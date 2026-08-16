# House Pricing API — MLOps end-to-end

**🔗 Servicio en producción:** https://mlops-pricing-app.onrender.com
**📄 Docs interactivos:** https://mlops-pricing-app.onrender.com/docs
**❤️ Health check:** https://mlops-pricing-app.onrender.com/health

> Desplegado en **Render** (plan Free). El plan gratuito "duerme" el servicio
> tras ~15 min de inactividad — el primer request tras eso puede tardar
> 30-50s en responder mientras el contenedor arranca. Es esperado, no un fallo.

Pipeline completo: entrenamiento → serialización de modelo → API FastAPI → contenedor Docker → CI/CD en GitHub Actions → despliegue automático en Render.

## Problema

Predecir el precio de venta de una vivienda (`SalePrice`, USD) a partir de variables numéricas del inmueble.

## Datos

Dataset: **Ames Housing** — competencia Kaggle "House Prices: Advanced Regression Techniques"
(`data/housing_train_raw.csv`, 1460 filas, 81 columnas originales, ~40 numéricas y el resto categóricas).

Se seleccionó un subconjunto de **6 columnas sin valores nulos** y con la correlación
más alta contra `SalePrice`:

| Columna original | Columna normalizada | Descripción                              | Corr. con SalePrice |
|-------------------|----------------------|-------------------------------------------|----------------------|
| OverallQual        | `overall_qual`       | Calidad general de materiales/acabados (1-10) | 0.79 |
| GrLivArea          | `gr_liv_area`         | Superficie habitable sobre el suelo (sqft)     | 0.71 |
| GarageCars         | `garage_cars`         | Capacidad del garaje (autos)                   | 0.64 |
| TotalBsmtSF        | `total_bsmt_sf`       | Superficie total del sótano (sqft)             | 0.61 |
| FullBath           | `full_bath`           | Baños completos sobre el suelo                 | 0.56 |
| YearBuilt          | `year_built`          | Año de construcción original                   | 0.52 |
| SalePrice (target) | `price`               | Precio de venta (USD)                          | —    |

El resto de las ~70 columnas (mayormente categóricas: `Neighborhood`, `SaleType`,
`ExterQual`, etc.) se descartan deliberadamente para mantener el pipeline simple y
100% numérico.

## Modelo

- Pipeline scikit-learn: `StandardScaler` + `RandomForestRegressor` (200 árboles, profundidad 12).
- Split 80/20 con `random_state=42` fijo (reproducible).
- Métrica evaluada sobre el conjunto de prueba (no visto en entrenamiento):

```json
{
  "rmse": 29374.31,
  "mae": 19207.23,
  "r2": 0.8875,
  "n_train": 1168,
  "n_test": 292
}
```

Interpretación: el modelo explica ~89% de la varianza del precio (R²) y se equivoca
en promedio ~$19.2k (MAE) sobre un precio medio de ~$180k — un error relativo de
~10-11%, razonable para un subconjunto de solo 6 variables numéricas sin
información de ubicación (`Neighborhood`), que en el dataset completo es uno de
los predictores más fuertes.

Artefactos versionados en `artifacts/` (generados automáticamente en cada build,
ver sección Docker):
- `model.pkl` — pipeline sklearn completo (preprocesador + modelo).
- `metadata.json` — features, fecha de entrenamiento (UTC), semilla, métricas, versión del modelo.

## Cómo levantar el proyecto

### 1. Local (sin Docker)

```bash
python3 -m venv pricing_app
source pricing_app/bin/activate   # Windows: pricing_app\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python src/train.py               # usa data/housing_train_raw.csv por defecto
uvicorn app.main:app --reload
```

### 2. Con Docker Compose (recomendado — un solo comando)

```bash
docker compose up --build
```

El modelo se entrena **dentro** del build de la imagen (ver `Dockerfile`), así que
esto funciona en una máquina limpia recién clonada, sin pasos manuales previos.
La API queda disponible en `http://localhost:8000`.

## Endpoints

| Método | Ruta               | Descripción                                   |
|--------|---------------------|------------------------------------------------|
| GET    | `/health`            | Estado del servicio y si el modelo está cargado |
| POST   | `/predict`           | Predicción para una sola vivienda              |
| POST   | `/predict/batch`     | Predicción para hasta 1000 viviendas            |
| GET    | `/model/schema`      | Features esperadas y sus tipos                 |
| GET    | `/docs`              | Documentación interactiva (Swagger UI)          |

### Ejemplo real — `POST /predict` (contra el servicio en producción)

```bash
curl -s -X POST https://mlops-pricing-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "overall_qual": 7,
    "gr_liv_area": 1710,
    "total_bsmt_sf": 856,
    "garage_cars": 2,
    "full_bath": 2,
    "year_built": 2003
  }'
```

Respuesta real (esta casa tiene `SalePrice` real de $208,500 en el dataset):

```json
{"predicted_price":190718.24,"model_version":"1.0.0","currency":"USD"}
```

### Ejemplo real — entrada inválida (nunca un 500 genérico)

```bash
curl -s -X POST https://mlops-pricing-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"overall_qual": 99, "gr_liv_area": 1710, "total_bsmt_sf": 856, "garage_cars": 2, "full_bath": 2, "year_built": 2003}'
```

```json
{"detail":"Entrada inválida","errors":[{"field":"overall_qual","message":"Input should be less than or equal to 10"}]}
```

Status HTTP: `422 Unprocessable Entity`.

## Tests y calidad

```bash
ruff check .              # lint
ruff format --check .     # formato
pytest tests/ -v          # 14 tests, sin red ni credenciales
```

Cobertura de tests: contrato de la API (`/health`, `/predict`, `/predict/batch`,
`/model/schema`), validación de entradas (tipo incorrecto, campo faltante, fuera de
rango), casos borde (batch vacío, un ítem inválido dentro de un batch) y errores
esperados (404, 422, 503 si el modelo no cargó). También hay tests de reproducibilidad
del entrenamiento y de generación correcta de artefactos.

## Docker

- Imagen base `python:3.11-slim`, dependencias fijadas por versión en `requirements.txt`.
- El modelo se entrena **durante el build** (`RUN python src/train.py` dentro del
  `Dockerfile`), no depende de artefactos pre-generados fuera de control de versión.
- Usuario no-root (`appuser`), `HEALTHCHECK` declarado, puerto configurable vía `$PORT`
  (requisito de Render y otros PaaS).
- `.dockerignore` excluye tests, docs y archivos de entorno.

## CI/CD (GitHub Actions)

Workflow en `.github/workflows/ci.yml`, disparado en `push`/`pull_request` sobre `main`
y en tags `v*`. Jobs encadenados:

1. **lint** — `ruff check` + `ruff format --check`.
2. **test** — entrena el modelo, corre `pytest` (14 tests).
3. **build** — construye la imagen Docker (el modelo se entrena dentro del build).
4. **smoke-test** — levanta el contenedor de verdad y llama a `/health` y `/predict` reales; falla el job si no responde.
5. **publish** *(condicional, solo en tags `v*`)* — publica la imagen en GHCR (`ghcr.io/<owner>/house-pricing-api`).
6. **deploy** *(en `main` y tags `v*`, solo si `smoke-test` pasó)* — dispara el deploy en Render vía su *Deploy Hook* y verifica que el servicio público responda en `/health`. Las credenciales (`RENDER_DEPLOY_HOOK_URL`, `RENDER_SERVICE_URL`) viven en **GitHub Secrets**, nunca en el repositorio.

> El auto-deploy nativo de Render (conectar el repo desde su dashboard) está
> **desactivado** a propósito — el despliegue lo dispara únicamente el job
> `deploy` de Actions, condicionado a que los jobs anteriores pasen.

## Variables de entorno

Ver `.env.example` para las variables de runtime de la aplicación. Ninguna requiere
secretos para correr localmente.

Para CI/CD, estos secretos viven en **GitHub Secrets** (Settings → Secrets and
variables → Actions), no en el código ni en `.env.example`:

| Secreto                  | Descripción                                              |
|---------------------------|-----------------------------------------------------------|
| `RENDER_DEPLOY_HOOK_URL`  | Deploy Hook del servicio (Render → Settings → Deploy Hook) |
| `RENDER_SERVICE_URL`      | URL pública del servicio, para el health check post-deploy |
| `GITHUB_TOKEN`             | Provisto automáticamente por Actions, usado para publicar en GHCR |

## Documentación adicional

- **Informe técnico:** [`docs/informe.pdf`](docs/informe.pdf) — problema, datos, modelo, decisiones, limitaciones y distribución del trabajo.
- **Video de demostración (≤5 min):** [COMPLETAR — enlace al video]

## Uso de asistentes de IA

Se usó Claude (Anthropic) como asistente durante el desarrollo para: generar el
andamiaje inicial del proyecto (estructura de carpetas, `Dockerfile`,
`docker-compose.yml`, workflow de GitHub Actions), redactar los tests de pytest
sobre el contrato ya definido de la API, y depurar errores de despliegue en Render
(configuración de `$PORT`, entorno Docker vs. runtime nativo). El diseño del
esquema de features, la selección de columnas del dataset, y las decisiones de
arquitectura del modelo fueron revisadas y validadas por el equipo.

## Limitaciones conocidas / qué haría el equipo con más tiempo

- Se descartaron ~70 columnas categóricas del dataset original (`Neighborhood`, `ExterQual`, `SaleType`, etc.) que en un modelo de producción real aportarían señal significativa — con más tiempo se agregaría un `OneHotEncoder`/`OrdinalEncoder` dentro del mismo `Pipeline` de sklearn.
- No hay monitoreo de *data drift* ni reentrenamiento automático.
- No hay autenticación en la API (pensado para uso académico/demo).
- El modelo no reporta intervalos de confianza por predicción.
- No hay versionado formal de datasets (DVC).
- El plan gratuito de Render duerme el servicio tras inactividad; el primer request post-inactividad puede tardar 30-50s.
