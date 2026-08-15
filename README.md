# House Pricing API — MLOps end-to-end

Pipeline completo: entrenamiento → serialización de modelo → API FastAPI → contenedor Docker → CI/CD en GitHub Actions.

## Problema

Predecir el precio de venta de una vivienda (`SalePrice`, USD) a partir de variables numéricas del inmueble.

## Datos

Dataset: **Ames Housing** — competencia Kaggle "House Prices: Advanced Regression Techniques"
(`housing_train.csv`, 1460 filas, 81 columnas originales, ~40 numéricas y el resto categóricas).

Para mantener el pipeline simple y 100% numérico, se seleccionó un subconjunto de
**6 columnas sin valores nulos** y con la correlación más alta contra `SalePrice`:

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
evitar encoders adicionales, tal como se pidió.

El CSV crudo vive en `data/housing_train_raw.csv`. `data/generate_synthetic_data.py`
se conserva como *fallback* offline (genera un dataset sintético con esquema distinto,
el de "USA Housing") por si se necesita reconstruir el pipeline sin conexión a ningún dato real.

## Modelo

- Pipeline scikit-learn: `StandardScaler` + `RandomForestRegressor` (200 árboles, profundidad 12).
- Split 80/20 con `random_state=42` fijo (reproducible).
- Métrica evaluada sobre el conjunto de prueba (no visto en entrenamiento), datos reales:

```json
{
  "rmse": 29374.31,
  "mae": 19207.23,
  "r2": 0.8875,
  "n_train": 1168,
  "n_test": 292
}
```

Artefactos versionados en `artifacts/`:
- `model.pkl` — pipeline sklearn completo (preprocesador + modelo).
- `metadata.json` — features, fecha de entrenamiento (UTC), semilla, métricas, versión del modelo.

## Cómo levantar el proyecto

### 1. Local (sin Docker)

```bash
pip install -r requirements-dev.txt
python src/train.py          # usa data/housing_train_raw.csv por defecto
uvicorn app.main:app --reload
```

### 2. Con Docker Compose (recomendado)

```bash
python src/train.py                # genera artifacts/ antes de construir la imagen
docker compose up --build
```

La API queda disponible en `http://localhost:8000`.

## Endpoints

| Método | Ruta               | Descripción                                   |
|--------|---------------------|------------------------------------------------|
| GET    | `/health`            | Estado del servicio y si el modelo está cargado |
| POST   | `/predict`           | Predicción para una sola vivienda              |
| POST   | `/predict/batch`     | Predicción para hasta 1000 viviendas            |
| GET    | `/model/schema`      | Features esperadas y sus tipos                 |
| GET    | `/docs`              | Documentación interactiva (Swagger UI)          |

### Ejemplo real — `POST /predict`

```bash
curl -s -X POST http://localhost:8000/predict \
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
curl -s -X POST http://localhost:8000/predict \
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

## CI/CD (GitHub Actions)

Workflow en `.github/workflows/ci.yml`, disparado en `push`/`pull_request` sobre `main`
y en tags `v*`. Jobs encadenados:

1. **lint** — `ruff check` + `ruff format --check`.
2. **test** — entrena el modelo con `data/housing_train_raw.csv`, corre `pytest`, sube `artifacts/` como artifact de CI.
3. **build** — construye la imagen Docker con el modelo entrenado dentro.
4. **smoke-test** — levanta el contenedor de verdad y llama a `/health` y `/predict` reales; falla el job si no responde.
5. **publish** *(condicional, solo en tags `v*`)* — publica la imagen en GHCR (`ghcr.io/<owner>/house-pricing-api`).

## Variables de entorno

Ver `.env.example`. Ninguna variable requiere secretos para correr localmente; `GITHUB_TOKEN` para GHCR lo provee GitHub Actions automáticamente.

## Limitaciones conocidas / qué haría el equipo con más tiempo

- Se descartaron ~70 columnas categóricas del dataset original (`Neighborhood`, `ExterQual`, `SaleType`, etc.) que en un modelo de producción real aportarían señal significativa — con más tiempo se agregaría un `OneHotEncoder`/`OrdinalEncoder` dentro del mismo `Pipeline` de sklearn.
- No hay monitoreo de *data drift* ni reentrenamiento automático — se agregaría un job programado que reentrene y compare métricas contra el modelo en producción antes de promoverlo.
- No hay autenticación en la API (pensado para uso interno/demo) — se añadiría API key o OAuth2 antes de exponerla públicamente.
- El modelo no reporta intervalos de confianza por predicción — se podría usar quantile regression o un ensemble con bandas de incertidumbre.
- No hay versionado formal de datasets (DVC) — se agregaría para trazabilidad completa dato→modelo→despliegue.
