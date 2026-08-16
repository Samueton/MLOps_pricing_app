from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = "docs/informe.pdf"

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="H1",
        fontSize=17,
        spaceAfter=10,
        spaceBefore=4,
        textColor=colors.HexColor("#111111"),
        fontName="Helvetica-Bold",
    )
)
styles.add(
    ParagraphStyle(
        name="H2",
        fontSize=12.5,
        spaceAfter=6,
        spaceBefore=14,
        textColor=colors.HexColor("#1a1a1a"),
        fontName="Helvetica-Bold",
    )
)
styles.add(
    ParagraphStyle(name="Body", fontSize=10, leading=14.5, spaceAfter=6, fontName="Helvetica")
)
styles.add(
    ParagraphStyle(
        name="Placeholder",
        fontSize=10,
        leading=14.5,
        spaceAfter=6,
        fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#b45309"),
    )
)
styles.add(
    ParagraphStyle(name="Small", fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"))
)

story = []

story.append(
    Paragraph("Servicio MLOps end-to-end: predicción de precios de vivienda", styles["H1"])
)
story.append(
    Paragraph(
        "[COMPLETAR] Curso · Sección · Segundo semestre 2026 &nbsp;|&nbsp; Integrantes: [NOMBRE 1], [NOMBRE 2], [NOMBRE 3], [NOMBRE 4]",
        styles["Placeholder"],
    )
)
story.append(Spacer(1, 10))

story.append(Paragraph("1. Problema", styles["H2"]))
story.append(
    Paragraph(
        "Se aborda un problema de regresión: predecir el precio de venta (SalePrice, en USD) de una "
        "vivienda a partir de un subconjunto reducido de variables numéricas del inmueble. El objetivo "
        "del trabajo no es maximizar la sofisticación del modelo, sino construir un sistema completo y "
        "operable: entrenamiento reproducible, contrato de API validado, contenedor portable y un "
        "pipeline de integración y despliegue continuo.",
        styles["Body"],
    )
)

story.append(Paragraph("2. Datos", styles["H2"]))
story.append(
    Paragraph(
        'Se utilizó el dataset público <b>Ames Housing</b> (competencia Kaggle "House Prices: Advanced '
        'Regression Techniques"), con 1460 registros y 81 columnas originales. De estas, se descartaron '
        "deliberadamente las ~70 columnas categóricas para mantener el pipeline simple y 100% numérico, "
        "y se seleccionaron las 6 columnas numéricas sin valores nulos con mayor correlación lineal "
        "respecto al precio de venta:",
        styles["Body"],
    )
)

data = [
    ["Variable", "Descripción", "Corr. con precio"],
    ["overall_qual", "Calidad general de materiales y acabados (1-10)", "0.79"],
    ["gr_liv_area", "Superficie habitable sobre el suelo (sqft)", "0.71"],
    ["garage_cars", "Capacidad del garaje (autos)", "0.64"],
    ["total_bsmt_sf", "Superficie total del sótano (sqft)", "0.61"],
    ["full_bath", "Baños completos sobre el suelo", "0.56"],
    ["year_built", "Año de construcción original", "0.52"],
]
t = Table(data, colWidths=[3.3 * cm, 9.2 * cm, 3.2 * cm])
t.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story.append(t)
story.append(Spacer(1, 8))
story.append(
    Paragraph(
        "El dataset crudo se versiona en <font face='Courier'>data/housing_train_raw.csv</font> dentro "
        "del repositorio.",
        styles["Body"],
    )
)

story.append(Paragraph("3. Modelo y entrenamiento", styles["H2"]))
story.append(
    Paragraph(
        "Se entrenó un pipeline de scikit-learn compuesto por un <font face='Courier'>StandardScaler</font> "
        "seguido de un <font face='Courier'>RandomForestRegressor</font> (200 árboles, profundidad máxima 12). "
        "Se usó una semilla fija (random_state=42) y una separación explícita 80/20 entre entrenamiento y "
        "prueba, evaluando las métricas únicamente sobre el conjunto de prueba (datos no vistos durante "
        "el ajuste).",
        styles["Body"],
    )
)

metric_cell_style = ParagraphStyle(
    name="MetricCell", fontSize=8.7, leading=11, fontName="Helvetica"
)
metrics_data = [
    ["Métrica", "Valor", "Interpretación"],
    [
        Paragraph("R<super rise=3 size=6>2</super>", metric_cell_style),
        "0.888",
        "El modelo explica ~89% de la varianza del precio",
    ],
    ["RMSE", "$29,374", "Error cuadrático medio sobre el conjunto de prueba"],
    ["MAE", "$19,207", "Error absoluto promedio (~10-11% del precio medio, ~$180k)"],
]
t2 = Table(metrics_data, colWidths=[2.3 * cm, 2.3 * cm, 11.1 * cm])
t2.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story.append(t2)
story.append(Spacer(1, 8))
story.append(
    Paragraph(
        "Los artefactos (<font face='Courier'>model.pkl</font> y <font face='Courier'>metadata.json</font> "
        "con features, fecha de entrenamiento, semilla y métricas) se generan de forma reproducible mediante "
        "<font face='Courier'>src/train.py</font>, y se regeneran automáticamente en cada build de la imagen "
        "Docker.",
        styles["Body"],
    )
)

story.append(Paragraph("4. Decisiones de diseño", styles["H2"]))
story.append(
    Paragraph(
        "[COMPLETAR — el equipo debe describir aquí, con sus propias palabras, decisiones concretas y el "
        "porqué. Ejemplos de preguntas que la defensa oral puede hacer sobre esta sección:]",
        styles["Placeholder"],
    )
)
bullets = [
    "¿Por qué RandomForest y no un modelo lineal? ¿Se probó alguna alternativa?",
    "¿Por qué se descartaron las columnas categóricas en vez de codificarlas?",
    "¿Por qué se entrena el modelo dentro del build de Docker y no se versiona el .pkl directamente?",
    "¿Qué llevó a fijar los rangos de validación de Pydantic (ej. overall_qual entre 1 y 10)?",
    "¿Por qué se eligió Render sobre otras opciones de despliegue?",
]
story.append(
    ListFlowable(
        [ListItem(Paragraph(b, styles["Placeholder"])) for b in bullets],
        bulletType="bullet",
        leftIndent=14,
    )
)

story.append(Paragraph("5. Resultados", styles["H2"]))
story.append(
    Paragraph(
        "[COMPLETAR — capturas de pantalla o resumen de: la ejecución verde del pipeline en la pestaña "
        "Actions, el servicio respondiendo en /health desde internet, y un ejemplo de /predict con salida "
        "real. Comentar si el error del modelo (MAE ~$19k) es aceptable para el caso de uso planteado.]",
        styles["Placeholder"],
    )
)

story.append(Paragraph("6. Limitaciones conocidas y trabajo futuro", styles["H2"]))
limits = [
    "Se descartaron ~70 columnas categóricas (ej. Neighborhood, ExterQual) que aportarían señal "
    "significativa en un modelo de producción real; con más tiempo se agregaría un encoder dentro "
    "del mismo Pipeline de sklearn.",
    "No hay monitoreo de data drift ni reentrenamiento automático.",
    "No hay autenticación en la API (pensado para uso académico/demo).",
    "El modelo no reporta intervalos de confianza por predicción.",
    "No hay versionado formal de datasets (ej. DVC).",
    "El plan gratuito de Render duerme el servicio tras inactividad; el primer request "
    "post-inactividad puede tardar 30-50s.",
]
story.append(
    ListFlowable(
        [ListItem(Paragraph(x, styles["Body"])) for x in limits],
        bulletType="bullet",
        leftIndent=14,
    )
)

story.append(Paragraph("7. Distribución del trabajo", styles["H2"]))
story.append(
    Paragraph(
        "[COMPLETAR — cada integrante debe responder por su parte en la defensa oral]",
        styles["Placeholder"],
    )
)
work_data = [
    ["Integrante", "Módulos / responsabilidades"],
    ["[Nombre 1]", "[ej. entrenamiento del modelo, selección de features]"],
    ["[Nombre 2]", "[ej. API FastAPI, contratos Pydantic]"],
    ["[Nombre 3]", "[ej. Docker, docker-compose, CI/CD]"],
    ["[Nombre 4]", "[ej. despliegue en Render, documentación, video]"],
]
t3 = Table(work_data, colWidths=[4.5 * cm, 11.2 * cm])
t3.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Oblique"),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#b45309")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story.append(t3)

story.append(Spacer(1, 14))
story.append(
    Paragraph(
        "Nota: este documento fue generado como plantilla inicial con asistencia de IA (Claude, Anthropic) "
        "a partir del código y las métricas reales del proyecto. Las secciones marcadas [COMPLETAR] "
        "requieren el análisis y la redacción propia del equipo antes de la entrega.",
        styles["Small"],
    )
)

doc = SimpleDocTemplate(
    OUT,
    pagesize=letter,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
    leftMargin=1.8 * cm,
    rightMargin=1.8 * cm,
)
doc.build(story)
print(f"Generado {OUT}")
