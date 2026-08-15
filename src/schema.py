"""Definición única de esquema de features, compartida por entrenamiento y API.

Dataset: Ames Housing / "House Prices - Advanced Regression Techniques" (Kaggle),
archivo housing_train.csv provisto por el usuario. Se seleccionó un subconjunto
pequeño de columnas 100% numéricas, sin nulos y con alta correlación con SalePrice.
"""

FEATURE_COLUMNS = [
    "overall_qual",
    "gr_liv_area",
    "total_bsmt_sf",
    "garage_cars",
    "full_bath",
    "year_built",
]

# Nombres originales de columnas tal como vienen en el CSV (Ames Housing)
RAW_COLUMN_MAP = {
    "OverallQual": "overall_qual",
    "GrLivArea": "gr_liv_area",
    "TotalBsmtSF": "total_bsmt_sf",
    "GarageCars": "garage_cars",
    "FullBath": "full_bath",
    "YearBuilt": "year_built",
    "SalePrice": "price",
}

TARGET_COLUMN = "price"

FEATURE_TYPES = {col: "float" for col in FEATURE_COLUMNS}
