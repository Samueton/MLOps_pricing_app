"""
Entrena un modelo de regresión para predecir precios de casas.

Uso:
    python src/train.py [--data data/usa_housing.csv] [--out artifacts]

Genera en `artifacts/`:
    - model.pkl       -> pipeline sklearn completo (preprocesador + modelo)
    - metadata.json    -> features, fecha de entrenamiento, métricas, versión
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.schema import FEATURE_COLUMNS, RAW_COLUMN_MAP, TARGET_COLUMN  # noqa: E402

SEED = 42
MODEL_VERSION = "1.0.0"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Acepta tanto los nombres originales del CSV de Kaggle (Ames Housing) como los
    # ya normalizados
    if "SalePrice" in df.columns:
        df = df.rename(columns=RAW_COLUMN_MAP)

    # Solo columnas numéricas relevantes (se descartan las ~70 columnas categóricas
    # del dataset original)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el dataset: {missing}")

    df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(data_path: str, out_dir: str) -> dict:
    df = load_data(data_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_path = out_path / "model.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "model_type": "RandomForestRegressor",
        "metrics": metrics,
        "data_path": str(data_path),
        "n_rows_total": int(len(df)),
    }
    with open(out_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Entrenamiento completo.")
    print(json.dumps(metrics, indent=2))
    print(f"Modelo guardado en: {model_path}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo de precios de casas")
    parser.add_argument("--data", default="data/housing_train_raw.csv")
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.out)
