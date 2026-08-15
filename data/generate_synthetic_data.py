"""
Generador de datos sintéticos con el MISMO esquema que el dataset de Kaggle
"USA Housing dataset" (https://www.kaggle.com/datasets/gpandi007/usa-housing-dataset).

Este entorno no tiene salida de red hacia kaggle.com, así que este script crea
un CSV con las mismas columnas, tipos y rangos aproximados que el dataset real,
para poder construir y probar todo el pipeline end-to-end.

Para usar el dataset REAL:
1. Descarga USA_Housing.csv desde el link de arriba.
2. Reemplaza data/usa_housing.csv por el archivo descargado (mismas columnas).
3. Vuelve a correr `python src/train.py` — no hace falta tocar nada más.
"""
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 5000
OUT_PATH = "data/usa_housing.csv"


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    avg_area_income = rng.normal(68583, 10658, n_rows).clip(min=15000)
    avg_area_house_age = rng.normal(5.98, 0.99, n_rows).clip(min=1)
    avg_area_number_of_rooms = rng.normal(6.99, 1.01, n_rows).clip(min=2)
    avg_area_number_of_bedrooms = rng.normal(3.98, 1.23, n_rows).clip(min=1)
    area_population = rng.normal(36163, 9925, n_rows).clip(min=1000)

    noise = rng.normal(0, 100000, n_rows)
    price = (
        21.5 * avg_area_income
        + 164000 * avg_area_house_age
        + 122000 * avg_area_number_of_rooms
        + 2500 * avg_area_number_of_bedrooms
        + 15.2 * area_population
        + noise
        - 2600000
    ).clip(min=15000)

    df = pd.DataFrame(
        {
            "Avg. Area Income": avg_area_income.round(2),
            "Avg. Area House Age": avg_area_house_age.round(2),
            "Avg. Area Number of Rooms": avg_area_number_of_rooms.round(2),
            "Avg. Area Number of Bedrooms": avg_area_number_of_bedrooms.round(2),
            "Area Population": area_population.round(2),
            "Price": price.round(2),
        }
    )
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Generados {len(df)} registros sintéticos en {OUT_PATH}")
    print(df.describe())
