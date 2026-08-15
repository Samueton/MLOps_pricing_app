import json

import pytest

from src.train import load_data, train


def test_load_data_rejects_missing_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("Foo,Bar\n1,2\n")
    with pytest.raises(ValueError, match="Faltan columnas"):
        load_data(str(bad_csv))


def test_train_is_reproducible_with_fixed_seed(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    meta1 = train("data/housing_train_raw.csv", str(out1))
    meta2 = train("data/housing_train_raw.csv", str(out2))
    assert meta1["metrics"]["rmse"] == meta2["metrics"]["rmse"]
    assert meta1["seed"] == meta2["seed"] == 42


def test_train_produces_model_and_metadata_artifacts(tmp_path):
    out_dir = tmp_path / "artifacts"
    train("data/housing_train_raw.csv", str(out_dir))
    assert (out_dir / "model.pkl").exists()
    assert (out_dir / "metadata.json").exists()
    with open(out_dir / "metadata.json") as f:
        metadata = json.load(f)
    assert "rmse" in metadata["metrics"]
    assert "r2" in metadata["metrics"]
    assert metadata["n_rows_total"] > 0
