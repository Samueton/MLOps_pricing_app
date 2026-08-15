from tests.conftest import VALID_PAYLOAD


def test_health_returns_ok_and_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] is not None


def test_model_schema_lists_expected_features(client):
    resp = client.get("/model/schema")
    assert resp.status_code == 200
    body = resp.json()
    names = {f["name"] for f in body["features"]}
    assert names == {
        "overall_qual",
        "gr_liv_area",
        "total_bsmt_sf",
        "garage_cars",
        "full_bath",
        "year_built",
    }
    assert body["target"] == "price"


def test_predict_valid_payload_returns_positive_price(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_price"] > 0
    assert body["currency"] == "USD"
    assert "model_version" in body


def test_predict_missing_field_returns_422_with_detail(client):
    payload = VALID_PAYLOAD.copy()
    del payload["overall_qual"]
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"] == "Entrada inválida"
    assert any(e["field"] == "overall_qual" for e in body["errors"])


def test_predict_wrong_type_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["overall_qual"] = "no-es-un-numero"
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_negative_value_returns_422_not_500(client):
    payload = VALID_PAYLOAD.copy()
    payload["gr_liv_area"] = -100
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "errors" in resp.json()


def test_predict_out_of_range_value_returns_422(client):
    payload = VALID_PAYLOAD.copy()
    payload["overall_qual"] = 99  # excede el máximo permitido (1-10)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch_valid_returns_all_predictions(client):
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert all(p["predicted_price"] > 0 for p in body["predictions"])


def test_predict_batch_empty_list_returns_422(client):
    resp = client.post("/predict/batch", json={"items": []})
    assert resp.status_code == 422


def test_predict_batch_one_invalid_item_returns_422(client):
    bad_payload = VALID_PAYLOAD.copy()
    bad_payload["overall_qual"] = -1
    resp = client.post("/predict/batch", json={"items": [VALID_PAYLOAD, bad_payload]})
    assert resp.status_code == 422


def test_unknown_route_returns_404(client):
    resp = client.get("/does-not-exist")
    assert resp.status_code == 404
