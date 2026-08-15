import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "overall_qual": 7,
    "gr_liv_area": 1710,
    "total_bsmt_sf": 856,
    "garage_cars": 2,
    "full_bath": 2,
    "year_built": 2003,
}
