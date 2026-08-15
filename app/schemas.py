from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HouseFeatures(BaseModel):
    overall_qual: int = Field(
        ..., ge=1, le=10, description="Calidad general de materiales y acabados (1-10)"
    )
    gr_liv_area: float = Field(
        ..., gt=0, le=15000, description="Superficie habitable sobre el suelo (sqft)"
    )
    total_bsmt_sf: float = Field(
        ..., ge=0, le=15000, description="Superficie total del sótano (sqft)"
    )
    garage_cars: int = Field(
        ..., ge=0, le=10, description="Capacidad del garaje en número de autos"
    )
    full_bath: int = Field(..., ge=0, le=10, description="Número de baños completos sobre el suelo")
    year_built: int = Field(..., ge=1800, le=2026, description="Año de construcción original")

    @field_validator("gr_liv_area", "total_bsmt_sf")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf check
            raise ValueError("el valor debe ser un número finito")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "overall_qual": 7,
                "gr_liv_area": 1710,
                "total_bsmt_sf": 856,
                "garage_cars": 2,
                "full_bath": 2,
                "year_built": 2003,
            }
        }
    }


class PredictResponse(BaseModel):
    predicted_price: float
    model_version: str
    currency: Literal["USD"] = "USD"


class BatchPredictRequest(BaseModel):
    items: list[HouseFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class FeatureSpec(BaseModel):
    name: str
    type: str
    required: bool = True


class SchemaResponse(BaseModel):
    features: list[FeatureSpec]
    target: str
    model_version: str | None = None
