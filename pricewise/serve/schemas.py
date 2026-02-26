"""Request/response schemas for the serving API.

The model was trained on ~79 raw features, but requiring all of them would make
the API painful. So we type the high-signal fields explicitly (for nice docs +
validation) and allow any other raw column via `extra="allow"`. Unset fields are
sent as null and imputed by the pipeline.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    OverallQual: Optional[int] = Field(None, ge=1, le=10, description="Overall material/finish quality (1-10)")
    OverallCond: Optional[int] = Field(None, ge=1, le=10, description="Overall condition (1-10)")
    GrLivArea: Optional[float] = Field(None, ge=0, description="Above-grade living area (sq ft)")
    TotalBsmtSF: Optional[float] = Field(None, ge=0, description="Total basement area (sq ft)")
    FirstFlrSF: Optional[float] = Field(None, ge=0, alias="1stFlrSF")
    SecondFlrSF: Optional[float] = Field(None, ge=0, alias="2ndFlrSF")
    LotArea: Optional[float] = Field(None, ge=0, description="Lot size (sq ft)")
    YearBuilt: Optional[int] = Field(None, ge=1800, le=2100)
    YearRemodAdd: Optional[int] = Field(None, ge=1800, le=2100)
    FullBath: Optional[int] = Field(None, ge=0, le=10)
    HalfBath: Optional[int] = Field(None, ge=0, le=10)
    BedroomAbvGr: Optional[int] = Field(None, ge=0, le=20)
    TotRmsAbvGrd: Optional[int] = Field(None, ge=0, le=30)
    GarageCars: Optional[int] = Field(None, ge=0, le=10)
    GarageArea: Optional[float] = Field(None, ge=0)
    Fireplaces: Optional[int] = Field(None, ge=0, le=10)
    YrSold: Optional[int] = Field(None, ge=1900, le=2100)
    Neighborhood: Optional[str] = Field(None, description="Ames neighborhood code, e.g. NridgHt")

    def to_features(self) -> dict:
        """Flatten to a {column: value} dict using real column names (aliases)."""
        return self.model_dump(by_alias=True, exclude_none=True)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"
    model_name: str
