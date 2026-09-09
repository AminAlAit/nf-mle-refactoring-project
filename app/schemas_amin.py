"""
Request and response models.
"""

from pydantic import BaseModel, ConfigDict, Field

from kc.config_amin import MAX_PLAUSIBLE_BEDROOMS


class HouseBase(BaseModel):
    """
    The five stored fields.
    """

    bedrooms: int = Field(ge=0, le=MAX_PLAUSIBLE_BEDROOMS)
    bathrooms: float = Field(ge=0, le=20)
    sqft_living: int = Field(ge=100)
    grade: int = Field(ge=1, le=13)
    zipcode: str = Field(min_length=5, max_length=10)


class HouseCreate(HouseBase):
    """
    POST /houses body.
    """


class HouseUpdate(HouseBase):
    """
    PUT /houses/{id} body.
    """


class HouseOut(HouseBase):
    """
    A stored house.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int


class PredictionRequest(BaseModel):
    """
    Everything the model needs to price a house.

    Longer than the five stored fields because the model uses nineteen features.
    No sqft_basement, since the pipeline rebuilds it. waterfront, view and yr_renovated
    default to 0, same as the cleaning rule.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bedrooms": 3,
                "bathrooms": 2.5,
                "sqft_living": 1800,
                "sqft_lot": 7500,
                "floors": 2.0,
                "condition": 3,
                "grade": 8,
                "sqft_above": 1800,
                "yr_built": 1995,
                "zipcode": 98103,
                "lat": 47.6567,
                "long": -122.3419,
                "sqft_living15": 1750,
                "sqft_lot15": 7200,
            }
        }
    )

    bedrooms: int = Field(ge=0, le=MAX_PLAUSIBLE_BEDROOMS)
    bathrooms: float = Field(ge=0, le=20)
    sqft_living: int = Field(ge=100)
    sqft_lot: int = Field(ge=100)
    floors: float = Field(ge=1, le=5)
    condition: int = Field(ge=1, le=5)
    grade: int = Field(ge=1, le=13)
    sqft_above: int = Field(ge=100)
    yr_built: int = Field(ge=1800, le=2100)
    zipcode: int = Field(ge=98000, le=98999)
    lat: float = Field(ge=47.0, le=48.0)
    long: float = Field(ge=-123.0, le=-121.0)
    sqft_living15: int = Field(ge=100)
    sqft_lot15: int = Field(ge=100)

    waterfront: int = Field(default=0, ge=0, le=1)
    view: int = Field(default=0, ge=0, le=4)
    yr_renovated: int = Field(default=0, ge=0, le=2100)


class PredictionResponse(BaseModel):
    """
    A predicted price in dollars.
    """

    predicted_price: float
    currency: str = "USD"
