"""
FastAPI app: CRUD over houses, plus /predict.

    uv run uvicorn app.main_amin:app --reload
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import models_amin, schemas_amin
from app.database_amin import Base, engine, get_db
from kc.config_amin import MODEL_PATH
from kc.predict_amin import get_model, predict_price


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Create the tables on startup.

    bonus_solution does this at import time, which makes the module awkward to
    import from a test or a script.
    """
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="King County Houses",
    description="Store houses and predict King County prices.",
    version="1.0.0",
    lifespan=lifespan,
)

DatabaseSession = Annotated[Session, Depends(get_db)]


def get_house_or_404(db: Session, house_id: int) -> models_amin.House:
    """
    Fetch a house or 404.
    """
    house = db.get(models_amin.House, house_id)
    if house is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"House {house_id} was not found.",
        )
    return house


@app.get("/", tags=["health"])
def index() -> dict[str, str | bool]:
    """
    Status, and whether a model is available.
    """
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.get("/houses", response_model=list[schemas_amin.HouseOut], tags=["houses"])
def list_houses(db: DatabaseSession) -> list[models_amin.House]:
    """
    All stored houses.
    """
    return list(db.scalars(select(models_amin.House).order_by(models_amin.House.id)))


@app.get("/houses/{house_id}", response_model=schemas_amin.HouseOut, tags=["houses"])
def read_house(house_id: int, db: DatabaseSession) -> models_amin.House:
    """
    One house.
    """
    return get_house_or_404(db, house_id)


@app.post(
    "/houses",
    response_model=schemas_amin.HouseOut,
    status_code=status.HTTP_201_CREATED,
    tags=["houses"],
)
def create_house(
    request: schemas_amin.HouseCreate, db: DatabaseSession
) -> models_amin.House:
    """
    Store a new house.
    """
    house = models_amin.House(**request.model_dump())
    db.add(house)
    db.commit()
    db.refresh(house)
    return house


@app.put("/houses/{house_id}", response_model=schemas_amin.HouseOut, tags=["houses"])
def update_house(
    house_id: int, request: schemas_amin.HouseUpdate, db: DatabaseSession
) -> models_amin.House:
    """
    Replace a house's fields.
    """
    house = get_house_or_404(db, house_id)
    for field, value in request.model_dump().items():
        setattr(house, field, value)
    db.commit()
    db.refresh(house)
    return house


@app.delete(
    "/houses/{house_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["houses"]
)
def delete_house(house_id: int, db: DatabaseSession) -> Response:
    """
    Delete a house.
    """
    house = get_house_or_404(db, house_id)
    db.delete(house)
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/predict", response_model=schemas_amin.PredictionResponse, tags=["model"])
def predict(request: schemas_amin.PredictionRequest) -> schemas_amin.PredictionResponse:
    """
    Predict one house's sale price.

    The body is raw house attributes, not the nineteen engineered features. The
    saved pipeline handles that.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No trained model available. Run `python -m kc.train_amin` to "
                f"create {MODEL_PATH.name}."
            ),
        )
    try:
        # Passed explicitly instead of relying on get_model's default, which is
        # bound at import time: the check above and the load here must agree.
        model = get_model(MODEL_PATH)
        price = predict_price(request.model_dump(), model=model)[0]
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    return schemas_amin.PredictionResponse(predicted_price=price)
