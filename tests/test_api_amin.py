"""
Tests for the FastAPI app. Each test gets its own throwaway SQLite file.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database_amin import Base, get_db
from app.main_amin import app
from kc.cleaning_amin import clean
from kc.features_amin import add_sqft_price
from kc.modeling_amin import split_dataset
from kc.pipeline_amin import build_model_pipeline

VALID_HOUSE = {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 1800,
    "grade": 8,
    "zipcode": "98103",
}

VALID_PREDICTION_INPUT = {
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


@pytest.fixture
def client(tmp_path):
    """
    Test client on a throwaway database.
    """
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        with TestingSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# CRUD


def test_create_returns_201_and_an_id(client) -> None:
    response = client.post("/houses", json=VALID_HOUSE)
    assert response.status_code == 201
    assert response.json()["id"] >= 1
    assert response.json()["bedrooms"] == 3


def test_full_crud_round_trip(client) -> None:
    house_id = client.post("/houses", json=VALID_HOUSE).json()["id"]

    assert client.get(f"/houses/{house_id}").json()["sqft_living"] == 1800

    updated = {**VALID_HOUSE, "bedrooms": 4, "sqft_living": 2000}
    assert client.put(f"/houses/{house_id}", json=updated).json()["bedrooms"] == 4
    assert client.get(f"/houses/{house_id}").json()["sqft_living"] == 2000

    assert client.delete(f"/houses/{house_id}").status_code == 204
    assert client.get(f"/houses/{house_id}").status_code == 404


def test_list_starts_empty_and_grows(client) -> None:
    assert client.get("/houses").json() == []
    client.post("/houses", json=VALID_HOUSE)
    client.post("/houses", json={**VALID_HOUSE, "zipcode": "98115"})
    assert len(client.get("/houses").json()) == 2


@pytest.mark.parametrize("method", ["get", "put", "delete"])
def test_missing_house_returns_404(client, method: str) -> None:
    kwargs = {"json": VALID_HOUSE} if method == "put" else {}
    response = getattr(client, method)("/houses/9999", **kwargs)
    assert response.status_code == 404
    assert "9999" in response.json()["detail"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("bedrooms", 33),  # the record we had to drop from the dataset
        ("bedrooms", -1),
        ("grade", 14),
        ("sqft_living", 10),
        ("zipcode", "981"),
    ],
)
def test_invalid_house_is_rejected(client, field: str, value) -> None:
    assert client.post("/houses", json={**VALID_HOUSE, field: value}).status_code == 422


# prediction


def test_predict_returns_a_plausible_price(client, tmp_path, monkeypatch) -> None:
    """
    Train a small model, point the app at it, price a house.
    """
    from kc.config_amin import RAW_DATA_PATH

    if not RAW_DATA_PATH.exists():
        pytest.skip("Dataset not available")

    import app.main_amin as main_module
    import kc.predict_amin as predict_module
    from kc.data_amin import load_raw
    from kc.modeling_amin import save_model

    prepared = add_sqft_price(clean(load_raw().sample(1200, random_state=3)))
    X_train, _, y_train, _ = split_dataset(prepared)
    model = build_model_pipeline().fit(X_train, y_train)

    model_path = tmp_path / "model.bin"
    save_model(model, model_path)
    monkeypatch.setattr(main_module, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_module, "MODEL_PATH", model_path)
    predict_module.get_model.cache_clear()

    response = client.post("/predict", json=VALID_PREDICTION_INPUT)
    assert response.status_code == 200
    price = response.json()["predicted_price"]
    assert 50_000 < price < 5_000_000, f"implausible prediction: {price}"


def test_predict_reports_a_missing_model(client, tmp_path, monkeypatch) -> None:
    import app.main_amin as main_module

    monkeypatch.setattr(main_module, "MODEL_PATH", tmp_path / "absent.bin")
    response = client.post("/predict", json=VALID_PREDICTION_INPUT)
    assert response.status_code == 503
    assert "kc.train_amin" in response.json()["detail"]


def test_predict_rejects_an_out_of_range_house(client) -> None:
    """
    Berlin is not in King County.
    """
    response = client.post("/predict", json={**VALID_PREDICTION_INPUT, "lat": 52.52})
    assert response.status_code == 422


def test_predict_defaults_the_optional_fields(client) -> None:
    """
    waterfront, view and yr_renovated can be left out.
    """
    from app.schemas_amin import PredictionRequest

    request = PredictionRequest(**VALID_PREDICTION_INPUT)
    assert request.waterfront == 0
    assert request.view == 0
    assert request.yr_renovated == 0
