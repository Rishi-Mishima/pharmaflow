from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_predict():
    request_data = {
        "lag_1": 40,
        "lag_7": 42,
        "lag_14": 38,
        "lag_28": 41,
        "rolling_mean_7": 40,
        "rolling_mean_28": 39,
        "dow_sin": 0,
        "dow_cos": 1,
        "month_sin": 0,
        "month_cos": 1,
        "doy_sin": 0,
        "doy_cos": 1,
        "is_weekend": 0
    }
    response = client.post(
        "/predict",
        json=request_data
    )

    assert response.status_code == 200
    response_data = response.json()

    assert "prediction" in response_data
    assert isinstance(response_data["prediction"], float)


def test_predict_invalid_input():
    request_data = {
        "lag_1": "Hello",
    }

    response = client.post(
        "/predict",
        json=request_data
    )

    assert response.status_code == 422