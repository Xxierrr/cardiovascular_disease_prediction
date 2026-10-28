from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == (
        "Cardiovascular Disease Prediction API is running"
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_prediction():

    patient = {
        "age_years": 55,
        "height": 170,
        "weight": 75,
        "ap_hi": 130,
        "ap_lo": 85,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 2
    }

    response = client.post(
        "/predict",
        json=patient
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "result" in result
    assert "probability" in result
    assert "bmi" in result

    assert result["prediction"] in [0, 1]