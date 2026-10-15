from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


# A known-valid request based on our earlier validation example.
VALID_PAYLOAD = {
    "Delivery_person_Age": 36.0,
    "Delivery_person_Ratings": 4.0,
    "Restaurant_latitude": 22.74806,
    "Restaurant_longitude": 75.8934,
    "Delivery_location_latitude": 22.80806,
    "Delivery_location_longitude": 75.9534,
    "distance_km": 9.074744674241003,
    "distance_bands": "long",
    "pickup_hour": 21,
    "pickup_minute": 59,
    "pickup_day_of_week": 3,
    "pickup_is_weekend": 0,
    "pickup_time_period": "night",
    "pickup_hour_sin": -0.7071067811865477,
    "pickup_hour_cos": 0.7071067811865474,
    "is_peak_pickup": 1,
    "Weatherconditions": "Sandstorms",
    "Road_traffic_density": "Jam",
    "Vehicle_condition": 0,
    "Type_of_order": "Drinks",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0,
    "Festival": "No",
    "City": "Metropolitian",
}

def test_health_endpoint() :
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()
    assert body['status'] == 'healthy'
    assert body["eta_model_loaded"] is True
    assert body["classifier_loaded"] is True

def test_valid_prediction () :
    response  = client.post("/predict",json = VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()

    # Check that the API returns the expected fields.
    assert "predicted_eta_minutes" in body
    assert "late_probability" in body
    assert "risk_level" in body
    assert "recommendation" in body

    # Check that predictions are within sensible output ranges.
    assert isinstance(body["predicted_eta_minutes"], (int, float))
    assert body["predicted_eta_minutes"] >= 0

    assert 0 <= body["late_probability"] <= 1

    assert body["risk_level"] in {
        "Low Risk",
        "Medium Risk",
        "High Risk",
    }


def test_invalid_pickup_hour():
    payload = VALID_PAYLOAD.copy()
    payload["pickup_hour"] = 25

    response = client.post("/predict", json=payload)

    # Pydantic should reject an hour outside 0–23.
    assert response.status_code == 422


def test_missing_required_field():
    payload = VALID_PAYLOAD.copy()
    del payload["Delivery_person_Age"]

    response = client.post("/predict", json=payload)

    # The required age field is missing.
    assert response.status_code == 422

