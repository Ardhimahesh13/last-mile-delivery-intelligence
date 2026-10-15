from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

# resolve model paths relative to the project Directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / 'models'

ETA_MODEL_PATH = MODEL_DIR / "eta_model.joblib"
CLASSIFIER_MODEL_PATH = (
    MODEL_DIR / 'late_delivery_classifier.joblib'
)

app = FastAPI(
    title='Last-Mile Delivery Intelligence API',
    description="ETA prediction and late-delivery risk assessment",
    version='1.0.0'
)

# Load models once api application starts
if not ETA_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ETA model not found: {ETA_MODEL_PATH}"
    )

if not CLASSIFIER_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Classifier not found: {CLASSIFIER_MODEL_PATH}"
    )

eta_model = joblib.load(ETA_MODEL_PATH)
late_classifier = joblib.load(CLASSIFIER_MODEL_PATH)

class DeliveryFeatures(BaseModel) :
    Delivery_person_Age : float = Field(ge = 18,le=100)
    Delivery_person_Ratings : float = Field(ge=0,le=5)
    Restaurant_latitude : float
    Restaurant_longitude : float
    Delivery_location_latitude : float
    Delivery_location_longitude : float

    distance_km : float = Field(ge = 0)

    distance_bands: str
    pickup_hour: int = Field(ge=0, le=23)
    pickup_minute: int = Field(ge=0, le=59)
    pickup_day_of_week: int = Field(ge=0, le=6)
    pickup_is_weekend: int = Field(ge=0, le=1)
    pickup_time_period: str
    pickup_hour_sin: float
    pickup_hour_cos: float
    is_peak_pickup: int = Field(ge=0, le=1)

    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int = Field(ge=0)
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: int = Field(ge=0)
    Festival: str
    City: str


@app.get("/health")
def health_check() :
    return {
        "status" : "healthy",
        "eta_model_loaded" : eta_model is not None,
        "classifier_loaded" : late_classifier is not None
    }





@app.post("/predict")
def predict_delivery(features: DeliveryFeatures):
    try:
        # 1. Convert validated request data into a DataFrame.
        input_df = pd.DataFrame([features.model_dump()])


        # 2. Predict delivery time in minutes.
        eta_output = eta_model.predict(input_df)
        predicted_eta = float(eta_output[0])

        # 3. Predict probability of late delivery.
        probability_output = late_classifier.predict_proba(input_df)
        late_probability = float(probability_output[0, 1])

        # 4. Map probability to a business risk level.
        if late_probability < 0.30:
            risk_level = "Low Risk"
            recommendation = "Normal monitoring"
        elif late_probability < 0.60:
            risk_level = "Medium Risk"
            recommendation = "Monitor delivery closely"
        else:
            risk_level = "High Risk"
            recommendation = "Prioritize intervention"

        # 5. Return the prediction response.
        return {
            "predicted_eta_minutes": predicted_eta,
            "late_probability": late_probability,
            "risk_level": risk_level,
            "recommendation": recommendation
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal processing error."
        ) from exc




# Comparing API and notebook predictions 
