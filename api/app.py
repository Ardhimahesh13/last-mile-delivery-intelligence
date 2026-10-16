
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Resolve model paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

ETA_MODEL_PATH = MODEL_DIR / "eta_model.joblib"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "late_delivery_classifier.joblib"


# --------------------------------------------------
# 2. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Last-Mile Delivery Intelligence API",
    description="ETA prediction and late-delivery risk assessment",
    version="1.1.0",
)


# --------------------------------------------------
# 3. Load models when the application module starts
# --------------------------------------------------

if not ETA_MODEL_PATH.is_file():
    raise FileNotFoundError(f"ETA model not found: {ETA_MODEL_PATH}")

if not CLASSIFIER_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Classifier model not found: {CLASSIFIER_MODEL_PATH}"
    )

eta_model = joblib.load(ETA_MODEL_PATH)
late_classifier = joblib.load(CLASSIFIER_MODEL_PATH)


# --------------------------------------------------
# 4. Validate incoming delivery features
# --------------------------------------------------

class DeliveryFeatures(BaseModel):
    Delivery_person_Age: float = Field(ge=18, le=100)
    Delivery_person_Ratings: float = Field(ge=0, le=5)

    Restaurant_latitude: float = Field(ge=-90, le=90)
    Restaurant_longitude: float = Field(ge=-180, le=180)
    Delivery_location_latitude: float = Field(ge=-90, le=90)
    Delivery_location_longitude: float = Field(ge=-180, le=180)

    distance_km: float = Field(ge=0)
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


# --------------------------------------------------
# 5. Health-check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "eta_model_loaded": eta_model is not None,
        "classifier_loaded": late_classifier is not None,
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_delivery(features: DeliveryFeatures):
    try:
        # Convert validated request data into a one-row DataFrame.
        input_df = pd.DataFrame([features.model_dump()])

        # Predict ETA in minutes.
        eta_output = eta_model.predict(input_df)
        predicted_eta = float(eta_output[0])

        # Find the probability column corresponding to the late class.
        classes = list(late_classifier.classes_)

        # This assumes the training target encodes late delivery as 1.
        if 1 not in classes:
            raise RuntimeError(
                "The classifier's late-class label could not be verified. "
                "Check the target encoding used during training."
            )

        late_class_index = classes.index(1)

        probability_output = late_classifier.predict_proba(input_df)
        late_probability = float(
            probability_output[0, late_class_index]
        )

        # Validate model outputs before returning them.
        if not 0.0 <= late_probability <= 1.0:
            raise RuntimeError("Classifier returned an invalid probability.")

        if not pd.notna(predicted_eta):
            raise RuntimeError("ETA model returned an invalid prediction.")

        # Apply the existing business risk thresholds.
        if late_probability < 0.30:
            risk_level = "Low Risk"
            recommendation = "Normal monitoring"
        elif late_probability < 0.60:
            risk_level = "Medium Risk"
            recommendation = "Monitor delivery closely"
        else:
            risk_level = "High Risk"
            recommendation = "Prioritize intervention"

        return {
            "predicted_eta_minutes": predicted_eta,
            "late_probability": late_probability,
            "risk_level": risk_level,
            "recommendation": recommendation,
        }

    except HTTPException:
        raise

    except Exception as exc:
        # Log the technical error for debugging without exposing
        # internal model details to the API caller.
        print(f"Prediction failed: {type(exc).__name__}: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal processing error.",
        ) from exc
