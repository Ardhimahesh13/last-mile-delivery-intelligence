
import math
from datetime import date

import requests
import streamlit as st


# --------------------------------------------------
# 1. FastAPI backend endpoint
# --------------------------------------------------

API_URL = "https://last-mile-delivery-intelligence.onrender.com/predict"


# --------------------------------------------------
# 2. Configure the Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Last-Mile Delivery Intelligence",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Last-Mile Delivery Intelligence")
st.write(
    "Estimate delivery time and assess the risk of a late delivery."
)
st.info("Start the FastAPI backend before requesting a prediction.")


# --------------------------------------------------
# 3. Calculate geographic distance using Haversine
# --------------------------------------------------

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance between two coordinates in km."""

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2],
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    # Protect against floating-point rounding errors.
    a = min(1.0, max(0.0, a))

    earth_radius_km = 6371.0

    return earth_radius_km * 2 * math.asin(math.sqrt(a))


# --------------------------------------------------
# 4. Derive distance band using training thresholds
# --------------------------------------------------

def get_distance_band(distance_km):
    if distance_km <= 1:
        return "very_short"
    elif distance_km <= 3:
        return "short"
    elif distance_km <= 5:
        return "medium"
    elif distance_km <= 10:
        return "long"
    else:
        return "very_long"


# --------------------------------------------------
# 5. Derive time period
# --------------------------------------------------

def get_time_period(hour):
    """
    Use the time-period boundaries identified for the
    training notebook.
    """

    if 5 <= hour < 11:
        return "morning"
    elif 11 <= hour < 15:
        return "afternoon"
    elif 15 <= hour < 19:
        return "evening"
    elif 19 <= hour < 24:
        return "night"
    else:
        return "late_night"


# --------------------------------------------------
# 6. Collect delivery details
# --------------------------------------------------

with st.form("delivery_form"):
    st.subheader("Delivery Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Delivery Person Age",
            min_value=18,
            max_value=70,
            value=36,
        )

        rating = st.number_input(
            "Delivery Person Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
        )

        pickup_date = st.date_input(
            "Pickup Date",
            value=date.today(),
        )

        pickup_hour = st.slider(
            "Pickup Hour",
            min_value=0,
            max_value=23,
            value=21,
        )

        pickup_minute = st.slider(
            "Pickup Minute",
            min_value=0,
            max_value=59,
            value=59,
        )

        weather = st.selectbox(
            "Weather",
            [
                "Sandstorms",
                "Cloudy",
                "Fog",
                "Stormy",
                "Sunny",
                "Windy",
            ],
        )

        traffic = st.selectbox(
            "Road Traffic",
            ["Jam", "High", "Medium", "Low"],
        )

    with col2:
        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            [0, 1, 2],
        )

        order_type = st.selectbox(
            "Order Type",
            ["Drinks", "Meal", "Snack", "Buffet"],
        )

        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["motorcycle", "scooter", "electric_scooter"],
        )

        multiple_deliveries = st.selectbox(
            "Multiple Deliveries",
            [0, 1, 2, 3],
        )

        festival = st.selectbox(
            "Festival",
            ["No", "Yes"],
        )

        city = st.selectbox(
            "City",
            ["Metropolitian", "Urban", "Semi-Urban"],
        )

    # --------------------------------------------------
    # 7. Collect restaurant and delivery coordinates
    # --------------------------------------------------

    st.subheader("Location Coordinates")

    loc1, loc2 = st.columns(2)

    with loc1:
        restaurant_lat = st.number_input(
            "Restaurant Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=22.74806,
            format="%.6f",
        )

        restaurant_lon = st.number_input(
            "Restaurant Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.8934,
            format="%.6f",
        )

    with loc2:
        delivery_lat = st.number_input(
            "Delivery Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=22.80806,
            format="%.6f",
        )

        delivery_lon = st.number_input(
            "Delivery Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=75.9534,
            format="%.6f",
        )

    submitted = st.form_submit_button(
        "Predict Delivery",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# 8. Engineer features and request a prediction
# --------------------------------------------------

if submitted:

    # Calculate distance from the coordinates.
    distance = haversine_distance_km(
        restaurant_lat,
        restaurant_lon,
        delivery_lat,
        delivery_lon,
    )

    # Derive the distance band automatically.
    distance_band = get_distance_band(distance)

    # Calendar features: Monday=0, Sunday=6.
    pickup_day_of_week = pickup_date.weekday()
    pickup_is_weekend = int(pickup_day_of_week >= 5)

    # Cyclical encoding of pickup hour.
    hour_sin = math.sin(2 * math.pi * pickup_hour / 24)
    hour_cos = math.cos(2 * math.pi * pickup_hour / 24)

    # Time period.
    time_period = get_time_period(pickup_hour)

    # Peak-hour rule from the training notebook.
    is_peak_pickup = int(
        pickup_hour in [11, 12, 13, 14, 19, 20, 21, 22]
    )

    # Construct the 24-feature API payload.
    payload = {
        "Delivery_person_Age": float(age),
        "Delivery_person_Ratings": float(rating),
        "Restaurant_latitude": restaurant_lat,
        "Restaurant_longitude": restaurant_lon,
        "Delivery_location_latitude": delivery_lat,
        "Delivery_location_longitude": delivery_lon,
        "distance_km": distance,
        "distance_bands": distance_band,
        "pickup_hour": pickup_hour,
        "pickup_minute": pickup_minute,
        "pickup_day_of_week": pickup_day_of_week,
        "pickup_is_weekend": pickup_is_weekend,
        "pickup_time_period": time_period,
        "pickup_hour_sin": hour_sin,
        "pickup_hour_cos": hour_cos,
        "is_peak_pickup": is_peak_pickup,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival,
        "City": city,
    }

    # Display engineered features for transparency.
    st.subheader("Calculated Features")

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    feature_col1.metric(
        "Calculated Distance",
        f"{distance:.2f} km",
    )

    feature_col2.metric(
        "Distance Band",
        distance_band,
    )

    feature_col3.metric(
        "Pickup Time Period",
        time_period,
    )

    st.caption(
        f"Pickup day: {pickup_date.strftime('%A')} | "
        f"Peak pickup: {'Yes' if is_peak_pickup else 'No'}"
    )

    # Send the features to FastAPI.
    try:
        with st.spinner("Getting delivery prediction..."):
            response = requests.post(
                API_URL,
                json=payload,
                timeout=30,
            )

        if response.ok:
            result = response.json()

            # Validate expected response fields.
            eta = float(result["predicted_eta_minutes"])
            late_probability = float(result["late_probability"])
            risk_level = result["risk_level"]
            recommendation = result["recommendation"]

            if not 0 <= late_probability <= 1:
                raise ValueError(
                    "The API returned an invalid probability."
                )

            st.subheader("Prediction Results")

            result_col1, result_col2, result_col3 = st.columns(3)

            result_col1.metric(
                "Estimated Delivery Time",
                f"{eta:.1f} min",
            )

            result_col2.metric(
                "Late-Delivery Probability",
                f"{late_probability:.1%}",
            )

            result_col3.metric(
                "Risk Level",
                risk_level,
            )

            st.write("**Recommended Action**")
            st.info(recommendation)

            with st.expander("View submitted features"):
                st.json(payload)

        else:
            st.error(
                f"API returned HTTP {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Start the backend in another "
            "terminal using: uvicorn api.app:app --reload"
        )

    except requests.exceptions.Timeout:
        st.error(
            "The API request timed out. Please try again."
        )

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

    except (KeyError, TypeError, ValueError) as exc:
        st.error(
            f"The API response was invalid or incomplete: {exc}"
        )
