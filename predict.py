"""
Standalone prediction CLI and module for Flood Severity & Inundation Risk.
Loads the benchmark-winning model trained on the Power BI cleaned dataset.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_flood_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Model artifacts not found in {MODELS_DIR}. Run train_models.py first."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def get_recommendation(severity_level, risk_score_pct):
    sev = severity_level.upper()
    if sev in ["CRITICAL", "SEVERE"]:
        return {
            "status_code": "RED_ALERT",
            "boats_recommended": 4,
            "evacuation_priority": "Immediate Emergency Evacuation",
            "action": "Deploy 4-6 Motorized Inflatable Rescue Boats and Diving Units immediately. Prepare rooftop winching and clear downstream causeways.",
            "shelter_advisory": "Relocate residents to high-elevation relief shelters (>25m elevation).",
        }
    elif sev == "HIGH":
        return {
            "status_code": "ORANGE_ALERT",
            "boats_recommended": 2,
            "evacuation_priority": "High Priority Ground-Floor Evacuation",
            "action": "Deploy 2-3 Rapid Rescue Rafts, station high-capacity dewatering pumps, and prioritize vulnerable residents.",
            "shelter_advisory": "Advise ground floor residents to move to first floor or designated relief centers.",
        }
    elif sev in ["MEDIUM", "MODERATE"]:
        return {
            "status_code": "YELLOW_ALERT",
            "boats_recommended": 1,
            "evacuation_priority": "Precautionary Monitoring",
            "action": "Deploy 1 Standby Patrol Boat, activate storm pump stations, and monitor dam inflow rates hourly.",
            "shelter_advisory": "Keep emergency go-bags ready, move vehicles and cattle to higher ground.",
        }
    else:
        return {
            "status_code": "GREEN_SAFE",
            "boats_recommended": 0,
            "evacuation_priority": "Normal Standby",
            "action": "Low flood hazard. Routine stormwater monitoring and clear street drains.",
            "shelter_advisory": "No evacuation required. Stay tuned to weather advisories.",
        }


def predict(input_data):
    model, preprocessor = load_artifacts()

    scaler = preprocessor["scaler"]
    label_encoder = preprocessor["label_encoder"]
    env_feature_cols = preprocessor["env_feature_cols"]
    all_feature_cols = preprocessor["all_feature_cols"]
    winner_name = preprocessor.get("winner_name", "Random Forest")
    use_scaled = preprocessor.get("use_scaled", False)
    risk_regressor = preprocessor.get("risk_regressor")

    # Handle sensor slider inputs and normalize feature names
    rain_val = float(input_data.get("rainfall_24h_mm") or input_data.get("rainfall_mm") or 120.0)
    river_val = float(input_data.get("river_water_level_m") or input_data.get("water_level_m") or 3.0)
    elev_val = float(input_data.get("elevation_m") or 15.0)
    soil_val = float(input_data.get("soil_saturation_pct") or input_data.get("soil_moisture_pct") or 65.0)
    drain_val = float(input_data.get("drainage_capacity_pct") or 50.0)
    dam_val = float(input_data.get("dam_discharge_cumecs") or input_data.get("discharge_cumecs") or 800.0)
    dist_val = float(input_data.get("distance_to_waterbody_m") or 500.0)
    pop_val = float(input_data.get("population_density") or 5000.0)

    # Convert distance to km if meters given
    dist_km = dist_val / 1000.0 if dist_val > 50 else dist_val

    # Convert drainage capacity to embankment condition (0: Good, 1: Moderate, 2: Poor)
    if drain_val > 65:
        embankment_code = 0
    elif drain_val > 35:
        embankment_code = 1
    else:
        embankment_code = 2

    # Baseline warning and danger marks (aligned with cleaned dataset averages)
    danger_mark = 7.0
    warning_mark = 5.8

    # The slider 'river_water_level_m' represents water height relative to flood danger mark (-2.0m to +5.0m)
    abs_water_level = danger_mark + river_val

    upstream_rain = float(input_data.get("upstream_rainfall_mm") or (rain_val * 0.75))
    forecast_rain = float(input_data.get("forecast_rainfall_next24_mm") or (rain_val * 0.85))
    total_rain = rain_val + upstream_rain + forecast_rain

    discharge_base = dam_val * 0.65
    reservoir_release = dam_val * 0.35
    total_discharge = discharge_base + reservoir_release

    water_danger_diff = abs_water_level - danger_mark
    water_warning_diff = abs_water_level - warning_mark
    water_danger_ratio = abs_water_level / (danger_mark + 1e-5)
    water_warning_ratio = abs_water_level / (warning_mark + 1e-5)
    danger_exceeded = 1 if abs_water_level >= danger_mark else 0
    warning_exceeded = 1 if abs_water_level >= warning_mark else 0

    feature_dict = {
        "rainfall_mm": rain_val,
        "upstream_rainfall_mm": upstream_rain,
        "forecast_rainfall_next24_mm": forecast_rain,
        "total_rainfall": total_rain,
        "water_level_m": abs_water_level,
        "warning_level_m": warning_mark,
        "danger_level_m": danger_mark,
        "water_danger_diff": water_danger_diff,
        "water_warning_diff": water_warning_diff,
        "water_danger_ratio": water_danger_ratio,
        "water_warning_ratio": water_warning_ratio,
        "danger_exceeded": danger_exceeded,
        "warning_exceeded": warning_exceeded,
        "discharge_cumecs": discharge_base,
        "reservoir_release_cumecs": reservoir_release,
        "total_discharge": total_discharge,
        "temperature_c": float(input_data.get("temperature_c") or 27.5),
        "humidity_pct": float(input_data.get("humidity_pct") or 80.0),
        "soil_moisture_pct": soil_val,
        "wind_speed_kmph": float(input_data.get("wind_speed_kmph") or 12.0),
        "embankment_encoded": embankment_code,
        "historical_flood_freq": int(input_data.get("historical_flood_freq") or 2),
        "distance_to_river_km": dist_km,
        "elevation_m": elev_val,
        "population_density": pop_val,
        "month": int(input_data.get("month") or 9),
    }

    env_df = pd.DataFrame([feature_dict])[env_feature_cols]

    # Calculate continuous flood risk score
    if "flood_risk_score" in input_data:
        calculated_risk_score = float(input_data["flood_risk_score"])
    elif risk_regressor is not None:
        calculated_risk_score = float(risk_regressor.predict(env_df)[0])
    else:
        calculated_risk_score = 0.5

    feature_dict["flood_risk_score"] = calculated_risk_score

    full_df = pd.DataFrame([feature_dict])[all_feature_cols]
    X_input = scaler.transform(full_df) if use_scaled else full_df

    # Predict severity class
    pred_encoded = model.predict(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Predict class probabilities
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        for cls_name, prob in zip(label_encoder.classes_, probs):
            probabilities[cls_name] = round(float(prob) * 100.0, 2)
    else:
        probabilities[pred_label] = 100.0

    # Ensure UI-compatible aliases: Moderate=Medium, Severe=Critical
    if "Medium" in probabilities:
        probabilities["Moderate"] = probabilities["Medium"]
    elif "Moderate" in probabilities:
        probabilities["Medium"] = probabilities["Moderate"]

    if "Critical" in probabilities:
        probabilities["Severe"] = probabilities["Critical"]
    elif "Severe" in probabilities:
        probabilities["Critical"] = probabilities["Severe"]

    # Calculate 0-100% Risk Index
    # Scaled from calculated_risk_score and severity probabilities
    raw_risk_pct = calculated_risk_score * 100.0
    if pred_label == "Critical":
        risk_index = max(raw_risk_pct, 82.0)
    elif pred_label == "High":
        risk_index = min(max(raw_risk_pct, 65.0), 84.9)
    elif pred_label == "Medium":
        risk_index = min(max(raw_risk_pct, 45.0), 64.9)
    else:
        risk_index = min(raw_risk_pct, 44.9)

    risk_index = round(min(100.0, max(0.0, risk_index)), 1)
    confidence = probabilities.get(pred_label, 95.0)
    recommendation = get_recommendation(pred_label, risk_index)

    output = {
        "success": True,
        "model_used": winner_name,
        "predicted_severity": pred_label,
        "risk_index": risk_index,
        "confidence_pct": confidence,
        "class_probabilities": probabilities,
        "recommendation": recommendation,
        "input_features": {
            "rainfall_24h_mm": rain_val,
            "river_water_level_m": river_val,
            "elevation_m": elev_val,
            "soil_saturation_pct": soil_val,
            "dam_discharge_cumecs": dam_val,
            "drainage_capacity_pct": drain_val,
            "flood_risk_score": round(calculated_risk_score, 3),
        },
    }

    return output


def main():
    try:
        raw_input = ""
        if len(sys.argv) > 1:
            arg = sys.argv[1].strip()
            if arg == "--stdin":
                raw_input = sys.stdin.read().strip()
            else:
                raw_input = arg

        if not raw_input:
            # Default test sample (Severe Monsoon Scenario)
            sample = {
                "rainfall_24h_mm": 240.0,
                "river_water_level_m": 3.4,
                "elevation_m": 6.0,
                "soil_saturation_pct": 92.0,
                "distance_to_waterbody_m": 150.0,
                "drainage_capacity_pct": 18.0,
                "dam_discharge_cumecs": 2600.0,
                "population_density": 8200.0,
            }
            res = predict(sample)
        else:
            data = json.loads(raw_input)
            res = predict(data)

        print(json.dumps(res, indent=2))
    except Exception as e:
        err_res = {"success": False, "error": str(e)}
        print(json.dumps(err_res))
        sys.exit(1)


if __name__ == "__main__":
    main()
