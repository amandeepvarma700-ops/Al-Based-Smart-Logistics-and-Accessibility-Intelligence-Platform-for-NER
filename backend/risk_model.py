import joblib
import pandas as pd

# Load the newly trained model when the server starts
model = joblib.load("xgboost_risk_model.pkl")

def predict_risk(elevation, rainfall_mm, slope_deg):
    try:
        # Format the input exactly how XGBoost expects it
        df_input = pd.DataFrame([{
            "elevation": elevation,
            "rainfall_mm": rainfall_mm,
            "slope_deg": slope_deg
        }])
        
        # Extract the probability of Class 1 (High Risk)
        risk_prob = float(model.predict_proba(df_input)[0][1])
        
        return {
            "status": "success",
            "risk_score": round(risk_prob, 4),
            "is_high_risk": bool(risk_prob > 0.6)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}