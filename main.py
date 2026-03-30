from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("peak_model.pkl")

@app.get("/")
def home():
    return {"message": "UV Model API Running 🚀"}

@app.post("/predict")
def predict(data: dict):
    peak_abs = data.get("peak_abs")

    if peak_abs is None:
        return {"error": "peak_abs is required"}

    try:
        peak_abs = float(peak_abs)
    except:
        return {"error": "Invalid peak_abs value"}

    pred_log = model.predict([[peak_abs]])[0]
    pred_n = 10**pred_log
    log_conc = -2 * pred_n

    return {
        "peak_abs": peak_abs,
        "log_potency": float(pred_log),
        "potency": float(pred_n),
        "log_concentration": float(log_conc)
    }