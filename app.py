```python id="m0shv9"
from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load model
model = joblib.load("sevencore_model.pkl")


@app.get("/")
def home():
    return {"message": "Sevencore ML API Running"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df).max()

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
```
