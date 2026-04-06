from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

# Define proper input format
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([data.features])
    return {"predicted_price": float(prediction[0])}