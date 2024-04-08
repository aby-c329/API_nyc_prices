from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load('/Users/alexa/Documents/MLOps /API_nyc_houseprice/model/linear_regression_model_NYC.pkl')

class PredictionRequest(BaseModel):
    beds: int
    bath: int
    property_sqft: float

class PredictionResponse(BaseModel):
    predicted_price: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    prediction = model.predict([[request.beds, request.bath, request.property_sqft]])
    return {"predicted_price": prediction[0]}

