import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(input_data: TextInput):
    text = "Test"
    sentiment = "Test"
    return {"text": text, "sentiment": sentiment}

# Run using: uvicorn filename:app --reload
"""curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This product is amazing!"}'"""