import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import os 
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# Initializations
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/model")
model = tf.saved_model.load(MODEL_PATH)
encoder = SentenceTransformer('all-mpnet-base-v2')
label_map = {1 : "Positive", 0: "Negative"}

# Input Schema for API payload
class TextInput(BaseModel):
    text: str

# Predicting Endpoint
@app.post("/predict/")
def predict_sentiment(input_data: TextInput):
    encoded_ = encoder.encode([input_data.text])
    prediction_prob = model.signatures["serving_default"]\
        (tf.constant(encoded_))['output_0'][0][0]
    predicted_label = 1 if prediction_prob > 0.5 else 0
    mapped_label = label_map[predicted_label]
    return {"text": input_data, "propbability":round(float(prediction_prob),2),\
             "sentiment": mapped_label}