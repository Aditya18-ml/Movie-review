from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from preprocessing import preprocess_text

app = FastAPI(title="IMDB Sentiment Analysis API")

model = joblib.load('logistic.joblib')
tfidf = joblib.load('tfidf.joblib')

class ReviewRequest(BaseModel):
    review: str

@app.get("/")
def home():
    return {"message": "IMDB Sentiment Analysis API is running."}

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    
    cleaned = preprocess_text([request.review])    
    vectorized = tfidf.transform(cleaned)
    prediction = model.predict(vectorized)[0]
    
    return {
        "review": request.review,
        "sentiment": prediction
    }