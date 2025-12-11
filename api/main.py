from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
import re
import string

with open("model/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

app = FastAPI(title="Finance Sentiment Analysis API")

class SentimentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: SentimentRequest):
    cleaned = clean_text(data.text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]

    reverse_map = {0:"negative", 1:"neutral", 2:"positive"}

    return {
        "input": data.text,
        "sentiment": reverse_map[pred]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
