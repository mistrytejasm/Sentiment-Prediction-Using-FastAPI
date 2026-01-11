from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
import re
from datetime import datetime
from sentiment_api.schema import PredictionResponse, ReviewRequest

# Load Train Model
print("Loading Train Model...")
import os
from pathlib import Path

# Get the correct path whether running locally or in Docker
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

# Load model
model_path = MODEL_DIR / "sentiment_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# If you have separate preprocessor
preprocessor_path = MODEL_DIR / "text_preprocessor.pkl"
if preprocessor_path.exists():
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

print("Model Loaded Successfully! :)")

# Initialize Fastapi APP
app = FastAPI(
    title="Sentiment Analysis API",
    description="API To predict sentiment (Positive/Negative) from Text",
    version="0.v1"
)

# Text preprocessing Function
def clean_text(text):
    """Clean text same as traning"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Create API Endpoints
@app.get("/")
def home():
    """Home Endpoint - API Health Check"""
    return {
        "message": "Sentiment Analysis API is running!",
        "status": "healthy",
        "version": "0.v1",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment of given text

    - **text**: The review text to analyze

    Returns sentiment (positive/negative) with confidence score
    """

    try:
        # clean the text
        cleaned = clean_text(request.text)

        # check if text is empty after cleaning
        if not cleaned or len(cleaned.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Text is empty after preprocessing. Please provide valid text."
            )

        # Make Prediction
        prediction = model.predict([cleaned])[0]
        print("Prediction: ",prediction)

        probabilities = model.predict_proba([cleaned])[0]

        # Get sentiment and confidence
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = float(max(probabilities))

        # return Response
        return PredictionResponse(
            original_text = request.text,
            cleaned_text = cleaned,
            sentiment = sentiment,
            confidence = confidence,
            timestamp = datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
def health():
    return {
        "status": "Sentiment Prediction is Running and Healthy",
        "version": "0.v1",
        "timestamp": datetime.now().isoformat()
    }

# Run the server
if __name__ == "__main__":
    print("Starting server....")
    print("API Docs: https://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)