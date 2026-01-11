# Define Request and Response Model
from pydantic import BaseModel

class ReviewRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing!"
            }
        }

class PredictionResponse(BaseModel):
    original_text: str
    cleaned_text: str
    sentiment: str
    confidence: float
    timestamp: str

