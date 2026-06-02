from pydantic import BaseModel, Field

class RecommendationRequest(BaseModel):
    predicted_label: str = Field(..., example="acne")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.85)
    
class SimpleAIResponse(BaseModel):
    predicted_label: str
    confidence: float
    recommendation: str
    
class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float