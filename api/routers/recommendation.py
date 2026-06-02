from fastapi import APIRouter, Depends, HTTPException

from api.schemas import RecommendationRequest, SimpleAIResponse
from api.deps import get_recommendation_service
from llm.recommendation import SkinRecommendationService


router = APIRouter()


@router.post("/generate", response_model=SimpleAIResponse)
def generate_recommendation(
    request: RecommendationRequest,
    service: SkinRecommendationService = Depends(get_recommendation_service),
):
    try:
        result = service.generate_recommendation(
            predicted_label=request.predicted_label,
            confidence=request.confidence,
        )

        return {
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "recommendation": result["recommendation"],
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error while generating recommendation",
        )