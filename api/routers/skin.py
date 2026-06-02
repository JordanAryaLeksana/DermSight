from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from api.schemas import PredictionResponse, SimpleAIResponse
from api.deps import get_skin_predictor, get_recommendation_service

from api.inference.predictor import SkinDiseasePredictor
from llm.recommendation import SkinRecommendationService


router = APIRouter()


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
}


@router.post("/predict", response_model=PredictionResponse)
async def predict_skin_image(
    image: UploadFile = File(...),
    predictor: SkinDiseasePredictor = Depends(get_skin_predictor),
):
    try:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, and PNG images are supported",
            )

        image_bytes = await image.read()

        prediction = predictor.predict(image_bytes)

        return {
            "predicted_label": prediction["predicted_label"],
            "confidence": prediction["confidence"],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to classify skin image",
        )


@router.post("/analyze", response_model=SimpleAIResponse)
async def analyze_skin_image(
    image: UploadFile = File(...),
    predictor: SkinDiseasePredictor = Depends(get_skin_predictor),
    recommendation_service: SkinRecommendationService = Depends(get_recommendation_service),
):
    try:
        if image.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG, and PNG images are supported",
            )

        image_bytes = await image.read()

        prediction = predictor.predict(image_bytes)

        recommendation_result = recommendation_service.generate_recommendation(
            predicted_label=prediction["predicted_label"],
            confidence=prediction["confidence"],
        )

        return {
            "predicted_label": prediction["predicted_label"],
            "confidence": prediction["confidence"],
            "recommendation": recommendation_result["recommendation"],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze skin image",
        )