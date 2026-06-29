from fastapi import APIRouter, HTTPException

from app.schemas.prediction import CaseInput, ModelInfoResponse, PredictionResponse
from app.services import inference_service
from app.services.inference_service import ModelNotTrainedError

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(case: CaseInput) -> PredictionResponse:
    try:
        result = inference_service.run_prediction(case.model_dump())
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**inference_service.get_model_info())
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
