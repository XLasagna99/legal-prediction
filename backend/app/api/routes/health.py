from fastapi import APIRouter

from app.services import inference_service

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": inference_service.is_ready()}
