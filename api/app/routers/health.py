from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    from api.app.services.predictor import get_predictor
    get_predictor()  # will raise if model can't load
    return {"status": "ready"}
