from fastapi import APIRouter, Depends

from api.app.schemas.request import TransactionRequest
from api.app.schemas.response import PredictionResponse
from api.app.services.predictor import Predictor, get_predictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest, predictor: Predictor = Depends(get_predictor)):
    return predictor.predict(tx)
