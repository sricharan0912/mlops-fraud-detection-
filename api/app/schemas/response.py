from pydantic import BaseModel


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_name: str = "lgbm"
