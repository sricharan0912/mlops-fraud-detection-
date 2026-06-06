from typing import Literal

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    step: int = Field(..., description="Time step in the simulation")
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(..., gt=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    model_config = {"json_schema_extra": {"example": {
        "step": 1, "type": "TRANSFER", "amount": 181.0,
        "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }}}
