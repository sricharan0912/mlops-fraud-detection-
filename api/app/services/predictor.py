import functools
import pickle

import pandas as pd

from api.app.schemas.request import TransactionRequest
from api.app.schemas.response import PredictionResponse


class Predictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        # Supports both pipeline dicts (xgb/lgbm) and sklearn Pipeline objects (baseline)
        if isinstance(artifact, dict):
            self.preprocessor = artifact.get("preprocessor")
            self.clf = artifact.get("clf") or artifact.get("pipeline")
            self.threshold = float(artifact["threshold"])
            self.model_name = artifact.get("model", "unknown")
        else:
            self.preprocessor = None
            self.clf = artifact
            self.threshold = 0.5
            self.model_name = "unknown"

    def predict(self, tx: TransactionRequest) -> PredictionResponse:
        row = pd.DataFrame([tx.model_dump()])
        row["balance_diff_orig"] = row["oldbalanceOrg"] - row["newbalanceOrig"]
        row["balance_diff_dest"] = row["newbalanceDest"] - row["oldbalanceDest"]

        assert self.clf is not None, "Model not loaded"
        if self.preprocessor is not None:
            X = self.preprocessor.transform(row)
            prob = float(self.clf.predict_proba(X)[0, 1])
        else:
            # sklearn Pipeline with built-in preprocessor
            prob = float(self.clf.predict_proba(row)[0, 1])

        return PredictionResponse(
            fraud_probability=round(prob, 6),
            is_fraud=prob >= self.threshold,
            threshold=self.threshold,
            model_name=self.model_name,
        )


@functools.lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    from api.app.config import settings
    return Predictor(settings.model_path)
