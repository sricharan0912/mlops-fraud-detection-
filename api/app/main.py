from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.app.routers import health, predict
from api.app.services.predictor import get_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()  # warm up model at startup
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction risk scoring",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["ops"])
app.include_router(predict.router, tags=["inference"])
