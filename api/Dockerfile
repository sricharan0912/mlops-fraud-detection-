FROM python:3.12-slim

WORKDIR /app

# libgomp required by LightGBM; libgomp1 ships with gcc on debian slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps before copying source — preserves layer cache on code changes
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi uvicorn[standard] pydantic pydantic-settings \
        scikit-learn xgboost lightgbm imbalanced-learn \
        pandas numpy mlflow psycopg2-binary boto3 sqlalchemy

# Copy source
COPY api/ ./api/
COPY training/ ./training/
COPY models/ ./models/

ENV MODEL_PATH=models/lgbm.pkl
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
