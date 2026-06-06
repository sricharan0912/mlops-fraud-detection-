FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[dev]"

COPY drift/ ./drift/
COPY training/ ./training/
COPY data/processed/ ./data/processed/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "drift/monitor.py"]
