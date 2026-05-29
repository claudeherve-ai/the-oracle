# The Oracle — Predictive Intelligence Engine
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY oracle/ oracle/
COPY pyproject.toml .
RUN pip install -e . --no-deps

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "oracle.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
