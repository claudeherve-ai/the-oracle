# THE ORACLE

> **"Specific, verifiable, time-bound predictions with calibrated confidence."**

A predictive intelligence engine that ingests global signals, generates specific forecasts, tracks every outcome, and displays its accuracy publicly. Not vague fortune-telling — every prediction has a deadline and gets resolved.

## Quick Start

```bash
# Install
cd ~/the-oracle
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]" --break-system-packages

# Set your LLM key
export OPENAI_API_KEY="sk-..."

# Start the server
oracle server &

# Generate predictions
oracle predict "Will Apple announce new hardware at WWDC 2026?"

# Auto-scan from current signals
oracle scan

# View calibration
oracle calibration

# Open the Matrix-themed dashboard
oracle dashboard
```

## API

```
POST   /v1/predict              Generate predictions
POST   /v1/predict/query        Predict for a specific question
GET    /v1/predictions           List predictions
GET    /v1/predictions/{id}      Get prediction details
POST   /v1/predictions/{id}/resolve  Mark as correct/incorrect
GET    /v1/calibration           Calibration report
GET    /v1/signals               Recent signals
POST   /v1/ingest                Trigger fresh ingestion
GET    /v1/dashboard             Dashboard data
```

## Python SDK

```python
import oracle

# Generate predictions
predictions = oracle.predict("Will NVDA hit $200 by July?")
for p in predictions:
    print(f"{p['statement']} — {p['confidence']:.0%} confidence")

# View your track record
cal = oracle.calibration()
print(f"Accuracy: {cal['overall_accuracy']:.1%}")
```

## Calibration Dashboard

Matrix-themed (green on black) real-time dashboard at `/dashboard/index.html`:
- Overall accuracy score
- Calibration curve (reliability diagram)
- Breakdown by confidence bucket
- Breakdown by category
- Recent predictions feed

## Architecture

```
Ingestion → Signal Extraction → Prediction Engine → Calibration Tracker
  (5 sources)    (NLP/patterns)    (LLM-powered)     (accuracy proof)
```

## Azure Deployment

```bash
az acr build --registry acroracle --image oracle:v1 .
az containerapp update --name the-oracle -g rg-oracle-prod \
  --image acroracle.azurecr.io/oracle:v1 \
  --env-vars OPENAI_API_KEY=... ORACLE_DB=postgresql://... \
  --ingress external --target-port 8001
```

## License

MIT
