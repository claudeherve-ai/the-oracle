# THE ORACLE — Design Document

> **"Specific, verifiable, time-bound predictions with calibrated confidence."**
> **v0.1.0 — May 2026**

## Product Vision

A predictive intelligence engine that ingests global signals, generates specific forecasts with confidence scores, tracks every outcome, and displays its accuracy publicly. Not vague fortune-telling — every prediction has a deadline and gets resolved. The calibration dashboard proves it works.

**"Holy shit" moment:** "The Oracle predicted Company X's product launch 14 days before it happened at 78% confidence." The dashboard shows this prediction alongside 500 others — 73% of its 70-80% confidence predictions were correct. That's a calibrated oracle.

## Architecture

```
                      ┌────────────────────────────┐
                      │        THE ORACLE           │
                      │                            │
  ┌──────────┐       │  ┌──────────────────────┐  │
  │ REST API │──────▶│  │   Ingestion Engine    │  │
  └──────────┘       │  │   (news, social,      │  │
                      │  │    financial, tech)   │  │
  ┌──────────┐       │  └──────────┬───────────┘  │
  │   CLI    │──────▶│             │              │
  └──────────┘       │             ▼              │
                      │  ┌──────────────────────┐  │
  ┌──────────┐       │  │   Signal Extraction   │  │
  │ Python   │──────▶│  │   (NLP, patterns,     │  │
  │ SDK      │       │  │    anomaly detection) │  │
  └──────────┘       │  └──────────┬───────────┘  │
                      │             │              │
  ┌──────────┐       │             ▼              │
  │Dashboard │──────▶│  ┌──────────────────────┐  │
  │ (HTML)   │       │  │   Prediction Engine   │  │
  └──────────┘       │  │   (LLM + ensemble)    │  │
                      │  └──────────┬───────────┘  │
                      │             │              │
                      │             ▼              │
                      │  ┌──────────────────────┐  │
                      │  │  Calibration System   │  │
                      │  │  (track → adjust →    │  │
                      │  │   prove accuracy)     │  │
                      │  └──────────────────────┘  │
                      │                            │
                      │  Storage: SQLite            │
                      └────────────────────────────┘
```

## Data Models

### Prediction
```python
class Prediction:
    id: str
    category: Category        # TECH_TREND, MARKET_MOVE, PRODUCT_LAUNCH, etc.
    statement: str            # The actual prediction text
    confidence: float         # 0.0 - 1.0
    reasoning: str            # Why the model made this prediction
    sources: List[str]        # URLs/data sources used
    deadline: datetime        # When this prediction resolves
    status: Status            # PENDING, CORRECT, INCORRECT, EXPIRED
    resolution: Optional[str] # What actually happened
    resolved_at: Optional[datetime]
    created_at: datetime
```

### Signal
```python
class Signal:
    id: str
    source: str               # twitter, reddit, hackernews, yfinance, etc.
    content: str              # Raw text/content
    entities: List[str]       # Companies, people, products mentioned
    sentiment: float          # -1.0 to 1.0
    relevance: float          # 0.0 to 1.0 — how relevant to predictions
    metadata: dict
    captured_at: datetime
```

### Calibration
```python
class CalibrationRecord:
    category: Category
    confidence_bucket: str    # "0.5-0.6", "0.6-0.7", etc.
    total: int
    correct: int
    accuracy: float           # correct / total
    brier_score: float        # Statistical calibration measure
```

## Prediction Categories

| Category | Example | Timeframe |
|----------|---------|-----------|
| TECH_TREND | "React Server Components will reach 40% adoption by Q4 2026" | 1-6 months |
| PRODUCT_LAUNCH | "Apple will announce AR glasses at WWDC 2026" | 1-30 days |
| MARKET_MOVE | "NVDA will close above $150 by June 15" | 1-14 days |
| REGULATORY | "EU AI Act implementing rules will be published by July" | 1-3 months |
| STARTUP_SUCCESS | "Company X will raise Series B above $50M by Q4" | 1-6 months |
| CULTURE | "Movie X will open above $100M domestic weekend" | 1-7 days |
| GITHUB_TREND | "Repo X will gain 5K stars within 30 days" | 7-30 days |

## API Surface

```
POST   /v1/predict              Generate predictions from current signals
POST   /v1/predict/query        Generate predictions for a specific question
GET    /v1/predictions           List predictions (filterable by category, status)
GET    /v1/predictions/{id}      Get specific prediction
POST   /v1/predictions/{id}/resolve   Mark as CORRECT/INCORRECT
GET    /v1/calibration           Calibration dashboard data
GET    /v1/signals               Recent signals ingested
GET    /v1/dashboard             HTML calibration dashboard
POST   /v1/ingest                Trigger a fresh ingestion cycle
```

## Ingestion Sources (v0)

| Source | Data | Method |
|--------|------|--------|
| Hacker News | Tech trends, launches | API (free) |
| Reddit (r/technology, r/programming) | Tech sentiment | RSS/API |
| Yahoo Finance | Stock prices, news | yfinance |
| GitHub Trending | Repo growth | API (free) |
| RSS Feeds (TechCrunch, Verge, Ars) | Product launches | feedparser |

## CLI

```bash
# Generate predictions
oracle predict "Will Apple release a new MacBook Pro in June 2026?"

# Generate predictions from current signals
oracle scan

# List predictions
oracle predictions --category tech_trend

# Resolve a prediction
oracle resolve <prediction-id> --outcome correct

# Show calibration
oracle calibration

# Open dashboard
oracle dashboard

# Start server
oracle server
```

## Calibration Dashboard

A single-page HTML dashboard showing:
- Overall accuracy: "73% of 847 predictions correct"
- Reliability diagram: confidence vs actual accuracy (calibration curve)
- Breakdown by category
- Recent predictions feed (resolved + pending)
- Prediction count over time
- Matrix/cyber-themed (green-on-black, glitch effects)

## v0 Scope

- [ ] 5 prediction categories
- [ ] 5 ingestion sources
- [ ] LLM-powered prediction engine
- [ ] Calibration tracking
- [ ] REST API
- [ ] CLI with Rich output
- [ ] Python SDK
- [ ] HTML calibration dashboard (Matrix-themed)
- [ ] Docker + Azure ready

## Azure Deployment

Same pattern as Genesis Engine and AgentSystem:
- Azure Container Apps (eastus2)
- Azure Key Vault for API keys
- SQLite (dev) → Azure PostgreSQL (prod)
- Managed Identity for secrets
