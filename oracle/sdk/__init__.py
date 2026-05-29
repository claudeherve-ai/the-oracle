"""The Oracle Python SDK."""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import httpx

DEFAULT_API_URL = "http://localhost:8001/v1"


@dataclass
class OracleClient:
    base_url: str = DEFAULT_API_URL
    timeout: float = 120.0

    def __post_init__(self):
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def predict(self, question: str, count: int = 5) -> List[Dict[str, Any]]:
        r = self.client.post("/predict/query", json={"question": question, "max_predictions": count})
        r.raise_for_status()
        return r.json().get("predictions", [])

    def scan(self, count: int = 5) -> List[Dict[str, Any]]:
        self.client.post("/ingest")
        r = self.client.post("/predict", json={"question": "", "max_predictions": count})
        r.raise_for_status()
        return r.json().get("predictions", [])

    def calibration(self, category: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if category:
            params["category"] = category
        r = self.client.get("/calibration", params=params)
        r.raise_for_status()
        return r.json()

    def list_predictions(self, category: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {"limit": 50}
        if category:
            params["category"] = category
        if status:
            params["status"] = status
        r = self.client.get("/predictions", params=params)
        r.raise_for_status()
        return r.json().get("items", [])

    def resolve(self, prediction_id: str, outcome: str, note: Optional[str] = None) -> Dict[str, Any]:
        r = self.client.post(f"/predictions/{prediction_id}/resolve", json={"outcome": outcome, "resolution": note})
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


_default: Optional[OracleClient] = None


def _get() -> OracleClient:
    global _default
    if _default is None:
        _default = OracleClient()
    return _default


def predict(question: str, count: int = 5) -> List[Dict[str, Any]]:
    return _get().predict(question, count)


def scan(count: int = 5) -> List[Dict[str, Any]]:
    return _get().scan(count)


def calibration(category: Optional[str] = None) -> Dict[str, Any]:
    return _get().calibration(category)
