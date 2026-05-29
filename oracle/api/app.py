"""FastAPI application for The Oracle."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from oracle.api.routes import router

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("oracle")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("The Oracle starting...")
    logger.info("  LLM: %s", "Azure Foundry (gpt-5.4)" if os.getenv("AZURE_OPENAI_API_KEY") else "not configured")
    logger.info("  DB: %s", os.getenv("ORACLE_DB", "sqlite:///oracle.db"))
    yield
    logger.info("The Oracle shutting down.")


app = FastAPI(
    title="The Oracle",
    description="Predictive intelligence engine — specific, verifiable, time-bound forecasts with calibrated confidence. Multi-source MCP grounding enabled.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dashboard static files
dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(dashboard_path):
    app.mount("/dashboard", StaticFiles(directory=dashboard_path, html=True), name="dashboard")

app.include_router(router)


@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "ui", "index.html")
    if os.path.isfile(ui_path):
        return FileResponse(ui_path, media_type="text/html")
    return {"name": "The Oracle", "version": "0.1.0", "status": "running", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def main():
    import uvicorn
    host = os.getenv("ORACLE_HOST", "0.0.0.0")
    port = int(os.getenv("ORACLE_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
