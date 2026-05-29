"""The Oracle — static dashboard server."""

from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()


@router.get("/dashboard")
async def dashboard():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(path, media_type="text/html")
