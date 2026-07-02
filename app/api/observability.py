"""
Health and metrics API routes.
"""

from fastapi import APIRouter

from app.db import connection
from app.services import metrics_service
from app.services import pipeline_service


router = APIRouter()


@router.get("/health")
def health():
    """
    Return service health and worker state.
    """

    cameras = pipeline_service.list_cameras()["items"]

    return {
        "status": "ok",
        "database_configured": connection.is_configured(),
        "active_camera_count": len(cameras),
        "cameras": cameras,
    }


@router.get("/metrics")
def metrics():
    """
    Return runtime metrics.
    """

    return {
        "pipeline": pipeline_service.get_pipeline_metrics(),
        "runtime": metrics_service.snapshot(),
    }
