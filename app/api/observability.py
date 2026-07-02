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

    video_alive = (
        pipeline_service.video_process is not None
        and pipeline_service.video_process.is_alive()
    )
    yolo_alive = (
        pipeline_service.yolo_process is not None
        and pipeline_service.yolo_process.is_alive()
    )

    return {
        "status": "ok",
        "database_configured": connection.is_configured(),
        "run_id": pipeline_service.current_run_id,
        "workers": {
            "video_reader_alive": video_alive,
            "yolo_worker_alive": yolo_alive,
        },
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
