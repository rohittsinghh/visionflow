"""
History API routes backed by PostgreSQL.
"""

from fastapi import APIRouter
from fastapi import Depends
from fastapi import Query

from app.core.auth import require_api_key
from app.services import history_service


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/detections/latest")
async def latest_detections(
    camera_id: str | None = None,
    limit: int = Query(20, ge=1, le=500),
):
    """
    Return newest detection rows.
    """

    return {
        "items": await history_service.latest_detections(
            limit=limit,
            camera_id=camera_id,
        ),
    }


@router.get("/detections/history")
async def detection_history(
    class_name: str | None = None,
    run_id: str | None = None,
    camera_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Return detection history with optional class/run filters.
    """

    return {
        "items": await history_service.detection_history(
            class_name=class_name,
            run_id=run_id,
            camera_id=camera_id,
            limit=limit,
        ),
    }


@router.get("/first-appearances/history")
async def first_appearance_history(
    run_id: str | None = None,
    camera_id: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Return first-appearance crop metadata history.
    """

    return {
        "items": await history_service.first_appearance_history(
            run_id=run_id,
            camera_id=camera_id,
            limit=limit,
        ),
    }


@router.get("/runs/{run_id}/summary")
async def run_summary(run_id: str):
    """
    Return one pipeline run summary.
    """

    return {
        "item": await history_service.run_summary(run_id),
    }
