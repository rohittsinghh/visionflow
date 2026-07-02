"""
Pipeline API routes.
"""

from fastapi import APIRouter
from fastapi import Depends
from pydantic import BaseModel

from app.core.auth import require_api_key
from app.services import history_service
from app.services import pipeline_service
from app.services import streaming_service


router = APIRouter()


class CameraStartRequest(BaseModel):
    """
    Request body for starting a camera/video source.
    """

    source: str = pipeline_service.VIDEO_PATH


@router.get("/")
def home():
    """
    Health check endpoint.
    """

    return {
        "status": "running",
    }


@router.post("/start-video", dependencies=[Depends(require_api_key)])
async def start_video():
    """
    Start the complete video processing pipeline.
    """

    response = pipeline_service.start_pipeline()

    if "run_id" in response:
        await history_service.create_run(
            response["camera_id"],
            response["run_id"],
            response["source"],
            pipeline_service.MODEL_PATH,
        )

    return response


@router.post("/stop-video", dependencies=[Depends(require_api_key)])
async def stop_video():
    """
    Stop the video processing pipeline and release shared memory.
    """

    run_id = pipeline_service.get_current_run_id()
    response = pipeline_service.stop_pipeline()
    streaming_service.clear_latest_detection()
    streaming_service.clear_latest_first_appearance()

    await history_service.stop_run(run_id)

    return response


@router.get("/buffer-status", dependencies=[Depends(require_api_key)])
def buffer_status():
    """
    Debug endpoint for inspecting ring buffer metadata.
    """

    return pipeline_service.get_buffer_status()


@router.get("/cameras", dependencies=[Depends(require_api_key)])
def cameras():
    """
    Return active camera pipelines.
    """

    return pipeline_service.list_cameras()


@router.post(
    "/cameras/{camera_id}/start",
    dependencies=[Depends(require_api_key)],
)
async def start_camera(camera_id: str, request: CameraStartRequest):
    """
    Start one camera/video source.
    """

    response = pipeline_service.start_camera_pipeline(
        camera_id=camera_id,
        source=request.source,
    )

    if "run_id" in response:
        await history_service.create_run(
            response["camera_id"],
            response["run_id"],
            response["source"],
            pipeline_service.MODEL_PATH,
        )

    return response


@router.post(
    "/cameras/{camera_id}/stop",
    dependencies=[Depends(require_api_key)],
)
async def stop_camera(camera_id: str):
    """
    Stop one camera/video source.
    """

    normalized_camera_id = pipeline_service.sanitize_camera_id(camera_id)
    run_id = pipeline_service.get_current_run_id(normalized_camera_id)
    response = pipeline_service.stop_camera_pipeline(normalized_camera_id)
    streaming_service.clear_latest_detection(normalized_camera_id)
    streaming_service.clear_latest_first_appearance(normalized_camera_id)

    await history_service.stop_run(run_id)

    return response


@router.get(
    "/cameras/{camera_id}/buffer-status",
    dependencies=[Depends(require_api_key)],
)
def camera_buffer_status(camera_id: str):
    """
    Debug endpoint for inspecting one camera's ring buffers.
    """

    return pipeline_service.get_camera_buffer_status(camera_id)
