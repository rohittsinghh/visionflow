"""
Pipeline API routes.
"""

from fastapi import APIRouter
from fastapi import Depends

from app.core.auth import require_api_key
from app.services import history_service
from app.services import pipeline_service
from app.services import streaming_service


router = APIRouter()


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
            response["run_id"],
            pipeline_service.VIDEO_PATH,
            pipeline_service.MODEL_PATH,
        )

    return response


@router.post("/stop-video", dependencies=[Depends(require_api_key)])
async def stop_video():
    """
    Stop the video processing pipeline and release shared memory.
    """

    run_id = pipeline_service.current_run_id
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
