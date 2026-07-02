"""
Streaming API routes.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services import streaming_service


router = APIRouter()


@router.get("/frame")
def stream_annotated_frames():
    """
    Stream annotated frames as MJPEG.
    """

    return StreamingResponse(
        streaming_service.annotated_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/cameras/{camera_id}/frame")
def stream_camera_annotated_frames(camera_id: str):
    """
    Stream one camera's annotated frames as MJPEG.
    """

    return StreamingResponse(
        streaming_service.annotated_frame_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/detections/events")
def stream_detection_events():
    """
    Stream detection JSON using Server-Sent Events.
    """

    return StreamingResponse(
        streaming_service.detection_event_generator(),
        media_type="text/event-stream",
    )


@router.get("/cameras/{camera_id}/detections/events")
def stream_camera_detection_events(camera_id: str):
    """
    Stream one camera's detection JSON using Server-Sent Events.
    """

    return StreamingResponse(
        streaming_service.detection_event_generator(camera_id),
        media_type="text/event-stream",
    )


@router.get("/first-appearances/events")
def stream_first_appearance_events():
    """
    Stream first-appearance crop events using Server-Sent Events.
    """

    return StreamingResponse(
        streaming_service.first_appearance_event_generator(),
        media_type="text/event-stream",
    )


@router.get("/cameras/{camera_id}/first-appearances/events")
def stream_camera_first_appearance_events(camera_id: str):
    """
    Stream one camera's first-appearance crop events using SSE.
    """

    return StreamingResponse(
        streaming_service.first_appearance_event_generator(camera_id),
        media_type="text/event-stream",
    )
