"""
Streaming service.

This module owns frame streaming and detection fanout logic. API route
handlers call these helpers instead of implementing stream loops directly.
"""

import asyncio
import json
from queue import Empty

import cv2

from app.core.state import first_appearance_queue
from app.core.state import result_queue
from app.services import db_writer_service
from app.services import metrics_service
from app.services import pipeline_service


queue_drain_task = None
latest_detection_by_camera = {}
detection_events_by_camera = {}
first_appearance_events_by_camera = {}
first_appearance_signal_by_camera = {}


def get_camera_event(event_map, camera_id):
    """
    Return the asyncio.Event used to wake streams for one camera.
    """

    if camera_id not in event_map:
        event_map[camera_id] = asyncio.Event()

    return event_map[camera_id]


async def drain_detection_queue():
    """
    Move worker detection results into shared FastAPI fanout state.
    """

    global latest_detection_by_camera
    global first_appearance_events_by_camera

    while True:
        try:
            while True:
                detection_payload = result_queue.get_nowait()
                camera_id = detection_payload.get(
                    "camera_id",
                    pipeline_service.DEFAULT_CAMERA_ID,
                )
                latest_detection_by_camera[camera_id] = detection_payload
                metrics_service.record_detection_payload(detection_payload)
                get_camera_event(detection_events_by_camera, camera_id).set()
                await db_writer_service.enqueue_detection_payload(
                    detection_payload
                )
        except Empty:
            pass

        try:
            while True:
                first_appearance_event = first_appearance_queue.get_nowait()
                camera_id = first_appearance_event.get(
                    "camera_id",
                    pipeline_service.DEFAULT_CAMERA_ID,
                )
                metrics_service.record_first_appearance_event()
                first_appearance_events_by_camera.setdefault(
                    camera_id,
                    [],
                ).append(first_appearance_event)
                get_camera_event(
                    first_appearance_signal_by_camera,
                    camera_id,
                ).set()
                await db_writer_service.enqueue_first_appearance_event(
                    first_appearance_event
                )
        except Empty:
            pass

        await asyncio.sleep(0.01)


async def start_detection_fanout():
    """
    Start the background queue-drain task.
    """

    global queue_drain_task

    queue_drain_task = asyncio.create_task(drain_detection_queue())


async def stop_detection_fanout():
    """
    Stop the background queue-drain task.
    """

    global queue_drain_task
    global latest_detection_by_camera
    global detection_events_by_camera
    global first_appearance_events_by_camera
    global first_appearance_signal_by_camera

    if queue_drain_task is not None:
        queue_drain_task.cancel()

        try:
            await queue_drain_task
        except asyncio.CancelledError:
            pass

        queue_drain_task = None

    latest_detection_by_camera = {}
    detection_events_by_camera = {}
    first_appearance_events_by_camera = {}
    first_appearance_signal_by_camera = {}


def clear_latest_detection(camera_id=None):
    """
    Clear the latest detection payload.
    """

    global latest_detection_by_camera

    if camera_id is None:
        latest_detection_by_camera = {}
        detection_events_by_camera.clear()
        return

    latest_detection_by_camera.pop(camera_id, None)
    detection_events_by_camera.pop(camera_id, None)


def clear_latest_first_appearance(camera_id=None):
    """
    Clear the latest first-appearance crop event.
    """

    global first_appearance_events_by_camera

    if camera_id is None:
        first_appearance_events_by_camera = {}
        first_appearance_signal_by_camera.clear()
        return

    first_appearance_events_by_camera.pop(camera_id, None)
    first_appearance_signal_by_camera.pop(camera_id, None)


def encode_mjpeg_frame(frame):
    """
    Encode one OpenCV frame as an MJPEG multipart payload.
    """

    success, encoded_frame = cv2.imencode(".jpg", frame)

    if not success:
        return None

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + encoded_frame.tobytes()
        + b"\r\n"
    )


async def annotated_frame_generator(camera_id=pipeline_service.DEFAULT_CAMERA_ID):
    """
    Stream newest annotated frames without consuming them.
    """

    camera_id = pipeline_service.sanitize_camera_id(camera_id)
    last_frame_id = -1
    metrics_service.increment_client("mjpeg")

    try:
        while True:
            annotated_ring_buffer = pipeline_service.get_annotated_ring_buffer(
                camera_id
            )

            if annotated_ring_buffer is None:
                await asyncio.sleep(0.05)
                continue

            frame_id, frame = annotated_ring_buffer.read_latest(last_frame_id)

            if frame is None:
                await asyncio.sleep(0.03)
                continue

            payload = encode_mjpeg_frame(frame)

            if payload is not None:
                last_frame_id = frame_id
                yield payload

            await asyncio.sleep(0.001)
    finally:
        metrics_service.decrement_client("mjpeg")


async def detection_event_generator(camera_id=pipeline_service.DEFAULT_CAMERA_ID):
    """
    Stream detection payloads as Server-Sent Events.
    """

    camera_id = pipeline_service.sanitize_camera_id(camera_id)
    last_frame_id = -1
    detection_event = get_camera_event(detection_events_by_camera, camera_id)
    metrics_service.increment_client("detections")

    try:
        while True:
            latest_detection = latest_detection_by_camera.get(camera_id)

            if latest_detection is None:
                await detection_event.wait()
                detection_event.clear()
                continue

            frame_id = latest_detection.get("frame", -1)

            if frame_id <= last_frame_id:
                await detection_event.wait()
                detection_event.clear()
                continue

            last_frame_id = frame_id
            payload = json.dumps(latest_detection)

            yield f"data: {payload}\n\n"
    finally:
        metrics_service.decrement_client("detections")


async def first_appearance_event_generator(
    camera_id=pipeline_service.DEFAULT_CAMERA_ID,
):
    """
    Stream first-appearance crop events as Server-Sent Events.
    """

    camera_id = pipeline_service.sanitize_camera_id(camera_id)
    next_event_index = 0
    first_appearance_signal = get_camera_event(
        first_appearance_signal_by_camera,
        camera_id,
    )
    metrics_service.increment_client("first_appearances")

    try:
        while True:
            first_appearance_events = first_appearance_events_by_camera.get(
                camera_id,
                [],
            )

            if next_event_index >= len(first_appearance_events):
                await first_appearance_signal.wait()
                first_appearance_signal.clear()
                continue

            event = first_appearance_events[next_event_index]
            next_event_index += 1
            payload = json.dumps(event)

            yield f"data: {payload}\n\n"
    finally:
        metrics_service.decrement_client("first_appearances")
