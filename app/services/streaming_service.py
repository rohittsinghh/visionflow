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
latest_detection = None
latest_detection_event = asyncio.Event()
latest_first_appearance = None
first_appearance_events = []
latest_first_appearance_event = asyncio.Event()


async def drain_detection_queue():
    """
    Move worker detection results into shared FastAPI fanout state.
    """

    global latest_detection
    global latest_first_appearance
    global first_appearance_events

    while True:
        try:
            while True:
                latest_detection = result_queue.get_nowait()
                metrics_service.record_detection_payload(latest_detection)
                latest_detection_event.set()
                await db_writer_service.enqueue_detection_payload(
                    latest_detection
                )
        except Empty:
            pass

        try:
            while True:
                latest_first_appearance = first_appearance_queue.get_nowait()
                metrics_service.record_first_appearance_event()
                first_appearance_events.append(latest_first_appearance)
                latest_first_appearance_event.set()
                await db_writer_service.enqueue_first_appearance_event(
                    latest_first_appearance
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
    global latest_detection
    global latest_first_appearance
    global first_appearance_events

    if queue_drain_task is not None:
        queue_drain_task.cancel()

        try:
            await queue_drain_task
        except asyncio.CancelledError:
            pass

        queue_drain_task = None

    latest_detection = None
    latest_first_appearance = None
    first_appearance_events = []


def clear_latest_detection():
    """
    Clear the latest detection payload.
    """

    global latest_detection

    latest_detection = None


def clear_latest_first_appearance():
    """
    Clear the latest first-appearance crop event.
    """

    global latest_first_appearance
    global first_appearance_events

    latest_first_appearance = None
    first_appearance_events = []


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


async def annotated_frame_generator():
    """
    Stream newest annotated frames without consuming them.
    """

    last_frame_id = -1
    metrics_service.increment_client("mjpeg")

    try:
        while True:
            annotated_ring_buffer = pipeline_service.get_annotated_ring_buffer()

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


async def detection_event_generator():
    """
    Stream detection payloads as Server-Sent Events.
    """

    last_frame_id = -1
    metrics_service.increment_client("detections")

    try:
        while True:
            if latest_detection is None:
                await latest_detection_event.wait()
                latest_detection_event.clear()
                continue

            frame_id = latest_detection.get("frame", -1)

            if frame_id <= last_frame_id:
                await latest_detection_event.wait()
                latest_detection_event.clear()
                continue

            last_frame_id = frame_id
            payload = json.dumps(latest_detection)

            yield f"data: {payload}\n\n"
    finally:
        metrics_service.decrement_client("detections")


async def first_appearance_event_generator():
    """
    Stream first-appearance crop events as Server-Sent Events.
    """

    next_event_index = 0
    metrics_service.increment_client("first_appearances")

    try:
        while True:
            if next_event_index >= len(first_appearance_events):
                await latest_first_appearance_event.wait()
                latest_first_appearance_event.clear()
                continue

            event = first_appearance_events[next_event_index]
            next_event_index += 1
            payload = json.dumps(event)

            yield f"data: {payload}\n\n"
    finally:
        metrics_service.decrement_client("first_appearances")
