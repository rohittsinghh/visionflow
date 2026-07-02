"""
Pipeline service.

This module owns runtime pipeline objects. In v1 multi-camera mode, each
camera id maps to an isolated pair of worker processes and ring buffers. Local
video files are treated as camera sources so the same API shape can later
accept RTSP URLs or webcam indexes.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process

from app.core.ring_buffer import RingBuffer
from app.core.ring_buffer import cleanup_shared_memory_names
from app.core.state import first_appearance_queue
from app.core.state import result_queue
from app.workers.video_reader import read_video
from app.workers.yolo_worker import yolo_worker


DEFAULT_CAMERA_ID = "default"
VIDEO_PATH = "videos/sample.mp4"
MODEL_PATH = "models/yolov8n.onnx"
FRAME_SHAPE = (480, 640, 3)
BUFFER_SIZE = 4

logger = logging.getLogger(__name__)
camera_pipelines = {}


@dataclass
class CameraPipelineState:
    camera_id: str
    source: str
    run_id: str
    shared_memory_names: dict
    raw_ring_buffer: RingBuffer
    annotated_ring_buffer: RingBuffer
    video_process: Process | None = None
    yolo_process: Process | None = None


def sanitize_camera_id(camera_id):
    """
    Keep camera ids URL-friendly and usable in short shared memory names.
    """

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", camera_id.strip().lower())
    return cleaned.strip("_") or DEFAULT_CAMERA_ID


def shared_memory_tag(camera_id):
    """
    Return a short stable tag for POSIX shared memory names.
    """

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "", camera_id.lower())
    return (cleaned or "default")[:8]


def build_shared_memory_names(camera_id):
    """
    Build short fixed shared memory names for one camera.
    """

    if sanitize_camera_id(camera_id) == DEFAULT_CAMERA_ID:
        return {
            "raw": {
                "frame": "psm_yrf",
                "ids": "psm_yri",
                "status": "psm_yrs",
            },
            "annotated": {
                "frame": "psm_yaf",
                "ids": "psm_yai",
                "status": "psm_yas",
            },
        }

    tag = shared_memory_tag(camera_id)

    return {
        "raw": {
            "frame": f"psm_{tag}_rf",
            "ids": f"psm_{tag}_ri",
            "status": f"psm_{tag}_rs",
        },
        "annotated": {
            "frame": f"psm_{tag}_af",
            "ids": f"psm_{tag}_ai",
            "status": f"psm_{tag}_as",
        },
    }


def flatten_shared_memory_names(name_map):
    """
    Return all shared memory names from a camera name map.
    """

    return [
        name_map["raw"]["frame"],
        name_map["raw"]["ids"],
        name_map["raw"]["status"],
        name_map["annotated"]["frame"],
        name_map["annotated"]["ids"],
        name_map["annotated"]["status"],
    ]


def cleanup_configured_shared_memory(camera_id):
    """
    Unlink this camera's fixed shared memory blocks when they already exist.
    """

    return cleanup_shared_memory_names(
        flatten_shared_memory_names(build_shared_memory_names(camera_id))
    )


def is_process_alive(process):
    """
    Return whether a process exists and is alive.
    """

    return process is not None and process.is_alive()


def cleanup_camera_pipeline(camera_id):
    """
    Stop one camera pipeline and release its shared memory.
    """

    camera_id = sanitize_camera_id(camera_id)
    state = camera_pipelines.get(camera_id)

    if state is None:
        cleanup_configured_shared_memory(camera_id)
        return None

    for process in (state.video_process, state.yolo_process):
        if is_process_alive(process):
            process.terminate()
            process.join(timeout=2)

    for buffer_name, ring_buffer in (
        ("raw", state.raw_ring_buffer),
        ("annotated", state.annotated_ring_buffer),
    ):
        try:
            ring_buffer.close()
            ring_buffer.unlink()
        except Exception:
            logger.exception(
                "ring_buffer_cleanup_failed camera_id=%s buffer=%s",
                camera_id,
                buffer_name,
            )

    cleanup_shared_memory_names(
        flatten_shared_memory_names(state.shared_memory_names)
    )
    camera_pipelines.pop(camera_id, None)
    return state


def cleanup_pipeline():
    """
    Stop all camera pipelines. Kept for FastAPI shutdown compatibility.
    """

    for camera_id in list(camera_pipelines):
        cleanup_camera_pipeline(camera_id)


def start_camera_pipeline(camera_id=DEFAULT_CAMERA_ID, source=VIDEO_PATH):
    """
    Start one camera pipeline.
    """

    camera_id = sanitize_camera_id(camera_id)

    existing_state = camera_pipelines.get(camera_id)

    if existing_state is not None and is_process_alive(existing_state.video_process):
        return {
            "message": "Camera pipeline already running.",
            "camera_id": camera_id,
            "run_id": existing_state.run_id,
        }

    cleanup_camera_pipeline(camera_id)
    name_map = build_shared_memory_names(camera_id)
    cleanup_configured_shared_memory(camera_id)
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S_%f")

    raw_ring_buffer = RingBuffer(
        frame_shape=FRAME_SHAPE,
        buffer_size=BUFFER_SIZE,
        frame_shm_name=name_map["raw"]["frame"],
        id_shm_name=name_map["raw"]["ids"],
        status_shm_name=name_map["raw"]["status"],
        create=True,
    )

    annotated_ring_buffer = RingBuffer(
        frame_shape=FRAME_SHAPE,
        buffer_size=BUFFER_SIZE,
        frame_shm_name=name_map["annotated"]["frame"],
        id_shm_name=name_map["annotated"]["ids"],
        status_shm_name=name_map["annotated"]["status"],
        create=True,
    )

    video_process = Process(
        target=read_video,
        args=(
            source,
            FRAME_SHAPE,
            BUFFER_SIZE,
            raw_ring_buffer.frame_shm_name,
            raw_ring_buffer.id_shm_name,
            raw_ring_buffer.status_shm_name,
            raw_ring_buffer.write_index,
            raw_ring_buffer.read_index,
            raw_ring_buffer.next_frame_id,
            raw_ring_buffer.lock,
        ),
    )

    yolo_process = Process(
        target=yolo_worker,
        args=(
            FRAME_SHAPE,
            BUFFER_SIZE,
            raw_ring_buffer.frame_shm_name,
            raw_ring_buffer.id_shm_name,
            raw_ring_buffer.status_shm_name,
            raw_ring_buffer.write_index,
            raw_ring_buffer.read_index,
            raw_ring_buffer.next_frame_id,
            raw_ring_buffer.lock,
            annotated_ring_buffer.frame_shm_name,
            annotated_ring_buffer.id_shm_name,
            annotated_ring_buffer.status_shm_name,
            annotated_ring_buffer.write_index,
            annotated_ring_buffer.read_index,
            annotated_ring_buffer.next_frame_id,
            annotated_ring_buffer.lock,
            MODEL_PATH,
            result_queue,
            first_appearance_queue,
            run_id,
            camera_id,
        ),
    )

    state = CameraPipelineState(
        camera_id=camera_id,
        source=source,
        run_id=run_id,
        shared_memory_names=name_map,
        raw_ring_buffer=raw_ring_buffer,
        annotated_ring_buffer=annotated_ring_buffer,
        video_process=video_process,
        yolo_process=yolo_process,
    )
    camera_pipelines[camera_id] = state

    video_process.start()
    yolo_process.start()
    logger.info(
        "camera_pipeline_started camera_id=%s run_id=%s source=%s model_path=%s",
        camera_id,
        run_id,
        source,
        MODEL_PATH,
    )

    return {
        "message": "Video reader and YOLO worker started successfully.",
        "camera_id": camera_id,
        "run_id": run_id,
        "source": source,
    }


def start_pipeline():
    """
    Legacy single-camera start endpoint.
    """

    return start_camera_pipeline(DEFAULT_CAMERA_ID, VIDEO_PATH)


def stop_camera_pipeline(camera_id=DEFAULT_CAMERA_ID):
    """
    Stop one camera pipeline.
    """

    camera_id = sanitize_camera_id(camera_id)
    state = camera_pipelines.get(camera_id)

    if state is None:
        return {
            "message": "Camera pipeline is not running.",
            "camera_id": camera_id,
        }

    run_id = state.run_id
    cleanup_camera_pipeline(camera_id)
    logger.info("camera_pipeline_stopped camera_id=%s run_id=%s", camera_id, run_id)

    return {
        "message": "Video pipeline stopped successfully.",
        "camera_id": camera_id,
        "run_id": run_id,
    }


def stop_pipeline():
    """
    Legacy single-camera stop endpoint.
    """

    return stop_camera_pipeline(DEFAULT_CAMERA_ID)


def camera_snapshot(state):
    """
    Return one camera pipeline snapshot.
    """

    return {
        "camera_id": state.camera_id,
        "source": state.source,
        "run_id": state.run_id,
        "video_reader_alive": is_process_alive(state.video_process),
        "yolo_worker_alive": is_process_alive(state.yolo_process),
    }


def list_cameras():
    """
    Return all known active camera pipelines.
    """

    return {
        "items": [
            camera_snapshot(state)
            for state in camera_pipelines.values()
        ]
    }


def get_camera_buffer_status(camera_id=DEFAULT_CAMERA_ID):
    """
    Return raw and annotated ring buffer metadata for one camera.
    """

    camera_id = sanitize_camera_id(camera_id)
    state = camera_pipelines.get(camera_id)

    if state is None:
        return {
            "message": "Camera pipeline has not been started.",
            "camera_id": camera_id,
        }

    return {
        "camera_id": camera_id,
        "source": state.source,
        "raw": state.raw_ring_buffer.snapshot(),
        "annotated": state.annotated_ring_buffer.snapshot(),
        "run_id": state.run_id,
    }


def get_buffer_status():
    """
    Legacy default-camera buffer status.
    """

    return get_camera_buffer_status(DEFAULT_CAMERA_ID)


def calculate_buffer_fill(snapshot):
    """
    Count slots that currently contain work.
    """

    if not snapshot:
        return 0

    return sum(
        1
        for status in snapshot.get("status", [])
        if status in {"READY", "PROCESSING"}
    )


def get_pipeline_metrics():
    """
    Return worker and ring buffer metrics for all cameras.
    """

    cameras = {}

    for camera_id, state in camera_pipelines.items():
        raw_snapshot = state.raw_ring_buffer.snapshot()
        annotated_snapshot = state.annotated_ring_buffer.snapshot()
        cameras[camera_id] = {
            "run_id": state.run_id,
            "source": state.source,
            "workers": {
                "video_reader_alive": is_process_alive(state.video_process),
                "yolo_worker_alive": is_process_alive(state.yolo_process),
            },
            "buffers": {
                "raw_fill": calculate_buffer_fill(raw_snapshot),
                "annotated_fill": calculate_buffer_fill(annotated_snapshot),
                "buffer_size": BUFFER_SIZE,
            },
        }

    return {
        "camera_count": len(camera_pipelines),
        "cameras": cameras,
    }


def get_annotated_ring_buffer(camera_id=DEFAULT_CAMERA_ID):
    """
    Return one camera's annotated frame ring buffer.
    """

    state = camera_pipelines.get(sanitize_camera_id(camera_id))

    if state is None:
        return None

    return state.annotated_ring_buffer


def get_current_run_id(camera_id=DEFAULT_CAMERA_ID):
    """
    Return one camera's current run id.
    """

    state = camera_pipelines.get(sanitize_camera_id(camera_id))
    return state.run_id if state is not None else None
