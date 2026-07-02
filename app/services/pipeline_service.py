"""
Pipeline service.

This module owns the long-lived runtime objects used by the video pipeline:
worker processes and shared-memory ring buffers.
"""

import logging
from datetime import datetime
from multiprocessing import Process

from app.core.ring_buffer import RingBuffer
from app.core.ring_buffer import cleanup_shared_memory_names
from app.core.state import first_appearance_queue
from app.core.state import result_queue
from app.workers.video_reader import read_video
from app.workers.yolo_worker import yolo_worker


VIDEO_PATH = "videos/sample.mp4"
MODEL_PATH = "models/yolov8n.onnx"

FRAME_SHAPE = (480, 640, 3)
BUFFER_SIZE = 4
logger = logging.getLogger(__name__)


video_process = None
yolo_process = None
ring_buffer = None
annotated_ring_buffer = None
current_run_id = None


def build_shared_memory_names():
    """
    Build fixed shared memory names for the local pipeline.

    The names are intentionally short because some platforms, including
    macOS, enforce a small POSIX shared memory name limit. Reusing fixed
    names is convenient for local debugging because the names are stable
    across server restarts.
    """

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


SHARED_MEMORY_NAMES = build_shared_memory_names()


def flatten_shared_memory_names(name_map=SHARED_MEMORY_NAMES):
    """
    Return all configured shared memory names.
    """

    return [
        name_map["raw"]["frame"],
        name_map["raw"]["ids"],
        name_map["raw"]["status"],
        name_map["annotated"]["frame"],
        name_map["annotated"]["ids"],
        name_map["annotated"]["status"],
    ]


def cleanup_configured_shared_memory():
    """
    Unlink this app's fixed shared memory blocks when they already exist.

    This is safe before creating a new local pipeline because start_pipeline()
    first verifies that the current process is not already running a pipeline.
    It should not be used as a broad system cleanup command for unrelated
    processes.
    """

    return cleanup_shared_memory_names(flatten_shared_memory_names())


def cleanup_pipeline():
    """
    Stop worker processes and release shared memory.
    """

    global video_process
    global yolo_process
    global ring_buffer
    global annotated_ring_buffer
    global current_run_id

    for process in (video_process, yolo_process):
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=2)

    video_process = None
    yolo_process = None

    if ring_buffer is not None:
        try:
            ring_buffer.close()
            ring_buffer.unlink()
        except Exception:
            logger.exception("ring_buffer_cleanup_failed")

        ring_buffer = None

    if annotated_ring_buffer is not None:
        try:
            annotated_ring_buffer.close()
            annotated_ring_buffer.unlink()
        except Exception:
            logger.exception("annotated_ring_buffer_cleanup_failed")

        annotated_ring_buffer = None

    cleanup_configured_shared_memory()
    current_run_id = None


def start_pipeline():
    """
    Start the video reader and YOLO worker processes.
    """

    global video_process
    global yolo_process
    global ring_buffer
    global annotated_ring_buffer
    global current_run_id

    if video_process is not None and video_process.is_alive():
        return {
            "message": "Video pipeline already running.",
        }

    cleanup_pipeline()
    name_map = SHARED_MEMORY_NAMES
    current_run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S_%f")

    ring_buffer = RingBuffer(
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
            VIDEO_PATH,
            FRAME_SHAPE,
            BUFFER_SIZE,
            ring_buffer.frame_shm_name,
            ring_buffer.id_shm_name,
            ring_buffer.status_shm_name,
            ring_buffer.write_index,
            ring_buffer.read_index,
            ring_buffer.next_frame_id,
            ring_buffer.lock,
        ),
    )

    yolo_process = Process(
        target=yolo_worker,
        args=(
            FRAME_SHAPE,
            BUFFER_SIZE,
            ring_buffer.frame_shm_name,
            ring_buffer.id_shm_name,
            ring_buffer.status_shm_name,
            ring_buffer.write_index,
            ring_buffer.read_index,
            ring_buffer.next_frame_id,
            ring_buffer.lock,
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
            current_run_id,
        ),
    )

    video_process.start()
    yolo_process.start()
    logger.info(
        "pipeline_started run_id=%s video_path=%s model_path=%s",
        current_run_id,
        VIDEO_PATH,
        MODEL_PATH,
    )

    return {
        "message": "Video reader and YOLO worker started successfully.",
        "run_id": current_run_id,
    }


def stop_pipeline():
    """
    Stop the video pipeline if it exists.
    """

    if (
        ring_buffer is None
        and annotated_ring_buffer is None
        and video_process is None
        and yolo_process is None
    ):
        return {
            "message": "Video pipeline is not running.",
        }

    stopped_run_id = current_run_id
    cleanup_pipeline()
    logger.info("pipeline_stopped run_id=%s", stopped_run_id)

    return {
        "message": "Video pipeline stopped successfully.",
    }


def get_buffer_status():
    """
    Return raw and annotated ring buffer metadata.
    """

    if ring_buffer is None:
        return {
            "message": "Video pipeline has not been started.",
        }

    return {
        "raw": ring_buffer.snapshot(),
        "annotated": (
            annotated_ring_buffer.snapshot()
            if annotated_ring_buffer is not None
            else None
        ),
        "run_id": current_run_id,
    }


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
    Return worker and ring buffer metrics.
    """

    raw_snapshot = ring_buffer.snapshot() if ring_buffer is not None else None
    annotated_snapshot = (
        annotated_ring_buffer.snapshot()
        if annotated_ring_buffer is not None
        else None
    )

    return {
        "run_id": current_run_id,
        "workers": {
            "video_reader_alive": (
                video_process is not None and video_process.is_alive()
            ),
            "yolo_worker_alive": (
                yolo_process is not None and yolo_process.is_alive()
            ),
        },
        "buffers": {
            "raw_fill": calculate_buffer_fill(raw_snapshot),
            "annotated_fill": calculate_buffer_fill(annotated_snapshot),
            "buffer_size": BUFFER_SIZE,
        },
    }


def get_annotated_ring_buffer():
    """
    Return the current annotated frame ring buffer.
    """

    return annotated_ring_buffer
