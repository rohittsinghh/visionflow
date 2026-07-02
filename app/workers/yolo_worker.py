"""
YOLO Worker

Responsibilities:
1. Connect to the shared ring buffer.
2. Read frames in production order.
3. Run YOLO object detection.
4. Push detection results into the queue.

This worker is the consumer.
"""

import logging
import re

import cv2

from app.core.config import LOG_EVERY_N_FRAMES
from app.core.logging_config import configure_logging
from app.core.ring_buffer import RingBuffer
from app.ml.yolo_onnx import YOLOONNXDetector
from app.services import storage_service


logger = logging.getLogger(__name__)


def safe_filename_part(value):
    """
    Convert a class name into a filesystem-safe filename segment.
    """

    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip().lower())
    return cleaned.strip("_") or "unknown"


def crop_detection(frame, bbox):
    """
    Return a copy of the detected object crop, clipped to the frame bounds.
    """

    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(x1), frame_width - 1))
    y1 = max(0, min(int(y1), frame_height - 1))
    x2 = max(0, min(int(x2), frame_width))
    y2 = max(0, min(int(y2), frame_height))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def create_first_appearance_event(camera_id, run_id, frame_id, frame, detection):
    """
    Save the first detected crop for a class and build an event payload.
    """

    crop = crop_detection(frame, detection["bbox"])

    if crop is None:
        return None

    class_name = detection["class_name"]
    safe_class_name = safe_filename_part(class_name)
    storage_service.ensure_crop_dir(camera_id, run_id)

    crop_filename = f"{frame_id}_{safe_class_name}.jpg"
    crop_path = storage_service.crop_path(camera_id, run_id, crop_filename)

    success = cv2.imwrite(str(crop_path), crop)

    if not success:
        return None

    return {
        "run_id": run_id,
        "camera_id": camera_id,
        "frame": frame_id,
        "class_id": detection["class_id"],
        "class_name": class_name,
        "confidence": detection["confidence"],
        "bbox": detection["bbox"],
        "crop_path": str(crop_path),
        "crop_url": storage_service.crop_url(camera_id, run_id, crop_filename),
    }


def draw_detections(frame, detections):
    """
    Draw YOLO detections onto a frame copy.

    OpenCV uses BGR frames, which matches the frames coming from the video
    reader. The function keeps the original frame untouched so detection and
    annotation remain separate steps.
    """

    annotated_frame = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        text_width, text_height = text_size
        label_y1 = max(y1 - text_height - baseline - 4, 0)

        cv2.rectangle(
            annotated_frame,
            (x1, label_y1),
            (x1 + text_width + 4, label_y1 + text_height + baseline + 4),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1 + 2, label_y1 + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated_frame


def yolo_worker(
    frame_shape: tuple[int, int, int],
    buffer_size: int,
    frame_shm_name: str,
    id_shm_name: str,
    status_shm_name: str,
    write_index,
    read_index,
    next_frame_id,
    lock,
    annotated_frame_shm_name: str,
    annotated_id_shm_name: str,
    annotated_status_shm_name: str,
    annotated_write_index,
    annotated_read_index,
    annotated_next_frame_id,
    annotated_lock,
    model_path: str,
    result_queue,
    first_appearance_queue,
    run_id: str,
    camera_id: str = "default",
):
    """
    Read frames from the ring buffer, run YOLO inference, and send detection
    results to FastAPI through a multiprocessing queue.

    The lock is only held while copying a frame out of shared memory and while
    changing slot status. It is not held during YOLO inference.
    """

    configure_logging()
    ring_buffer = RingBuffer(
        frame_shape=frame_shape,
        buffer_size=buffer_size,
        frame_shm_name=frame_shm_name,
        id_shm_name=id_shm_name,
        status_shm_name=status_shm_name,
        write_index=write_index,
        read_index=read_index,
        next_frame_id=next_frame_id,
        lock=lock,
        create=False,
    )

    annotated_ring_buffer = RingBuffer(
        frame_shape=frame_shape,
        buffer_size=buffer_size,
        frame_shm_name=annotated_frame_shm_name,
        id_shm_name=annotated_id_shm_name,
        status_shm_name=annotated_status_shm_name,
        write_index=annotated_write_index,
        read_index=annotated_read_index,
        next_frame_id=annotated_next_frame_id,
        lock=annotated_lock,
        create=False,
    )

    detector = YOLOONNXDetector(
        model_path=model_path,
    )
    seen_classes = set()

    logger.info(
        "yolo_worker_started camera_id=%s run_id=%s model_path=%s",
        camera_id,
        run_id,
        model_path,
    )

    try:
        while True:
            frame_id, frame, slot_index = ring_buffer.read()

            try:
                detections = detector.detect(frame)
                annotated_frame = draw_detections(frame, detections)
                annotated_ring_buffer.write_latest(
                    annotated_frame,
                    frame_id=frame_id,
                )

                result_queue.put(
                    {
                        "camera_id": camera_id,
                        "run_id": run_id,
                        "frame": frame_id,
                        "detections": detections,
                    }
                )

                if frame_id % LOG_EVERY_N_FRAMES == 0:
                    logger.info(
                        "frame_processed camera_id=%s run_id=%s frame_id=%s detections=%s",
                        camera_id,
                        run_id,
                        frame_id,
                        len(detections),
                    )

                for detection in detections:
                    class_name = detection["class_name"]

                    if class_name not in seen_classes:
                        crop_event = create_first_appearance_event(
                            camera_id,
                            run_id,
                            frame_id,
                            frame,
                            detection,
                        )

                        if crop_event is not None:
                            seen_classes.add(class_name)
                            first_appearance_queue.put(crop_event)
                            logger.info(
                                "first_appearance run_id=%s frame_id=%s "
                                "camera_id=%s class_name=%s confidence=%.2f crop_url=%s",
                                run_id,
                                frame_id,
                                camera_id,
                                class_name,
                                detection["confidence"],
                                crop_event["crop_url"],
                            )

            finally:
                ring_buffer.mark_empty(slot_index)

    finally:
        ring_buffer.close()
        annotated_ring_buffer.close()
        logger.info("yolo_worker_finished camera_id=%s run_id=%s", camera_id, run_id)
