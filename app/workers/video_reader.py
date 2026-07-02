"""
Video Reader Worker

Responsibilities:
1. Open a video file.
2. Read frames continuously.
3. Resize frames.
4. Write frames into the shared ring buffer.

This worker is the producer.
It does not run object detection.
"""

import logging
import time

import cv2

from app.core.config import LOG_EVERY_N_FRAMES
from app.core.logging_config import configure_logging
from app.core.ring_buffer import RingBuffer


logger = logging.getLogger(__name__)


def read_video(
    video_path: str,
    frame_shape: tuple[int, int, int],
    buffer_size: int,
    frame_shm_name: str,
    id_shm_name: str,
    status_shm_name: str,
    write_index,
    read_index,
    next_frame_id,
    lock,
):
    """
    Read video frames and write them into the ring buffer.

    The producer writes only into EMPTY slots.
    If the buffer is full, RingBuffer.write() waits briefly and retries.
    """

    configure_logging()
    logger.info("video_reader_started video_path=%s", video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("video_open_failed video_path=%s", video_path)
        return

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

    try:
        while True:
            success, frame = cap.read()

            if not success:
                logger.info("video_finished_restarting video_path=%s", video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(
                frame,
                (frame_shape[1], frame_shape[0]),
            )

            frame_id = ring_buffer.write(frame)

            if frame_id % LOG_EVERY_N_FRAMES == 0:
                logger.info(
                    "frame_written frame_id=%s video_path=%s",
                    frame_id,
                    video_path,
                )

            time.sleep(0.03)

    finally:
        cap.release()
        ring_buffer.close()
        logger.info("video_reader_finished video_path=%s", video_path)
