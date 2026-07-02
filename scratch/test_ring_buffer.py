"""
Standalone producer-consumer RingBuffer test.

This test does not use:
- OpenCV
- YOLO
- FastAPI

It verifies:
- Frame ordering
- Frame IDs
- Slot state transitions
- Shared memory correctness
- Producer-consumer synchronization
"""

import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.ring_buffer import RingBuffer


FRAME_SHAPE = (480, 640, 3)
BUFFER_SIZE = 4
TOTAL_FRAMES = 12


def producer(
    frame_shape,
    buffer_size,
    frame_shm_name,
    id_shm_name,
    status_shm_name,
    write_index,
    read_index,
    next_frame_id,
    lock,
):
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
        for frame_number in range(TOTAL_FRAMES):
            pixel_value = frame_number % 256

            frame = np.full(
                frame_shape,
                pixel_value,
                dtype=np.uint8,
            )

            frame_id = ring_buffer.write(frame)

            print(f"Produced frame_id={frame_id}, pixel_value={pixel_value}")

            time.sleep(0.01)

    finally:
        ring_buffer.close()


def consumer(
    frame_shape,
    buffer_size,
    frame_shm_name,
    id_shm_name,
    status_shm_name,
    write_index,
    read_index,
    next_frame_id,
    lock,
    output_queue,
):
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

    consumed = []

    try:
        for _ in range(TOTAL_FRAMES):
            frame_id, frame, slot_index = ring_buffer.read()

            try:
                mean_pixel = int(frame.mean())

                consumed.append(
                    {
                        "frame_id": frame_id,
                        "mean_pixel": mean_pixel,
                        "slot_index": slot_index,
                    }
                )

                print(
                    f"Consumed frame_id={frame_id}, "
                    f"mean_pixel={mean_pixel}, "
                    f"slot_index={slot_index}"
                )

                time.sleep(0.03)

            finally:
                ring_buffer.mark_empty(slot_index)

        output_queue.put(consumed)

    finally:
        ring_buffer.close()


def main():
    ring_buffer = RingBuffer(
        frame_shape=FRAME_SHAPE,
        buffer_size=BUFFER_SIZE,
        create=True,
    )

    output_queue = Queue()

    producer_process = Process(
        target=producer,
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
        ),
    )

    consumer_process = Process(
        target=consumer,
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
            output_queue,
        ),
    )

    try:
        producer_process.start()
        consumer_process.start()

        producer_process.join()
        consumer_process.join()

        consumed = output_queue.get()

        frame_ids = [
            item["frame_id"]
            for item in consumed
        ]

        expected_frame_ids = list(range(TOTAL_FRAMES))

        assert frame_ids == expected_frame_ids, (
            f"Frame order mismatch. Expected {expected_frame_ids}, got {frame_ids}"
        )

        for item in consumed:
            expected_pixel = item["frame_id"] % 256

            assert item["mean_pixel"] == expected_pixel, (
                f"Frame data mismatch for frame_id={item['frame_id']}. "
                f"Expected mean pixel {expected_pixel}, "
                f"got {item['mean_pixel']}"
            )

        snapshot = ring_buffer.snapshot()

        print()
        print("=" * 60)
        print("Final Ring Buffer Snapshot")
        print("=" * 60)
        print(snapshot)

        print()
        print("Ring buffer producer-consumer test passed.")

    finally:
        ring_buffer.close()
        ring_buffer.unlink()


if __name__ == "__main__":
    main()