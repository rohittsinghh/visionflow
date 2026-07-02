"""
Ring Buffer

This module implements a producer-consumer ring buffer using
multiprocessing shared memory.

The ring buffer stores multiple video frames so that the video reader
producer never overwrites a frame that the YOLO worker consumer has not
finished reading.

Shared memory is split into three separate blocks:

1. Frame buffer
   Stores raw image data.

2. Frame IDs
   Stores one monotonically increasing frame ID per slot.

3. Slot status
   Stores the lifecycle state of each slot.

Slot lifecycle:

    EMPTY -> READY -> PROCESSING -> EMPTY

A single multiprocessing.Lock protects all shared metadata and all frame
slot state transitions. The lock is held while copying frames into or out
of shared memory, but it is not held during YOLO inference.
"""

import time
from enum import IntEnum
from multiprocessing import Lock, Value
from multiprocessing.shared_memory import SharedMemory

import numpy as np


def shared_memory_exists(name):
    """
    Return True when a shared memory block with this name exists.
    """

    try:
        shared_memory = SharedMemory(name=name)
    except FileNotFoundError:
        return False

    shared_memory.close()
    return True


def unlink_shared_memory_by_name(name):
    """
    Unlink one shared memory block by name.

    Returns True when a block was found and unlinked, otherwise False.
    """

    try:
        shared_memory = SharedMemory(name=name)
    except FileNotFoundError:
        return False

    shared_memory.close()
    shared_memory.unlink()
    return True


def cleanup_shared_memory_names(names):
    """
    Unlink all existing shared memory blocks from a known name list.
    """

    unlinked_names = []

    for name in names:
        if name and unlink_shared_memory_by_name(name):
            unlinked_names.append(name)

    return unlinked_names


class SlotStatus(IntEnum):
    """
    State of a slot inside the ring buffer.
    """

    EMPTY = 0
    READY = 1
    PROCESSING = 2


class RingBuffer:
    """
    Producer-consumer ring buffer backed by multiprocessing shared memory.

    Parameters
    ----------
    frame_shape:
        Shape of one video frame, for example:
        (480, 640, 3)

    buffer_size:
        Number of frame slots in the ring buffer.

    frame_shm_name:
        Existing frame shared memory name. Use this when attaching from
        another process.

    id_shm_name:
        Existing frame ID shared memory name. Use this when attaching from
        another process.

    status_shm_name:
        Existing slot status shared memory name. Use this when attaching
        from another process.

    write_index:
        Shared multiprocessing.Value containing the producer index.

    read_index:
        Shared multiprocessing.Value containing the consumer index.

    next_frame_id:
        Shared multiprocessing.Value containing the next frame ID.

    lock:
        Shared multiprocessing.Lock protecting all reads, writes, metadata,
        and status transitions.

    create:
        If True, allocate new shared memory blocks.
        If False, attach to existing shared memory blocks.
    """

    def __init__(
        self,
        frame_shape,
        buffer_size=4,
        frame_shm_name=None,
        id_shm_name=None,
        status_shm_name=None,
        write_index=None,
        read_index=None,
        next_frame_id=None,
        lock=None,
        create=True,
    ):
        self.frame_shape = frame_shape
        self.buffer_size = buffer_size
        self.create = create

        frame_bytes = int(np.prod(frame_shape)) * np.uint8().nbytes
        id_bytes = buffer_size * np.int64().nbytes
        status_bytes = buffer_size * np.uint8().nbytes

        if create:
            self.frame_shm = SharedMemory(
                name=frame_shm_name,
                create=True,
                size=frame_bytes * buffer_size,
            )

            self.id_shm = SharedMemory(
                name=id_shm_name,
                create=True,
                size=id_bytes,
            )

            self.status_shm = SharedMemory(
                name=status_shm_name,
                create=True,
                size=status_bytes,
            )
        else:
            self.frame_shm = SharedMemory(
                name=frame_shm_name,
            )

            self.id_shm = SharedMemory(
                name=id_shm_name,
            )

            self.status_shm = SharedMemory(
                name=status_shm_name,
            )

        self.frames = np.ndarray(
            (
                buffer_size,
                frame_shape[0],
                frame_shape[1],
                frame_shape[2],
            ),
            dtype=np.uint8,
            buffer=self.frame_shm.buf,
        )

        self.frame_ids = np.ndarray(
            (buffer_size,),
            dtype=np.int64,
            buffer=self.id_shm.buf,
        )

        self.status = np.ndarray(
            (buffer_size,),
            dtype=np.uint8,
            buffer=self.status_shm.buf,
        )

        self.write_index = write_index if write_index is not None else Value("i", 0)
        self.read_index = read_index if read_index is not None else Value("i", 0)
        self.next_frame_id = next_frame_id if next_frame_id is not None else Value("q", 0)
        self.lock = lock if lock is not None else Lock()

        if create:
            self.frames[:] = 0
            self.frame_ids[:] = -1
            self.status[:] = SlotStatus.EMPTY

    @property
    def frame_shm_name(self):
        return self.frame_shm.name

    @property
    def id_shm_name(self):
        return self.id_shm.name

    @property
    def status_shm_name(self):
        return self.status_shm.name

    def write(self, frame, retry_sleep=0.005):
        """
        Write one frame into the next available ring buffer slot.

        The producer only writes into the current write slot when that slot
        is EMPTY. If the slot is not EMPTY, the buffer is full from the
        producer's point of view, so this method waits briefly and retries.

        Returns
        -------
        int
            The assigned frame ID.
        """

        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Expected frame shape {self.frame_shape}, got {frame.shape}"
            )

        while True:
            with self.lock:
                slot_index = self.write_index.value
                slot_status = SlotStatus(int(self.status[slot_index]))

                if slot_status == SlotStatus.EMPTY:
                    frame_id = self.next_frame_id.value

                    self.frames[slot_index][:] = frame
                    self.frame_ids[slot_index] = frame_id
                    self.status[slot_index] = SlotStatus.READY

                    self.next_frame_id.value += 1
                    self.write_index.value = (
                        self.write_index.value + 1
                    ) % self.buffer_size

                    return frame_id

            time.sleep(retry_sleep)

    def write_latest(self, frame, frame_id=None):
        """
        Write a latest-value frame, overwriting the current write slot.

        This method is for fanout-style outputs such as the annotated frame
        stream. Unlike write(), it does not wait for an EMPTY slot because
        HTTP streaming clients read frames non-destructively with read_latest().
        Slow or disconnected clients may miss old frames, but they cannot
        block the producer.
        """

        if frame.shape != self.frame_shape:
            raise ValueError(
                f"Expected frame shape {self.frame_shape}, got {frame.shape}"
            )

        with self.lock:
            slot_index = self.write_index.value

            if frame_id is None:
                frame_id = self.next_frame_id.value
                self.next_frame_id.value += 1
            else:
                self.next_frame_id.value = max(
                    self.next_frame_id.value,
                    frame_id + 1,
                )

            self.frames[slot_index][:] = frame
            self.frame_ids[slot_index] = frame_id
            self.status[slot_index] = SlotStatus.READY
            self.write_index.value = (
                self.write_index.value + 1
            ) % self.buffer_size

            return frame_id

    def read(self, retry_sleep=0.005):
        """
        Read the next frame in production order.

        The consumer only reads from the current read slot when that slot is
        READY. The slot is changed to PROCESSING while holding the lock, then
        the frame and frame ID are copied locally. After the copy is complete,
        read_index advances and the lock is released.

        The slot remains PROCESSING while YOLO inference runs. The caller must
        call mark_empty(slot_index) after inference completes.

        Returns
        -------
        tuple[int, numpy.ndarray, int]
            frame_id, frame_copy, slot_index
        """

        while True:
            with self.lock:
                slot_index = self.read_index.value
                slot_status = SlotStatus(int(self.status[slot_index]))

                if slot_status == SlotStatus.READY:
                    self.status[slot_index] = SlotStatus.PROCESSING

                    frame_id = int(self.frame_ids[slot_index])
                    frame = self.frames[slot_index].copy()

                    self.read_index.value = (
                        self.read_index.value + 1
                    ) % self.buffer_size

                    return frame_id, frame, slot_index

            time.sleep(retry_sleep)

    def read_latest(self, last_frame_id=-1):
        """
        Copy the newest available frame without consuming it.

        This is intentionally different from read(). Streaming HTTP clients
        should not move read_index or mark slots PROCESSING, otherwise one
        browser tab could steal frames from another. The method scans READY
        slots while holding the lock, copies the newest frame, and leaves the
        ring buffer state unchanged.

        Parameters
        ----------
        last_frame_id:
            The newest frame ID already sent to this client.

        Returns
        -------
        tuple[int | None, numpy.ndarray | None]
            A new frame ID and frame copy when available, otherwise
            (None, None).
        """

        with self.lock:
            latest_slot_index = None
            latest_frame_id = last_frame_id

            for slot_index in range(self.buffer_size):
                slot_status = SlotStatus(int(self.status[slot_index]))
                frame_id = int(self.frame_ids[slot_index])

                if (
                    slot_status == SlotStatus.READY
                    and frame_id > latest_frame_id
                ):
                    latest_slot_index = slot_index
                    latest_frame_id = frame_id

            if latest_slot_index is None:
                return None, None

            return latest_frame_id, self.frames[latest_slot_index].copy()

    def mark_empty(self, slot_index):
        """
        Mark a PROCESSING slot as EMPTY after inference is finished.

        This completes the valid lifecycle:

            READY -> PROCESSING -> EMPTY

        The slot can then be reused by the producer.
        """

        with self.lock:
            slot_status = SlotStatus(int(self.status[slot_index]))

            if slot_status != SlotStatus.PROCESSING:
                raise RuntimeError(
                    f"Cannot mark slot {slot_index} EMPTY because "
                    f"its current status is {slot_status.name}"
                )

            self.status[slot_index] = SlotStatus.EMPTY
            self.frame_ids[slot_index] = -1

    def snapshot(self):
        """
        Return a small debugging snapshot of ring buffer metadata.
        """

        with self.lock:
            return {
                "frame_shm_name": self.frame_shm_name,
                "id_shm_name": self.id_shm_name,
                "status_shm_name": self.status_shm_name,
                "write_index": self.write_index.value,
                "read_index": self.read_index.value,
                "next_frame_id": self.next_frame_id.value,
                "frame_ids": self.frame_ids.copy().tolist(),
                "status": [
                    SlotStatus(int(value)).name
                    for value in self.status.copy().tolist()
                ],
            }

    def close(self):
        """
        Close this process's handles to shared memory.
        """

        self.frame_shm.close()
        self.id_shm.close()
        self.status_shm.close()

    def unlink(self):
        """
        Destroy shared memory blocks.

        This should only be called by the process that created the shared
        memory, after all worker processes have stopped.
        """

        for shared_memory in (
            self.frame_shm,
            self.id_shm,
            self.status_shm,
        ):
            try:
                shared_memory.unlink()
            except FileNotFoundError:
                pass
