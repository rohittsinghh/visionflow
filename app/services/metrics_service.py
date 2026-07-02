"""
Runtime metrics and health snapshots.

Metrics are kept in process memory because they describe the currently running
FastAPI process and worker fanout state. They are intentionally simple JSON
values so they can be returned from /health and /metrics without extra
dependencies.
"""

import time


started_at = time.time()
frames_processed = 0
detections_seen = 0
first_appearance_events_seen = 0
sse_detection_clients = 0
sse_first_appearance_clients = 0
mjpeg_clients = 0
last_detection_frame = None
last_detection_at = None
last_first_appearance_at = None
db_detection_batches_inserted = 0
db_detection_rows_inserted = 0
db_crop_rows_inserted = 0
db_insert_failures = 0
db_retry_rows_written = 0
last_db_insert_latency_ms = None


def record_detection_payload(payload):
    """
    Record a detection payload drained from the worker queue.
    """

    global frames_processed
    global detections_seen
    global last_detection_frame
    global last_detection_at

    frames_processed += 1
    detections_seen += len(payload.get("detections", []))
    last_detection_frame = payload.get("frame")
    last_detection_at = time.time()


def record_first_appearance_event():
    """
    Record one first-appearance crop event.
    """

    global first_appearance_events_seen
    global last_first_appearance_at

    first_appearance_events_seen += 1
    last_first_appearance_at = time.time()


def record_db_insert(row_count, latency_ms, kind):
    """
    Record a successful database insert.
    """

    global db_detection_batches_inserted
    global db_detection_rows_inserted
    global db_crop_rows_inserted
    global last_db_insert_latency_ms

    last_db_insert_latency_ms = round(latency_ms, 3)

    if kind == "detection":
        db_detection_batches_inserted += 1
        db_detection_rows_inserted += row_count
    elif kind == "crop":
        db_crop_rows_inserted += row_count


def record_db_failure(row_count):
    """
    Record a failed database write and retry-file fallback.
    """

    global db_insert_failures
    global db_retry_rows_written

    db_insert_failures += 1
    db_retry_rows_written += row_count


def increment_client(kind):
    """
    Increment a streaming client counter.
    """

    global sse_detection_clients
    global sse_first_appearance_clients
    global mjpeg_clients

    if kind == "detections":
        sse_detection_clients += 1
    elif kind == "first_appearances":
        sse_first_appearance_clients += 1
    elif kind == "mjpeg":
        mjpeg_clients += 1


def decrement_client(kind):
    """
    Decrement a streaming client counter.
    """

    global sse_detection_clients
    global sse_first_appearance_clients
    global mjpeg_clients

    if kind == "detections":
        sse_detection_clients = max(0, sse_detection_clients - 1)
    elif kind == "first_appearances":
        sse_first_appearance_clients = max(0, sse_first_appearance_clients - 1)
    elif kind == "mjpeg":
        mjpeg_clients = max(0, mjpeg_clients - 1)


def snapshot():
    """
    Return current runtime metrics.
    """

    uptime_seconds = max(0, time.time() - started_at)

    return {
        "uptime_seconds": round(uptime_seconds, 3),
        "frames_processed": frames_processed,
        "detections_seen": detections_seen,
        "first_appearance_events_seen": first_appearance_events_seen,
        "average_fps": round(frames_processed / uptime_seconds, 3)
        if uptime_seconds
        else 0,
        "last_detection_frame": last_detection_frame,
        "last_detection_at": last_detection_at,
        "last_first_appearance_at": last_first_appearance_at,
        "clients": {
            "mjpeg": mjpeg_clients,
            "detections_sse": sse_detection_clients,
            "first_appearances_sse": sse_first_appearance_clients,
        },
        "db": {
            "detection_batches_inserted": db_detection_batches_inserted,
            "detection_rows_inserted": db_detection_rows_inserted,
            "crop_rows_inserted": db_crop_rows_inserted,
            "insert_failures": db_insert_failures,
            "retry_rows_written": db_retry_rows_written,
            "last_insert_latency_ms": last_db_insert_latency_ms,
        },
    }
