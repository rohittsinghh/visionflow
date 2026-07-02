"""
Detection metadata batch writer.

YOLO inference should not wait on PostgreSQL. This service receives detection
payloads from the live streaming fanout, flattens them into database rows, and
inserts rows in batches.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from sqlalchemy import text

from app.db import connection
from app.services import metrics_service


BATCH_SIZE = 1000
RETRY_FILE_PATH = Path("data/db_retry_queue.jsonl")

db_writer_queue = None
db_writer_task = None
crop_writer_queue = None
crop_writer_task = None
enabled = False
logger = logging.getLogger(__name__)


INSERT_DETECTIONS_SQL = text("""
INSERT INTO detections (
    camera_id,
    run_id,
    frame_id,
    class_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2
)
VALUES (
    :camera_id,
    :run_id,
    :frame_id,
    :class_id,
    :class_name,
    :confidence,
    :x1,
    :y1,
    :x2,
    :y2
)
""")

INSERT_FIRST_APPEARANCE_CROP_SQL = text("""
INSERT INTO first_appearance_crops (
    camera_id,
    run_id,
    frame_id,
    class_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
    crop_path,
    crop_url
)
VALUES (
    :camera_id,
    :run_id,
    :frame_id,
    :class_id,
    :class_name,
    :confidence,
    :x1,
    :y1,
    :x2,
    :y2,
    :crop_path,
    :crop_url
)
ON CONFLICT (camera_id, run_id, class_name) DO NOTHING
""")


def flatten_detection_payload(payload):
    """
    Convert one detection payload into database rows.
    """

    frame_id = payload.get("frame")
    run_id = payload.get("run_id")
    camera_id = payload.get("camera_id", "default")
    rows = []

    for detection in payload.get("detections", []):
        x1, y1, x2, y2 = detection["bbox"]

        rows.append(
            {
                "camera_id": camera_id,
                "run_id": run_id,
                "frame_id": frame_id,
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "confidence": detection["confidence"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    return rows


def flatten_first_appearance_event(event):
    """
    Convert one first-appearance crop event into a database row.
    """

    x1, y1, x2, y2 = event["bbox"]

    return {
        "camera_id": event.get("camera_id", "default"),
        "run_id": event["run_id"],
        "frame_id": event["frame"],
        "class_id": event["class_id"],
        "class_name": event["class_name"],
        "confidence": event["confidence"],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "crop_path": event["crop_path"],
        "crop_url": event["crop_url"],
    }


def append_retry_rows(rows, kind):
    """
    Persist failed rows to a local JSONL retry file.

    This keeps failed writes inspectable and replayable after transient DB
    issues without blocking the YOLO pipeline.
    """

    if not rows:
        return

    RETRY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RETRY_FILE_PATH.open("a", encoding="utf-8") as retry_file:
        for row in rows:
            retry_file.write(
                json.dumps(
                    {
                        "kind": kind,
                        "row": row,
                    }
                )
                + "\n"
            )

    metrics_service.record_db_failure(len(rows))


def load_retry_records():
    """
    Load retry-file records from disk.
    """

    if not RETRY_FILE_PATH.exists():
        return []

    records = []

    for line in RETRY_FILE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("invalid_db_retry_record_skipped")

    return records


def rewrite_retry_records(records):
    """
    Replace the retry file with records that still need replay.
    """

    if not records:
        RETRY_FILE_PATH.unlink(missing_ok=True)
        return

    RETRY_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RETRY_FILE_PATH.open("w", encoding="utf-8") as retry_file:
        for record in records:
            retry_file.write(json.dumps(record) + "\n")


async def replay_retry_file():
    """
    Try to insert rows from the local retry file.
    """

    records = load_retry_records()

    if not records:
        return

    logger.info("db_retry_replay_started records=%s", len(records))
    remaining_records = []

    for record in records:
        kind = record.get("kind")
        row = record.get("row")

        if not kind or not row:
            continue

        before_failures = metrics_service.snapshot()["db"]["insert_failures"]

        if kind == "detection":
            await insert_rows([row], retry_on_failure=False)
        elif kind == "crop":
            await insert_first_appearance_rows([row], retry_on_failure=False)
        else:
            continue

        after_failures = metrics_service.snapshot()["db"]["insert_failures"]

        if after_failures > before_failures:
            remaining_records.append(record)

    rewrite_retry_records(remaining_records)
    logger.info(
        "db_retry_replay_finished attempted=%s remaining=%s",
        len(records),
        len(remaining_records),
    )


async def insert_rows(rows, retry_on_failure=True):
    """
    Insert rows using one short-lived SQLAlchemy AsyncSession.

    The session is closed after the insert, which returns its underlying
    PostgreSQL connection to the engine pool. The pool itself remains alive
    until application shutdown.
    """

    if not rows:
        return

    if not connection.is_configured():
        return

    try:
        started_at = time.perf_counter()

        async with connection.get_session() as session:
            await session.execute(INSERT_DETECTIONS_SQL, rows)
            await session.commit()

        metrics_service.record_db_insert(
            len(rows),
            (time.perf_counter() - started_at) * 1000,
            "detection",
        )
    except Exception as exc:
        logger.exception("detection_batch_insert_failed rows=%s", len(rows))

        if retry_on_failure:
            append_retry_rows(rows, "detection")
        else:
            metrics_service.record_db_failure(len(rows))


async def insert_first_appearance_rows(rows, retry_on_failure=True):
    """
    Insert first-appearance crop metadata rows.
    """

    if not rows:
        return

    if not connection.is_configured():
        return

    try:
        started_at = time.perf_counter()

        async with connection.get_session() as session:
            await session.execute(INSERT_FIRST_APPEARANCE_CROP_SQL, rows)
            await session.commit()

        metrics_service.record_db_insert(
            len(rows),
            (time.perf_counter() - started_at) * 1000,
            "crop",
        )
    except Exception as exc:
        logger.exception("first_appearance_crop_insert_failed rows=%s", len(rows))

        if retry_on_failure:
            append_retry_rows(rows, "crop")
        else:
            metrics_service.record_db_failure(len(rows))


async def db_writer_loop():
    """
    Collect rows and flush only when BATCH_SIZE is reached.
    """

    batch = []

    while True:
        try:
            payload = await db_writer_queue.get()

            batch.extend(flatten_detection_payload(payload))

            if len(batch) >= BATCH_SIZE:
                rows_to_insert = batch[:BATCH_SIZE]
                batch = batch[BATCH_SIZE:]
                await insert_rows(rows_to_insert)

        except asyncio.CancelledError:
            if batch:
                await insert_rows(batch)

            raise


async def crop_writer_loop():
    """
    Insert first-appearance crop metadata as soon as events arrive.

    Unlike normal detections, first-appearance events are rare and important
    for the UI/DB history. Waiting for the large detection batch size would
    leave these rows in memory for too long, so each event is committed
    immediately.
    """

    while True:
        event = await crop_writer_queue.get()
        row = flatten_first_appearance_event(event)
        await insert_first_appearance_rows([row])


async def start_db_writer():
    """
    Start the database writer if PostgreSQL settings are configured.
    """

    global db_writer_queue
    global db_writer_task
    global crop_writer_queue
    global crop_writer_task
    global enabled

    if not await connection.init_db_engine():
        enabled = False
        return

    if db_writer_task is not None:
        enabled = True
        return

    await replay_retry_file()

    db_writer_queue = asyncio.Queue()
    crop_writer_queue = asyncio.Queue()
    db_writer_task = asyncio.create_task(db_writer_loop())
    crop_writer_task = asyncio.create_task(crop_writer_loop())
    enabled = True


async def stop_db_writer():
    """
    Stop the database writer and flush remaining rows.
    """

    global db_writer_task
    global db_writer_queue
    global crop_writer_task
    global crop_writer_queue
    global enabled

    if db_writer_task is not None:
        db_writer_task.cancel()

        try:
            await db_writer_task
        except asyncio.CancelledError:
            pass

        db_writer_task = None

    if crop_writer_task is not None:
        crop_writer_task.cancel()

        try:
            await crop_writer_task
        except asyncio.CancelledError:
            pass

        crop_writer_task = None

    db_writer_queue = None
    crop_writer_queue = None
    enabled = False

    await connection.dispose_db_engine()


async def enqueue_detection_payload(payload):
    """
    Queue a detection payload for database storage.
    """

    if not enabled or db_writer_queue is None:
        return

    await db_writer_queue.put(payload)


async def enqueue_first_appearance_event(event):
    """
    Queue a first-appearance crop event for database storage.
    """

    if not enabled or crop_writer_queue is None:
        return

    await crop_writer_queue.put(event)
