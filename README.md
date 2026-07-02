# VisionFlow

VisionFlow is a real-time video analytics backend built with FastAPI, multiprocessing, shared-memory ring buffers, ONNX YOLO inference, SSE event streaming, MJPEG annotated-frame streaming, PostgreSQL persistence, and local crop artifact storage.

The project is structured like a production backend system, not just a YOLO script. Video frames move between processes through shared memory, detections are streamed to clients, metadata is written to PostgreSQL in batches, and first-appearance object crops are saved as storage artifacts.

## Features

- Multi-camera/video pipeline support using `camera_id`
- Raw frame producer and YOLO consumer running in separate processes
- Shared-memory ring buffers using `multiprocessing.shared_memory`
- Annotated-frame ring buffer for live MJPEG streaming
- Server-Sent Events for live detection events
- PostgreSQL persistence with SQLAlchemy async engine and asyncpg driver
- Detection batch writer with retry-file fallback
- First-appearance object crop storage and metadata persistence
- Health and metrics endpoints
- API key protection for control/history endpoints
- Docker Compose setup for app + PostgreSQL
- Pytest coverage for auth, DB row conversion, crop events, retry file logic, and pipeline naming

## Architecture

```mermaid
flowchart LR
    Client["Client / Browser / API User"]
    API["FastAPI App"]
    Pipeline["Pipeline Service"]
    VideoReader["Video Reader Process"]
    RawRing["Raw Frame Ring Buffer<br/>Shared Memory"]
    YOLO["YOLO Worker Process<br/>ONNX Runtime"]
    AnnotatedRing["Annotated Frame Ring Buffer<br/>Shared Memory"]
    SSE["SSE Detection Streams"]
    MJPEG["MJPEG Frame Stream"]
    Queue["Multiprocessing Queues"]
    DBWriter["DB Writer Service"]
    Postgres["PostgreSQL"]
    Storage["Local Crop Storage<br/>storage/crops"]

    Client --> API
    API --> Pipeline
    Pipeline --> VideoReader
    Pipeline --> YOLO
    VideoReader --> RawRing
    RawRing --> YOLO
    YOLO --> AnnotatedRing
    AnnotatedRing --> MJPEG
    MJPEG --> Client
    YOLO --> Queue
    Queue --> SSE
    SSE --> Client
    Queue --> DBWriter
    DBWriter --> Postgres
    YOLO --> Storage
    DBWriter --> Postgres
```

## Frame Processing Flow

```mermaid
sequenceDiagram
    participant VR as Video Reader
    participant RB as Raw Ring Buffer
    participant YW as YOLO Worker
    participant ARB as Annotated Ring Buffer
    participant API as FastAPI Streaming
    participant DB as PostgreSQL

    VR->>RB: write(frame, frame_id)
    Note over RB: Slot status EMPTY -> READY
    YW->>RB: read next READY slot
    Note over RB: Slot status READY -> PROCESSING
    RB-->>YW: copied frame + frame_id
    Note over YW: Lock released before YOLO inference
    YW->>YW: run ONNX YOLO detection
    YW->>ARB: write annotated frame
    YW->>API: publish detection JSON through queue
    YW->>RB: mark slot EMPTY
    API->>DB: batch insert detection metadata
    API-->>API: stream SSE + MJPEG to clients
```

## Shared Memory Ring Buffer

The project avoids sending full video frames through normal Python queues because that would copy large image arrays between processes. Instead, it uses `multiprocessing.shared_memory` so the video reader and YOLO worker can access the same frame memory.

Each ring buffer uses separate shared memory blocks:

- frame buffer: image data
- frame IDs: one continuously increasing ID per slot
- slot statuses: lifecycle state per slot

Each slot follows this lifecycle:

```text
EMPTY -> READY -> PROCESSING -> EMPTY
```

The ring buffer uses one global `multiprocessing.Lock` per buffer. The lock protects frame copies, slot status, frame IDs, `write_index`, `read_index`, and `next_frame_id`. The lock is not held during YOLO inference, so expensive model execution does not block the producer from using the buffer longer than necessary.

There are two ring buffers per camera:

| Buffer | Producer | Consumer | Purpose |
| --- | --- | --- | --- |
| Raw frame ring buffer | Video reader | YOLO worker | Moves original frames into inference |
| Annotated frame ring buffer | YOLO worker | FastAPI `/frame` stream | Exposes latest annotated frames to clients |

For the default camera, shared memory names remain short and fixed:

```text
psm_yrf  raw frames
psm_yri  raw frame IDs
psm_yrs  raw slot statuses
psm_yaf  annotated frames
psm_yai  annotated frame IDs
psm_yas  annotated slot statuses
```

Additional cameras use camera-scoped names derived from `camera_id`.

## Multi-Camera Model

A camera is identified by a `camera_id`, for example:

```text
front_gate
back_gate
parking
warehouse_1
```

Each camera gets an isolated runtime pipeline:

- one video reader process
- one YOLO worker process
- one raw shared-memory ring buffer
- one annotated shared-memory ring buffer
- one detection SSE stream
- one first-appearance crop stream
- camera-scoped database rows
- camera-scoped crop folder

Current sources are OpenCV-compatible inputs. In local development, that usually means video files:

```json
{
  "source": "videos/sample.mp4"
}
```

Later, the same source field can point to RTSP streams:

```json
{
  "source": "rtsp://user:password@camera-ip:554/stream1"
}
```

## Backend Components

| Path | Responsibility |
| --- | --- |
| `app/main.py` | Creates FastAPI app, includes routers, registers startup/shutdown hooks |
| `app/api/pipeline.py` | Start/stop camera pipelines and expose buffer status |
| `app/api/streaming.py` | MJPEG and SSE streaming endpoints |
| `app/api/history.py` | Database-backed detection and run history APIs |
| `app/api/observability.py` | `/health` and `/metrics` |
| `app/api/ui.py` | Small manual UI for first-appearance crop viewing |
| `app/core/ring_buffer.py` | Shared-memory ring buffer implementation |
| `app/core/auth.py` | API key dependency |
| `app/core/logging_config.py` | Central logging setup |
| `app/services/pipeline_service.py` | Multi-camera process and ring-buffer ownership |
| `app/services/streaming_service.py` | Queue fanout, SSE generators, MJPEG generator |
| `app/services/db_writer_service.py` | Detection batching, crop metadata writes, retry file handling |
| `app/services/history_service.py` | PostgreSQL read/query helpers |
| `app/services/storage_service.py` | Local object-storage style crop paths |
| `app/workers/video_reader.py` | Reads frames from video source into raw ring buffer |
| `app/workers/yolo_worker.py` | Runs YOLO, draws boxes, writes annotated frames, emits detections |
| `app/ml/yolo_onnx.py` | ONNX Runtime YOLO inference wrapper |
| `app/db/schema.sql` | PostgreSQL tables and indexes |

## Database Flow

Detection payloads are produced by the YOLO worker and sent to FastAPI through a multiprocessing queue. FastAPI drains that queue once, then fans out the same payload to:

- SSE clients
- metrics
- PostgreSQL writer

This prevents SSE clients and DB writing from competing for the same queue messages.

Normal detection rows are accumulated in memory and inserted in batches. First-appearance crop rows are inserted immediately because they are rare and important for the UI.

PostgreSQL access uses:

- SQLAlchemy async engine
- `asyncpg` driver
- small connection pool
- short-lived sessions for inserts/queries

The app keeps the engine pool alive while the server runs, but individual sessions are opened and closed per DB operation.

## First-Appearance Crops

When YOLO sees a class for the first time in a pipeline run, the worker crops the detected bounding box from the original frame and saves it under:

```text
storage/crops/{camera_id}/{run_id}/{frame_id}_{class_name}.jpg
```

The database stores crop metadata and the URL path, not the raw image bytes. This follows an object-storage style pattern and makes it easier to move from local disk to S3 or MinIO later.

## API Overview

### Pipeline

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Basic status |
| `POST` | `/start-video` | Start default camera using `videos/sample.mp4` |
| `POST` | `/stop-video` | Stop default camera |
| `GET` | `/buffer-status` | Inspect default camera buffers |
| `GET` | `/cameras` | List active camera pipelines |
| `POST` | `/cameras/{camera_id}/start` | Start a named camera/video source |
| `POST` | `/cameras/{camera_id}/stop` | Stop a named camera |
| `GET` | `/cameras/{camera_id}/buffer-status` | Inspect one camera's buffers |

### Streaming

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/frame` | Default camera MJPEG annotated stream |
| `GET` | `/cameras/{camera_id}/frame` | Camera-specific MJPEG annotated stream |
| `GET` | `/detections/events` | Default camera detection SSE stream |
| `GET` | `/cameras/{camera_id}/detections/events` | Camera-specific detection SSE stream |
| `GET` | `/first-appearances/events` | Default camera crop SSE stream |
| `GET` | `/cameras/{camera_id}/first-appearances/events` | Camera-specific crop SSE stream |
| `GET` | `/first-appearances` | Simple browser UI for crop events |

### History And Observability

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service health and active cameras |
| `GET` | `/metrics` | Runtime metrics |
| `GET` | `/detections/latest` | Latest detection rows |
| `GET` | `/detections/history` | Detection history with filters |
| `GET` | `/first-appearances/history` | Crop metadata history |
| `GET` | `/runs/{run_id}/summary` | Pipeline run summary |
| `GET` | `/crops/...` | Static crop image files |

## API Key Auth

Control and history endpoints are protected when `API_KEY` is configured.

Set this in `.env`:

```env
API_KEY=change-me
```

Then call protected endpoints with:

```bash
curl -X POST http://127.0.0.1:8000/start-video \
  -H "x-api-key: change-me"
```

If `API_KEY` is empty, auth is disabled for local development.

## Setup

Create a virtual environment and install dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

Make sure these files exist:

```text
models/yolov8n.onnx
videos/sample.mp4
```

Run the app:

```bash
.venv/bin/uvicorn app.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## Docker Compose

Start the app and PostgreSQL:

```bash
docker compose up --build
```

The app runs on:

```text
http://127.0.0.1:8000
```

PostgreSQL uses the values from `.env`.

## Manual Testing

Start the default camera:

```bash
curl -X POST http://127.0.0.1:8000/start-video \
  -H "x-api-key: change-me"
```

Start a named camera:

```bash
curl -X POST http://127.0.0.1:8000/cameras/front_gate/start \
  -H "x-api-key: change-me" \
  -H "Content-Type: application/json" \
  -d '{"source":"videos/sample.mp4"}'
```

Watch detection events:

```bash
curl -N http://127.0.0.1:8000/cameras/front_gate/detections/events
```

Open annotated frames in a browser:

```text
http://127.0.0.1:8000/cameras/front_gate/frame
```

Check ring buffer state:

```bash
curl http://127.0.0.1:8000/cameras/front_gate/buffer-status \
  -H "x-api-key: change-me"
```

Stop the camera:

```bash
curl -X POST http://127.0.0.1:8000/cameras/front_gate/stop \
  -H "x-api-key: change-me"
```

## PostgreSQL Checks

Connect with `psql`:

```bash
psql "postgresql://yolo_user:yolo_password@127.0.0.1:5432/yolo_video"
```

Useful queries:

```sql
SELECT camera_id, run_id, frame_id, class_name, confidence, created_at
FROM detections
ORDER BY id DESC
LIMIT 20;

SELECT camera_id, run_id, class_name, crop_url, created_at
FROM first_appearance_crops
ORDER BY id DESC
LIMIT 20;

SELECT run_id, camera_id, status, started_at, stopped_at
FROM pipeline_runs
ORDER BY started_at DESC
LIMIT 10;
```

## Tests

Run compile checks:

```bash
.venv/bin/python -m compileall app scratch test_sse.py tests
```

Run unit tests:

```bash
.venv/bin/python -m pytest tests
```

## Current Limitations

- Each camera loads its own YOLO worker and model instance, which is simple but memory-heavy.
- Multi-camera support currently treats video files as camera sources; RTSP support should work if OpenCV can open the stream, but it has not been hardened.
- Pending in-memory DB batches can be lost if the process crashes before a flush.
- Local crop storage is not durable object storage yet.
- Fixed shared-memory names assume a single local app instance, not multiple uvicorn workers.

## Good Next Improvements

- Shared inference worker pool or GPU batch inference
- RTSP reconnect/backoff handling
- Redis Streams or Kafka for event fanout
- MinIO/S3 crop storage backend
- Alembic migrations
- Prometheus metrics endpoint
- JWT/API key management
- CI with test and lint jobs
- Load testing for multi-camera workloads

## Resume Highlights

- Built a real-time video analytics backend using FastAPI, multiprocessing, shared-memory ring buffers, and ONNX YOLO inference.
- Implemented MJPEG annotated-frame streaming and SSE detection events with shared-latest fanout semantics.
- Designed PostgreSQL persistence for detection metadata, pipeline runs, and first-appearance crop artifacts.
- Added SQLAlchemy async pooling, batch writes, retry-file fallback, API key auth, health checks, metrics, and Docker Compose deployment.
- Extended the system to support multiple independent camera pipelines with camera-scoped buffers, streams, storage, and database rows.
