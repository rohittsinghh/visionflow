CREATE TABLE IF NOT EXISTS detections (
    id BIGSERIAL PRIMARY KEY,
    frame_id BIGINT NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE detections
    ADD COLUMN IF NOT EXISTS run_id TEXT;

CREATE INDEX IF NOT EXISTS idx_detections_frame_id
    ON detections(frame_id);

CREATE INDEX IF NOT EXISTS idx_detections_class_name
    ON detections(class_name);

CREATE INDEX IF NOT EXISTS idx_detections_created_at
    ON detections(created_at);

CREATE INDEX IF NOT EXISTS idx_detections_confidence
    ON detections(confidence);

CREATE INDEX IF NOT EXISTS idx_detections_run_id
    ON detections(run_id);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stopped_at TIMESTAMPTZ,
    video_path TEXT,
    model_path TEXT
);

CREATE TABLE IF NOT EXISTS first_appearance_crops (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    frame_id BIGINT NOT NULL,
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    crop_path TEXT NOT NULL,
    crop_url TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE first_appearance_crops
    ADD COLUMN IF NOT EXISTS crop_url TEXT;

UPDATE first_appearance_crops
    SET crop_url = crop_path
    WHERE crop_url IS NULL;

ALTER TABLE first_appearance_crops
    ALTER COLUMN crop_url SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_first_appearance_crops_run_class
    ON first_appearance_crops(run_id, class_name);

CREATE INDEX IF NOT EXISTS idx_first_appearance_crops_run_id
    ON first_appearance_crops(run_id);

CREATE INDEX IF NOT EXISTS idx_first_appearance_crops_class_name
    ON first_appearance_crops(class_name);
