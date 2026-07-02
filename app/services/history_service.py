"""
Database-backed query helpers for history endpoints.
"""

from sqlalchemy import text

from app.db import connection


async def fetch_all(statement, params=None):
    """
    Run a SELECT query and return rows as dictionaries.
    """

    if not connection.is_configured():
        return []

    async with connection.get_session() as session:
        result = await session.execute(text(statement), params or {})
        return [dict(row._mapping) for row in result.fetchall()]


async def fetch_one(statement, params=None):
    """
    Run a SELECT query and return one dictionary.
    """

    rows = await fetch_all(statement, params)
    return rows[0] if rows else None


async def latest_detections(limit=20):
    """
    Return the newest detection rows.
    """

    return await fetch_all(
        """
        SELECT id, run_id, frame_id, class_id, class_name, confidence,
               x1, y1, x2, y2, created_at
        FROM detections
        ORDER BY id DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


async def detection_history(class_name=None, run_id=None, limit=100):
    """
    Return detection history with optional filters.
    """

    filters = []
    params = {"limit": limit}

    if class_name:
        filters.append("class_name = :class_name")
        params["class_name"] = class_name

    if run_id:
        filters.append("run_id = :run_id")
        params["run_id"] = run_id

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""

    return await fetch_all(
        f"""
        SELECT id, run_id, frame_id, class_id, class_name, confidence,
               x1, y1, x2, y2, created_at
        FROM detections
        {where_sql}
        ORDER BY id DESC
        LIMIT :limit
        """,
        params,
    )


async def first_appearance_history(run_id=None, limit=100):
    """
    Return first-appearance crop metadata.
    """

    params = {"limit": limit}
    where_sql = ""

    if run_id:
        where_sql = "WHERE run_id = :run_id"
        params["run_id"] = run_id

    return await fetch_all(
        f"""
        SELECT id, run_id, frame_id, class_id, class_name, confidence,
               x1, y1, x2, y2, crop_path, crop_url, created_at
        FROM first_appearance_crops
        {where_sql}
        ORDER BY id DESC
        LIMIT :limit
        """,
        params,
    )


async def run_summary(run_id):
    """
    Return one run with aggregate detection/crop counts.
    """

    return await fetch_one(
        """
        SELECT r.run_id, r.status, r.started_at, r.stopped_at,
               r.video_path, r.model_path,
               COUNT(d.id) AS detection_rows,
               COUNT(DISTINCT d.class_name) AS detection_classes,
               COUNT(DISTINCT c.class_name) AS first_appearance_classes
        FROM pipeline_runs r
        LEFT JOIN detections d ON d.run_id = r.run_id
        LEFT JOIN first_appearance_crops c ON c.run_id = r.run_id
        WHERE r.run_id = :run_id
        GROUP BY r.run_id, r.status, r.started_at, r.stopped_at,
                 r.video_path, r.model_path
        """,
        {"run_id": run_id},
    )


async def create_run(run_id, video_path, model_path):
    """
    Insert a running pipeline run record.
    """

    if not connection.is_configured():
        return

    async with connection.get_session() as session:
        await session.execute(
            text(
                """
                INSERT INTO pipeline_runs (run_id, status, video_path, model_path)
                VALUES (:run_id, 'running', :video_path, :model_path)
                ON CONFLICT (run_id) DO UPDATE
                SET status = 'running',
                    stopped_at = NULL,
                    video_path = EXCLUDED.video_path,
                    model_path = EXCLUDED.model_path
                """
            ),
            {
                "run_id": run_id,
                "video_path": video_path,
                "model_path": model_path,
            },
        )
        await session.commit()


async def stop_run(run_id, status="stopped"):
    """
    Mark a pipeline run as stopped.
    """

    if not run_id or not connection.is_configured():
        return

    async with connection.get_session() as session:
        await session.execute(
            text(
                """
                UPDATE pipeline_runs
                SET status = :status,
                    stopped_at = NOW()
                WHERE run_id = :run_id
                """
            ),
            {
                "run_id": run_id,
                "status": status,
            },
        )
        await session.commit()
