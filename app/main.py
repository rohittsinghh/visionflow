"""
Main entry point of the application.

This file only creates the FastAPI app, includes routers, and wires
application lifecycle hooks. Pipeline and streaming logic live in services.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.history import router as history_router
from app.api.observability import router as observability_router
from app.api.pipeline import router as pipeline_router
from app.api.streaming import router as streaming_router
from app.api.ui import router as ui_router
from app.core.logging_config import configure_logging
from app.services import db_writer_service
from app.services import pipeline_service
from app.services import storage_service
from app.services import streaming_service


configure_logging()

app = FastAPI(title="YOLO Backend")

storage_service.CROP_ROOT.mkdir(parents=True, exist_ok=True)

app.include_router(pipeline_router)
app.include_router(streaming_router)
app.include_router(ui_router)
app.include_router(history_router)
app.include_router(observability_router)
app.mount("/crops", StaticFiles(directory=storage_service.CROP_ROOT), name="crops")


@app.on_event("startup")
async def startup_event():
    """
    Start background services.
    """

    await db_writer_service.start_db_writer()
    await streaming_service.start_detection_fanout()


@app.on_event("shutdown")
async def shutdown_event():
    """
    Stop background services and release pipeline resources.
    """

    await streaming_service.stop_detection_fanout()
    await db_writer_service.stop_db_writer()
    pipeline_service.cleanup_pipeline()
