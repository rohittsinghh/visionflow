from pathlib import Path

import numpy as np

from app.services import storage_service
from app.workers import yolo_worker


def test_create_first_appearance_event_uses_storage_style_path(tmp_path):
    original_crop_root = storage_service.CROP_ROOT
    storage_service.CROP_ROOT = tmp_path / "storage" / "crops"

    try:
        frame = np.full((40, 60, 3), 127, dtype=np.uint8)
        detection = {
            "class_id": 5,
            "class_name": "city bus",
            "confidence": 0.91,
            "bbox": [5, 6, 25, 30],
        }

        event = yolo_worker.create_first_appearance_event(
            "run_test",
            42,
            frame,
            detection,
        )

        assert event["class_name"] == "city bus"
        assert event["crop_url"] == "/crops/run_test/42_city_bus.jpg"
        assert Path(event["crop_path"]).exists()
    finally:
        storage_service.CROP_ROOT = original_crop_root


def test_crop_detection_clips_bbox_to_frame():
    frame = np.full((20, 30, 3), 255, dtype=np.uint8)

    crop = yolo_worker.crop_detection(frame, [-10, -10, 40, 25])

    assert crop.shape == frame.shape
