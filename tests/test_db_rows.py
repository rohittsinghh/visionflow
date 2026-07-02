from app.services.db_writer_service import flatten_detection_payload
from app.services.db_writer_service import flatten_first_appearance_event


def test_flatten_detection_payload_includes_run_id():
    payload = {
        "run_id": "run_1",
        "frame": 10,
        "detections": [
            {
                "class_id": 2,
                "class_name": "car",
                "confidence": 0.8,
                "bbox": [1, 2, 3, 4],
            }
        ],
    }

    rows = flatten_detection_payload(payload)

    assert rows == [
        {
            "run_id": "run_1",
            "frame_id": 10,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8,
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
        }
    ]


def test_flatten_first_appearance_event():
    event = {
        "run_id": "run_1",
        "frame": 10,
        "class_id": 5,
        "class_name": "bus",
        "confidence": 0.9,
        "bbox": [5, 6, 7, 8],
        "crop_path": "storage/crops/run_1/10_bus.jpg",
        "crop_url": "/crops/run_1/10_bus.jpg",
    }

    row = flatten_first_appearance_event(event)

    assert row["run_id"] == "run_1"
    assert row["frame_id"] == 10
    assert row["class_name"] == "bus"
    assert row["crop_url"] == "/crops/run_1/10_bus.jpg"
