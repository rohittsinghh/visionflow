from app.services import db_writer_service


def test_retry_file_round_trip(tmp_path):
    original_retry_file = db_writer_service.RETRY_FILE_PATH
    db_writer_service.RETRY_FILE_PATH = tmp_path / "retry.jsonl"

    try:
        row = {
            "run_id": "run_1",
            "frame_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8,
            "x1": 1,
            "y1": 2,
            "x2": 3,
            "y2": 4,
        }

        db_writer_service.append_retry_rows([row], "detection")
        records = db_writer_service.load_retry_records()

        assert records == [
            {
                "kind": "detection",
                "row": row,
            }
        ]

        db_writer_service.rewrite_retry_records([])

        assert not db_writer_service.RETRY_FILE_PATH.exists()
    finally:
        db_writer_service.RETRY_FILE_PATH = original_retry_file
