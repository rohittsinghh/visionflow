from app.services import pipeline_service


def test_sanitize_camera_id_keeps_safe_url_identifier():
    assert pipeline_service.sanitize_camera_id(" Front Gate 01 ") == "front_gate_01"


def test_build_shared_memory_names_are_camera_scoped():
    names = pipeline_service.build_shared_memory_names("front_gate")
    flat_names = pipeline_service.flatten_shared_memory_names(names)

    assert flat_names == [
        "psm_frontgat_rf",
        "psm_frontgat_ri",
        "psm_frontgat_rs",
        "psm_frontgat_af",
        "psm_frontgat_ai",
        "psm_frontgat_as",
    ]
