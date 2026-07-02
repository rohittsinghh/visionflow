"""
Local object-storage style helpers.

The app currently stores crop images on local disk, but callers use object-key
style paths so the implementation can later move to S3 or MinIO with less
pipeline churn.
"""

from pathlib import Path


STORAGE_ROOT = Path("storage")
CROP_ROOT = STORAGE_ROOT / "crops"


def crop_object_key(camera_id, run_id, filename):
    """
    Return the object key for a crop image.
    """

    return f"{camera_id}/{run_id}/{filename}"


def crop_path(camera_id, run_id, filename):
    """
    Return the local filesystem path for a crop image.
    """

    return CROP_ROOT / crop_object_key(camera_id, run_id, filename)


def crop_url(camera_id, run_id, filename):
    """
    Return the HTTP URL path for a crop image.
    """

    return f"/crops/{crop_object_key(camera_id, run_id, filename)}"


def ensure_crop_dir(camera_id, run_id):
    """
    Ensure the local crop directory exists for a run.
    """

    directory = CROP_ROOT / camera_id / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory
