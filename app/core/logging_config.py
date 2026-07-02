"""
Central logging configuration.
"""

import logging
import os

from app.db.connection import load_env_file


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_LOG_LEVEL = "INFO"


def configure_logging():
    """
    Configure standard Python logging for the whole application.
    """

    load_env_file()
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        force=True,
    )

    logging.getLogger("multipart").setLevel(logging.WARNING)
