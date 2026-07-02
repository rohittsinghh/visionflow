"""
Application configuration.
"""

import os

from app.db.connection import load_env_file


load_env_file()

API_KEY = os.getenv("API_KEY", "")
LOG_EVERY_N_FRAMES = int(os.getenv("LOG_EVERY_N_FRAMES", "100"))
