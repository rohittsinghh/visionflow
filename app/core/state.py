"""
Shared application state.

These queues are used to send worker events from the YOLO process to FastAPI.
"""

from multiprocessing import Queue

# Normal per-frame detection results.
result_queue = Queue()

# First appearance crop events. The YOLO worker puts one event here the first
# time each class appears during a pipeline run.
first_appearance_queue = Queue()
