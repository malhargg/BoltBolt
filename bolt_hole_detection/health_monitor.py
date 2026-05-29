from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from config import AppConfig, get_logger
from models import RuntimeMetrics


class HealthMonitor:
    def __init__(
        self,
        config: AppConfig,
        metrics: RuntimeMetrics,
        frame_queue: queue.Queue,
        detection_queue: queue.Queue,
        event_queue: queue.Queue,
        lock: threading.RLock,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.frame_queue = frame_queue
        self.detection_queue = detection_queue
        self.event_queue = event_queue
        self.lock = lock
        self.logger = get_logger("system.health", config)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_capture_count = 0
        self._last_process_count = 0
        self._last_time = time.perf_counter()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="HealthMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        interval = self.config.logging.health_log_interval_seconds
        while not self._stop_event.wait(interval):
            now = time.perf_counter()
            elapsed = max(0.001, now - self._last_time)
            with self.lock:
                self.metrics.capture_fps = (self.metrics.frames_captured - self._last_capture_count) / elapsed
                self.metrics.processing_fps = (self.metrics.frames_processed - self._last_process_count) / elapsed
                self.metrics.frame_queue_size = self.frame_queue.qsize()
                self.metrics.detection_queue_size = self.detection_queue.qsize()
                self.metrics.event_queue_size = self.event_queue.qsize()
                snapshot = RuntimeMetrics(**self.metrics.__dict__)
                self._last_capture_count = self.metrics.frames_captured
                self._last_process_count = self.metrics.frames_processed
            self._last_time = now
            self.logger.info(
                "health frames_captured=%s frames_processed=%s holes=%s capture_fps=%.2f processing_fps=%.2f queues=(%s,%s,%s) rss_mb=%s",
                snapshot.frames_captured,
                snapshot.frames_processed,
                snapshot.holes_detected,
                snapshot.capture_fps,
                snapshot.processing_fps,
                snapshot.frame_queue_size,
                snapshot.detection_queue_size,
                snapshot.event_queue_size,
                self._rss_mb(),
            )

    @staticmethod
    def _rss_mb() -> str:
        try:
            import psutil  # type: ignore

            return f"{psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.1f}"
        except Exception:
            return "unavailable"
