from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime

from config import AppConfig, get_logger
from processor import BoltHoleProcessor


@dataclass(frozen=True)
class RuntimeSession:
    processor: BoltHoleProcessor
    task_started_at: datetime


class DashboardRuntime:
    """Owns the one active background processor inside the Streamlit server."""

    def __init__(self) -> None:
        self.config = AppConfig.load()
        self.logger = get_logger("system.runtime", self.config)
        self._lock = threading.RLock()
        self._session: RuntimeSession | None = None

    def new_task_session(self, clear_database: bool = True) -> RuntimeSession:
        with self._lock:
            if self._session is not None:
                try:
                    self._session.processor.stop()
                except Exception as exc:
                    self.logger.exception("Failed to stop previous task during dashboard refresh: %s", exc)

            processor = BoltHoleProcessor(self.config)
            if clear_database:
                processor.database.clear_all()
            self._session = RuntimeSession(processor=processor, task_started_at=datetime.utcnow())
            self.logger.info("Prepared new dashboard task session clear_database=%s", clear_database)
            return self._session

    def current_session(self) -> RuntimeSession:
        with self._lock:
            if self._session is None:
                return self.new_task_session(clear_database=True)
            return self._session
