from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from models import DetectionRecord


class DetectionDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hole_id TEXT NOT NULL UNIQUE,
                    distance TEXT NOT NULL,
                    frame_number INTEGER NOT NULL,
                    detection_confidence REAL NOT NULL,
                    ocr_confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_distance ON detections(distance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_detections_unique_distance "
                "ON detections(distance) WHERE distance != ''"
            )
            conn.commit()

    def insert_detection(self, record: DetectionRecord) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO detections
                (hole_id, distance, frame_number, detection_confidence, ocr_confidence, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.hole_id,
                    record.distance,
                    record.frame_number,
                    record.detection_confidence,
                    record.ocr_confidence,
                    record.timestamp.isoformat(),
                ),
            )
            conn.commit()
            return cursor.rowcount == 1

    def fetch_all(self) -> list[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            return list(conn.execute("SELECT * FROM detections ORDER BY timestamp ASC"))

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM detections").fetchone()
            return int(row["c"])

    def clear_all(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM detections")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'detections'")
            conn.commit()

    def iter_unique_for_report(self) -> Iterable[sqlite3.Row]:
        rows = self.fetch_all()
        seen: set[str] = set()
        for row in rows:
            if row["hole_id"] in seen:
                continue
            seen.add(row["hole_id"])
            yield row
