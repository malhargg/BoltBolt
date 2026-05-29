from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from database import DetectionDatabase


class ReportManager:
    def __init__(self, database: DetectionDatabase, reports_dir: Path) -> None:
        self.database = database
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_csv(self, filename: str | None = None) -> Path:
        rows = list(self.database.iter_unique_for_report())
        rows.sort(key=lambda row: self._distance_sort_key(str(row["distance"])))
        report_path = self.reports_dir / (filename or f"bolt_hole_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Hole ID", "Distance", "Detection Confidence", "OCR Confidence", "Timestamp"])
            for row in rows:
                writer.writerow(
                    [
                        row["hole_id"],
                        row["distance"],
                        f"{float(row['detection_confidence']):.4f}",
                        f"{float(row['ocr_confidence']):.2f}",
                        row["timestamp"],
                    ]
                )
        return report_path

    @staticmethod
    def _distance_sort_key(distance: str) -> tuple[int, str]:
        digits = "".join(ch for ch in distance if ch.isdigit())
        if not digits:
            return (10**18, distance)
        return (int(digits), distance)

