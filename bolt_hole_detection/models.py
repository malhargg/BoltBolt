from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PixelROI:
    x: int
    y: int
    width: int
    height: int

    def slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_number: int
    timestamp: datetime
    window_rect: tuple[int, int, int, int]


@dataclass(frozen=True)
class Detection:
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]
    area: float
    confidence: float
    frame_number: int
    timestamp: datetime


@dataclass(frozen=True)
class ValidatedDetection:
    detection: Detection
    support_frames: int


@dataclass
class Track:
    hole_id: str
    centroid_x: float
    centroid_y: float
    last_frame: int
    confidence: float
    active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TrackEvent:
    track: Track
    detection: Detection
    is_new: bool


@dataclass(frozen=True)
class OCROutput:
    text: str
    confidence: float
    accepted: bool


@dataclass(frozen=True)
class DetectionRecord:
    hole_id: str
    distance: str
    location: str
    gps_location: str
    frame_number: int
    detection_confidence: float
    ocr_confidence: float
    timestamp: datetime


@dataclass
class RuntimeMetrics:
    frames_captured: int = 0
    frames_processed: int = 0
    detections_seen: int = 0
    holes_detected: int = 0
    ocr_events: int = 0
    capture_fps: float = 0.0
    processing_fps: float = 0.0
    frame_queue_size: int = 0
    detection_queue_size: int = 0
    event_queue_size: int = 0
    last_error: Optional[str] = None
