from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import ValidationConfig
from models import Detection, ValidatedDetection


@dataclass
class CandidateHistory:
    centroid_x: float
    centroid_y: float
    first_frame: int
    last_frame: int
    support_frames: int = 1
    missed_frames: int = 0
    best_detection: Detection | None = None
    confirmed: bool = False

    def update(self, detection: Detection) -> None:
        alpha = 0.35
        self.centroid_x = (self.centroid_x * (1.0 - alpha)) + (detection.centroid_x * alpha)
        self.centroid_y = (self.centroid_y * (1.0 - alpha)) + (detection.centroid_y * alpha)
        if detection.frame_number > self.last_frame:
            self.support_frames += 1
        self.last_frame = max(self.last_frame, detection.frame_number)
        self.missed_frames = 0
        if self.best_detection is None or detection.confidence > self.best_detection.confidence:
            self.best_detection = detection


class TemporalValidator:
    def __init__(self, config: ValidationConfig) -> None:
        self.config = config
        self._candidates: list[CandidateHistory] = []
        self._confirmed_keys: set[tuple[int, int]] = set()

    def validate(self, detections: list[Detection], frame_number: int) -> list[ValidatedDetection]:
        matched: set[int] = set()
        for detection in detections:
            index = self._find_candidate(detection)
            if index is None:
                candidate = CandidateHistory(
                    centroid_x=detection.centroid_x,
                    centroid_y=detection.centroid_y,
                    first_frame=detection.frame_number,
                    last_frame=detection.frame_number,
                    best_detection=detection,
                )
                self._candidates.append(candidate)
                matched.add(len(self._candidates) - 1)
            else:
                self._candidates[index].update(detection)
                matched.add(index)

        for index, candidate in enumerate(self._candidates):
            if index not in matched and frame_number > candidate.last_frame:
                candidate.missed_frames += 1

        confirmed: list[ValidatedDetection] = []
        for candidate in self._candidates:
            if candidate.support_frames < self.config.validation_frames:
                continue
            if candidate.best_detection is None:
                continue
            key = self._confirmation_key(candidate)
            if candidate.confirmed and key in self._confirmed_keys:
                continue
            candidate.confirmed = True
            self._confirmed_keys.add(key)
            confirmed.append(ValidatedDetection(candidate.best_detection, candidate.support_frames))

        self._cleanup(frame_number)
        return confirmed

    def _find_candidate(self, detection: Detection) -> int | None:
        best_index: int | None = None
        best_distance = float("inf")
        for index, candidate in enumerate(self._candidates):
            if detection.frame_number - candidate.last_frame > self.config.max_missing_frames + 1:
                continue
            distance = math.hypot(detection.centroid_x - candidate.centroid_x, detection.centroid_y - candidate.centroid_y)
            if distance <= self.config.spatial_threshold_pixels and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _confirmation_key(self, candidate: CandidateHistory) -> tuple[int, int]:
        threshold = max(1.0, self.config.spatial_threshold_pixels)
        return int(candidate.centroid_x // threshold), int(candidate.centroid_y // threshold)

    def _cleanup(self, frame_number: int) -> None:
        self._candidates = [
            candidate
            for candidate in self._candidates
            if frame_number - candidate.last_frame <= self.config.stale_after_frames
            or candidate.missed_frames <= self.config.max_missing_frames
        ]

