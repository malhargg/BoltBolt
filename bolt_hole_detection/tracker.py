from __future__ import annotations

import math
import threading
from datetime import datetime

from config import TrackerConfig
from models import Track, TrackEvent, ValidatedDetection


class CentroidTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._next_id = 1
        self.active_tracks: dict[str, Track] = {}
        self.inactive_tracks: dict[str, Track] = {}

    def update(self, detections: list[ValidatedDetection], frame_number: int) -> list[TrackEvent]:
        events: list[TrackEvent] = []
        with self._lock:
            for validated in detections:
                detection = validated.detection
                track = self._match_track(detection.centroid_x, detection.centroid_y)
                is_new = False
                if track is None:
                    track = self._create_track(detection.centroid_x, detection.centroid_y, frame_number, detection.confidence)
                    is_new = True
                else:
                    track.centroid_x = detection.centroid_x
                    track.centroid_y = detection.centroid_y
                    track.last_frame = frame_number
                    track.confidence = max(track.confidence, detection.confidence)
                    track.updated_at = datetime.utcnow()
                    track.active = True
                    self.active_tracks[track.hole_id] = track
                    self.inactive_tracks.pop(track.hole_id, None)
                events.append(TrackEvent(track=track, detection=detection, is_new=is_new))
            self._cleanup(frame_number)
        return events

    def _create_track(self, x: float, y: float, frame_number: int, confidence: float) -> Track:
        hole_id = f"BH{self._next_id}"
        self._next_id += 1
        now = datetime.utcnow()
        track = Track(hole_id, x, y, frame_number, confidence, True, now, now)
        self.active_tracks[hole_id] = track
        return track

    def _match_track(self, x: float, y: float) -> Track | None:
        best_track: Track | None = None
        best_distance = float("inf")
        for track in list(self.active_tracks.values()) + list(self.inactive_tracks.values()):
            distance = math.hypot(x - track.centroid_x, y - track.centroid_y)
            if distance <= self.config.tracker_distance_threshold and distance < best_distance:
                best_track = track
                best_distance = distance
        return best_track

    def _cleanup(self, frame_number: int) -> None:
        for hole_id, track in list(self.active_tracks.items()):
            if frame_number - track.last_frame > self.config.stale_cleanup_frames:
                track.active = False
                self.inactive_tracks[hole_id] = track
                del self.active_tracks[hole_id]
        for hole_id, track in list(self.inactive_tracks.items()):
            if frame_number - track.last_frame > self.config.inactive_retention_frames:
                del self.inactive_tracks[hole_id]

    def total_tracks(self) -> int:
        with self._lock:
            return len(self.active_tracks) + len(self.inactive_tracks)

