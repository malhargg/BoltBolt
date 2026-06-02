from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig, DetectorConfig, get_logger
from models import Detection


class BaseDetector(ABC):
    def __init__(self, config: AppConfig) -> None:
        self.app_config = config
        self.config: DetectorConfig = config.detector
        self.logger = get_logger("detector", config)
        self._last_logged_frame = -1

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        kernel_size = self.config.gaussian_kernel if self.config.gaussian_kernel % 2 == 1 else self.config.gaussian_kernel + 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        block_size = self.config.adaptive_block_size
        if block_size % 2 == 0:
            block_size += 1
        threshold = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.config.adaptive_c,
        )
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )
        cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, morph_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, morph_kernel)
        return gray, cleaned

    @abstractmethod
    def detect(self, image: np.ndarray, frame_number: int, timestamp: datetime) -> list[Detection]:
        raise NotImplementedError

    def _score_candidate(self, area: float, aspect_ratio: float, solidity: float) -> float:
        area_span = max(1.0, self.config.max_blob_area - self.config.min_blob_area)
        area_score = 1.0 - abs(area - ((self.config.min_blob_area + self.config.max_blob_area) / 2.0)) / area_span
        aspect_score = 1.0 - min(1.0, abs(1.0 - aspect_ratio))
        solidity_score = min(1.0, max(0.0, solidity))
        return float(max(0.0, min(1.0, (area_score * 0.35) + (aspect_score * 0.30) + (solidity_score * 0.35))))

    def _filter_by_pair_distance(self, detections: list[Detection]) -> list[Detection]:
        if len(detections) < 2:
            return detections
        accepted: list[Detection] = []
        for det in detections:
            has_neighbor = any(
                self.config.pair_distance_min
                <= math.hypot(det.centroid_x - other.centroid_x, det.centroid_y - other.centroid_y)
                <= self.config.pair_distance_max
                for other in detections
                if other is not det
            )
            if has_neighbor:
                accepted.append(det)
        return accepted or detections

    def _save_debug(self, name: str, image: np.ndarray, frame_number: int) -> None:
        if not self.app_config.debug.enabled:
            return
        path = self.app_config.paths.debug_dir / f"{frame_number:08d}_{name}.png"
        cv2.imwrite(str(path), image)

    def _log_count(self, frame_number: int, detector_name: str, count: int) -> None:
        if frame_number == 1 or frame_number - self._last_logged_frame >= 25:
            self.logger.info("frame=%s %s_detections=%s", frame_number, detector_name, count)
            self._last_logged_frame = frame_number


class ContourDetector(BaseDetector):
    def detect(self, image: np.ndarray, frame_number: int, timestamp: datetime) -> list[Detection]:
        _, threshold = self.preprocess(image)
        self._save_debug("threshold", threshold, frame_number)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_blob_area or area > self.config.max_blob_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.config.min_solidity:
                continue
            confidence = self._score_candidate(area, aspect_ratio, solidity)
            if confidence < self.config.min_confidence:
                continue
            detections.append(Detection(x + w / 2.0, y + h / 2.0, (x, y, w, h), area, confidence, frame_number, timestamp))
        detections = self._filter_by_pair_distance(detections)
        self._log_count(frame_number, "contour", len(detections))
        return detections


class BlobDetector(BaseDetector):
    def detect(self, image: np.ndarray, frame_number: int, timestamp: datetime) -> list[Detection]:
        _, threshold = self.preprocess(image)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold, connectivity=8)
        detections: list[Detection] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            area = float(area)
            if area < self.config.min_blob_area or area > self.config.max_blob_area:
                continue
            aspect_ratio = w / max(h, 1)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            component = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_area = max((cv2.contourArea(c) for c in contours), default=area)
            hull_area = max((cv2.contourArea(cv2.convexHull(c)) for c in contours), default=area)
            solidity = float(contour_area / hull_area) if hull_area > 0 else 0.0
            if solidity < self.config.min_solidity:
                continue
            confidence = self._score_candidate(area, aspect_ratio, solidity)
            if confidence < self.config.min_confidence:
                continue
            cx, cy = centroids[label]
            detections.append(Detection(float(cx), float(cy), (int(x), int(y), int(w), int(h)), area, confidence, frame_number, timestamp))
        detections = self._filter_by_pair_distance(detections)
        self._log_count(frame_number, "blob", len(detections))
        return detections


class DashPatternDetector(BaseDetector):
    """Detect bolt-hole echo clusters made of short horizontal dash marks."""

    def detect(self, image: np.ndarray, frame_number: int, timestamp: datetime) -> list[Detection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        band_top, band_bottom = self._find_bscan_band(image)
        dark_mask = cv2.inRange(gray, 0, 185)
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            color_mask = ((saturation > 35) & (value > 45) & (value < 245)).astype(np.uint8) * 255
            mask = cv2.bitwise_or(dark_mask, color_mask)
        else:
            mask = dark_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        self._save_debug("dash_pattern_mask", mask, frame_number)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        strokes: list[tuple[int, int, int, int]] = []
        image_height, image_width = image.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(cv2.contourArea(contour))
            if y < band_top or y > band_bottom:
                continue
            if w < 2 or w > 48 or h < 1 or h > 14:
                continue
            if area > 180:
                continue
            if w / max(h, 1) < 0.8:
                continue
            # Ignore the long blue/red B-scan guide lines and border fragments.
            if w > image_width * 0.06:
                continue
            strokes.append((x, y, w, h))

        clusters = self._cluster_strokes(strokes)
        detections: list[Detection] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            x1 = min(x for x, _, _, _ in cluster)
            y1 = min(y for _, y, _, _ in cluster)
            x2 = max(x + w for x, y, w, h in cluster)
            y2 = max(y + h for x, y, w, h in cluster)
            width = x2 - x1
            height = y2 - y1
            if width < 10 or width > 110 or height > 55:
                continue
            confidence = min(0.95, 0.45 + (0.10 * len(cluster)))
            detections.append(
                Detection(
                    centroid_x=x1 + width / 2.0,
                    centroid_y=y1 + height / 2.0,
                    bbox=(x1, y1, width, height),
                    area=float(width * height),
                    confidence=confidence,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
            )
        self._log_count(frame_number, "dash_pattern", len(detections))
        return detections

    def _find_bscan_band(self, image: np.ndarray) -> tuple[int, int]:
        if image.ndim != 3:
            return 4, image.shape[0] - 4
        b, g, r = cv2.split(image)
        blue_mask = (b > 120) & (g < 170) & (r < 170)
        red_mask = (r > 130) & (g < 150) & (b < 150)
        blue_rows = np.where(np.mean(blue_mask, axis=1) > 0.18)[0]
        red_rows = np.where(np.mean(red_mask, axis=1) > 0.18)[0]
        if len(blue_rows) and len(red_rows):
            for red_y_raw in red_rows:
                blue_candidates = blue_rows[blue_rows < red_y_raw - 10]
                if len(blue_candidates) == 0:
                    continue
                blue_y = int(blue_candidates[-1])
                red_y = int(red_y_raw)
                return min(image.shape[0] - 4, blue_y + 2), max(4, red_y - 2)
        return int(image.shape[0] * 0.30), int(image.shape[0] * 0.86)

    def _cluster_strokes(self, strokes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
        clusters: list[list[tuple[int, int, int, int]]] = []
        for stroke in sorted(strokes, key=lambda item: item[0]):
            sx, sy, sw, sh = stroke
            scx = sx + sw / 2.0
            scy = sy + sh / 2.0
            best_cluster: list[tuple[int, int, int, int]] | None = None
            best_distance = float("inf")
            for cluster in clusters:
                centers = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in cluster]
                cx = sum(x for x, _ in centers) / len(centers)
                cy = sum(y for _, y in centers) / len(centers)
                distance = math.hypot(scx - cx, scy - cy)
                if abs(scx - cx) <= 45 and abs(scy - cy) <= 24 and distance < best_distance:
                    best_cluster = cluster
                    best_distance = distance
            if best_cluster is None:
                clusters.append([stroke])
            else:
                best_cluster.append(stroke)
        return clusters


class HybridDetector(BaseDetector):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._contour = ContourDetector(config)
        self._blob = BlobDetector(config)

    def detect(self, image: np.ndarray, frame_number: int, timestamp: datetime) -> list[Detection]:
        candidates = self._contour.detect(image, frame_number, timestamp) + self._blob.detect(image, frame_number, timestamp)
        merged: list[Detection] = []
        for candidate in sorted(candidates, key=lambda d: d.confidence, reverse=True):
            if any(math.hypot(candidate.centroid_x - existing.centroid_x, candidate.centroid_y - existing.centroid_y) < 8 for existing in merged):
                continue
            merged.append(candidate)
        self._log_count(frame_number, "hybrid", len(merged))
        return merged


def create_detector(config: AppConfig) -> BaseDetector:
    strategy = config.detector.strategy.lower().strip()
    if strategy == "contour":
        return ContourDetector(config)
    if strategy == "blob":
        return BlobDetector(config)
    if strategy == "dash_pattern":
        return DashPatternDetector(config)
    if strategy == "hybrid":
        return HybridDetector(config)
    raise ValueError(f"Unsupported detector strategy: {config.detector.strategy}")
