from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from capture_window import WindowCapture
from config import AppConfig, get_logger
from database import DetectionDatabase
from detector import create_detector
from health_monitor import HealthMonitor
from models import Detection, DetectionRecord, FramePacket, RuntimeMetrics, TrackEvent, ValidatedDetection
from ocr_reader import OCRReader
from report_manager import ReportManager
from roi_manager import ROIManager
from state_machine import StateMachine, SystemState
from tracker import CentroidTracker
from validator import TemporalValidator


@dataclass
class DetectionPacket:
    validated_detections: list[ValidatedDetection]
    bscan_image: np.ndarray
    dst_image: np.ndarray
    frame_number: int
    timestamp: datetime


@dataclass
class OCREventPacket:
    event: TrackEvent
    dst_image: np.ndarray
    frame_number: int
    timestamp: datetime


@dataclass(frozen=True)
class LiveSnapshot:
    annotated_bscan: np.ndarray | None
    frame_number: int
    records: list[dict[str, object]]


class BoltHoleProcessor:
    MOTION_DIFF_THRESHOLD = 10
    MOTION_CHANGED_RATIO_THRESHOLD = 0.003
    MOTION_MEAN_DIFF_THRESHOLD = 0.25
    MOTION_ACTIVE_SECONDS = 2.0
    DETECTION_MAX_WIDTH = 800
    LIVE_HOLD_FRAMES = 20

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("system.processor", config)
        self.state_machine = StateMachine()
        self.state_machine.add_listener(lambda t: self.logger.info("state %s -> %s reason=%s", t.previous.value, t.current.value, t.reason))

        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=config.queues.frame_queue_size)
        self.detection_queue: queue.Queue[DetectionPacket] = queue.Queue(maxsize=config.queues.detection_queue_size)
        self.event_queue: queue.Queue[OCREventPacket] = queue.Queue(maxsize=config.queues.event_queue_size)

        self.database = DetectionDatabase(config.paths.database)
        self.report_manager = ReportManager(self.database, config.paths.reports_dir)
        self.capture = WindowCapture(config, self.frame_queue, self.state_machine)
        self.roi_manager = ROIManager(config.roi)
        self.detector = create_detector(config)
        self.validator = TemporalValidator(config.validation)
        self.tracker = CentroidTracker(config.tracker)
        self.ocr_reader = OCRReader(config)

        self.metrics = RuntimeMetrics()
        self.metrics_lock = threading.RLock()
        self.health_monitor = HealthMonitor(
            config,
            self.metrics,
            self.frame_queue,
            self.detection_queue,
            self.event_queue,
            self.metrics_lock,
        )

        self._stop_event = threading.Event()
        self._processing_thread: threading.Thread | None = None
        self._tracking_thread: threading.Thread | None = None
        self._last_ocr_reject_log_frame = -1
        self._last_motion_frame: np.ndarray | None = None
        self._last_motion_log_frame = -1
        self._motion_active_until_frame = -1
        self._live_lock = threading.RLock()
        self._latest_annotated_bscan: np.ndarray | None = None
        self._latest_frame_number = 0
        self._latest_live_rows: list[dict[str, object]] = []
        self._latest_distance = ""
        self._last_visible_detections: list[ValidatedDetection] = []
        self._last_visible_detection_frame = -1
        self._detected_hole_count = 0
        self._known_hole_rows: dict[str, dict[str, object]] = {}

    def start(self) -> None:
        if self.state_machine.is_running_state():
            return
        self._stop_event.clear()
        self._drain_queues()
        self.state_machine.transition_to(SystemState.WAITING_FOR_WINDOW, "start requested")
        self.capture.start()
        self._processing_thread = threading.Thread(target=self._processing_loop, name="FrameProcessing", daemon=True)
        self._tracking_thread = threading.Thread(target=self._tracking_ocr_loop, name="TrackingOCR", daemon=True)
        self._processing_thread.start()
        self._tracking_thread.start()
        self.health_monitor.start()
        self.logger.info("Bolt-hole processor started")

    def stop(self) -> None:
        self._stop_event.set()
        self.capture.stop()
        for thread in (self._processing_thread, self._tracking_thread):
            if thread:
                thread.join(timeout=5.0)
        self.health_monitor.stop()
        self._drain_queues()
        if self.state_machine.state != SystemState.ERROR:
            try:
                self.state_machine.transition_to(SystemState.STOPPED, "stop requested")
            except ValueError:
                self.state_machine.transition_to(SystemState.ERROR, "stop transition failed")
        self._reset_live_metrics_after_stop()
        self.logger.info("Bolt-hole processor stopped")

    def pause(self) -> None:
        if self.state_machine.state == SystemState.PROCESSING:
            self.state_machine.transition_to(SystemState.PAUSED, "pause requested")

    def resume(self) -> None:
        if self.state_machine.state == SystemState.PAUSED:
            self.state_machine.transition_to(SystemState.PROCESSING, "resume requested")

    def generate_report(self) -> str:
        return str(self.report_manager.generate_csv())

    def snapshot_metrics(self) -> RuntimeMetrics:
        with self.metrics_lock:
            self.metrics.frame_queue_size = self.frame_queue.qsize()
            self.metrics.detection_queue_size = self.detection_queue.qsize()
            self.metrics.event_queue_size = self.event_queue.qsize()
            self.metrics.holes_detected = max(
                self._detected_hole_count,
                self.tracker.total_tracks(),
                self.database.count(),
            )
            if not self.state_machine.is_running_state():
                self.metrics.capture_fps = 0.0
                self.metrics.processing_fps = 0.0
                self.metrics.frame_queue_size = 0
                self.metrics.detection_queue_size = 0
                self.metrics.event_queue_size = 0
            return RuntimeMetrics(**self.metrics.__dict__)

    def snapshot_live_view(self) -> LiveSnapshot:
        with self._live_lock:
            image = None if self._latest_annotated_bscan is None else self._latest_annotated_bscan.copy()
            frame_number = self._latest_frame_number
            live_rows = [dict(row) for row in self._latest_live_rows]
            known_rows = [dict(row) for row in self._known_hole_rows.values()]
        saved_rows = [
            {
                "Bolt Hole": row["hole_id"],
                "Distance": row["distance"],
                "Frame": row["frame_number"],
                "Detection Confidence": round(float(row["detection_confidence"]), 3),
                "OCR Confidence": round(float(row["ocr_confidence"]), 1),
            }
            for row in self.database.fetch_all()
        ]
        records = live_rows if live_rows else known_rows if known_rows else saved_rows
        return LiveSnapshot(image, frame_number, records)

    def _drain_queues(self) -> None:
        for q in (self.frame_queue, self.detection_queue, self.event_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _reset_live_metrics_after_stop(self) -> None:
        with self.metrics_lock:
            self.metrics.capture_fps = 0.0
            self.metrics.processing_fps = 0.0
            self.metrics.frame_queue_size = 0
            self.metrics.detection_queue_size = 0
            self.metrics.event_queue_size = 0

    def _processing_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.state_machine.state == SystemState.PAUSED:
                time.sleep(0.1)
                continue
            try:
                packet = self.frame_queue.get(timeout=self.config.processing.empty_queue_sleep_seconds)
            except queue.Empty:
                continue

            with self.metrics_lock:
                self.metrics.frames_captured = max(self.metrics.frames_captured, packet.frame_number)

            if packet.frame_number % max(1, self.config.processing.process_every_n_frames) != 0:
                continue

            try:
                bscan_image, _ = self.roi_manager.get_bscan_roi(packet.frame)
                dst_image, _ = self.roi_manager.get_dst_roi(packet.frame)
                if self.config.debug.enabled and self.config.debug.save_roi_images:
                    self._save_debug_image("bscan_roi", bscan_image, packet.frame_number)
                    self._save_debug_image("dst_roi", dst_image, packet.frame_number)
                self._should_process_frame(bscan_image, packet.frame_number)
                if self.state_machine.state == SystemState.CAPTURING:
                    self.state_machine.transition_to(SystemState.PROCESSING, "B-scan frame received")
                detection_image, scale = self._resize_for_detection(bscan_image)
                detections = self._scale_detections(
                    self.detector.detect(detection_image, packet.frame_number, packet.timestamp),
                    scale,
                )
                validated = self.validator.validate(detections, packet.frame_number)
                if validated:
                    self._last_visible_detections = validated
                    self._last_visible_detection_frame = packet.frame_number
                elif packet.frame_number - self._last_visible_detection_frame <= self.LIVE_HOLD_FRAMES:
                    validated = self._last_visible_detections
                self._update_live_bscan(bscan_image, validated, packet.frame_number)
                self._put_latest(self.detection_queue, DetectionPacket(validated, bscan_image, dst_image, packet.frame_number, packet.timestamp))
                with self.metrics_lock:
                    self.metrics.frames_processed += 1
                    self.metrics.detections_seen += len(detections)
            except Exception as exc:
                self.logger.exception("Frame processing failed: %s", exc)
                with self.metrics_lock:
                    self.metrics.last_error = str(exc)

    def _tracking_ocr_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                packet = self.detection_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                events = self.tracker.update(packet.validated_detections, packet.frame_number)
                if events:
                    with self.metrics_lock:
                        self._detected_hole_count = max(self._detected_hole_count, self.tracker.total_tracks())
                        self.metrics.holes_detected = self._detected_hole_count
                    with self._live_lock:
                        for event in events:
                            self._known_hole_rows[event.track.hole_id] = {
                                "Bolt Hole": event.track.hole_id,
                                "Distance": self._latest_distance or "reading...",
                                "Frame": packet.frame_number,
                                "Detection Confidence": round(float(event.detection.confidence), 3),
                                "OCR Confidence": "",
                            }
                self._update_live_bscan(packet.bscan_image, packet.validated_detections, packet.frame_number, events)
                for event in events:
                    if event.is_new:
                        self._put_latest(self.event_queue, OCREventPacket(event, packet.dst_image, packet.frame_number, packet.timestamp))
                self._consume_ocr_events()
            except Exception as exc:
                self.logger.exception("Tracking/OCR loop failed: %s", exc)
                with self.metrics_lock:
                    self.metrics.last_error = str(exc)

    def _consume_ocr_events(self) -> None:
        while True:
            try:
                packet = self.event_queue.get_nowait()
            except queue.Empty:
                return
            ocr = self.ocr_reader.read_distance(packet.dst_image, packet.frame_number)
            with self.metrics_lock:
                self.metrics.ocr_events += 1
            if not ocr.accepted:
                if not self.ocr_reader.available:
                    continue
                if packet.frame_number - self._last_ocr_reject_log_frame >= 25:
                    self.logger.warning("hole=%s OCR rejected; database insert skipped", packet.event.track.hole_id)
                    self._last_ocr_reject_log_frame = packet.frame_number
                continue
            with self._live_lock:
                self._latest_distance = ocr.text
                if packet.event.track.hole_id in self._known_hole_rows:
                    self._known_hole_rows[packet.event.track.hole_id]["Distance"] = ocr.text
                    self._known_hole_rows[packet.event.track.hole_id]["OCR Confidence"] = round(float(ocr.confidence), 1)
            record = DetectionRecord(
                hole_id=packet.event.track.hole_id,
                distance=ocr.text,
                frame_number=packet.frame_number,
                detection_confidence=packet.event.detection.confidence,
                ocr_confidence=ocr.confidence,
                timestamp=packet.timestamp,
            )
            inserted = self.database.insert_detection(record)
            with self.metrics_lock:
                self.metrics.holes_detected = self.database.count()
            self.logger.info("hole=%s distance=%s inserted=%s", record.hole_id, record.distance, inserted)

    def _put_latest(self, q: queue.Queue, item: object) -> None:
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(item)

    def _should_process_frame(self, image: np.ndarray, frame_number: int) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        sample = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        previous = self._last_motion_frame
        self._last_motion_frame = sample
        if previous is None:
            return False

        diff = cv2.absdiff(previous, sample)
        changed_ratio = float(np.mean(diff > self.MOTION_DIFF_THRESHOLD))
        mean_diff = float(np.mean(diff))
        has_motion = (
            changed_ratio >= self.MOTION_CHANGED_RATIO_THRESHOLD
            or mean_diff >= self.MOTION_MEAN_DIFF_THRESHOLD
        )
        if has_motion:
            grace_frames = int(round(self.config.capture.capture_fps * self.MOTION_ACTIVE_SECONDS))
            self._motion_active_until_frame = max(self._motion_active_until_frame, frame_number + grace_frames)
            return True
        if frame_number <= self._motion_active_until_frame:
            return True
        if not has_motion and (frame_number == 1 or frame_number - self._last_motion_log_frame >= 25):
            self.logger.info(
                "frame=%s skipped detection because B-scan ROI is static changed_ratio=%.4f mean_diff=%.2f",
                frame_number,
                changed_ratio,
                mean_diff,
            )
            self._last_motion_log_frame = frame_number
        return False

    def _resize_for_detection(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        if width <= self.DETECTION_MAX_WIDTH:
            return image, 1.0
        scale = self.DETECTION_MAX_WIDTH / width
        resized = cv2.resize(
            image,
            (self.DETECTION_MAX_WIDTH, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _scale_detections(self, detections: list[Detection], scale: float) -> list[Detection]:
        if scale == 1.0:
            return detections
        scaled: list[Detection] = []
        for detection in detections:
            x, y, w, h = detection.bbox
            scaled.append(
                Detection(
                    centroid_x=detection.centroid_x / scale,
                    centroid_y=detection.centroid_y / scale,
                    bbox=(
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(w / scale)),
                        int(round(h / scale)),
                    ),
                    area=detection.area / (scale * scale),
                    confidence=detection.confidence,
                    frame_number=detection.frame_number,
                    timestamp=detection.timestamp,
                )
            )
        return scaled

    def _update_live_bscan(
        self,
        bscan_image: np.ndarray,
        validated: list[ValidatedDetection],
        frame_number: int,
        events: list[TrackEvent] | None = None,
    ) -> None:
        annotated = bscan_image.copy()
        labels_by_detection: dict[int, str] = {}
        if events:
            for event in events:
                labels_by_detection[id(event.detection)] = event.track.hole_id
        rows: list[dict[str, object]] = []
        for index, item in enumerate(validated, start=1):
            detection = item.detection
            x, y, w, h = detection.bbox
            pad = 8
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(annotated.shape[1] - 1, x + w + pad)
            y2 = min(annotated.shape[0] - 1, y + h + pad)
            label = labels_by_detection.get(id(detection), f"BH{index}")
            rows.append(
                {
                    "Bolt Hole": label,
                    "Distance": self._latest_distance or "reading...",
                    "Frame": frame_number,
                    "Detection Confidence": round(float(detection.confidence), 3),
                    "OCR Confidence": "",
                }
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{label} {detection.confidence:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        with self._live_lock:
            self._latest_annotated_bscan = annotated
            self._latest_frame_number = frame_number
            if rows or frame_number - self._last_visible_detection_frame > self.LIVE_HOLD_FRAMES:
                self._latest_live_rows = rows

    def _save_debug_image(self, name: str, image: np.ndarray, frame_number: int) -> None:
        path = self.config.paths.debug_dir / f"{frame_number:08d}_{name}.png"
        cv2.imwrite(str(path), image)
