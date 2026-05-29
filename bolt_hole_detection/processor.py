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
from models import DetectionRecord, FramePacket, RuntimeMetrics, TrackEvent, ValidatedDetection
from ocr_reader import OCRReader
from report_manager import ReportManager
from roi_manager import ROIManager
from state_machine import StateMachine, SystemState
from tracker import CentroidTracker
from validator import TemporalValidator


@dataclass
class DetectionPacket:
    validated_detections: list[ValidatedDetection]
    dst_image: np.ndarray
    frame_number: int
    timestamp: datetime


@dataclass
class OCREventPacket:
    event: TrackEvent
    dst_image: np.ndarray
    frame_number: int
    timestamp: datetime


class BoltHoleProcessor:
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
            self.metrics.holes_detected = self.database.count()
            if not self.state_machine.is_running_state():
                self.metrics.capture_fps = 0.0
                self.metrics.processing_fps = 0.0
                self.metrics.frame_queue_size = 0
                self.metrics.detection_queue_size = 0
                self.metrics.event_queue_size = 0
            return RuntimeMetrics(**self.metrics.__dict__)

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
                if self.state_machine.state == SystemState.CAPTURING:
                    self.state_machine.transition_to(SystemState.PROCESSING, "first frame received")
                bscan_image, _ = self.roi_manager.get_bscan_roi(packet.frame)
                dst_image, _ = self.roi_manager.get_dst_roi(packet.frame)
                if self.config.debug.enabled and self.config.debug.save_roi_images:
                    self._save_debug_image("bscan_roi", bscan_image, packet.frame_number)
                    self._save_debug_image("dst_roi", dst_image, packet.frame_number)
                detections = self.detector.detect(bscan_image, packet.frame_number, packet.timestamp)
                validated = self.validator.validate(detections, packet.frame_number)
                self._put_latest(self.detection_queue, DetectionPacket(validated, dst_image, packet.frame_number, packet.timestamp))
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

    def _save_debug_image(self, name: str, image: np.ndarray, frame_number: int) -> None:
        path = self.config.paths.debug_dir / f"{frame_number:08d}_{name}.png"
        cv2.imwrite(str(path), image)
