from __future__ import annotations

import ctypes
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import mss
import numpy as np

from config import AppConfig, get_logger
from models import FramePacket
from state_machine import StateMachine, SystemState

try:
    import win32con
    import win32gui
    import win32ui
except ImportError:  # pragma: no cover - import is platform dependent.
    win32con = None
    win32gui = None
    win32ui = None


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    rect: tuple[int, int, int, int]


class WindowCapture:
    WINDOW_REFRESH_SECONDS = 1.0

    def __init__(self, config: AppConfig, frame_queue: queue.Queue[FramePacket], state_machine: StateMachine) -> None:
        self.config = config
        self.frame_queue = frame_queue
        self.state_machine = state_machine
        self.logger = get_logger("system.capture", config)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_number = 0
        self._last_window: WindowInfo | None = None
        self._last_window_lookup_at = 0.0
        self._set_dpi_awareness()

    def _set_dpi_awareness(self) -> None:
        if not self.config.capture.dpi_aware:
            return
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as exc:
                self.logger.warning("Unable to set DPI awareness: %s", exc)

    def _find_window(self) -> WindowInfo | None:
        now = time.perf_counter()
        if self._last_window is not None and now - self._last_window_lookup_at < self.WINDOW_REFRESH_SECONDS:
            refreshed = self._refresh_window(self._last_window.hwnd)
            if refreshed is not None:
                return refreshed
        self._last_window_lookup_at = now
        if win32gui is None:
            self.logger.error("pywin32 is required for SRT_BScan window discovery on Windows.")
            return None
        matches: list[WindowInfo] = []

        def enum_handler(hwnd: int, _: object) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if self.config.window_title.lower() not in title.lower():
                return
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left <= 0 or bottom - top <= 0:
                return
            if left < -10000 or top < -10000:
                return
            matches.append(WindowInfo(hwnd=hwnd, title=title, rect=(left, top, right, bottom)))

        win32gui.EnumWindows(enum_handler, None)
        return matches[0] if matches else None

    def _refresh_window(self, hwnd: int) -> WindowInfo | None:
        if win32gui is None:
            return self._last_window
        try:
            if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
                return None
            title = win32gui.GetWindowText(hwnd)
            if self.config.window_title.lower() not in title.lower():
                return None
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            if right - left <= 0 or bottom - top <= 0:
                return None
            if left < -10000 or top < -10000:
                return None
            return WindowInfo(hwnd=hwnd, title=title, rect=(left, top, right, bottom))
        except Exception:
            return None

    def _client_capture_rect(self, window: WindowInfo) -> tuple[int, int, int, int] | None:
        if win32gui is None:
            return window.rect
        try:
            left, top, right, bottom = win32gui.GetClientRect(window.hwnd)
            screen_left, screen_top = win32gui.ClientToScreen(window.hwnd, (left, top))
            screen_right, screen_bottom = win32gui.ClientToScreen(window.hwnd, (right, bottom))
            if screen_right <= screen_left or screen_bottom <= screen_top:
                return None
            return screen_left, screen_top, screen_right, screen_bottom
        except Exception as exc:
            self.logger.warning("Unable to calculate client capture rect: %s", exc)
            return window.rect

    def _capture_with_printwindow(self, window: WindowInfo) -> np.ndarray | None:
        if win32gui is None or win32ui is None:
            return None
        hwnd_dc = None
        mfc_dc = None
        save_dc = None
        bitmap = None
        try:
            _, _, width, height = win32gui.GetClientRect(window.hwnd)
            if width <= 0 or height <= 0:
                return None
            hwnd_dc = win32gui.GetWindowDC(window.hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(bitmap)
            result = ctypes.windll.user32.PrintWindow(window.hwnd, save_dc.GetSafeHdc(), 3)
            if result != 1:
                return None
            bitmap_info = bitmap.GetInfo()
            bitmap_bytes = bitmap.GetBitmapBits(True)
            image = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape(
                (bitmap_info["bmHeight"], bitmap_info["bmWidth"], 4)
            )
            frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            if frame.size == 0 or float(frame.mean()) < 1.0:
                return None
            return self._trim_printwindow_padding(frame)
        except Exception as exc:
            self.logger.warning("PrintWindow capture failed, using region fallback: %s", exc)
            return None
        finally:
            if bitmap is not None:
                win32gui.DeleteObject(bitmap.GetHandle())
            if save_dc is not None:
                save_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            if hwnd_dc is not None:
                win32gui.ReleaseDC(window.hwnd, hwnd_dc)

    @staticmethod
    def _trim_printwindow_padding(frame: np.ndarray) -> np.ndarray:
        """Remove black DPI padding sometimes added by PrintWindow."""
        mask = np.any(frame > 8, axis=2)
        if not np.any(mask):
            return frame
        ys, xs = np.where(mask)
        return frame[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1].copy()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="WindowCapture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _put_latest(self, packet: FramePacket) -> None:
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(packet)

    def _run(self) -> None:
        interval = 1.0 / max(self.config.capture.capture_fps, 0.1)
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                start = time.perf_counter()
                window = self._find_window()
                if window is None:
                    self._last_window = None
                    if self.state_machine.state != SystemState.WAITING_FOR_WINDOW:
                        self.state_machine.transition_to(SystemState.WAITING_FOR_WINDOW, "SRT_BScan window not found")
                    time.sleep(self.config.capture.reconnect_interval_seconds)
                    continue

                if self._last_window is None or self._last_window.rect != window.rect:
                    self.logger.info("Connected to window '%s' rect=%s", window.title, window.rect)
                self._last_window = window
                if self.state_machine.state == SystemState.WAITING_FOR_WINDOW:
                    self.state_machine.transition_to(SystemState.CAPTURING, "SRT_BScan window available")

                capture_rect = self._client_capture_rect(window)
                if capture_rect is None:
                    time.sleep(interval)
                    continue
                left, top, right, bottom = capture_rect
                try:
                    monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
                    if self._is_foreground_window(window):
                        raw = np.asarray(sct.grab(monitor))
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    else:
                        frame = self._capture_with_printwindow(window)
                        if frame is None:
                            self._bring_window_to_front(window)
                            raw = np.asarray(sct.grab(monitor))
                            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                    self._frame_number += 1
                    self._put_latest(FramePacket(frame, self._frame_number, datetime.utcnow(), capture_rect))
                except Exception as exc:
                    self.logger.exception("Window capture failed: %s", exc)
                    self.state_machine.transition_to(SystemState.ERROR, f"capture failed: {exc}")
                    time.sleep(self.config.capture.reconnect_interval_seconds)

                elapsed = time.perf_counter() - start
                time.sleep(max(0.0, interval - elapsed))

    def _bring_window_to_front(self, window: WindowInfo) -> None:
        if win32gui is None:
            return
        try:
            if win32gui.IsIconic(window.hwnd):
                win32gui.ShowWindow(window.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(window.hwnd)
        except Exception as exc:
            self.logger.warning("Unable to focus SRT_BScan before screen fallback: %s", exc)

    def _is_foreground_window(self, window: WindowInfo) -> bool:
        if win32gui is None:
            return True
        try:
            return win32gui.GetForegroundWindow() == window.hwnd
        except Exception:
            return False
