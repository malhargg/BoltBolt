from __future__ import annotations

import cv2
import numpy as np

from config import ROIConfig, RectPercent
from models import PixelROI


class ROIManager:
    def __init__(self, config: ROIConfig) -> None:
        self.config = config
        self._last_shape: tuple[int, int] | None = None
        self._bscan_roi: PixelROI | None = None
        self._dst_roi: PixelROI | None = None

    def _calculate(self, rect: RectPercent, width: int, height: int) -> PixelROI:
        x = max(0, int(round(width * rect.x)))
        y = max(0, int(round(height * rect.y)))
        roi_width = max(1, int(round(width * rect.width)))
        roi_height = max(1, int(round(height * rect.height)))
        if x + roi_width > width:
            roi_width = width - x
        if y + roi_height > height:
            roi_height = height - y
        if roi_width <= 0 or roi_height <= 0:
            raise ValueError(f"Calculated invalid ROI for frame {width}x{height}: {rect}")
        return PixelROI(x=x, y=y, width=roi_width, height=roi_height)

    def recalculate(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        shape = (height, width)
        self._bscan_roi = self._find_bscan_roi(frame) or self._calculate(self.config.bscan, width, height)
        if shape != self._last_shape or self._dst_roi is None:
            self._dst_roi = self._calculate(self.config.dst, width, height)
        self._last_shape = shape

    def _find_bscan_roi(self, frame: np.ndarray) -> PixelROI | None:
        if frame.ndim != 3:
            return None
        height, width = frame.shape[:2]
        b, g, r = cv2.split(frame)
        blue_mask = (b > 110) & (g < 190) & (r < 190)
        red_mask = (r > 120) & (g < 170) & (b < 170)
        search_right = int(width * 0.42)
        blue_mask[:, search_right:] = False
        red_mask[:, search_right:] = False

        lower_start = int(height * 0.55)
        min_line_pixels = max(80, int(search_right * 0.45))
        blue_counts = np.sum(blue_mask[lower_start:], axis=1)
        red_counts = np.sum(red_mask[lower_start:], axis=1)
        blue_rows = np.where(blue_counts > min_line_pixels)[0] + lower_start
        red_rows = np.where(red_counts > min_line_pixels)[0] + lower_start
        if len(blue_rows) == 0 or len(red_rows) == 0:
            return None

        best: tuple[int, int, int] | None = None
        for blue_y in blue_rows:
            candidates = red_rows[(red_rows > blue_y + 25) & (red_rows < blue_y + int(height * 0.15))]
            if len(candidates) == 0:
                continue
            red_y = int(candidates[0])
            score = int(np.sum(blue_mask[blue_y]) + np.sum(red_mask[red_y]))
            if best is None or score > best[0]:
                best = (score, int(blue_y), red_y)
        if best is None:
            return None

        _, blue_y, red_y = best
        line_mask = blue_mask[blue_y] | red_mask[red_y]
        run = self._longest_true_run(line_mask)
        if run is None:
            return None
        run_start, run_end = run
        if run_end - run_start < min_line_pixels:
            return None

        x1 = max(0, int(run_start) - 8)
        x2 = min(width, int(run_end) + 9)
        y1 = max(0, blue_y - int(height * 0.06))
        y2 = min(height, red_y + int(height * 0.035))
        if x2 - x1 < 120 or y2 - y1 < 50:
            return None
        return PixelROI(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @staticmethod
    def _longest_true_run(values: np.ndarray) -> tuple[int, int] | None:
        best_start = best_end = -1
        start: int | None = None
        for index, value in enumerate(values):
            if bool(value) and start is None:
                start = index
            elif not bool(value) and start is not None:
                if index - start > best_end - best_start:
                    best_start, best_end = start, index
                start = None
        if start is not None and len(values) - start > best_end - best_start:
            best_start, best_end = start, len(values)
        if best_start < 0:
            return None
        return best_start, best_end

    def get_bscan_roi(self, frame: np.ndarray) -> tuple[np.ndarray, PixelROI]:
        self.recalculate(frame)
        assert self._bscan_roi is not None
        return frame[self._bscan_roi.slice()].copy(), self._bscan_roi

    def get_dst_roi(self, frame: np.ndarray) -> tuple[np.ndarray, PixelROI]:
        self.recalculate(frame)
        assert self._dst_roi is not None
        return frame[self._dst_roi.slice()].copy(), self._dst_roi

    @staticmethod
    def save_roi(path: str, image: np.ndarray) -> None:
        cv2.imwrite(path, image)

