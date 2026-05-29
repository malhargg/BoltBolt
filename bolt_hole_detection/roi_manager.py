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
        if shape == self._last_shape and self._bscan_roi and self._dst_roi:
            return
        self._bscan_roi = self._calculate(self.config.bscan, width, height)
        self._dst_roi = self._calculate(self.config.dst, width, height)
        self._last_shape = shape

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

