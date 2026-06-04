from __future__ import annotations

import re
from datetime import datetime

import cv2
import numpy as np
import pytesseract

from config import AppConfig, get_logger
from models import OCROutput


class OCRReader:
    DISTANCE_PATTERN = re.compile(r"\d{3,5}:\d{3,5}:\d{3,5}")
    DECIMAL_DISTANCE_PATTERN = re.compile(r"\d+(?:\.\d+)?")
    GPS_PATTERN = re.compile(r"\d{1,3}\s*[°º']?\s*\d{1,3}[']?\s*\d{1,3}(?:\.\d+)?[\"']?\s*[NSEW]", re.IGNORECASE)

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger("ocr", config)
        if config.ocr.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = config.ocr.tesseract_cmd
        self._available = self._check_tesseract_available()

    def read_distance(self, image: np.ndarray, frame_number: int) -> OCROutput:
        if not self._available:
            return OCROutput(text="", confidence=0.0, accepted=False)
        best = OCROutput(text="", confidence=0.0, accepted=False)
        variants = self._preprocess_variants(image)
        max_variants = min(len(variants), max(1, self.config.ocr.retry_attempts) * 3)
        for attempt, processed in enumerate(variants[:max_variants]):
            if self.config.debug.enabled and self.config.debug.save_ocr_crops:
                cv2.imwrite(str(self.config.paths.debug_dir / f"{frame_number:08d}_distance_ocr_{attempt}.png"), processed)
            output = self._run_tesseract(
                processed,
                whitelist="0123456789:.",
                pattern=None,
            )
            clean = self._clean_distance_text(output.text)
            result = OCROutput(
                text=clean,
                confidence=output.confidence,
                accepted=bool(clean) and output.confidence >= self.config.ocr.ocr_confidence_threshold,
            )
            if result.confidence > best.confidence:
                best = result
            if result.accepted:
                self.logger.info("frame=%s distance_text=%s confidence=%.2f", frame_number, result.text, result.confidence)
                return result
        self.logger.warning("frame=%s distance OCR rejected best_text=%s confidence=%.2f", frame_number, best.text, best.confidence)
        return best

    def read_location(self, image: np.ndarray, frame_number: int) -> OCROutput:
        if not self._available:
            return OCROutput(text="", confidence=0.0, accepted=False)
        best = OCROutput(text="", confidence=0.0, accepted=False)
        variants = self._preprocess_variants(image)
        max_variants = min(len(variants), max(1, self.config.ocr.retry_attempts) * 3)
        for attempt, processed in enumerate(variants[:max_variants]):
            if self.config.debug.enabled and self.config.debug.save_ocr_crops:
                cv2.imwrite(str(self.config.paths.debug_dir / f"{frame_number:08d}_ocr_{attempt}.png"), processed)
            output = self._run_tesseract(
                processed,
                whitelist="0123456789:",
                pattern=self.DISTANCE_PATTERN,
            )
            if output.confidence > best.confidence:
                best = output
            if output.accepted:
                self.logger.info("frame=%s location_text=%s confidence=%.2f", frame_number, output.text, output.confidence)
                return output
        self.logger.warning("frame=%s location OCR rejected best_text=%s confidence=%.2f", frame_number, best.text, best.confidence)
        return best

    def read_gps_location(self, image: np.ndarray, frame_number: int) -> OCROutput:
        if not self._available:
            return OCROutput(text="", confidence=0.0, accepted=False)
        best = OCROutput(text="", confidence=0.0, accepted=False)
        variants = self._preprocess_variants(image)
        max_variants = min(len(variants), max(1, self.config.ocr.retry_attempts) * 3)
        for attempt, processed in enumerate(variants[:max_variants]):
            if self.config.debug.enabled and self.config.debug.save_ocr_crops:
                cv2.imwrite(str(self.config.paths.debug_dir / f"{frame_number:08d}_gps_ocr_{attempt}.png"), processed)
            output = self._run_tesseract(
                processed,
                whitelist="0123456789NSEWnsew.:'°º",
                pattern=None,
            )
            if output.confidence > best.confidence:
                best = output
            if output.text and output.confidence >= self.config.ocr.ocr_confidence_threshold:
                clean = self._clean_gps_text(output.text)
                accepted = bool(clean)
                result = OCROutput(text=clean, confidence=output.confidence, accepted=accepted)
                if accepted:
                    self.logger.info("frame=%s gps_location_text=%s confidence=%.2f", frame_number, result.text, result.confidence)
                    return result
        self.logger.warning("frame=%s GPS OCR rejected best_text=%s confidence=%.2f", frame_number, best.text, best.confidence)
        return OCROutput(text=self._clean_gps_text(best.text), confidence=best.confidence, accepted=False)

    def _check_tesseract_available(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as exc:
            self.logger.error(
                "Tesseract executable is unavailable; OCR events will be skipped until ocr.tesseract_cmd is configured. error=%s",
                exc,
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    def _preprocess_variants(self, image: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        scale = max(1.0, self.config.ocr.scale_factor)
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
        inverted = cv2.bitwise_not(otsu)
        return [sharpened, enhanced, otsu, adaptive, inverted]

    def _run_tesseract(
        self,
        image: np.ndarray,
        whitelist: str | None = None,
        pattern: re.Pattern[str] | None = None,
    ) -> OCROutput:
        chars = whitelist if whitelist is not None else self.config.ocr.whitelist
        config = f"--oem 3 --psm {self.config.ocr.psm} -c tessedit_char_whitelist={chars}"
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        except Exception as exc:
            self.logger.exception("Tesseract OCR failed: %s", exc)
            return OCROutput(text="", confidence=0.0, accepted=False)

        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = str(text).strip()
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                confidence = -1.0
            if text:
                words.append(text)
            if confidence >= 0:
                confidences.append(confidence)
        joined = "".join(words).replace(" ", "")
        match = pattern.search(joined) if pattern is not None else None
        clean_text = match.group(0) if match else joined
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        accepted = (bool(match) if pattern is not None else bool(clean_text)) and avg_confidence >= self.config.ocr.ocr_confidence_threshold
        return OCROutput(text=clean_text, confidence=avg_confidence, accepted=accepted)

    def _clean_gps_text(self, text: str) -> str:
        clean = " ".join(text.replace("|", " ").split())
        matches = self.GPS_PATTERN.findall(clean)
        return " ".join(matches) if matches else clean

    def _clean_distance_text(self, text: str) -> str:
        clean = text.replace("O", "0").replace("o", "0").replace(",", ".")
        clean = "".join(ch for ch in clean if ch.isdigit() or ch in {":", "."})
        colon_match = self.DISTANCE_PATTERN.search(clean)
        if colon_match:
            return colon_match.group(0)
        decimal_match = self.DECIMAL_DISTANCE_PATTERN.search(clean)
        return decimal_match.group(0) if decimal_match else ""
