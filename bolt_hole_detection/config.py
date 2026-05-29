from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the project's simple nested YAML without requiring PyYAML."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a mapping.")
        return loaded
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


@dataclass(frozen=True)
class RectPercent:
    x: float
    y: float
    width: float
    height: float

    def validate(self, name: str) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(v < 0 for v in values):
            raise ValueError(f"{name} ROI contains negative values: {self}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{name} ROI width/height must be positive: {self}")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError(f"{name} ROI must remain inside frame: {self}")


@dataclass(frozen=True)
class CaptureConfig:
    capture_fps: float
    reconnect_interval_seconds: float
    dpi_aware: bool


@dataclass(frozen=True)
class ProcessingConfig:
    process_every_n_frames: int
    empty_queue_sleep_seconds: float


@dataclass(frozen=True)
class ROIConfig:
    bscan: RectPercent
    dst: RectPercent


@dataclass(frozen=True)
class DetectorConfig:
    strategy: str
    min_blob_area: int
    max_blob_area: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_solidity: float
    gaussian_kernel: int
    adaptive_block_size: int
    adaptive_c: int
    morphology_kernel: int
    min_confidence: float
    pair_distance_min: float
    pair_distance_max: float


@dataclass(frozen=True)
class ValidationConfig:
    validation_frames: int
    spatial_threshold_pixels: float
    max_missing_frames: int
    stale_after_frames: int


@dataclass(frozen=True)
class TrackerConfig:
    tracker_distance_threshold: float
    stale_cleanup_frames: int
    inactive_retention_frames: int


@dataclass(frozen=True)
class OCRConfig:
    ocr_confidence_threshold: float
    retry_attempts: int
    tesseract_cmd: str
    whitelist: str
    psm: int
    scale_factor: float


@dataclass(frozen=True)
class QueueConfig:
    frame_queue_size: int
    detection_queue_size: int
    event_queue_size: int


@dataclass(frozen=True)
class PathConfig:
    database: Path
    reports_dir: Path
    logs_dir: Path
    debug_dir: Path


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool
    save_roi_images: bool
    save_threshold_images: bool
    save_contour_images: bool
    save_ocr_crops: bool
    max_debug_files: int


@dataclass(frozen=True)
class LoggingConfig:
    max_bytes: int
    backup_count: int
    health_log_interval_seconds: float


@dataclass(frozen=True)
class AppConfig:
    window_title: str
    base_dir: Path
    capture: CaptureConfig
    processing: ProcessingConfig
    roi: ROIConfig
    detector: DetectorConfig
    validation: ValidationConfig
    tracker: TrackerConfig
    ocr: OCRConfig
    queues: QueueConfig
    paths: PathConfig
    debug: DebugConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "AppConfig":
        root_dir = Path(__file__).resolve().parent
        path = Path(config_path) if config_path else root_dir / "config" / "config.yaml"
        raw = _load_yaml(path)
        base_dir = (root_dir / str(raw.get("app", {}).get("base_dir", "."))).resolve()

        def section(name: str) -> dict[str, Any]:
            value = raw.get(name, {})
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section '{name}' must be a mapping.")
            return value

        roi_raw = section("roi")
        bscan = RectPercent(**roi_raw["bscan"])
        dst = RectPercent(**roi_raw["dst"])
        bscan.validate("BScan")
        dst.validate("DST")

        paths_raw = section("paths")
        paths = PathConfig(
            database=(base_dir / str(paths_raw["database"])).resolve(),
            reports_dir=(base_dir / str(paths_raw["reports_dir"])).resolve(),
            logs_dir=(base_dir / str(paths_raw["logs_dir"])).resolve(),
            debug_dir=(base_dir / str(paths_raw["debug_dir"])).resolve(),
        )

        cfg = cls(
            window_title=str(section("app").get("window_title", "SRT_BScan")),
            base_dir=base_dir,
            capture=CaptureConfig(**section("capture")),
            processing=ProcessingConfig(**section("processing")),
            roi=ROIConfig(bscan=bscan, dst=dst),
            detector=DetectorConfig(**section("detector")),
            validation=ValidationConfig(**section("validation")),
            tracker=TrackerConfig(**section("tracker")),
            ocr=OCRConfig(**section("ocr")),
            queues=QueueConfig(**section("queues")),
            paths=paths,
            debug=DebugConfig(**section("debug")),
            logging=LoggingConfig(**section("logging")),
        )
        cfg.ensure_directories()
        return cfg

    def ensure_directories(self) -> None:
        for path in (self.paths.database.parent, self.paths.reports_dir, self.paths.logs_dir, self.paths.debug_dir):
            path.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, config: AppConfig) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    log_name = "system.log"
    if name.lower().startswith("detector"):
        log_name = "detector.log"
    elif name.lower().startswith("ocr"):
        log_name = "ocr.log"

    handler = RotatingFileHandler(
        config.paths.logs_dir / log_name,
        maxBytes=config.logging.max_bytes,
        backupCount=config.logging.backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    return logger
