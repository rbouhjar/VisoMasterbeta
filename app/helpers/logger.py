import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_logger = None


def get_logger(name: str = "visomaster") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    log_dir = Path(__file__).resolve().parents[2] / "debug"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "detection_gaps.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = RotatingFileHandler(str(log_path), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    _logger = logger
    return logger


def log_detection_gap(frame_number: int, extra: dict | None = None):
    logger = get_logger()
    if extra is None:
        extra = {}
    # Keep it concise to avoid noise
    parts = [f"frame={frame_number}"]
    if 'fallback' in extra:
        parts.append(f"fallback={extra['fallback']}")
    if 'roi_redetect' in extra:
        parts.append(f"roi_redetect={extra['roi_redetect']}")
    if 'soft_retry' in extra:
        parts.append(f"soft_retry={extra['soft_retry']}")
    if 'reason' in extra:
        parts.append(f"reason={extra['reason']}")
    logger.info("no_face_detected " + " ".join(parts))


def log_swap_gap(frame_number: int, extra: dict | None = None):
    logger = get_logger()
    if extra is None:
        extra = {}
    parts = [f"frame={frame_number}"]
    # Context: how many faces detected/matched/attempted/applied
    for k in ("detected", "matched", "attempted", "applied"):
        if k in extra:
            parts.append(f"{k}={extra[k]}")
    # Detection path flags
    if 'soft_retry' in extra:
        parts.append(f"soft_retry={extra['soft_retry']}")
    if 'fallback' in extra:
        parts.append(f"fallback={extra['fallback']}")
    if 'roi_redetect' in extra:
        parts.append(f"roi_redetect={extra['roi_redetect']}")
    if 'reason' in extra:
        parts.append(f"reason={extra['reason']}")
    logger.info("no_swap_applied " + " ".join(parts))


def log_threshold_tune(frame_number: int, face_id: str, base: float, eff: float, delta: float, streak: int, reason: str):
    logger = get_logger()
    parts = [
        f"frame={frame_number}",
        f"face_id={face_id}",
        f"base={base}",
        f"eff={eff}",
        f"delta={delta}",
        f"streak={streak}",
        f"reason={reason}",
    ]
    logger.info("auto_threshold " + " ".join(parts))
