import logging
from pathlib import Path


def setup_logger(log_file: Path, level: "int | str" = logging.INFO) -> logging.Logger:
    """Configure a dual (file + console) logger for the pipeline."""

    logger = logging.getLogger("agentcity")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


__all__ = ["setup_logger"]
