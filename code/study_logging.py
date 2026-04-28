"""Shared logging setup: console + per-run timestamped file under ``code/logs/``."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class _FlushingFileHandler(logging.FileHandler):
    """Flush after each record so ``code/logs/*.log`` updates while the process runs."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def configure_study_logging(*, study_name: str, level: int) -> Path:
    """Attach stream + file handlers to the root logger.

    Log path: ``<code>/logs/{study_name}_YYMMDD_HHMMSS.log`` (one file per run).

    Parameters
    ----------
    study_name
        Filename prefix, e.g. ``"study1"`` or ``"study2"``.
    level
        Root logging level (e.g. ``logging.INFO``).
    """
    code_dir = Path(__file__).resolve().parent
    logs_dir = code_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{study_name}_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = _FlushingFileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.getLogger(__name__).info("Writing logs to %s", log_path)
    return log_path
