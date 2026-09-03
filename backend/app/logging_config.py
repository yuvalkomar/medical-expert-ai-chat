import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.app.config import Settings
from backend.app.models import utc_now


class JsonFormatter(logging.Formatter):
    """Compact JSON-lines formatter suitable for concurrent logging handlers."""

    _EXTRA_FIELDS = ("message_id", "question", "response", "error", "retry_count", "worker")

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": utc_now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in self._EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger("medical_chat")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    formatter = JsonFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(settings.log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

