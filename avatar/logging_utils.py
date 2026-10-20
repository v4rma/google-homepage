"""Privacy-preserving diagnostic logging.

Episode content, interview answers and LLM output must never reach the logs. All logging in this
package goes through :func:`log_event`, which accepts only identifiers, numbers, booleans and
short enum-like strings. A filter rejects anything that looks like free text as a second line of
defence.
"""

from __future__ import annotations

import logging
from typing import Any

MAX_VALUE_LEN = 64

_logger = logging.getLogger("avatar")


class _NoFreeTextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if len(msg) > 400 or "\n" in msg:
            record.msg = "log record suppressed (looked like free text)"
            record.args = ()
        return True


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not any(isinstance(f, _NoFreeTextFilter) for f in _logger.filters):
        _logger.addFilter(_NoFreeTextFilter())


def _safe(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) or value is None:
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return "[" + ",".join(_safe(v) for v in list(value)[:10]) + "]"
    text = str(value)
    if len(text) > MAX_VALUE_LEN or "\n" in text or " " in text.strip():
        return f"<redacted:{len(text)}chars>"
    return text


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Log a structured event. Values are coerced to id-like tokens; anything resembling prose is redacted."""
    parts = " ".join(f"{k}={_safe(v)}" for k, v in fields.items())
    _logger.log(level, "%s %s", event, parts)


def log_error(event: str, exc: BaseException, **fields: Any) -> None:
    """Log an exception by type only; exception messages may echo user content."""
    log_event(event, level=logging.ERROR, error_type=type(exc).__name__, **fields)
