"""Exact conversion between screenplay timestamps and integer milliseconds."""
import re

_TIMESTAMP = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")


class TimestampError(ValueError):
    pass


def parse_timestamp(value: str) -> int:
    """Parse "MM:SS.ss" (or "H:MM:SS.ss", fraction optional, up to 3 digits) into milliseconds."""
    text = value.strip()
    match = _TIMESTAMP.match(text)
    if not match:
        raise TimestampError(f"unrecognised timestamp: {value!r}")
    hours, minutes, seconds, fraction = match.groups()
    minutes_i, seconds_i = int(minutes), int(seconds)
    if seconds_i >= 60 or (hours is not None and minutes_i >= 60):
        raise TimestampError(f"out-of-range timestamp: {value!r}")
    millis = int((fraction or "").ljust(3, "0"))
    return ((int(hours or 0) * 60 + minutes_i) * 60 + seconds_i) * 1000 + millis


def format_timestamp(ms: int) -> str:
    """Format milliseconds as "MM:SS.ss" (3 decimals only if needed; hours prefix only if needed)."""
    if ms < 0:
        raise TimestampError(f"negative time: {ms}")
    total_seconds, millis = divmod(ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    fraction = f"{millis // 10:02d}" if millis % 10 == 0 else f"{millis:03d}"
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction}"
    return f"{minutes:02d}:{seconds:02d}.{fraction}"
