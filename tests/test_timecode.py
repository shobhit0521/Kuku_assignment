import pytest

from promopilot.timecode import TimestampError, format_timestamp, parse_timestamp


@pytest.mark.parametrize("raw, ms", [
    ("00:00.00", 0),
    ("00:09.80", 9800),
    ("02:18.00", 138000),
    ("01:59.63", 119630),
    ("3:20.5", 200500),
    ("00:07", 7000),
    ("00:01.234", 1234),
    ("1:02:03.50", 3723500),
    (" 00:11.30 ", 11300),
])
def test_parse_valid(raw, ms):
    assert parse_timestamp(raw) == ms


@pytest.mark.parametrize("raw", ["", "abc", "00:61.00", "1:60:00.00", "00:01.2345", "-00:01.00", "00.01:00", "12"])
def test_parse_invalid(raw):
    with pytest.raises(TimestampError):
        parse_timestamp(raw)


@pytest.mark.parametrize("ms, text", [
    (0, "00:00.00"), (9800, "00:09.80"), (138000, "02:18.00"), (1234, "00:01.234"), (3723500, "1:02:03.50"),
])
def test_format(ms, text):
    assert format_timestamp(ms) == text


def test_round_trip_is_exact_for_all_centiseconds_in_ten_minutes():
    for ms in range(0, 600_000, 10):
        assert parse_timestamp(format_timestamp(ms)) == ms


def test_no_float_drift_in_durations():
    assert parse_timestamp("00:00.30") - parse_timestamp("00:00.10") == 200


def test_negative_format_rejected():
    with pytest.raises(TimestampError):
        format_timestamp(-1)
