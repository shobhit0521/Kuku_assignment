"""Checks against the real input pack. Expected values were measured independently in Phase 0.

Skipped when PROMOPILOT_INPUT_DIR is not set. No screenplay text is stored in this file.
"""
import hashlib
from collections import Counter

from promopilot.parser import discover_episode_files, parse_pack


def by_number(episodes):
    return {e.episode_number: e for e in episodes}


def anomaly_codes(episodes):
    return Counter(a.code for e in episodes for a in e.anomalies)


def test_pack_totals(parsed_pack):
    episodes, summary = parsed_pack
    assert summary.episode_count == 50
    assert [e.episode_number for e in episodes] == list(range(1, 51))
    assert summary.scene_count == 100
    assert summary.block_count == 3727
    assert summary.block_type_counts == {"ACTION": 372, "CAMERA": 1754, "DIALOGUE": 1487, "SOUND_FX": 114}
    assert summary.pack_anomalies == []


def test_no_parse_errors_or_unrecognised_lines(parsed_pack):
    episodes, _ = parsed_pack
    severities = Counter(a.severity.value for e in episodes for a in e.anomalies)
    assert severities.get("error", 0) == 0
    codes = anomaly_codes(episodes)
    for code in ("UNPARSED_LINE", "BLOCK_MISSING_TIME", "INVALID_TIMESTAMP", "BLOCK_NO_TEXT",
                 "MULTILINE_BLOCK_TEXT", "MISSING_HEADER_FIELD", "EPISODE_NUMBER_MISMATCH", "TIME_FORMAT_VARIANT"):
        assert codes.get(code, 0) == 0, code
    assert all(b.has_valid_time for e in episodes for b in e.blocks)


def test_known_source_anomalies_are_preserved(parsed_pack):
    episodes, _ = parsed_pack
    eps = by_number(episodes)
    codes = anomaly_codes(episodes)
    assert codes["HEADER_CASE_VARIANT"] == 1
    assert codes["BLOCK_OVERRUNS_SCENE"] == 1
    assert codes["OVERLAPPING_DIALOGUE"] == 10
    assert codes["OVERLAPPING_CAMERA"] == 5
    assert codes["FILE_ORDER_NOT_TIME_ORDER"] == 8
    assert codes["UNCOVERED_TIME"] == 4

    order_eps = sorted({a.episode_id for e in episodes for a in e.anomalies if a.code == "FILE_ORDER_NOT_TIME_ORDER"})
    assert order_eps == ["EP004", "EP012", "EP020", "EP021", "EP032", "EP046"]

    overrun = [a for a in eps[22].anomalies if a.code == "BLOCK_OVERRUNS_SCENE"][0]
    block = eps[22].get_block(overrun.block_id)
    assert block.block_type.value == "SOUND_FX"
    assert (block.start_raw, block.end_raw) == ("00:41.50", "00:42.80")
    assert eps[22].scenes[0].end_raw == "00:42.50"


def test_episode_6_uppercase_variant(parsed_pack):
    episodes, _ = parsed_pack
    ep6 = by_number(episodes)[6]
    assert [(s.start_raw, s.end_raw) for s in ep6.scenes] == [("00:00.00", "00:33.70"), ("00:33.70", "01:55.00")]
    assert all(s.heading for s in ep6.scenes)
    assert [a.code for a in ep6.anomalies if a.code == "HEADER_CASE_VARIANT"] == ["HEADER_CASE_VARIANT"]
    assert sum(len(s.block_ids) for s in ep6.scenes) == len(ep6.blocks)


def test_durations_and_camera_coverage(parsed_pack):
    episodes, _ = parsed_pack
    durations = {e.episode_number: e.stats.duration_ms for e in episodes}
    assert min(durations.values()) == 70500 and min(durations, key=durations.get) == 44
    assert max(durations.values()) == 200750 and max(durations, key=durations.get) == 36
    eps = by_number(episodes)
    assert eps[18].stats.max_camera_gap_ms == 25500
    assert eps[18].stats.camera_coverage_ratio < 0.6
    assert max(e - s for ep in episodes for s, e in ep.stats.uncovered_intervals_ms) == 2000


def test_golden_block_episode_1(parsed_pack):
    episodes, _ = parsed_pack
    ep1 = by_number(episodes)[1]
    hits = [b for b in ep1.blocks if (b.start_raw, b.end_raw) == ("00:11.30", "00:13.10")]
    assert len(hits) == 1
    b = hits[0]
    assert (b.block_type.value, b.speaker, b.delivery, b.start_ms, b.end_ms) == \
        ("DIALOGUE", "MATRIARCH", ["Cold", "Commanding"], 11300, 13100)
    assert ep1.block_timestamps(b.block_id) == ("00:11.30", "00:13.10")
    assert ep1.stats.duration_ms == 138000


def test_every_block_text_and_time_match_source_lines(pack_dir, parsed_pack):
    episodes, _ = parsed_pack
    files = {p.name: p for p in discover_episode_files(pack_dir)}
    checked = 0
    for ep in episodes:
        lines = files[ep.source_file].read_text(encoding="utf-8").splitlines()
        for b in ep.blocks:
            assert lines[b.line_number - 1].strip() == f"**{b.label_raw}:**"
            assert lines[b.line_number].strip() == f"Start: ({b.start_raw}) | End: ({b.end_raw})"
            assert lines[b.line_number + 1].strip() == b.text
            checked += 1
    assert checked == 3727


def test_block_ids_unique_and_sequential(parsed_pack):
    episodes, _ = parsed_pack
    all_ids = [b.block_id for e in episodes for b in e.blocks]
    assert len(all_ids) == len(set(all_ids))
    for e in episodes:
        assert [b.block_id for b in e.blocks] == [f"{e.episode_id}_B{i:03d}" for i in range(1, len(e.blocks) + 1)]
        assert all(b.scene_id is not None for b in e.blocks)


def test_parse_is_deterministic_and_read_only(pack_dir, parsed_pack):
    files = discover_episode_files(pack_dir)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    episodes_again, summary_again = parse_pack(pack_dir)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert before == after
    episodes, summary = parsed_pack
    assert [e.model_dump_json() for e in episodes] == [e.model_dump_json() for e in episodes_again]
    assert summary.model_dump_json() == summary_again.model_dump_json()
    assert {e.source_file: e.source_sha256 for e in episodes} == before
