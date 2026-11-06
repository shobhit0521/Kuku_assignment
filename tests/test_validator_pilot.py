"""Validator regression cases from the Flash and Flash-Lite pilots, rebuilt from block IDs only.

Skipped when PROMOPILOT_INPUT_DIR is not set. No screenplay text is stored here: candidates are rebuilt from
the pilot's block IDs and checked against timings measured from the parsed pack.
"""
import json

import pytest

from promopilot import config
from promopilot.schemas import BeatInput, BeatSource, RuleStatus, SemanticInputs
from promopilot.validator import candidate_from_blocks, validate_and_repair, validate_candidate

# run, candidate ID -> (start, end, hook, generator cliffhanger) as recorded in the pilot outputs
PILOT = {
    ("pilot", "EP004_C02"): ("EP004_B086", "EP004_B101", "EP004_B087", "EP004_B099"),
    ("pilot_lite", "EP004_C01"): ("EP004_B004", "EP004_B012", "EP004_B005", "EP004_B012"),
    ("pilot_lite", "EP004_C02"): ("EP004_B086", "EP004_B099", "EP004_B087", "EP004_B099"),
    ("pilot", "EP018_C02"): ("EP018_B071", "EP018_B081", "EP018_B071", "EP018_B080"),
    ("pilot_lite", "EP018_C02"): ("EP018_B070", "EP018_B081", "EP018_B071", "EP018_B081"),
    ("pilot", "EP001_C02"): ("EP001_B063", "EP001_B095", "EP001_B064", "EP001_B092"),
}


@pytest.fixture(scope="module")
def episodes(parsed_pack):
    eps, _ = parsed_pack
    return {e.episode_id: e for e in eps}


def build(episodes, run, cid):
    start, end, hook, cliff = PILOT[(run, cid)]
    ep = episodes[cid[:5]]
    return ep, candidate_from_blocks(ep, cid, start, end, hook, cliff)


def inputs(c, beat_id, source=BeatSource.TEST):
    return SemanticInputs(hook=BeatInput(block_id=c.hook_block_id, source=BeatSource.GENERATOR_ADVISORY),
                          suspense_beat=BeatInput(block_id=beat_id, source=source))


def rule(v, name):
    return next(c for c in v.checks if c.rule == name)


def test_recorded_pilot_ids_match_this_file():
    for (run, cid), ids in PILOT.items():
        path = config.runs_dir() / run / "candidates" / f"{cid[:5]}.json"
        if not path.exists():
            pytest.skip(f"pilot output {path} not available")
        c = next(c for c in json.loads(path.read_text(encoding="utf-8"))["candidates"] if c["candidate_id"] == cid)
        assert (c["start_block_id"], c["end_block_id"], c["hook_block_id"], c["cliffhanger_block_id"]) == ids


def test_flash_ep004_c02_trim_then_extend_start(episodes):
    ep, c = build(episodes, "pilot", "EP004_C02")
    rec = validate_and_repair(ep, c, inputs(c, "EP004_B099"))
    assert rule(rec.validation, "SUSPENSE_TAIL_TOLERANCE").value_ms == 4_300
    assert rec.validation.failed and all(f.repairable for f in rec.validation.failed)
    assert rec.repair.status == "repaired"
    r = rec.repair.repaired_candidate
    assert (r.start_ms, r.end_ms, r.duration_ms) == (113_000, 134_500, 21_500)
    assert (r.start_block_id, r.end_block_id, r.hook_offset_ms) == ("EP004_B084", "EP004_B099", 2_000)
    assert [a.action for a in rec.repair.actions] == ["trim_end_to_suspense_beat", "extend_start_backwards"]
    assert rec.repair.repaired_validation.status == RuleStatus.PASS
    assert (c.start_ms, c.end_ms) == (114_800, 138_800)


def test_lite_ep004_c02_too_short_is_extended_without_a_suspense_beat(episodes):
    ep, c = build(episodes, "pilot_lite", "EP004_C02")
    rec = validate_and_repair(ep, c)
    assert rule(rec.validation, "DURATION_RANGE").value_ms == 19_700
    assert rec.repair.status == "repaired"
    assert (rec.repair.repaired_candidate.start_ms, rec.repair.repaired_candidate.end_ms) == (113_000, 134_500)
    assert rec.repair.repaired_validation.status == RuleStatus.PENDING


def test_lite_ep004_c01_is_unrepairable(episodes):
    ep, c = build(episodes, "pilot_lite", "EP004_C01")
    rec = validate_and_repair(ep, c, inputs(c, "EP004_B012"))
    assert rule(rec.validation, "DURATION_RANGE").value_ms == 11_000
    assert rec.repair.status == "unrepairable" and rec.repair.repaired_candidate is None
    assert "DURATION_RANGE" in rec.repair.reason


@pytest.mark.parametrize("beat_id, tail_ms", [("EP018_B081", 0), ("EP018_B080", 5_500)])
def test_ep018_c02_both_versions_get_the_same_verdict_for_the_same_beat(episodes, beat_id, tail_ms):
    results = []
    for run in ("pilot", "pilot_lite"):
        ep, c = build(episodes, run, "EP018_C02")
        rec = validate_and_repair(ep, c, inputs(c, beat_id))
        tail = rule(rec.validation, "SUSPENSE_TAIL_TOLERANCE")
        results.append((tail.status, tail.value_ms, rule(rec.validation, "SUSPENSE_BEAT_IS_STORY_BEAT").status,
                        rec.repair.status,
                        rec.repair.repaired_validation.status if rec.repair.repaired_validation else None))
    assert results[0] == results[1]
    status, value, story, repair, repaired_status = results[0]
    assert value == tail_ms and story == RuleStatus.PASS
    if tail_ms == 0:
        assert status == RuleStatus.PASS and repair == "not_needed"
    else:
        assert status == RuleStatus.FAIL and repair == "repaired" and repaired_status == RuleStatus.PASS


def test_ep001_cut_to_black_is_a_transition_but_the_push_in_is_a_story_beat(episodes):
    ep, c = build(episodes, "pilot", "EP001_C02")
    v = validate_candidate(ep, c, inputs(c, "EP001_B095"))
    assert rule(v, "SUSPENSE_BEAT_IS_STORY_BEAT").status == RuleStatus.FAIL
    assert v.ending.transition_block_ids == ["EP001_B095"] and "EP001_B094" in v.ending.score_cue_block_ids
    assert rule(v, "ENDING_NOT_TRANSITION_ONLY").status == RuleStatus.PASS
    v2 = validate_candidate(ep, c, inputs(c, "EP001_B092"))
    assert v2.status == RuleStatus.PASS and rule(v2, "SUSPENSE_TAIL_TOLERANCE").value_ms == 1_000
