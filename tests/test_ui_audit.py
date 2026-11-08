"""Needs-attention and repair classification over fixture outputs written by the real pipeline (fake judge,
no network). Nothing here re-validates, re-judges or re-ranks: every value comes from stored files."""
import json
import shutil

import pytest

import ui_fixtures as fx
from promopilot.schemas import Episode, QualityDecision
from promopilot.validator import run_validation
from promopilot_ui import audit, data


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("ui_audit"))


@pytest.fixture
def out(built, tmp_path):
    return shutil.copytree(built, tmp_path / "outputs")


def load(out):
    run = next(r for r in data.discover_runs(out) if r.name == fx.RUN)
    return data.load_run(run, root=out)


def ids(views):
    return [v.candidate_id for v in views]


DECISIONS = ("runs", fx.RUN, "judge", fx.V1, "decisions", "EP009.json")


def edit_decision(out, cid, **changes):
    path = out.joinpath(*DECISIONS)
    raw = json.loads(path.read_text("utf-8"))
    for d in raw["decisions"]:
        if d["candidate_id"] == cid:
            d.update(changes)
    path.write_text(json.dumps(raw), encoding="utf-8")


# --- categories ---------------------------------------------------------------------------------------

def test_each_candidate_is_in_exactly_one_failure_category(built):
    rd = load(built)
    s = audit.sections(rd)
    assert ids(s["technical"]) == ["EP009_C01", "EP009_C08"]
    assert ids(s["quality"]) == ["EP009_C04"]
    assert ids(s["unusable"]) == ["EP009_C05", "EP009_C06"]
    assert ids(s["pending"]) == ["EP009_C07"]
    failures = [c for k in ("technical", "quality", "unusable", "pending") for c in ids(s[k])]
    assert len(failures) == len(set(failures))
    eligible = {c for c, lane in fx.EXPECTED_LANES.items() if lane == "eligible"}
    assert not eligible & set(failures)


def test_no_usable_judgement_is_never_a_quality_failure(built):
    rd = load(built)
    for cid in ("EP009_C05", "EP009_C06", "EP009_C07"):
        v = rd.candidates[cid]
        assert audit.category(v) in ("unusable", "pending")
        assert v.score_role is None and audit.gate_failures(v) == [] and v.scores == {}
    assert ids(audit.sections(rd)["quality"]) == ["EP009_C04"]


def test_unusable_outputs_explain_cause_and_next_step(built):
    rd = load(built)
    provider = audit.unusable_info(rd.candidates["EP009_C05"], fx.RUN)
    assert provider.kind == "Provider request failed" and provider.retry_on_resume
    assert fx.FAKE_KEY not in provider.detail and "[redacted]" in provider.detail
    malformed = audit.unusable_info(rd.candidates["EP009_C06"], fx.RUN)
    assert malformed.kind == "Malformed judge reply" and not malformed.retry_on_resume
    assert "manual review" in malformed.next_step and "--refresh" in malformed.next_step
    pending = audit.unusable_info(rd.candidates["EP009_C07"], fx.RUN)
    assert pending.kind == "No judge decision recorded" and "judge --run fixture" in pending.next_step
    assert audit.unusable_info(rd.candidates["EP009_C02"], fx.RUN) is None


def test_generation_pending_episode_is_listed_as_pending_work(built):
    rd = load(built)
    assert [e.episode_id for e in audit.pending_episodes(rd)] == ["EP010"]


# --- review flags -------------------------------------------------------------------------------------

def test_review_flag_does_not_override_eligibility(built):
    rd = load(built)
    v = rd.candidates["EP009_C03"]
    assert v.lane == "eligible" and audit.category(v) is None
    assert audit.review_status(v) == "needs_review" and v in audit.sections(rd)["review"]
    assert v.ranking_status == "not_selected_episode_cap"
    [item] = audit.review_items(v)
    assert (item.dimension, item.original, item.revised, item.effective) == ("context_independence", 4, 5, 4)
    assert item.still_flagged and item.outcome.startswith("Revision rejected")


def test_resolved_reassessment_shows_original_and_effective(built):
    rd = load(built)
    v = rd.candidates["EP009_C04"]
    assert audit.review_status(v) == "resolved" and audit.category(v) == "quality"
    [item] = audit.review_items(v)
    assert (item.original, item.revised, item.effective, item.still_flagged) == (3, 2, 2, False)
    assert item.outcome == "Revision accepted"


# --- technical failures -------------------------------------------------------------------------------

def test_rule_failures_report_observed_and_required_values(built):
    rd = load(built)
    [cut] = audit.rule_failures(rd, rd.candidates["EP009_C01"])
    assert cut.rule == "DIALOGUE_BOUNDARY" and cut.observed.startswith("cuts EP009_B") and cut.repairable
    [short] = audit.rule_failures(rd, rd.candidates["EP009_C08"])
    assert (short.rule, short.observed, short.required) == ("DURATION_RANGE", "10.0 s", "20 s to 60 s")
    assert audit.rule_failures(rd, rd.candidates["EP009_C02"]) == []


def test_failure_recorded_before_judging_is_parsed_without_inventing_values(built):
    rd = load(built)
    d = QualityDecision(candidate_id="EP009_C99", episode_id="EP009", verdict="hard_fail", hard_rules="fail",
                        hard_rule_failures=["DURATION_RANGE: duration 75.00s (allowed 20-60s inclusive)"],
                        note="not sent to the judge: objective failure no repair can fix")
    v = data.CandidateView("EP009_C99", "EP009", decision=d)
    [f] = audit.rule_failures(rd, v)
    assert f.rule == "DURATION_RANGE" and f.observed == "duration 75.00s (allowed 20-60s inclusive)"
    state = audit.repair_state(rd, v)
    assert state.key == "not_attempted" and "not sent to the judge" in state.detail
    assert v.score_role is None


def test_thresholds_come_from_stored_validation_settings_when_present(out):
    rd = load(out)
    assert audit.settings_for(rd, "EP009")[1].startswith("current configuration")
    ep = Episode.model_validate_json((out / "parsed" / "episodes" / "EP009.json").read_text("utf-8"))
    run_validation(out / "runs" / fx.RUN, {9: ep}, settings={**fx.SETTINGS, "min_clip_ms": 15_000})
    rd = load(out)
    settings, source = audit.settings_for(rd, "EP009")
    assert source == "stored with the validation output" and settings["min_clip_ms"] == 15_000
    pre = audit.preliminary(rd.candidates["EP009_C01"])
    assert pre.status == "fail" and pre.provisional and pre.failed[0].startswith("DIALOGUE_BOUNDARY")


# --- repairs ------------------------------------------------------------------------------------------

def test_repair_records_link_original_and_repaired(built):
    rd = load(built)
    recs = {r.original.candidate_id: r for r in audit.repair_records(rd)}
    assert set(recs) == {"EP009_C01", "EP009_C08"}
    ok = recs["EP009_C01"]
    assert ok.succeeded and ok.repaired.candidate_id == "EP009_C01_R1" and ok.became_eligible and ok.shortlisted
    assert ok.new_judgement and [t.rule for t in ok.triggers] == ["DIALOGUE_BOUNDARY"]
    assert ok.provenance.original_end_ms == 30_000 and ok.provenance.new_end_ms == 31_000
    assert [a.action for a in ok.provenance.actions] == ["expand_end_for_dialogue"]
    bad = recs["EP009_C08"]
    assert bad.state.key == "unrepairable" and bad.repaired is None and not bad.succeeded
    assert "no conservative boundary change" in bad.state.detail
    assert audit.repair_key(rd, rd.candidates["EP009_C01"]) == "repaired_original"
    assert audit.repair_key(rd, rd.candidates["EP009_C01_R1"]) == "repair_version"
    assert audit.repair_key(rd, rd.candidates["EP009_C08"]) == "repair_unresolved"
    assert audit.repair_key(rd, rd.candidates["EP009_C02"]) == "none"


def test_repair_counts_are_derived_once_per_original(built):
    s = audit.repair_summary(load(built))
    assert (s.attempts, s.succeeded, s.unrepairable, s.not_resolved) == (2, 1, 1, 1)
    assert (s.became_eligible, s.shortlisted, s.technical_without_attempt) == (1, 1, 0)


def test_repair_is_not_called_successful_unless_the_repaired_decision_passes(out):
    edit_decision(out, "EP009_C01_R1", hard_rules="fail", verdict="hard_fail",
                  hard_rule_failures=["SUSPENSE_TAIL_TOLERANCE: clip ends 3.00s after the suspense beat"])
    rd = load(out)
    [rec] = [r for r in audit.repair_records(rd) if r.original.candidate_id == "EP009_C01"]
    assert rec.state.key == "repaired_still_failing" and not rec.succeeded and not rec.became_eligible
    assert "SUSPENSE_TAIL_TOLERANCE" in rec.state.detail
    assert audit.repair_summary(rd).succeeded == 0


def test_repair_without_a_recorded_decision_is_reported_as_such(out):
    path = out.joinpath(*DECISIONS)
    raw = json.loads(path.read_text("utf-8"))
    raw["decisions"] = [d for d in raw["decisions"] if d["candidate_id"] != "EP009_C01_R1"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    rd = load(out)
    state = audit.repair_state(rd, rd.candidates["EP009_C01"])
    assert state.key == "repaired_no_decision" and audit.repair_summary(rd).awaiting_decision == 1


def test_malformed_preliminary_record_is_ignored(built):
    v = data.CandidateView("EP009_C01", "EP009", validation_record={"candidate_id": "EP009_C01", "validation": 5})
    assert audit.preliminary(v) is None


# --- scores and filters -------------------------------------------------------------------------------

def test_score_roles(built):
    rd = load(built)
    roles = {c: v.score_role for c, v in rd.candidates.items()}
    assert roles["EP009_C02"] == roles["EP009_C01_R1"] == "eligible"
    assert roles["EP009_C01"] == roles["EP009_C04"] == roles["EP009_C08"] == "diagnostic"
    assert roles["EP009_C05"] is roles["EP009_C06"] is roles["EP009_C07"] is None
    assert rd.candidates["EP009_C04"].weighted_score == rd.candidates["EP009_C04"].decision.weighted_score


def test_filters(built):
    rd = load(built)
    only = lambda **kw: {k: ids(v) for k, v in audit.sections(rd, audit.Filters(**kw)).items() if v}
    assert only(categories={"technical"}) == {"technical": ["EP009_C01", "EP009_C08"]}
    assert only(repairs={"repair_unresolved"}) == {"technical": ["EP009_C08"]}
    assert only(reviews={"needs_review"}) == {"review": ["EP009_C03"]}
    assert only(focus="EP009_C04") == {"quality": ["EP009_C04"], "review": ["EP009_C04"]}
    assert only(archetypes={"not_classified"}) == {"unusable": ["EP009_C05", "EP009_C06"], "pending": ["EP009_C07"]}
    assert only(episodes={"EP010"}) == {}
