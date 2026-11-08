"""UI data layer: loading, status lanes, rubric separation, staleness, data issues, fingerprinting and the
read-only guarantee. Fixture outputs are produced by the real pipeline with a fake judge; no network."""
import hashlib
import json
import os
import shutil

import pytest

import ui_fixtures as fx
from promopilot_ui import data
from promopilot_ui.status import rank_reason, selection_note


def _row(rank, score, cliff=4, hook=4, context=4):
    return {"rank": rank, "score": score,
            "dimension_scores": {"cliffhanger": cliff, "hook": hook, "context_independence": context}}


def test_rank_reason_follows_the_rankings_own_sort_order():
    rows = [_row(1, 5.0), _row(2, 4.6, cliff=5), _row(3, 4.6, cliff=4, hook=5), _row(4, 4.6, cliff=4, hook=5),
            _row(5, 4.6, cliff=4, hook=5), _row(6, 4.2)]
    rows[4]["dimension_scores"]["context_independence"] = 3
    assert rank_reason(rows, 0) == "Weighted score 5.00 of 5, ahead of #2 (4.60)."
    assert rank_reason(rows, 1) == "Weighted score 4.60 of 5, level with #3; placed above it on the cliffhanger " \
                                   "tie-break (5 vs 4)."
    assert rank_reason(rows, 2).endswith("level with #4 on every tie-break score; ordered by candidate ID.")
    assert rank_reason(rows, 3).endswith("context independence tie-break (4 vs 3).")
    assert rank_reason(rows, 5) == "Weighted score 4.20 of 5, behind #5 (4.60); last of the 6 shortlisted clips."
    assert rank_reason(rows[:1], 0) == "Weighted score 5.00 of 5; the only clip on the shortlist."


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("ui"))


@pytest.fixture
def out(built, tmp_path):
    """A private copy for tests that edit files."""
    return shutil.copytree(built, tmp_path / "outputs")


def load(out, judge=None):
    run = next(r for r in data.discover_runs(out) if r.name == fx.RUN)
    return data.load_run(run, judge, out)


def tree_hashes(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


# --- discovery and rubric separation -----------------------------------------------------------------

def test_discovery_pins_production_to_judge_v1_regardless_of_environment(built, monkeypatch):
    monkeypatch.setenv("PROMOPILOT_JUDGE_PROMPT_VERSION", "judge-v2")
    run = next(r for r in data.discover_runs(built) if r.name == fx.RUN)
    assert {j.name: j.production for j in run.judges} == {fx.V1: True, fx.V2: False}
    assert data.production_judge(run).name == fx.V1
    rd = data.load_run(run, root=built)
    assert rd.judge.prompt_version == "judge-v1" and rd.rubric_is_production
    assert rd.judge.checker_version == "c1" and rd.judge.model == "fake-judge-1"


def test_production_scores_come_only_from_v1_judgements(built):
    rd = load(built)
    v2 = json.loads((built / "runs" / fx.RUN / "judge" / fx.V2 / "judgements" / "EP009_C02.json").read_text("utf-8"))
    assert v2["dimensions"]["hook"]["score"] == 2
    view = rd.candidates["EP009_C02"]
    assert view.scores["hook"] == 4 and view.raw.prompt_version.startswith("judge-v1")
    assert all(v.raw is None or v.raw.prompt_version.startswith("judge-v1") for v in rd.candidates.values())


def test_experimental_judge_loads_separately_and_is_marked(built):
    run = next(r for r in data.discover_runs(built) if r.name == fx.RUN)
    rd = data.load_run(run, run.experimental_judges[0], built)
    assert not rd.rubric_is_production and rd.ranking is None
    assert rd.candidates["EP009_C02"].raw.prompt_version.startswith("judge-v2")


def test_judgement_from_another_rubric_in_the_production_folder_is_reported(out):
    jdir = out / "runs" / fx.RUN / "judge"
    shutil.copy(jdir / fx.V2 / "judgements" / "EP009_C02.json", jdir / fx.V1 / "judgements" / "EP009_C02.json")
    rd = load(out)
    assert any("does not match folder rubric" in i.problem for i in rd.issues)


# --- status lanes -------------------------------------------------------------------------------------

def test_every_candidate_state_is_distinguished(built):
    rd = load(built)
    assert {c: v.lane for c, v in rd.candidates.items()} == fx.EXPECTED_LANES
    assert rd.issues == []


def test_scores_are_shown_only_for_successfully_judged_candidates(built):
    rd = load(built)
    for cid in ("EP009_C05", "EP009_C06", "EP009_C07"):
        v = rd.candidates[cid]
        assert not v.judged and v.scores == {} and v.raw_scores == {} and v.weighted_score is None
        assert v.archetype is None
    assert rd.candidates["EP009_C01"].judged  # judged, then failed a hard rule; scores stay visible


def test_not_judged_is_never_a_quality_failure(built):
    rd = load(built)
    v = rd.candidates["EP009_C05"]
    assert v.lane == "not_judged" and v.decision.verdict == "not_judged"
    assert v not in rd.by_lane("quality_fail")


def test_effective_scores_are_production_and_raw_scores_stay_inspectable(built):
    rd = load(built)
    q = rd.candidates["EP009_C04"]
    assert q.raw_scores["cliffhanger"] == 3 and q.scores["cliffhanger"] == 2
    assert q.reassessment_status == "resolved" and q.lane == "quality_fail"
    assert q.scores == {k: d.score for k, d in q.effective.dimensions.items()}


def test_review_flag_coexists_with_eligibility(built):
    rd = load(built)
    v = rd.candidates["EP009_C03"]
    assert v.lane == "eligible" and v.review_flags and v.reassessment_status == "revision_rejected"
    status, accepted, rejected = v.reassessment_outcome()
    assert status == "revision_rejected" and not accepted and rejected[0][0] == "context_independence"
    assert v.scores["context_independence"] == 4


def test_repaired_candidate_provenance(built):
    rd = load(built)
    orig, rep = rd.candidates["EP009_C01"], rd.candidates["EP009_C01_R1"]
    assert orig.lane == "technical_fail" and orig.repaired_to == "EP009_C01_R1"
    assert rep.repaired_from == "EP009_C01" and rep.repair.original_candidate_id == "EP009_C01"
    assert [a.action for a in rep.repair.actions] == ["expand_end_for_dialogue"]
    assert rep.span == (0, 31_000) and rep.repair.original_end_ms == 30_000
    assert list(rd.candidates).index("EP009_C01_R1") == list(rd.candidates).index("EP009_C01") + 1


def test_final_validation_comes_from_the_judge_decision(built):
    rd = load(built)
    v = rd.candidates["EP009_C01_R1"].decision.validation
    assert not v.provisional and v.hook_source == "judge"
    assert all(c.status.value == "pass" for c in v.checks)


def test_ranking_order_and_reasons_for_unselected_eligible_candidates(built):
    rd = load(built)
    assert [v.candidate_id for v in rd.shortlist] == ["EP009_C01_R1"]
    others = {v.candidate_id: v for v in rd.eligible_not_selected}
    assert others["EP009_C02"].ranking_status == "suppressed_duplicate"
    assert "Near-duplicate" in selection_note(others["EP009_C02"].ordered_row, rd.ranking)
    assert "EP009_C01_R1" in selection_note(others["EP009_C02"].ordered_row, rd.ranking)
    note = selection_note(others["EP009_C03"].ordered_row, rd.ranking)
    assert others["EP009_C03"].ranking_status == "not_selected_episode_cap"
    assert "ranked higher (EP009_C01_R1)" in note and "per episode" in note


def test_pending_generation_is_not_no_strong_promo(built):
    rd = load(built)
    assert rd.episodes["EP009"].state == "promo_found"
    pending = rd.episodes["EP010"]
    assert pending.state == "generation_pending" and pending.candidate_ids == []
    assert "quota exhausted" in pending.generation_reason and fx.FAKE_KEY not in pending.generation_reason


def test_redact_removes_key_like_values():
    text = f"HTTP 400 ?key={fx.FAKE_KEY} api_key=sk-{'x' * 20}\nsecond line"
    red = data.redact(text)
    assert fx.FAKE_KEY not in red and "sk-" not in red and "second line" not in red


# --- staleness ----------------------------------------------------------------------------------------

def test_decision_newer_than_ranking_is_stale(out):
    path = out / "runs" / fx.RUN / "judge" / fx.V1 / "decisions" / "EP009.json"
    later = (out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json").stat().st_mtime + 60
    os.utime(path, (later, later))
    rd = load(out)
    assert rd.staleness.status == "stale" and "after the ranking" in rd.staleness.reasons[0]


def test_published_snapshot_ignores_checkout_mtimes_but_not_content(out, monkeypatch):
    monkeypatch.setenv("PROMOPILOT_DASHBOARD_DIR", str(out))
    path = out / "runs" / fx.RUN / "judge" / fx.V1 / "decisions" / "EP009.json"
    later = (out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json").stat().st_mtime + 60
    os.utime(path, (later, later))
    assert load(out).staleness.status == "current"
    rpath = out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json"
    rep = json.loads(rpath.read_text("utf-8"))
    next(r for r in rep["table"] if r["candidate_id"] == "EP009_C04")["verdict"] = "eligible"
    rpath.write_text(json.dumps(rep), encoding="utf-8")
    assert load(out).staleness.status == "stale"


def test_ranking_that_disagrees_with_decisions_is_stale_even_with_newer_mtime(out):
    rpath = out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json"
    rep = json.loads(rpath.read_text("utf-8"))
    next(r for r in rep["table"] if r["candidate_id"] == "EP009_C04")["verdict"] = "eligible"
    rpath.write_text(json.dumps(rep), encoding="utf-8")
    rd = load(out)
    assert rd.staleness.status == "stale" and any("differ" in r for r in rd.staleness.reasons)


def test_missing_ranking_is_reported_not_invented(out):
    (out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json").unlink()
    rd = load(out)
    assert rd.staleness.status == "missing" and rd.ranking is None
    assert rd.shortlist == [] and rd.eligible_not_selected == []
    assert rd.candidates["EP009_C02"].lane == "eligible"


# --- missing, unreadable and schema-invalid files -------------------------------------------------------

def test_unreadable_judgement_is_reported_and_scores_withheld(out):
    (out / "runs" / fx.RUN / "judge" / fx.V1 / "judgements" / "EP009_C02.json").write_text("{broken", "utf-8")
    rd = load(out)
    assert any(i.path.endswith("EP009_C02.json") and "unreadable" in i.problem for i in rd.issues)
    v = rd.candidates["EP009_C02"]
    assert v.decision is not None and not v.judged and v.scores == {}


def test_schema_invalid_decision_is_reported_and_others_still_load(out):
    path = out / "runs" / fx.RUN / "judge" / fx.V1 / "decisions" / "EP009.json"
    raw = json.loads(path.read_text("utf-8"))
    raw["decisions"][2]["gates"] = "not a list"
    path.write_text(json.dumps(raw), encoding="utf-8")
    rd = load(out)
    bad = raw["decisions"][2]["candidate_id"]
    assert any("does not match QualityDecision" in i.problem for i in rd.issues)
    assert rd.candidates[bad].decision is None and rd.candidates[bad].lane == "pending"
    assert rd.candidates["EP009_C04"].lane == "quality_fail"


def test_missing_manifest_and_episode_are_reported(out):
    (out / "runs" / fx.RUN / "manifest.json").unlink()
    (out / "parsed" / "episodes" / "EP009.json").unlink()
    rd = load(out)
    assert any(i.path.endswith("manifest.json") and "missing" in i.problem for i in rd.issues)
    ep, issue = data.load_episode("EP009", out)
    assert ep is None and "missing" in issue.problem


def test_missing_judgement_referenced_by_a_decision_is_reported(out):
    (out / "runs" / fx.RUN / "judge" / fx.V1 / "judgements" / "EP009_C03.json").unlink()
    rd = load(out)
    assert any("referenced by a decision is missing" in i.problem for i in rd.issues)
    assert not rd.candidates["EP009_C03"].judged


def test_no_runs_found_in_empty_output_dir(tmp_path):
    assert data.discover_runs(tmp_path) == []


def test_output_dir_inside_the_project_triggers_a_warning(tmp_path):
    from promopilot import config
    assert data.output_location_warning(config.PROJECT_ROOT / "outputs")
    assert data.output_location_warning(tmp_path) is None


# --- fingerprint / cache invalidation -----------------------------------------------------------------

def test_fingerprint_detects_same_size_edit_with_unchanged_mtime(out):
    run = next(r for r in data.discover_runs(out) if r.name == fx.RUN)
    judge = data.production_judge(run)
    before = data.run_fingerprint(run, judge, out)
    path = out / "runs" / fx.RUN / "judge" / fx.V1 / "decisions" / "EP009.json"
    st = path.stat()
    path.write_text(path.read_text("utf-8").replace("quality_fail", "quality_fai1", 1), encoding="utf-8")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert path.stat().st_size == st.st_size and path.stat().st_mtime_ns == st.st_mtime_ns
    assert data.run_fingerprint(run, judge, out) != before


def test_fingerprint_tracks_added_and_removed_files_but_ignores_history(out):
    run = next(r for r in data.discover_runs(out) if r.name == fx.RUN)
    judge = data.production_judge(run)
    base = data.run_fingerprint(run, judge, out)
    hist = out / "runs" / fx.RUN / "judge" / fx.V1 / "history"
    hist.mkdir(exist_ok=True)
    (hist / "EP009.old.json").write_text("{}", "utf-8")
    assert data.run_fingerprint(run, judge, out) == base
    (out / "runs" / fx.RUN / "judge" / fx.V1 / "reassessments" / "EP009_C04.json").unlink()
    assert data.run_fingerprint(run, judge, out) != base


def test_reload_after_edit_reflects_new_content(out):
    first = load(out)
    path = out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json"
    rep = json.loads(path.read_text("utf-8"))
    rep["disclaimer"] = "edited"
    path.write_text(json.dumps(rep), encoding="utf-8")
    second = load(out)
    assert first.fingerprint != second.fingerprint and second.ranking["disclaimer"] == "edited"


# --- read-only ----------------------------------------------------------------------------------------

def test_loading_never_modifies_source_files(built):
    before = tree_hashes(built)
    for run in data.discover_runs(built):
        rd = data.load_run(run, root=built)
        for j in run.experimental_judges:
            data.load_run(run, j, built)
        for eid in rd.episodes:
            data.load_episode(eid, built)
    assert tree_hashes(built) == before
