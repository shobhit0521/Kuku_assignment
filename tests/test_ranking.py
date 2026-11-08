"""Phase 4D ranking: eligibility, duplicate suppression, diversity caps, sensitivity and reports. No API calls."""
import json

import pytest

from promopilot.ranking import EQUAL_WEIGHTS, build_report, overlap_ratio, rank, sensitivity, write_report
from promopilot.schemas import GateResult, QualityDecision, RuleStatus, ValidationResult


def dec(cid, start, end, hook=3, cliff=3, ctx=3, emo=3, verdict="eligible", arche="danger_threat", hard="pass"):
    ep = cid[:5]
    v = ValidationResult(candidate_id=cid, episode_id=ep, validator_version="t", status=RuleStatus(hard),
                         provisional=False, start_ms=start * 1000, end_ms=end * 1000,
                         duration_ms=(end - start) * 1000)
    scores = {"hook": hook, "narrative_tension": 3, "cliffhanger": cliff, "context_independence": ctx,
              "curiosity_gap": 3, "emotional_impact": emo} if verdict in ("eligible", "quality_fail") else {}
    return QualityDecision(candidate_id=cid, episode_id=ep, verdict=verdict, eligible=verdict == "eligible",
                           dimension_scores=scores, archetype=arche, validation=v, hard_rules=hard,
                           gates=[GateResult(gate="cliffhanger_score", passed=verdict == "eligible", detail="")])


def test_overlap_ratio_uses_shorter_clip():
    assert overlap_ratio((0, 40), (20, 60)) == 0.5
    assert overlap_ratio((0, 60), (10, 30)) == 1.0
    assert overlap_ratio((0, 20), (20, 40)) == 0.0


def test_only_eligible_clips_are_ranked():
    ds = [dec("EP001_C01", 0, 30, hook=5), dec("EP001_C02", 100, 130, verdict="not_judged"),
          dec("EP001_C03", 200, 230, verdict="quality_fail"), dec("EP001_C04", 300, 330, verdict="hard_fail",
                                                                  hard="fail")]
    r = rank(ds)
    assert [x["candidate_id"] for x in r["ordered"]] == ["EP001_C01"]


def test_near_duplicates_over_half_overlap_are_suppressed_within_episode():
    ds = [dec("EP001_C01", 0, 40, hook=4), dec("EP001_C02", 10, 40, hook=3),   # 100% of shorter: suppressed
          dec("EP001_C03", 20, 60, hook=3, cliff=2),                            # exactly 50% of C01: kept
          dec("EP002_C01", 10, 40, hook=3)]                                     # other episode: kept
    r = rank(ds)
    status = {x["candidate_id"]: (x["status"], x["suppressed_by"]) for x in r["ordered"]}
    assert status["EP001_C02"] == ("suppressed_duplicate", "EP001_C01")
    assert status["EP001_C03"][0] != "suppressed_duplicate"
    assert status["EP002_C01"][0] == "selected"


def test_one_clip_per_episode_and_three_per_archetype():
    ds = [dec(f"EP00{i}_C01", 0, 30, hook=5 - i % 2, arche="betrayal_humiliation") for i in range(1, 6)]
    ds += [dec("EP001_C02", 100, 130, arche="reversal"), dec("EP006_C01", 0, 30, hook=1, arche="unclassified"),
           dec("EP007_C01", 0, 30, hook=1, arche="unclassified"), dec("EP008_C01", 0, 30, hook=1, arche="unclassified"),
           dec("EP009_C01", 0, 30, hook=1, arche="unclassified")]
    r = rank(ds)
    sel = [x for x in r["ordered"] if x["status"] == "selected"]
    assert len({x["episode_id"] for x in sel}) == len(sel)
    assert sum(x["archetype"] == "betrayal_humiliation" for x in sel) == 3
    assert sum(x["archetype"] == "unclassified" for x in sel) == 4
    status = {x["candidate_id"]: x["status"] for x in r["ordered"]}
    assert status["EP001_C02"] == "not_selected_episode_cap"
    assert list(status.values()).count("not_selected_archetype_cap") == 2
    assert [x["rank"] for x in r["shortlist"]] == list(range(1, len(r["shortlist"]) + 1))


def test_best_per_episode_ignores_caps_but_not_duplicates():
    ds = [dec("EP001_C01", 0, 30, hook=5), dec("EP001_C02", 5, 30, hook=4), dec("EP001_C03", 100, 130, hook=3)]
    r = rank(ds)
    assert r["best_per_episode"]["EP001"]["candidate_id"] == "EP001_C01"


def test_weights_change_order_and_sensitivity_reports_it():
    a = dec("EP001_C01", 0, 30, hook=5, cliff=4, ctx=5, emo=1)   # approved 4.15, equal 3.75
    b = dec("EP002_C01", 0, 30, hook=4, cliff=4, ctx=4, emo=4)   # approved 4.0, equal 4.0
    c = dec("EP003_C01", 0, 30, hook=3, cliff=3, ctx=3, emo=3)   # 3.0 either way
    primary = rank([a, b, c])
    assert [x["candidate_id"] for x in primary["ordered"]] == ["EP001_C01", "EP002_C01", "EP003_C01"]
    s = sensitivity([a, b, c], primary)
    assert s["shortlist_equal_weight"] == ["EP002_C01", "EP001_C01", "EP003_C01"]
    assert s["spearman_rho"] == 0.5
    assert s["rank_changes"] and s["spearman_rho"] is not None and s["weights"] == EQUAL_WEIGHTS


def write_decisions(jdir, decisions):
    (jdir / "decisions").mkdir(parents=True)
    by_ep = {}
    for d in decisions:
        by_ep.setdefault(d.episode_id, []).append(d.model_dump(mode="json"))
    for ep, rows in by_ep.items():
        (jdir / "decisions" / f"{ep}.json").write_text(json.dumps({"episode_id": ep, "decisions": rows}),
                                                       encoding="utf-8")


def test_report_accounting_and_no_strong_promo(tmp_path):
    run = tmp_path / "run"
    jdir = run / "judge" / "fake__m"
    write_decisions(jdir, [dec("EP001_C01", 0, 30, hook=4), dec("EP001_C02", 50, 80, verdict="not_judged"),
                           dec("EP002_C01", 0, 30, verdict="quality_fail"),
                           dec("EP002_C02", 40, 70, verdict="hard_fail", hard="fail")])
    (run / "manifest.json").write_text(json.dumps({"requested": ["EP001", "EP002", "EP003"],
                                                   "episodes": {"EP003": {"status": "pending"}}}), encoding="utf-8")
    rep = build_report(run, jdir)
    c = rep["counts"]
    assert (c["total_candidates"], c["passing_quality_gates"], c["not_judged_provider_or_budget"],
            c["excluded_technical_failure"], c["quality_fail"]) == (4, 1, 1, 1, 1)
    eps = rep["episodes"]
    assert eps["EP001"]["status"] == "promo found" and eps["EP001"]["incomplete"]
    assert eps["EP002"]["status"] == "no strong promo"
    assert eps["EP003"]["status"] == "no candidates (generation pending)"
    assert rep["baseline"] is None and "not objective" in rep["disclaimer"]


def test_write_report_outputs_and_archives_previous(tmp_path):
    run = tmp_path / "run"
    jdir = run / "judge" / "fake__m"
    write_decisions(jdir, [dec("EP001_C01", 0, 30, hook=4)])
    out = write_report(run, jdir)
    out2 = write_report(run, jdir)
    assert out == out2
    for name in ("ranking.json", "candidates.csv", "shortlist.csv", "report.md"):
        assert (out / name).exists()
    assert len(list((out / "history").glob("report.*.md"))) == 1
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "EP001_C01" in md and "Deferred" in md
    assert (out / "candidates.csv").read_text(encoding="utf-8").splitlines()[0].startswith("candidate_id,")
