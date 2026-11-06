"""Rubric validation support: selecting existing candidates, version separation, stop-on-error and the
v1-vs-v2 comparison. Fake provider only."""
import hashlib
import json

import pytest

from promopilot import config
from promopilot.judge import JudgeRun, judge_dir, load_selected_candidates, run_judging
from promopilot.judge_compare import compare, write_comparison
from promopilot.llm import ProviderError
from test_judge import SETTINGS, FakeJudge, candidate, client_for, default_reply, episode, make_run


def digest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_default_rubric_is_v1(monkeypatch):
    monkeypatch.setenv("PROMOPILOT_JUDGE_PROMPT_VERSION", "")
    assert config.judge_settings()["prompt_version"] == "judge-v1"


def test_select_loads_originals_and_stored_repairs(tmp_path):
    ep, n = episode([("cut", "DIALOGUE", 29, 31, "Don't open it.")])
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    v1 = judge_dir(run, "fake", "fake-judge-1")
    got = load_selected_candidates(run, ["EP009_C01_R1", "EP009_C02"], [v1 / "decisions"])
    assert [c.candidate_id for c in got["EP009"]] == ["EP009_C01_R1", "EP009_C02"]
    assert got["EP009"][0].repair.original_candidate_id == "EP009_C01"
    with pytest.raises(ValueError, match="EP009_C07"):
        load_selected_candidates(run, ["EP009_C07"], [v1 / "decisions"])


def judge_selected(run, ep, client, ids, version, **kw):
    v1 = judge_dir(run, "fake", "fake-judge-1")
    by_ep = load_selected_candidates(run, ids, [v1 / "decisions"])
    out = judge_dir(run, client.provider.name, client.provider.model, version) / "rubric_validation"
    return JudgeRun(run, {9: ep}, client, temperature=0, max_tokens=4096, settings=SETTINGS, prompt_version=version,
                    out_dir=out, allow_repair=False, **kw).run_candidates(by_ep), out


def test_v2_selection_keeps_v1_outputs_and_cache_separate(tmp_path):
    ep, n = episode([("cut", "DIALOGUE", 29, 31, "Don't open it.")])
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    v1 = judge_dir(run, "fake", "fake-judge-1")
    before, cache_before = digest(v1), digest(tmp_path / "cache")

    p = FakeJudge()
    summary, out = judge_selected(run, ep, client_for(tmp_path, p), ["EP009_C01_R1", "EP009_C02"], "judge-v2")
    assert len(p.requests) == 2 and all("give 4" in r.messages[0]["content"] for r in p.requests)
    assert [r["candidate_id"] for r in summary["episodes"]["EP009"]] == ["EP009_C01_R1", "EP009_C02"]
    assert digest(v1) == before
    cache_after = digest(tmp_path / "cache")
    assert all(cache_after[k] == v for k, v in cache_before.items()) and len(cache_after) == len(cache_before) + 2
    assert out == run / "judge" / "fake__fake-judge-1__judge-v2" / "rubric_validation"

    cmp = compare(v1, out, ["EP009_C01_R1", "EP009_C02"], {9: ep})
    assert cmp["not_compared"] == [] and all(r["new"]["prompt_version"] == "judge-v2" for r in cmp["rows"])
    assert all(r["base"]["prompt_version"] == "judge-v1" for r in cmp["rows"])


def test_stop_on_provider_error_halts_the_selection(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    p = FakeJudge([ProviderError("HTTP 503", status=503, kind="transient"), default_reply])
    summary, _ = judge_selected(run, ep, client_for(tmp_path, p, max_attempts=1), ["EP009_C01", "EP009_C02"],
                                "judge-v2", stop_on_provider_error=True)
    assert len(p.requests) == 1
    assert [r["verdict"] for r in summary["episodes"]["EP009"]] == ["not_judged", "not_judged"]
    assert "provider error" in summary["stopped_early"]


def test_comparison_reports_deltas_gate_changes_and_checks(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    weaker = lambda t: default_reply(t, cliffhanger__score=2, cliffhanger__reason="A tearful look, no question.",
                                     cliffhanger__outcome="partly_resolved")
    _, out = judge_selected(run, ep, client_for(tmp_path, FakeJudge([weaker])), ["EP009_C01"], "judge-v2")
    v1 = judge_dir(run, "fake", "fake-judge-1")
    cmp = compare(v1, out, ["EP009_C01"], {9: ep})
    row = cmp["rows"][0]
    assert row["delta"]["cliffhanger"] == -2 and row["gate_change"] == "eligible -> quality_fail"
    assert cmp["consistency"]["new"]["resolved_with_passing_cliffhanger"] == 0
    assert row["new"]["evidence"]["cliffhanger"]["n"] >= 1
    path = write_comparison(run, v1, out, ["EP009_C01"], {9: ep}, "judge-v1", "judge-v2")
    md = (path / "report.md").read_text(encoding="utf-8")
    assert "eligible -> quality_fail" in md and "not by itself evidence" in md
