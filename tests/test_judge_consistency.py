"""Phase 4C.1: reason/score consistency checks, one targeted reassessment, retry limits, resume without
duplicate calls, history instead of overwrite, and the run lock. Fake provider only; no network."""
import json
import subprocess
import sys

import pytest

from promopilot.judge import (RunLocked, consistency_flags, effective_judgement, judge_candidate, reassess,
                              run_judging, run_lock)
from promopilot.llm import DuplicateRequest, LLMRequest, ProviderError
from test_judge import (SETTINGS, FakeJudge, alias, candidate, client_for, default_reply, episode,
                        judgement_from, make_run, response)

OUT = "judge/fake__fake-judge-1"


def codes(j, ep):
    return {f.code for f in consistency_flags(j, ep)}


def judged(ep, n, **override):
    c = candidate(ep, n)
    j, view = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                       **override))
    return c, j, view


# --- consistency flags -----------------------------------------------------------------------------

def test_clean_judgement_has_no_flags():
    ep, n = episode()
    _, j, _ = judged(ep, n)
    assert codes(j, ep) == set()


@pytest.mark.parametrize("reason", ["The clip ends on an emotional button rather than an active plot question.",
                                    "The scene settles into a peaceful moment.",
                                    "The exit concludes the scene."])
def test_cliffhanger_reason_contradicting_score_is_flagged(reason):
    ep, n = episode()
    _, j, _ = judged(ep, n, cliffhanger__score=3, cliffhanger__reason=reason)
    assert "REASON_DESCRIBES_WEAK_ENDING" in codes(j, ep)


def test_same_reason_with_low_score_is_not_flagged():
    ep, n = episode()
    _, j, _ = judged(ep, n, cliffhanger__score=2, cliffhanger__reason="An emotional button with no real question.")
    assert codes(j, ep) == set()


def test_resolved_ending_with_passing_cliffhanger_score_is_flagged():
    ep, n = episode()
    _, j, _ = judged(ep, n, cliffhanger__outcome="resolved", cliffhanger__score=3)
    assert "RESOLVED_OUTCOME_PASSING_SCORE" in codes(j, ep)
    _, j5, _ = judged(ep, n, cliffhanger__outcome="partly_resolved", cliffhanger__score=5)
    assert "PARTLY_RESOLVED_TOP_SCORE" in codes(j5, ep)


def test_missing_open_question_with_passing_score_is_flagged():
    ep, n = episode()
    _, j, _ = judged(ep, n, cliffhanger__open_question="")
    assert "NO_OPEN_QUESTION" in codes(j, ep)


def test_strong_context_dependency_with_high_context_score_is_flagged():
    ep, n = episode()
    two = lambda v: [{"issue": "who is Kai?", "block_ids": [alias(v, n["d1"])]},
                     {"issue": "what lie?", "block_ids": [alias(v, n["d1"])]}]
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    confusions=two(v)))
    assert "CONFUSIONS_WITH_HIGH_SCORE" in codes(j, ep)
    j2, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                     context_independence__score=3,
                                                     promo_dna__context_dependency={"level": "high",
                                                                                    "missing_context": ["x"]}))
    assert "DEPENDENCY_LEVEL_MISMATCH" in codes(j2, ep)
    _, j3, _ = judged(ep, n, context_independence__reason="A viewer cannot understand why she is there.")
    assert "REASON_DESCRIBES_DEPENDENCY" in codes(j3, ep)


def test_positive_context_wording_is_not_flagged():
    ep, n = episode()
    _, j, _ = judged(ep, n, context_independence__reason="The conflict is easy to grasp without prior episodes.")
    assert codes(j, ep) == set()


def test_high_emotion_score_needs_enough_cited_story_beats():
    ep, n = episode()
    _, j, _ = judged(ep, n, emotional_impact__score=4)
    assert "THIN_EVIDENCE_FOR_SCORE" in codes(j, ep)
    c = candidate(ep, n)
    j2, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                     emotional_impact__score=4,
                                                     emotional_impact__evidence=[alias(v, n["d1"]), alias(v, n["x"])]))
    assert "THIN_EVIDENCE_FOR_SCORE" not in codes(j2, ep)


def test_flags_never_change_scores_by_themselves():
    ep, n = episode()
    _, j, _ = judged(ep, n, cliffhanger__outcome="resolved", cliffhanger__score=4)
    before = j.model_dump()
    consistency_flags(j, ep)
    assert j.model_dump() == before


# --- targeted reassessment -------------------------------------------------------------------------

def flagged_reply(text):
    return default_reply(text, cliffhanger__score=3, cliffhanger__reason="An emotional button rather than an active question.")


def cliff_only(score, reason, outcome="unresolved"):
    def reply(text):
        full = json.loads(default_reply(text))
        full["cliffhanger"].update(score=score, reason=reason, outcome=outcome)
        return json.dumps({"cliffhanger": full["cliffhanger"]})
    return reply


def test_reassessment_resolving_contradiction_is_applied_without_touching_raw(tmp_path):
    ep, n = episode()
    c = candidate(ep, n)
    provider = FakeJudge([flagged_reply, cliff_only(2, "An emotional button with no open question.")])
    client = client_for(tmp_path, provider)
    j = judge_candidate(ep, c, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    rec = reassess(ep, c, j, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert rec.status == "resolved" and rec.original_scores == {"cliffhanger": 3}
    assert rec.revised_scores == {"cliffhanger": 2}
    assert j.dimensions["cliffhanger"].score == 3
    eff = effective_judgement(j, rec)
    assert eff.dimensions["cliffhanger"].score == 2 and not eff.dimensions["cliffhanger"].review
    last = provider.requests[-1].messages
    assert last[2]["role"] == "assistant" and "cliffhanger" in last[3]["content"]
    assert len(provider.requests) == 2


def test_unresolved_contradiction_keeps_original_and_marks_review(tmp_path):
    ep, n = episode()
    c = candidate(ep, n)
    still = cliff_only(3, "Still an emotional button rather than an active question.")
    provider = FakeJudge([flagged_reply, still])
    client = client_for(tmp_path, provider)
    j = judge_candidate(ep, c, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    rec = reassess(ep, c, j, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert rec.status == "unresolved" and rec.remaining_flags
    eff = effective_judgement(j, rec)
    cliff = eff.dimensions["cliffhanger"]
    assert cliff.score == 3 and cliff.review and "REASON_DESCRIBES_WEAK_ENDING" in cliff.review_reasons[0]


@pytest.mark.parametrize("reply, status", [("not json", "invalid_reply"),
                                           ('{"something_else": {}}', "invalid_reply"),
                                           (ProviderError("HTTP 503", status=503, kind="transient"), "provider_error")])
def test_reassessment_failure_preserves_original(tmp_path, reply, status):
    ep, n = episode()
    c = candidate(ep, n)
    provider = FakeJudge([flagged_reply, reply])
    client = client_for(tmp_path, provider, max_attempts=1)
    j = judge_candidate(ep, c, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    rec = reassess(ep, c, j, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert rec.status == status
    eff = effective_judgement(j, rec)
    assert eff.dimensions["cliffhanger"].score == 3 and eff.dimensions["cliffhanger"].review


def context_flagged_reply(text):
    full = json.loads(default_reply(text))
    hook = full["hook"]["block_id"]
    full["confusions"] = [{"issue": "who is Kai?", "block_ids": [hook]}, {"issue": "what lie?", "block_ids": [hook]}]
    return json.dumps(full)


def context_revision(score, keep_confusions):
    def reply(text):
        full = json.loads(context_flagged_reply(text))
        full["context_independence"]["score"] = score
        return json.dumps({"context_independence": full["context_independence"], "blind_summary": "x",
                           "confusions": full["confusions"] if keep_confusions else []})
    return reply


@pytest.mark.parametrize("revision, status, final", [
    (context_revision(5, keep_confusions=False), "revision_rejected", 4),   # raised score + dropped confusions
    (context_revision(4, keep_confusions=False), "revision_rejected", 4),   # dropped confusions, same score
    (context_revision(3, keep_confusions=True), "resolved", 3)])            # genuine fix
def test_revision_that_games_the_check_is_rejected(tmp_path, revision, status, final):
    ep, n = episode()
    c = candidate(ep, n)
    provider = FakeJudge([context_flagged_reply, revision])
    client = client_for(tmp_path, provider)
    j = judge_candidate(ep, c, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert "CONFUSIONS_WITH_HIGH_SCORE" in codes(j, ep)
    rec = reassess(ep, c, j, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert rec.status == status
    ctx = effective_judgement(j, rec).dimensions["context_independence"]
    assert ctx.score == final and ctx.review == (status != "resolved")


def test_partly_resolved_applies_only_the_genuine_revision(tmp_path):
    ep, n = episode()
    c = candidate(ep, n)

    def both_flagged(text):
        full = json.loads(context_flagged_reply(text))
        full["cliffhanger"].update(score=3, reason="An emotional button rather than an active question.")
        return json.dumps(full)

    def revision(text):
        full = json.loads(both_flagged(text))
        full["cliffhanger"].update(score=2, reason="A tearful look with no open question.")
        full["context_independence"]["score"] = 5
        return json.dumps({"cliffhanger": full["cliffhanger"], "context_independence": full["context_independence"],
                           "blind_summary": "x", "confusions": []})

    client = client_for(tmp_path, FakeJudge([both_flagged, revision]))
    j = judge_candidate(ep, c, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    rec = reassess(ep, c, j, client, temperature=0, max_tokens=4096, settings=SETTINGS)
    eff = effective_judgement(j, rec)
    assert rec.status == "partly_resolved"
    assert eff.dimensions["cliffhanger"].score == 2 and not eff.dimensions["cliffhanger"].review
    assert eff.dimensions["context_independence"].score == 4 and eff.dimensions["context_independence"].review


def test_at_most_one_reassessment_per_judgement_across_resumes(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    still = cliff_only(3, "Still an emotional button rather than an active question.")
    p1 = FakeJudge([flagged_reply, still])
    s1 = run_judging(run, {9: ep}, client_for(tmp_path, p1), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(p1.requests) == 2 and s1["episodes"]["EP009"][0]["reassessment"] == "unresolved"
    p2 = FakeJudge([])
    s2 = run_judging(run, {9: ep}, client_for(tmp_path, p2), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert p2.requests == [] and s2["resumed_reassessments"] == 1
    decision = json.loads((run / OUT / "decisions" / "EP009.json").read_text(encoding="utf-8"))["decisions"][0]
    assert decision["review_flags"] and decision["dimension_scores"]["cliffhanger"] == 3


def test_reassessment_provider_error_is_retried_once_on_next_invocation(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p1 = FakeJudge([flagged_reply, ProviderError("HTTP 503", status=503, kind="transient")])
    run_judging(run, {9: ep}, client_for(tmp_path, p1, max_attempts=1), temperature=0, max_tokens=4096,
                settings=SETTINGS)
    p2 = FakeJudge([cliff_only(2, "An emotional button with no open question.")])
    s2 = run_judging(run, {9: ep}, client_for(tmp_path, p2), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(p2.requests) == 1 and s2["episodes"]["EP009"][0]["reassessment"] == "resolved"
    assert list((run / OUT / "reassessments" / "history").glob("EP009_C01.*.json"))


def test_no_reassess_flag_marks_review_without_calls(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p = FakeJudge([flagged_reply])
    s = run_judging(run, {9: ep}, client_for(tmp_path, p), temperature=0, max_tokens=4096, settings=SETTINGS,
                    reassess=False)
    assert len(p.requests) == 1 and s["episodes"]["EP009"][0]["reassessment"] == "not_run"
    assert s["episodes"]["EP009"][0]["review_flags"] == 1


# --- retries, resume, duplicates -------------------------------------------------------------------

def test_retry_limit_one_attempt_keeps_not_judged(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p = FakeJudge([ProviderError("HTTP 503", status=503, kind="transient")] * 3)
    s = run_judging(run, {9: ep}, client_for(tmp_path, p, max_attempts=1), temperature=0, max_tokens=4096,
                    settings=SETTINGS)
    assert len(p.requests) == 1 and s["episodes"]["EP009"][0]["verdict"] == "not_judged"
    j = json.loads((run / OUT / "judgements" / "EP009_C01.json").read_text(encoding="utf-8"))
    assert j["failure_kind"] == "provider_error"


def test_invalid_model_output_is_distinguished_from_api_failure(tmp_path):
    ep, n = episode()
    j = judge_candidate(ep, candidate(ep, n), client_for(tmp_path, FakeJudge(["bad", "bad"])), temperature=0,
                        max_tokens=4096, settings=SETTINGS)
    assert j.failure_kind == "invalid_output"


def test_new_judgements_restricted_to_listed_ids(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    p = FakeJudge()
    s = run_judging(run, {9: ep}, client_for(tmp_path, p), temperature=0, max_tokens=4096, settings=SETTINGS,
                    new_judgements={"EP009_C02"})
    assert len(p.requests) == 1 and p.requests[0].messages[1]["content"]
    verdicts = {r["candidate_id"]: r["verdict"] for r in s["episodes"]["EP009"]}
    assert verdicts == {"EP009_C01": "not_judged", "EP009_C02": "eligible"}


def test_retrying_provider_failure_archives_instead_of_overwriting(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge([ProviderError("HTTP 503", status=503,
                                                                            kind="transient")]), max_attempts=1),
                temperature=0, max_tokens=4096, settings=SETTINGS)
    run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    history = list((run / OUT / "judgements" / "history").glob("EP009_C01.*.json"))
    assert len(history) == 1 and json.loads(history[0].read_text(encoding="utf-8"))["failure_kind"] == "provider_error"
    assert json.loads((run / OUT / "judgements" / "EP009_C01.json").read_text(encoding="utf-8"))["status"] == "valid"


def test_resume_makes_no_duplicate_calls_and_reuses_cache(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p1 = FakeJudge()
    run_judging(run, {9: ep}, client_for(tmp_path, p1), temperature=0, max_tokens=4096, settings=SETTINGS)
    p2 = FakeJudge([])
    s2 = run_judging(run, {9: ep}, client_for(tmp_path, p2), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(p1.requests) == 1 and p2.requests == [] and s2["resumed_judgements"] == 1
    log = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({e["invocation_id"] for e in log}) == 1 and all(e["invocation_id"] for e in log)


def test_client_once_per_key_blocks_a_repeat_request(tmp_path):
    p = FakeJudge([ProviderError("HTTP 503", status=503, kind="transient"), "{}"])
    client = client_for(tmp_path, p, max_attempts=1, once_per_key=True)
    req = LLMRequest(messages=[{"role": "user", "content": "x"}, {"role": "user", "content": "y"}], temperature=0,
                     max_tokens=10)
    with pytest.raises(ProviderError):
        client.complete(req, prompt_version="v", stage="judge", item_id="A")
    with pytest.raises(DuplicateRequest):
        client.complete(req, prompt_version="v", stage="judge", item_id="A")
    assert len(p.requests) == 1


def test_concurrent_run_is_refused_while_lock_is_held(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p = FakeJudge()
    with run_lock(run / OUT, "other-invocation"):
        with pytest.raises(RunLocked, match="other-invocation"):
            run_judging(run, {9: ep}, client_for(tmp_path, p), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert p.requests == []
    run_judging(run, {9: ep}, client_for(tmp_path, p), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(p.requests) == 1 and not (run / OUT / "judge.lock").exists()


def test_stale_lock_from_dead_process_is_replaced(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (run / OUT).mkdir(parents=True)
    (run / OUT / "judge.lock").write_text(json.dumps({"pid": dead.pid, "invocation_id": "old"}), encoding="utf-8")
    s = run_judging(run, {9: ep}, client_for(tmp_path, FakeJudge()), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert s["episodes"]["EP009"][0]["verdict"] == "eligible"


def test_prompt_versions_use_separate_output_folders(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    p = FakeJudge()
    run_judging(run, {9: ep}, client_for(tmp_path, p), temperature=0, max_tokens=4096, settings=SETTINGS,
                prompt_version="judge-v2")
    assert (run / "judge" / "fake__fake-judge-1__judge-v2" / "judgements" / "EP009_C01.json").exists()
    assert not (run / OUT).exists()
    assert "If you hesitate between 4 and 5, give 4." in p.requests[0].messages[0]["content"]
