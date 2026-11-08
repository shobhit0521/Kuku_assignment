"""Blind judge, evidence validation, Promo DNA, gates and the judge-repair loop. Synthetic episodes and a fake
provider only: no network, no API quota."""
import json
import re

import pytest

from promopilot import config
from promopilot.judge import (MASK_TOKEN, build_judgement, decide, judge_candidate, parse_response, render_clip,
                              run_judging, weighted_score)
from promopilot.llm import LLMClient, LLMResponse, ProviderError, QuotaExhausted, ResponseCache
from promopilot.schemas import EpisodeGeneration
from test_validator import SETTINGS, make_episode
from promopilot.validator import candidate_from_blocks

SECRETS = {"hook_type": "GEN_HOOK_TYPE_X", "unresolved_question": "GEN_QUESTION_X",
           "new_viewer_context": "GEN_CONTEXT_X", "why_it_works": "GEN_WHY_X"}


def episode(extra=()):
    ep, n = make_episode([("d1", "DIALOGUE", 0.5, 2.5, "You lied to me."),
                          ("x", "ACTION", 10, 12, "A suspenseful pause before the Cliffhanger."),
                          ("m", "SOUND_FX", 27, 29, "Dramatic musical chord swelling.")] + list(extra), track_end=60)
    ep.logline, ep.continuity_notes = "LOGLINE_SECRET", "CONTINUITY_SECRET"
    return ep, n


def candidate(ep, n, start="c0", end="c28", cid="EP009_C01"):
    c = candidate_from_blocks(ep, cid, n[start], n[end], n["d1"], n[end])
    return c.model_copy(update=SECRETS)


def alias(view, real):
    return next(a for a, r in view.aliases.items() if r == real)


def response(hook, beat, ev, **override):
    data = {
        "blind_summary": "A woman confronts someone who lied; a door stays shut.",
        "confusions": [],
        "hook": {"block_id": hook, "score": 4, "reason": "direct accusation", "evidence": [hook] if hook else []},
        "narrative_tension": {"score": 3, "escalation_beats": [ev], "reason": "builds", "evidence": [ev]},
        "cliffhanger": {"suspense_beat_id": beat, "mechanism": "unanswered_question", "outcome": "unresolved",
                        "open_question": "What is behind the door?", "score": 4, "reason": "open",
                        "evidence": [beat] if beat else []},
        "context_independence": {"score": 4, "reason": "clear roles", "evidence": [ev]},
        "curiosity_gap": {"revealed": ["a lie"], "withheld": ["what the lie was"], "balance": "balanced",
                          "score": 3, "reason": "ok", "evidence": [ev]},
        "emotional_impact": {"emotion": "anger", "score": 3, "reason": "accusation", "evidence": [ev]},
        "promo_dna": {
            "archetype": {"primary": "betrayal_humiliation", "other_label": None, "secondary": "reversal",
                          "confidence": "medium", "evidence": [ev]},
            "hook_mechanism": {"type": "accusation_or_question", "evidence": [ev]},
            "emotional_trigger": {"type": "outrage_injustice", "evidence": [ev]},
            "stakes": {"summary": "trust", "evidence": [ev]},
            "context_dependency": {"level": "low", "missing_context": []},
            "appeal": "a betrayal confronted"},
        "soft_rejections": [{"code": "LOW_TENSION", "severity": "minor", "explanation": "flat middle",
                             "block_ids": [ev]}],
    }
    for path, value in override.items():
        node = data
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        if value is DELETE:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    return data


DELETE = object()


def default_reply(user_text, **override):
    """Pick the first eligible hook and the last eligible final beat from the rendered clip."""
    hooks = re.search(r"ELIGIBLE HOOK BLOCKS[^:]*: (.*)", user_text).group(1).split(", ")
    beats = re.search(r"ELIGIBLE FINAL BEATS[^:]*: (.*)", user_text).group(1).split(", ")
    hook = hooks[0] if hooks != ["none"] else None
    beat = beats[-1] if beats != ["none"] else None
    return json.dumps(response(hook, beat, hook or beat, **override))


class FakeJudge:
    """Scripted fake provider. Each script item is a string, an Exception, or a callable(user_text) -> str."""

    def __init__(self, script=None, name="fake", model="fake-judge-1"):
        self.name, self.model = name, model
        self.script = list(script) if script is not None else None
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        item = default_reply if self.script is None else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item(request.messages[1]["content"]) if callable(item) else item
        return LLMResponse(text=text, provider=self.name, model=self.model, prompt_tokens=100,
                           completion_tokens=50, total_tokens=150, latency_ms=5)


def client_for(tmp_path, provider, **kw):
    return LLMClient(provider, ResponseCache(tmp_path / "cache"), tmp_path / "calls.jsonl", min_interval_s=0,
                     sleep=lambda s: None, jitter=lambda d: 0, **kw)


def judge(tmp_path, ep, c, provider):
    return judge_candidate(ep, c, client_for(tmp_path, provider), temperature=0, max_tokens=4096, settings=SETTINGS)


def judgement_from(ep, c, data):
    view = render_clip(ep, c, SETTINGS)
    resp = parse_response(json.dumps(data(view) if callable(data) else data))
    return build_judgement(ep, c, view, resp, provider="fake", model="m", prompt_sha="0", errors=[], calls=[],
                           raw_text=None), view


# --- blind rendering -------------------------------------------------------------------------------

def test_blind_render_excludes_generator_reasoning_and_choices():
    ep, n = episode()
    view = render_clip(ep, candidate(ep, n), SETTINGS)
    for secret in list(SECRETS.values()) + ["LOGLINE_SECRET", "CONTINUITY_SECRET", "EP009", "C01"]:
        assert secret not in view.text
    assert "hook_block" not in view.text and "cliffhanger_block" not in view.text
    assert not any(real in view.text for real in view.aliases.values())


def test_masking_preserves_block_ids_and_timestamps():
    ep, n = episode()
    view = render_clip(ep, candidate(ep, n), SETTINGS)
    line = next(l for l in view.text.splitlines() if l.startswith(alias(view, n["x"]) + " |"))
    assert line == f"{alias(view, n['x'])} | +00:10.00-+00:12.00 | ACTION |  | A {MASK_TOKEN} pause before the {MASK_TOKEN}."
    assert view.masked_terms == 2
    assert sorted(view.aliases.values()) == sorted(b.block_id for b in ep.blocks_intersecting(0, 30_000))
    assert len(set(view.aliases.values())) == len(view.aliases)


def test_times_are_clip_relative_and_boundary_blocks_flagged():
    ep, n = episode([("long", "DIALOGUE", 9, 11, "Wait.")])
    c = candidate_from_blocks(ep, "EP009_C02", n["c10"], n["c38"], n["c10"], n["c38"])
    view = render_clip(ep, c, SETTINGS)
    rows = [l for l in view.text.splitlines() if l.startswith("K")]
    assert "| -00:01.00-+00:01.00 |" in rows[0] and "began before the clip" in rows[0]
    assert alias(view, n["long"]) not in view.eligible_hooks
    assert alias(view, n["c10"]) in view.eligible_hooks


# --- evidence validation ---------------------------------------------------------------------------

def test_valid_judgement_is_eligible_with_weighted_score(tmp_path):
    ep, n = episode()
    c = candidate(ep, n)
    j = judge(tmp_path, ep, c, FakeJudge())
    d = decide(ep, c, j, SETTINGS)
    assert j.status == "valid" and d.verdict == "eligible" and d.eligible
    assert all(g.passed for g in d.gates) and d.validation.status.value == "pass"
    assert not d.validation.provisional and d.validation.hook_source == "judge"
    assert d.weighted_score == weighted_score(j) == round(0.30 * 4 + 0.25 * 4 + 0.30 * 4 + 0.15 * 3, 2)


def test_hook_outside_first_three_seconds_is_rejected():
    ep, n = episode()
    c = candidate(ep, n)
    j, view = judgement_from(ep, c, lambda v: response(alias(v, n["c6"]), alias(v, n["c28"]), alias(v, n["d1"])))
    assert not j.dimensions["hook"].valid and "hook window" in j.dimensions["hook"].issues[0]
    assert j.status == "invalid" and decide(ep, c, j, SETTINGS).verdict == "judge_invalid"


def test_no_suitable_hook_fails_the_hook_gate():
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(None, alias(v, n["c28"]), alias(v, n["d1"]),
                                                    hook__score=1))
    d = decide(ep, c, j, SETTINGS)
    assert j.status == "valid" and d.verdict == "quality_fail"
    assert {g.gate for g in d.gates if not g.passed} == {"hook_score", "hook_block_in_window"}


@pytest.mark.parametrize("beat", ["c10", "K99"])
def test_suspense_beat_outside_final_window_or_clip_is_rejected(beat):
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n[beat]) if beat in n else beat,
                                                    alias(v, n["d1"])))
    assert not j.dimensions["cliffhanger"].valid and j.status == "invalid"


def test_music_cue_cannot_be_the_suspense_beat():
    ep, n = episode()
    c = candidate(ep, n)
    view = render_clip(ep, c, SETTINGS)
    assert alias(view, n["m"]) not in view.eligible_beats
    assert alias(view, n["c28"]) in view.eligible_beats


@pytest.mark.parametrize("bad", ["K99", "EP009_B001"])
def test_evidence_ids_must_belong_to_the_clip(bad):
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    context_independence__evidence=[bad]))
    assert not j.dimensions["context_independence"].valid and j.status == "invalid"


def test_score_without_evidence_and_bad_confusion_ids_invalidate():
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    context_independence__evidence=[],
                                                    confusions=[{"issue": "who is Kai?", "block_ids": ["K99"]}]))
    issues = j.dimensions["context_independence"].issues
    assert any("without valid evidence" in i for i in issues) and any("confusion" in i for i in issues)


def test_emotion_backed_only_by_music_cue_is_invalid():
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    emotional_impact__evidence=[alias(v, n["m"])]))
    assert not j.dimensions["emotional_impact"].valid and j.status == "invalid"
    assert weighted_score(j) is None


def test_resolved_outcome_fails_gate():
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    cliffhanger__outcome="resolved", cliffhanger__score=1))
    d = decide(ep, c, j, SETTINGS)
    assert d.verdict == "quality_fail"
    assert {g.gate for g in d.gates if not g.passed} == {"cliffhanger_score", "outcome_not_resolved"}


def test_resolved_outcome_with_high_score_is_flagged_not_invalidated():
    from promopilot.judge import consistency_flags
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    cliffhanger__outcome="resolved"))
    assert j.dimensions["cliffhanger"].valid
    assert "RESOLVED_OUTCOME_PASSING_SCORE" in {f.code for f in consistency_flags(j, ep)}
    assert decide(ep, c, j, SETTINGS).verdict == "quality_fail"


# --- format and schema -----------------------------------------------------------------------------

def test_invalid_json_gets_exactly_one_repair_call(tmp_path):
    ep, n = episode()
    provider = FakeJudge(["not json at all", default_reply])
    j = judge(tmp_path, ep, candidate(ep, n), provider)
    assert j.status == "valid" and len(provider.requests) == 2
    assert [c["stage"] for c in j.calls] == ["judge", "judge_repair"]
    assert "not valid JSON" in j.response_errors[0]


def test_second_bad_reply_marks_judgement_invalid(tmp_path):
    ep, n = episode()
    provider = FakeJudge(["nope", "still nope"])
    j = judge(tmp_path, ep, candidate(ep, n), provider)
    assert j.status == "invalid" and len(provider.requests) == 2 and j.dimensions == {}


@pytest.mark.parametrize("field", ["cliffhanger__score", "hook__score", "context_independence", "promo_dna"])
def test_missing_required_fields_invalidate(tmp_path, field):
    ep, n = episode()
    bad = lambda text: default_reply(text, **{field: DELETE})
    provider = FakeJudge([bad, bad])
    j = judge(tmp_path, ep, candidate(ep, n), provider)
    assert j.status == "invalid" and "schema mismatch" in j.response_errors[-1]
    assert decide(ep, candidate(ep, n), j, SETTINGS).verdict == "judge_invalid"


def test_score_out_of_range_is_a_schema_error():
    with pytest.raises(Exception, match="schema mismatch"):
        parse_response(json.dumps(response("K01", "K02", "K01", hook__score=7)))


# --- hard rules are authoritative ------------------------------------------------------------------

def test_judge_cannot_override_hard_rule_failure(tmp_path):
    ep, n = episode()
    c = candidate(ep, n, end="c8")  # 10 s clip: too short, cannot extend before 0
    perfect = lambda text: default_reply(text, hook__score=5, cliffhanger__score=5, context_independence__score=5,
                                         emotional_impact__score=5)
    j = judge(tmp_path, ep, c, FakeJudge([perfect]))
    d = decide(ep, c, j, SETTINGS)
    assert j.status == "valid" and d.verdict == "hard_fail" and not d.eligible
    assert any(f.startswith("DURATION_RANGE") for f in d.hard_rule_failures)


def test_repaired_candidate_is_judged_from_scratch(tmp_path):
    ep, n = episode([("cut", "DIALOGUE", 29, 31, "Don't open it.")])
    c = candidate(ep, n)
    run = tmp_path / "run"
    gen = EpisodeGeneration(episode_id="EP009", status="complete", provider="g", model="g", prompt_version="v",
                            prompt_sha256="0", candidates=[c])
    (run / "candidates").mkdir(parents=True)
    (run / "candidates" / "EP009.json").write_text(gen.model_dump_json(), encoding="utf-8")
    provider = FakeJudge()
    summary = run_judging(run, {9: ep}, client_for(tmp_path, provider), temperature=0, max_tokens=4096,
                          settings=SETTINGS)
    rows = summary["episodes"]["EP009"]
    assert [r["candidate_id"] for r in rows] == ["EP009_C01", "EP009_C01_R1"]
    assert rows[0]["verdict"] == "hard_fail" and rows[0]["repair"] == "repaired"
    assert len(provider.requests) == 2
    first, second = (r.messages[1]["content"] for r in provider.requests)
    assert first != second and "C01" not in second and "accusation" not in second
    out = run / "judge" / "fake__fake-judge-1"
    j1 = json.loads((out / "judgements" / "EP009_C01.json").read_text(encoding="utf-8"))
    j2 = json.loads((out / "judgements" / "EP009_C01_R1.json").read_text(encoding="utf-8"))
    assert j1["clip_sha256"] != j2["clip_sha256"] and j2["clip_end_ms"] == 31_000
    decisions = json.loads((out / "decisions" / "EP009.json").read_text(encoding="utf-8"))["decisions"]
    assert decisions[1]["repaired_from"] == "EP009_C01" and decisions[1]["verdict"] == "eligible"
    assert decisions[1]["validation"]["hook_source"] == "judge"


# --- Promo DNA and soft rejections -----------------------------------------------------------------

@pytest.mark.parametrize("primary, label, evidence, expected", [
    ("other", "found family", True, "other"),
    ("other", None, True, "unclassified"),
    ("space_opera", None, True, "unclassified"),
    ("unclassified", None, False, "unclassified"),
    ("danger_threat", None, False, "unclassified"),
    ("danger_threat", None, True, "danger_threat")])
def test_dna_supports_other_and_unclassified(primary, label, evidence, expected):
    ep, n = episode()
    c = candidate(ep, n)

    def data(v):
        d = response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]))
        d["promo_dna"]["archetype"].update(primary=primary, other_label=label,
                                           evidence=[alias(v, n["d1"])] if evidence else [])
        return d

    j, _ = judgement_from(ep, c, data)
    assert j.promo_dna.archetype.value == expected
    assert j.promo_dna.archetype_other_label == (label if expected == "other" else None)
    assert j.status == "valid"
    assert j.promo_dna.cliffhanger_mechanism.value == "unanswered_question"
    assert j.promo_dna.cliffhanger_mechanism.evidence == [n["c28"]]


def test_dna_attribute_with_bad_evidence_is_unclassified():
    ep, n = episode()
    c = candidate(ep, n)
    j, _ = judgement_from(ep, c, lambda v: response(alias(v, n["d1"]), alias(v, n["c28"]), alias(v, n["d1"]),
                                                    promo_dna__hook_mechanism={"type": "threat", "evidence": ["K99"]}))
    assert j.promo_dna.hook_mechanism.value == "unclassified"


def test_every_kept_soft_rejection_has_valid_evidence():
    ep, n = episode()
    c = candidate(ep, n)

    def data(v):
        ok = alias(v, n["d1"])
        return response(ok, alias(v, n["c28"]), ok, soft_rejections=[
            {"code": "WEAK_HOOK", "severity": "minor", "explanation": "soft", "block_ids": [ok]},
            {"code": "TOO_VAGUE", "severity": "major", "explanation": "vague", "block_ids": []},
            {"code": "REVEALS_TOO_MUCH", "severity": "major", "explanation": "x", "block_ids": ["K99"]},
            {"code": "DURATION_RANGE", "severity": "major", "explanation": "x", "block_ids": [ok]},
            {"code": "LOW_TENSION", "severity": "critical", "explanation": "x", "block_ids": [ok]}])

    j, view = judgement_from(ep, c, data)
    assert [r.code for r in j.soft_rejections] == ["WEAK_HOOK"]
    assert all(r.block_ids and set(r.block_ids) <= set(view.aliases.values()) for r in j.soft_rejections)
    assert {d["problem"] for d in j.dropped_rejections} == {"no block IDs", "block IDs not in the clip",
                                                            "unknown code", "unknown severity"}


# --- cache, resume, quota --------------------------------------------------------------------------

def make_run(tmp_path, ep, cands):
    run = tmp_path / "run"
    (run / "candidates").mkdir(parents=True, exist_ok=True)
    gen = EpisodeGeneration(episode_id="EP009", status="complete", provider="g", model="g", prompt_version="v",
                            prompt_sha256="0", candidates=cands)
    (run / "candidates" / "EP009.json").write_text(gen.model_dump_json(), encoding="utf-8")
    return run


def test_resume_skips_existing_judgements_and_cache_serves_deleted_ones(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    p1 = FakeJudge()
    s1 = run_judging(run, {9: ep}, client_for(tmp_path, p1), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(p1.requests) == 2 and s1["requests_made_this_invocation"] == 2

    p2 = FakeJudge([])
    s2 = run_judging(run, {9: ep}, client_for(tmp_path, p2), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert p2.requests == [] and s2["resumed_judgements"] == 2

    for f in (run / "judge" / "fake__fake-judge-1" / "judgements").glob("*.json"):
        f.unlink()
    p3 = FakeJudge([])
    c3 = client_for(tmp_path, p3)
    run_judging(run, {9: ep}, c3, temperature=0, max_tokens=4096, settings=SETTINGS)
    assert p3.requests == [] and c3.cache_hits == 2
    assert [r["verdict"] for r in s1["episodes"]["EP009"]] == [r["verdict"] for r in s2["episodes"]["EP009"]]


def test_quota_exhaustion_stops_and_marks_rest_pending(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n), candidate(ep, n, start="c4", end="c32", cid="EP009_C02")])
    provider = FakeJudge([QuotaExhausted("daily quota", status=429, kind="quota")])
    s = run_judging(run, {9: ep}, client_for(tmp_path, provider, max_attempts=1), temperature=0, max_tokens=4096,
                    settings=SETTINGS)
    assert [r["verdict"] for r in s["episodes"]["EP009"]] == ["not_judged", "not_judged"]
    assert "quota" in s["stopped_early"] and len(provider.requests) == 1


def test_provider_outage_is_not_judged_and_retried_on_resume(tmp_path):
    ep, n = episode([("cut", "DIALOGUE", 29, 31, "Don't open it.")])
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    down = FakeJudge([ProviderError("HTTP 503", status=503, kind="transient")])
    s1 = run_judging(run, {9: ep}, client_for(tmp_path, down, max_attempts=1), temperature=0, max_tokens=4096,
                     settings=SETTINGS)
    assert s1["episodes"]["EP009"][0]["verdict"] == "not_judged" and s1["stopped_early"] is None
    up = FakeJudge()
    s2 = run_judging(run, {9: ep}, client_for(tmp_path, up), temperature=0, max_tokens=4096, settings=SETTINGS)
    assert len(up.requests) == 2 and s2["resumed_judgements"] == 0
    assert [r["verdict"] for r in s2["episodes"]["EP009"]] == ["hard_fail", "eligible"]


def test_request_cap_zero_uses_no_quota(tmp_path):
    ep, n = episode()
    run = make_run(tmp_path, ep, [candidate(ep, n)])
    provider = FakeJudge()
    s = run_judging(run, {9: ep}, client_for(tmp_path, provider, max_requests=0), temperature=0, max_tokens=4096,
                    settings=SETTINGS)
    assert provider.requests == [] and s["episodes"]["EP009"][0]["verdict"] == "not_judged"


def test_judge_settings_defaults(monkeypatch):
    for var in ("PROMOPILOT_JUDGE_PROVIDER", "PROMOPILOT_JUDGE_MODEL"):
        monkeypatch.setenv(var, "")
    s = config.judge_settings()
    assert (s["provider"], s["model"]) == ("gemini", "gemini-3.5-flash-lite")
    assert sum(config.JUDGE_WEIGHTS.values()) == pytest.approx(1.0)
