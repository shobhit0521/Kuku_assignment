"""A synthetic output directory for UI tests, produced by the real pipeline code with a scripted fake judge.

One episode (EP009) covers every candidate state the UI distinguishes; a second requested episode (EP010) is
still pending generation. A judge-v2 output exists beside the judge-v1 output to test rubric separation.
"""
import json
from pathlib import Path

from promopilot import ranking
from promopilot.judge import run_judging
from promopilot.llm import ProviderError
from promopilot.schemas import EpisodeGeneration
from promopilot.validator import candidate_from_blocks
from test_judge import SECRETS, SETTINGS, FakeJudge, candidate, client_for, default_reply, episode
from test_judge_consistency import cliff_only, context_flagged_reply, context_revision, flagged_reply

RUN = "fixture"
V1 = "fake__fake-judge-1"
V2 = "fake__fake-judge-1__judge-v2"
FAKE_KEY = "AIza" + "FAKE" * 6

EXPECTED_LANES = {
    "EP009_C01": "technical_fail", "EP009_C01_R1": "eligible", "EP009_C02": "eligible", "EP009_C03": "eligible",
    "EP009_C04": "quality_fail", "EP009_C05": "not_judged", "EP009_C06": "judge_invalid", "EP009_C07": "pending",
    "EP009_C08": "technical_fail",
}


def _write_generation(run: Path, cands) -> None:
    gen = EpisodeGeneration(episode_id="EP009", status="complete", provider="gen", model="gen-model-1",
                            prompt_version="gen-v1", prompt_sha256="1" * 64, candidates=cands)
    (run / "candidates").mkdir(parents=True, exist_ok=True)
    (run / "candidates" / "EP009.json").write_text(gen.model_dump_json(indent=1), encoding="utf-8")


def build(root: Path) -> Path:
    """Write the fixture under root/outputs and return that output directory."""
    ep, n = episode([("cut", "DIALOGUE", 29, 31, "Don't open it.")])
    out = root / "outputs"
    (out / "parsed" / "episodes").mkdir(parents=True)
    (out / "parsed" / "episodes" / "EP009.json").write_text(ep.model_dump_json(), encoding="utf-8")
    (out / "parsed" / "pack_summary.json").write_text(json.dumps(
        {"pack_name": "Fixture pack", "parser_version": "test", "episode_count": 2}), encoding="utf-8")

    run = out / "runs" / RUN
    cands = [candidate(ep, n),                                              # cuts dialogue -> repaired
             candidate(ep, n, start="c4", end="c32", cid="EP009_C02"),      # near-duplicate of the repair
             candidate(ep, n, start="c34", end="c56", cid="EP009_C03"),     # eligible, review flag, episode cap
             candidate(ep, n, start="c10", end="c40", cid="EP009_C04"),     # quality fail after reassessment
             candidate(ep, n, start="c20", end="c50", cid="EP009_C05"),     # provider error
             candidate_from_blocks(ep, "EP009_C06", n["c12"], n["c44"], n["c12"], n["c44"])
             .model_copy(update=SECRETS),                                    # unusable judge output
             candidate(ep, n, start="c0", end="c8", cid="EP009_C08")]       # 10 s at 0 s: repair not possible
    _write_generation(run, cands)
    script = [default_reply, default_reply, default_reply,
              context_flagged_reply, context_revision(5, keep_confusions=False),
              flagged_reply, cliff_only(2, "An emotional button with no open question."),
              ProviderError("HTTP 503 key=" + FAKE_KEY, status=503, kind="transient"),
              "not json", "not json", default_reply]
    run_judging(run, {9: ep}, client_for(root / "v1", FakeJudge(script), max_attempts=1), temperature=0,
                max_tokens=4096, settings=SETTINGS)
    v2 = [lambda t: default_reply(t, hook__score=2)] * 20
    run_judging(run, {9: ep}, client_for(root / "v2", FakeJudge(v2), max_attempts=1), temperature=0,
                max_tokens=4096, settings=SETTINGS, prompt_version="judge-v2")
    ranking.write_report(run, run / "judge" / V1)

    _write_generation(run, cands[:-1] + [candidate(ep, n, start="c6", end="c36", cid="EP009_C07"), cands[-1]])
    (run / "manifest.json").write_text(json.dumps({
        "episodes": {"EP009": {"status": "complete", "candidates": 7},
                     "EP010": {"status": "pending", "reason": f"quota exhausted: HTTP 429 key={FAKE_KEY}"}},
        "provider": "gen", "model": "gen-model-1", "prompt_version": "gen-v1", "prompt_sha256": "1" * 64,
        "requested": ["EP009", "EP010"], "stopped_early": "quota exhausted"}), encoding="utf-8")
    return out
