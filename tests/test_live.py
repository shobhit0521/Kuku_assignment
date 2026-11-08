"""Upload jobs: limits, isolation, the stage runner, failure reporting, locking, the daily cap and cleanup. The full
pipeline runs in-process against a fake provider; no network, no API quota."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pytest

from promopilot import live, pipeline
from promopilot.llm import ConfigError, LLMResponse
from test_generator import build_episode_text, ok_reply
from test_judge import default_reply

CLEARED = ["PROMOPILOT_INPUT_DIR", "PROMOPILOT_PARSED_DIR", "PROMOPILOT_JUDGE_PROVIDER", "PROMOPILOT_JUDGE_MODEL",
           "PROMOPILOT_JUDGE_PROMPT_VERSION", "PROMOPILOT_PROVIDERS", "GEMINI_BASE_URL", "GEMINI_JSON_MODE",
           "GEMINI_SEND_TEMPERATURE", "GEMINI_MAX_TOKENS_PARAM", "GEMINI_JUDGE_MODEL", "PROMOPILOT_DASHBOARD_DIR"]
SETTINGS = {"max_files": 3, "max_file_kb": 50, "gen_requests_per_file": 2, "judge_requests_per_file": 12,
            "daily_request_cap": 100, "job_ttl_min": 60}


class PipelineFake:
    """Answers generation prompts with a grounded proposal and judge prompts with a valid judgement."""

    def __init__(self, name="gemini", model="fake-gen-1"):
        self.name, self.model, self.endpoint, self.calls = name, model, None, 0

    def complete(self, request):
        self.calls += 1
        user = request.messages[-1]["content"]
        if "ELIGIBLE HOOK BLOCKS" in user:
            text = default_reply(user)
        else:
            text = ok_reply(int(re.search(r"\(EP(\d{3})\)", user).group(1)))
        return LLMResponse(text=text, provider=self.name, model=self.model, finish_reason="stop",
                           prompt_tokens=100, completion_tokens=50, total_tokens=150, latency_ms=5)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for var in CLEARED:
        monkeypatch.delenv(var, raising=False)
    prod = tmp_path / "prod"
    (prod / "runs").mkdir(parents=True)
    (prod / "runs" / "sentinel.txt").write_text("production outputs", encoding="utf-8")
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(prod))
    monkeypatch.setenv("PROMOPILOT_LIVE_DIR", str(tmp_path / "live"))
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("PROMOPILOT_MIN_REQUEST_INTERVAL_S", "0")
    monkeypatch.setenv("PROMOPILOT_RETRY_BASE_DELAY_S", "0")
    monkeypatch.setattr(pipeline, "get_provider", lambda name=None, *, model=None, **k: PipelineFake(
        name or "gemini", model or "fake-gen-1"))
    return monkeypatch


def upload(*numbers):
    return [(f"episode-{n:03d}.txt", build_episode_text(n, logline=f"Upload {n}.").encode("utf-8")) for n in numbers]


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --- upload pre-check --------------------------------------------------------------------------------

def test_precheck_accepts_a_well_formed_episode(env):
    name, raw = upload(7)[0]
    row = live.precheck(name, raw, SETTINGS)
    assert row["ok"] and row["episode_id"] == "EP007" and row["blocks"] == 12 and row["duration"] == "00:40.00"


@pytest.mark.parametrize("name,raw,problem", [
    ("notes.pdf", b"x", "Only .txt"),
    ("big.txt", b"x" * (51 * 1024), "Larger than 50 KB"),
    ("latin.txt", "EPISODE NUMBER: Episode 1 caf\xe9".encode("latin-1"), "Not a UTF-8"),
    ("notes.txt", b"hello", "cannot determine episode number"),
    ("episode-004.txt", b"EPISODE NUMBER: Episode 4\n--- SCRIPT START ---\n--- SCRIPT END ---", "No timed blocks"),
])
def test_precheck_explains_why_a_file_cannot_be_analysed(env, name, raw, problem):
    row = live.precheck(name, raw, SETTINGS)
    assert not row["ok"] and problem in row["problem"]


def test_duplicate_episode_numbers_are_detected(env):
    rows = [live.precheck(n, r, SETTINGS) for n, r in upload(1, 2)]
    rows.append(live.precheck("copy.txt", upload(1)[0][1], SETTINGS))
    assert live.duplicate_episodes(rows) == {"EP001": ["episode-001.txt", "copy.txt"]}


# --- job creation ------------------------------------------------------------------------------------

def test_create_job_writes_inputs_and_a_queued_status(env):
    jid = live.create_job(upload(1, 2), settings=SETTINGS)
    job = live.job_dir(jid)
    assert sorted(p.name for p in (job / "input").iterdir()) == ["episode-001.txt", "episode-002.txt"]
    status = live.read_status(job)
    assert status["state"] == "queued" and status["budget"] == {"generate": 4, "judge": 24}
    assert all(s["state"] == "pending" for s in status["stages"].values())


@pytest.mark.parametrize("files,message", [
    ([], "no files"),
    (upload(1, 2, 3, 4), "too many files"),
    ([("notes.pdf", b"x")], "only .txt"),
    ([("big.txt", b"x" * (51 * 1024))], "larger than 50 KB"),
])
def test_create_job_rejects_bad_uploads(env, files, message):
    with pytest.raises(live.JobError, match=message):
        live.create_job(files, settings=SETTINGS)
    assert not (live.root() / "jobs").exists() or not any((live.root() / "jobs").iterdir())


def test_upload_names_are_sanitised_and_deduplicated(env):
    jid = live.create_job([("../../evil episode 1.txt", b"a"), ("evil_episode_1.txt", b"b")], settings=SETTINGS)
    names = sorted(p.name for p in (live.job_dir(jid) / "input").iterdir())
    assert names == ["evil_episode_1.txt", "evil_episode_1_1.txt"]


def test_job_ids_cannot_escape_the_jobs_folder(env):
    for bad in ("../x", "", "ABCDEF123456", "0" * 13):
        with pytest.raises(live.JobError):
            live.job_dir(bad)
    assert live.job_state("../../etc") is None


# --- the worker --------------------------------------------------------------------------------------

def test_full_run_ranks_inside_the_job_and_leaves_production_untouched(env, tmp_path):
    prod = tmp_path / "prod"
    before = tree(prod)
    saved = dict(os.environ)
    jid = live.create_job(upload(1, 2), settings=SETTINGS)
    status = live.run_job(live.job_dir(jid))

    assert status["state"] == "done", status
    assert [status["stages"][s]["state"] for s in live.STAGES] == ["done"] * 5
    assert status["stages"]["parse"]["detail"]["episodes"] == 2
    assert status["stages"]["generate"]["detail"]["candidates"] == 2
    assert status["stages"]["judge"]["detail"]["decisions"] == 2
    # both episodes share one identical clip, so the second judge request is a cache hit
    assert status["requests"]["generate"] == 2 and status["requests"]["judge"] == 1
    job = live.job_dir(jid)
    out = live.judge_output(job)
    assert out is not None and out.name.startswith("gemini__")
    assert (live.run_dir(job) / "ranking" / out.name / "ranking.json").exists()
    assert (job / "llm_cache").is_dir() and (job / "log.txt").exists()
    assert tree(prod) == before
    assert dict(os.environ) == saved
    assert live.usage_today() == sum(status["requests"].values())
    assert live.active_job() is None


def test_results_load_with_the_dashboard_loader(env):
    from promopilot_ui import data

    jid = live.create_job(upload(3), settings=SETTINGS)
    live.run_job(live.job_dir(jid))
    job = live.job_dir(jid)
    runs = data.discover_runs(job)
    ctx = data.load_run(runs[0], None, job)
    assert ctx.ranking is not None and not ctx.issues
    assert {v.episode_id for v in ctx.candidates.values()} == {"EP003"}
    episode, issue = data.load_episode("EP003", job)
    assert episode is not None and issue is None


def test_spent_generation_budget_fails_honestly(env):
    jid = live.create_job(upload(1), settings={**SETTINGS, "gen_requests_per_file": 0})
    status = live.run_job(live.job_dir(jid))
    assert status["state"] == "failed"
    assert status["stages"]["parse"]["state"] == "done"
    assert status["stages"]["generate"]["state"] == "failed"
    assert status["stages"]["validate"]["state"] == "pending"
    assert "failed" in status["error"]


def test_provider_errors_are_reported_without_secrets(env):
    secret = "AIza" + "SECRET" * 6
    env.setenv("GEMINI_API_KEY", secret)

    def broken(*a, **k):
        raise ConfigError(f"GEMINI_API_KEY is not set in .env (saw {secret})")

    env.setattr(pipeline, "get_provider", broken)
    jid = live.create_job(upload(1), settings=SETTINGS)
    status = live.run_job(live.job_dir(jid))
    msg = status["stages"]["generate"]["message"]
    assert status["state"] == "failed" and "not set" in msg
    assert secret not in json.dumps(status) and secret not in (live.job_dir(jid) / "log.txt").read_text("utf-8")


def test_a_crashing_stage_still_leaves_a_final_status(env):
    def crash(argv=None):
        raise RuntimeError("boom")

    env.setattr(pipeline, "main", crash)
    jid = live.create_job(upload(1), settings=SETTINGS)
    status = live.run_job(live.job_dir(jid))
    assert status["state"] == "failed" and status["stages"]["parse"]["exit_code"] == -1
    assert "RuntimeError: boom" in status["stages"]["parse"]["message"]


# --- locking, daily cap, liveness and cleanup ----------------------------------------------------------

def test_only_one_job_runs_at_a_time(env):
    live._write_json(live.root() / "active.json", {"job_id": "a" * 12, "pid": os.getpid()})
    assert "Another analysis is running" in live.start_blocker(1, settings=SETTINGS)
    live._write_json(live.root() / "active.json", {"job_id": "a" * 12, "pid": 2 ** 30})
    assert live.start_blocker(1, settings=SETTINGS) is None
    assert not (live.root() / "active.json").exists()


def test_daily_cap_blocks_new_jobs(env):
    live._add_usage(80)
    assert live.start_blocker(1, settings=SETTINGS) is None
    assert "request limit" in live.start_blocker(2, settings=SETTINGS)


def test_a_dead_worker_is_reported_as_failed(env):
    jid = live.create_job(upload(1), settings=SETTINGS)
    status = live.read_status(live.job_dir(jid))
    status.update(state="running", pid=2 ** 30)
    live._write_json(live.status_path(live.job_dir(jid)), status)
    state = live.job_state(jid)
    assert state["state"] == "failed" and "stopped unexpectedly" in state["error"]


def test_sweep_removes_only_expired_jobs_and_delete_removes_a_job(env):
    old, new = live.create_job(upload(1), settings=SETTINGS), live.create_job(upload(2), settings=SETTINGS)
    past = time.time() - 2 * 3600
    os.utime(live.status_path(live.job_dir(old)), (past, past))
    assert live.sweep(settings=SETTINGS) == 1
    assert not live.job_dir(old).exists() and live.job_dir(new).exists()
    live.delete_job(new)
    assert not live.job_dir(new).exists() and live.job_state(new) is None


def test_start_job_launches_a_worker_that_finishes(env, tmp_path):
    """The real subprocess path. Keys are blanked and the provider is unknown, so no model call is possible
    (the fake provider does not apply in a subprocess); the file has no episode number, so parsing fails."""
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        env.setenv(key, "")
    env.setenv("LLM_PROVIDER", "none-such")
    jid = live.create_job([("notes.txt", b"not a screenplay")], settings=SETTINGS)
    live.start_job(jid, settings=SETTINGS)
    deadline = time.time() + 120
    while time.time() < deadline:
        state = live.job_state(jid)
        if state and state["state"] not in ("queued", "starting", "running"):
            break
        time.sleep(0.5)
    assert state["state"] == "failed" and state["stages"]["parse"]["state"] == "failed"
    assert state["requests"] == {"generate": 0, "judge": 0}
    assert live.active_job() is None
