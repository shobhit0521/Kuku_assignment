"""Uploaded-screenplay jobs for the public upload page.

Each job is a self-contained output root (input/, parsed/, runs/live/, llm_cache/) under `config.live_dir()`.
A background worker (`python -m promopilot.live JOBDIR`) runs the unchanged pipeline stages through
`pipeline.main` with PROMOPILOT_OUTPUT_DIR pointing at the job, so a job can never read or write the production
outputs. Progress is reported only from what each stage actually wrote.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .llm import redact

RUN_NAME = "live"
JUDGE_PROMPT = "judge-v1"
STAGES = ("parse", "generate", "validate", "judge", "rank")
STAGE_LABELS = {"parse": "Parse screenplays", "generate": "Generate candidate clips (Gemini)",
                "validate": "Validate hard rules", "judge": "Blind quality judge (Gemini)",
                "rank": "Rank and select the shortlist"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SECRET_ENVS = ("GEMINI_API_KEY", "GROQ_API_KEY")
_ACTIVE = "active.json"
_USAGE = "usage.json"


class JobError(Exception):
    """A job cannot be created or started; the message is safe to show to the visitor."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pid_alive(pid) -> bool:
    from .judge import _pid_alive as alive

    return bool(pid) and alive(int(pid))


def _secrets() -> list[str]:
    return [os.environ.get(k, "") for k in _SECRET_ENVS]


# --- paths ------------------------------------------------------------------------------------------

def root(base: Optional[Path] = None) -> Path:
    return Path(base) if base else config.live_dir()


def job_dir(job_id: str, base: Optional[Path] = None) -> Path:
    if not JOB_ID_RE.match(job_id or ""):
        raise JobError("invalid job reference")
    return root(base) / "jobs" / job_id


def status_path(job: Path) -> Path:
    return Path(job) / "status.json"


def read_status(job: Path) -> Optional[dict]:
    return _read_json(status_path(job))


def safe_name(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or "").name).strip("._") or "episode"
    return stem if stem.lower().endswith(".txt") else f"{stem}.txt"


# --- limits -----------------------------------------------------------------------------------------

def budgets(n_files: int, settings: Optional[dict] = None) -> dict:
    s = settings or config.live_settings()
    return {"generate": n_files * s["gen_requests_per_file"], "judge": n_files * s["judge_requests_per_file"]}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def usage_today(base: Optional[Path] = None) -> int:
    data = _read_json(root(base) / _USAGE) or {}
    return int(data.get("requests", 0)) if data.get("date") == _today() else 0


def _add_usage(requests: int, base: Optional[Path] = None) -> None:
    path = root(base) / _USAGE
    _write_json(path, {"date": _today(), "requests": usage_today(base) + int(requests)})


def active_job(base: Optional[Path] = None) -> Optional[str]:
    """The job currently running, if its worker process is alive. A stale marker is removed."""
    path = root(base) / _ACTIVE
    data = _read_json(path)
    if not data:
        return None
    if _pid_alive(data.get("pid")):
        return data.get("job_id")
    path.unlink(missing_ok=True)
    return None


def start_blocker(n_files: int, base: Optional[Path] = None, settings: Optional[dict] = None) -> Optional[str]:
    """Why a new job cannot start now, or None."""
    s = settings or config.live_settings()
    if active_job(base):
        return "Another analysis is running right now. Only one runs at a time; please try again in a few minutes."
    need = sum(budgets(n_files, s).values())
    used = usage_today(base)
    if used + need > s["daily_request_cap"]:
        return (f"Today's request limit for this demo is nearly used ({used} of {s['daily_request_cap']} model "
                f"requests; this upload may need up to {need}). Try fewer files or come back tomorrow (UTC).")
    return None


def precheck(name: str, raw: bytes, settings: Optional[dict] = None) -> dict:
    """Parse one upload with the production parser (nothing is written). `ok` means it can be analysed;
    `problem` explains why not, or lists format notes for a file that is still usable."""
    from .parser import ParseError, parse_episode_text
    from .schemas import Severity
    from .timecode import format_timestamp

    s = settings or config.live_settings()
    row = {"file": name, "ok": False, "episode_id": "", "scenes": None, "blocks": None, "duration": "",
           "problem": ""}
    if not str(name).lower().endswith(".txt"):
        row["problem"] = "Only .txt files are accepted."
        return row
    if len(raw) > s["max_file_kb"] * 1024:
        row["problem"] = f"Larger than {s['max_file_kb']} KB."
        return row
    try:
        ep = parse_episode_text(raw.decode("utf-8-sig"), safe_name(name))
    except UnicodeDecodeError:
        row["problem"] = "Not a UTF-8 text file."
        return row
    except ParseError as exc:
        row["problem"] = str(exc)
        return row
    except Exception as exc:  # the parser must never take the page down
        row["problem"] = f"Could not be parsed ({type(exc).__name__})."
        return row
    duration = ep.stats.duration_ms if ep.stats else 0
    row.update(episode_id=ep.episode_id, scenes=len(ep.scenes), blocks=len(ep.blocks),
               duration=format_timestamp(duration) if duration else "")
    if not ep.blocks:
        row["problem"] = "No timed blocks found. Check the expected format."
    elif duration < config.validation_settings()["min_clip_ms"]:
        row["problem"] = "Shorter than the minimum clip length, so no clip could pass the duration rule."
    else:
        row["ok"] = True
        if ep.anomalies:
            serious = sum(1 for a in ep.anomalies if a.severity == Severity.ERROR)
            row["problem"] = (f"{len(ep.anomalies)} format note(s), {serious} serious. It can still be analysed; "
                              "the hard rules catch timing problems.")
    return row


def duplicate_episodes(rows: list[dict]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for row in rows:
        if row["ok"]:
            seen.setdefault(row["episode_id"], []).append(row["file"])
    return {eid: files for eid, files in seen.items() if len(files) > 1}


# --- job lifecycle ----------------------------------------------------------------------------------

def create_job(files: list[tuple[str, bytes]], base: Optional[Path] = None, settings: Optional[dict] = None) -> str:
    """Write the uploaded files into a new job folder. Returns the job ID. Checks count, type and size only;
    the screenplay content is checked by the parse stage (and by the page's pre-check)."""
    s = settings or config.live_settings()
    if not files:
        raise JobError("no files uploaded")
    if len(files) > s["max_files"]:
        raise JobError(f"too many files: {len(files)} (limit {s['max_files']})")
    names: dict[str, bytes] = {}
    for name, raw in files:
        if not str(name).lower().endswith(".txt"):
            raise JobError(f"{name}: only .txt episode files are accepted")
        if len(raw) > s["max_file_kb"] * 1024:
            raise JobError(f"{name}: larger than {s['max_file_kb']} KB")
        clean = safe_name(name)
        while clean in names:
            clean = clean[:-4] + "_1.txt"
        names[clean] = raw
    job_id = uuid.uuid4().hex[:12]
    job = job_dir(job_id, base)
    (job / "input").mkdir(parents=True)
    for clean, raw in names.items():
        (job / "input" / clean).write_bytes(raw)
    stages = {st: {"state": "pending"} for st in STAGES}
    _write_json(status_path(job), {"job_id": job_id, "created_at": _now(), "state": "queued",
                                   "files": sorted(names), "budget": budgets(len(names), s),
                                   "requests": {"generate": 0, "judge": 0}, "stages": stages})
    return job_id


def start_job(job_id: str, base: Optional[Path] = None, settings: Optional[dict] = None) -> None:
    """Launch the background worker for a queued job."""
    job = job_dir(job_id, base)
    status = read_status(job)
    if not status or status.get("state") != "queued":
        raise JobError("this job is not waiting to start")
    blocker = start_blocker(len(status["files"]), base, settings)
    if blocker:
        raise JobError(blocker)
    log = open(job / "worker.log", "ab")
    kwargs = {"creationflags": 0x08000000 | 0x00000200} if os.name == "nt" else {"start_new_session": True}
    env = dict(os.environ)
    if base:
        env["PROMOPILOT_LIVE_DIR"] = str(root(base))
    proc = subprocess.Popen([sys.executable, "-m", "promopilot.live", str(job)], cwd=str(config.PROJECT_ROOT),
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kwargs)
    log.close()
    _write_json(root(base) / _ACTIVE, {"job_id": job_id, "pid": proc.pid, "started_at": _now()})
    status.update(state="starting", pid=proc.pid)
    _write_json(status_path(job), status)


def job_state(job_id: str, base: Optional[Path] = None) -> Optional[dict]:
    """The job's status, with a worker that died mid-run reported as such. None if the job no longer exists."""
    try:
        job = job_dir(job_id, base)
    except JobError:
        return None
    status = read_status(job)
    if status is None:
        return None
    if status.get("state") in ("starting", "running") and not _pid_alive(status.get("pid")):
        time.sleep(0.5)
        status = read_status(job) or status
        if status.get("state") in ("starting", "running"):
            status["state"] = "failed"
            status["error"] = "The analysis process stopped unexpectedly (the server may have restarted)."
    return status


def cancel_job(job_id: str, base: Optional[Path] = None) -> None:
    status = job_state(job_id, base)
    if status and status.get("state") in ("starting", "running") and _pid_alive(status.get("pid")):
        with contextlib.suppress(OSError):
            os.kill(int(status["pid"]), signal.SIGTERM)
    active = _read_json(root(base) / _ACTIVE) or {}
    if active.get("job_id") == job_id:
        (root(base) / _ACTIVE).unlink(missing_ok=True)


def delete_job(job_id: str, base: Optional[Path] = None) -> None:
    cancel_job(job_id, base)
    job = job_dir(job_id, base)
    for _ in range(5):
        shutil.rmtree(job, ignore_errors=True)
        if not job.exists():
            return
        time.sleep(0.3)


def sweep(base: Optional[Path] = None, settings: Optional[dict] = None) -> int:
    """Delete jobs older than the time limit that are not running. Returns how many were removed."""
    s = settings or config.live_settings()
    jobs = root(base) / "jobs"
    if not jobs.is_dir():
        return 0
    cutoff = time.time() - s["job_ttl_min"] * 60
    active, removed = active_job(base), 0
    for job in jobs.iterdir():
        if not job.is_dir() or job.name == active:
            continue
        marker = status_path(job) if status_path(job).exists() else job
        if marker.stat().st_mtime < cutoff:
            shutil.rmtree(job, ignore_errors=True)
            removed += 1
    return removed


# --- worker -----------------------------------------------------------------------------------------

@contextlib.contextmanager
def job_environment(job: Path):
    """Point every stage at the job folder for the duration of the run, then restore the environment."""
    overrides = {"PROMOPILOT_OUTPUT_DIR": str(job), "PROMOPILOT_PARSED_DIR": str(job / "parsed"),
                 "PROMOPILOT_INPUT_DIR": str(job / "input"), "PROMOPILOT_JUDGE_PROMPT_VERSION": JUDGE_PROMPT}
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def episode_numbers(job: Path) -> list[int]:
    nums = []
    for path in sorted((Path(job) / "parsed" / "episodes").glob("EP*.json")):
        data = _read_json(path) or {}
        if "episode_number" in data:
            nums.append(int(data["episode_number"]))
    return nums


def _argv(stage: str, job: Path, status: dict) -> list[str]:
    if stage == "parse":
        return ["parse", "--input", str(job / "input"), "--out", str(job / "parsed")]
    if stage == "generate":
        return ["generate", "--run", RUN_NAME, "--episodes", ",".join(map(str, episode_numbers(job))),
                "--max-requests", str(status["budget"]["generate"])]
    if stage == "validate":
        return ["validate", "--run", RUN_NAME]
    if stage == "judge":
        return ["judge", "--run", RUN_NAME, "--prompt-version", JUDGE_PROMPT,
                "--max-requests", str(status["budget"]["judge"])]
    return ["rank", "--run", RUN_NAME, "--prompt-version", JUDGE_PROMPT]


def run_dir(job: Path) -> Path:
    return Path(job) / "runs" / RUN_NAME


def judge_output(job: Path) -> Optional[Path]:
    base = run_dir(job) / "judge"
    found = sorted(p for p in base.iterdir() if (p / "summary.json").exists()) if base.is_dir() else []
    return found[0] if found else None


def _stage_detail(stage: str, job: Path) -> tuple[dict, int]:
    """Facts the stage wrote to disk, and the API requests it made."""
    if stage == "parse":
        pack = _read_json(job / "parsed" / "pack_summary.json") or {}
        return {"episodes": pack.get("episode_count", 0), "scenes": pack.get("scene_count", 0),
                "blocks": pack.get("block_count", 0)}, 0
    if stage == "generate":
        m = _read_json(run_dir(job) / "manifest.json") or {}
        eps = m.get("episodes") or {}
        return {"episodes_complete": sum(1 for e in eps.values() if e.get("status") == "complete"),
                "episodes": len(eps), "candidates": sum(int(e.get("candidates") or 0) for e in eps.values()),
                "stopped_early": m.get("stopped_early")}, int(m.get("requests_made_this_invocation") or 0)
    if stage == "validate":
        s = _read_json(run_dir(job) / "validation" / "summary.json") or {}
        records = [r for rows in (s.get("episodes") or {}).values() for r in rows]
        statuses: dict[str, int] = {}
        for r in records:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        return {"checked": len(records), "statuses": statuses}, 0
    if stage == "judge":
        out = judge_output(job)
        s = _read_json(out / "summary.json") if out else {}
        rows = [r for rs in ((s or {}).get("episodes") or {}).values() for r in rs]
        verdicts: dict[str, int] = {}
        for r in rows:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        return {"decisions": len(rows), "verdicts": verdicts, "stopped_early": (s or {}).get("stopped_early")}, \
            int((s or {}).get("requests_made_this_invocation") or 0)
    out = judge_output(job)
    rep = _read_json(run_dir(job) / "ranking" / out.name / "ranking.json") if out else None
    return {"shortlisted": len((rep or {}).get("shortlist") or []),
            "eligible": ((rep or {}).get("counts") or {}).get("passing_quality_gates", 0)}, 0


def _outcome(stage: str, rc: int, job: Path) -> str:
    """done | partial (stopped early or some episodes incomplete, but there is something to continue with) | failed"""
    if rc == 0:
        return "done"
    if stage == "generate" and rc in (1, 3) and any((run_dir(job) / "candidates").glob("EP*.json")):
        return "partial"
    if stage == "judge" and rc in (1, 3) and judge_output(job) is not None:
        return "partial"
    return "failed"


def _tail(text: str, lines: int = 6) -> str:
    kept = [l for l in text.splitlines() if l.strip()][-lines:]
    return redact("\n".join(kept), _secrets())[:800]


def run_job(job: Path, base: Optional[Path] = None) -> dict:
    """Run every stage in order inside the job folder. Writes status.json after each stage."""
    from . import pipeline

    job = Path(job).resolve()
    status = read_status(job)
    if status is None:
        raise JobError(f"no job at {job}")
    status.update(state="running", pid=os.getpid(), started_at=_now())
    _write_json(status_path(job), status)
    final = "done"
    try:
        with job_environment(job), open(job / "log.txt", "a", encoding="utf-8") as log:
            for stage in STAGES:
                info = status["stages"][stage]
                info.update(state="running", started_at=_now())
                _write_json(status_path(job), status)
                out, err = io.StringIO(), io.StringIO()
                try:
                    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                        rc = pipeline.main(_argv(stage, job, status))
                except Exception as exc:  # a crash in one stage must still leave an honest status
                    rc, err = -1, io.StringIO(f"{type(exc).__name__}: {exc}")
                log.write(f"== {stage} (exit {rc})\n{redact(out.getvalue() + err.getvalue(), _secrets())}\n")
                log.flush()
                detail, requests = _stage_detail(stage, job)
                if stage in status["requests"]:
                    status["requests"][stage] = requests
                state = _outcome(stage, rc, job)
                info.update(state=state, exit_code=rc, finished_at=_now(), detail=detail)
                if state != "done":
                    info["message"] = _tail(err.getvalue() or out.getvalue())
                elif stage == "parse" and not detail.get("episodes"):
                    state = info["state"] = "failed"
                    info["message"] = "No episode could be parsed from the uploaded files."
                _write_json(status_path(job), status)
                if state == "failed":
                    final = "failed"
                    status["error"] = f"{STAGE_LABELS[stage]} failed. " + (info.get("message") or "")
                    break
                if state == "partial":
                    final = "partial"
    finally:
        status.update(state=final, finished_at=_now())
        _write_json(status_path(job), status)
        _add_usage(sum(status["requests"].values()), base)
        active = _read_json(root(base) / _ACTIVE) or {}
        if active.get("job_id") == status["job_id"]:
            (root(base) / _ACTIVE).unlink(missing_ok=True)
    return status


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m promopilot.live JOBDIR", file=sys.stderr)
        return 2
    status = run_job(Path(args[0]))
    return 0 if status["state"] in ("done", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
