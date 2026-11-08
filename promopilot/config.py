"""Project settings. Everything environment-specific comes from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PARSER_VERSION = "1.0.0"

# Two blocks' times are treated as equal within this tolerance (timestamps have 2 decimals).
TIME_TOLERANCE_MS = 10

# Episode-level informational flags.
LARGE_CAMERA_GAP_MS = 5000
UNCOVERED_TIME_FLAG_MS = 1000

def input_dir_from_env() -> Path | None:
    value = os.environ.get("PROMOPILOT_INPUT_DIR", "").strip()
    return Path(value) if value else None


def output_dir() -> Path:
    """Root for all screenplay-derived outputs; falls back to the gitignored ./outputs folder."""
    value = os.environ.get("PROMOPILOT_OUTPUT_DIR", "").strip()
    return Path(value) if value else PROJECT_ROOT / "outputs"


def parsed_dir() -> Path:
    """Default parsed dataset: PROMOPILOT_PARSED_DIR if set, else <output dir>/parsed."""
    value = os.environ.get("PROMOPILOT_PARSED_DIR", "").strip()
    return Path(value) if value else output_dir() / "parsed"


def runs_dir() -> Path:
    return output_dir() / "runs"


def llm_cache_dir() -> Path:
    return output_dir() / "llm_cache"


def _env_number(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    return float(value) if value else default


VALIDATOR_VERSION = "1.0.0"


# Hard-rule limits (all overridable in .env). Bounds are inclusive.
def validation_settings() -> dict:
    return {
        "min_clip_ms": int(_env_number("PROMOPILOT_MIN_CLIP_MS", 20_000)),
        "max_clip_ms": int(_env_number("PROMOPILOT_MAX_CLIP_MS", 60_000)),
        "hook_window_ms": int(_env_number("PROMOPILOT_HOOK_WINDOW_MS", 3_000)),
        "suspense_tail_ms": int(_env_number("PROMOPILOT_SUSPENSE_TAIL_MS", 2_000)),
        "ending_window_ms": int(_env_number("PROMOPILOT_ENDING_WINDOW_MS", 8_000)),
    }


JUDGE_WEIGHTS = {"hook": 0.30, "cliffhanger": 0.25, "context_independence": 0.30, "emotional_impact": 0.15}
JUDGE_GATE_MIN = {"hook": 3, "cliffhanger": 3, "context_independence": 3}


def judge_settings(provider: str | None = None, model: str | None = None) -> dict:
    """Judge configuration; provider/model precedence is documented in `llm.resolve_judge`. `model` is None when
    no model can be resolved for the provider (the judge command then reports a configuration error)."""
    from .llm import ConfigError, resolve_judge

    try:
        provider, model = resolve_judge(provider, model)
    except ConfigError:
        provider = (provider or os.environ.get("PROMOPILOT_JUDGE_PROVIDER", "")).strip().lower() or "gemini"
        model = None
    return {
        "provider": provider,
        "model": model,
        "prompt_version": os.environ.get("PROMOPILOT_JUDGE_PROMPT_VERSION", "").strip() or "judge-v1",
        "temperature": _env_number("PROMOPILOT_JUDGE_TEMPERATURE", 0.0),
        "max_tokens": int(_env_number("PROMOPILOT_JUDGE_MAX_TOKENS", 4096)),
        "max_requests_per_run": int(_env_number("PROMOPILOT_JUDGE_MAX_REQUESTS_PER_RUN", 60)),
    }


# LLM call controls (all overridable in .env).
def llm_settings() -> dict:
    return {
        "provider": os.environ.get("LLM_PROVIDER", "gemini").strip().lower(),
        "timeout_s": _env_number("PROMOPILOT_REQUEST_TIMEOUT_S", 120),
        "min_interval_s": _env_number("PROMOPILOT_MIN_REQUEST_INTERVAL_S", 4),
        "max_requests_per_run": int(_env_number("PROMOPILOT_MAX_REQUESTS_PER_RUN", 30)),
        "retry_max_attempts": int(_env_number("PROMOPILOT_RETRY_MAX_ATTEMPTS", 5)),
        "retry_base_delay_s": _env_number("PROMOPILOT_RETRY_BASE_DELAY_S", 2),
        "retry_max_delay_s": _env_number("PROMOPILOT_RETRY_MAX_DELAY_S", 65),
        "gen_temperature": _env_number("PROMOPILOT_GEN_TEMPERATURE", 0.2),
        "gen_max_tokens": int(_env_number("PROMOPILOT_GEN_MAX_TOKENS", 8192)),
        "gen_candidates": int(_env_number("PROMOPILOT_GEN_CANDIDATES", 5)),
    }


def dashboard_dir() -> Path:
    """Output root the dashboard pages read: PROMOPILOT_DASHBOARD_DIR (relative paths resolve against the project),
    else the output dir."""
    value = os.environ.get("PROMOPILOT_DASHBOARD_DIR", "").strip()
    if not value:
        return output_dir()
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def live_dir() -> Path:
    """Scratch root for uploaded-screenplay jobs. Never inside the dashboard or production output folders."""
    import tempfile

    value = os.environ.get("PROMOPILOT_LIVE_DIR", "").strip()
    return Path(value) if value else Path(tempfile.gettempdir()) / "promopilot_live"


# Limits for the public upload page (all overridable in .env or deployment secrets).
def live_settings() -> dict:
    return {
        "max_files": int(_env_number("PROMOPILOT_LIVE_MAX_FILES", 5)),
        "max_file_kb": int(_env_number("PROMOPILOT_LIVE_MAX_FILE_KB", 100)),
        "gen_requests_per_file": int(_env_number("PROMOPILOT_LIVE_GEN_REQUESTS_PER_FILE", 2)),
        "judge_requests_per_file": int(_env_number("PROMOPILOT_LIVE_JUDGE_REQUESTS_PER_FILE", 12)),
        "daily_request_cap": int(_env_number("PROMOPILOT_LIVE_DAILY_REQUEST_CAP", 150)),
        "job_ttl_min": int(_env_number("PROMOPILOT_LIVE_JOB_TTL_MIN", 120)),
    }
