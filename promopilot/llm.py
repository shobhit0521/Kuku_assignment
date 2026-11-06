"""Provider-neutral LLM access: one interface, retries, throttling, response cache and call log.

Providers are configured only through environment variables; there is no automatic fallback to
another provider, so a run never moves to a different (possibly paid) service on its own.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from pydantic import BaseModel, Field

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

PROVIDER_REGISTRY = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY", "model_env": "GEMINI_MODEL", "base_url_env": "GEMINI_BASE_URL"},
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "model_env": "GROQ_MODEL", "base_url_env": "GROQ_BASE_URL"},
}


class ConfigError(Exception):
    pass


class ProviderError(Exception):
    """A failed provider call. kind: transient | quota | auth | client."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after_s: Optional[float] = None,
                 kind: str = "client"):
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s
        self.kind = kind
        self.retries: list[dict] = []

    @property
    def retryable(self) -> bool:
        return self.kind == "transient"


class QuotaExhausted(ProviderError):
    """Daily/free-tier quota is used up; retrying today will not help."""


class RequestCapReached(Exception):
    """The per-run request budget (PROMOPILOT_MAX_REQUESTS_PER_RUN) is spent."""


class DuplicateRequest(Exception):
    """The same cache key was already sent to the provider in this invocation (once_per_key clients)."""


class LLMRequest(BaseModel):
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    json_mode: bool = True


class LLMResponse(BaseModel):
    """Normalised response, identical in shape for every provider."""
    text: str
    provider: str
    model: str
    reported_model: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    latency_ms: int = 0
    rate_limit: dict[str, str] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def hidden_tokens(self) -> Optional[int]:
        """Tokens billed but not reported as prompt or visible completion (e.g. Gemini thinking)."""
        if None in (self.prompt_tokens, self.completion_tokens, self.total_tokens):
            return None
        return max(0, self.total_tokens - self.prompt_tokens - self.completion_tokens)


class Provider(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def _retry_after_seconds(headers: Any, body_text: str) -> Optional[float]:
    value = None
    try:
        value = headers.get("retry-after") if headers is not None else None
    except Exception:
        value = None
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
    match = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", body_text)
    return float(match.group(1)) if match else None


def classify_status_error(status: int, headers: Any, body: Any, secrets: list[str]) -> ProviderError:
    body_text = body if isinstance(body, str) else json.dumps(body, default=str)
    body_text = redact(body_text, secrets)
    retry_after = _retry_after_seconds(headers, body_text)
    message = f"HTTP {status}: {body_text[:400]}"
    if status == 429 and re.search(r"PerDay|per day|daily", body_text, re.I):
        return QuotaExhausted(message, status=status, retry_after_s=retry_after, kind="quota")
    if status in RETRYABLE_STATUS:
        return ProviderError(message, status=status, retry_after_s=retry_after, kind="transient")
    if status in (401, 403):
        return ProviderError(message, status=status, kind="auth")
    return ProviderError(message, status=status, kind="client")


class OpenAICompatibleProvider:
    """Any endpoint speaking the OpenAI chat-completions protocol (Gemini and Groq both do)."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout_s: float = 120,
                 client: Any = None):
        self.name = name
        self.model = model
        self._secrets = [api_key]
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout_s)
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        import openai

        params: dict[str, Any] = {"model": self.model, "messages": request.messages,
                                  "temperature": request.temperature, "max_tokens": request.max_tokens}
        if request.json_mode:
            params["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        try:
            raw = self._client.chat.completions.with_raw_response.create(**params)
        except openai.APIStatusError as exc:
            headers = exc.response.headers if getattr(exc, "response", None) is not None else None
            raise classify_status_error(exc.status_code, headers, exc.body if exc.body is not None else exc.message,
                                        self._secrets) from None
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise ProviderError(f"{type(exc).__name__} while calling {self.name}", kind="transient") from None
        latency_ms = int((time.perf_counter() - started) * 1000)
        completion = raw.parse()
        return normalise_completion(self.name, self.model, completion, raw.headers, latency_ms)


def normalise_completion(provider: str, model: str, completion: Any, headers: Any, latency_ms: int) -> LLMResponse:
    choice = completion.choices[0]
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    try:
        raw = completion.model_dump()
    except AttributeError:
        raw = {}
    rate_limit = {}
    if headers is not None:
        rate_limit = {k: v for k, v in headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    return LLMResponse(
        text=choice.message.content or "",
        provider=provider, model=model,
        reported_model=getattr(completion, "model", None),
        finish_reason=getattr(choice, "finish_reason", None),
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        latency_ms=latency_ms, rate_limit=rate_limit, raw=raw,
    )


def get_provider(name: Optional[str] = None, *, model: Optional[str] = None, timeout_s: float = 120,
                 client: Any = None) -> OpenAICompatibleProvider:
    """Build a provider from environment variables. Raises ConfigError without revealing secrets.
    `model` overrides the provider's *_MODEL variable (used for a separately configured judge model)."""
    name = (name or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if name not in PROVIDER_REGISTRY:
        raise ConfigError(f"unknown LLM provider {name!r}; expected one of {sorted(PROVIDER_REGISTRY)}")
    spec = PROVIDER_REGISTRY[name]
    api_key = os.environ.get(spec["key_env"], "").strip()
    model = (model or os.environ.get(spec["model_env"], "")).strip()
    if not api_key:
        raise ConfigError(f"{spec['key_env']} is not set in .env")
    if not model:
        raise ConfigError(f"{spec['model_env']} is not set in .env")
    base_url = os.environ.get(spec["base_url_env"], "").strip() or spec["base_url"]
    return OpenAICompatibleProvider(name, base_url, api_key, model, timeout_s=timeout_s, client=client)


def call_with_retry(provider: Provider, request: LLMRequest, *, max_attempts: int = 5, base_delay_s: float = 2,
                    max_delay_s: float = 65, sleep: Callable[[float], None] = time.sleep,
                    jitter: Callable[[float], float] = lambda d: d * random.uniform(0, 0.1)
                    ) -> tuple[LLMResponse, list[dict]]:
    """Bounded exponential backoff on transient errors; honours Retry-After / retryDelay when present."""
    retries: list[dict] = []
    for attempt in range(1, max_attempts + 1):
        try:
            return provider.complete(request), retries
        except ProviderError as exc:
            if not exc.retryable or attempt == max_attempts:
                exc.retries = retries
                raise
            if exc.retry_after_s is not None:
                delay = min(exc.retry_after_s, max_delay_s)
            else:
                delay = min(max_delay_s, base_delay_s * 2 ** (attempt - 1))
                delay = min(max_delay_s, delay + jitter(delay))
            retries.append({"attempt": attempt, "status": exc.status, "delay_s": round(delay, 2),
                            "error": str(exc)[:200]})
            sleep(delay)
    raise AssertionError("unreachable")


class ResponseCache:
    """One JSON file per request fingerprint. Writes are atomic; unreadable files count as misses."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(provider: str, model: str, prompt_version: str, request: LLMRequest) -> str:
        payload = {"provider": provider, "model": model, "prompt_version": prompt_version,
                   "request": request.model_dump()}
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[LLMResponse]:
        path = self.path(key)
        if not path.exists():
            return None
        try:
            return LLMResponse.model_validate(json.loads(path.read_text(encoding="utf-8"))["response"])
        except (ValueError, KeyError):
            return None

    def put(self, key: str, meta: dict, response: LLMResponse) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "meta": meta, "response": response.model_dump()},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)


class CallResult(BaseModel):
    response: LLMResponse
    cache_key: str
    cache_hit: bool
    retries: list[dict] = Field(default_factory=list)

    def summary(self, stage: str) -> dict:
        r = self.response
        return {"stage": stage, "cache_key": self.cache_key, "cache_hit": self.cache_hit,
                "retries": len(self.retries), "model": r.model, "reported_model": r.reported_model,
                "finish_reason": r.finish_reason, "latency_ms": r.latency_ms,
                "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens, "hidden_tokens": r.hidden_tokens}


class LLMClient:
    """Cache-first, throttled, budget-capped access to one provider, with a JSONL call log."""

    def __init__(self, provider: Provider, cache: ResponseCache, log_path: Optional[Path], *,
                 min_interval_s: float = 4, max_requests: int = 30, max_attempts: int = 5,
                 base_delay_s: float = 2, max_delay_s: float = 65, bypass_cache: bool = False,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 jitter: Optional[Callable[[float], float]] = None, once_per_key: bool = False,
                 log_extra: Optional[dict] = None):
        self.provider = provider
        self.once_per_key = once_per_key
        self.log_extra = dict(log_extra or {})
        self._issued: set[str] = set()
        self.cache = cache
        self.log_path = Path(log_path) if log_path else None
        self.min_interval_s = min_interval_s
        self.max_requests = max_requests
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.bypass_cache = bypass_cache
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._last_call: Optional[float] = None
        self.requests_made = 0
        self.cache_hits = 0

    def complete(self, request: LLMRequest, *, prompt_version: str, stage: str, item_id: str) -> CallResult:
        key = ResponseCache.key(self.provider.name, self.provider.model, prompt_version, request)
        if not self.bypass_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                result = CallResult(response=cached, cache_key=key, cache_hit=True)
                self._log(stage, item_id, prompt_version, "ok", result=result)
                return result
        if self.once_per_key and key in self._issued:
            self._log(stage, item_id, prompt_version, "duplicate_blocked", key=key,
                      error="same request already sent in this invocation")
            raise DuplicateRequest(f"{stage} {item_id}: request {key[:12]} already sent in this invocation")
        if self.requests_made >= self.max_requests:
            raise RequestCapReached(f"request budget of {self.max_requests} for this run is spent")
        self._issued.add(key)
        if self._last_call is not None:
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self.requests_made += 1
        kwargs = {"jitter": self._jitter} if self._jitter else {}
        try:
            response, retries = call_with_retry(
                self.provider, request, max_attempts=self.max_attempts, base_delay_s=self.base_delay_s,
                max_delay_s=self.max_delay_s, sleep=self._sleep, **kwargs)
        except ProviderError as exc:
            self._log(stage, item_id, prompt_version, "quota_exhausted" if isinstance(exc, QuotaExhausted)
                      else "error", key=key, error=str(exc), retries=exc.retries)
            raise
        finally:
            self._last_call = self._clock()
        self.cache.put(key, {"provider": self.provider.name, "model": self.provider.model,
                             "prompt_version": prompt_version, "stage": stage, "item_id": item_id,
                             "request": request.model_dump(),
                             "created_at": datetime.now(timezone.utc).isoformat()}, response)
        result = CallResult(response=response, cache_key=key, cache_hit=False, retries=retries)
        self._log(stage, item_id, prompt_version, "ok", result=result)
        return result

    def _log(self, stage: str, item_id: str, prompt_version: str, status: str, *, result: Optional[CallResult] = None,
             key: Optional[str] = None, error: Optional[str] = None, retries: Optional[list] = None) -> None:
        if self.log_path is None:
            return
        entry = {"time": datetime.now(timezone.utc).isoformat(), "provider": self.provider.name,
                 "model": self.provider.model, "prompt_version": prompt_version, "stage": stage,
                 "item_id": item_id, "status": status, **self.log_extra}
        if result is not None:
            entry.update(result.summary(stage))
            entry["retry_log"] = result.retries
            entry["rate_limit"] = result.response.rate_limit if not result.cache_hit else {}
        else:
            entry.update({"cache_key": key, "cache_hit": False, "error": error, "retry_log": retries or []})
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
