"""Provider abstraction, error classification, retries, caching and logging. No network access."""
import json
from types import SimpleNamespace

import pytest

from promopilot.llm import (ConfigError, LLMClient, LLMRequest, LLMResponse, OpenAICompatibleProvider,
                            ProviderError, QuotaExhausted, RequestCapReached, ResponseCache, call_with_retry,
                            classify_status_error, get_provider, redact)

REQ = LLMRequest(messages=[{"role": "user", "content": "hi"}], temperature=0, max_tokens=100)
FAKE_KEY = "test-secret-key-123"


class ScriptedProvider:
    """Returns or raises the scripted items in order."""

    def __init__(self, script, name="mock", model="mock-1"):
        self.name, self.model = name, model
        self.script = list(script)
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, provider=self.name, model=self.model, prompt_tokens=10,
                           completion_tokens=5, total_tokens=30, latency_ms=7)


def transient(status=503, retry_after=None):
    return ProviderError(f"HTTP {status}", status=status, retry_after_s=retry_after, kind="transient")


# --- configuration ---------------------------------------------------------------------------------

def test_get_provider_reads_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GROQ_MODEL", "some/model")
    p = get_provider("groq", client=object())
    assert (p.name, p.model) == ("groq", "some/model")


def test_provider_and_model_swap_through_env_only(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "model-a")
    assert get_provider(client=object()).model == "model-a"
    monkeypatch.setenv("GEMINI_MODEL", "model-b")
    assert get_provider(client=object()).model == "model-b"


@pytest.mark.parametrize("missing", ["GEMINI_API_KEY", "GEMINI_MODEL"])
def test_missing_config_is_reported_without_secret(monkeypatch, missing):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv(missing, "")
    with pytest.raises(ConfigError) as exc:
        get_provider("gemini", client=object())
    assert missing in str(exc.value) and FAKE_KEY not in str(exc.value)


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError):
        get_provider("someone-else", client=object())


# --- normalisation ---------------------------------------------------------------------------------

class FakeRaw:
    def __init__(self, completion, headers):
        self._completion, self.headers = completion, headers

    def parse(self):
        return self._completion


def fake_client(completion, headers):
    create = lambda **params: FakeRaw(completion, headers)  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))


def test_openai_compatible_response_is_normalised():
    completion = SimpleNamespace(
        model="reported-model", choices=[SimpleNamespace(message=SimpleNamespace(content='{"a": 1}'),
                                                         finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=24, completion_tokens=12, total_tokens=225,
                              completion_tokens_details=None))
    provider = OpenAICompatibleProvider("gemini", "http://x", FAKE_KEY, "cfg-model",
                                        client=fake_client(completion, {"x-ratelimit-limit-tokens": "8000",
                                                                        "content-type": "json"}))
    r = provider.complete(REQ)
    assert (r.text, r.provider, r.model, r.reported_model, r.finish_reason) == \
        ('{"a": 1}', "gemini", "cfg-model", "reported-model", "stop")
    assert (r.prompt_tokens, r.completion_tokens, r.total_tokens, r.hidden_tokens) == (24, 12, 225, 189)
    assert r.rate_limit == {"x-ratelimit-limit-tokens": "8000"}


# --- error classification --------------------------------------------------------------------------

@pytest.mark.parametrize("status, kind", [(429, "transient"), (500, "transient"), (502, "transient"),
                                          (503, "transient"), (504, "transient"), (400, "client"),
                                          (404, "client"), (401, "auth"), (403, "auth")])
def test_status_classification(status, kind):
    assert classify_status_error(status, {}, {"error": "x"}, []).kind == kind


def test_daily_quota_is_not_retryable():
    body = [{"error": {"code": 429, "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}]
    err = classify_status_error(429, {}, body, [])
    assert isinstance(err, QuotaExhausted) and not err.retryable


def test_retry_after_from_header_and_gemini_body():
    assert classify_status_error(429, {"retry-after": "7"}, "slow down", []).retry_after_s == 7.0
    body = [{"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "21s"}]}}]
    assert classify_status_error(429, {}, body, []).retry_after_s == 21.0


def test_secrets_are_redacted_from_errors():
    err = classify_status_error(400, {}, f"bad key {FAKE_KEY} supplied", [FAKE_KEY])
    assert FAKE_KEY not in str(err) and "[REDACTED]" in str(err)
    assert redact("abc", [""]) == "abc"


# --- retries ---------------------------------------------------------------------------------------

def test_transient_errors_retry_with_exponential_backoff():
    sleeps = []
    provider = ScriptedProvider([transient(503), transient(502), '{"ok": 1}'])
    resp, retries = call_with_retry(provider, REQ, max_attempts=5, base_delay_s=2, max_delay_s=65,
                                    sleep=sleeps.append, jitter=lambda d: 0)
    assert resp.text == '{"ok": 1}' and provider.calls == 3
    assert sleeps == [2, 4] and [r["status"] for r in retries] == [503, 502]


def test_retry_after_is_respected_and_bounded():
    sleeps = []
    provider = ScriptedProvider([transient(429, retry_after=9), transient(429, retry_after=999), "{}"])
    call_with_retry(provider, REQ, max_attempts=5, base_delay_s=2, max_delay_s=65, sleep=sleeps.append,
                    jitter=lambda d: 0)
    assert sleeps == [9, 65]


def test_backoff_is_capped():
    sleeps = []
    provider = ScriptedProvider([transient()] * 4 + ["{}"])
    call_with_retry(provider, REQ, max_attempts=5, base_delay_s=10, max_delay_s=30, sleep=sleeps.append,
                    jitter=lambda d: 0)
    assert sleeps == [10, 20, 30, 30]


def test_gives_up_after_max_attempts_and_keeps_retry_log():
    provider = ScriptedProvider([transient()] * 3)
    with pytest.raises(ProviderError) as exc:
        call_with_retry(provider, REQ, max_attempts=3, sleep=lambda s: None, jitter=lambda d: 0)
    assert provider.calls == 3 and len(exc.value.retries) == 2


@pytest.mark.parametrize("error", [ProviderError("HTTP 400", status=400, kind="client"),
                                   ProviderError("HTTP 401", status=401, kind="auth"),
                                   QuotaExhausted("HTTP 429 PerDay", status=429, kind="quota")])
def test_non_transient_errors_are_not_retried(error):
    provider = ScriptedProvider([error, "{}"])
    with pytest.raises(ProviderError):
        call_with_retry(provider, REQ, sleep=lambda s: None)
    assert provider.calls == 1


# --- cache, throttle, budget, log ------------------------------------------------------------------

def make_client(tmp_path, provider, **kw):
    sleeps = []
    clock = iter(range(0, 10_000, 1))
    client = LLMClient(provider, ResponseCache(tmp_path / "cache"), tmp_path / "calls.jsonl",
                       sleep=sleeps.append, clock=lambda: next(clock), jitter=lambda d: 0, **kw)
    return client, sleeps


def test_cache_prevents_duplicate_requests(tmp_path):
    provider = ScriptedProvider(['{"x": 1}'])
    client, _ = make_client(tmp_path, provider)
    first = client.complete(REQ, prompt_version="v1", stage="s", item_id="EP001")
    second = client.complete(REQ, prompt_version="v1", stage="s", item_id="EP001")
    assert provider.calls == 1 and not first.cache_hit and second.cache_hit
    assert second.response.text == '{"x": 1}' and first.cache_key == second.cache_key


def test_cache_survives_new_client_instance(tmp_path):
    client, _ = make_client(tmp_path, ScriptedProvider(['{"x": 1}']))
    client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    fresh_provider = ScriptedProvider([])
    client2, _ = make_client(tmp_path, fresh_provider)
    assert client2.complete(REQ, prompt_version="v1", stage="s", item_id="a").cache_hit
    assert fresh_provider.calls == 0


def test_cache_key_changes_with_prompt_model_or_request():
    base = ResponseCache.key("gemini", "m1", "v1", REQ)
    assert base != ResponseCache.key("gemini", "m1", "v2", REQ)
    assert base != ResponseCache.key("gemini", "m2", "v1", REQ)
    assert base != ResponseCache.key("groq", "m1", "v1", REQ)
    assert base != ResponseCache.key("gemini", "m1", "v1", REQ.model_copy(update={"temperature": 0.5}))
    assert base == ResponseCache.key("gemini", "m1", "v1", LLMRequest(**REQ.model_dump()))


def test_corrupted_cache_file_is_a_miss(tmp_path):
    provider = ScriptedProvider(['{"x": 1}', '{"x": 2}'])
    client, _ = make_client(tmp_path, provider)
    r = client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    client.cache.path(r.cache_key).write_text("{not json", encoding="utf-8")
    again = client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    assert not again.cache_hit and again.response.text == '{"x": 2}'


def test_bypass_cache_forces_a_call(tmp_path):
    provider = ScriptedProvider(['{"x": 1}', '{"x": 2}'])
    client, _ = make_client(tmp_path, provider)
    client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    client.bypass_cache = True
    assert client.complete(REQ, prompt_version="v1", stage="s", item_id="a").response.text == '{"x": 2}'


def test_request_budget_is_enforced_but_cache_hits_are_free(tmp_path):
    provider = ScriptedProvider(['{"a": 1}', '{"b": 2}'])
    client, _ = make_client(tmp_path, provider, max_requests=1)
    client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    assert client.complete(REQ, prompt_version="v1", stage="s", item_id="a").cache_hit
    with pytest.raises(RequestCapReached):
        client.complete(REQ, prompt_version="v2", stage="s", item_id="b")
    assert provider.calls == 1


def test_zero_budget_allows_cache_hits_only(tmp_path):
    from promopilot.pipeline import request_budget

    assert request_budget(0, 30) == 0
    assert request_budget(None, 30) == 30
    provider = ScriptedProvider(['{"a": 1}'])
    warm, _ = make_client(tmp_path, provider)
    warm.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    client, _ = make_client(tmp_path, provider, max_requests=request_budget(0, 30))
    assert client.complete(REQ, prompt_version="v1", stage="s", item_id="a").cache_hit
    with pytest.raises(RequestCapReached):
        client.complete(REQ, prompt_version="v2", stage="s", item_id="b")
    assert provider.calls == 1


def test_throttle_waits_between_real_calls(tmp_path):
    provider = ScriptedProvider(["{}", "{}"])
    client, sleeps = make_client(tmp_path, provider, min_interval_s=4)
    client.complete(REQ, prompt_version="v1", stage="s", item_id="a")
    client.complete(REQ, prompt_version="v2", stage="s", item_id="b")
    assert sleeps and 0 < sleeps[0] <= 4


def test_call_log_has_usage_and_never_the_key(tmp_path, monkeypatch):
    provider = ScriptedProvider(['{"x": 1}', transient(503), transient(503), transient(503)])
    client, _ = make_client(tmp_path, provider, max_attempts=3)
    client.complete(REQ, prompt_version="v1", stage="generate", item_id="EP001")
    client.complete(REQ, prompt_version="v1", stage="generate", item_id="EP001")
    with pytest.raises(ProviderError):
        client.complete(REQ, prompt_version="v9", stage="generate", item_id="EP002")
    lines = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(x["status"], x["cache_hit"]) for x in lines] == [("ok", False), ("ok", True), ("error", False)]
    first = lines[0]
    for field in ("model", "prompt_version", "latency_ms", "prompt_tokens", "completion_tokens",
                  "total_tokens", "hidden_tokens", "retries"):
        assert field in first
    assert first["hidden_tokens"] == 15
    assert len(lines[2]["retry_log"]) == 2
    assert FAKE_KEY not in (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
