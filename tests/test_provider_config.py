"""Provider configuration: judge model resolution, generic OpenAI-compatible providers, request capabilities and
endpoint-aware cache isolation. Environment is faked per test; no network, no real keys."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from promopilot import config
from promopilot.judge import judge_dir
from promopilot.llm import (DEFAULT_CAPABILITIES, ConfigError, LLMClient, LLMRequest, LLMResponse,
                            OpenAICompatibleProvider, ResponseCache, endpoint_identity, endpoint_label, get_provider,
                            provider_registry, request_capabilities, resolve_judge)

REQ = LLMRequest(messages=[{"role": "user", "content": "hi"}], temperature=0, max_tokens=100)
FAKE_KEY = "test-secret-key-123"
ENV_VARS = ["LLM_PROVIDER", "PROMOPILOT_PROVIDERS", "PROMOPILOT_JUDGE_PROVIDER", "PROMOPILOT_JUDGE_MODEL"] + [
    f"{p}_{s}" for p in ("GEMINI", "GROQ", "LOCAL", "LAB_2")
    for s in ("MODEL", "JUDGE_MODEL", "BASE_URL", "API_KEY", "API_KEY_ENV", "JSON_MODE", "SEND_TEMPERATURE",
              "MAX_TOKENS_PARAM")]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS + ["MY_SECRET_VAR"]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


# --- judge provider and model resolution (Fix D) ---------------------------------------------------

def test_default_is_gemini_with_its_judge_model():
    assert resolve_judge() == ("gemini", "gemini-3.5-flash-lite")
    assert (config.judge_settings()["provider"], config.judge_settings()["model"]) == ("gemini", "gemini-3.5-flash-lite")


def test_groq_without_a_model_uses_groqs_configured_model_not_geminis(clean_env):
    clean_env.setenv("GROQ_MODEL", "groq-model-x")
    assert resolve_judge("groq") == ("groq", "groq-model-x")
    clean_env.setenv("GROQ_JUDGE_MODEL", "groq-judge-y")
    assert resolve_judge("groq") == ("groq", "groq-judge-y")


def test_groq_with_no_model_at_all_is_a_clear_error():
    with pytest.raises(ConfigError, match="no judge model for provider 'groq'"):
        resolve_judge("groq")
    assert config.judge_settings("groq")["model"] is None


def test_explicit_model_wins_over_every_variable(clean_env):
    clean_env.setenv("PROMOPILOT_JUDGE_MODEL", "env-judge")
    clean_env.setenv("GEMINI_JUDGE_MODEL", "gemini-judge")
    assert resolve_judge(model="cli-model") == ("gemini", "cli-model")
    assert resolve_judge("groq", "cli-model") == ("groq", "cli-model")


def test_env_judge_provider_and_model(clean_env):
    clean_env.setenv("PROMOPILOT_JUDGE_PROVIDER", "groq")
    clean_env.setenv("PROMOPILOT_JUDGE_MODEL", "env-judge")
    clean_env.setenv("GROQ_MODEL", "groq-gen")
    assert resolve_judge() == ("groq", "env-judge")
    # the env judge model belongs to the env judge provider; another provider does not inherit it
    clean_env.setenv("GEMINI_MODEL", "gemini-gen")
    assert resolve_judge("gemini") == ("gemini", "gemini-3.5-flash-lite")
    assert config.judge_settings()["model"] == "env-judge"


def test_cli_provider_overrides_env_provider(clean_env):
    clean_env.setenv("PROMOPILOT_JUDGE_PROVIDER", "groq")
    clean_env.setenv("GROQ_MODEL", "groq-gen")
    assert resolve_judge("gemini") == ("gemini", "gemini-3.5-flash-lite")
    assert resolve_judge("gemini", "g-1") == ("gemini", "g-1")


def test_unknown_judge_provider_is_rejected():
    with pytest.raises(ConfigError, match="unknown LLM provider"):
        resolve_judge("nope")


# --- generic OpenAI-compatible providers (Fix E) ----------------------------------------------------

def test_generic_provider_from_env(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "local, lab-2")
    clean_env.setenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1/")
    clean_env.setenv("LOCAL_MODEL", "llama-test")
    clean_env.setenv("LOCAL_API_KEY", FAKE_KEY)
    reg = provider_registry()
    assert {"gemini", "groq", "local", "lab-2"} <= set(reg)
    assert reg["lab-2"]["base_url_env"] == "LAB_2_BASE_URL"
    p = get_provider("local", client=object())
    assert (p.name, p.model) == ("local", "llama-test")
    assert p.endpoint == "http://127.0.0.1:8000/v1"
    assert FAKE_KEY not in p.endpoint


def test_generic_provider_key_can_come_from_a_named_variable(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "local")
    clean_env.setenv("LOCAL_BASE_URL", "http://localhost:9000/v1")
    clean_env.setenv("LOCAL_MODEL", "m")
    clean_env.setenv("LOCAL_API_KEY_ENV", "MY_SECRET_VAR")
    with pytest.raises(ConfigError, match="MY_SECRET_VAR is not set") as err:
        get_provider("local", client=object())
    clean_env.setenv("MY_SECRET_VAR", FAKE_KEY)
    assert get_provider("local", client=object()).model == "m"
    assert FAKE_KEY not in str(err.value)


def test_generic_provider_needs_a_base_url_and_a_valid_name(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "local")
    clean_env.setenv("LOCAL_API_KEY", FAKE_KEY)
    clean_env.setenv("LOCAL_MODEL", "m")
    with pytest.raises(ConfigError, match="LOCAL_BASE_URL is not set"):
        get_provider("local", client=object())
    clean_env.setenv("PROMOPILOT_PROVIDERS", "Bad Name!")
    with pytest.raises(ConfigError, match="invalid provider name"):
        provider_registry()


def test_builtin_providers_cannot_be_redefined(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "gemini")
    assert provider_registry()["gemini"]["base_url"].startswith("https://generativelanguage")


def test_generic_provider_can_be_a_judge(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "local")
    clean_env.setenv("LOCAL_BASE_URL", "http://localhost:9000/v1")
    clean_env.setenv("LOCAL_MODEL", "local-m")
    assert resolve_judge("local") == ("local", "local-m")
    clean_env.setenv("LOCAL_JUDGE_MODEL", "local-judge")
    assert resolve_judge("local") == ("local", "local-judge")


# --- request capabilities ---------------------------------------------------------------------------

def test_default_capabilities_send_the_original_parameters():
    assert request_capabilities("gemini") == DEFAULT_CAPABILITIES
    p = OpenAICompatibleProvider("x", "http://h/v1", FAKE_KEY, "m", client=object())
    req = REQ.model_copy(update={"json_mode": True})
    assert p.request_params(req) == {"model": "m", "messages": REQ.messages, "temperature": 0, "max_tokens": 100,
                                     "response_format": {"type": "json_object"}}


def test_capabilities_from_env_change_the_request(clean_env):
    clean_env.setenv("LOCAL_JSON_MODE", "false")
    clean_env.setenv("LOCAL_SEND_TEMPERATURE", "0")
    clean_env.setenv("LOCAL_MAX_TOKENS_PARAM", "max_completion_tokens")
    caps = request_capabilities("local")
    assert caps == {"json_mode": False, "temperature": False, "max_tokens_param": "max_completion_tokens"}
    p = OpenAICompatibleProvider("local", "http://h/v1", FAKE_KEY, "m", client=object(), capabilities=caps)
    params = p.request_params(REQ.model_copy(update={"json_mode": True}))
    assert params == {"model": "m", "messages": REQ.messages, "max_completion_tokens": 100}
    clean_env.setenv("LOCAL_MAX_TOKENS_PARAM", "none")
    p.capabilities = request_capabilities("local")
    assert "max_tokens" not in p.request_params(REQ) and "max_completion_tokens" not in p.request_params(REQ)


@pytest.mark.parametrize("var, value", [("LOCAL_JSON_MODE", "maybe"), ("LOCAL_MAX_TOKENS_PARAM", "tokens")])
def test_bad_capability_values_are_rejected(clean_env, var, value):
    clean_env.setenv(var, value)
    with pytest.raises(ConfigError, match=var):
        request_capabilities("local")


class FakeOpenAI:
    """Stands in for the openai client: records the parameters of each chat-completions call."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create)))

    def _create(self, **params):
        self.calls.append(params)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, completion_tokens_details=None),
            model="m")
        return SimpleNamespace(parse=lambda: completion, headers={})


def test_provider_sends_the_capability_shaped_request(clean_env):
    clean_env.setenv("PROMOPILOT_PROVIDERS", "local")
    clean_env.setenv("LOCAL_BASE_URL", "http://localhost:9000/v1")
    clean_env.setenv("LOCAL_MODEL", "m")
    clean_env.setenv("LOCAL_API_KEY", FAKE_KEY)
    clean_env.setenv("LOCAL_MAX_TOKENS_PARAM", "max_completion_tokens")
    fake = FakeOpenAI()
    resp = get_provider("local", client=fake).complete(REQ.model_copy(update={"json_mode": True}))
    assert resp.text == '{"ok": true}'
    assert "max_completion_tokens" in fake.calls[0] and "max_tokens" not in fake.calls[0]


# --- endpoint identity and cache isolation ----------------------------------------------------------

def test_builtin_default_endpoints_have_no_identity_so_existing_keys_stay_valid():
    assert endpoint_identity("gemini") is None and endpoint_identity("groq") is None


def test_endpoint_label_drops_credentials_query_and_fragment():
    assert endpoint_label("https://user:pw@llm.example.com:8443/v1/?key=abc#x") == "https://llm.example.com:8443/v1"


def test_overridden_base_url_or_capabilities_get_an_identity(clean_env):
    clean_env.setenv("GEMINI_BASE_URL", "https://proxy.example.com/v1beta/openai/")
    assert endpoint_identity("gemini") == "https://proxy.example.com/v1beta/openai"
    clean_env.delenv("GEMINI_BASE_URL")
    clean_env.setenv("GEMINI_JSON_MODE", "false")
    assert endpoint_identity("gemini") == \
        'https://generativelanguage.googleapis.com/v1beta/openai {"json_mode": false}'
    clean_env.delenv("GEMINI_JSON_MODE")
    clean_env.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    assert endpoint_identity("gemini") is None


def test_legacy_cache_key_format_is_unchanged():
    legacy = hashlib.sha256(json.dumps({"provider": "gemini", "model": "m", "prompt_version": "v",
                                        "request": REQ.model_dump()}, sort_keys=True,
                                       ensure_ascii=False).encode("utf-8")).hexdigest()
    assert ResponseCache.key("gemini", "m", "v", REQ) == legacy
    assert ResponseCache.key("gemini", "m", "v", REQ, endpoint=None) == legacy


def test_cache_key_separates_endpoints_models_and_prompts():
    keys = {ResponseCache.key("local", "m", "v", REQ, endpoint="http://a/v1"),
            ResponseCache.key("local", "m", "v", REQ, endpoint="http://b/v1"),
            ResponseCache.key("local", "m", "v", REQ),
            ResponseCache.key("local", "m2", "v", REQ, endpoint="http://a/v1"),
            ResponseCache.key("local", "m", "v2", REQ, endpoint="http://a/v1")}
    assert len(keys) == 5


class EndpointProvider:
    def __init__(self, endpoint, name="local", model="m"):
        self.name, self.model, self.endpoint, self.calls = name, model, endpoint, 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(text=f"from {self.endpoint}", provider=self.name, model=self.model)


def test_same_name_and_model_on_another_endpoint_never_reads_the_other_cache(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    a, b = EndpointProvider("http://a/v1"), EndpointProvider("http://b/v1")
    kw = dict(min_interval_s=0, sleep=lambda s: None)
    LLMClient(a, cache, tmp_path / "a.jsonl", **kw).complete(REQ, prompt_version="v", stage="s", item_id="i")
    rb = LLMClient(b, cache, tmp_path / "b.jsonl", **kw).complete(REQ, prompt_version="v", stage="s", item_id="i")
    assert b.calls == 1 and not rb.cache_hit and rb.response.text == "from http://b/v1"
    ra = LLMClient(a, cache, tmp_path / "a.jsonl", **kw).complete(REQ, prompt_version="v", stage="s", item_id="i")
    assert ra.cache_hit and ra.response.text == "from http://a/v1"
    log = [json.loads(line) for line in (tmp_path / "b.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[0]["endpoint"] == "http://b/v1"


def test_judge_output_folder_includes_a_non_default_endpoint(tmp_path):
    assert judge_dir(tmp_path, "gemini", "m") == tmp_path / "judge" / "gemini__m"
    a = judge_dir(tmp_path, "local", "m", endpoint="http://a/v1")
    b = judge_dir(tmp_path, "local", "m", endpoint="http://b/v1")
    assert a != b and a.name.startswith("local__m__ep-") and len(a.name.split("__ep-")[1]) == 8
