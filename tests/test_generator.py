"""Candidate generation: JSON parsing, grounding against blocks, repair, resumability. No network access."""
import json

import pytest

from promopilot.generator import (PROMPT_VERSION, ground_proposals, generate_for_episode, parse_model_json,
                                  quote_matches, render_episode, run_generation, ResponseFormatError)
from promopilot.llm import LLMClient, LLMResponse, ProviderError, QuotaExhausted, ResponseCache
from promopilot.parser import parse_episode_text

BLOCKS = [  # (type label, start, end, text) - invented content
    ("CAMERA", "00:00.00", "00:05.00", "Wide shot of a lighthouse at night."),
    ("ACTION", "00:01.00", "00:05.00", "Ines climbs the spiral stairs."),
    ("CAMERA", "00:05.00", "00:12.00", "Close-up on a stranger at the top."),
    ("STRANGER (Cold)", "00:06.00", "00:08.50", '"You were never supposed to find this place."'),
    ("INES (Shaken)", "00:09.00", "00:11.00", '"Who are you?"'),
    ("ACTION", "00:12.00", "00:16.00", "The stranger slaps the lantern from her hand."),
    ("SOUND FX", "00:12.50", "00:13.50", "Glass shatters."),
    ("STRANGER (Whispering)", "00:16.00", "00:20.00", '"Your father built this lighthouse to hide me."'),
    ("CAMERA", "00:12.00", "00:40.00", "Two-shot in the dark."),
    ("INES (Breaking)", "00:21.00", "00:24.00", '"That is impossible. He died ten years ago."'),
    ("STRANGER (Grim)", "00:25.00", "00:38.00", '"He did not die. He is downstairs."'),
    ("ACTION", "00:38.00", "00:40.00", "Ines turns toward the dark stairwell."),
]


def build_episode_text(number=3, logline="A keeper meets a stranger."):
    parts = [f"EPISODE NUMBER: Episode {number}", f"EPISODE LOGLINE: {logline}", "CONTINUITY NOTES: none", "",
             "--- SCRIPT START ---", "", "SCENE 01", "Scene heading: INT. LIGHTHOUSE – NIGHT",
             "Scene start: 00:00.00", "Scene end: 00:40.00", "", "Actions and dialogue:", ""]
    for label, s, e, text in BLOCKS:
        parts += [f"**{label}:**", f"Start: ({s}) | End: ({e})", text, ""]
    parts.append("--- SCRIPT END ---")
    return "\n".join(parts)


@pytest.fixture
def episode():
    return parse_episode_text(build_episode_text(), "episode-003.txt")


def proposal(**over):
    p = {"start_block_id": "EP003_B003", "end_block_id": "EP003_B011", "hook_block_id": "EP003_B004",
         "hook_quote": "You were never supposed to find this place", "hook_type": "threat",
         "cliffhanger_block_id": "EP003_B011", "cliffhanger_quote": "He is downstairs.",
         "unresolved_question": "Who is downstairs?", "new_viewer_context": "A woman meets a threatening man.",
         "why_it_works": "Threat up front, impossible reveal at the end."}
    p.update(over)
    return p


def codes(c):
    return [r.code for r in c.rejection_reasons]


# --- JSON parsing ----------------------------------------------------------------------------------

def test_parse_plain_and_fenced_json():
    assert parse_model_json('{"candidates": []}') == ([], [])
    assert parse_model_json('```json\n{"candidates": [1]}\n```') == ([1], ["CODE_FENCE_STRIPPED"])


@pytest.mark.parametrize("text", ["", "not json", '{"candidates": {}}', "[]", '{"other": []}', '{"candidates": [1,'])
def test_parse_rejects_unusable_json(text):
    with pytest.raises(ResponseFormatError):
        parse_model_json(text)


def test_quote_matching_is_verbatim_modulo_quotes_case_and_spacing():
    src = '"He did not die.  He is downstairs."'
    assert quote_matches("He is downstairs", src)
    assert quote_matches("“he did not die. he is downstairs.”", src)
    assert not quote_matches("He is upstairs", src)
    assert not quote_matches("", src)


# --- grounding -------------------------------------------------------------------------------------

def test_valid_proposal_is_grounded_with_times_from_source(episode):
    [c] = ground_proposals(episode, [proposal()])
    assert c.status.value == "grounded" and codes(c) == []
    assert (c.start_ms, c.end_ms, c.duration_ms) == (5000, 38000, 33000)
    assert (c.start_ts, c.end_ts) == ("00:05.00", "00:38.00")
    assert (c.hook_offset_ms, c.cliffhanger_end_gap_ms) == (1000, 0)
    assert c.hook_source_text == '"You were never supposed to find this place."'
    assert c.segment_block_ids[0] == "EP003_B003" and "EP003_B001" not in c.segment_block_ids
    assert "EP003_B012" not in c.segment_block_ids
    assert c.scene_ids == ["EP003_S01"] and c.hard_rule_validation.startswith("pending")


def test_model_supplied_timestamps_are_ignored_and_flagged(episode):
    [c] = ground_proposals(episode, [proposal(start_time="00:00.00", end_time="09:99.00")])
    assert c.status.value == "grounded" and c.start_ms == 5000
    assert any(f.startswith("MODEL_EXTRA_FIELDS_IGNORED") and "start_time" in f for f in c.flags)


@pytest.mark.parametrize("over, code", [
    ({"start_block_id": "EP003_B099"}, "INVALID_BLOCK_ID"),
    ({"hook_block_id": "EP007_B004"}, "INVALID_BLOCK_ID"),
    ({"end_block_id": "B011"}, "INVALID_BLOCK_ID"),
    ({"start_block_id": "EP003_B011", "end_block_id": "EP003_B003"}, "BOUNDARY_ORDER"),
    ({"hook_block_id": "EP003_B012"}, "HOOK_OUTSIDE_CLIP"),
    ({"cliffhanger_block_id": "EP003_B001", "cliffhanger_quote": "Wide shot"}, "CLIFFHANGER_OUTSIDE_CLIP"),
    ({"hook_quote": "Welcome to the lighthouse"}, "QUOTE_MISMATCH"),
])
def test_bad_references_are_rejected_with_reason(episode, over, code):
    [c] = ground_proposals(episode, [proposal(**over)])
    assert c.status.value == "rejected" and code in codes(c)


def test_cross_episode_id_names_the_other_episode(episode):
    [c] = ground_proposals(episode, [proposal(hook_block_id="EP007_B004")])
    assert "belongs to EP007" in c.rejection_reasons[0].message


@pytest.mark.parametrize("raw", ["just text", {"start_block_id": "EP003_B003"},
                                 dict(proposal(), start_block_id=5)])
def test_schema_invalid_candidates_are_kept_as_rejected(episode, raw):
    [c] = ground_proposals(episode, [raw])
    assert c.status.value == "rejected" and codes(c) == ["SCHEMA_INVALID"]


def test_duplicates_and_overlaps(episode):
    cs = ground_proposals(episode, [proposal(), proposal(),
                                    proposal(start_block_id="EP003_B004", end_block_id="EP003_B011")])
    assert [c.status.value for c in cs] == ["grounded", "rejected", "grounded"]
    assert codes(cs[1]) == ["DUPLICATE_PROPOSAL"]
    assert "OVERLAPS_MORE_THAN_HALF:EP003_C01" in cs[2].flags
    assert [c.candidate_id for c in cs] == ["EP003_C01", "EP003_C02", "EP003_C03"]


def test_episode_end_flag(episode):
    [c] = ground_proposals(episode, [proposal(end_block_id="EP003_B012", cliffhanger_block_id="EP003_B012",
                                              cliffhanger_quote="turns toward the dark stairwell")])
    assert "ENDS_AT_EPISODE_END" in c.flags


def test_render_is_time_ordered_and_keeps_source_text(episode):
    text = render_episode(episode, "Previous: x")
    lines = [ln for ln in text.splitlines() if ln.startswith("EP003_B")]
    assert len(lines) == len(BLOCKS)
    assert lines.index(next(ln for ln in lines if ln.startswith("EP003_B009"))) < \
        lines.index(next(ln for ln in lines if ln.startswith("EP003_B008")))
    assert any(ln.endswith('"He did not die. He is downstairs."') for ln in lines)
    assert "Previous episode summary (context only): Previous: x" in text


# --- end to end with a scripted provider ----------------------------------------------------------

class ScriptedProvider:
    def __init__(self, script, name="mock", model="mock-1"):
        self.name, self.model, self.script, self.calls = name, model, list(script), 0

    def complete(self, request):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text, finish = item if isinstance(item, tuple) else (item, "stop")
        return LLMResponse(text=text, provider=self.name, model=self.model, finish_reason=finish,
                           prompt_tokens=100, completion_tokens=50, total_tokens=400, latency_ms=5)


def client_for(tmp_path, provider, **kw):
    return LLMClient(provider, ResponseCache(tmp_path / "cache"), tmp_path / "run" / "calls.jsonl",
                     sleep=lambda s: None, jitter=lambda d: 0, **kw)


GOOD = json.dumps({"candidates": [proposal(), proposal(hook_block_id="EP003_B099")]})
GEN_KW = dict(n_candidates=5, temperature=0.2, max_tokens=4000)


def test_generate_episode_end_to_end(tmp_path, episode):
    gen = generate_for_episode(episode, None, client_for(tmp_path, ScriptedProvider([GOOD])), **GEN_KW)
    assert gen.status == "complete" and gen.prompt_version == PROMPT_VERSION
    assert [c.status.value for c in gen.candidates] == ["grounded", "rejected"]
    assert gen.calls[0]["cache_hit"] is False and gen.calls[0]["hidden_tokens"] == 250


def test_invalid_json_gets_one_repair_attempt(tmp_path, episode):
    provider = ScriptedProvider(["Sure! here are candidates", GOOD])
    gen = generate_for_episode(episode, None, client_for(tmp_path, provider), **GEN_KW)
    assert gen.status == "complete" and provider.calls == 2
    assert [c["stage"] for c in gen.calls] == ["generate", "generate_repair"]
    assert gen.response_errors[0].startswith("generate: not valid JSON")


def test_truncated_then_still_invalid_marks_episode_invalid(tmp_path, episode):
    provider = ScriptedProvider([('{"candidates": [', "length"), "still not json"])
    gen = generate_for_episode(episode, None, client_for(tmp_path, provider), **GEN_KW)
    assert gen.status == "invalid_response" and gen.candidates == []
    assert "cut off" in gen.response_errors[0] and len(gen.response_errors) == 2


def write_parsed(tmp_path, numbers):
    parsed = tmp_path / "parsed" / "episodes"
    parsed.mkdir(parents=True)
    for n in numbers:
        ep = parse_episode_text(build_episode_text(n), f"episode-{n:03d}.txt")
        (parsed / f"{ep.episode_id}.json").write_text(ep.model_dump_json(), encoding="utf-8")
    return tmp_path / "parsed"


def ok_reply(n):
    return json.dumps({"candidates": [{k: (v.replace("EP003", f"EP{n:03d}") if k.endswith("_id") else v)
                                       for k, v in proposal().items()}]})


def test_run_is_resumable_and_skips_completed_episodes(tmp_path):
    parsed = write_parsed(tmp_path, [1, 2, 3])
    run_dir = tmp_path / "run"
    provider = ScriptedProvider([ok_reply(1), QuotaExhausted("HTTP 429 PerDay", status=429, kind="quota")])
    m1 = run_generation([1, 2, 3], parsed_dir=parsed, run_dir=run_dir, client=client_for(tmp_path, provider), **GEN_KW)
    assert m1["episodes"]["EP001"]["status"] == "complete"
    assert m1["episodes"]["EP002"]["status"] == "pending" and m1["episodes"]["EP003"]["status"] == "pending"
    assert m1["stopped_early"].startswith("quota exhausted")
    assert not (run_dir / "candidates" / "EP002.json").exists()

    provider2 = ScriptedProvider([ok_reply(2), ok_reply(3)])
    m2 = run_generation([1, 2, 3], parsed_dir=parsed, run_dir=run_dir, client=client_for(tmp_path, provider2), **GEN_KW)
    assert provider2.calls == 2
    assert m2["episodes"]["EP001"] == {"status": "complete", "resumed": True, "candidates": 1}
    assert all(m2["episodes"][e]["status"] == "complete" for e in ("EP002", "EP003"))
    assert m2["stopped_early"] is None

    provider3 = ScriptedProvider([])
    client3 = client_for(tmp_path, provider3)
    m3 = run_generation([1, 2, 3], parsed_dir=parsed, run_dir=run_dir, client=client3, refresh=True, **GEN_KW)
    assert provider3.calls == 0 and client3.cache_hits == 3
    assert all(m3["episodes"][e]["status"] == "complete" for e in ("EP001", "EP002", "EP003"))


def test_model_change_invalidates_resume_but_not_other_models_cache(tmp_path):
    parsed = write_parsed(tmp_path, [1])
    run_dir = tmp_path / "run"
    run_generation([1], parsed_dir=parsed, run_dir=run_dir,
                   client=client_for(tmp_path, ScriptedProvider([ok_reply(1)], model="m1")), **GEN_KW)
    other = ScriptedProvider([ok_reply(1)], model="m2")
    run_generation([1], parsed_dir=parsed, run_dir=run_dir, client=client_for(tmp_path, other), **GEN_KW)
    assert other.calls == 1


def test_provider_error_is_recorded_and_run_continues(tmp_path):
    parsed = write_parsed(tmp_path, [1, 2])
    run_dir = tmp_path / "run"
    provider = ScriptedProvider([ProviderError("HTTP 400: bad", status=400, kind="client"), ok_reply(2)])
    m = run_generation([1, 2], parsed_dir=parsed, run_dir=run_dir, client=client_for(tmp_path, provider), **GEN_KW)
    assert m["episodes"]["EP001"]["status"] == "provider_error"
    assert m["episodes"]["EP002"]["status"] == "complete"
    saved = json.loads((run_dir / "candidates" / "EP001.json").read_text(encoding="utf-8"))
    assert "HTTP 400" in saved["response_errors"][0]


def test_missing_parsed_data_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_generation([1], parsed_dir=tmp_path / "none", run_dir=tmp_path / "run",
                       client=client_for(tmp_path, ScriptedProvider([])), **GEN_KW)
