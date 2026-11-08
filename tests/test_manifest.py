"""Cumulative run manifest across batched generation invocations. Synthetic episodes and a scripted provider."""
import json

import pytest

from promopilot.dataset import DatasetError
from promopilot.generator import run_generation
from promopilot.llm import ProviderError, QuotaExhausted
from promopilot.ranking import build_report
from test_generator import GEN_KW, ScriptedProvider, client_for, ok_reply, write_parsed
from test_ranking import dec


def ids(numbers):
    return [f"EP{n:03d}" for n in numbers]


def batch(tmp_path, parsed, numbers, script=None, **kw):
    provider = ScriptedProvider([ok_reply(n) for n in numbers] if script is None else script)
    manifest = run_generation(list(numbers), parsed_dir=parsed, run_dir=tmp_path / "run",
                              client=client_for(tmp_path, provider), **GEN_KW, **kw)
    return manifest, provider


def saved(tmp_path):
    return json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))


def test_batches_accumulate_every_requested_episode(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 51))
    for numbers in (range(1, 21), range(21, 41), range(41, 51)):
        m, _ = batch(tmp_path, parsed, numbers)
        assert m["last_requested"] == ids(numbers)
    m = saved(tmp_path)
    assert m["requested"] == ids(range(1, 51))
    assert len(m["requested"]) == len(set(m["requested"]))
    assert set(m["episodes"]) == set(ids(range(1, 51)))
    assert all(v["status"] == "complete" for v in m["episodes"].values())
    assert m["dataset"]["episode_count"] == 50


def test_rerunning_a_completed_batch_changes_nothing_and_calls_nothing(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 11))
    batch(tmp_path, parsed, range(1, 6))
    batch(tmp_path, parsed, range(6, 11))
    before = {k: v["candidates"] for k, v in saved(tmp_path)["episodes"].items()}
    m, provider = batch(tmp_path, parsed, range(1, 6), script=[])
    assert provider.calls == 0
    assert m["requested"] == ids(range(1, 11)) and m["last_requested"] == ids(range(1, 6))
    assert all(m["episodes"][e] == {"status": "complete", "resumed": True, "candidates": before[e]}
               for e in ids(range(1, 6)))
    assert all(m["episodes"][e]["status"] == "complete" for e in ids(range(6, 11)))


def test_quota_stop_then_resume_keeps_completed_work(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 6))
    m, _ = batch(tmp_path, parsed, range(1, 6),
                 script=[ok_reply(1), ok_reply(2), QuotaExhausted("HTTP 429 PerDay", status=429, kind="quota")])
    assert [m["episodes"][e]["status"] for e in ids(range(1, 6))] == ["complete", "complete", "pending", "pending",
                                                                     "pending"]
    m, provider = batch(tmp_path, parsed, range(1, 6), script=[ok_reply(3), ok_reply(4), ok_reply(5)])
    assert provider.calls == 3 and m["stopped_early"] is None
    assert m["episodes"]["EP001"]["resumed"] is True and m["episodes"]["EP003"]["resumed"] is False
    assert all(m["episodes"][e]["status"] == "complete" for e in ids(range(1, 6)))
    assert m["requested"] == ids(range(1, 6))


def test_failed_episode_later_completes(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 4))
    m, _ = batch(tmp_path, parsed, range(1, 4),
                 script=[ok_reply(1), ProviderError("HTTP 400: bad", status=400, kind="client"), ok_reply(3)])
    assert m["episodes"]["EP002"]["status"] == "provider_error"
    m, provider = batch(tmp_path, parsed, [2])
    assert provider.calls == 1 and m["episodes"]["EP002"]["status"] == "complete"
    assert m["requested"] == ids(range(1, 4)) and m["last_requested"] == ["EP002"]
    assert m["episodes"]["EP001"]["status"] == m["episodes"]["EP003"]["status"] == "complete"


def test_episodes_requested_out_of_order_are_stored_in_episode_order(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 13))
    batch(tmp_path, parsed, [12, 3])
    m, _ = batch(tmp_path, parsed, [10, 3, 1])
    assert m["requested"] == ["EP001", "EP003", "EP010", "EP012"]


def test_run_refuses_a_different_parsed_dataset(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 4))
    batch(tmp_path, parsed, range(1, 4))
    other = write_parsed(tmp_path / "other", range(1, 3))
    with pytest.raises(DatasetError, match="Use a new --run name"):
        batch(tmp_path, other, [1], script=[])
    assert saved(tmp_path)["requested"] == ids(range(1, 4))


def test_consolidated_report_covers_every_requested_episode(tmp_path):
    parsed = write_parsed(tmp_path, range(1, 51))
    for numbers in (range(1, 21), range(21, 41), range(41, 51)):
        batch(tmp_path, parsed, numbers)
    jdir = tmp_path / "run" / "judge" / "fake__m"
    (jdir / "decisions").mkdir(parents=True)
    for eid in ("EP001", "EP030", "EP050"):
        d = dec(f"{eid}_C01", 5, 38)
        (jdir / "decisions" / f"{eid}.json").write_text(json.dumps({"decisions": [d.model_dump(mode="json")]}),
                                                        encoding="utf-8")
    rep = build_report(tmp_path / "run", jdir)
    assert list(rep["episodes"]) == ids(range(1, 51))
    assert rep["episodes"]["EP030"]["status"] == "promo found"
    assert rep["episodes"]["EP002"]["status"] == "no candidates (generation complete)"
