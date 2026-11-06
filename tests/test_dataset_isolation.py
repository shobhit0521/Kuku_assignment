"""Fresh-input safety: separate packs never mix, every stage reads the run's own parsed dataset, and a custom parse
output works end to end. Synthetic packs in temp folders, scripted providers; the production dataset is only
hashed (read-only) to show it is untouched."""
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from promopilot import config, dataset, pipeline
from test_generator import ScriptedProvider, build_episode_text, ok_reply
from test_judge import FakeJudge

PRODUCTION_PARSED = config.parsed_dir()
CLEARED = ["PROMOPILOT_INPUT_DIR", "PROMOPILOT_PARSED_DIR", "PROMOPILOT_JUDGE_PROVIDER", "PROMOPILOT_JUDGE_MODEL",
           "PROMOPILOT_JUDGE_PROMPT_VERSION", "PROMOPILOT_PROVIDERS", "GEMINI_BASE_URL", "GEMINI_JSON_MODE",
           "GEMINI_SEND_TEMPERATURE", "GEMINI_MAX_TOKENS_PARAM", "GEMINI_JUDGE_MODEL"]


def tree_hashes(root: Path) -> dict:
    if not root.is_dir():
        return {}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    for var in CLEARED:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PROMOPILOT_MIN_REQUEST_INTERVAL_S", "0")
    monkeypatch.setenv("PROMOPILOT_RETRY_BASE_DELAY_S", "0")
    return monkeypatch


def make_pack(root: Path, numbers, tag="") -> Path:
    root.mkdir(parents=True)
    for n in numbers:
        (root / f"episode-{n:03d}.txt").write_text(build_episode_text(n, logline=f"Pack {tag} episode {n}."),
                                                   encoding="utf-8")
    return root


def parsed_ids(parsed: Path) -> list[str]:
    return sorted(p.stem for p in (parsed / "episodes").glob("EP*.json"))


class RecordingProvider(ScriptedProvider):
    def __init__(self, script):
        super().__init__(script)
        self.prompts = []

    def complete(self, request):
        self.prompts.append(json.dumps(request.messages))
        return super().complete(request)


def use_provider(env, provider):
    provider.endpoint = getattr(provider, "endpoint", None)
    env.setattr(pipeline, "get_provider", lambda *a, **k: provider)
    return provider


def generate(*extra):
    return pipeline.main(["generate", "--episodes", "1-10", "--run", "r", *extra])


def test_second_pack_cannot_leak_into_or_inherit_the_first(tmp_path, env, capsys):
    pack50, pack10 = make_pack(tmp_path / "p50", range(1, 51), "A"), make_pack(tmp_path / "p10", range(1, 11), "B")
    default = config.parsed_dir()
    assert pipeline.main(["parse", "--input", str(pack50)]) == 0
    assert parsed_ids(default) == [f"EP{n:03d}" for n in range(1, 51)]
    before = tree_hashes(default)

    assert pipeline.main(["parse", "--input", str(pack10)]) == 1
    assert "already holds a different parsed dataset" in capsys.readouterr().err
    assert tree_hashes(default) == before

    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack10), "--out", str(custom)]) == 0
    assert parsed_ids(custom) == [f"EP{n:03d}" for n in range(1, 11)]
    assert all("Pack B" in json.loads(p.read_text(encoding="utf-8"))["logline"]
               for p in (custom / "episodes").glob("*.json"))
    assert tree_hashes(default) == before

    assert pipeline.main(["parse", "--input", str(pack10), "--replace"]) == 0
    assert parsed_ids(default) == [f"EP{n:03d}" for n in range(1, 11)]
    [archive] = list((default / "history").iterdir())
    assert len(list((archive / "episodes").glob("EP*.json"))) == 50
    assert dataset.read_record(default)["episode_count"] == 10


def test_reparsing_the_same_pack_is_allowed(tmp_path, env):
    pack = make_pack(tmp_path / "p", range(1, 4))
    assert pipeline.main(["parse", "--input", str(pack)]) == 0
    assert pipeline.main(["parse", "--input", str(pack)]) == 0
    assert not (config.parsed_dir() / "history").exists()


def test_pack_with_duplicate_episode_numbers_is_refused(tmp_path, env, capsys):
    pack = make_pack(tmp_path / "p", range(1, 3))
    (pack / "episode-002-copy.txt").write_text(build_episode_text(2), encoding="utf-8")
    assert pipeline.main(["parse", "--input", str(pack)]) == 1
    assert "same episode number" in capsys.readouterr().err
    assert not config.parsed_dir().exists()


def test_generation_reads_the_custom_parsed_dataset(tmp_path, env):
    make_pack(tmp_path / "p50", range(1, 51), "A")
    pack10 = make_pack(tmp_path / "p10", range(1, 11), "B")
    assert pipeline.main(["parse", "--input", str(tmp_path / "p50")]) == 0
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack10), "--out", str(custom)]) == 0
    provider = use_provider(env, RecordingProvider([ok_reply(n) for n in range(1, 11)]))
    assert pipeline.main(["generate", "--episodes", "1-11", "--run", "r", "--parsed", str(custom)]) == 1
    m = json.loads((config.runs_dir() / "r" / "manifest.json").read_text(encoding="utf-8"))
    assert m["dataset"]["parsed_dir"] == str(custom.resolve()) and m["dataset"]["episode_count"] == 10
    assert m["episodes"]["EP011"]["status"] == "missing_parsed_episode"
    assert all(m["episodes"][f"EP{n:03d}"]["status"] == "complete" for n in range(1, 11))
    assert len(provider.prompts) == 10
    assert all("Pack B" in p and "Pack A" not in p for p in provider.prompts)


def test_stale_extra_episode_in_parsed_folder_is_detected(tmp_path, env, capsys):
    pack = make_pack(tmp_path / "p", range(1, 4))
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(custom)]) == 0
    other = make_pack(tmp_path / "o", [7])
    assert pipeline.main(["parse", "--input", str(other), "--out", str(tmp_path / "o_parsed")]) == 0
    shutil.copy(tmp_path / "o_parsed" / "episodes" / "EP007.json", custom / "episodes" / "EP007.json")
    use_provider(env, ScriptedProvider([]))
    assert generate("--parsed", str(custom)) == 2
    err = capsys.readouterr().err
    assert "unexpected or stale episodes" in err and "EP007" in err


def test_legacy_folder_without_a_record_is_checked_against_its_pack_summary(tmp_path, env):
    pack = make_pack(tmp_path / "p", range(1, 4))
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(custom)]) == 0
    (custom / dataset.DATASET_FILE).unlink()
    episodes, info = dataset.load(custom)
    assert sorted(episodes) == [1, 2, 3] and info["recorded"] is False
    (custom / "episodes" / "EP003.json").rename(custom / "episodes" / "EP003.json.bak")
    with pytest.raises(dataset.DatasetError, match="missing episodes"):
        dataset.load(custom)


def test_hand_renamed_episode_file_is_detected(tmp_path, env):
    pack = make_pack(tmp_path / "p", range(1, 3))
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(custom)]) == 0
    (custom / "episodes" / "EP002.json").rename(custom / "episodes" / "EP009.json")
    with pytest.raises(dataset.DatasetError, match="contains EP002"):
        dataset.load(custom)


def test_generation_refuses_a_dataset_that_no_longer_matches_the_input_pack(tmp_path, env, capsys):
    pack = make_pack(tmp_path / "p", range(1, 4))
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(custom)]) == 0
    (pack / "episode-002.txt").write_text(build_episode_text(2, logline="edited"), encoding="utf-8")
    (pack / "episode-004.txt").write_text(build_episode_text(4), encoding="utf-8")
    provider = use_provider(env, ScriptedProvider([ok_reply(n) for n in (1, 2, 3)]))
    assert pipeline.main(["generate", "--episodes", "1-3", "--run", "r", "--parsed", str(custom)]) == 2
    err = capsys.readouterr().err
    assert "episode-002.txt changed since it was parsed" in err
    assert "episode-004.txt is in the input folder but not in the parsed dataset" in err
    assert provider.calls == 0
    assert pipeline.main(["generate", "--episodes", "1-3", "--run", "r", "--parsed", str(custom),
                          "--no-input-check"]) == 0


def test_later_stages_refuse_a_dataset_other_than_the_runs(tmp_path, env, capsys):
    pack, other = make_pack(tmp_path / "p", range(1, 3)), make_pack(tmp_path / "o", range(1, 4))
    a, b = tmp_path / "a", tmp_path / "b"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(a)]) == 0
    assert pipeline.main(["parse", "--input", str(other), "--out", str(b)]) == 0
    use_provider(env, ScriptedProvider([ok_reply(1), ok_reply(2)]))
    assert pipeline.main(["generate", "--episodes", "1-2", "--run", "r", "--parsed", str(a)]) == 0
    assert pipeline.main(["validate", "--run", "r", "--parsed", str(b)]) == 2
    assert "was generated from dataset" in capsys.readouterr().err
    assert pipeline.main(["validate", "--run", "r"]) == 0
    use_provider(env, ScriptedProvider([]))
    assert pipeline.main(["generate", "--episodes", "3", "--run", "r", "--parsed", str(b)]) == 2


def test_custom_parse_output_works_end_to_end_and_production_is_untouched(tmp_path, env):
    production_before = tree_hashes(PRODUCTION_PARSED)
    pack = make_pack(tmp_path / "p", range(1, 4))
    custom = tmp_path / "fresh"
    assert pipeline.main(["parse", "--input", str(pack), "--out", str(custom)]) == 0
    use_provider(env, ScriptedProvider([ok_reply(n) for n in range(1, 4)]))
    assert pipeline.main(["generate", "--episodes", "1-3", "--run", "r", "--parsed", str(custom)]) == 0
    assert pipeline.main(["validate", "--run", "r"]) == 0
    judge = use_provider(env, FakeJudge(name="gemini", model="fake-judge-1"))
    assert pipeline.main(["judge", "--run", "r", "--provider", "gemini", "--model", "fake-judge-1"]) == 0
    assert judge.requests
    assert pipeline.main(["rank", "--run", "r", "--provider", "gemini", "--model", "fake-judge-1"]) == 0
    out = config.runs_dir() / "r" / "ranking" / "gemini__fake-judge-1"
    rows = list(csv.DictReader(io.StringIO((out / "catalogue.csv").read_text(encoding="utf-8"))))
    assert rows and {r["episode_id"] for r in rows} <= {"EP001", "EP002", "EP003"}
    episodes, _ = dataset.load(custom)
    by_id = {e.episode_id: e for e in episodes.values()}
    for r in rows:
        assert r["hook_text"] == by_id[r["episode_id"]].get_block(r["hook_block_id"]).text
        assert r["cliffhanger_text"] == by_id[r["episode_id"]].get_block(r["cliffhanger_block_id"]).text
    assert not config.parsed_dir().exists()
    assert tree_hashes(PRODUCTION_PARSED) == production_before
