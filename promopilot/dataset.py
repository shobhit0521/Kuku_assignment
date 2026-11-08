"""Parsed-dataset identity and isolation.

A parsed dataset is one folder (`episodes/EPnnn.json`, `pack_summary.json`, `anomalies.csv`, `dataset.json`).
`dataset.json` records which source files, with which hashes, the episodes came from; its fingerprint binds a
run to the dataset it was generated from. Loading refuses mixed data: files whose name and content disagree,
duplicate episode IDs, and episodes the dataset record or pack summary does not list. Nothing is deleted:
replacing a dataset moves the old one into `history/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .schemas import Episode

DATASET_FILE = "dataset.json"
_DATASET_PARTS = ("episodes", "pack_summary.json", "anomalies.csv", DATASET_FILE)


class DatasetError(ValueError):
    """The parsed dataset is inconsistent, stale, or not the one a run or input pack expects."""


def fingerprint(entries: dict[str, tuple[str, str]], parser_version: str) -> str:
    """entries: episode_id -> (source_file, source_sha256)."""
    blob = json.dumps({"parser_version": parser_version, "episodes": sorted([k, *v] for k, v in entries.items())},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _entries(episodes) -> dict[str, tuple[str, str]]:
    return {e.episode_id: (e.source_file, e.source_sha256) for e in episodes}


def describe(episodes, input_dir: Path, pack_name: str) -> dict:
    entries = _entries(episodes)
    return {"schema": 1, "fingerprint": fingerprint(entries, config.PARSER_VERSION),
            "parser_version": config.PARSER_VERSION, "pack_name": pack_name,
            "input_dir": str(Path(input_dir).resolve()), "episode_count": len(entries),
            "episodes": {eid: {"source_file": f, "source_sha256": s} for eid, (f, s) in sorted(entries.items())},
            "created_at": datetime.now(timezone.utc).isoformat()}


def read_record(parsed_dir: Path) -> Optional[dict]:
    path = Path(parsed_dir) / DATASET_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise DatasetError(f"{path} is unreadable ({exc}); re-run the parse stage") from None


def duplicate_episode_ids(episodes) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for e in episodes:
        seen.setdefault(e.episode_id, []).append(e.source_file)
    return {eid: files for eid, files in seen.items() if len(files) > 1}


def _existing_entries(parsed_dir: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for path in sorted((Path(parsed_dir) / "episodes").glob("EP*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[data["episode_id"]] = (data["source_file"], data["source_sha256"])
        except (ValueError, KeyError):
            out[path.stem] = ("<unreadable>", "")
    return out


def prepare_target(parsed_dir: Path, episodes, *, replace: bool = False) -> Optional[Path]:
    """Check that writing `episodes` into parsed_dir cannot mix two packs. Re-parsing the same pack is allowed.
    A different existing dataset is refused unless replace=True, which moves it to history/ (never deletes).
    Returns the archive folder when something was moved."""
    parsed_dir = Path(parsed_dir)
    existing = _existing_entries(parsed_dir)
    if not existing:
        return None
    if existing == _entries(episodes):
        return None
    if not replace:
        raise DatasetError(
            f"{parsed_dir} already holds a different parsed dataset ({len(existing)} episodes; the new pack has "
            f"{len(episodes)}). Parse into a new folder with --out DIR (later stages: --parsed DIR), or pass "
            f"--replace to move the existing dataset to {parsed_dir / 'history'} first.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = parsed_dir / "history" / stamp
    archive.mkdir(parents=True)
    for part in _DATASET_PARTS:
        src = parsed_dir / part
        if src.exists():
            shutil.move(str(src), str(archive / part))
    return archive


def load(parsed_dir: Path) -> tuple[dict[int, Episode], dict]:
    """Load and check a parsed dataset. Returns ({episode_number: Episode}, info) where info has the resolved
    folder, fingerprint and whether a dataset record exists. Raises FileNotFoundError if there are no parsed
    episodes and DatasetError if the folder holds mixed, stale or duplicate data."""
    parsed_dir = Path(parsed_dir)
    files = sorted((parsed_dir / "episodes").glob("EP*.json"))
    if not files:
        raise FileNotFoundError(f"no parsed episodes in {parsed_dir / 'episodes'}; run the parse stage first")
    episodes = []
    for f in files:
        try:
            ep = Episode.model_validate_json(f.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise DatasetError(f"{f} is not a valid parsed episode ({str(exc)[:200]})") from None
        if ep.episode_id != f.stem:
            raise DatasetError(f"{f.name} contains {ep.episode_id}; the parsed folder has been altered by hand")
        episodes.append(ep)
    numbers: dict[int, list[str]] = {}
    for ep in episodes:
        numbers.setdefault(ep.episode_number, []).append(ep.episode_id)
    clashes = {n: ids for n, ids in numbers.items() if len(ids) > 1}
    if clashes:
        raise DatasetError(f"duplicate episode numbers in {parsed_dir}: {clashes}")

    found = _entries(episodes)
    record = read_record(parsed_dir)
    if record is not None:
        listed = {eid: (v["source_file"], v["source_sha256"]) for eid, v in record.get("episodes", {}).items()}
        _compare(parsed_dir, listed, found, "dataset.json")
        if record.get("fingerprint") != fingerprint(listed, record.get("parser_version", "")):
            raise DatasetError(f"{parsed_dir / DATASET_FILE} has an inconsistent fingerprint; re-run the parse stage")
    else:
        summary_path = parsed_dir / "pack_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        if summary.get("episodes"):
            listed_ids = {e["episode_id"] for e in summary["episodes"]}
            extra, missing = sorted(set(found) - listed_ids), sorted(listed_ids - set(found))
            if extra or missing:
                raise DatasetError(f"{parsed_dir} does not match its pack_summary.json: "
                                   + (f"unexpected or stale episodes {extra}" if extra else "")
                                   + ("; " if extra and missing else "")
                                   + (f"missing episodes {missing}" if missing else ""))
    parser_version = (record or {}).get("parser_version") or config.PARSER_VERSION
    info = {"parsed_dir": str(parsed_dir.resolve()), "fingerprint": fingerprint(found, parser_version),
            "episode_count": len(found), "recorded": record is not None,
            "input_dir": (record or {}).get("input_dir")}
    return {e.episode_number: e for e in episodes}, info


def _compare(parsed_dir: Path, listed: dict, found: dict, what: str) -> None:
    extra = sorted(set(found) - set(listed))
    missing = sorted(set(listed) - set(found))
    changed = sorted(e for e in set(found) & set(listed) if found[e] != listed[e])
    problems = []
    if extra:
        problems.append(f"unexpected or stale episodes not in {what}: {extra}")
    if missing:
        problems.append(f"episodes listed in {what} but missing: {missing}")
    if changed:
        problems.append(f"episodes whose source differs from {what}: {changed}")
    if problems:
        raise DatasetError(f"{parsed_dir}: " + "; ".join(problems) + ". Re-parse into a clean folder.")


def check_input(episodes: dict[int, Episode], input_dir: Path) -> list[str]:
    """Differences between a parsed dataset and the screenplay files currently in input_dir (by file name and
    SHA-256). Reads the input files; never writes."""
    from .parser import ParseError, discover_episode_files

    try:
        files = discover_episode_files(Path(input_dir))
    except ParseError as exc:
        return [str(exc)]
    current: dict[str, str] = {}
    problems = []
    for f in files:
        if f.name in current:
            problems.append(f"two input files are named {f.name}")
        current[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    parsed = {e.source_file: e.source_sha256 for e in episodes.values()}
    for name in sorted(set(parsed) - set(current)):
        problems.append(f"{name} was parsed but is not in the input folder")
    for name in sorted(set(current) - set(parsed)):
        problems.append(f"{name} is in the input folder but not in the parsed dataset")
    for name in sorted(set(parsed) & set(current)):
        if parsed[name] != current[name]:
            problems.append(f"{name} changed since it was parsed")
    return problems


def write_record(parsed_dir: Path, record: dict) -> None:
    path = Path(parsed_dir) / DATASET_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.replace(tmp, path)
