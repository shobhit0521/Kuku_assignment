"""Read-only side-by-side check of two generation runs (model comparison experiments).

Applies the agreed hard rules identically to both runs and writes a shuffled, model-blind scoring
sheet. Nothing in the runs is modified. Boundary expansion is not performed: dialogue lines cut by a
boundary are reported, and whether expansion would still pass is shown for information only.

    python scripts/compare_runs.py --runs pilot pilot_lite --name flash_vs_lite
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promopilot import config  # noqa: E402
from promopilot.generator import load_parsed_episodes, quote_matches  # noqa: E402
from promopilot.schemas import BlockType, Candidate, Episode, EpisodeGeneration  # noqa: E402

MIN_MS, MAX_MS = 20_000, 60_000
HOOK_WINDOW_MS = 3_000
CLIFF_TOLERANCE_MS = 2_000


def straddling_dialogue(ep: Episode, start: int, end: int) -> list[str]:
    return [b.block_id for b in ep.blocks_intersecting(start, end)
            if b.block_type == BlockType.DIALOGUE and (b.start_ms < start or b.end_ms > end)]


def timing_checks(start: int, end: int, hook_start: int, cliff_end: int) -> dict:
    return {"duration_20_60": MIN_MS <= end - start <= MAX_MS,
            "hook_within_3s": 0 <= hook_start - start <= HOOK_WINDOW_MS,
            "cliff_within_tolerance": 0 <= end - cliff_end <= CLIFF_TOLERANCE_MS}


def check_candidate(ep: Episode, c: Candidate) -> dict:
    row = {"candidate_id": c.candidate_id, "status": c.status.value, "flags": c.flags,
           "rejection_codes": [r.code for r in c.rejection_reasons]}
    ids = [c.start_block_id, c.end_block_id, c.hook_block_id, c.cliffhanger_block_id]
    row["ids_exist_in_episode"] = all(i and ep.has_block(i) and i.startswith(ep.episode_id) for i in ids)
    if c.status.value != "grounded" or not row["ids_exist_in_episode"]:
        row["hard_pass"] = False
        return row
    hook, cliff = ep.get_block(c.hook_block_id), ep.get_block(c.cliffhanger_block_id)
    first, last = ep.get_block(c.start_block_id), ep.get_block(c.end_block_id)
    row["times_from_blocks"] = (c.start_ms == first.start_ms and c.end_ms == last.end_ms)
    row["text_from_source"] = (c.hook_source_text == hook.text and c.cliffhanger_source_text == cliff.text
                               and quote_matches(c.proposal.get("hook_quote", ""), hook.text)
                               and quote_matches(c.proposal.get("cliffhanger_quote", ""), cliff.text))
    row.update(timing_checks(c.start_ms, c.end_ms, hook.start_ms, cliff.end_ms))
    row.update({"duration_s": c.duration_ms / 1000, "hook_offset_s": (hook.start_ms - c.start_ms) / 1000,
                "cliff_gap_s": (c.end_ms - cliff.end_ms) / 1000,
                "hook_block_type": hook.block_type.value, "cliff_block_type": cliff.block_type.value})
    cut = straddling_dialogue(ep, c.start_ms, c.end_ms)
    row["dialogue_cut_by_boundary"] = cut
    if cut:
        blocks = [ep.get_block(i) for i in cut]
        s = min([c.start_ms] + [b.start_ms for b in blocks])
        e = max([c.end_ms] + [b.end_ms for b in blocks])
        row["after_expansion_info"] = {"start_ms": s, "end_ms": e,
                                       "passes": all(timing_checks(s, e, hook.start_ms, cliff.end_ms).values())}
    row["hard_pass"] = all(row[k] for k in ("ids_exist_in_episode", "times_from_blocks", "text_from_source",
                                            "duration_20_60", "hook_within_3s", "cliff_within_tolerance"))
    return row


def clip_text(ep: Episode, c: Candidate) -> list[str]:
    return [f"{b.block_type.value[:4]} {b.label_raw}: {b.text}" for b in ep.blocks_intersecting(c.start_ms, c.end_ms)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    episodes = load_parsed_episodes(config.parsed_dir())
    out = config.output_dir() / "comparisons" / args.name
    out.mkdir(parents=True, exist_ok=True)
    report, sheet = {"runs": {}}, []
    for run in args.runs:
        run_dir = config.runs_dir() / run
        rows = {}
        for f in sorted((run_dir / "candidates").glob("EP*.json")):
            gen = EpisodeGeneration.model_validate_json(f.read_text(encoding="utf-8"))
            ep = episodes[int(gen.episode_id[2:])]
            rows[gen.episode_id] = {"status": gen.status, "model": gen.model, "prompt_version": gen.prompt_version,
                                    "prompt_sha256": gen.prompt_sha256, "calls": gen.calls,
                                    "candidates": [check_candidate(ep, c) for c in gen.candidates]}
            for c in gen.candidates:
                if c.status.value == "grounded":
                    sheet.append({"run": run, "candidate_id": c.candidate_id, "episode_id": gen.episode_id,
                                  "duration_s": c.duration_ms / 1000, "hook_block": c.hook_block_id,
                                  "cliff_block": c.cliffhanger_block_id,
                                  "unresolved_question": c.unresolved_question, "clip": clip_text(ep, c)})
        report["runs"][run] = rows
    random.Random(args.seed).shuffle(sheet)
    key = {}
    for i, item in enumerate(sheet, 1):
        item["blind_id"] = f"X{i:02d}"
        key[item["blind_id"]] = {"run": item.pop("run"), "candidate_id": item.pop("candidate_id")}
    (out / "hard_rules.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "blind_sheet.json").write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    for run, rows in report["runs"].items():
        cands = [c for r in rows.values() for c in r["candidates"]]
        passed = sum(c["hard_pass"] for c in cands)
        print(f"\n== {run}: {len(cands)} candidates, {passed} pass hard rules")
        for eid, r in rows.items():
            for c in r["candidates"]:
                fails = [k for k in ("ids_exist_in_episode", "times_from_blocks", "text_from_source",
                                     "duration_20_60", "hook_within_3s", "cliff_within_tolerance") if not c.get(k)]
                print(f"  {c['candidate_id']} pass={c['hard_pass']} dur={c.get('duration_s')} "
                      f"hook={c.get('hook_offset_s')}({c.get('hook_block_type')}) gap={c.get('cliff_gap_s')}"
                      f"({c.get('cliff_block_type')}) cut={c.get('dialogue_cut_by_boundary')} "
                      f"exp={c.get('after_expansion_info', {}).get('passes')} fails={fails} flags={c['flags']}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
