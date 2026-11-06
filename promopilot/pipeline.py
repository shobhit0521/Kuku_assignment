"""Command-line entry point.

    python -m promopilot.pipeline parse [--input DIR] [--out DIR] [--replace]
    python -m promopilot.pipeline check-providers
    python -m promopilot.pipeline generate --episodes 1,4,6 --run pilot [--provider NAME] [--refresh] [--parsed DIR]
    python -m promopilot.pipeline validate --run pilot [--suspense-beats FILE] [--preview-generator-beats]
    python -m promopilot.pipeline judge --run pilot [--episodes 1,4] [--provider NAME] [--model NAME]
        [--prompt-version judge-v1|judge-v2] [--new-judgements all|none|IDS] [--no-reassess] [--baseline]
    python -m promopilot.pipeline rank --run pilot [--prompt-version judge-v1]

A run is bound to the parsed dataset it was generated from (recorded in its manifest); validate, judge and rank
read that dataset unless --parsed is given, and refuse a different one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import config, dataset
from .llm import (ConfigError, LLMClient, LLMRequest, ProviderError, ResponseCache, call_with_retry,
                  endpoint_identity, get_provider, provider_registry, resolve_judge)
from .parser import ParseError, parse_pack


def write_parsed(episodes, summary, out_dir: Path, record: dict | None = None) -> None:
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        (episodes_dir / f"{ep.episode_id}.json").write_text(ep.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "pack_summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    with open(out_dir / "anomalies.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["code", "severity", "episode_id", "scene_id", "block_id", "line_number", "message",
                         "details"])
        anomalies = [a for ep in episodes for a in ep.anomalies] + summary.pack_anomalies
        for a in anomalies:
            writer.writerow([a.code, a.severity.value, a.episode_id or "", a.scene_id or "", a.block_id or "",
                             a.line_number or "", a.message, json.dumps(a.details, sort_keys=True)])
    if record is not None:
        dataset.write_record(out_dir, record)


def cmd_parse(args) -> int:
    input_dir = Path(args.input) if args.input else config.input_dir_from_env()
    if input_dir is None:
        print("No input folder: pass --input or set PROMOPILOT_INPUT_DIR in .env", file=sys.stderr)
        return 2
    try:
        episodes, summary = parse_pack(input_dir)
    except ParseError as exc:
        print(f"Parse failed: {exc}", file=sys.stderr)
        return 1
    duplicates = dataset.duplicate_episode_ids(episodes)
    if duplicates:
        print("Parse refused: more than one input file has the same episode number: "
              + "; ".join(f"{eid} in {', '.join(files)}" for eid, files in sorted(duplicates.items())), file=sys.stderr)
        return 1
    out_dir = Path(args.out) if args.out else config.parsed_dir()
    try:
        archived = dataset.prepare_target(out_dir, episodes, replace=args.replace)
    except dataset.DatasetError as exc:
        print(f"Parse refused: {exc}", file=sys.stderr)
        return 1
    if archived:
        print(f"Previous parsed dataset moved to {archived}")
    record = dataset.describe(episodes, input_dir, summary.pack_name)
    write_parsed(episodes, summary, out_dir, record)
    print(f"Parsed {summary.episode_count} episodes, {summary.scene_count} scenes, "
          f"{summary.block_count} blocks -> {out_dir} (dataset {record['fingerprint'][:12]})")
    print("Block types:", summary.block_type_counts)
    print("Anomalies:", summary.anomaly_counts or "none")
    if out_dir.resolve() != config.parsed_dir().resolve():
        print(f"This is not the default parsed folder: pass --parsed {out_dir} to generate for a new run.")
    return 0


def _run_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_run_dataset(args, run_dir: Path):
    """The parsed dataset for a run: --parsed, else the folder the run was generated from, else the default.
    A run bound to a dataset refuses any other one. Returns (episodes, info) or raises DatasetError /
    FileNotFoundError."""
    bound = _run_manifest(run_dir).get("dataset") or {}
    parsed = Path(getattr(args, "parsed", None) or bound.get("parsed_dir") or config.parsed_dir())
    episodes, info = dataset.load(parsed)
    if bound.get("fingerprint") and bound["fingerprint"] != info["fingerprint"]:
        raise dataset.DatasetError(
            f"run {Path(run_dir).name} was generated from dataset {bound['fingerprint'][:12]} "
            f"({bound.get('parsed_dir')}), but {parsed} holds dataset {info['fingerprint'][:12]}")
    return episodes, info


def cmd_check_providers(args) -> int:
    """One tiny, uncached JSON request per configured provider. Sends no screenplay content."""
    settings = config.llm_settings()
    request = LLMRequest(messages=[{"role": "system", "content": "Reply with JSON only."},
                                   {"role": "user", "content": 'Return {"ok": true}'}],
                         temperature=0, max_tokens=512)
    failures = 0
    try:
        names = list(provider_registry())
    except ConfigError as exc:
        print(f"Provider configuration error: {exc}", file=sys.stderr)
        return 2
    for name in names:
        try:
            provider = get_provider(name, timeout_s=settings["timeout_s"])
        except ConfigError as exc:
            print(f"{name}: not configured ({exc})")
            failures += 1
            continue
        try:
            resp, retries = call_with_retry(provider, request, max_attempts=2,
                                            base_delay_s=settings["retry_base_delay_s"])
        except ProviderError as exc:
            print(f"{name} [{provider.model}]: FAILED {exc}")
            failures += 1
            continue
        ok = resp.text.strip().startswith("{")
        print(f"{name} [{provider.model}]: {'OK' if ok else 'UNEXPECTED REPLY'} in {resp.latency_ms} ms, "
              f"tokens prompt={resp.prompt_tokens} completion={resp.completion_tokens} total={resp.total_tokens}, "
              f"retries={len(retries)}")
        failures += 0 if ok else 1
    return 1 if failures else 0


def _parse_episode_list(text: str) -> list[int]:
    numbers = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            numbers.extend(range(int(lo), int(hi) + 1))
        elif part:
            numbers.append(int(part))
    return list(dict.fromkeys(numbers))


def request_budget(override: int | None, default: int) -> int:
    """`--max-requests 0` is a valid budget (cache-only run), so only None falls back to the default."""
    return default if override is None else override


def cmd_generate(args) -> int:
    from .generator import run_generation

    settings = config.llm_settings()
    try:
        provider = get_provider(args.provider or settings["provider"], timeout_s=settings["timeout_s"])
    except ConfigError as exc:
        print(f"Provider not configured: {exc}", file=sys.stderr)
        return 2
    run_dir = config.runs_dir() / args.run
    try:
        parsed_episodes, info = load_run_dataset(args, run_dir)
    except (FileNotFoundError, dataset.DatasetError) as exc:
        print(f"Parsed dataset not usable: {exc}", file=sys.stderr)
        return 2
    if not args.no_input_check:
        input_dir = Path(args.input) if args.input else \
            Path(info["input_dir"]) if info.get("input_dir") else config.input_dir_from_env()
        if input_dir is None:
            print("Cannot check the parsed dataset against an input pack: pass --input DIR, set "
                  "PROMOPILOT_INPUT_DIR, or pass --no-input-check", file=sys.stderr)
            return 2
        problems = dataset.check_input(parsed_episodes, input_dir)
        if problems:
            print(f"Parsed dataset {info['parsed_dir']} does not match the input pack {input_dir}:", file=sys.stderr)
            for p in problems[:20]:
                print(f"  {p}", file=sys.stderr)
            print("Re-run the parse stage (into a new folder, or with --replace) or pass --no-input-check.",
                  file=sys.stderr)
            return 2
    client = LLMClient(provider, ResponseCache(config.llm_cache_dir()), run_dir / "llm_calls.jsonl",
                       min_interval_s=settings["min_interval_s"],
                       max_requests=request_budget(args.max_requests, settings["max_requests_per_run"]),
                       max_attempts=settings["retry_max_attempts"], base_delay_s=settings["retry_base_delay_s"],
                       max_delay_s=settings["retry_max_delay_s"], bypass_cache=args.bypass_cache)
    episodes = _parse_episode_list(args.episodes)
    print(f"Generating with {provider.name} [{provider.model}] for {len(episodes)} episodes from dataset "
          f"{info['fingerprint'][:12]} ({info['parsed_dir']}) -> {run_dir}")
    try:
        manifest = run_generation(episodes, parsed_dir=Path(info["parsed_dir"]), run_dir=run_dir, client=client,
                                  n_candidates=settings["gen_candidates"], temperature=settings["gen_temperature"],
                                  max_tokens=settings["gen_max_tokens"], refresh=args.refresh)
    except (FileNotFoundError, dataset.DatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for eid in manifest["last_requested"]:
        print(f"  {eid}: {manifest['episodes'].get(eid)}")
    print(f"API requests this invocation: {manifest['requests_made_this_invocation']}, "
          f"cache hits: {manifest['cache_hits_this_invocation']}")
    if manifest.get("stopped_early"):
        print(f"Stopped early: {manifest['stopped_early']}", file=sys.stderr)
        return 3
    statuses = Counter(manifest["episodes"][eid]["status"] for eid in manifest["last_requested"])
    return 0 if set(statuses) <= {"complete"} else 1


def _dataset_or_exit(args, run_dir: Path):
    try:
        return load_run_dataset(args, run_dir)[0]
    except (FileNotFoundError, dataset.DatasetError) as exc:
        print(f"Parsed dataset not usable: {exc}", file=sys.stderr)
        return None


def cmd_validate(args) -> int:
    from .validator import load_beats_file, run_validation

    run_dir = config.runs_dir() / args.run
    if not (run_dir / "candidates").is_dir():
        print(f"No candidates in {run_dir}; run the generate stage first", file=sys.stderr)
        return 2
    episodes = _dataset_or_exit(args, run_dir)
    if episodes is None:
        return 2
    beats = load_beats_file(Path(args.suspense_beats)) if args.suspense_beats else None
    out_name = args.out_name or ("validation_preview" if args.preview_generator_beats else "validation")
    summary = run_validation(run_dir, episodes, out_name=out_name, beats=beats,
                             preview_generator_beats=args.preview_generator_beats)
    counts = Counter()
    for records in summary["episodes"].values():
        for r in records:
            counts[r["status"]] += 1
            counts[f"repair:{r['repair']}"] += 1
            print(f"  {r['candidate_id']}: {r['status']} failed={r['failed']} repair={r['repair']}"
                  + (f" -> {r['repaired_status']}" if r["repaired_status"] else ""))
    print(f"{dict(counts)} -> {run_dir / out_name}")
    return 0


def cmd_judge(args) -> int:
    import uuid

    from .judge import JudgeRun, RunLocked, judge_dir, load_selected_candidates, run_baseline, run_judging

    llm, judge = config.llm_settings(), config.judge_settings()
    prompt_version = args.prompt_version or judge["prompt_version"]
    try:
        name, model = resolve_judge(args.provider, args.model)
        provider = get_provider(name, model=model, timeout_s=llm["timeout_s"])
    except ConfigError as exc:
        print(f"Judge provider not configured: {exc}", file=sys.stderr)
        return 2
    run_dir = config.runs_dir() / args.run
    if not (run_dir / "candidates").is_dir():
        print(f"No candidates in {run_dir}; run the generate stage first", file=sys.stderr)
        return 2
    episodes = _dataset_or_exit(args, run_dir)
    if episodes is None:
        return 2
    out = judge_dir(run_dir, provider.name, provider.model, prompt_version, provider.endpoint) / \
        ("baseline" if args.baseline else (args.out_subdir or ""))
    invocation = uuid.uuid4().hex[:12]
    client = LLMClient(provider, ResponseCache(config.llm_cache_dir()), out / "llm_calls.jsonl",
                       min_interval_s=llm["min_interval_s"],
                       max_requests=request_budget(args.max_requests, judge["max_requests_per_run"]),
                       max_attempts=args.max_attempts or llm["retry_max_attempts"],
                       base_delay_s=llm["retry_base_delay_s"], max_delay_s=llm["retry_max_delay_s"],
                       once_per_key=True, log_extra={"invocation_id": invocation})
    episode_ids = [f"EP{n:03d}" for n in _parse_episode_list(args.episodes)] if args.episodes else None
    new = None if args.new_judgements in (None, "all") else \
        set() if args.new_judgements == "none" else {s.strip() for s in args.new_judgements.split(",") if s.strip()}
    print(f"Judging {run_dir.name} with {provider.name} [{provider.model}], prompt {prompt_version}, "
          f"invocation {invocation}" + (" (baseline)" if args.baseline else ""))
    common = dict(temperature=judge["temperature"], max_tokens=judge["max_tokens"], prompt_version=prompt_version,
                  new_judgements=new, reassess=not args.no_reassess, invocation_id=invocation,
                  stop_on_provider_error=args.stop_on_error)
    try:
        if args.select:
            v1_decisions = judge_dir(run_dir, provider.name, provider.model, "judge-v1", provider.endpoint) / "decisions"
            by_episode = load_selected_candidates(run_dir, [s.strip() for s in args.select.split(",") if s.strip()],
                                                  [v1_decisions])
            summary = JudgeRun(run_dir, episodes, client, out_dir=out, allow_repair=False,
                               **common).run_candidates(by_episode, {e: "selected existing candidates"
                                                                     for e in by_episode})
        elif args.baseline:
            ids = episode_ids or sorted(p.stem for p in (run_dir / "candidates").glob("EP*.json"))
            summary = run_baseline(run_dir, episodes, client, ids, **common)
        else:
            summary = run_judging(run_dir, episodes, client, refresh=args.refresh, episode_ids=episode_ids, **common)
    except RunLocked as exc:
        print(f"Refusing to start: {exc}", file=sys.stderr)
        return 4
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    counts = Counter()
    for rows in summary["episodes"].values():
        for r in rows:
            counts[r["verdict"]] += 1
            print(f"  {r['candidate_id']}: {r['verdict']} score={r['weighted_score']} hard={r['hard_rules']} "
                  f"failed_gates={r['failed_gates']}" + (f" repair={r['repair']}" if r["repair"] else "")
                  + (f" reassess={r['reassessment']}" if r["reassessment"] else "")
                  + (f" review_flags={r['review_flags']}" if r["review_flags"] else ""))
    print(f"{dict(counts)}; API requests: {summary['requests_made_this_invocation']}, "
          f"cache hits: {summary['cache_hits_this_invocation']}, resumed: {summary['resumed_judgements']} "
          f"judgements / {summary['resumed_reassessments']} reassessments")
    if summary["stopped_early"]:
        print(f"Stopped early: {summary['stopped_early']}", file=sys.stderr)
        return 3
    return 0


def _judge_target(args) -> tuple[str, str, str | None]:
    """Judge provider, model and endpoint identity, without needing an API key (no request is made)."""
    name, model = resolve_judge(getattr(args, "provider", None), getattr(args, "model", None))
    return name, model, endpoint_identity(name)


def cmd_compare_judges(args) -> int:
    from .judge import judge_dir
    from .judge_compare import write_comparison

    try:
        name, model, endpoint = _judge_target(args)
    except ConfigError as exc:
        print(f"Judge provider not configured: {exc}", file=sys.stderr)
        return 2
    run_dir = config.runs_dir() / args.run
    episodes = _dataset_or_exit(args, run_dir)
    if episodes is None:
        return 2
    base = judge_dir(run_dir, name, model, args.base, endpoint)
    new = judge_dir(run_dir, name, model, args.new, endpoint) / (args.new_subdir or "")
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    out = write_comparison(run_dir, base, new, ids, episodes, args.base, args.new)
    print(f"Comparison -> {out}")
    return 0


def cmd_rank(args) -> int:
    from .judge import judge_dir
    from .ranking import write_report

    judge = config.judge_settings()
    try:
        name, model, endpoint = _judge_target(args)
    except ConfigError as exc:
        print(f"Judge provider not configured: {exc}", file=sys.stderr)
        return 2
    run_dir = config.runs_dir() / args.run
    jdir = judge_dir(run_dir, name, model, args.prompt_version or judge["prompt_version"], endpoint)
    if not (jdir / "decisions").is_dir():
        print(f"No judge decisions in {jdir}; run the judge stage first", file=sys.stderr)
        return 2
    episodes = _dataset_or_exit(args, run_dir)
    if episodes is None:
        return 2
    out = write_report(run_dir, jdir, episodes)
    report = json.loads((out / "ranking.json").read_text(encoding="utf-8"))
    print(f"Counts: {report['counts']}")
    for s in report["shortlist"]:
        print(f"  #{s['rank']} {s['candidate_id']} {s['start']}-{s['end']} score={s['score']} {s['archetype']}")
    for eid, e in report["episodes"].items():
        print(f"  {eid}: {e['status']} best={e['best']} ({e['best_score']})" + (" incomplete" if e["incomplete"] else ""))
    print(f"Reports -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="promopilot")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("parse", help="parse screenplay files into structured JSON")
    p.add_argument("--input", help="folder of episode .txt files (default: PROMOPILOT_INPUT_DIR)")
    p.add_argument("--out", help="parsed dataset folder (default: PROMOPILOT_PARSED_DIR or "
                                 "PROMOPILOT_OUTPUT_DIR/parsed); later stages read it with --parsed")
    p.add_argument("--replace", action="store_true",
                   help="if --out holds a different dataset, move it to <out>/history/ first (nothing is deleted)")
    p.set_defaults(func=cmd_parse)

    def add_parsed(p):
        p.add_argument("--parsed", help="parsed dataset folder (default: the folder this run was generated from, "
                                        "else PROMOPILOT_PARSED_DIR or PROMOPILOT_OUTPUT_DIR/parsed)")

    p = sub.add_parser("check-providers", help="one minimal request per configured provider")
    p.set_defaults(func=cmd_check_providers)

    p = sub.add_parser("generate", help="LLM candidate generation (block IDs only, grounded by code)")
    p.add_argument("--episodes", required=True, help="e.g. 1,4,6 or 1-10")
    p.add_argument("--run", required=True, help="run name; outputs go to PROMOPILOT_OUTPUT_DIR/runs/<run>")
    p.add_argument("--provider", help="override LLM_PROVIDER for this run")
    p.add_argument("--refresh", action="store_true",
                   help="re-ground completed episodes (cached responses are still reused)")
    p.add_argument("--bypass-cache", action="store_true", help="force new API calls (costs quota)")
    p.add_argument("--max-requests", type=int, help="override PROMOPILOT_MAX_REQUESTS_PER_RUN")
    add_parsed(p)
    p.add_argument("--input", help="input pack to check the parsed dataset against (default: the folder it was "
                                   "parsed from, else PROMOPILOT_INPUT_DIR)")
    p.add_argument("--no-input-check", action="store_true",
                   help="skip comparing the parsed dataset with the input pack's file hashes")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="deterministic hard rules and conservative repair (no API calls)")
    p.add_argument("--run", required=True, help="generation run name under PROMOPILOT_OUTPUT_DIR/runs")
    p.add_argument("--suspense-beats", help="JSON file of judge/human-supplied beats per candidate ID")
    p.add_argument("--preview-generator-beats", action="store_true",
                   help="provisional what-if: use the generator's cliffhanger as an advisory suspense beat")
    p.add_argument("--out-name", help="output folder name inside the run (default: validation[_preview])")
    add_parsed(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("judge", help="blind quality judge + Promo DNA on a generation run")
    p.add_argument("--run", required=True, help="generation run name under PROMOPILOT_OUTPUT_DIR/runs")
    p.add_argument("--episodes", help="limit to these episodes, e.g. 1,4")
    p.add_argument("--provider", help="override PROMOPILOT_JUDGE_PROVIDER")
    p.add_argument("--model", help="override PROMOPILOT_JUDGE_MODEL")
    p.add_argument("--refresh", action="store_true", help="re-judge even if a judgement file exists")
    p.add_argument("--max-requests", type=int, help="override PROMOPILOT_JUDGE_MAX_REQUESTS_PER_RUN")
    p.add_argument("--max-attempts", type=int, help="HTTP attempts per request (default PROMOPILOT_RETRY_MAX_ATTEMPTS)")
    p.add_argument("--prompt-version", help="judge-v1 or judge-v2 (default PROMOPILOT_JUDGE_PROMPT_VERSION)")
    p.add_argument("--new-judgements", help="'all' (default), 'none', or candidate IDs allowed new judge calls")
    p.add_argument("--no-reassess", action="store_true", help="flag contradictions but make no reassessment calls")
    p.add_argument("--baseline", action="store_true", help="judge the last-30-seconds baseline instead")
    p.add_argument("--select", help="judge only these existing candidate IDs (repairs taken from v1 decisions); "
                                    "no repair step")
    p.add_argument("--out-subdir", help="sub-folder of the judge output directory (use with --select)")
    p.add_argument("--stop-on-error", action="store_true", help="stop at the first provider error")
    add_parsed(p)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("compare-judges", help="compare two judge outputs on the same candidates (no API calls)")
    p.add_argument("--run", required=True)
    p.add_argument("--ids", required=True, help="candidate IDs to compare")
    p.add_argument("--base", default="judge-v1", help="prompt version of the reference judge output")
    p.add_argument("--new", default="judge-v2", help="prompt version of the judge output under test")
    p.add_argument("--new-subdir", default="rubric_validation", help="sub-folder of the new judge output")
    p.add_argument("--provider", help="judge provider (default PROMOPILOT_JUDGE_PROVIDER)")
    p.add_argument("--model", help="judge model (default: see the judge model precedence in the README)")
    add_parsed(p)
    p.set_defaults(func=cmd_compare_judges)

    p = sub.add_parser("rank", help="rank eligible clips and write JSON/CSV/Markdown reports (no API calls)")
    p.add_argument("--run", required=True, help="generation run name under PROMOPILOT_OUTPUT_DIR/runs")
    p.add_argument("--provider", help="judge provider (default PROMOPILOT_JUDGE_PROVIDER)")
    p.add_argument("--model", help="judge model (default: see the judge model precedence in the README)")
    p.add_argument("--prompt-version", help="judge prompt version whose outputs to rank")
    add_parsed(p)
    p.set_defaults(func=cmd_rank)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
