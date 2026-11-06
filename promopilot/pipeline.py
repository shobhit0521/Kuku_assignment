"""Command-line entry point.

    python -m promopilot.pipeline parse [--input DIR] [--out DIR]
    python -m promopilot.pipeline check-providers
    python -m promopilot.pipeline generate --episodes 1,4,6 --run pilot [--provider NAME] [--refresh]
    python -m promopilot.pipeline validate --run pilot [--suspense-beats FILE] [--preview-generator-beats]
    python -m promopilot.pipeline judge --run pilot [--episodes 1,4] [--provider NAME] [--model NAME]
        [--prompt-version judge-v1|judge-v2] [--new-judgements all|none|IDS] [--no-reassess] [--baseline]
    python -m promopilot.pipeline rank --run pilot [--prompt-version judge-v1]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from . import config
from .llm import (PROVIDER_REGISTRY, ConfigError, LLMClient, LLMRequest, ProviderError, ResponseCache,
                  call_with_retry, get_provider)
from .parser import ParseError, parse_pack


def write_parsed(episodes, summary, out_dir: Path) -> None:
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
    out_dir = Path(args.out)
    write_parsed(episodes, summary, out_dir)
    print(f"Parsed {summary.episode_count} episodes, {summary.scene_count} scenes, "
          f"{summary.block_count} blocks -> {out_dir}")
    print("Block types:", summary.block_type_counts)
    print("Anomalies:", summary.anomaly_counts or "none")
    return 0


def cmd_check_providers(args) -> int:
    """One tiny, uncached JSON request per configured provider. Sends no screenplay content."""
    settings = config.llm_settings()
    request = LLMRequest(messages=[{"role": "system", "content": "Reply with JSON only."},
                                   {"role": "user", "content": 'Return {"ok": true}'}],
                         temperature=0, max_tokens=512)
    failures = 0
    for name in PROVIDER_REGISTRY:
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
    client = LLMClient(provider, ResponseCache(config.llm_cache_dir()), run_dir / "llm_calls.jsonl",
                       min_interval_s=settings["min_interval_s"],
                       max_requests=request_budget(args.max_requests, settings["max_requests_per_run"]),
                       max_attempts=settings["retry_max_attempts"], base_delay_s=settings["retry_base_delay_s"],
                       max_delay_s=settings["retry_max_delay_s"], bypass_cache=args.bypass_cache)
    episodes = _parse_episode_list(args.episodes)
    print(f"Generating with {provider.name} [{provider.model}] for {len(episodes)} episodes -> {run_dir}")
    try:
        manifest = run_generation(episodes, parsed_dir=config.parsed_dir(), run_dir=run_dir, client=client,
                                  n_candidates=settings["gen_candidates"], temperature=settings["gen_temperature"],
                                  max_tokens=settings["gen_max_tokens"], refresh=args.refresh)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for eid, info in manifest["episodes"].items():
        if eid in manifest["requested"]:
            print(f"  {eid}: {info}")
    print(f"API requests this invocation: {manifest['requests_made_this_invocation']}, "
          f"cache hits: {manifest['cache_hits_this_invocation']}")
    if manifest.get("stopped_early"):
        print(f"Stopped early: {manifest['stopped_early']}", file=sys.stderr)
        return 3
    statuses = Counter(info["status"] for eid, info in manifest["episodes"].items() if eid in manifest["requested"])
    return 0 if set(statuses) <= {"complete"} else 1


def cmd_validate(args) -> int:
    from .generator import load_parsed_episodes
    from .validator import load_beats_file, run_validation

    run_dir = config.runs_dir() / args.run
    if not (run_dir / "candidates").is_dir():
        print(f"No candidates in {run_dir}; run the generate stage first", file=sys.stderr)
        return 2
    beats = load_beats_file(Path(args.suspense_beats)) if args.suspense_beats else None
    out_name = args.out_name or ("validation_preview" if args.preview_generator_beats else "validation")
    summary = run_validation(run_dir, load_parsed_episodes(config.parsed_dir()), out_name=out_name, beats=beats,
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
    from .generator import load_parsed_episodes
    import uuid

    from .judge import JudgeRun, RunLocked, judge_dir, load_selected_candidates, run_baseline, run_judging

    llm, judge = config.llm_settings(), config.judge_settings()
    name, model = args.provider or judge["provider"], args.model or judge["model"]
    prompt_version = args.prompt_version or judge["prompt_version"]
    try:
        provider = get_provider(name, model=model, timeout_s=llm["timeout_s"])
    except ConfigError as exc:
        print(f"Judge provider not configured: {exc}", file=sys.stderr)
        return 2
    run_dir = config.runs_dir() / args.run
    if not (run_dir / "candidates").is_dir():
        print(f"No candidates in {run_dir}; run the generate stage first", file=sys.stderr)
        return 2
    out = judge_dir(run_dir, provider.name, provider.model, prompt_version) / \
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
    episodes = load_parsed_episodes(config.parsed_dir())
    try:
        if args.select:
            v1_decisions = judge_dir(run_dir, provider.name, provider.model, "judge-v1") / "decisions"
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


def cmd_compare_judges(args) -> int:
    from .generator import load_parsed_episodes
    from .judge import judge_dir
    from .judge_compare import write_comparison

    judge = config.judge_settings()
    run_dir = config.runs_dir() / args.run
    base = judge_dir(run_dir, judge["provider"], judge["model"], args.base)
    new = judge_dir(run_dir, judge["provider"], judge["model"], args.new) / (args.new_subdir or "")
    ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    out = write_comparison(run_dir, base, new, ids, load_parsed_episodes(config.parsed_dir()), args.base, args.new)
    print(f"Comparison -> {out}")
    return 0


def cmd_rank(args) -> int:
    from .judge import judge_dir
    from .ranking import write_report

    judge = config.judge_settings()
    run_dir = config.runs_dir() / args.run
    jdir = judge_dir(run_dir, args.provider or judge["provider"], args.model or judge["model"],
                     args.prompt_version or judge["prompt_version"])
    if not (jdir / "decisions").is_dir():
        print(f"No judge decisions in {jdir}; run the judge stage first", file=sys.stderr)
        return 2
    out = write_report(run_dir, jdir)
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
    p.add_argument("--out", default=str(config.parsed_dir()),
                   help="output folder (default: PROMOPILOT_OUTPUT_DIR/parsed)")
    p.set_defaults(func=cmd_parse)

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
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("validate", help="deterministic hard rules and conservative repair (no API calls)")
    p.add_argument("--run", required=True, help="generation run name under PROMOPILOT_OUTPUT_DIR/runs")
    p.add_argument("--suspense-beats", help="JSON file of judge/human-supplied beats per candidate ID")
    p.add_argument("--preview-generator-beats", action="store_true",
                   help="provisional what-if: use the generator's cliffhanger as an advisory suspense beat")
    p.add_argument("--out-name", help="output folder name inside the run (default: validation[_preview])")
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
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("compare-judges", help="compare two judge outputs on the same candidates (no API calls)")
    p.add_argument("--run", required=True)
    p.add_argument("--ids", required=True, help="candidate IDs to compare")
    p.add_argument("--base", default="judge-v1", help="prompt version of the reference judge output")
    p.add_argument("--new", default="judge-v2", help="prompt version of the judge output under test")
    p.add_argument("--new-subdir", default="rubric_validation", help="sub-folder of the new judge output")
    p.set_defaults(func=cmd_compare_judges)

    p = sub.add_parser("rank", help="rank eligible clips and write JSON/CSV/Markdown reports (no API calls)")
    p.add_argument("--run", required=True, help="generation run name under PROMOPILOT_OUTPUT_DIR/runs")
    p.add_argument("--provider", help="judge provider (default PROMOPILOT_JUDGE_PROVIDER)")
    p.add_argument("--model", help="judge model (default PROMOPILOT_JUDGE_MODEL)")
    p.add_argument("--prompt-version", help="judge prompt version whose outputs to rank")
    p.set_defaults(func=cmd_rank)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
