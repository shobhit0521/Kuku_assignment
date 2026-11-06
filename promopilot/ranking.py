"""Stage 5: ranking, duplicate suppression, diversity and reporting over judge decisions. No API calls.

Only clips that passed the deterministic hard rules, have a valid judgement and passed every quality gate
(verdict "eligible") are ranked. not_judged clips stay unranked and are reported separately; they are not
quality failures. Scores are one model's opinion, not ground truth.
"""
from __future__ import annotations

import csv
import io
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .judge import WEIGHTED, _archive, effective_judgement
from .schemas import Judgement, QualityDecision, Reassessment
from .timecode import format_timestamp

OVERLAP_LIMIT = 0.5
MAX_PER_ARCHETYPE = 3
UNCAPPED_ARCHETYPES = {"unclassified"}
EQUAL_WEIGHTS = {k: 1 / len(WEIGHTED) for k in WEIGHTED}


# --- loading ----------------------------------------------------------------------------------------

def load_decisions(directory: Path) -> list[QualityDecision]:
    return [QualityDecision.model_validate(d) for p in sorted(Path(directory).glob("EP*.json"))
            for d in json.loads(p.read_text(encoding="utf-8"))["decisions"]]


def load_effective_judgements(judge_dir: Path) -> dict[str, Judgement]:
    out = {}
    for p in sorted((Path(judge_dir) / "judgements").glob("*.json")):
        j = Judgement.model_validate_json(p.read_text(encoding="utf-8"))
        r_path = Path(judge_dir) / "reassessments" / p.name
        rec = Reassessment.model_validate_json(r_path.read_text(encoding="utf-8")) if r_path.exists() else None
        out[j.candidate_id] = effective_judgement(j, rec)
    return out


# --- ranking ----------------------------------------------------------------------------------------

def score(d: QualityDecision, weights: dict) -> float:
    return round(sum(weights[k] * d.dimension_scores[k] for k in WEIGHTED), 3)


def overlap_ratio(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Shared time as a fraction of the shorter clip."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return inter / shorter if shorter > 0 else 0.0


def _span(d: QualityDecision) -> tuple[int, int]:
    return d.validation.start_ms, d.validation.end_ms


def rank(decisions: list[QualityDecision], *, weights: Optional[dict] = None, overlap_limit: float = OVERLAP_LIMIT,
         max_per_archetype: int = MAX_PER_ARCHETYPE) -> dict:
    """Order eligible clips, suppress near-duplicates within an episode, then select at most one clip per
    episode and `max_per_archetype` per primary archetype."""
    weights = weights or config.JUDGE_WEIGHTS
    eligible = [d for d in decisions if d.verdict == "eligible" and d.eligible]

    def key(d):
        s = d.dimension_scores
        return (-score(d, weights), -s["cliffhanger"], -s["hook"], -s["context_independence"], d.candidate_id)

    rows = []
    kept: dict[str, list[dict]] = defaultdict(list)
    for d in sorted(eligible, key=key):
        row = {"candidate_id": d.candidate_id, "episode_id": d.episode_id, "score": score(d, weights),
               "archetype": d.archetype or "unclassified", "start_ms": _span(d)[0], "end_ms": _span(d)[1],
               "status": None, "suppressed_by": None, "overlap": None, "rank": None}
        dup = next(((k, overlap_ratio(_span(d), (k["start_ms"], k["end_ms"]))) for k in kept[d.episode_id]
                    if overlap_ratio(_span(d), (k["start_ms"], k["end_ms"])) > overlap_limit), None)
        if dup:
            row.update(status="suppressed_duplicate", suppressed_by=dup[0]["candidate_id"], overlap=round(dup[1], 3))
        else:
            kept[d.episode_id].append(row)
        rows.append(row)

    taken_episodes, per_archetype, shortlist = set(), Counter(), []
    for row in rows:
        if row["status"]:
            continue
        capped = row["archetype"] not in UNCAPPED_ARCHETYPES and per_archetype[row["archetype"]] >= max_per_archetype
        if row["episode_id"] in taken_episodes:
            row["status"] = "not_selected_episode_cap"
        elif capped:
            row["status"] = "not_selected_archetype_cap"
        else:
            row["status"], row["rank"] = "selected", len(shortlist) + 1
            taken_episodes.add(row["episode_id"])
            per_archetype[row["archetype"]] += 1
            shortlist.append(row)
    best = {}
    for row in rows:
        if row["status"] != "suppressed_duplicate":
            best.setdefault(row["episode_id"], row)
    return {"weights": weights, "ordered": rows, "shortlist": shortlist, "best_per_episode": best}


def _spearman(a: list[str], b: list[str]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    pos = {cid: i for i, cid in enumerate(b)}
    return round(1 - 6 * sum((i - pos[cid]) ** 2 for i, cid in enumerate(a)) / (n * (n * n - 1)), 3)


def sensitivity(decisions: list[QualityDecision], primary: dict) -> dict:
    alt = rank(decisions, weights=EQUAL_WEIGHTS)
    order_p = [r["candidate_id"] for r in primary["ordered"]]
    order_e = [r["candidate_id"] for r in alt["ordered"]]
    pos_e = {cid: i + 1 for i, cid in enumerate(order_e)}
    return {
        "weights": EQUAL_WEIGHTS,
        "spearman_rho": _spearman(order_p, order_e),
        "rank_changes": [{"candidate_id": cid, "approved_rank": i + 1, "equal_weight_rank": pos_e[cid]}
                         for i, cid in enumerate(order_p) if pos_e[cid] != i + 1],
        "shortlist_approved": [r["candidate_id"] for r in primary["shortlist"]],
        "shortlist_equal_weight": [r["candidate_id"] for r in alt["shortlist"]],
        "best_per_episode_changes": {
            eid: {"approved": primary["best_per_episode"][eid]["candidate_id"],
                  "equal_weight": alt["best_per_episode"].get(eid, {}).get("candidate_id")}
            for eid in primary["best_per_episode"]
            if primary["best_per_episode"][eid]["candidate_id"] != alt["best_per_episode"].get(eid, {}).get("candidate_id")},
    }


# --- report data ------------------------------------------------------------------------------------

def _repairs(decisions: list[QualityDecision]) -> list[dict]:
    out = []
    for d in decisions:
        if d.repaired_candidate and d.repaired_candidate.repair:
            r = d.repaired_candidate.repair
            out.append({"original": d.candidate_id, "repaired": d.repaired_candidate.candidate_id,
                        "reason": d.hard_rule_failures, "original_span": [format_timestamp(r.original_start_ms),
                                                                          format_timestamp(r.original_end_ms)],
                        "new_span": [format_timestamp(r.new_start_ms), format_timestamp(r.new_end_ms)],
                        "actions": [a.action for a in r.actions], "suspense_beat": r.suspense_beat_id,
                        "beat_source": r.suspense_beat_source})
    return out


def _api_usage(log_paths: list[Path]) -> dict:
    rows = []
    for p in log_paths:
        if p.exists():
            rows += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    live_ok = [r for r in rows if r["status"] == "ok" and not r.get("cache_hit")]
    lat = sorted(r["latency_ms"] for r in live_ok if r.get("latency_ms"))
    return {
        "log_rows": len(rows),
        "by_stage_status": dict(Counter(f"{r['stage']}:{r['status']}" for r in rows)),
        "cache_hits": sum(bool(r.get("cache_hit")) for r in rows),
        "provider_retries": sum(len(r.get("retry_log") or []) for r in rows),
        "approx_http_attempts": sum(1 + len(r.get("retry_log") or []) for r in rows if not r.get("cache_hit")
                                    and r["status"] != "duplicate_blocked"),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in live_ok),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in live_ok),
        "hidden_tokens": sum(r.get("hidden_tokens") or 0 for r in live_ok),
        "latency_ms": {"n": len(lat), "median": statistics.median(lat) if lat else None,
                       "min": lat[0] if lat else None, "max": lat[-1] if lat else None},
        "invocations": sorted({r.get("invocation_id") or "untagged (before Phase 4C.1)" for r in rows}),
    }


def build_report(run_dir: Path, judge_dir: Path) -> dict:
    run_dir, judge_dir = Path(run_dir), Path(judge_dir)
    decisions = load_decisions(judge_dir / "decisions")
    judgements = load_effective_judgements(judge_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    requested = manifest.get("requested") or sorted({d.episode_id for d in decisions})
    primary = rank(decisions)
    sens = sensitivity(decisions, primary)
    verdicts = Counter(d.verdict for d in decisions)
    judged_ok = [d for d in decisions if d.candidate_id in judgements and judgements[d.candidate_id].status == "valid"]

    counts = {
        "total_candidates": len(decisions),
        "original_candidates": sum(d.repaired_from is None for d in decisions),
        "repaired_candidates": sum(d.repaired_from is not None for d in decisions),
        "deterministically_valid": sum(d.hard_rules == "pass" for d in decisions),
        "deterministic_pending": sum(d.hard_rules in ("pending", None) for d in decisions),
        "judged_successfully": len(judged_ok),
        "passing_quality_gates": verdicts["eligible"],
        "quality_fail": verdicts["quality_fail"],
        "excluded_technical_failure": verdicts["hard_fail"],
        "not_judged_provider_or_budget": verdicts["not_judged"],
        "judge_invalid_output": verdicts["judge_invalid"],
        "marked_for_review": sum(bool(d.review_flags) for d in decisions),
    }
    episodes = {}
    for eid in requested:
        mine = [d for d in decisions if d.episode_id == eid]
        best = primary["best_per_episode"].get(eid)
        gen_status = (manifest.get("episodes") or {}).get(eid, {}).get("status")
        if not mine:
            status = f"no candidates (generation {gen_status or 'missing'})"
        elif best:
            status = "promo found"
        else:
            status = "no strong promo"
        pending = sum(d.verdict == "not_judged" for d in mine)
        episodes[eid] = {"status": status, "candidates": len(mine), "eligible": sum(d.verdict == "eligible" for d in mine),
                         "not_judged": pending, "incomplete": pending > 0,
                         "best": best["candidate_id"] if best else None, "best_score": best["score"] if best else None,
                         "verdicts": dict(Counter(d.verdict for d in mine))}

    def dna_counts(ids, attr):
        return dict(Counter(getattr(judgements[i].promo_dna, attr).value for i in ids
                            if i in judgements and judgements[i].promo_dna))

    judged_ids = [d.candidate_id for d in judged_ok]
    eligible_ids = [r["candidate_id"] for r in primary["ordered"]]
    shortlist_ids = [r["candidate_id"] for r in primary["shortlist"]]
    dna = {scope: {a: dna_counts(ids, a) for a in ("archetype", "hook_mechanism", "cliffhanger_mechanism",
                                                    "emotional_trigger")}
           for scope, ids in (("judged", judged_ids), ("eligible", eligible_ids), ("shortlist", shortlist_ids))}

    rejections = {
        "hard_rule_failures": dict(Counter(f.split(":")[0] for d in decisions if d.verdict == "hard_fail"
                                           for f in d.hard_rule_failures)),
        "gate_failures": dict(Counter(g.gate for d in decisions if d.verdict == "quality_fail"
                                      for g in d.gates if not g.passed)),
        "soft_rejections": dict(Counter(f"{r.code}:{r.severity}" for i in judged_ids
                                        for r in judgements[i].soft_rejections)),
        "review_flags": dict(Counter(f.split(":")[1].strip() for d in decisions for f in d.review_flags)),
    }

    baseline = None
    base_dir = judge_dir / "baseline" / "decisions"
    if base_dir.is_dir():
        base = {d.episode_id: d for d in load_decisions(base_dir)}
        baseline = {}
        for eid, b in sorted(base.items()):
            best = primary["best_per_episode"].get(eid)
            baseline[eid] = {"baseline_id": b.candidate_id, "verdict": b.verdict, "hard_rules": b.hard_rules,
                             "hard_rule_failures": [f.split(":")[0] for f in b.hard_rule_failures],
                             "failed_gates": [g.gate for g in b.gates if not g.passed],
                             "score": score(b, config.JUDGE_WEIGHTS) if b.dimension_scores else None,
                             "dimension_scores": b.dimension_scores,
                             "best_candidate": best["candidate_id"] if best else None,
                             "best_candidate_score": best["score"] if best else None}

    by_id = {d.candidate_id: d for d in decisions}
    table = []
    for d in decisions:
        row = next((r for r in primary["ordered"] if r["candidate_id"] == d.candidate_id), None)
        alt_rank = next((x["equal_weight_rank"] for x in sens["rank_changes"] if x["candidate_id"] == d.candidate_id),
                        None)
        table.append({
            "candidate_id": d.candidate_id, "episode_id": d.episode_id, "repaired_from": d.repaired_from,
            "start": format_timestamp(d.validation.start_ms) if d.validation and d.validation.start_ms is not None else "",
            "end": format_timestamp(d.validation.end_ms) if d.validation and d.validation.end_ms is not None else "",
            "verdict": d.verdict, "hard_rules": d.hard_rules, "weighted_score": d.weighted_score,
            "equal_weight_score": score(d, EQUAL_WEIGHTS) if d.dimension_scores else None,
            **{k: d.dimension_scores.get(k) for k in ("hook", "narrative_tension", "cliffhanger",
                                                      "context_independence", "curiosity_gap", "emotional_impact")},
            "archetype": d.archetype, "ranking_status": row["status"] if row else None,
            "shortlist_rank": row["rank"] if row else None, "suppressed_by": row["suppressed_by"] if row else None,
            "equal_weight_rank_if_changed": alt_rank,
            "failed_gates": ";".join(g.gate for g in d.gates if not g.passed),
            "hard_rule_failures": ";".join(f.split(":")[0] for f in d.hard_rule_failures),
            "soft_rejections": ";".join(d.soft_rejection_codes), "review_flags": " | ".join(d.review_flags),
            "reassessment": d.reassessment_status, "note": d.note or "",
        })
    shortlist = []
    for r in primary["shortlist"]:
        d, j = by_id[r["candidate_id"]], judgements.get(r["candidate_id"])
        dims = j.dimensions if j else {}
        shortlist.append({**r, "start": format_timestamp(r["start_ms"]), "end": format_timestamp(r["end_ms"]),
                          "duration_s": round((r["end_ms"] - r["start_ms"]) / 1000, 2),
                          "hook_block_id": dims["hook"].block_id if dims else None,
                          "suspense_beat_id": dims["cliffhanger"].suspense_beat_id if dims else None,
                          "cliffhanger_mechanism": dims["cliffhanger"].mechanism if dims else None,
                          "appeal": j.promo_dna.appeal if j and j.promo_dna else None,
                          "dimension_scores": d.dimension_scores, "review_flags": d.review_flags,
                          "repaired_from": d.repaired_from})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "run": run_dir.name, "judge_dir": judge_dir.name,
        "disclaimer": "Scores come from one LLM judge (same model family as the generator). They are an initial "
                      "model-based evaluation, not objective quality.",
        "settings": {"weights": config.JUDGE_WEIGHTS, "gate_minimums": config.JUDGE_GATE_MIN,
                     "overlap_limit": OVERLAP_LIMIT, "max_per_archetype": MAX_PER_ARCHETYPE,
                     "uncapped_archetypes": sorted(UNCAPPED_ARCHETYPES), "max_per_episode": 1},
        "counts": counts, "episodes": episodes, "ranking": primary, "sensitivity": sens, "shortlist": shortlist,
        "repairs": _repairs(decisions), "rejections": rejections, "dna": dna, "baseline": baseline,
        "api_usage": _api_usage([judge_dir / "llm_calls.jsonl", judge_dir / "baseline" / "llm_calls.jsonl"]),
        "table": table,
    }


# --- writing ----------------------------------------------------------------------------------------

def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    return buf.getvalue()


def _md_table(header: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]


def render_markdown(rep: dict) -> str:
    c, L = rep["counts"], []
    L += [f"# PromoPilot ranking report: run `{rep['run']}`", "",
          f"Judge output: `{rep['judge_dir']}`. Generated {rep['generated_at'][:19]} UTC.", "",
          f"> {rep['disclaimer']}", "", "## Candidate accounting", ""]
    L += _md_table(["Category", "Count"], [
        ["Total candidates (originals + repairs)", c["total_candidates"]],
        ["Original candidates", c["original_candidates"]], ["Repaired candidates", c["repaired_candidates"]],
        ["Deterministically valid (all hard rules pass with judge beats)", c["deterministically_valid"]],
        ["Judged successfully", c["judged_successfully"]], ["Passing all quality gates (ranked)", c["passing_quality_gates"]],
        ["Quality fail", c["quality_fail"]], ["Excluded: technical (hard-rule) failure", c["excluded_technical_failure"]],
        ["Not judged (provider error or budget)", c["not_judged_provider_or_budget"]],
        ["Invalid judge output", c["judge_invalid_output"]], ["Marked for review", c["marked_for_review"]]])
    L += ["", "## Ranked shortlist (one per episode, at most "
          f"{rep['settings']['max_per_archetype']} per archetype)", ""]
    L += _md_table(["Rank", "Candidate", "Span", "Score", "Hook/Cliff/Context/Emotion", "Archetype", "Appeal", "Review"],
                   [[s["rank"], s["candidate_id"], f"{s['start']}-{s['end']} ({s['duration_s']}s)", s["score"],
                     "/".join(str(s["dimension_scores"][k]) for k in WEIGHTED), s["archetype"], s["appeal"],
                     "yes" if s["review_flags"] else ""] for s in rep["shortlist"]])
    L += ["", "## Best promo per episode", ""]
    L += _md_table(["Episode", "Result", "Best", "Score", "Candidates", "Eligible", "Not judged"],
                   [[e, v["status"] + (" (incomplete)" if v["incomplete"] else ""), v["best"], v["best_score"],
                     v["candidates"], v["eligible"], v["not_judged"]] for e, v in rep["episodes"].items()])
    L += ["", "## All eligible clips, ordered", ""]
    L += _md_table(["Order", "Candidate", "Score", "Status", "Suppressed by (overlap)"],
                   [[i + 1, r["candidate_id"], r["score"], r["status"],
                     f"{r['suppressed_by']} ({r['overlap']})" if r["suppressed_by"] else ""]
                    for i, r in enumerate(rep["ranking"]["ordered"])])
    s = rep["sensitivity"]
    L += ["", "## Weight sensitivity (equal weights 25/25/25/25 vs approved 30/25/30/15)", "",
          f"- Spearman rank correlation over eligible clips: {s['spearman_rho']}",
          f"- Shortlist (approved): {', '.join(s['shortlist_approved']) or 'none'}",
          f"- Shortlist (equal weights): {', '.join(s['shortlist_equal_weight']) or 'none'}",
          f"- Rank changes: " + (", ".join(f"{x['candidate_id']} {x['approved_rank']}->{x['equal_weight_rank']}"
                                           for x in s["rank_changes"]) or "none"),
          f"- Best-per-episode changes: " + (", ".join(f"{e}: {v['approved']}->{v['equal_weight']}"
                                                       for e, v in s["best_per_episode_changes"].items()) or "none")]
    L += ["", "## Repairs and provenance", ""]
    L += _md_table(["Original", "Repaired", "Failed rule", "Original span", "New span", "Actions", "Beat (source)"],
                   [[r["original"], r["repaired"], "; ".join(x.split(":")[0] for x in r["reason"]),
                     "-".join(r["original_span"]), "-".join(r["new_span"]), ", ".join(r["actions"]),
                     f"{r['suspense_beat']} ({r['beat_source']})"] for r in rep["repairs"]])
    L += ["", "## Rejection reasons", ""]
    for name, counts in rep["rejections"].items():
        L.append(f"- {name.replace('_', ' ')}: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items())) or "none"))
    L += ["", "## Promo DNA", ""]
    for scope, attrs in rep["dna"].items():
        L.append(f"**{scope}**")
        for attr, counts in attrs.items():
            L.append(f"- {attr}: " + (", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
                                       or "none"))
        L.append("")
    L += ["## Baseline (last ~30 s of each episode, same judge and rubric)", ""]
    if rep["baseline"]:
        L += _md_table(["Episode", "Baseline verdict", "Hard-rule failures", "Failed gates", "Baseline score",
                        "Best candidate", "Best score"],
                       [[e, b["verdict"], ", ".join(b["hard_rule_failures"]), ", ".join(b["failed_gates"]), b["score"],
                         b["best_candidate"], b["best_candidate_score"]] for e, b in rep["baseline"].items()])
    else:
        L.append("Deferred: not run for this report.")
    a = rep["api_usage"]
    L += ["", "## API usage", "",
          f"- Logged requests: {a['log_rows']} ({', '.join(f'{k} x{v}' for k, v in sorted(a['by_stage_status'].items()))})",
          f"- Provider retries: {a['provider_retries']}; approximate HTTP attempts: {a['approx_http_attempts']}; "
          f"cache hits: {a['cache_hits']}",
          f"- Tokens (successful live calls): prompt {a['prompt_tokens']}, completion {a['completion_tokens']}, "
          f"hidden {a['hidden_tokens']}",
          f"- Latency: median {a['latency_ms']['median']} ms (n={a['latency_ms']['n']}, "
          f"{a['latency_ms']['min']}-{a['latency_ms']['max']} ms)",
          "", "## Limitations", "",
          "- One judge model, from the same family as the generator: the evaluation is only partly independent.",
          "- Scores cluster at 3-4 and moved by one point between near-identical clips; treat differences below "
          "about 0.5 as noise.",
          "- Diversity caps can only choose among the candidates the generator proposed for these episodes.",
          "- Episodes marked incomplete have candidates that the provider never judged (HTTP 503).", ""]
    return "\n".join(L)


def write_report(run_dir: Path, judge_dir: Path) -> Path:
    rep = build_report(run_dir, judge_dir)
    out = Path(run_dir) / "ranking" / Path(judge_dir).name
    out.mkdir(parents=True, exist_ok=True)
    files = {"ranking.json": json.dumps(rep, indent=2, default=str), "candidates.csv": _csv(rep["table"]),
             "shortlist.csv": _csv([{k: v for k, v in s.items() if k not in ("start_ms", "end_ms")}
                                    for s in rep["shortlist"]]),
             "report.md": render_markdown(rep)}
    for name, text in files.items():
        _archive(out / name)
        tmp = out / f"{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(out / name)
    return out
