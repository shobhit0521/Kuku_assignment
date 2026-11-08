"""Compare two judge outputs (e.g. rubric judge-v1 vs judge-v2) on the same existing candidates. No API calls.

Each side is read from its own directory and never mixed into a ranking. A wider score spread is reported
as a fact, not as evidence of accuracy; evidence support is assessed separately.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .judge import DIMENSIONS, WEIGHTED, _archive, consistency_flags, effective_judgement, reassessment_status
from .schemas import Episode, Judgement, QualityDecision, Reassessment
from .validator import beat_class


def _load_side(judge_dir: Path, cid: str) -> Optional[dict]:
    jp = Path(judge_dir) / "judgements" / f"{cid}.json"
    if not jp.exists():
        return None
    raw = Judgement.model_validate_json(jp.read_text(encoding="utf-8"))
    rp = Path(judge_dir) / "reassessments" / f"{cid}.json"
    rec = Reassessment.model_validate_json(rp.read_text(encoding="utf-8")) if rp.exists() else None
    decision = None
    for p in sorted((Path(judge_dir) / "decisions").glob("EP*.json")):
        for d in json.loads(p.read_text(encoding="utf-8"))["decisions"]:
            if d["candidate_id"] == cid:
                decision = QualityDecision.model_validate(d)
    return {"raw": raw, "rec": rec, "eff": effective_judgement(raw, rec), "decision": decision}


def _side_summary(side: dict, episode: Episode) -> dict:
    raw, eff, d = side["raw"], side["eff"], side["decision"]
    if raw.status != "valid":
        return {"status": raw.status, "failure_kind": raw.failure_kind}
    dims = eff.dimensions

    def ev(name):
        ids = dims[name].evidence
        story = [b for b in ids if episode.has_block(b) and beat_class(episode.get_block(b)) == "story"]
        return {"n": len(ids), "story": len(story), "ids": ids}

    return {
        "status": raw.status, "prompt_version": raw.prompt_version,
        "raw_scores": {n: raw.dimensions[n].score for n in DIMENSIONS},
        "scores": {n: dims[n].score for n in DIMENSIONS},
        "weighted": d.weighted_score if d else None, "verdict": d.verdict if d else None,
        "hard_rules": d.hard_rules if d else None,
        "failed_gates": [g.gate for g in d.gates if not g.passed] if d else [],
        "hook_block": dims["hook"].block_id, "suspense_beat": dims["cliffhanger"].suspense_beat_id,
        "outcome": dims["cliffhanger"].outcome, "mechanism": dims["cliffhanger"].mechanism,
        "open_question": dims["cliffhanger"].open_question,
        "cliff_reason": dims["cliffhanger"].reason, "context_reason": dims["context_independence"].reason,
        "confusions": len(dims["context_independence"].confusions),
        "context_dependency": eff.promo_dna.context_dependency if eff.promo_dna else None,
        "raw_flags": [f.code for f in consistency_flags(raw, episode)],
        "reassessment": reassessment_status(raw, side["rec"]),
        "review_flags": d.review_flags if d else [],
        "evidence": {n: ev(n) for n in DIMENSIONS},
        "soft_rejections": [f"{r.code}:{r.severity}" for r in eff.soft_rejections],
        "archetype": eff.promo_dna.archetype.value if eff.promo_dna else None,
    }


def _dist(rows: list[dict], side: str) -> dict:
    out = {}
    for n in DIMENSIONS:
        vals = [r[side]["scores"][n] for r in rows if r.get(side, {}).get("status") == "valid"]
        out[n] = {"values": vals, "mean": round(statistics.mean(vals), 2) if vals else None,
                  "stdev": round(statistics.pstdev(vals), 2) if len(vals) > 1 else None,
                  "counts": dict(sorted(Counter(vals).items()))}
    return out


def compare(base_dir: Path, new_dir: Path, ids: list[str], episodes: dict[int, Episode]) -> dict:
    rows = []
    for cid in ids:
        ep = episodes[int(cid[2:5])]
        base, new = _load_side(base_dir, cid), _load_side(new_dir, cid)
        row = {"candidate_id": cid, "base": _side_summary(base, ep) if base else {"status": "missing"},
               "new": _side_summary(new, ep) if new else {"status": "missing"}}
        b, n = row["base"], row["new"]
        if b.get("status") == "valid" and n.get("status") == "valid":
            row["delta"] = {k: n["scores"][k] - b["scores"][k] for k in DIMENSIONS}
            row["gate_change"] = None if b["verdict"] == n["verdict"] else f"{b['verdict']} -> {n['verdict']}"
            row["same_hook"] = b["hook_block"] == n["hook_block"]
            row["same_beat"] = b["suspense_beat"] == n["suspense_beat"]
        rows.append(row)
    valid = [r for r in rows if "delta" in r]

    def checks(side):
        res = [r[side] for r in valid]
        return {
            "resolved_with_passing_cliffhanger": sum(s["outcome"] == "resolved" and s["scores"]["cliffhanger"] >= 3
                                                     for s in res),
            "partly_resolved_scored_5": sum(s["outcome"] == "partly_resolved" and s["scores"]["cliffhanger"] >= 5
                                            for s in res),
            "context_ge4_with_2plus_confusions": sum(s["scores"]["context_independence"] >= 4 and s["confusions"] >= 2
                                                     for s in res),
            "context_ge3_with_high_dependency": sum(s["scores"]["context_independence"] >= 3
                                                    and s["context_dependency"] == "high" for s in res),
            "raw_flags": dict(Counter(f for s in res for f in s["raw_flags"])),
            "score_5_count": sum(v == 5 for s in res for k, v in s["scores"].items() if k in WEIGHTED),
            "mean_evidence_ids": round(statistics.mean(s["evidence"][k]["n"] for s in res for k in DIMENSIONS), 2)
            if res else None,
            "non_story_evidence_ids": sum(s["evidence"][k]["n"] - s["evidence"][k]["story"] for s in res
                                          for k in DIMENSIONS),
        }

    return {"generated_at": datetime.now(timezone.utc).isoformat(), "base_dir": str(base_dir), "new_dir": str(new_dir),
            "ids": ids, "rows": rows, "distribution": {"base": _dist(valid, "base"), "new": _dist(valid, "new")},
            "consistency": {"base": checks("base"), "new": checks("new")},
            "gate_changes": {r["candidate_id"]: r["gate_change"] for r in valid if r["gate_change"]},
            "not_compared": [r["candidate_id"] for r in rows if "delta" not in r]}


def render_markdown(cmp: dict, base_label: str, new_label: str) -> str:
    L = [f"# Judge rubric comparison: {base_label} vs {new_label}", "",
         f"Generated {cmp['generated_at'][:19]} UTC. Same judge model, same existing candidates, separate outputs. "
         "Scores are model opinions; a wider spread is not by itself evidence of better accuracy.", "",
         "## Scores (effective, after any accepted reassessment)", "",
         "| Candidate | " + " | ".join(f"{d[:6]} {base_label[-2:]}->{new_label[-2:]}" for d in DIMENSIONS)
         + " | Weighted | Verdict |", "|" + "---|" * (len(DIMENSIONS) + 3)]
    for r in cmp["rows"]:
        b, n = r["base"], r["new"]
        if "delta" not in r:
            L.append(f"| {r['candidate_id']} | not compared ({b.get('status')} / {n.get('status')}) |")
            continue
        L.append(f"| {r['candidate_id']} | " + " | ".join(f"{b['scores'][d]}->{n['scores'][d]}" for d in DIMENSIONS)
                 + f" | {b['weighted']}->{n['weighted']} | {b['verdict']}->{n['verdict']} |")
    L += ["", "## Quality-gate decisions", ""]
    L += [f"- {cid}: {chg}" for cid, chg in cmp["gate_changes"].items()] or ["- no changes"]
    L += ["", "## Cliffhanger outcome and context details", "",
          "| Candidate | Outcome | Mechanism | Same beat | Confusions | Dependency | Failed gates (new) | Raw flags (new) | Reassessment (new) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in cmp["rows"]:
        if "delta" not in r:
            continue
        b, n = r["base"], r["new"]
        L.append(f"| {r['candidate_id']} | {b['outcome']}->{n['outcome']} | {b['mechanism']}->{n['mechanism']} | "
                 f"{'yes' if r['same_beat'] else 'no'} | {b['confusions']}->{n['confusions']} | "
                 f"{b['context_dependency']}->{n['context_dependency']} | {', '.join(n['failed_gates']) or '-'} | "
                 f"{', '.join(n['raw_flags']) or '-'} | {n['reassessment'] or '-'} |")
    L += ["", "## Consistency checks", "", "| Check | " + base_label + " | " + new_label + " |", "|---|---|---|"]
    for k in cmp["consistency"]["base"]:
        L.append(f"| {k} | {cmp['consistency']['base'][k]} | {cmp['consistency']['new'][k]} |")
    L += ["", "## Score distribution over these candidates", "",
          "| Dimension | " + base_label + " mean (sd) | " + new_label + " mean (sd) | " + base_label + " counts | "
          + new_label + " counts |", "|---|---|---|---|---|"]
    for d in DIMENSIONS:
        b, n = cmp["distribution"]["base"][d], cmp["distribution"]["new"][d]
        L.append(f"| {d} | {b['mean']} ({b['stdev']}) | {n['mean']} ({n['stdev']}) | {b['counts']} | {n['counts']} |")
    if cmp["not_compared"]:
        L += ["", f"Not compared: {', '.join(cmp['not_compared'])}"]
    return "\n".join(L) + "\n"


def write_comparison(run_dir: Path, base_dir: Path, new_dir: Path, ids: list[str], episodes: dict[int, Episode],
                     base_label: str, new_label: str) -> Path:
    cmp = compare(base_dir, new_dir, ids, episodes)
    out = Path(run_dir) / "comparisons" / f"{base_label}_vs_{new_label}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("comparison.json", json.dumps(cmp, indent=2, default=str)),
                       ("report.md", render_markdown(cmp, base_label, new_label))):
        _archive(out / name)
        (out / name).write_text(text, encoding="utf-8")
    return out
