"""Presentation vocabulary: labels and explanations for verdicts, ranking statuses, gates and rules.

Pure mapping from values the pipeline already stored; nothing here decides a verdict or a rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from promopilot.judge import DIMENSIONS, WEIGHTED


@dataclass(frozen=True)
class Lane:
    key: str
    label: str
    tone: str  # ok | quality | technical | neutral | review
    explanation: str


LANES = {
    "eligible": Lane("eligible", "Eligible", "ok",
                     "Passed every technical rule and every promotional-quality gate."),
    "quality_fail": Lane("quality_fail", "Quality gate not met", "quality",
                         "Technically valid, but the judge's assessment did not meet a promotional-quality gate."),
    "technical_fail": Lane("technical_fail", "Technical failure", "technical",
                           "Broke a deterministic rule (boundaries, duration, hook window or ending). "
                           "The judge cannot override these."),
    "not_judged": Lane("not_judged", "Not judged", "neutral",
                       "The judge request did not complete (provider error or request budget). "
                       "This is not a quality result."),
    "judge_invalid": Lane("judge_invalid", "Judge output unusable", "neutral",
                          "The judge replied, but the reply failed schema or evidence checks, so no scores are used."),
    "pending": Lane("pending", "Pending evaluation", "neutral", "No judge decision exists for this candidate yet."),
    "unknown": Lane("unknown", "Unknown status", "neutral", "The stored verdict is not recognised."),
}

EPISODE_STATES = {
    "promo_found": ("Promo found", "ok"),
    "no_strong_promo": ("No strong promo", "quality"),
    "generation_pending": ("Generation pending", "neutral"),
    "evaluation_incomplete": ("Evaluation incomplete", "neutral"),
    "not_ranked": ("Not ranked yet", "neutral"),
}

DIMENSION_LABELS = {
    "hook": "Hook", "narrative_tension": "Narrative tension", "cliffhanger": "Cliffhanger",
    "context_independence": "Context independence", "curiosity_gap": "Curiosity gap",
    "emotional_impact": "Emotional impact",
}
SHORT_LABELS = {"hook": "Hook", "narrative_tension": "Tension", "cliffhanger": "Cliffhanger",
                "context_independence": "Context", "curiosity_gap": "Curiosity", "emotional_impact": "Emotion"}

GATE_LABELS = {
    "hook_score": "Hook score at or above the minimum",
    "cliffhanger_score": "Cliffhanger score at or above the minimum",
    "context_independence_score": "Context-independence score at or above the minimum",
    "hook_block_in_window": "Hook starts inside the opening window",
    "suspense_beat_in_clip": "Suspense beat lies inside the clip",
    "outcome_not_resolved": "Ending is not resolved",
}

RULE_LABELS = {
    "REFERENCES_VALID": "Block references resolve",
    "BOUNDARY_ORDER": "Start precedes end",
    "DURATION_RANGE": "Duration within range",
    "HOOK_WINDOW": "Hook inside the opening window",
    "DIALOGUE_BOUNDARY": "No dialogue cut at the boundaries",
    "ENDING_NOT_TRANSITION_ONLY": "Ending is not a bare transition",
    "SUSPENSE_BEAT_IN_CLIP": "Suspense beat inside the clip",
    "SUSPENSE_BEAT_IS_STORY_BEAT": "Suspense beat is a story beat",
    "SUSPENSE_TAIL_TOLERANCE": "Clip ends soon after the suspense beat",
}

TRACK_LABELS = {"DIALOGUE": "Dialogue", "CAMERA": "Camera", "ACTION": "Action", "SOUND_FX": "Sound"}

OUTCOME_LABELS = {"unresolved": "Unresolved", "partly_resolved": "Partly resolved", "resolved": "Resolved"}

REASSESSMENT_LABELS = {
    "resolved": "Contradiction resolved; revision accepted",
    "partly_resolved": "Partly resolved; some revisions accepted, others rejected",
    "revision_rejected": "Revision rejected; original score kept and marked for review",
    "unresolved": "Contradiction persisted; original kept and marked for review",
    "invalid_reply": "Reassessment reply unusable; original kept and marked for review",
    "provider_error": "Reassessment request failed; original kept and marked for review",
    "not_run": "Flagged, reassessment not run; marked for review",
    "not_needed": "No contradictions found",
}

_ARCHETYPES = {
    "betrayal_humiliation": "Betrayal & humiliation", "confrontation_power_shift": "Confrontation & power shift",
    "danger_threat": "Danger & threat", "mystery_revelation": "Mystery & revelation",
    "romance_devotion": "Romance & devotion", "deception_secret_identity": "Deception & secret identity",
    "vulnerability": "Vulnerability", "reversal": "Reversal", "unclassified": "Unclassified", "other": "Other",
}


def humanize(value: Optional[str]) -> str:
    if not value:
        return "—"
    if value in _ARCHETYPES:
        return _ARCHETYPES[value]
    text = value.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def lane(key: str) -> Lane:
    return LANES.get(key, LANES["unknown"])


def gate_label(gate: str) -> str:
    return GATE_LABELS.get(gate, humanize(gate))


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, humanize(rule))


def selection_note(ordered_row: Optional[dict], ranking: Optional[dict]) -> Optional[str]:
    """Why an eligible candidate is or is not on the shortlist, from the stored ranking row and settings."""
    if not ordered_row or not ranking:
        return None
    status = ordered_row.get("status")
    settings = ranking.get("settings", {})
    if status == "selected":
        return f"Shortlisted at rank {ordered_row.get('rank')}"
    if status == "not_selected_episode_cap":
        chosen = next((r["candidate_id"] for r in ranking.get("ranking", {}).get("shortlist", [])
                       if r["episode_id"] == ordered_row["episode_id"]), None)
        cap = settings.get("max_per_episode", 1)
        limit = f"the shortlist takes {cap} clip{'s' if cap != 1 else ''} per episode"
        return (f"Another candidate from {ordered_row['episode_id']} ranked higher ({chosen}); {limit}"
                if chosen else f"Another candidate from {ordered_row['episode_id']} ranked higher; {limit}")
    if status == "suppressed_duplicate":
        limit = settings.get("overlap_limit")
        overlap = ordered_row.get("overlap")
        pct = f"{overlap:.0%}" if isinstance(overlap, (int, float)) else "more than the limit"
        lim = f" (limit {limit:.0%} of the shorter clip)" if isinstance(limit, (int, float)) else ""
        return f"Near-duplicate: overlaps {pct} with higher-ranked {ordered_row.get('suppressed_by')}{lim}"
    if status == "not_selected_archetype_cap":
        cap = settings.get("max_per_archetype")
        return (f"Archetype limit reached: the shortlist already holds {cap} "
                f"{humanize(ordered_row.get('archetype'))} clips")
    return humanize(status)


def fmt_seconds(ms: Optional[int]) -> str:
    if ms is None:
        return "—"
    return f"{ms / 1000:.1f} s"


__all__ = ["DIMENSIONS", "WEIGHTED", "LANES", "Lane", "lane", "humanize", "selection_note", "gate_label",
           "rule_label", "DIMENSION_LABELS", "SHORT_LABELS", "EPISODE_STATES", "OUTCOME_LABELS",
           "REASSESSMENT_LABELS", "TRACK_LABELS", "fmt_seconds"]
