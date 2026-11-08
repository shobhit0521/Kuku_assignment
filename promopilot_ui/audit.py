"""Audit views behind the Needs attention and Repairs pages: which candidates need attention and why, and what
each repair did. Pure functions over `RunData`; nothing here validates, judges, repairs or ranks. Every value is
read from the stored decisions, judgements, reassessments and validation records."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from promopilot import config
from promopilot.judge import provider_failed, review_outcome
from promopilot.schemas import CandidateValidation, RepairAction, RepairProvenance, RuleStatus
from promopilot.validator import REPAIRABLE_RULES, RULES
from promopilot_ui.data import CandidateView, EpisodeView, RunData, redact
from promopilot_ui.status import fmt_seconds, humanize

CATEGORIES = {
    "technical": "Technical failures",
    "quality": "Quality-gate failures",
    "unusable": "Not judged / unusable judge output",
    "pending": "Pending or incomplete processing",
    "review": "Flagged for review",
}
_LANE_CATEGORY = {"technical_fail": "technical", "quality_fail": "quality", "not_judged": "unusable",
                  "judge_invalid": "unusable", "pending": "pending", "unknown": "pending"}

REVIEW_STATUSES = {"needs_review": "Needs review", "resolved": "Resolved by reassessment", "none": "No flags"}
REPAIR_KEYS = {"repaired_original": "Original with a successful repair", "repair_version": "Repaired version",
               "repair_unresolved": "Repair attempted, not resolved", "not_attempted": "Technical failure, no repair",
               "none": "No repair involved"}
GATE_DIMENSION = {"hook_score": "hook", "hook_block_in_window": "hook", "cliffhanger_score": "cliffhanger",
                  "suspense_beat_in_clip": "cliffhanger", "outcome_not_resolved": "cliffhanger",
                  "context_independence_score": "context_independence"}
NOT_SENT_NOTE = "not sent to the judge"


# --- classification ---------------------------------------------------------------------------------

def category(view: CandidateView) -> Optional[str]:
    """The one failure category of a candidate, from its stored verdict. Eligible candidates have none; a review
    flag is a separate, overlapping marker (see `review_status`)."""
    return _LANE_CATEGORY.get(view.lane)


def review_status(view: CandidateView) -> str:
    if view.review_flags:
        return "needs_review"
    rec = view.reassessment
    return "resolved" if rec is not None and rec.flags else "none"


def archetype_key(view: CandidateView) -> str:
    return view.archetype or "not_classified"


# --- technical failures -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleFailure:
    rule: str
    message: str
    observed: str
    required: str
    repairable: Optional[bool]
    block_ids: tuple[str, ...] = ()


def settings_for(rd: RunData, episode_id: str) -> tuple[dict, str]:
    """Validation thresholds as stored by `validate` for this episode, else the current configuration."""
    stored = rd.validation_settings.get(episode_id)
    if stored:
        return stored, "stored with the validation output"
    return config.validation_settings(), "current configuration (no stored validation settings)"


def _secs(ms: Optional[int]) -> str:
    return "—" if ms is None else f"{ms / 1000:g} s"


def required(rule: str, settings: dict) -> str:
    tail = _secs(settings.get("suspense_tail_ms"))
    return {
        "REFERENCES_VALID": "Start, end and hook blocks exist in the episode and have times",
        "BOUNDARY_ORDER": "End after start",
        "DURATION_RANGE": f"{_secs(settings.get('min_clip_ms'))} to {_secs(settings.get('max_clip_ms'))}",
        "HOOK_WINDOW": f"Hook starts within {_secs(settings.get('hook_window_ms'))} of the clip start",
        "DIALOGUE_BOUNDARY": "No dialogue line cut at the start or end",
        "ENDING_NOT_TRANSITION_ONLY": f"At most {tail} of transition or music after the last story beat",
        "SUSPENSE_BEAT_IN_CLIP": "Suspense beat starts inside the clip",
        "SUSPENSE_BEAT_IS_STORY_BEAT": "Suspense beat is a story beat, not a transition or music cue",
        "SUSPENSE_TAIL_TOLERANCE": f"Clip ends at most {tail} after the suspense beat",
    }.get(rule, "See the recorded message")


def observed(rule: str, value_ms: Optional[int], block_ids, message: str) -> str:
    if rule == "DURATION_RANGE" and value_ms is not None:
        return fmt_seconds(value_ms)
    if rule == "HOOK_WINDOW" and value_ms is not None:
        return (f"starts {fmt_seconds(-value_ms)} before the clip" if value_ms < 0
                else f"starts {fmt_seconds(value_ms)} after the clip start")
    if rule == "SUSPENSE_TAIL_TOLERANCE" and value_ms is not None:
        return f"ends {fmt_seconds(value_ms)} after the suspense beat"
    if rule == "ENDING_NOT_TRANSITION_ONLY":
        return (f"{fmt_seconds(value_ms)} of transition or music after the last story beat"
                if value_ms is not None else "no story beat in the clip")
    if rule == "DIALOGUE_BOUNDARY" and block_ids:
        return "cuts " + ", ".join(block_ids)
    return message


def rule_failures(rd: RunData, view: CandidateView) -> list[RuleFailure]:
    """Failed deterministic rules from the candidate's final validation. Candidates rejected before judging have
    no stored validation, only `RULE: message` strings; those are parsed without inventing values."""
    d = view.decision
    if d is None:
        return []
    settings, _ = settings_for(rd, view.episode_id)
    if d.validation is not None and d.validation.checks:
        return [RuleFailure(c.rule, c.message, observed(c.rule, c.value_ms, c.block_ids, c.message),
                            required(c.rule, settings), c.repairable, tuple(c.block_ids))
                for c in d.validation.checks if c.status == RuleStatus.FAIL]
    out = []
    for text in d.hard_rule_failures:
        rule, _, msg = text.partition(": ")
        if rule not in RULES:
            rule, msg = "", text
        out.append(RuleFailure(rule, msg, msg, required(rule, settings) if rule else "See the recorded message",
                               (rule in REPAIRABLE_RULES) if rule else None))
    return out


# --- repairs ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RepairState:
    key: str    # repaired | repaired_still_failing | repaired_no_decision | unrepairable | not_attempted | is_repair
    label: str
    detail: str = ""
    target: Optional[str] = None


def repair_state(rd: RunData, view: CandidateView) -> Optional[RepairState]:
    """What happened to this candidate's repair, confirmed against the repaired clip's own stored decision."""
    d = view.decision
    if view.repaired_from:
        return RepairState("is_repair", f"Repair of {view.repaired_from}", target=view.repaired_from)
    if d is None:
        return None
    if d.repair_status == "repaired":
        target = view.repaired_to or (d.repaired_candidate.candidate_id if d.repaired_candidate else None)
        tv = rd.candidates.get(target) if target else None
        td = tv.decision if tv else None
        if td is None:
            return RepairState("repaired_no_decision", "Repair created; no decision recorded for it yet",
                               d.repair_reason or "", target)
        if td.hard_rules == "pass" and td.validation is not None and td.validation.status == RuleStatus.PASS:
            return RepairState("repaired", "Repaired; every rule passes after revalidation",
                               d.repair_reason or "", target)
        failed = ", ".join(f.split(":")[0] for f in td.hard_rule_failures) or "hard rules not confirmed"
        return RepairState("repaired_still_failing", "Repair created, but it still fails", failed, target)
    if d.repair_status == "unrepairable":
        return RepairState("unrepairable", "Repair attempted; not possible", d.repair_reason or "")
    if view.lane != "technical_fail":
        return None
    if d.note and NOT_SENT_NOTE in d.note:
        return RepairState("not_attempted", "Not attempted", "objective failure no repair can fix; not sent to "
                                                             "the judge")
    if view.raw is None or view.raw.status != "valid":
        return RepairState("not_attempted", "Not attempted", "repairs need a valid judgement for the suspense beat")
    return RepairState("not_attempted", "Not attempted", "no repair was recorded for this candidate")


def repair_key(rd: RunData, view: CandidateView) -> str:
    st = repair_state(rd, view)
    if st is None:
        return "none"
    return {"repaired": "repaired_original", "is_repair": "repair_version", "not_attempted": "not_attempted"} \
        .get(st.key, "repair_unresolved")


@dataclass
class Preliminary:
    """The `validate` command's check, run before judging with the generator's beats (always provisional)."""
    status: str
    provisional: bool
    failed: list[str]
    repair_status: str
    repair_reason: str
    repair_actions: list[RepairAction]


def preliminary(view: CandidateView) -> Optional[Preliminary]:
    if not view.validation_record:
        return None
    try:
        rec = CandidateValidation.model_validate(view.validation_record)
    except ValueError:
        return None
    v = rec.validation
    return Preliminary(v.status.value, v.provisional, [f"{c.rule}: {c.message}" for c in v.failed],
                       rec.repair.status, rec.repair.reason, list(rec.repair.actions))


@dataclass
class RepairRecord:
    original: CandidateView
    repaired: Optional[CandidateView]
    state: RepairState
    triggers: list[RuleFailure]
    provenance: Optional[RepairProvenance]
    preliminary: Optional[Preliminary]
    new_judgement: bool = False

    @property
    def succeeded(self) -> bool:
        return self.state.key == "repaired"

    @property
    def became_eligible(self) -> bool:
        return self.succeeded and self.repaired is not None and self.repaired.lane == "eligible"

    @property
    def shortlisted(self) -> bool:
        return self.became_eligible and self.repaired.ranking_status == "selected"


def repair_records(rd: RunData) -> list[RepairRecord]:
    """One record per original candidate whose stored decision records a repair attempt."""
    out = []
    for view in rd.candidates.values():
        d = view.decision
        if view.repaired_from or d is None or d.repair_status not in ("repaired", "unrepairable"):
            continue
        state = repair_state(rd, view)
        rv = rd.candidates.get(state.target) if state.target else None
        prov = (rv.repair if rv else None) or (d.repaired_candidate.repair if d.repaired_candidate else None)
        new_j = bool(rv and rv.raw and view.raw and rv.raw.clip_sha256 != view.raw.clip_sha256)
        out.append(RepairRecord(view, rv, state, rule_failures(rd, view), prov, preliminary(view), new_j))
    return out


@dataclass
class RepairSummary:
    attempts: int
    succeeded: int
    unrepairable: int
    still_failing: int
    awaiting_decision: int
    became_eligible: int
    shortlisted: int
    technical_without_attempt: int

    @property
    def not_resolved(self) -> int:
        return self.attempts - self.succeeded


def repair_summary(rd: RunData, records: Optional[list[RepairRecord]] = None) -> RepairSummary:
    records = repair_records(rd) if records is None else records
    keys = [r.state.key for r in records]
    no_attempt = sum(1 for v in rd.candidates.values()
                     if (s := repair_state(rd, v)) is not None and s.key == "not_attempted")
    return RepairSummary(len(records), keys.count("repaired"), keys.count("unrepairable"),
                         keys.count("repaired_still_failing"), keys.count("repaired_no_decision"),
                         sum(r.became_eligible for r in records), sum(r.shortlisted for r in records), no_attempt)


# --- quality failures -------------------------------------------------------------------------------

@dataclass(frozen=True)
class GateFailure:
    gate: str
    detail: str
    dimension: Optional[str]
    reason: str  # the judge's explanation for that dimension (effective judgement)
    evidence: tuple[str, ...] = ()


def gate_failures(view: CandidateView) -> list[GateFailure]:
    d = view.decision
    if d is None:
        return []
    dims = view.effective.dimensions if view.judged else {}
    out = []
    for g in d.gates:
        if g.passed:
            continue
        name = GATE_DIMENSION.get(g.gate)
        dim = dims.get(name) if name else None
        out.append(GateFailure(g.gate, g.detail, name, dim.reason if dim else "",
                               tuple(dim.evidence) if dim else ()))
    return out


# --- unusable / pending -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Unusable:
    kind: str
    detail: str
    next_step: str
    retry_on_resume: bool  # re-running `judge` picks it up without extra flags


def unusable_info(view: CandidateView, run_name: str) -> Optional[Unusable]:
    d, raw = view.decision, view.raw
    judge_cmd = f"python -m promopilot.pipeline judge --run {run_name}"
    if view.lane == "not_judged":
        if raw is not None and provider_failed(raw):
            detail = redact(next(iter(raw.invalid_reasons), "") or (d.note if d else ""))
            return Unusable("Provider request failed", detail,
                            f"Re-running `{judge_cmd}` retries this candidate (one new API request).", True)
        return Unusable("Judge request not completed", redact(d.note if d else ""),
                        f"Re-running `{judge_cmd}` judges it when quota or budget allows.", True)
    if view.lane == "judge_invalid":
        kind = {"invalid_output": "Malformed judge reply",
                "invalid_assessment": "Judge reply failed evidence checks"}.get(
            raw.failure_kind if raw else None, "Judge output unusable")
        reasons = "; ".join(redact(r, 140) for r in (raw.invalid_reasons if raw else [])[:3])
        return Unusable(kind, reasons or redact(d.note if d else ""),
                        "Not retried on resume (the stored reply is reused). Needs manual review, or an explicit "
                        f"re-judge with `{judge_cmd} --refresh`, which re-judges the selected episodes.", False)
    if view.lane in ("pending", "unknown"):
        return Unusable("No judge decision recorded", "",
                        f"Run `{judge_cmd}` to evaluate it.", True)
    return None


def pending_episodes(rd: RunData) -> list[EpisodeView]:
    return [e for e in rd.episodes.values() if e.state in ("generation_pending", "evaluation_incomplete")]


# --- review -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReviewItem:
    dimension: str
    code: str
    detail: str
    original: Optional[int]
    revised: Optional[int]
    effective: Optional[int]
    outcome: str
    still_flagged: bool


def review_items(view: CandidateView) -> list[ReviewItem]:
    rec, raw = view.reassessment, view.raw
    if rec is None or raw is None or not rec.flags:
        return []
    status, accepted, rejected = review_outcome(raw, rec)
    why = dict(rejected)
    eff = view.effective.dimensions if view.effective else {}
    items = []
    for f in rec.flags:
        name = f.dimension
        orig = rec.original_scores.get(name)
        if orig is None and name in raw.dimensions:
            orig = raw.dimensions[name].score
        if name in accepted:
            outcome = "Revision accepted"
        elif name in why:
            outcome = f"Revision rejected: {why[name]}"
        elif status in ("unresolved", "invalid_reply", "provider_error", "not_run"):
            outcome = humanize(status)
        else:
            outcome = "Not revised"
        effective = eff[name].score if name in eff else None
        items.append(ReviewItem(name, f.code, f.detail, orig, rec.revised_scores.get(name), effective, outcome,
                                bool(name in eff and eff[name].review)))
    return items


# --- filtering --------------------------------------------------------------------------------------

@dataclass
class Filters:
    episodes: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    archetypes: set[str] = field(default_factory=set)
    repairs: set[str] = field(default_factory=set)
    reviews: set[str] = field(default_factory=set)
    focus: Optional[str] = None

    def keep(self, rd: RunData, view: CandidateView, section: str) -> bool:
        if self.focus and view.candidate_id != self.focus:
            return False
        return ((not self.categories or section in self.categories)
                and (not self.episodes or view.episode_id in self.episodes)
                and (not self.archetypes or archetype_key(view) in self.archetypes)
                and (not self.repairs or repair_key(rd, view) in self.repairs)
                and (not self.reviews or review_status(view) in self.reviews))


def sections(rd: RunData, f: Optional[Filters] = None) -> dict[str, list[CandidateView]]:
    """Candidates per attention section, in run order. A candidate appears in at most one failure section, and
    additionally under review when it carries (or carried) a consistency flag."""
    f = f or Filters()
    out: dict[str, list[CandidateView]] = {k: [] for k in CATEGORIES}
    for view in rd.candidates.values():
        cat = category(view)
        if cat and f.keep(rd, view, cat):
            out[cat].append(view)
        if review_status(view) != "none" and f.keep(rd, view, "review"):
            out["review"].append(view)
    return out
