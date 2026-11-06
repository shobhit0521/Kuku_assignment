"""Stage 4: blind promotional-quality judge and Promo DNA.

The judge sees only the clip: its blocks (under neutral aliases K01, K02, ...), clip-relative times, tracks,
speaker labels and text, with labelling words such as "cliffhanger" masked. It never sees the generator's
hook, cliffhanger, reasoning, scores, the episode number, loglines or another model's judgement.

The judge chooses the hook (from blocks starting in the first 3 s) and the final suspense-bearing beat
(from story beats in the last 8 s). Python validates every cited ID, then re-runs the hard-rule validator
with the judge's beats as the authoritative source. Python hard-rule failures can never be overridden.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from . import config
from .generator import _write_json
from .llm import DuplicateRequest, LLMClient, LLMRequest, ProviderError, QuotaExhausted, RequestCapReached
from .schemas import (BeatInput, BeatSource, Block, BlockType, Candidate, ConsistencyFlag, DimensionAssessment,
                      DNAAttribute, Episode, EpisodeGeneration, GateResult, Judgement, PromoDNA, QualityDecision,
                      Reassessment, RuleStatus, SemanticInputs, SoftRejection)
from .validator import (_dialogue_cut, analyse_ending, beat_class, candidate_from_blocks, repair_candidate,
                        validate_candidate)

PROMPT_VERSION = "judge-v1"  # default for library calls; the CLI default comes from config.judge_settings()
PROMPT_FILES = {"judge-v1": "judge_candidate.md", "judge-v2": "judge_candidate_v2.md"}
PROMPT_PATH = Path(__file__).parent / "prompts" / PROMPT_FILES[PROMPT_VERSION]
REPAIR_INSTRUCTION = ("Your previous reply could not be used: {error}. Reply again with JSON only, exactly in the "
                      "requested shape, with every field present and only block IDs from the clip.")

DIMENSIONS = ("hook", "narrative_tension", "cliffhanger", "context_independence", "curiosity_gap",
              "emotional_impact")
WEIGHTED = tuple(config.JUDGE_WEIGHTS)
ARCHETYPES = ("betrayal_humiliation", "confrontation_power_shift", "danger_threat", "mystery_revelation",
              "romance_devotion", "vulnerability", "reversal", "deception_secret_identity")
HOOK_MECHANISMS = ("shocking_line", "accusation_or_question", "threat", "visual_event", "physical_action",
                   "reveal", "in_medias_res", "other")
EMOTIONAL_TRIGGERS = ("outrage_injustice", "fear", "longing_romance", "humiliation_sympathy", "triumph", "grief",
                      "suspicion", "other")
CLIFF_MECHANISMS = ("unanswered_question", "imminent_danger", "revelation", "pending_decision", "reaction_hold",
                    "interrupted_action", "other", "none")
OUTCOMES = ("unresolved", "partly_resolved", "resolved")
BALANCES = ("too_little", "balanced", "too_much")
SOFT_REJECTION_CODES = ("WEAK_HOOK", "WEAK_CLIFFHANGER", "RESOLVED_ENDING", "LOW_TENSION", "CONTEXT_DEPENDENT",
                        "EXCESSIVE_EXPOSITION", "REVEALS_TOO_MUCH", "TOO_VAGUE")
SEVERITIES = ("major", "minor")
PROVIDER_ERROR_PREFIX = "provider_error:"

MASK_RE = re.compile(r"\b(cliff[\s-]?hangers?|suspense(?:ful(?:ly)?)?)\b", re.I)
MASK_TOKEN = "[label]"


# --- response shape (lenient types, required fields) ------------------------------------------------

class _Scored(BaseModel):
    score: int = Field(ge=1, le=5)
    reason: str
    evidence: list[str] = Field(default_factory=list)


class _Hook(_Scored):
    block_id: Optional[str]


class _Tension(_Scored):
    escalation_beats: list[str] = Field(default_factory=list)


class _Cliff(_Scored):
    suspense_beat_id: Optional[str]
    mechanism: str
    outcome: str
    open_question: str = ""


class _Curiosity(_Scored):
    revealed: list[str] = Field(default_factory=list)
    withheld: list[str] = Field(default_factory=list)
    balance: Optional[str] = None


class _Emotion(_Scored):
    emotion: str = ""


class _Confusion(BaseModel):
    issue: str
    block_ids: list[str] = Field(default_factory=list)


class _Archetype(BaseModel):
    primary: str
    other_label: Optional[str] = None
    secondary: Optional[str] = None
    confidence: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)


class _Typed(BaseModel):
    type: str
    evidence: list[str] = Field(default_factory=list)


class _Stakes(BaseModel):
    summary: str
    evidence: list[str] = Field(default_factory=list)


class _ContextDep(BaseModel):
    level: str
    missing_context: list[str] = Field(default_factory=list)


class _DNA(BaseModel):
    archetype: _Archetype
    hook_mechanism: _Typed
    emotional_trigger: _Typed
    stakes: _Stakes
    context_dependency: _ContextDep
    appeal: str = ""


class _Rejection(BaseModel):
    code: str
    severity: str
    explanation: str
    block_ids: list[str] = Field(default_factory=list)


class JudgeResponse(BaseModel):
    blind_summary: str
    confusions: list[_Confusion] = Field(default_factory=list)
    hook: _Hook
    narrative_tension: _Tension
    cliffhanger: _Cliff
    context_independence: _Scored
    curiosity_gap: _Curiosity
    emotional_impact: _Emotion
    promo_dna: _DNA
    soft_rejections: list[_Rejection] = Field(default_factory=list)


class ResponseFormatError(ValueError):
    pass


# --- blind rendering --------------------------------------------------------------------------------

class ClipView(BaseModel):
    """What the judge sees, plus the private alias map Python uses to translate IDs back."""
    text: str
    sha256: str
    start_ms: int
    end_ms: int
    aliases: dict[str, str]            # alias -> real block ID
    eligible_hooks: list[str]          # aliases
    eligible_beats: list[str]          # aliases
    masked_terms: int = 0


def load_prompt(version: str = PROMPT_VERSION) -> tuple[str, str]:
    if version not in PROMPT_FILES:
        raise ValueError(f"unknown judge prompt version {version!r} (known: {', '.join(PROMPT_FILES)})")
    text = (PROMPT_PATH.parent / PROMPT_FILES[version]).read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def mask(text: str) -> tuple[str, int]:
    return MASK_RE.subn(MASK_TOKEN, text)


def _rel(ms: int) -> str:
    sign = "-" if ms < 0 else "+"
    ms = abs(ms)
    return f"{sign}{ms // 60000:02d}:{(ms % 60000) / 1000:05.2f}"


def render_clip(episode: Episode, candidate: Candidate, settings: Optional[dict] = None) -> ClipView:
    """Render only the clip's own content. Uses candidate start/end times, nothing else from the candidate."""
    settings = settings or config.validation_settings()
    s, e = candidate.start_ms, candidate.end_ms
    blocks = episode.blocks_intersecting(s, e)
    aliases = {f"K{i:02d}": b.block_id for i, b in enumerate(blocks, 1)}
    by_real = {v: k for k, v in aliases.items()}
    scenes = {sc.scene_id: sc for sc in episode.scenes}
    masked = 0
    lines = [f"CLIP duration {_rel(e - s)[1:]}. Times are relative to the clip start (+00:00.00).", "",
             "BLOCKS (ID | start-end | track | speaker | text):"]
    current_scene = None
    for b in blocks:
        if b.scene_id != current_scene:
            current_scene = b.scene_id
            heading = scenes[b.scene_id].heading if b.scene_id in scenes and scenes[b.scene_id].heading else ""
            if heading:
                heading, n = mask(heading)
                masked += n
                lines.append(f"[scene: {heading}]")
        label, n1 = mask(b.label_raw) if b.block_type == BlockType.DIALOGUE else ("", 0)
        text, n2 = mask(b.text)
        masked += n1 + n2
        notes = []
        if b.start_ms < s:
            notes.append("began before the clip")
        if b.end_ms > e:
            notes.append("continues after the cut")
        suffix = f"   [{'; '.join(notes)}]" if notes else ""
        lines.append(f"{by_real[b.block_id]} | {_rel(b.start_ms - s)}-{_rel(b.end_ms - s)} | {b.block_type.value} | "
                     f"{label} | {text}{suffix}")
    window = settings["hook_window_ms"]
    hooks = [by_real[b.block_id] for b in blocks if s <= b.start_ms <= s + window]
    options = analyse_ending(episode, s, e, settings).suspense_beat_options
    beats = [by_real[i] for i in options if i in by_real]
    lines += ["", f"ELIGIBLE HOOK BLOCKS (start within the first {window / 1000:g} s): " + (", ".join(hooks) or "none"),
              f"ELIGIBLE FINAL BEATS (story beats in the last {settings['ending_window_ms'] / 1000:g} s): "
              + (", ".join(beats) or "none")]
    text = "\n".join(lines)
    return ClipView(text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), start_ms=s, end_ms=e,
                    aliases=aliases, eligible_hooks=hooks, eligible_beats=beats, masked_terms=masked)


# --- parsing and evidence validation ----------------------------------------------------------------

def parse_response(text: str, finish_reason: Optional[str] = None) -> JudgeResponse:
    if finish_reason == "length":
        raise ResponseFormatError("reply was cut off at the token limit")
    body = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(f"not valid JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(data, dict):
        raise ResponseFormatError("top level must be a JSON object")
    try:
        return JudgeResponse.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:8])
        raise ResponseFormatError(f"schema mismatch ({problems})") from None


class _Evidence:
    def __init__(self, view: ClipView, episode: Episode):
        self.view = view
        self.episode = episode

    def resolve(self, ids: list[str]) -> tuple[list[str], list[str]]:
        good, bad = [], []
        for raw in ids or []:
            alias = str(raw).strip().upper()
            (good if alias in self.view.aliases else bad).append(self.view.aliases.get(alias, str(raw)))
        return list(dict.fromkeys(good)), bad

    def one(self, raw: Optional[str]) -> tuple[Optional[str], Optional[str], bool]:
        """(alias, real id, ok) for a single optional ID."""
        if raw is None or str(raw).strip().lower() in ("", "null", "none"):
            return None, None, True
        alias = str(raw).strip().upper()
        return alias, self.view.aliases.get(alias), alias in self.view.aliases

    def block(self, real_id: str) -> Block:
        return self.episode.get_block(real_id)


def _scored(name: str, part: _Scored, ev: _Evidence) -> DimensionAssessment:
    good, bad = ev.resolve(part.evidence)
    dim = DimensionAssessment(name=name, score=part.score, reason=part.reason.strip(), evidence=good)
    if bad:
        dim.issues.append(f"evidence IDs not in the clip: {', '.join(bad)}")
    if part.score >= 2 and not good:
        dim.issues.append("score of 2 or higher without valid evidence")
    if not dim.reason:
        dim.issues.append("no reason given")
    return dim


def _assess(resp: JudgeResponse, ev: _Evidence) -> dict[str, DimensionAssessment]:
    v = ev.view
    dims: dict[str, DimensionAssessment] = {}

    hook = _scored("hook", resp.hook, ev)
    alias, real, ok = ev.one(resp.hook.block_id)
    if not ok:
        hook.issues.append(f"hook {resp.hook.block_id!r} is not a block of the clip")
    elif alias and alias not in v.eligible_hooks:
        hook.issues.append(f"hook {alias} does not start within the hook window")
    elif real:
        hook.block_id = real
        hook.evidence = list(dict.fromkeys([real] + hook.evidence))
        hook.issues = [i for i in hook.issues if not i.startswith("score of 2")]
    if hook.block_id is None and resp.hook.score >= 3:
        hook.issues.append("hook scored 3 or higher but no hook block selected")
    dims["hook"] = hook

    tension = _scored("narrative_tension", resp.narrative_tension, ev)
    good, bad = ev.resolve(resp.narrative_tension.escalation_beats)
    tension.escalation_beats = good
    if bad:
        tension.issues.append(f"escalation beats not in the clip: {', '.join(bad)}")
    dims["narrative_tension"] = tension

    c = resp.cliffhanger
    cliff = _scored("cliffhanger", c, ev)
    cliff.mechanism = c.mechanism.strip().lower()
    if cliff.mechanism not in CLIFF_MECHANISMS:
        cliff.mechanism = "other"
    cliff.outcome = c.outcome.strip().lower()
    cliff.open_question = c.open_question.strip()
    if cliff.outcome not in OUTCOMES:
        cliff.issues.append(f"outcome {c.outcome!r} is not one of {', '.join(OUTCOMES)}")
    alias, real, ok = ev.one(c.suspense_beat_id)
    if not ok:
        cliff.issues.append(f"suspense beat {c.suspense_beat_id!r} is not a block of the clip")
    elif alias and alias not in v.eligible_beats:
        cliff.issues.append(f"suspense beat {alias} is not an eligible final story beat")
    elif real:
        cliff.suspense_beat_id = real
        cliff.evidence = list(dict.fromkeys([real] + cliff.evidence))
        cliff.issues = [i for i in cliff.issues if not i.startswith("score of 2")]
    if cliff.suspense_beat_id is None and c.score >= 3:
        cliff.issues.append("cliffhanger scored 3 or higher but no suspense beat selected")
    dims["cliffhanger"] = cliff

    ctx = _scored("context_independence", resp.context_independence, ev)
    ctx.blind_summary = resp.blind_summary.strip()
    if not ctx.blind_summary:
        ctx.issues.append("blind summary missing")
    for conf in resp.confusions:
        good, bad = ev.resolve(conf.block_ids)
        if bad or not good:
            ctx.issues.append(f"confusion {conf.issue[:60]!r} lacks valid block IDs")
        ctx.confusions.append({"issue": conf.issue, "block_ids": good})
    dims["context_independence"] = ctx

    cur = _scored("curiosity_gap", resp.curiosity_gap, ev)
    cur.revealed, cur.withheld = resp.curiosity_gap.revealed, resp.curiosity_gap.withheld
    bal = (resp.curiosity_gap.balance or "").strip().lower()
    cur.balance = bal if bal in BALANCES else None
    dims["curiosity_gap"] = cur

    emo = _scored("emotional_impact", resp.emotional_impact, ev)
    emo.emotion = resp.emotional_impact.emotion.strip()
    if resp.emotional_impact.score >= 2 and emo.evidence and \
            not any(beat_class(ev.block(b)) == "story" for b in emo.evidence):
        emo.issues.append("emotion evidence is only music or transition cues")
    dims["emotional_impact"] = emo

    for d in dims.values():
        d.valid = not d.issues
    return dims


def _dna(resp: JudgeResponse, dims: dict[str, DimensionAssessment], ev: _Evidence) -> PromoDNA:
    d = resp.promo_dna
    dna = PromoDNA()

    def attr(value: str, allowed: tuple, evidence: list[str], label: str) -> DNAAttribute:
        good, bad = ev.resolve(evidence)
        value = (value or "").strip().lower()
        if value == "unclassified":
            return DNAAttribute(value="unclassified", note="judge found no clear basis")
        if value not in allowed:
            return DNAAttribute(value="unclassified", note=f"{label} {value!r} is not in the taxonomy")
        if bad or not good:
            return DNAAttribute(value="unclassified", note=f"{label} lacks valid evidence")
        return DNAAttribute(value=value, evidence=good)

    dna.archetype = attr(d.archetype.primary, ARCHETYPES + ("other",), d.archetype.evidence, "archetype")
    if dna.archetype.value == "other":
        label = (d.archetype.other_label or "").strip()
        if label:
            dna.archetype_other_label = label
        else:
            dna.archetype = DNAAttribute(value="unclassified", note="archetype 'other' without a label")
    sec = (d.archetype.secondary or "").strip().lower()
    dna.archetype_secondary = sec if sec in ARCHETYPES and sec != dna.archetype.value else None
    conf = (d.archetype.confidence or "").strip().lower()
    dna.archetype_confidence = conf if conf in ("low", "medium", "high") else None
    dna.hook_mechanism = attr(d.hook_mechanism.type, HOOK_MECHANISMS, d.hook_mechanism.evidence, "hook mechanism")
    dna.emotional_trigger = attr(d.emotional_trigger.type, EMOTIONAL_TRIGGERS, d.emotional_trigger.evidence,
                                 "emotional trigger")
    good, bad = ev.resolve(d.stakes.evidence)
    dna.stakes = DNAAttribute(value=d.stakes.summary.strip(), evidence=good) \
        if d.stakes.summary.strip() and good and not bad else DNAAttribute(note="stakes lack valid evidence")
    cliff = dims["cliffhanger"]
    dna.cliffhanger_mechanism = DNAAttribute(value=cliff.mechanism, evidence=[cliff.suspense_beat_id]) \
        if cliff.valid and cliff.suspense_beat_id and cliff.mechanism != "none" \
        else DNAAttribute(note="no valid suspense beat")
    level = d.context_dependency.level.strip().lower()
    dna.context_dependency = level if level in ("low", "medium", "high") else "unclassified"
    dna.missing_context = d.context_dependency.missing_context
    cur = dims["curiosity_gap"]
    dna.curiosity_revealed, dna.curiosity_withheld, dna.curiosity_balance = cur.revealed, cur.withheld, cur.balance
    dna.appeal = d.appeal.strip() or None
    dna.supporting_block_ids = list(dict.fromkeys(
        b for a in (dna.archetype, dna.hook_mechanism, dna.emotional_trigger, dna.stakes, dna.cliffhanger_mechanism)
        for b in a.evidence))
    return dna


def _rejections(resp: JudgeResponse, ev: _Evidence) -> tuple[list[SoftRejection], list[dict]]:
    kept, dropped = [], []
    for r in resp.soft_rejections:
        code, sev = r.code.strip().upper(), r.severity.strip().lower()
        good, bad = ev.resolve(r.block_ids)
        problem = ("unknown code" if code not in SOFT_REJECTION_CODES else
                   "unknown severity" if sev not in SEVERITIES else
                   "block IDs not in the clip" if bad else
                   "no block IDs" if not good else
                   "no explanation" if not r.explanation.strip() else None)
        if problem:
            dropped.append({"code": r.code, "severity": r.severity, "problem": problem})
        elif code not in {k.code for k in kept}:
            kept.append(SoftRejection(code=code, severity=sev, explanation=r.explanation.strip(), block_ids=good))
    return kept, dropped


def build_judgement(episode: Episode, candidate: Candidate, view: ClipView, resp: Optional[JudgeResponse], *,
                    provider: str, model: str, prompt_sha: str, errors: list[str], calls: list[dict],
                    raw_text: Optional[str], prompt_version: str = PROMPT_VERSION) -> Judgement:
    j = Judgement(candidate_id=candidate.candidate_id, episode_id=candidate.episode_id, status="invalid",
                  provider=provider, model=model, prompt_version=prompt_version, prompt_sha256=prompt_sha,
                  clip_sha256=view.sha256, clip_start_ms=view.start_ms, clip_end_ms=view.end_ms,
                  masked_terms=view.masked_terms, eligible_hook_ids=[view.aliases[a] for a in view.eligible_hooks],
                  eligible_beat_ids=[view.aliases[a] for a in view.eligible_beats], response_errors=errors,
                  calls=calls, raw_response_text=raw_text, judged_at=datetime.now(timezone.utc).isoformat())
    if resp is None:
        j.invalid_reasons.append(errors[-1] if errors else "no usable response")
        j.failure_kind = "provider_error" if errors and errors[-1].startswith(PROVIDER_ERROR_PREFIX) \
            else "invalid_output"
        return j
    ev = _Evidence(view, episode)
    j.dimensions = _assess(resp, ev)
    j.promo_dna = _dna(resp, j.dimensions, ev)
    j.soft_rejections, j.dropped_rejections = _rejections(resp, ev)
    for name in WEIGHTED:
        if not j.dimensions[name].valid:
            j.invalid_reasons.append(f"{name}: " + "; ".join(j.dimensions[name].issues))
    j.status = "invalid" if j.invalid_reasons else "valid"
    j.failure_kind = "invalid_assessment" if j.invalid_reasons else None
    return j


def provider_failed(judgement: Judgement) -> bool:
    return judgement.failure_kind == "provider_error" or \
        any(r.startswith(PROVIDER_ERROR_PREFIX) for r in judgement.invalid_reasons)


# --- calling the judge ------------------------------------------------------------------------------

def _messages(view: ClipView, prompt_version: str) -> tuple[list[dict], str, str]:
    prompt, prompt_sha = load_prompt(prompt_version)
    return ([{"role": "system", "content": prompt}, {"role": "user", "content": view.text}], prompt_sha,
            f"{prompt_version}:{prompt_sha[:12]}")


def judge_candidate(episode: Episode, candidate: Candidate, client: LLMClient, *, temperature: float,
                    max_tokens: int, settings: Optional[dict] = None,
                    prompt_version: str = PROMPT_VERSION) -> Judgement:
    """One blind judgement (plus at most one format-repair call). Raises QuotaExhausted / RequestCapReached."""
    view = render_clip(episode, candidate, settings)
    messages, prompt_sha, version = _messages(view, prompt_version)
    common = dict(provider=client.provider.name, model=client.provider.model, prompt_sha=prompt_sha,
                  prompt_version=prompt_version)
    calls, errors = [], []
    try:
        result = client.complete(LLMRequest(messages=messages, temperature=temperature, max_tokens=max_tokens),
                                 prompt_version=version, stage="judge", item_id=candidate.candidate_id)
    except (QuotaExhausted, RequestCapReached):
        raise
    except ProviderError as exc:
        return build_judgement(episode, candidate, view, None, errors=[f"{PROVIDER_ERROR_PREFIX} {exc}"],
                               calls=calls, raw_text=None, **common)
    calls.append(result.summary("judge"))
    try:
        resp = parse_response(result.response.text, result.response.finish_reason)
    except ResponseFormatError as first:
        errors.append(f"judge: {first}")
        repair = LLMRequest(messages=messages + [
            {"role": "assistant", "content": result.response.text},
            {"role": "user", "content": REPAIR_INSTRUCTION.format(error=first)}],
            temperature=temperature, max_tokens=max_tokens)
        try:
            result = client.complete(repair, prompt_version=version, stage="judge_repair",
                                     item_id=candidate.candidate_id)
        except (QuotaExhausted, RequestCapReached):
            raise
        except ProviderError as exc:
            errors.append(f"{PROVIDER_ERROR_PREFIX} {exc}")
            return build_judgement(episode, candidate, view, None, errors=errors, calls=calls,
                                   raw_text=result.response.text, **common)
        calls.append(result.summary("judge_repair"))
        try:
            resp = parse_response(result.response.text, result.response.finish_reason)
        except ResponseFormatError as second:
            errors.append(f"judge_repair: {second}")
            return build_judgement(episode, candidate, view, None, errors=errors, calls=calls,
                                   raw_text=result.response.text, **common)
    return build_judgement(episode, candidate, view, resp, errors=errors, calls=calls,
                           raw_text=result.response.text, **common)


# --- reason/score consistency and targeted reassessment ---------------------------------------------
# These checks only decide whether to ask the judge to look again. They never change a score themselves.

CHECKER_VERSION = "c1"
_WEAK_ENDING_RE = re.compile(
    r"emotional button|rather than (?:an? )?(?:active|real|clear|open|plot|narrative|sharp|specific)"
    r"|no (?:real|clear|active|open|specific) question|\bresolv(?:es|ed|ing)\b|\bconclud(?:es|ed|ing)?\b"
    r"|\bconclude\b|\bsettl(?:es|ed|ing)\b|\bwraps? up\b|\bcloses out\b|\bserene\b|\bpeaceful\b", re.I)
_DEPENDENCY_RE = re.compile(
    r"\b(?:cannot|can't|could not|unable to) (?:understand|follow|tell)\b"
    r"|\b(?:unclear|confusing|lost|hard to follow) without\b|\brequires? (?:outside|prior|earlier|previous)\b"
    r"|\brelies on (?:prior|earlier|outside)\b|\bunexplained\b", re.I)
_REASSESS_KEYS = {"cliffhanger": ["cliffhanger"], "emotional_impact": ["emotional_impact"],
                  "context_independence": ["context_independence", "confusions", "blind_summary"]}
REASSESS_INSTRUCTION = (
    "An automatic consistency check found possible contradictions in your assessment:\n{issues}\n"
    "Re-read the clip and the scoring anchors for {dims} only, and reassess them from the evidence. Keep a score "
    "only if your reason matches that score's anchor; otherwise change the score or the reason so they agree. "
    "Do not change any other part of your assessment.\n"
    "Reply with JSON only, containing exactly these keys: {keys}. Each has the same fields as in your previous reply.")


def consistency_flags(judgement: Judgement, episode: Episode) -> list[ConsistencyFlag]:
    if judgement.status != "valid":
        return []
    d, flags = judgement.dimensions, []

    def flag(dim, code, detail):
        flags.append(ConsistencyFlag(dimension=dim, code=code, detail=detail))

    c = d["cliffhanger"]
    if c.outcome == "resolved" and c.score >= 3:
        flag("cliffhanger", "RESOLVED_OUTCOME_PASSING_SCORE", f"outcome is resolved but the score is {c.score}")
    if c.outcome == "partly_resolved" and c.score >= 5:
        flag("cliffhanger", "PARTLY_RESOLVED_TOP_SCORE", "outcome is partly resolved but the score is 5")
    if c.score >= 3 and (c.mechanism == "none" or not c.open_question):
        flag("cliffhanger", "NO_OPEN_QUESTION", f"score {c.score} but no open question / mechanism is none")
    m = _WEAK_ENDING_RE.search(c.reason or "")
    if c.score >= 3 and m:
        flag("cliffhanger", "REASON_DESCRIBES_WEAK_ENDING",
             f"score {c.score} but the reason describes a weak or settled ending ({m.group(0)!r})")

    x = d["context_independence"]
    n_conf = len(x.confusions)
    if (x.score >= 4 and n_conf >= 2) or (x.score >= 5 and n_conf >= 1):
        flag("context_independence", "CONFUSIONS_WITH_HIGH_SCORE",
             f"score {x.score} with {n_conf} confusing reference(s) listed")
    level = judgement.promo_dna.context_dependency if judgement.promo_dna else None
    if (level == "high" and x.score >= 3) or (level == "medium" and x.score >= 5):
        flag("context_independence", "DEPENDENCY_LEVEL_MISMATCH",
             f"score {x.score} but context dependency is {level}")
    m = _DEPENDENCY_RE.search(x.reason or "")
    if x.score >= 4 and m:
        flag("context_independence", "REASON_DESCRIBES_DEPENDENCY",
             f"score {x.score} but the reason describes missing context ({m.group(0)!r})")

    e = d["emotional_impact"]
    story = [b for b in e.evidence if episode.has_block(b) and beat_class(episode.get_block(b)) == "story"]
    need = {4: 2, 5: 3}.get(e.score)
    if need and len(story) < need:
        flag("emotional_impact", "THIN_EVIDENCE_FOR_SCORE",
             f"score {e.score} needs at least {need} cited story beats; {len(story)} cited")
    return flags


def _load_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    body = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    try:
        data = json.loads(fenced.group(1) if fenced else body)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def reassess(episode: Episode, candidate: Candidate, judgement: Judgement, client: LLMClient, *,
             temperature: float, max_tokens: int, settings: Optional[dict] = None) -> Reassessment:
    """At most one targeted reassessment call for the flagged dimensions. Raises QuotaExhausted /
    RequestCapReached / DuplicateRequest; other provider errors are recorded."""
    flags = consistency_flags(judgement, episode)
    rec = Reassessment(candidate_id=candidate.candidate_id, checker_version=CHECKER_VERSION,
                       judgement_clip_sha256=judgement.clip_sha256, judgement_prompt_sha256=judgement.prompt_sha256,
                       status="not_needed", flags=flags, created_at=datetime.now(timezone.utc).isoformat(),
                       original_scores={f.dimension: judgement.dimensions[f.dimension].score for f in flags})
    if not flags:
        return rec
    original = _load_json(judgement.raw_response_text)
    if original is None:
        rec.status, rec.note = "invalid_reply", "original reply not available for reassessment"
        return rec
    dims = list(dict.fromkeys(f.dimension for f in flags))
    keys = [k for dim in dims for k in _REASSESS_KEYS[dim]]
    view = render_clip(episode, candidate, settings)
    messages, prompt_sha, version = _messages(view, judgement.prompt_version)
    issues = "\n".join(f"- {f.dimension}: {f.detail}" for f in flags)
    request = LLMRequest(messages=messages + [
        {"role": "assistant", "content": judgement.raw_response_text},
        {"role": "user", "content": REASSESS_INSTRUCTION.format(issues=issues, dims=", ".join(dims),
                                                                keys=", ".join(keys))}],
        temperature=temperature, max_tokens=max_tokens)
    try:
        result = client.complete(request, prompt_version=f"{version}:{CHECKER_VERSION}", stage="reassess",
                                 item_id=candidate.candidate_id)
    except (QuotaExhausted, RequestCapReached):
        raise
    except ProviderError as exc:
        rec.status, rec.note = "provider_error", f"{PROVIDER_ERROR_PREFIX} {exc}"
        return rec
    rec.calls.append(result.summary("reassess"))
    rec.raw_response_text = result.response.text
    reply = _load_json(result.response.text) if result.response.finish_reason != "length" else None
    missing = [k for dim in dims for k in _REASSESS_KEYS[dim][:1] if not reply or k not in reply]
    if reply is None or missing:
        rec.status, rec.note = "invalid_reply", f"reply unusable (missing: {', '.join(missing) or 'JSON'})"
        return rec
    merged = {**original, **{k: reply[k] for k in keys if k in reply}}
    try:
        resp = JudgeResponse.model_validate(merged)
    except ValidationError as exc:
        rec.status, rec.note = "invalid_reply", f"schema mismatch ({exc.errors()[0]['msg']})"
        return rec
    revised = build_judgement(episode, candidate, view, resp, provider=judgement.provider, model=judgement.model,
                              prompt_sha=prompt_sha, errors=[], calls=[], raw_text=None,
                              prompt_version=judgement.prompt_version)
    bad = [d for d in dims if not revised.dimensions[d].valid]
    if bad:
        rec.status = "invalid_reply"
        rec.note = "; ".join(f"{d}: {'; '.join(revised.dimensions[d].issues)}" for d in bad)
        return rec
    rec.revised_scores = {d: revised.dimensions[d].score for d in dims}
    rec.remaining_flags = [f for f in consistency_flags(revised, episode) if f.dimension in dims]
    if rec.remaining_flags:
        rec.status = "unresolved"
        rec.note = "contradiction remains; original judgement kept and marked for review"
        return rec
    rec.revised_dimensions = {d: revised.dimensions[d] for d in dims}
    rec.status = "resolved"
    rec.status, accepted, rejected = review_outcome(judgement, rec)
    if rejected:
        rec.note = "; ".join(f"{d}: {why}" for d, why in rejected)
    return rec


def _revision_problem(orig: DimensionAssessment, new: DimensionAssessment) -> Optional[str]:
    """Every consistency flag says a score is too high for its reasons. A revision that clears the flag by
    raising the score, or by deleting the listed confusions, is not a resolution."""
    if new.score > orig.score:
        return f"revision raised the score {orig.score}->{new.score} instead of resolving the contradiction"
    if orig.name == "context_independence" and len(new.confusions) < len(orig.confusions) \
            and new.score >= orig.score:
        return (f"revision removed listed confusions ({len(orig.confusions)}->{len(new.confusions)}) "
                f"without lowering the score")
    return None


def review_outcome(judgement: Judgement, rec: Reassessment) -> tuple[str, dict, list[tuple[str, str]]]:
    """(status, accepted revised dimensions, rejected (dimension, reason)) for a reassessment record."""
    if rec.status not in ("resolved", "partly_resolved", "revision_rejected"):
        return rec.status, {}, []
    accepted, rejected = {}, []
    for name, new in rec.revised_dimensions.items():
        problem = _revision_problem(judgement.dimensions[name], new)
        if problem:
            rejected.append((name, problem))
        else:
            accepted[name] = new
    status = "resolved" if not rejected else "partly_resolved" if accepted else "revision_rejected"
    return status, accepted, rejected


def effective_judgement(judgement: Judgement, rec: Optional[Reassessment]) -> Judgement:
    """The judgement used for decisions: accepted revisions applied; otherwise the original, with the
    flagged dimensions marked for review."""
    if rec is None or rec.status == "not_needed" or not rec.flags:
        return judgement
    eff = judgement.model_copy(deep=True)
    status, accepted, rejected = review_outcome(judgement, rec)
    if status in ("resolved", "partly_resolved", "revision_rejected"):
        eff.dimensions.update({k: v.model_copy(deep=True) for k, v in accepted.items()})
        cliff = eff.dimensions["cliffhanger"]
        if "cliffhanger" in accepted and eff.promo_dna:
            eff.promo_dna.cliffhanger_mechanism = DNAAttribute(value=cliff.mechanism, evidence=[cliff.suspense_beat_id]) \
                if cliff.suspense_beat_id and cliff.mechanism != "none" else DNAAttribute(note="no valid suspense beat")
        for name, why in rejected:
            dim = eff.dimensions[name]
            dim.review = True
            dim.review_reasons += [f"{f.code}: {f.detail} (reassessment rejected: {why})"
                                   for f in rec.flags if f.dimension == name]
        return eff
    flags = rec.remaining_flags if rec.status == "unresolved" else rec.flags
    for f in flags:
        dim = eff.dimensions[f.dimension]
        dim.review = True
        dim.review_reasons.append(f"{f.code}: {f.detail} (reassessment {rec.status})")
    return eff


def reassessment_status(judgement: Judgement, rec: Optional[Reassessment]) -> Optional[str]:
    return review_outcome(judgement, rec)[0] if rec else None


# --- combining with the hard rules ------------------------------------------------------------------

def judge_inputs(judgement: Judgement) -> SemanticInputs:
    hook = judgement.dimensions.get("hook")
    cliff = judgement.dimensions.get("cliffhanger")
    return SemanticInputs(
        hook=BeatInput(block_id=hook.block_id, source=BeatSource.JUDGE) if hook and hook.block_id else None,
        suspense_beat=BeatInput(block_id=cliff.suspense_beat_id, source=BeatSource.JUDGE)
        if cliff and cliff.suspense_beat_id else None,
        outcome=cliff.outcome if cliff else None)


def weighted_score(judgement: Judgement, weights: Optional[dict] = None) -> Optional[float]:
    weights = weights or config.JUDGE_WEIGHTS
    dims = judgement.dimensions
    if any(n not in dims or not dims[n].valid for n in WEIGHTED):
        return None
    return round(sum(weights[n] * dims[n].score for n in WEIGHTED), 2)


def decide(episode: Episode, candidate: Candidate, judgement: Judgement,
           settings: Optional[dict] = None) -> QualityDecision:
    """Python's hard rules are authoritative; the judge's quality gates apply only on top of them."""
    settings = settings or config.validation_settings()
    usable = judgement.status == "valid"
    inputs = judge_inputs(judgement) if usable else None
    validation = validate_candidate(episode, candidate, inputs, settings)
    decision = QualityDecision(candidate_id=candidate.candidate_id, episode_id=candidate.episode_id,
                               verdict="judge_invalid", hard_rules=validation.status.value,
                               hard_rule_failures=[f"{c.rule}: {c.message}" for c in validation.failed],
                               validation=validation)
    if usable:
        dims = judgement.dimensions
        rule = {c.rule: c for c in validation.checks}
        for name, minimum in config.JUDGE_GATE_MIN.items():
            d = dims[name]
            decision.gates.append(GateResult(gate=f"{name}_score", passed=d.valid and d.score >= minimum,
                                             detail=f"{d.score} (minimum {minimum})"))
        hook_ok = bool(inputs.hook) and rule["HOOK_WINDOW"].status == RuleStatus.PASS
        decision.gates.append(GateResult(gate="hook_block_in_window", passed=hook_ok,
                                         detail=rule["HOOK_WINDOW"].message if inputs.hook else
                                         "judge found no suitable hook in the first 3 s"))
        beat_ok = bool(inputs.suspense_beat) and all(
            rule[r].status == RuleStatus.PASS for r in ("SUSPENSE_BEAT_IN_CLIP", "SUSPENSE_BEAT_IS_STORY_BEAT"))
        decision.gates.append(GateResult(gate="suspense_beat_in_clip", passed=beat_ok,
                                         detail=rule["SUSPENSE_BEAT_IN_CLIP"].message if inputs.suspense_beat
                                         else "judge found no suspense-bearing final beat"))
        outcome = dims["cliffhanger"].outcome
        decision.gates.append(GateResult(gate="outcome_not_resolved", passed=outcome in ("unresolved",
                                                                                       "partly_resolved"),
                                         detail=f"outcome at the cut: {outcome}"))
        decision.weighted_score = weighted_score(judgement)
        decision.soft_rejection_codes = [r.code for r in judgement.soft_rejections]
        decision.dimension_scores = {n: d.score for n, d in dims.items() if d.score is not None}
        decision.archetype = judgement.promo_dna.archetype.value if judgement.promo_dna else None
        decision.review_flags = [f"{n}: {r}" for n, d in dims.items() for r in d.review_reasons]
    if provider_failed(judgement):
        decision.verdict = "not_judged"
        decision.note = "judge unavailable (retried on resume): " + judgement.invalid_reasons[0][:200]
    elif validation.status == RuleStatus.FAIL:
        decision.verdict = "hard_fail"
    elif not usable:
        decision.verdict = "judge_invalid"
        decision.note = "; ".join(judgement.invalid_reasons)
    elif validation.status != RuleStatus.PASS or not all(g.passed for g in decision.gates):
        decision.verdict = "quality_fail"
    else:
        decision.verdict, decision.eligible = "eligible", True
    return decision


# --- run safety -------------------------------------------------------------------------------------

class RunLocked(Exception):
    """Another judge invocation is already writing to this output directory."""


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel32.GetLastError() == 5  # access denied: the process exists
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def run_lock(directory: Path, invocation_id: str):
    """Exclusive lock file for one judge output directory. A lock left by a dead process is replaced."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "judge.lock"
    info = json.dumps({"pid": os.getpid(), "invocation_id": invocation_id,
                       "started_at": datetime.now(timezone.utc).isoformat()})
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                prior = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
            if prior.get("pid") and _pid_alive(prior["pid"]):
                raise RunLocked(f"another judge run (pid {prior['pid']}, invocation {prior.get('invocation_id')}, "
                                f"started {prior.get('started_at')}) is using {directory}") from None
            path.unlink(missing_ok=True)
    else:
        raise RunLocked(f"could not acquire {path}")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(info)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _archive(path: Path) -> None:
    """Move an existing output into history/ instead of overwriting it."""
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = path.parent / "history" / f"{path.stem}.{stamp}{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)


# --- run --------------------------------------------------------------------------------------------

def model_slug(provider: str, model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{provider}__{model}")


def judge_dir(run_dir: Path, provider: str, model: str, prompt_version: str = PROMPT_VERSION,
              endpoint: Optional[str] = None) -> Path:
    """judge-v1 outputs keep their original location; other prompt versions get their own folder. A non-default
    endpoint (see `llm.endpoint_identity`) adds a short endpoint hash, so outputs from two endpoints never mix."""
    slug = model_slug(provider, model)
    if endpoint:
        slug += "__ep-" + hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:8]
    return Path(run_dir) / "judge" / (slug if prompt_version == "judge-v1" else f"{slug}__{prompt_version}")


def _unjudgeable(episode: Episode, candidate: Candidate, settings: dict) -> Optional[str]:
    """Objective failures no repair can fix; such clips are not sent to the judge."""
    pre = validate_candidate(episode, candidate, None, settings)
    for c in pre.failed:
        if c.rule in ("REFERENCES_VALID", "BOUNDARY_ORDER"):
            return f"{c.rule}: {c.message}"
        if c.rule == "DURATION_RANGE" and c.value_ms > settings["max_clip_ms"]:
            return f"{c.rule}: {c.message}"
    return None


class _NewCallNotAllowed(Exception):
    pass


class JudgeRun:
    """Resumable judging of one run. new_judgements=None allows new judge calls for every candidate; a set
    restricts them to those IDs (others are only resumed). Never overwrites a raw judgement."""

    def __init__(self, run_dir: Path, episodes: dict[int, Episode], client: LLMClient, *, temperature: float,
                 max_tokens: int, settings: Optional[dict] = None, refresh: bool = False,
                 prompt_version: str = PROMPT_VERSION, out_dir: Optional[Path] = None,
                 new_judgements: Optional[set[str]] = None, reassess: bool = True, allow_repair: bool = True,
                 invocation_id: Optional[str] = None, stop_on_provider_error: bool = False):
        self.stop_on_provider_error = stop_on_provider_error
        self.run_dir = Path(run_dir)
        self.episodes = episodes
        self.client = client
        self.temperature, self.max_tokens = temperature, max_tokens
        self.settings = settings or config.validation_settings()
        self.refresh = refresh
        self.prompt_version = prompt_version
        self.out_dir = Path(out_dir) if out_dir else judge_dir(run_dir, client.provider.name, client.provider.model,
                                                               prompt_version, getattr(client.provider, "endpoint", None))
        self.new_judgements = new_judgements
        self.do_reassess = reassess
        self.allow_repair = allow_repair
        self.invocation_id = invocation_id or uuid.uuid4().hex[:12]
        client.log_extra.setdefault("invocation_id", self.invocation_id)
        self.stop_reason: Optional[str] = None
        self.resumed = 0
        self.reassess_resumed = 0
        self._judged_now: set[str] = set()
        _, self.prompt_sha = load_prompt(prompt_version)

    def judgement_path(self, candidate_id: str) -> Path:
        return self.out_dir / "judgements" / f"{candidate_id}.json"

    def reassessment_path(self, candidate_id: str) -> Path:
        return self.out_dir / "reassessments" / f"{candidate_id}.json"

    def judge(self, episode: Episode, candidate: Candidate) -> Judgement:
        cid = candidate.candidate_id
        path = self.judgement_path(cid)
        view_sha = render_clip(episode, candidate, self.settings).sha256
        if path.exists() and not self.refresh:
            prior = Judgement.model_validate_json(path.read_text(encoding="utf-8"))
            same = (prior.provider, prior.model, prior.prompt_sha256, prior.clip_sha256) == (
                self.client.provider.name, self.client.provider.model, self.prompt_sha, view_sha)
            if same and not provider_failed(prior):
                self.resumed += 1
                return prior
            if same and self.new_judgements is not None and cid not in self.new_judgements:
                return prior
        if self.new_judgements is not None and cid not in self.new_judgements:
            raise _NewCallNotAllowed(f"new judge calls are not allowed for {cid} in this invocation")
        if cid in self._judged_now:
            raise DuplicateRequest(f"{cid} was already judged in this invocation")
        self._judged_now.add(cid)
        j = judge_candidate(episode, candidate, self.client, temperature=self.temperature,
                            max_tokens=self.max_tokens, settings=self.settings, prompt_version=self.prompt_version)
        j.invocation_id = self.invocation_id
        _archive(path)
        _write_json(path, j)
        return j

    def reassessment(self, episode: Episode, candidate: Candidate, judgement: Judgement) -> Optional[Reassessment]:
        if judgement.status != "valid":
            return None
        path = self.reassessment_path(candidate.candidate_id)
        if path.exists():
            prior = Reassessment.model_validate_json(path.read_text(encoding="utf-8"))
            if (prior.checker_version, prior.judgement_clip_sha256, prior.judgement_prompt_sha256) == (
                    CHECKER_VERSION, judgement.clip_sha256, judgement.prompt_sha256) \
                    and (prior.status != "provider_error" or not self.do_reassess):
                self.reassess_resumed += 1
                return prior
        flags = consistency_flags(judgement, episode)
        if not flags:
            return None
        if not self.do_reassess or self.stop_reason:
            return Reassessment(candidate_id=candidate.candidate_id, checker_version=CHECKER_VERSION,
                                judgement_clip_sha256=judgement.clip_sha256,
                                judgement_prompt_sha256=judgement.prompt_sha256, status="not_run", flags=flags,
                                note=self.stop_reason or "reassessment disabled for this invocation",
                                created_at=datetime.now(timezone.utc).isoformat())
        try:
            rec = reassess(episode, candidate, judgement, self.client, temperature=self.temperature,
                           max_tokens=self.max_tokens, settings=self.settings)
        except (QuotaExhausted, RequestCapReached, DuplicateRequest) as exc:
            self.stop_reason = self.stop_reason or str(exc)
            return Reassessment(candidate_id=candidate.candidate_id, checker_version=CHECKER_VERSION,
                                judgement_clip_sha256=judgement.clip_sha256,
                                judgement_prompt_sha256=judgement.prompt_sha256, status="not_run", flags=flags,
                                note=str(exc), created_at=datetime.now(timezone.utc).isoformat())
        rec.invocation_id = self.invocation_id
        _archive(path)
        _write_json(path, rec)
        return rec

    def process(self, episode: Episode, candidate: Candidate) -> list[QualityDecision]:
        reason = _unjudgeable(episode, candidate, self.settings)
        if reason:
            return [QualityDecision(candidate_id=candidate.candidate_id, episode_id=candidate.episode_id,
                                    verdict="hard_fail", hard_rules="fail", hard_rule_failures=[reason],
                                    note="not sent to the judge: objective failure no repair can fix")]
        try:
            if self.stop_reason and not self.judgement_path(candidate.candidate_id).exists():
                return [self._pending(candidate, self.stop_reason)]
            j = self.judge(episode, candidate)
        except (QuotaExhausted, RequestCapReached, DuplicateRequest) as exc:
            self.stop_reason = self.stop_reason or str(exc)
            return [self._pending(candidate, str(exc))]
        except _NewCallNotAllowed as exc:
            return [self._pending(candidate, str(exc))]
        if provider_failed(j) and self.stop_on_provider_error:
            self.stop_reason = self.stop_reason or f"provider error while judging {candidate.candidate_id}; stopped"
        rec = self.reassessment(episode, candidate, j)
        if rec and rec.status == "provider_error" and self.stop_on_provider_error:
            self.stop_reason = self.stop_reason or f"provider error while reassessing {candidate.candidate_id}; stopped"
        d = decide(episode, candidate, effective_judgement(j, rec), self.settings)
        d.reassessment_status = reassessment_status(j, rec)
        d.judgement_file = str(self.judgement_path(candidate.candidate_id))
        d.repaired_from = candidate.repair.original_candidate_id if candidate.repair else None
        out = [d]
        if (d.verdict != "hard_fail" or candidate.repair is not None or j.status != "valid"
                or not self.allow_repair):
            return out
        outcome = repair_candidate(episode, candidate, d.validation, judge_inputs(effective_judgement(j, rec)),
                                   self.settings)
        d.repair_status, d.repair_reason = outcome.status, outcome.reason
        if outcome.repaired_candidate is None:
            return out
        d.repaired_candidate = outcome.repaired_candidate
        out.extend(self.process(episode, outcome.repaired_candidate))
        return out

    def _pending(self, candidate: Candidate, reason: str) -> QualityDecision:
        return QualityDecision(candidate_id=candidate.candidate_id, episode_id=candidate.episode_id,
                               verdict="not_judged", note=f"pending: {reason}")

    def run_candidates(self, by_episode: dict[str, list[Candidate]], sources: Optional[dict[str, str]] = None
                       ) -> dict:
        with run_lock(self.out_dir, self.invocation_id):
            decisions_dir = self.out_dir / "decisions"
            summary = {"provider": self.client.provider.name, "model": self.client.provider.model,
                       "endpoint": getattr(self.client.provider, "endpoint", None),
                       "prompt_version": self.prompt_version, "prompt_sha256": self.prompt_sha,
                       "checker_version": CHECKER_VERSION, "invocation_id": self.invocation_id,
                       "weights": config.JUDGE_WEIGHTS, "gate_minimums": config.JUDGE_GATE_MIN,
                       "validator_version": config.VALIDATOR_VERSION, "episodes": {}}
            for eid, cands in by_episode.items():
                episode = self.episodes[int(eid[2:])]
                decisions = [d for c in cands for d in self.process(episode, c)]
                path = decisions_dir / f"{eid}.json"
                _archive(path)
                _write_json(path, {"episode_id": eid, "source_file": (sources or {}).get(eid),
                                   "invocation_id": self.invocation_id,
                                   "decisions": [d.model_dump(mode="json") for d in decisions]})
                summary["episodes"][eid] = [
                    {"candidate_id": d.candidate_id, "verdict": d.verdict, "weighted_score": d.weighted_score,
                     "hard_rules": d.hard_rules, "failed_gates": [g.gate for g in d.gates if not g.passed],
                     "repair": d.repair_status, "reassessment": d.reassessment_status,
                     "review_flags": len(d.review_flags)} for d in decisions]
            summary.update({"stopped_early": self.stop_reason,
                            "requests_made_this_invocation": self.client.requests_made,
                            "cache_hits_this_invocation": self.client.cache_hits, "resumed_judgements": self.resumed,
                            "resumed_reassessments": self.reassess_resumed,
                            "updated_at": datetime.now(timezone.utc).isoformat()})
            _archive(self.out_dir / "summary.json")
            _write_json(self.out_dir / "summary.json", summary)
            return summary

    def run(self, episode_ids: Optional[list[str]] = None) -> dict:
        by_episode, sources = {}, {}
        for path in sorted((self.run_dir / "candidates").glob("EP*.json")):
            gen = EpisodeGeneration.model_validate_json(path.read_text(encoding="utf-8"))
            if episode_ids and gen.episode_id not in episode_ids:
                continue
            by_episode[gen.episode_id], sources[gen.episode_id] = gen.candidates, str(path)
        return self.run_candidates(by_episode, sources)


def load_selected_candidates(run_dir: Path, ids: list[str], decisions_dirs: list[Path]) -> dict[str, list[Candidate]]:
    """Existing candidates by ID, grouped by episode in the given order. Originals come from the generation
    files; repaired candidates (`_R1`) from stored judge decisions, so nothing is regenerated or re-repaired."""
    wanted, found = list(dict.fromkeys(ids)), {}
    for path in sorted((Path(run_dir) / "candidates").glob("EP*.json")):
        for c in EpisodeGeneration.model_validate_json(path.read_text(encoding="utf-8")).candidates:
            if c.candidate_id in wanted:
                found[c.candidate_id] = c
    for directory in decisions_dirs:
        for path in sorted(Path(directory).glob("EP*.json")):
            for raw in json.loads(path.read_text(encoding="utf-8"))["decisions"]:
                rc = raw.get("repaired_candidate")
                if rc and rc["candidate_id"] in wanted and rc["candidate_id"] not in found:
                    found[rc["candidate_id"]] = Candidate.model_validate(rc)
    missing = [i for i in wanted if i not in found]
    if missing:
        raise ValueError(f"candidates not found: {', '.join(missing)}")
    by_episode: dict[str, list[Candidate]] = {}
    for cid in wanted:
        by_episode.setdefault(found[cid].episode_id, []).append(found[cid])
    return by_episode


def run_judging(run_dir: Path, episodes: dict[int, Episode], client: LLMClient, *, temperature: float,
                max_tokens: int, settings: Optional[dict] = None, refresh: bool = False,
                episode_ids: Optional[list[str]] = None, **kw) -> dict:
    return JudgeRun(run_dir, episodes, client, temperature=temperature, max_tokens=max_tokens, settings=settings,
                    refresh=refresh, **kw).run(episode_ids)


# --- last-30-seconds baseline -----------------------------------------------------------------------

def baseline_candidate(episode: Episode, seconds: int = 30) -> Candidate:
    """Naive baseline: the episode's final ~`seconds`, starting at the latest block start that does not cut
    dialogue. Hook/cliffhanger fields are placeholders; the judge picks its own beats."""
    blocks = episode.time_ordered_blocks()
    end = max(b.end_ms for b in blocks)
    end_block = [b for b in blocks if b.end_ms == end][-1]
    target = end - seconds * 1000
    starts = sorted({b.start_ms for b in blocks if b.start_ms <= target}, reverse=True)
    start = next((t for t in starts if not _dialogue_cut(episode, t)), starts[-1] if starts else blocks[0].start_ms)
    start_block = next(b for b in blocks if b.start_ms == start)
    c = candidate_from_blocks(episode, f"{episode.episode_id}_BASE{seconds}", start_block.block_id,
                              end_block.block_id, start_block.block_id, end_block.block_id)
    c.flags.append(f"BASELINE_LAST_{seconds}S")
    return c


def run_baseline(run_dir: Path, episodes: dict[int, Episode], client: LLMClient, episode_ids: list[str], *,
                 temperature: float, max_tokens: int, prompt_version: str = PROMPT_VERSION,
                 settings: Optional[dict] = None, **kw) -> dict:
    out = judge_dir(run_dir, client.provider.name, client.provider.model, prompt_version) / "baseline"
    jr = JudgeRun(run_dir, episodes, client, temperature=temperature, max_tokens=max_tokens, settings=settings,
                  prompt_version=prompt_version, out_dir=out, allow_repair=False, **kw)
    by_episode = {eid: [baseline_candidate(episodes[int(eid[2:])])] for eid in episode_ids}
    return jr.run_candidates(by_episode, {eid: "baseline:last-30s" for eid in episode_ids})


