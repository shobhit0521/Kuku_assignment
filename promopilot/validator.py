"""Stage 3: deterministic hard-rule validation and conservative boundary repair.

Python owns objective facts only: references, ordering, duration, the hook window, dialogue boundaries,
transition-only endings and the trailing time after a supplied suspense-bearing beat. Which beat carries
the suspense, and whether it is unresolved, is a semantic judgement supplied from outside (later: the
blind judge). The generator's cliffhanger is never used as that beat unless a caller explicitly passes it
as an advisory input, and then results are marked provisional.

Original candidates are never modified. A repair is one deterministic attempt that produces a new
candidate (`<id>_R1`) with provenance, and is re-validated against every rule.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import config
from .schemas import (AUTHORITATIVE_BEAT_SOURCES, BeatInput, BeatSource, Block, BlockType, Candidate,
                      CandidateStatus, CandidateValidation, ClipEnding, Episode, EpisodeGeneration,
                      EpisodeValidation, RepairAction, RepairOutcome, RepairProvenance, RuleCheck, RuleStatus,
                      SemanticInputs, ValidationResult)
from .timecode import format_timestamp

RULES = ("REFERENCES_VALID", "BOUNDARY_ORDER", "DURATION_RANGE", "HOOK_WINDOW", "DIALOGUE_BOUNDARY",
         "ENDING_NOT_TRANSITION_ONLY", "SUSPENSE_BEAT_IN_CLIP", "SUSPENSE_BEAT_IS_STORY_BEAT",
         "SUSPENSE_TAIL_TOLERANCE")
SUSPENSE_RULES = ("SUSPENSE_BEAT_IN_CLIP", "SUSPENSE_BEAT_IS_STORY_BEAT", "SUSPENSE_TAIL_TOLERANCE")
REPAIRABLE_RULES = {"DURATION_RANGE", "DIALOGUE_BOUNDARY", "ENDING_NOT_TRANSITION_ONLY", "SUSPENSE_TAIL_TOLERANCE"}
EPISODE_END_WINDOW_MS = 2000
MAX_EXPANSION_ROUNDS = 50

# A block is a transition only if its whole text is a transition instruction. A shot that merely
# mentions "cut to black" while describing a character or event is a real beat.
_MODIFIER = r"(?:(?:slow|slowly|quick|hard|abrupt|abruptly|sudden|suddenly|smash|final|then|and|the|screen|picture|image|frame)\s+)*"
TRANSITION_RE = re.compile(
    rf"^{_MODIFIER}(?:cut|cuts|fade|fades|dissolve|dissolves|goes|smash cut|match cut)\s+"
    r"(?:to\s+black|to\s+white|out|to\s+credits|to\s+title\s+card|to\s+end\s+credits)"
    r"(?:\s+(?:slowly|abruptly))?$"
    r"|^(?:black screen|black|end credits|credits roll|roll credits|title card|end of episode|to be continued|"
    r"the end|fade in|smash cut)$")
# Non-diegetic score cues (music only). They may sit in the tail but cannot carry the suspense.
_MUSIC_RE = re.compile(r"\b(music|musical|score|chord|chords|drone|drones|strings?|swell|swells|swelling|sting|"
                       r"stinger|melody|theme|crescendo|soundtrack|orchestral|choir|piano|violins?|cello)\b")
_DIEGETIC_RE = re.compile(r"\b(door|doors|glass|crash|crashes|crashing|footsteps?|scream|screams|screaming|thunder|"
                          r"rain|fire|flames?|gun|gunshot|shot|knock|knocks|knocking|bell|bells|wood|wooden|beam|"
                          r"horse|hooves|breath|breathing|gasp|slap|thud|click|clicks|latch|wind|creak|creaks|"
                          r"creaking|shatter|shatters|splash|cough|coughing|sob|sobs|voice|voices|bang|explosion)\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()


def is_transition(block: Block) -> bool:
    return bool(TRANSITION_RE.match(_norm(block.text)))


def is_score_cue(block: Block) -> bool:
    if block.block_type != BlockType.SOUND_FX:
        return False
    text = _norm(block.text)
    return bool(_MUSIC_RE.search(text)) and not _DIEGETIC_RE.search(text)


def beat_class(block: Block) -> str:
    if is_transition(block):
        return "transition"
    if is_score_cue(block):
        return "score_cue"
    return "story"


# --- candidate construction ------------------------------------------------------------------------

def _boundary_blocks(episode: Episode, start_ms: int, end_ms: int, prefer_end: Optional[str] = None
                     ) -> tuple[Block, Block]:
    ordered = episode.time_ordered_blocks()
    start = next(b for b in ordered if b.start_ms == start_ms)
    ends = [b for b in ordered if b.end_ms == end_ms]
    end = next((b for b in ends if b.block_id == prefer_end), ends[-1])
    return start, end


def candidate_from_blocks(episode: Episode, candidate_id: str, start_block_id: str, end_block_id: str,
                          hook_block_id: str, cliffhanger_block_id: str, *, base: Optional[Candidate] = None
                          ) -> Candidate:
    """Build a grounded candidate from block IDs (times from the blocks). Generator claims are copied
    from `base` when given; nothing is invented."""
    start, end = episode.get_block(start_block_id), episode.get_block(end_block_id)
    hook, cliff = episode.get_block(hook_block_id), episode.get_block(cliffhanger_block_id)
    s, e = start.start_ms, end.end_ms
    segment = episode.blocks_intersecting(s, e)
    claims = {k: getattr(base, k) for k in ("hook_type", "unresolved_question", "new_viewer_context",
                                            "why_it_works")} if base else {}
    duration = episode.stats.duration_ms if episode.stats else max(b.end_ms for b in episode.time_ordered_blocks())
    flags = ["ENDS_AT_EPISODE_END"] if duration - e <= EPISODE_END_WINDOW_MS else []
    return Candidate(
        candidate_id=candidate_id, episode_id=episode.episode_id, status=CandidateStatus.GROUNDED, flags=flags,
        start_block_id=start.block_id, end_block_id=end.block_id, hook_block_id=hook.block_id,
        cliffhanger_block_id=cliff.block_id, start_ms=s, end_ms=e, start_ts=format_timestamp(s),
        end_ts=format_timestamp(e), duration_ms=e - s, hook_offset_ms=hook.start_ms - s,
        cliffhanger_end_gap_ms=e - cliff.end_ms, scene_ids=sorted({b.scene_id for b in segment if b.scene_id}),
        segment_block_ids=[b.block_id for b in segment], hook_source_text=hook.text,
        cliffhanger_source_text=cliff.text, **claims)


def advisory_inputs(candidate: Candidate, *, include_generator_cliffhanger: bool = False) -> SemanticInputs:
    """Default semantic inputs: the generator's hook as advisory. Its cliffhanger is only used when a caller
    explicitly asks for a provisional preview."""
    hook = BeatInput(block_id=candidate.hook_block_id, source=BeatSource.GENERATOR_ADVISORY) \
        if candidate.hook_block_id else None
    beat = BeatInput(block_id=candidate.cliffhanger_block_id, source=BeatSource.GENERATOR_ADVISORY) \
        if include_generator_cliffhanger and candidate.cliffhanger_block_id else None
    return SemanticInputs(hook=hook, suspense_beat=beat)


def _resolve_inputs(candidate: Candidate, inputs: Optional[SemanticInputs]) -> SemanticInputs:
    """Supplied beats win; a missing hook falls back to the generator's hook as an advisory input."""
    if inputs is None:
        return advisory_inputs(candidate)
    if inputs.hook is None and candidate.hook_block_id:
        return inputs.model_copy(update={"hook": advisory_inputs(candidate).hook})
    return inputs


# --- validation ------------------------------------------------------------------------------------

def _dialogue_cut(episode: Episode, at_ms: int) -> list[Block]:
    return [b for b in episode.time_ordered_blocks()
            if b.block_type == BlockType.DIALOGUE and b.start_ms < at_ms < b.end_ms]


def analyse_ending(episode: Episode, start_ms: int, end_ms: int, settings: dict,
                   suspense: Optional[BeatInput] = None, outcome: Optional[str] = None) -> ClipEnding:
    inside = episode.blocks_intersecting(start_ms, end_ms)
    classes = {b.block_id: beat_class(b) for b in inside}
    story = [b for b in inside if classes[b.block_id] == "story"]
    dialogue = [b for b in inside if b.block_type == BlockType.DIALOGUE]

    def eff_end(b: Block) -> int:
        return min(b.end_ms, end_ms)

    last_story = max(story, key=lambda b: (eff_end(b), b.start_ms, b.file_index), default=None)
    last_dialogue = max(dialogue, key=lambda b: (b.end_ms, b.start_ms, b.file_index), default=None)
    window_start = end_ms - settings["ending_window_ms"]
    ending = ClipEnding(
        clip_end_ms=end_ms,
        last_dialogue_block_id=last_dialogue.block_id if last_dialogue else None,
        last_dialogue_end_ms=last_dialogue.end_ms if last_dialogue else None,
        last_story_beat_id=last_story.block_id if last_story else None,
        last_story_beat_end_ms=eff_end(last_story) if last_story else None,
        trailing_after_last_story_ms=end_ms - eff_end(last_story) if last_story else None,
        transition_block_ids=[b.block_id for b in inside if classes[b.block_id] == "transition"],
        score_cue_block_ids=[b.block_id for b in inside if classes[b.block_id] == "score_cue"],
        suspense_beat_options=[b.block_id for b in story if b.end_ms > window_start and b.start_ms >= start_ms],
        outcome=outcome)
    if suspense and episode.has_block(suspense.block_id):
        beat = episode.get_block(suspense.block_id)
        ending.suspense_beat_id, ending.suspense_beat_source = beat.block_id, suspense.source.value
        ending.suspense_beat_end_ms = beat.end_ms
        ending.trailing_after_suspense_ms = max(0, end_ms - beat.end_ms)
    elif suspense:
        ending.suspense_beat_id, ending.suspense_beat_source = suspense.block_id, suspense.source.value
    return ending


def validate_candidate(episode: Episode, candidate: Candidate, inputs: Optional[SemanticInputs] = None,
                       settings: Optional[dict] = None) -> ValidationResult:
    """Apply every hard rule. Never modifies `candidate`."""
    settings = settings or config.validation_settings()
    inputs = _resolve_inputs(candidate, inputs)
    beats = [b for b in (inputs.hook, inputs.suspense_beat) if b]
    provisional = not beats or any(b.source not in AUTHORITATIVE_BEAT_SOURCES for b in beats) \
        or inputs.suspense_beat is None
    result = ValidationResult(candidate_id=candidate.candidate_id, episode_id=candidate.episode_id,
                              validator_version=config.VALIDATOR_VERSION, status=RuleStatus.FAIL,
                              provisional=provisional)
    checks = result.checks

    def add(rule, status, message, block_ids=(), value_ms=None):
        checks.append(RuleCheck(rule=rule, status=status, message=message, block_ids=list(block_ids),
                                value_ms=value_ms, repairable=status == RuleStatus.FAIL and rule in REPAIRABLE_RULES))

    def skip_rest(reason):
        done = {c.rule for c in checks}
        for rule in RULES:
            if rule not in done:
                add(rule, RuleStatus.SKIPPED, reason)

    # References and episode consistency.
    problems, ref_ids = [], []
    if candidate.status != CandidateStatus.GROUNDED:
        codes = ", ".join(r.code for r in candidate.rejection_reasons) or "not grounded"
        problems.append(f"rejected at grounding ({codes})")
    if candidate.episode_id != episode.episode_id:
        problems.append(f"candidate belongs to {candidate.episode_id}, not {episode.episode_id}")
    roles = {"start": candidate.start_block_id, "end": candidate.end_block_id,
             "hook": inputs.hook.block_id if inputs.hook else None}
    for role, bid in roles.items():
        if bid is None:
            problems.append(f"{role} block missing")
        elif not bid.startswith(episode.episode_id + "_") or not episode.has_block(bid):
            problems.append(f"{role} block {bid!r} is not a block of {episode.episode_id}")
        elif not episode.get_block(bid).has_valid_time:
            problems.append(f"{role} block {bid} has no valid time")
        else:
            ref_ids.append(bid)
    if problems:
        add("REFERENCES_VALID", RuleStatus.FAIL, "; ".join(problems), ref_ids)
        skip_rest("references invalid")
        return result
    add("REFERENCES_VALID", RuleStatus.PASS, "all block IDs exist in this episode and have times", ref_ids)

    start_b, end_b = episode.get_block(candidate.start_block_id), episode.get_block(candidate.end_block_id)
    s, e = start_b.start_ms, end_b.end_ms
    if e <= s:
        add("BOUNDARY_ORDER", RuleStatus.FAIL, f"end {format_timestamp(e)} is not after start {format_timestamp(s)}",
            [start_b.block_id, end_b.block_id])
        skip_rest("boundary order invalid")
        return result
    add("BOUNDARY_ORDER", RuleStatus.PASS, f"{format_timestamp(s)}-{format_timestamp(e)}",
        [start_b.block_id, end_b.block_id])
    result.start_ms, result.end_ms, result.duration_ms = s, e, e - s

    d = e - s
    lo, hi = settings["min_clip_ms"], settings["max_clip_ms"]
    add("DURATION_RANGE", RuleStatus.PASS if lo <= d <= hi else RuleStatus.FAIL,
        f"duration {d / 1000:.2f}s (allowed {lo / 1000:g}-{hi / 1000:g}s inclusive)", value_ms=d)

    hook = episode.get_block(inputs.hook.block_id)
    offset = hook.start_ms - s
    result.hook_block_id, result.hook_source, result.hook_offset_ms = hook.block_id, inputs.hook.source.value, offset
    window = settings["hook_window_ms"]
    if offset < 0:
        add("HOOK_WINDOW", RuleStatus.FAIL, f"hook block starts {-offset / 1000:.2f}s before the clip",
            [hook.block_id], offset)
    else:
        add("HOOK_WINDOW", RuleStatus.PASS if offset <= window else RuleStatus.FAIL,
            f"hook starts {offset / 1000:.2f}s after clip start (limit {window / 1000:g}s; "
            f"hook source: {inputs.hook.source.value})", [hook.block_id], offset)

    cut = sorted({b.block_id for b in _dialogue_cut(episode, s) + _dialogue_cut(episode, e)})
    add("DIALOGUE_BOUNDARY", RuleStatus.FAIL if cut else RuleStatus.PASS,
        f"boundary cuts dialogue block(s) {', '.join(cut)}" if cut else "no dialogue block is cut", cut)

    ending = analyse_ending(episode, s, e, settings, inputs.suspense_beat, inputs.outcome)
    result.ending = ending
    tail = settings["suspense_tail_ms"]
    if ending.last_story_beat_id is None:
        add("ENDING_NOT_TRANSITION_ONLY", RuleStatus.FAIL, "clip contains no story beat")
    else:
        trailing = ending.trailing_after_last_story_ms
        ok = trailing <= tail
        add("ENDING_NOT_TRANSITION_ONLY", RuleStatus.PASS if ok else RuleStatus.FAIL,
            f"last story beat {ending.last_story_beat_id} ends {trailing / 1000:.2f}s before the cut"
            + ("" if ok else f"; the rest is transition/score only (limit {tail / 1000:g}s)"),
            [ending.last_story_beat_id] + ending.transition_block_ids + ending.score_cue_block_ids, trailing)

    beat_in = inputs.suspense_beat
    if beat_in is None:
        for rule in SUSPENSE_RULES:
            add(rule, RuleStatus.PENDING, "suspense-bearing beat not supplied yet (semantic judge input)")
    elif not beat_in.block_id.startswith(episode.episode_id + "_") or not episode.has_block(beat_in.block_id):
        add("SUSPENSE_BEAT_IN_CLIP", RuleStatus.FAIL, f"{beat_in.block_id!r} is not a block of {episode.episode_id}")
        for rule in SUSPENSE_RULES[1:]:
            add(rule, RuleStatus.SKIPPED, "suspense beat invalid")
    else:
        beat = episode.get_block(beat_in.block_id)
        inside = beat.intersects(s, e) and beat.start_ms >= s
        add("SUSPENSE_BEAT_IN_CLIP", RuleStatus.PASS if inside else RuleStatus.FAIL,
            f"{beat.block_id} ({beat.start_raw}-{beat.end_raw}) "
            + ("is inside the clip" if inside else f"is not inside {format_timestamp(s)}-{format_timestamp(e)}")
            + f" (source: {beat_in.source.value})", [beat.block_id])
        cls = beat_class(beat)
        add("SUSPENSE_BEAT_IS_STORY_BEAT", RuleStatus.PASS if cls == "story" else RuleStatus.FAIL,
            f"{beat.block_id} is a {cls}" + ("" if cls == "story" else " and cannot carry the suspense"),
            [beat.block_id])
        if inside:
            trailing = ending.trailing_after_suspense_ms
            add("SUSPENSE_TAIL_TOLERANCE", RuleStatus.PASS if trailing <= tail else RuleStatus.FAIL,
                f"clip ends {trailing / 1000:.2f}s after the suspense beat (limit {tail / 1000:g}s)",
                [beat.block_id], trailing)
        else:
            add("SUSPENSE_TAIL_TOLERANCE", RuleStatus.SKIPPED, "suspense beat not inside clip")

    statuses = {c.status for c in checks}
    result.status = RuleStatus.FAIL if RuleStatus.FAIL in statuses else \
        RuleStatus.PENDING if RuleStatus.PENDING in statuses else RuleStatus.PASS
    return result


# --- repair ----------------------------------------------------------------------------------------

def repair_candidate(episode: Episode, candidate: Candidate, validation: ValidationResult,
                     inputs: Optional[SemanticInputs] = None, settings: Optional[dict] = None) -> RepairOutcome:
    """One conservative repair attempt: trim the tail, expand for cut dialogue, extend the start backwards
    if too short. The result is re-validated against every rule. Never modifies `candidate`."""
    settings = settings or config.validation_settings()
    inputs = _resolve_inputs(candidate, inputs)
    failed = validation.failed
    if not failed:
        return RepairOutcome(status="not_needed", reason=f"validation status is {validation.status.value}")
    fixed = [c.rule for c in failed if not c.repairable]
    if fixed:
        return RepairOutcome(status="unrepairable", reason="non-repairable rule(s) failed: " + ", ".join(fixed))
    duration_check = next((c for c in failed if c.rule == "DURATION_RANGE"), None)
    if duration_check and duration_check.value_ms > settings["max_clip_ms"]:
        return RepairOutcome(status="unrepairable", reason="clip is longer than the maximum; shortening the "
                                                           "start is outside the conservative repair rules")

    s, e = validation.start_ms, validation.end_ms
    hook = episode.get_block(inputs.hook.block_id)
    beat = episode.get_block(inputs.suspense_beat.block_id) if inputs.suspense_beat else None
    actions: list[RepairAction] = []
    failed_rules = {c.rule for c in failed}

    if "SUSPENSE_TAIL_TOLERANCE" in failed_rules and beat and beat.end_ms < e:
        actions.append(RepairAction(action="trim_end_to_suspense_beat", block_ids=[beat.block_id],
                                    detail=f"end {format_timestamp(e)} -> {format_timestamp(beat.end_ms)}"))
        e = beat.end_ms
    elif "ENDING_NOT_TRANSITION_ONLY" in failed_rules and validation.ending.last_story_beat_id:
        last = validation.ending.last_story_beat_end_ms
        actions.append(RepairAction(action="trim_transition_tail", block_ids=[validation.ending.last_story_beat_id],
                                    detail=f"end {format_timestamp(e)} -> {format_timestamp(last)}"))
        e = last

    for _ in range(MAX_EXPANSION_ROUNDS):
        at_start, at_end = _dialogue_cut(episode, s), _dialogue_cut(episode, e)
        if not at_start and not at_end:
            break
        if at_start:
            new_s = min(b.start_ms for b in at_start)
            actions.append(RepairAction(action="expand_start_for_dialogue", block_ids=[b.block_id for b in at_start],
                                        detail=f"start {format_timestamp(s)} -> {format_timestamp(new_s)}"))
            s = new_s
        if at_end:
            new_e = max(b.end_ms for b in at_end)
            actions.append(RepairAction(action="expand_end_for_dialogue", block_ids=[b.block_id for b in at_end],
                                        detail=f"end {format_timestamp(e)} -> {format_timestamp(new_e)}"))
            e = new_e

    if e - s < settings["min_clip_ms"]:
        earlier = sorted({b.start_ms for b in episode.time_ordered_blocks() if b.start_ms < s}, reverse=True)
        steps = []
        for t in earlier:
            if _dialogue_cut(episode, t):
                continue
            if hook.start_ms - t > settings["hook_window_ms"]:
                break
            steps.append(t)
            if e - t >= settings["min_clip_ms"]:
                break
        if steps:
            actions.append(RepairAction(action="extend_start_backwards",
                                        detail=f"start {format_timestamp(s)} -> {format_timestamp(steps[-1])} "
                                               f"in {len(steps)} block step(s)"))
            s = steps[-1]

    if (s, e) == (validation.start_ms, validation.end_ms) or not actions:
        return RepairOutcome(status="unrepairable", reason="no conservative boundary change is available",
                             actions=actions)

    start_b, end_b = _boundary_blocks(episode, s, e, prefer_end=beat.block_id if beat else None)
    repaired = candidate_from_blocks(episode, f"{candidate.candidate_id}_R1", start_b.block_id, end_b.block_id,
                                     candidate.hook_block_id, candidate.cliffhanger_block_id, base=candidate)
    repaired.flags.append(f"REPAIRED_FROM:{candidate.candidate_id}")
    repaired.repair = RepairProvenance(
        original_candidate_id=candidate.candidate_id, original_start_block_id=candidate.start_block_id,
        original_end_block_id=candidate.end_block_id, original_start_ms=validation.start_ms,
        original_end_ms=validation.end_ms, new_start_ms=s, new_end_ms=e, actions=actions,
        suspense_beat_id=beat.block_id if beat else None,
        suspense_beat_source=inputs.suspense_beat.source.value if inputs.suspense_beat else None,
        validator_version=config.VALIDATOR_VERSION)
    revalidation = validate_candidate(episode, repaired, inputs, settings)
    repaired.hard_rule_validation = f"{revalidation.status.value} (validator {config.VALIDATOR_VERSION})"
    if revalidation.status == RuleStatus.FAIL:
        return RepairOutcome(status="unrepairable",
                             reason="repair attempt still fails: " + ", ".join(c.rule for c in revalidation.failed),
                             actions=actions, attempted_start_ms=s, attempted_end_ms=e,
                             repaired_validation=revalidation)
    return RepairOutcome(status="repaired", reason="all hard rules pass after repair" if revalidation.status ==
                         RuleStatus.PASS else "objective rules pass after repair; suspense beat still pending",
                         actions=actions, attempted_start_ms=s, attempted_end_ms=e, repaired_candidate=repaired,
                         repaired_validation=revalidation)


def validate_and_repair(episode: Episode, candidate: Candidate, inputs: Optional[SemanticInputs] = None,
                        settings: Optional[dict] = None) -> CandidateValidation:
    settings = settings or config.validation_settings()
    inputs = _resolve_inputs(candidate, inputs)
    validation = validate_candidate(episode, candidate, inputs, settings)
    return CandidateValidation(candidate_id=candidate.candidate_id, validation=validation,
                               repair=repair_candidate(episode, candidate, validation, inputs, settings))


# --- run-level -------------------------------------------------------------------------------------

def load_beats_file(path: Path) -> dict[str, SemanticInputs]:
    """JSON: {"EP004_C02": {"suspense_beat": {"block_id": "EP004_B099", "source": "human"}, ...}, ...}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {cid: SemanticInputs.model_validate(v) for cid, v in data.items()}


def run_validation(run_dir: Path, episodes: dict[int, Episode], *, out_name: str = "validation",
                   beats: Optional[dict[str, SemanticInputs]] = None, preview_generator_beats: bool = False,
                   settings: Optional[dict] = None) -> dict:
    """Validate every candidate of a generation run. Reads candidates/*.json, writes <out_name>/*.json."""
    settings = settings or config.validation_settings()
    run_dir = Path(run_dir)
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"validator_version": config.VALIDATOR_VERSION, "settings": settings,
               "preview_generator_beats": preview_generator_beats, "episodes": {}}
    for path in sorted((run_dir / "candidates").glob("EP*.json")):
        raw = path.read_bytes()
        gen = EpisodeGeneration.model_validate_json(raw)
        episode = episodes[int(gen.episode_id[2:])]
        record = EpisodeValidation(episode_id=gen.episode_id, validator_version=config.VALIDATOR_VERSION,
                                   settings=settings, source_file=str(path),
                                   source_sha256=hashlib.sha256(raw).hexdigest())
        for c in gen.candidates:
            inputs = advisory_inputs(c, include_generator_cliffhanger=preview_generator_beats)
            supplied = (beats or {}).get(c.candidate_id)
            if supplied:
                inputs = SemanticInputs(hook=supplied.hook or inputs.hook,
                                        suspense_beat=supplied.suspense_beat or inputs.suspense_beat,
                                        outcome=supplied.outcome)
            record.records.append(validate_and_repair(episode, c, inputs, settings))
        tmp = out_dir / f"{gen.episode_id}.tmp"
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(out_dir / f"{gen.episode_id}.json")
        summary["episodes"][gen.episode_id] = [
            {"candidate_id": r.candidate_id, "status": r.validation.status.value,
             "failed": [c.rule for c in r.validation.failed], "repair": r.repair.status,
             "repaired_status": r.repair.repaired_validation.status.value if r.repair.repaired_validation else None}
            for r in record.records]
    summary["updated_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
