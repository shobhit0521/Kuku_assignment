"""Data models for parsed screenplays.

Canonical times are integer milliseconds; the original timestamp strings are kept alongside them.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr, computed_field

from .timecode import format_timestamp


class BlockType(str, Enum):
    CAMERA = "CAMERA"
    ACTION = "ACTION"
    DIALOGUE = "DIALOGUE"
    SOUND_FX = "SOUND_FX"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Anomaly(BaseModel):
    code: str
    severity: Severity
    message: str
    episode_id: Optional[str] = None
    block_id: Optional[str] = None
    scene_id: Optional[str] = None
    line_number: Optional[int] = None
    details: dict = Field(default_factory=dict)


class Block(BaseModel):
    block_id: str
    episode_id: str
    file_index: int
    scene_id: Optional[str]
    block_type: BlockType
    label_raw: str
    speaker: Optional[str] = None
    delivery: list[str] = Field(default_factory=list)
    offscreen: Optional[str] = None
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    text: str
    line_number: int

    @property
    def has_valid_time(self) -> bool:
        return self.start_ms is not None and self.end_ms is not None and self.end_ms >= self.start_ms

    @computed_field
    @property
    def duration_ms(self) -> Optional[int]:
        return self.end_ms - self.start_ms if self.has_valid_time else None

    def intersects(self, start_ms: int, end_ms: int) -> bool:
        return self.has_valid_time and self.start_ms < end_ms and self.end_ms > start_ms


class Scene(BaseModel):
    scene_id: str
    episode_id: str
    number: Optional[int]
    heading: Optional[str] = None
    start_raw: Optional[str] = None
    end_raw: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    line_number: int
    block_ids: list[str] = Field(default_factory=list)


class TimelineStats(BaseModel):
    duration_ms: int
    camera_coverage_ratio: float
    max_camera_gap_ms: int
    uncovered_intervals_ms: list[tuple[int, int]] = Field(default_factory=list)


class Episode(BaseModel):
    episode_id: str
    episode_number: int
    source_file: str
    source_sha256: str
    parser_version: str
    logline: Optional[str] = None
    continuity_notes: Optional[str] = None
    extra_header: dict[str, str] = Field(default_factory=dict)
    scenes: list[Scene] = Field(default_factory=list)
    blocks: list[Block] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    stats: Optional[TimelineStats] = None

    _by_id: dict[str, Block] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context) -> None:
        self._by_id = {b.block_id: b for b in self.blocks}

    def get_block(self, block_id: str) -> Block:
        """Look up a block by ID; raises KeyError for unknown IDs."""
        return self._by_id[block_id]

    def has_block(self, block_id: str) -> bool:
        return block_id in self._by_id

    def time_ordered_blocks(self) -> list[Block]:
        """Blocks with valid times, sorted by (start, end, file position). File order is not time order."""
        timed = [b for b in self.blocks if b.has_valid_time]
        return sorted(timed, key=lambda b: (b.start_ms, b.end_ms, b.file_index))

    def blocks_intersecting(self, start_ms: int, end_ms: int) -> list[Block]:
        """Blocks on any track whose interval overlaps [start_ms, end_ms), in time order. Empty interval -> []."""
        if end_ms <= start_ms:
            return []
        return [b for b in self.time_ordered_blocks() if b.intersects(start_ms, end_ms)]

    def block_timestamps(self, block_id: str) -> tuple[str, str]:
        block = self.get_block(block_id)
        if not block.has_valid_time:
            raise ValueError(f"{block_id} has no valid time")
        return format_timestamp(block.start_ms), format_timestamp(block.end_ms)


class CandidateStatus(str, Enum):
    GROUNDED = "grounded"
    REJECTED = "rejected"


class Reason(BaseModel):
    code: str
    message: str


class RepairAction(BaseModel):
    action: str
    detail: str
    block_ids: list[str] = Field(default_factory=list)


class RepairProvenance(BaseModel):
    """Where a repaired candidate came from. The original candidate record is never modified."""
    original_candidate_id: str
    original_start_block_id: str
    original_end_block_id: str
    original_start_ms: int
    original_end_ms: int
    new_start_ms: int
    new_end_ms: int
    actions: list[RepairAction] = Field(default_factory=list)
    suspense_beat_id: Optional[str] = None
    suspense_beat_source: Optional[str] = None
    validator_version: str


class Candidate(BaseModel):
    """One model proposal after grounding. Times come only from parsed blocks, never from the model.

    "grounded" means every reference resolved to real source blocks; it is not a quality judgement.
    Duration, hook-window and cliffhanger-gap values are computed here but enforced in Phase 4.
    """
    candidate_id: str
    episode_id: str
    status: CandidateStatus
    rejection_reasons: list[Reason] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    proposal: dict = Field(default_factory=dict)

    start_block_id: Optional[str] = None
    end_block_id: Optional[str] = None
    hook_block_id: Optional[str] = None
    cliffhanger_block_id: Optional[str] = None
    hook_type: Optional[str] = None
    unresolved_question: Optional[str] = None
    new_viewer_context: Optional[str] = None
    why_it_works: Optional[str] = None

    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    duration_ms: Optional[int] = None
    hook_offset_ms: Optional[int] = None
    cliffhanger_end_gap_ms: Optional[int] = None
    scene_ids: list[str] = Field(default_factory=list)
    segment_block_ids: list[str] = Field(default_factory=list)
    hook_source_text: Optional[str] = None
    cliffhanger_source_text: Optional[str] = None

    hard_rule_validation: str = "pending (Phase 4)"
    repair: Optional[RepairProvenance] = None


class EpisodeGeneration(BaseModel):
    episode_id: str
    status: str
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    candidates: list[Candidate] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    calls: list[dict] = Field(default_factory=list)


class BeatSource(str, Enum):
    """Who chose a semantic beat. Only judge/human choices make a validation final."""
    JUDGE = "judge"
    HUMAN = "human"
    TEST = "test"
    GENERATOR_ADVISORY = "generator_advisory"


AUTHORITATIVE_BEAT_SOURCES = {BeatSource.JUDGE, BeatSource.HUMAN}


class BeatInput(BaseModel):
    block_id: str
    source: BeatSource


class SemanticInputs(BaseModel):
    """What a later blind judge supplies. Python never decides whether an ending is unresolved."""
    hook: Optional[BeatInput] = None
    suspense_beat: Optional[BeatInput] = None
    outcome: Optional[str] = None  # "unresolved" | "partially_resolved" | "resolved"; recorded, not enforced


class RuleStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    SKIPPED = "skipped"


class RuleCheck(BaseModel):
    rule: str
    status: RuleStatus
    message: str
    block_ids: list[str] = Field(default_factory=list)
    value_ms: Optional[int] = None
    repairable: bool = False


class ClipEnding(BaseModel):
    """Deterministic facts about how a clip ends. The suspense-bearing beat itself is a semantic input."""
    clip_end_ms: int
    last_dialogue_block_id: Optional[str] = None
    last_dialogue_end_ms: Optional[int] = None
    last_story_beat_id: Optional[str] = None
    last_story_beat_end_ms: Optional[int] = None
    trailing_after_last_story_ms: Optional[int] = None
    transition_block_ids: list[str] = Field(default_factory=list)
    score_cue_block_ids: list[str] = Field(default_factory=list)
    suspense_beat_options: list[str] = Field(default_factory=list)
    suspense_beat_id: Optional[str] = None
    suspense_beat_source: Optional[str] = None
    suspense_beat_end_ms: Optional[int] = None
    trailing_after_suspense_ms: Optional[int] = None
    outcome: Optional[str] = None


class ValidationResult(BaseModel):
    candidate_id: str
    episode_id: str
    validator_version: str
    status: RuleStatus  # pass | fail | pending (all objective rules pass, suspense beat not yet supplied)
    provisional: bool  # True unless the semantic beats came from a judge or a human
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    hook_block_id: Optional[str] = None
    hook_source: Optional[str] = None
    hook_offset_ms: Optional[int] = None
    checks: list[RuleCheck] = Field(default_factory=list)
    ending: Optional[ClipEnding] = None

    @property
    def failed(self) -> list[RuleCheck]:
        return [c for c in self.checks if c.status == RuleStatus.FAIL]


class RepairOutcome(BaseModel):
    status: str  # not_needed | repaired | unrepairable
    reason: str
    actions: list[RepairAction] = Field(default_factory=list)
    attempted_start_ms: Optional[int] = None
    attempted_end_ms: Optional[int] = None
    repaired_candidate: Optional[Candidate] = None
    repaired_validation: Optional[ValidationResult] = None


class CandidateValidation(BaseModel):
    candidate_id: str
    validation: ValidationResult
    repair: RepairOutcome


class EpisodeValidation(BaseModel):
    episode_id: str
    validator_version: str
    settings: dict
    source_file: str
    source_sha256: str
    records: list[CandidateValidation] = Field(default_factory=list)


class DimensionAssessment(BaseModel):
    """One judged dimension after Python's evidence checks. valid=False means the assessment is unusable."""
    name: str
    score: Optional[int] = None
    valid: bool = False
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    block_id: Optional[str] = None            # hook: selected hook block
    suspense_beat_id: Optional[str] = None    # cliffhanger
    mechanism: Optional[str] = None
    outcome: Optional[str] = None
    open_question: Optional[str] = None
    escalation_beats: list[str] = Field(default_factory=list)  # narrative_tension
    blind_summary: Optional[str] = None       # context_independence
    confusions: list[dict] = Field(default_factory=list)
    revealed: list[str] = Field(default_factory=list)          # curiosity_gap
    withheld: list[str] = Field(default_factory=list)
    balance: Optional[str] = None
    emotion: Optional[str] = None             # emotional_impact
    review: bool = False                      # contradiction survived the targeted reassessment
    review_reasons: list[str] = Field(default_factory=list)


class DNAAttribute(BaseModel):
    value: str = "unclassified"
    evidence: list[str] = Field(default_factory=list)
    note: Optional[str] = None


class PromoDNA(BaseModel):
    archetype: DNAAttribute = Field(default_factory=DNAAttribute)
    archetype_other_label: Optional[str] = None
    archetype_secondary: Optional[str] = None
    archetype_confidence: Optional[str] = None
    hook_mechanism: DNAAttribute = Field(default_factory=DNAAttribute)
    emotional_trigger: DNAAttribute = Field(default_factory=DNAAttribute)
    stakes: DNAAttribute = Field(default_factory=DNAAttribute)
    cliffhanger_mechanism: DNAAttribute = Field(default_factory=DNAAttribute)
    context_dependency: Optional[str] = None
    missing_context: list[str] = Field(default_factory=list)
    curiosity_revealed: list[str] = Field(default_factory=list)
    curiosity_withheld: list[str] = Field(default_factory=list)
    curiosity_balance: Optional[str] = None
    appeal: Optional[str] = None
    supporting_block_ids: list[str] = Field(default_factory=list)


class SoftRejection(BaseModel):
    code: str
    severity: str
    explanation: str
    block_ids: list[str] = Field(default_factory=list)
    source: str = "judge"


class Judgement(BaseModel):
    """Raw blind judgement of one clip, after schema and evidence validation. Never combined with ranking."""
    candidate_id: str
    episode_id: str
    status: str  # valid | invalid
    invalid_reasons: list[str] = Field(default_factory=list)
    failure_kind: Optional[str] = None  # provider_error | invalid_output | invalid_assessment
    invocation_id: Optional[str] = None
    provider: str
    model: str
    prompt_version: str
    prompt_sha256: str
    clip_sha256: str
    clip_start_ms: int
    clip_end_ms: int
    masked_terms: int = 0
    eligible_hook_ids: list[str] = Field(default_factory=list)
    eligible_beat_ids: list[str] = Field(default_factory=list)
    dimensions: dict[str, DimensionAssessment] = Field(default_factory=dict)
    promo_dna: Optional[PromoDNA] = None
    soft_rejections: list[SoftRejection] = Field(default_factory=list)
    dropped_rejections: list[dict] = Field(default_factory=list)
    response_errors: list[str] = Field(default_factory=list)
    raw_response_text: Optional[str] = None
    calls: list[dict] = Field(default_factory=list)
    judged_at: str


class ConsistencyFlag(BaseModel):
    dimension: str
    code: str
    detail: str


class Reassessment(BaseModel):
    """One targeted reassessment of a judgement's flagged dimensions. Stored apart from the raw judgement."""
    candidate_id: str
    checker_version: str
    judgement_clip_sha256: str
    judgement_prompt_sha256: str
    status: str  # not_needed | resolved | unresolved | invalid_reply | provider_error
    flags: list[ConsistencyFlag] = Field(default_factory=list)
    remaining_flags: list[ConsistencyFlag] = Field(default_factory=list)
    original_scores: dict[str, int] = Field(default_factory=dict)
    revised_scores: dict[str, int] = Field(default_factory=dict)
    revised_dimensions: dict[str, DimensionAssessment] = Field(default_factory=dict)
    note: Optional[str] = None
    raw_response_text: Optional[str] = None
    calls: list[dict] = Field(default_factory=list)
    invocation_id: Optional[str] = None
    created_at: str


class GateResult(BaseModel):
    gate: str
    passed: bool
    detail: str


class QualityDecision(BaseModel):
    """Combines Python's hard rules (authoritative) with the judge's quality gates for one clip."""
    candidate_id: str
    episode_id: str
    verdict: str  # eligible | quality_fail | hard_fail | judge_invalid | not_judged
    eligible: bool = False
    weighted_score: Optional[float] = None
    hard_rules: Optional[str] = None
    hard_rule_failures: list[str] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    soft_rejection_codes: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, int] = Field(default_factory=dict)
    archetype: Optional[str] = None
    review_flags: list[str] = Field(default_factory=list)
    reassessment_status: Optional[str] = None
    validation: Optional[ValidationResult] = None
    judgement_file: Optional[str] = None
    repaired_from: Optional[str] = None
    repair_status: Optional[str] = None
    repair_reason: Optional[str] = None
    repaired_candidate: Optional[Candidate] = None
    note: Optional[str] = None


class EpisodeSummary(BaseModel):
    episode_id: str
    episode_number: int
    source_file: str
    duration_ms: int
    scene_count: int
    block_count: int
    anomaly_count: int


class PackSummary(BaseModel):
    pack_name: str
    parser_version: str
    episode_count: int
    scene_count: int
    block_count: int
    block_type_counts: dict[str, int]
    anomaly_counts: dict[str, int]
    pack_anomalies: list[Anomaly] = Field(default_factory=list)
    speakers: dict[str, list[int]] = Field(default_factory=dict)
    episodes: list[EpisodeSummary] = Field(default_factory=list)
