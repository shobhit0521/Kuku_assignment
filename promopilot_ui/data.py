"""Read-only access to persisted PromoPilot outputs for the UI.

Nothing here generates, validates, judges, scores or ranks. Verdicts, scores and ranking order are read from the
files the pipeline wrote; the only computation is joining those files by candidate ID. Paths are resolved from
`promopilot.config`, never from absolute paths stored inside the files. Unreadable or schema-invalid files are
reported as `DataIssue`s instead of raising. No function in this module writes to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

from promopilot import config
from promopilot.judge import effective_judgement, model_slug, reassessment_status, review_outcome
from promopilot.schemas import (Candidate, EpisodeGeneration, EpisodeValidation, Episode, Judgement,
                                QualityDecision, Reassessment, RepairProvenance)

PRODUCTION_RUBRIC = "judge-v1"
_VERSION_SUFFIX = re.compile(r"__(judge-v[\w.-]+)$")
_KEYLIKE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|sk-[0-9A-Za-z_\-]{16,}|(?i:api[_-]?key|key|token)=[^&\s\"']+)")

VERDICT_LANES = {"eligible": "eligible", "quality_fail": "quality_fail", "hard_fail": "technical_fail",
                 "not_judged": "not_judged", "judge_invalid": "judge_invalid"}


@dataclass(frozen=True)
class DataIssue:
    path: str
    problem: str


@dataclass(frozen=True)
class JudgeRef:
    name: str
    path: Path
    provider: Optional[str]
    model: Optional[str]
    prompt_version: Optional[str]
    prompt_sha256: Optional[str]
    checker_version: Optional[str]
    validator_version: Optional[str]

    @property
    def production(self) -> bool:
        return (self.prompt_version == PRODUCTION_RUBRIC and _VERSION_SUFFIX.search(self.name) is None
                and (self.provider is None or self.name == model_slug(self.provider, self.model or "")))

    @property
    def label(self) -> str:
        return f"{self.prompt_version or 'unknown rubric'} · {self.model or self.name}"


@dataclass(frozen=True)
class RunRef:
    name: str
    path: Path
    judges: tuple[JudgeRef, ...]

    @property
    def production_judges(self) -> list[JudgeRef]:
        return [j for j in self.judges if j.production]

    @property
    def experimental_judges(self) -> list[JudgeRef]:
        return [j for j in self.judges if not j.production]


@dataclass
class Staleness:
    status: str  # current | stale | missing
    reasons: list[str] = field(default_factory=list)
    generated_at: Optional[str] = None


@dataclass
class CandidateView:
    candidate_id: str
    episode_id: str
    candidate: Optional[Candidate] = None
    decision: Optional[QualityDecision] = None
    raw: Optional[Judgement] = None
    reassessment: Optional[Reassessment] = None
    effective: Optional[Judgement] = None
    table_row: Optional[dict] = None
    ordered_row: Optional[dict] = None
    validation_record: Optional[dict] = None
    repaired_to: Optional[str] = None

    @property
    def lane(self) -> str:
        if self.decision is None:
            return "pending"
        return VERDICT_LANES.get(self.decision.verdict, "unknown")

    @property
    def judged(self) -> bool:
        """A valid judgement exists and the decision used it. Only then are scores shown."""
        return (self.decision is not None and self.decision.verdict not in ("not_judged", "judge_invalid")
                and self.effective is not None and self.effective.status == "valid"
                and bool(self.decision.dimension_scores))

    @property
    def scores(self) -> dict[str, int]:
        return dict(self.decision.dimension_scores) if self.judged else {}

    @property
    def raw_scores(self) -> dict[str, int]:
        if not self.judged or self.raw is None:
            return {}
        return {k: d.score for k, d in self.raw.dimensions.items() if d.score is not None}

    @property
    def weighted_score(self) -> Optional[float]:
        return self.decision.weighted_score if self.judged else None

    @property
    def score_role(self) -> Optional[str]:
        """eligible: the score competes in the ranking. diagnostic: a persisted score of a candidate that failed a
        hard rule or a gate, shown for diagnosis only. None: no score exists."""
        if self.weighted_score is None:
            return None
        return "eligible" if self.lane == "eligible" else "diagnostic"

    @property
    def review_flags(self) -> list[str]:
        return list(self.decision.review_flags) if self.decision else []

    @property
    def repaired_from(self) -> Optional[str]:
        if self.decision and self.decision.repaired_from:
            return self.decision.repaired_from
        return self.candidate.repair.original_candidate_id if self.candidate and self.candidate.repair else None

    @property
    def repair(self) -> Optional[RepairProvenance]:
        return self.candidate.repair if self.candidate else None

    @property
    def span(self) -> tuple[Optional[int], Optional[int]]:
        v = self.decision.validation if self.decision else None
        if v and v.start_ms is not None and v.end_ms is not None:
            return v.start_ms, v.end_ms
        if self.candidate:
            return self.candidate.start_ms, self.candidate.end_ms
        return None, None

    @property
    def ranking_status(self) -> Optional[str]:
        return self.ordered_row["status"] if self.ordered_row else None

    @property
    def shortlist_rank(self) -> Optional[int]:
        return self.ordered_row["rank"] if self.ordered_row else None

    @property
    def archetype(self) -> Optional[str]:
        if not self.judged:
            return None
        return self.effective.promo_dna.archetype.value if self.effective.promo_dna else self.decision.archetype

    @property
    def reassessment_status(self) -> Optional[str]:
        return reassessment_status(self.raw, self.reassessment) if self.raw and self.reassessment else None

    def reassessment_outcome(self) -> Optional[tuple[str, dict, list[tuple[str, str]]]]:
        return review_outcome(self.raw, self.reassessment) if self.raw and self.reassessment else None


@dataclass
class EpisodeView:
    episode_id: str
    generation_status: str  # complete | pending | missing | <manifest value>
    generation_reason: Optional[str]
    candidate_ids: list[str]
    ranking_entry: Optional[dict]

    @property
    def state(self) -> str:
        if self.generation_status != "complete" and not self.candidate_ids:
            return "generation_pending"
        if self.ranking_entry is None:
            return "not_ranked"
        if self.ranking_entry.get("best"):
            return "promo_found"
        if self.ranking_entry.get("incomplete"):
            return "evaluation_incomplete"
        return "no_strong_promo"


@dataclass
class RunData:
    run: RunRef
    judge: Optional[JudgeRef]
    output_root: Path
    manifest: dict
    pack: Optional[dict]
    judge_summary: Optional[dict]
    ranking: Optional[dict]
    generation: dict[str, dict]
    candidates: dict[str, CandidateView]
    episodes: dict[str, EpisodeView]
    staleness: Staleness
    issues: list[DataIssue]
    fingerprint: str
    loaded_at: str
    validation_settings: dict[str, dict] = field(default_factory=dict)  # episode -> settings stored by `validate`

    @property
    def rubric_is_production(self) -> bool:
        return bool(self.judge and self.judge.production)

    def by_lane(self, lane: str) -> list[CandidateView]:
        return [c for c in self.candidates.values() if c.lane == lane]

    @property
    def shortlist(self) -> list[CandidateView]:
        rows = (self.ranking or {}).get("ranking", {}).get("shortlist", [])
        return [self.candidates[r["candidate_id"]] for r in rows if r["candidate_id"] in self.candidates]

    @property
    def eligible_not_selected(self) -> list[CandidateView]:
        rows = (self.ranking or {}).get("ranking", {}).get("ordered", [])
        return [self.candidates[r["candidate_id"]] for r in rows
                if r["status"] != "selected" and r["candidate_id"] in self.candidates]


# --- low-level reading ------------------------------------------------------------------------------

class _Reader:
    def __init__(self, root: Path):
        self.root, self.issues = root, []

    def rel(self, path: Path) -> str:
        try:
            return Path(path).relative_to(self.root).as_posix()
        except ValueError:
            return Path(path).name

    def json(self, path: Path, required: bool = False) -> Optional[dict]:
        if not path.exists():
            if required:
                self.issues.append(DataIssue(self.rel(path), "file is missing"))
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.issues.append(DataIssue(self.rel(path), f"unreadable JSON: {exc}"))
            return None

    def model(self, path: Path, cls: type[BaseModel], data: Optional[dict] = None):
        raw = data if data is not None else self.json(path)
        if raw is None:
            return None
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            self.issues.append(DataIssue(self.rel(path), f"does not match {cls.__name__}: "
                                                         f"{exc.error_count()} schema error(s)"))
            return None


def redact(text: Optional[str], limit: int = 160) -> str:
    """First line of a log/error string, with anything key-like removed."""
    if not text:
        return ""
    line = _KEYLIKE.sub("[redacted]", str(text).strip().splitlines()[0])
    return line if len(line) <= limit else line[: limit - 1] + "…"


# --- discovery --------------------------------------------------------------------------------------

def output_root() -> Path:
    return config.dashboard_dir()


def output_location_warning(root: Optional[Path] = None) -> Optional[str]:
    if os.environ.get("PROMOPILOT_DASHBOARD_DIR", "").strip():
        return None  # an explicitly published dashboard folder is meant to live in the project
    root = Path(root or output_root()).resolve()
    project = config.PROJECT_ROOT.resolve()
    if root == project or project in root.parents:
        return ("Outputs resolve inside the project workspace. Set PROMOPILOT_OUTPUT_DIR so screenplay-derived "
                "data stays outside the synced folder.")
    return None


def _judge_ref(path: Path) -> JudgeRef:
    summary = {}
    try:
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    suffix = _VERSION_SUFFIX.search(path.name)
    version = summary.get("prompt_version") or (suffix.group(1) if suffix else None)
    return JudgeRef(name=path.name, path=path, provider=summary.get("provider"), model=summary.get("model"),
                    prompt_version=version, prompt_sha256=summary.get("prompt_sha256"),
                    checker_version=summary.get("checker_version"),
                    validator_version=summary.get("validator_version"))


def discover_runs(root: Optional[Path] = None) -> list[RunRef]:
    runs_dir = Path(root) / "runs" if root else config.runs_dir()
    if not runs_dir.is_dir():
        return []
    runs = []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not ((path / "manifest.json").exists() or (path / "candidates").is_dir()):
            continue
        judge_root = path / "judge"
        judges = tuple(_judge_ref(j) for j in sorted(judge_root.iterdir())
                       if j.is_dir() and ((j / "summary.json").exists() or (j / "decisions").is_dir()
                                          or _VERSION_SUFFIX.search(j.name))) \
            if judge_root.is_dir() else ()
        runs.append(RunRef(name=path.name, path=path, judges=judges))
    return runs


def production_judge(run: RunRef) -> Optional[JudgeRef]:
    """The judge-v1 output for this run. Environment variables are deliberately not consulted."""
    ranked = [j for j in run.production_judges if (run.path / "ranking" / j.name / "ranking.json").exists()]
    choices = ranked or run.production_judges
    return choices[0] if choices else None


# --- fingerprint ------------------------------------------------------------------------------------

def _listing(folder: Path, pattern: str = "*.json") -> list[Path]:
    return sorted(folder.glob(pattern)) if folder.is_dir() else []


def run_files(run: RunRef, judge: Optional[JudgeRef], root: Path) -> list[Path]:
    """Every file the run view reads. Archived copies under history/ are excluded on purpose."""
    files = [run.path / "manifest.json", run.path / "llm_calls.jsonl", root / "parsed" / "pack_summary.json"]
    files += _listing(run.path / "candidates") + _listing(run.path / "validation")
    if judge:
        jp = judge.path
        files += [jp / "summary.json", jp / "llm_calls.jsonl"]
        for sub in ("decisions", "judgements", "reassessments", "baseline/decisions", "baseline/judgements",
                    "baseline/reassessments"):
            files += _listing(jp / sub)
        files += [jp / "baseline" / "summary.json", jp / "baseline" / "llm_calls.jsonl",
                  run.path / "ranking" / judge.name / "ranking.json"]
    return [f for f in files if f.is_file()]


def fingerprint(files: list[Path], root: Path) -> str:
    """SHA-256 over every file's relative path and full content: detects edits, additions and removals,
    including same-size edits that keep the modification time."""
    h = hashlib.sha256()
    for f in sorted(files):
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            rel = f.as_posix()
        h.update(rel.encode("utf-8") + b"\0")
        try:
            h.update(hashlib.sha256(f.read_bytes()).digest())
        except OSError:
            h.update(b"<unreadable>")
    return h.hexdigest()


def run_fingerprint(run: RunRef, judge: Optional[JudgeRef], root: Optional[Path] = None) -> str:
    root = Path(root or output_root())
    return fingerprint(run_files(run, judge, root), root)


# --- loading ----------------------------------------------------------------------------------------

def _staleness(judge: Optional[JudgeRef], ranking_path: Path, ranking: Optional[dict],
               decisions: dict[str, QualityDecision]) -> Staleness:
    if judge is None:
        return Staleness("missing", ["no judge output for this run"])
    if ranking is None:
        return Staleness("missing", ["ranking has not been generated for this judge output"])
    st = Staleness("current", generated_at=ranking.get("generated_at"))
    ranked_at = ranking_path.stat().st_mtime
    # A published snapshot is restored from git, which resets modification times; the content checks below still run.
    published = bool(os.environ.get("PROMOPILOT_DASHBOARD_DIR", "").strip())
    newer = [] if published else [p.name for sub in ("decisions", "judgements", "reassessments")
                                  for p in _listing(judge.path / sub) if p.stat().st_mtime > ranked_at]
    if newer:
        st.reasons.append(f"{len(newer)} judge file(s) changed after the ranking was generated")
    table = {r["candidate_id"]: r for r in ranking.get("table", [])}
    if set(table) != set(decisions):
        st.reasons.append("ranking and judge decisions list different candidates")
    changed = [cid for cid, d in decisions.items() if cid in table and
               (table[cid].get("verdict") != d.verdict or table[cid].get("weighted_score") != d.weighted_score)]
    if changed:
        st.reasons.append(f"{len(changed)} decision(s) differ from the ranking ({', '.join(changed[:3])})")
    if st.reasons:
        st.status = "stale"
    return st


def load_run(run: RunRef, judge: Optional[JudgeRef] = None, root: Optional[Path] = None) -> RunData:
    """Load one run with one judge output (default: the production judge-v1 output). Never writes."""
    root = Path(root or output_root())
    judge = judge or production_judge(run)
    rd = _Reader(root)
    fp = run_fingerprint(run, judge, root)

    manifest = rd.json(run.path / "manifest.json", required=True) or {}
    pack = rd.json(root / "parsed" / "pack_summary.json")

    generation, candidates = {}, {}
    for path in _listing(run.path / "candidates"):
        gen = rd.model(path, EpisodeGeneration)
        if gen is None:
            continue
        generation[gen.episode_id] = {"provider": gen.provider, "model": gen.model, "status": gen.status,
                                      "prompt_version": gen.prompt_version, "prompt_sha256": gen.prompt_sha256,
                                      "response_errors": list(gen.response_errors)}
        for c in gen.candidates:
            candidates[c.candidate_id] = CandidateView(c.candidate_id, c.episode_id, candidate=c)

    validation_settings = {}
    for path in _listing(run.path / "validation"):
        if path.name == "summary.json":
            continue
        ev = rd.model(path, EpisodeValidation)
        if ev:
            validation_settings[ev.episode_id] = dict(ev.settings)
        for rec in (ev.records if ev else []):
            if rec.candidate_id in candidates:
                candidates[rec.candidate_id].validation_record = rec.model_dump(mode="json")

    summary, decisions, ranking = None, {}, None
    ranking_path = run.path / "ranking" / judge.name / "ranking.json" if judge else run.path / "ranking.json"
    if judge:
        summary = rd.json(judge.path / "summary.json")
        for path in _listing(judge.path / "decisions"):
            raw = rd.json(path)
            if raw is None:
                continue
            for i, item in enumerate(raw.get("decisions", [])):
                d = rd.model(path, QualityDecision, data=item)
                if d is not None:
                    decisions[d.candidate_id] = d
        for cid, d in decisions.items():
            view = candidates.setdefault(cid, CandidateView(cid, d.episode_id))
            view.decision = d
            if d.repaired_candidate and d.repaired_candidate.candidate_id != cid:
                view.repaired_to = d.repaired_candidate.candidate_id
                target = candidates.setdefault(view.repaired_to, CandidateView(view.repaired_to, d.episode_id))
                target.candidate = target.candidate or d.repaired_candidate
        for cid, view in candidates.items():
            jpath = judge.path / "judgements" / f"{cid}.json"
            if not jpath.exists():
                if view.decision and view.decision.judgement_file:
                    rd.issues.append(DataIssue(rd.rel(jpath), "judgement file referenced by a decision is missing"))
                continue
            view.raw = rd.model(jpath, Judgement)
            rpath = judge.path / "reassessments" / f"{cid}.json"
            view.reassessment = rd.model(rpath, Reassessment) if rpath.exists() else None
            if view.raw is not None:
                view.effective = effective_judgement(view.raw, view.reassessment)
                if view.raw.prompt_version.split(":")[0] != (judge.prompt_version or "").split(":")[0]:
                    rd.issues.append(DataIssue(rd.rel(jpath), f"judgement rubric {view.raw.prompt_version} does not "
                                                              f"match folder rubric {judge.prompt_version}"))
            if view.judged:
                eff = {k: d.score for k, d in view.effective.dimensions.items()}
                if any(eff.get(k) != v for k, v in view.decision.dimension_scores.items()):
                    rd.issues.append(DataIssue(rd.rel(jpath), "decision scores differ from the effective judgement"))
        ranking = rd.json(ranking_path)
        if ranking is not None and ranking.get("judge_dir") not in (None, judge.name):
            rd.issues.append(DataIssue(rd.rel(ranking_path), "ranking was built from a different judge folder"))
            ranking = None

    if ranking:
        table = {r["candidate_id"]: r for r in ranking.get("table", [])}
        ordered = {r["candidate_id"]: r for r in ranking.get("ranking", {}).get("ordered", [])}
        for cid, view in candidates.items():
            view.table_row, view.ordered_row = table.get(cid), ordered.get(cid)

    staleness = _staleness(judge, ranking_path, ranking, decisions)
    candidates = dict(sorted(candidates.items(), key=lambda kv: (kv[1].episode_id, kv[1].repaired_from or kv[0],
                                                                 kv[1].repaired_from is not None, kv[0])))

    manifest_eps = manifest.get("episodes") or {}
    ids = list(dict.fromkeys(list(manifest.get("requested") or []) + list(manifest_eps) +
                             sorted({c.episode_id for c in candidates.values()})))
    rank_eps = (ranking or {}).get("episodes", {})
    episodes = {}
    for eid in ids:
        m = manifest_eps.get(eid, {})
        status = m.get("status") or ("complete" if eid in generation else "missing")
        episodes[eid] = EpisodeView(eid, status, redact(m.get("reason")) or None,
                                    [c.candidate_id for c in candidates.values() if c.episode_id == eid],
                                    rank_eps.get(eid))

    return RunData(run=run, judge=judge, output_root=root, manifest=manifest, pack=pack, judge_summary=summary,
                   ranking=ranking, generation=generation, candidates=candidates, episodes=episodes,
                   staleness=staleness, issues=rd.issues, fingerprint=fp,
                   loaded_at=datetime.now(timezone.utc).isoformat(), validation_settings=validation_settings)


def episode_path(episode_id: str, root: Optional[Path] = None) -> Path:
    base = Path(root) / "parsed" if root else config.parsed_dir()
    return base / "episodes" / f"{episode_id}.json"


def load_episode(episode_id: str, root: Optional[Path] = None) -> tuple[Optional[Episode], Optional[DataIssue]]:
    path = episode_path(episode_id, root)
    rd = _Reader(Path(root or output_root()))
    ep = rd.model(path, Episode, data=rd.json(path, required=True))
    return ep, (rd.issues[0] if rd.issues else None)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).is_file() else ""
