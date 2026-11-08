"""Stage 2: LLM candidate generation, grounded against parsed blocks.

The model returns block IDs and quotes only. This module resolves every time from the parser output
and rejects references that do not exist. It does not judge quality or enforce duration/hook rules.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from . import dataset
from .llm import LLMClient, LLMRequest, ProviderError, QuotaExhausted, RequestCapReached
from .schemas import Candidate, CandidateStatus, Episode, EpisodeGeneration, Reason
from .timecode import format_timestamp

PROMPT_VERSION = "gen-v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / "generate_candidates.md"
EPISODE_END_WINDOW_MS = 2000
REPAIR_INSTRUCTION = ("Your previous reply could not be used: {error}. Reply again with JSON only, "
                      "exactly in the requested shape, using only block IDs from the list.")


class Proposal(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)
    start_block_id: str
    end_block_id: str
    hook_block_id: str
    hook_quote: str
    hook_type: str
    cliffhanger_block_id: str
    cliffhanger_quote: str
    unresolved_question: str
    new_viewer_context: str
    why_it_works: str


class ResponseFormatError(ValueError):
    pass


def load_prompt() -> tuple[str, str]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_episode(episode: Episode, previous_logline: Optional[str]) -> str:
    """Compact, time-ordered listing of every block, with scene headers. Source text is unchanged."""
    scenes = {s.scene_id: s for s in episode.scenes}
    lines = [f"EPISODE {episode.episode_number} ({episode.episode_id}), duration "
             f"{format_timestamp(episode.stats.duration_ms)}"]
    if previous_logline:
        lines.append(f"Previous episode summary (context only): {previous_logline}")
    if episode.logline:
        lines.append(f"This episode summary (context only): {episode.logline}")
    lines.append("")
    lines.append("BLOCKS (ID | start-end | type | speaker | text):")
    current_scene = None
    for b in episode.time_ordered_blocks():
        if b.scene_id != current_scene:
            current_scene = b.scene_id
            s = scenes.get(b.scene_id)
            heading = s.heading if s and s.heading else ""
            lines.append(f"[{b.scene_id} {heading}]")
        who = b.label_raw if b.block_type.value == "DIALOGUE" else ""
        lines.append(f"{b.block_id} | {format_timestamp(b.start_ms)}-{format_timestamp(b.end_ms)} | "
                     f"{b.block_type.value} | {who} | {b.text}")
    return "\n".join(lines)


def parse_model_json(text: str) -> tuple[list, list[str]]:
    """Return (candidate list, notes). Raises ResponseFormatError on anything unusable."""
    notes = []
    body = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", body, re.S)
    if fenced:
        body = fenced.group(1)
        notes.append("CODE_FENCE_STRIPPED")
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(f"not valid JSON ({exc.msg} at char {exc.pos})") from None
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ResponseFormatError('top level must be an object with a "candidates" list')
    return data["candidates"], notes


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2026", "...")
    text = re.sub(r"[\"'\u201c\u201d\u2018\u2019]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" .,!?;:-\u2014\u2013")


def quote_matches(quote: str, source: str) -> bool:
    q = _normalise(quote)
    return bool(q) and q in _normalise(source)


def ground_proposals(episode: Episode, proposals: list, *, start_index: int = 1) -> list[Candidate]:
    """Resolve each proposal against the episode's blocks. Every proposal yields a Candidate record."""
    out: list[Candidate] = []
    grounded_spans: list[tuple[str, int, int]] = []
    duration = episode.stats.duration_ms if episode.stats else 0
    final_block = max(episode.time_ordered_blocks(), key=lambda b: (b.end_ms, b.file_index), default=None)

    for offset, raw in enumerate(proposals):
        cid = f"{episode.episode_id}_C{start_index + offset:02d}"
        reasons: list[Reason] = []
        flags: list[str] = []
        if not isinstance(raw, dict):
            out.append(Candidate(candidate_id=cid, episode_id=episode.episode_id, status=CandidateStatus.REJECTED,
                                 rejection_reasons=[Reason(code="SCHEMA_INVALID", message="candidate is not an object")],
                                 proposal={"value": raw}))
            continue
        try:
            p = Proposal.model_validate(raw)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            out.append(Candidate(candidate_id=cid, episode_id=episode.episode_id, status=CandidateStatus.REJECTED,
                                 rejection_reasons=[Reason(code="SCHEMA_INVALID", message=problems)], proposal=raw,
                                 start_block_id=raw.get("start_block_id") if isinstance(raw.get("start_block_id"), str) else None,
                                 end_block_id=raw.get("end_block_id") if isinstance(raw.get("end_block_id"), str) else None))
            continue
        if p.model_extra:
            flags.append("MODEL_EXTRA_FIELDS_IGNORED:" + ",".join(sorted(p.model_extra)))

        blocks = {}
        for role in ("start_block_id", "end_block_id", "hook_block_id", "cliffhanger_block_id"):
            bid = getattr(p, role)
            if not episode.has_block(bid):
                other = re.match(r"^(EP\d{3})_B\d+$", bid)
                where = f" (belongs to {other.group(1)})" if other and other.group(1) != episode.episode_id else ""
                reasons.append(Reason(code="INVALID_BLOCK_ID", message=f"{role}={bid!r} is not a block of "
                                      f"{episode.episode_id}{where}"))
            elif not episode.get_block(bid).has_valid_time:
                reasons.append(Reason(code="BLOCK_WITHOUT_TIME", message=f"{role}={bid} has no valid time"))
            else:
                blocks[role] = episode.get_block(bid)

        c = Candidate(candidate_id=cid, episode_id=episode.episode_id, status=CandidateStatus.REJECTED,
                      proposal=raw, start_block_id=p.start_block_id, end_block_id=p.end_block_id,
                      hook_block_id=p.hook_block_id, cliffhanger_block_id=p.cliffhanger_block_id,
                      hook_type=p.hook_type, unresolved_question=p.unresolved_question,
                      new_viewer_context=p.new_viewer_context, why_it_works=p.why_it_works)

        start, end = blocks.get("start_block_id"), blocks.get("end_block_id")
        if start and end:
            c.start_ms, c.end_ms = start.start_ms, end.end_ms
            if c.end_ms <= c.start_ms:
                reasons.append(Reason(code="BOUNDARY_ORDER",
                                      message=f"end {end.block_id} ({end.end_raw}) is not after start "
                                              f"{start.block_id} ({start.start_raw})"))
            else:
                c.start_ts, c.end_ts = format_timestamp(c.start_ms), format_timestamp(c.end_ms)
                c.duration_ms = c.end_ms - c.start_ms
                segment = episode.blocks_intersecting(c.start_ms, c.end_ms)
                c.segment_block_ids = [b.block_id for b in segment]
                c.scene_ids = sorted({b.scene_id for b in segment if b.scene_id})
                for role, code in (("hook_block_id", "HOOK_OUTSIDE_CLIP"),
                                   ("cliffhanger_block_id", "CLIFFHANGER_OUTSIDE_CLIP")):
                    blk = blocks.get(role)
                    if blk and not blk.intersects(c.start_ms, c.end_ms):
                        reasons.append(Reason(code=code, message=f"{blk.block_id} ({blk.start_raw}-{blk.end_raw}) "
                                              f"is outside {c.start_ts}-{c.end_ts}"))
                hook, cliff = blocks.get("hook_block_id"), blocks.get("cliffhanger_block_id")
                if hook:
                    c.hook_offset_ms = hook.start_ms - c.start_ms
                if cliff:
                    c.cliffhanger_end_gap_ms = c.end_ms - cliff.end_ms
                if duration - c.end_ms <= EPISODE_END_WINDOW_MS or (final_block and end.block_id == final_block.block_id):
                    flags.append("ENDS_AT_EPISODE_END")

        for role, quote, attr in (("hook_block_id", p.hook_quote, "hook_source_text"),
                                  ("cliffhanger_block_id", p.cliffhanger_quote, "cliffhanger_source_text")):
            blk = blocks.get(role)
            if blk is None:
                continue
            setattr(c, attr, blk.text)
            if not quote_matches(quote, blk.text):
                reasons.append(Reason(code="QUOTE_MISMATCH",
                                      message=f"{role.replace('_block_id', '')} quote not found verbatim in {blk.block_id}"))

        if not reasons and c.duration_ms:
            for other_id, s, e in grounded_spans:
                if (s, e) == (c.start_ms, c.end_ms):
                    reasons.append(Reason(code="DUPLICATE_PROPOSAL", message=f"same span as {other_id}"))
                    break
                inter = max(0, min(e, c.end_ms) - max(s, c.start_ms))
                union = max(e, c.end_ms) - min(s, c.start_ms)
                if union and inter / union > 0.5:
                    flags.append(f"OVERLAPS_MORE_THAN_HALF:{other_id}")

        c.rejection_reasons = reasons
        c.flags = flags
        if not reasons:
            c.status = CandidateStatus.GROUNDED
            grounded_spans.append((cid, c.start_ms, c.end_ms))
        out.append(c)
    return out


def generate_for_episode(episode: Episode, previous_logline: Optional[str], client: LLMClient, *,
                         n_candidates: int, temperature: float, max_tokens: int) -> EpisodeGeneration:
    prompt, prompt_sha = load_prompt()
    system = prompt.replace("{n_candidates}", str(n_candidates))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": render_episode(episode, previous_logline)}]
    version = f"{PROMPT_VERSION}:{prompt_sha[:12]}"
    gen = EpisodeGeneration(episode_id=episode.episode_id, status="in_progress", provider=client.provider.name,
                            model=client.provider.model, endpoint=getattr(client.provider, "endpoint", None),
                            prompt_version=PROMPT_VERSION, prompt_sha256=prompt_sha)

    request = LLMRequest(messages=messages, temperature=temperature, max_tokens=max_tokens)
    result = client.complete(request, prompt_version=version, stage="generate", item_id=episode.episode_id)
    gen.calls.append(result.summary("generate"))
    try:
        proposals, notes = _parse_or_raise(result.response)
    except ResponseFormatError as first_error:
        gen.response_errors.append(f"generate: {first_error}")
        repair = LLMRequest(messages=messages + [
            {"role": "assistant", "content": result.response.text},
            {"role": "user", "content": REPAIR_INSTRUCTION.format(error=first_error)}],
            temperature=temperature, max_tokens=max_tokens)
        result = client.complete(repair, prompt_version=version, stage="generate_repair", item_id=episode.episode_id)
        gen.calls.append(result.summary("generate_repair"))
        try:
            proposals, notes = _parse_or_raise(result.response)
        except ResponseFormatError as second_error:
            gen.response_errors.append(f"generate_repair: {second_error}")
            gen.status = "invalid_response"
            return gen
    gen.response_errors.extend(f"note: {n}" for n in notes)
    gen.candidates = ground_proposals(episode, proposals)
    gen.status = "complete"
    return gen


def _parse_or_raise(response) -> tuple[list, list[str]]:
    if response.finish_reason == "length":
        raise ResponseFormatError("reply was cut off at the token limit")
    return parse_model_json(response.text)


def load_parsed_episodes(parsed_dir: Path) -> dict[int, Episode]:
    """Checked load of one parsed dataset (see `dataset.load`); raises DatasetError on mixed or stale data."""
    return dataset.load(parsed_dir)[0]


def _episode_key(eid: str) -> tuple[int, str]:
    digits = eid[2:]
    return (int(digits), eid) if digits.isdigit() else (10 ** 9, eid)


def bind_dataset(manifest: dict, info: dict, run_dir: Path) -> None:
    """Record the dataset a run is generated from; refuse to continue a run on a different dataset."""
    bound = manifest.get("dataset")
    if bound and bound.get("fingerprint") != info["fingerprint"]:
        raise dataset.DatasetError(
            f"run {Path(run_dir).name} was generated from parsed dataset {bound.get('fingerprint', '')[:12]} "
            f"({bound.get('parsed_dir')}); the current dataset is {info['fingerprint'][:12]} "
            f"({info['parsed_dir']}). Use a new --run name for a different dataset.")
    manifest["dataset"] = {"parsed_dir": info["parsed_dir"], "fingerprint": info["fingerprint"],
                           "episode_count": info["episode_count"]}


def _write_json(path: Path, model: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    text = model.model_dump_json(indent=2) if isinstance(model, BaseModel) else json.dumps(model, indent=2)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_generation(episode_numbers: list[int], *, parsed_dir: Path, run_dir: Path, client: LLMClient,
                   n_candidates: int, temperature: float, max_tokens: int, refresh: bool = False) -> dict:
    """Generate candidates per episode, resumably. Completed episodes with the same model, endpoint and prompt are
    skipped unless refresh=True; cached responses are reused either way unless the client bypasses cache.
    `requested` in the manifest is the union over every invocation of this run; `last_requested` is this one."""
    episodes, info = dataset.load(parsed_dir)
    _, prompt_sha = load_prompt()
    endpoint = getattr(client.provider, "endpoint", None)
    manifest_path = Path(run_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"episodes": {}}
    manifest.setdefault("episodes", {})
    bind_dataset(manifest, info, run_dir)
    now = list(dict.fromkeys(f"EP{n:03d}" for n in episode_numbers))
    manifest.update({"provider": client.provider.name, "model": client.provider.model, "endpoint": endpoint,
                     "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha,
                     "requested": sorted(set(manifest.get("requested") or []) | set(now), key=_episode_key),
                     "last_requested": now})
    stop_reason = None

    for n in episode_numbers:
        eid = f"EP{n:03d}"
        if stop_reason:
            manifest["episodes"][eid] = {"status": "pending", "reason": stop_reason}
            continue
        if n not in episodes:
            manifest["episodes"][eid] = {"status": "missing_parsed_episode"}
            continue
        out_path = Path(run_dir) / "candidates" / f"{eid}.json"
        if out_path.exists() and not refresh:
            prior = EpisodeGeneration.model_validate_json(out_path.read_text(encoding="utf-8"))
            if (prior.status == "complete" and prior.model == client.provider.model
                    and prior.provider == client.provider.name and prior.prompt_sha256 == prompt_sha
                    and prior.endpoint == endpoint):
                manifest["episodes"][eid] = {"status": "complete", "resumed": True,
                                             "candidates": len(prior.candidates)}
                continue
        prev = episodes.get(n - 1)
        try:
            gen = generate_for_episode(episodes[n], prev.logline if prev else None, client,
                                       n_candidates=n_candidates, temperature=temperature, max_tokens=max_tokens)
        except QuotaExhausted as exc:
            stop_reason = f"quota exhausted: {exc}"
            manifest["episodes"][eid] = {"status": "pending", "reason": stop_reason}
            continue
        except RequestCapReached as exc:
            stop_reason = str(exc)
            manifest["episodes"][eid] = {"status": "pending", "reason": stop_reason}
            continue
        except ProviderError as exc:
            gen = EpisodeGeneration(episode_id=eid, status="provider_error", provider=client.provider.name,
                                    model=client.provider.model, endpoint=endpoint, prompt_version=PROMPT_VERSION,
                                    prompt_sha256=prompt_sha, response_errors=[str(exc)])
        _write_json(out_path, gen)
        manifest["episodes"][eid] = {"status": gen.status, "resumed": False, "candidates": len(gen.candidates)}

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stopped_early"] = stop_reason
    manifest["requests_made_this_invocation"] = client.requests_made
    manifest["cache_hits_this_invocation"] = client.cache_hits
    _write_json(manifest_path, manifest)
    return manifest
