"""Deterministic screenplay parser and anomaly detector.

Source anomalies are recorded, never corrected: block times, order and text stay as written.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from . import config
from .schemas import (
    Anomaly, Block, BlockType, Episode, EpisodeSummary, PackSummary, Scene, Severity, TimelineStats,
)
from .timecode import TimestampError, parse_timestamp

RE_SCRIPT_START = re.compile(r"^-+\s*SCRIPT START\s*-+$", re.I)
RE_SCRIPT_END = re.compile(r"^-+\s*SCRIPT END\s*-+$", re.I)
RE_HEADER = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
RE_HEADER_NUMBER = re.compile(r"^EPISODE NUMBER\s*:\s*(?:Episode\s+)?(\d+)", re.I | re.M)
RE_FILE_NUMBER = re.compile(r"episode[-_ ]?(\d+)", re.I)
RE_SCENE = re.compile(r"^(SCENE)\s+(\d+)\s*$", re.I)
RE_SCENE_FIELD = re.compile(r"^(Scene heading|Scene start|Scene end)\s*:\s*(.*)$", re.I)
RE_SECTION = re.compile(r"^(Actions and dialogue)\s*:?\s*$", re.I)
RE_LABEL = re.compile(r"^\*\*(.+?):\*\*$")
RE_TIME = re.compile(
    r"^Start:\s*(\()?\s*([^()|]+?)\s*(\))?\s*\|\s*End:\s*(\()?\s*([^()|]+?)\s*(\))?\s*$", re.I
)
RE_LABEL_PARTS = re.compile(r"^(.*?)\s*((?:\([^)]*\)\s*)*)$")

CANONICAL_SCENE_KEYS = {"scene heading": "Scene heading", "scene start": "Scene start", "scene end": "Scene end"}
TYPE_LABELS = {"CAMERA": BlockType.CAMERA, "ACTION": BlockType.ACTION,
               "SOUND FX": BlockType.SOUND_FX, "SFX": BlockType.SOUND_FX}
OFFSCREEN_TAGS = {"O.S.", "V.O.", "O.C.", "OS", "VO", "OC"}


class ParseError(Exception):
    pass


def parse_label(label: str) -> tuple[BlockType, Optional[str], list[str], Optional[str]]:
    """Split a block label into (type, speaker, delivery notes, off-screen tag)."""
    match = RE_LABEL_PARTS.match(label.strip())
    name = match.group(1).strip()
    groups = re.findall(r"\(([^)]*)\)", match.group(2))
    tokens = [t.strip() for g in groups for t in g.split(",") if t.strip()]
    offscreen = [t for t in tokens if t.upper() in OFFSCREEN_TAGS]
    delivery = [t for t in tokens if t.upper() not in OFFSCREEN_TAGS]
    block_type = TYPE_LABELS.get(name.upper(), BlockType.DIALOGUE)
    speaker = name if block_type is BlockType.DIALOGUE else None
    return block_type, speaker, delivery, (offscreen[0] if offscreen else None)


def _episode_number(text: str, source_name: str) -> tuple[int, list[Anomaly]]:
    anomalies = []
    header = RE_HEADER_NUMBER.search(text)
    from_file = RE_FILE_NUMBER.search(source_name)
    header_n = int(header.group(1)) if header else None
    file_n = int(from_file.group(1)) if from_file else None
    if header_n is None and file_n is None:
        raise ParseError(f"{source_name}: cannot determine episode number from header or file name")
    if header_n is not None and file_n is not None and header_n != file_n:
        anomalies.append(Anomaly(code="EPISODE_NUMBER_MISMATCH", severity=Severity.WARNING,
                                 message=f"header says {header_n}, file name says {file_n}; using header",
                                 details={"header": header_n, "file_name": file_n}))
    if header_n is None:
        anomalies.append(Anomaly(code="MISSING_HEADER_FIELD", severity=Severity.WARNING,
                                 message="EPISODE NUMBER missing; using file name",
                                 details={"field": "EPISODE NUMBER"}))
    return (header_n if header_n is not None else file_n), anomalies


def _parse_time(raw: str, anomalies: list[Anomaly], episode_id: str, line_no: int,
                block_id: Optional[str] = None, scene_id: Optional[str] = None) -> Optional[int]:
    try:
        return parse_timestamp(raw)
    except TimestampError as exc:
        anomalies.append(Anomaly(code="INVALID_TIMESTAMP", severity=Severity.ERROR, message=str(exc),
                                 episode_id=episode_id, block_id=block_id, scene_id=scene_id,
                                 line_number=line_no, details={"raw": raw}))
        return None


def parse_episode_text(text: str, source_name: str, source_sha256: Optional[str] = None) -> Episode:
    """Parse one episode's screenplay text. Raises ParseError only if the episode cannot be identified."""
    if source_sha256 is None:
        source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    number, anomalies = _episode_number(text, source_name)
    episode_id = f"EP{number:03d}"
    for a in anomalies:
        a.episode_id = episode_id

    def flag(code, severity, message, line_no=None, block_id=None, scene_id=None, **details):
        anomalies.append(Anomaly(code=code, severity=severity, message=message, episode_id=episode_id,
                                 block_id=block_id, scene_id=scene_id, line_number=line_no, details=details))

    header: dict[str, str] = {}
    extra_header: dict[str, str] = {}
    scenes: list[Scene] = []
    blocks: list[Block] = []
    pending: Optional[dict] = None
    seen_start = seen_end = False
    case_variant_lines: list[int] = []
    bare_time_lines: list[int] = []

    def finish_pending():
        nonlocal pending
        if pending is None:
            return
        idx = len(blocks) + 1
        block_id = f"{episode_id}_B{idx:03d}"
        block_type, speaker, delivery, offscreen = parse_label(pending["label"])
        start_ms = end_ms = None
        if pending["time"] is None:
            flag("BLOCK_MISSING_TIME", Severity.ERROR, "block has no Start/End line",
                 pending["line"], block_id=block_id, label=pending["label"])
        else:
            start_raw, end_raw, time_line = pending["time"]
            start_ms = _parse_time(start_raw, anomalies, episode_id, time_line, block_id=block_id)
            end_ms = _parse_time(end_raw, anomalies, episode_id, time_line, block_id=block_id)
        if not pending["text"]:
            flag("BLOCK_NO_TEXT", Severity.WARNING, "block has no text line", pending["line"], block_id=block_id)
        if len(pending["text"]) > 1:
            flag("MULTILINE_BLOCK_TEXT", Severity.INFO, f"block text spans {len(pending['text'])} lines",
                 pending["line"], block_id=block_id)
        scene = scenes[-1] if scenes else None
        if scene is None:
            flag("BLOCK_BEFORE_SCENE", Severity.WARNING, "block appears before any SCENE line",
                 pending["line"], block_id=block_id)
        else:
            scene.block_ids.append(block_id)
        blocks.append(Block(
            block_id=block_id, episode_id=episode_id, file_index=idx - 1,
            scene_id=scene.scene_id if scene else None, block_type=block_type, label_raw=pending["label"],
            speaker=speaker, delivery=delivery, offscreen=offscreen,
            start_raw=pending["time"][0] if pending["time"] else None,
            end_raw=pending["time"][1] if pending["time"] else None,
            start_ms=start_ms, end_ms=end_ms, text="\n".join(pending["text"]), line_number=pending["line"],
        ))
        pending = None

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            if pending is not None and pending["text"]:
                pending["closed"] = True
            continue

        if RE_SCRIPT_START.match(line):
            finish_pending()
            seen_start = True
            continue
        if RE_SCRIPT_END.match(line):
            finish_pending()
            seen_end = True
            continue
        if not seen_start:
            m = RE_HEADER.match(line)
            if m:
                header[m.group(1).upper()] = m.group(2).strip()
            elif RE_SCENE.match(line) or RE_LABEL.match(line):
                flag("MISSING_SCRIPT_START", Severity.WARNING, "script content before SCRIPT START marker", line_no)
                seen_start = True
            else:
                key, _, value = line.partition(":")
                extra_header[key.strip()] = value.strip()
                flag("UNKNOWN_HEADER_LINE", Severity.INFO, "unrecognised header line kept in extra_header", line_no)
                continue
            if not seen_start:
                continue

        m = RE_SCENE.match(line)
        if m:
            finish_pending()
            if m.group(1) != "SCENE":
                case_variant_lines.append(line_no)
            scenes.append(Scene(scene_id=f"{episode_id}_S{len(scenes) + 1:02d}", episode_id=episode_id,
                                number=int(m.group(2)), line_number=line_no))
            continue
        m = RE_SCENE_FIELD.match(line)
        if m:
            finish_pending()
            key = m.group(1)
            if key != CANONICAL_SCENE_KEYS[key.lower()]:
                case_variant_lines.append(line_no)
            if not scenes:
                flag("UNPARSED_LINE", Severity.WARNING, "scene field outside any scene", line_no, line=line)
                continue
            scene = scenes[-1]
            value = m.group(2).strip()
            if key.lower() == "scene heading":
                scene.heading = value
            elif key.lower() == "scene start":
                scene.start_raw = value
                scene.start_ms = _parse_time(value, anomalies, episode_id, line_no, scene_id=scene.scene_id)
            else:
                scene.end_raw = value
                scene.end_ms = _parse_time(value, anomalies, episode_id, line_no, scene_id=scene.scene_id)
            continue
        m = RE_SECTION.match(line)
        if m:
            finish_pending()
            if m.group(1) != "Actions and dialogue":
                case_variant_lines.append(line_no)
            continue
        m = RE_LABEL.match(line)
        if m:
            finish_pending()
            pending = {"label": m.group(1).strip(), "line": line_no, "time": None, "text": [], "closed": False}
            continue
        if pending is not None and pending["time"] is None and not pending["text"]:
            m = RE_TIME.match(line)
            if m:
                if not (m.group(1) and m.group(3) and m.group(4) and m.group(6)):
                    bare_time_lines.append(line_no)
                pending["time"] = (m.group(2).strip(), m.group(5).strip(), line_no)
                continue
        if pending is not None and not pending["closed"]:
            pending["text"].append(line)
            continue
        flag("UNPARSED_LINE", Severity.WARNING, "line not recognised and not inside a block", line_no, line=line)

    finish_pending()

    if not seen_start:
        flag("MISSING_SCRIPT_START", Severity.WARNING, "no SCRIPT START marker")
    if not seen_end:
        flag("MISSING_SCRIPT_END", Severity.WARNING, "no SCRIPT END marker")
    for field in ("EPISODE LOGLINE", "CONTINUITY NOTES"):
        if field not in header:
            flag("MISSING_HEADER_FIELD", Severity.WARNING, f"{field} missing", field=field)
    if case_variant_lines:
        flag("HEADER_CASE_VARIANT", Severity.INFO, "structural keys use non-standard capitalisation",
             case_variant_lines[0], lines=case_variant_lines)
    if bare_time_lines:
        flag("TIME_FORMAT_VARIANT", Severity.INFO, "block times without parentheses",
             bare_time_lines[0], lines=bare_time_lines)

    episode = Episode(
        episode_id=episode_id, episode_number=number, source_file=source_name, source_sha256=source_sha256,
        parser_version=config.PARSER_VERSION, logline=header.get("EPISODE LOGLINE"),
        continuity_notes=header.get("CONTINUITY NOTES"), extra_header=extra_header,
        scenes=scenes, blocks=blocks, anomalies=anomalies,
    )
    stats, timeline_anomalies = detect_timeline_anomalies(episode)
    episode.stats = stats
    episode.anomalies.extend(timeline_anomalies)
    return episode


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + config.TIME_TOLERANCE_MS:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _overlaps_within_track(episode: Episode, track: BlockType, code: str) -> list[Anomaly]:
    tol = config.TIME_TOLERANCE_MS
    found = []
    latest: Optional[Block] = None
    for b in (x for x in episode.time_ordered_blocks() if x.block_type is track):
        if latest is not None and b.start_ms < latest.end_ms - tol:
            found.append(Anomaly(
                code=code, severity=Severity.INFO, episode_id=episode.episode_id, block_id=b.block_id,
                line_number=b.line_number,
                message=f"{b.block_id} starts before {latest.block_id} ends",
                details={"overlaps": latest.block_id, "overlap_ms": latest.end_ms - b.start_ms}))
        if latest is None or b.end_ms > latest.end_ms:
            latest = b
    return found


def detect_timeline_anomalies(episode: Episode) -> tuple[TimelineStats, list[Anomaly]]:
    """Episode-level timing checks. Uses only blocks with valid times; nothing is modified."""
    tol = config.TIME_TOLERANCE_MS
    eid = episode.episode_id
    found: list[Anomaly] = []

    def flag(code, severity, message, block=None, scene=None, **details):
        found.append(Anomaly(code=code, severity=severity, message=message, episode_id=eid,
                             block_id=block.block_id if block else None,
                             scene_id=scene.scene_id if scene else None,
                             line_number=(block or scene).line_number if (block or scene) else None,
                             details=details))

    if not episode.blocks:
        flag("EMPTY_EPISODE", Severity.ERROR, "episode has no blocks")

    for b in episode.blocks:
        if b.start_ms is not None and b.end_ms is not None:
            if b.end_ms < b.start_ms:
                flag("END_BEFORE_START", Severity.ERROR, f"{b.end_raw} < {b.start_raw}", block=b)
            elif b.end_ms == b.start_ms:
                flag("ZERO_DURATION", Severity.WARNING, "block has zero duration", block=b)

    for s in episode.scenes:
        if s.start_ms is None or s.end_ms is None:
            flag("SCENE_MISSING_TIME", Severity.WARNING, "scene start or end time missing", scene=s)
        elif s.end_ms < s.start_ms:
            flag("SCENE_TIME_INVERTED", Severity.ERROR, f"scene ends ({s.end_raw}) before it starts", scene=s)
    for a, b in zip(episode.scenes, episode.scenes[1:]):
        if a.end_ms is not None and b.start_ms is not None:
            if b.start_ms > a.end_ms + tol:
                flag("SCENE_GAP", Severity.WARNING, f"{b.scene_id} starts after {a.scene_id} ends", scene=b,
                     gap_ms=b.start_ms - a.end_ms)
            elif b.start_ms < a.end_ms - tol:
                flag("SCENE_OVERLAP", Severity.WARNING, f"{b.scene_id} starts before {a.scene_id} ends",
                     scene=b, overlap_ms=a.end_ms - b.start_ms)

    scenes_by_id = {s.scene_id: s for s in episode.scenes}
    for b in episode.blocks:
        s = scenes_by_id.get(b.scene_id)
        if not (s and b.has_valid_time and s.start_ms is not None and s.end_ms is not None):
            continue
        if b.start_ms < s.start_ms - tol:
            flag("BLOCK_STARTS_BEFORE_SCENE", Severity.WARNING,
                 f"{b.block_id} starts {b.start_raw}, scene starts {s.start_raw}", block=b,
                 scene_start_ms=s.start_ms, block_start_ms=b.start_ms)
        if b.end_ms > s.end_ms + tol:
            flag("BLOCK_OVERRUNS_SCENE", Severity.WARNING,
                 f"{b.block_id} ends {b.end_raw}, scene ends {s.end_raw}", block=b,
                 scene_end_ms=s.end_ms, block_end_ms=b.end_ms)

    timed_file_order = [b for b in episode.blocks if b.has_valid_time]
    for prev, cur in zip(timed_file_order, timed_file_order[1:]):
        if cur.start_ms < prev.start_ms - tol:
            flag("FILE_ORDER_NOT_TIME_ORDER", Severity.INFO,
                 f"{cur.block_id} ({cur.start_raw}) is listed after {prev.block_id} ({prev.start_raw})",
                 block=cur, previous=prev.block_id)

    found += _overlaps_within_track(episode, BlockType.DIALOGUE, "OVERLAPPING_DIALOGUE")
    found += _overlaps_within_track(episode, BlockType.CAMERA, "OVERLAPPING_CAMERA")

    timed = episode.time_ordered_blocks()
    duration = max((b.end_ms for b in timed), default=0)

    camera = _union([(b.start_ms, b.end_ms) for b in timed if b.block_type is BlockType.CAMERA])
    camera_gaps = [(a[1], b[0]) for a, b in zip(camera, camera[1:])]
    if camera and camera[0][0] > tol:
        camera_gaps.insert(0, (0, camera[0][0]))
    if camera and duration - camera[-1][1] > tol:
        camera_gaps.append((camera[-1][1], duration))
    for gs, ge in camera_gaps:
        if ge - gs >= config.LARGE_CAMERA_GAP_MS:
            flag("LARGE_CAMERA_GAP", Severity.INFO, f"no CAMERA block for {ge - gs} ms (other tracks may continue)",
                 start_ms=gs, end_ms=ge)

    covered = _union([(b.start_ms, b.end_ms) for b in timed])
    uncovered = []
    cursor = 0
    for s, e in covered:
        if s > cursor + tol:
            uncovered.append((cursor, s))
        cursor = max(cursor, e)
    for us, ue in uncovered:
        if ue - us >= config.UNCOVERED_TIME_FLAG_MS:
            flag("UNCOVERED_TIME", Severity.INFO, f"no block on any track for {ue - us} ms",
                 start_ms=us, end_ms=ue)

    stats = TimelineStats(
        duration_ms=duration,
        camera_coverage_ratio=round(sum(e - s for s, e in camera) / duration, 4) if duration else 0.0,
        max_camera_gap_ms=max((ge - gs for gs, ge in camera_gaps), default=0),
        uncovered_intervals_ms=uncovered,
    )
    return stats, found


def discover_episode_files(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ParseError(f"input folder not found: {input_dir}")
    files = sorted(p for p in input_dir.rglob("*.txt") if p.is_file())
    if not files:
        raise ParseError(f"no .txt files under {input_dir}")
    return files


def parse_episode_file(path: Path) -> Episode:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path.name}: not valid UTF-8 ({exc})") from exc
    return parse_episode_text(text, Path(path).name, hashlib.sha256(raw).hexdigest())


def parse_pack(input_dir: Path) -> tuple[list[Episode], PackSummary]:
    """Parse every episode file under input_dir. Unreadable files are reported, not fatal."""
    input_dir = Path(input_dir)
    episodes: list[Episode] = []
    pack_anomalies: list[Anomaly] = []
    for path in discover_episode_files(input_dir):
        try:
            episodes.append(parse_episode_file(path))
        except ParseError as exc:
            pack_anomalies.append(Anomaly(code="UNREADABLE_FILE", severity=Severity.ERROR, message=str(exc),
                                          details={"file": path.name}))
    episodes.sort(key=lambda e: (e.episode_number, e.source_file))

    counts = Counter(e.episode_number for e in episodes)
    for number, n in sorted(counts.items()):
        if n > 1:
            pack_anomalies.append(Anomaly(
                code="DUPLICATE_EPISODE_NUMBER", severity=Severity.ERROR,
                message=f"episode {number} appears in {n} files",
                details={"files": [e.source_file for e in episodes if e.episode_number == number]}))
    if counts:
        missing = sorted(set(range(min(counts), max(counts) + 1)) - set(counts))
        if missing:
            pack_anomalies.append(Anomaly(code="EPISODE_NUMBER_GAP", severity=Severity.WARNING,
                                          message=f"missing episode numbers: {missing}",
                                          details={"missing": missing}))

    speakers: dict[str, set[int]] = defaultdict(set)
    for e in episodes:
        for b in e.blocks:
            if b.speaker:
                speakers[b.speaker].add(e.episode_number)

    all_anomalies = [a for e in episodes for a in e.anomalies] + pack_anomalies
    summary = PackSummary(
        pack_name=input_dir.name,
        parser_version=config.PARSER_VERSION,
        episode_count=len(episodes),
        scene_count=sum(len(e.scenes) for e in episodes),
        block_count=sum(len(e.blocks) for e in episodes),
        block_type_counts=dict(sorted(Counter(b.block_type.value for e in episodes for b in e.blocks).items())),
        anomaly_counts=dict(sorted(Counter(a.code for a in all_anomalies).items())),
        pack_anomalies=pack_anomalies,
        speakers={name: sorted(eps) for name, eps in sorted(speakers.items())},
        episodes=[EpisodeSummary(
            episode_id=e.episode_id, episode_number=e.episode_number, source_file=e.source_file,
            duration_ms=e.stats.duration_ms if e.stats else 0, scene_count=len(e.scenes),
            block_count=len(e.blocks), anomaly_count=len(e.anomalies)) for e in episodes],
    )
    return episodes, summary
