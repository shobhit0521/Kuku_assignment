"""Evidence chips and the screenplay clip panel. Chips link to anchored lines in the panel."""
from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Iterable, Optional

from promopilot.schemas import Block, Episode, Judgement
from promopilot.timecode import format_timestamp
from promopilot_ui.status import SHORT_LABELS, TRACK_LABELS


def anchor(block_id: str) -> str:
    return f"blk-{block_id}"


def short_id(block_id: str) -> str:
    return block_id.split("_")[-1]


class Clip:
    """Blocks of the episode that intersect the clip, as stored by the parser."""

    def __init__(self, episode: Optional[Episode], start_ms: Optional[int], end_ms: Optional[int]):
        self.episode, self.start_ms, self.end_ms = episode, start_ms, end_ms
        ok = episode is not None and start_ms is not None and end_ms is not None
        self.blocks: list[Block] = episode.blocks_intersecting(start_ms, end_ms) if ok else []
        self.ids = {b.block_id for b in self.blocks}

    def block(self, block_id: str) -> Optional[Block]:
        if self.episode is not None and self.episode.has_block(block_id):
            return self.episode.get_block(block_id)
        return None

    def rel(self, ms: Optional[int]) -> str:
        if ms is None or self.start_ms is None:
            return ""
        return f"+{(ms - self.start_ms) / 1000:.1f} s"


def chip(block_id: str, clip: Clip, label: str = "") -> str:
    b = clip.block(block_id)
    if b is None:
        return (f'<span class="pp-chip pp-chip--missing" title="Not found in the parsed screenplay">'
                f'{escape(block_id)}</span>')
    time = b.start_raw or ""
    title = f"{TRACK_LABELS.get(b.block_type.value, b.block_type.value)} · {time}"
    text = f"{escape(label)} " if label else ""
    if block_id not in clip.ids:
        return (f'<span class="pp-chip pp-chip--outside" title="{escape(title)} · outside this clip">'
                f'{text}{short_id(block_id)} <em>{escape(time)}</em></span>')
    return (f'<a class="pp-chip" href="#{anchor(block_id)}" title="{escape(title)}">'
            f'{text}{short_id(block_id)} <em>{escape(time)}</em></a>')


def chips(ids: Iterable[str], clip: Clip) -> str:
    ids = [i for i in dict.fromkeys(ids) if i]
    if not ids:
        return '<span class="pp-none">none cited</span>'
    return "".join(chip(i, clip) for i in ids)


def citations(j: Optional[Judgement]) -> dict[str, list[str]]:
    """block ID -> labels of the judgement parts that cite it."""
    out: dict[str, list[str]] = defaultdict(list)
    if j is None or j.status != "valid":
        return out
    for name, d in j.dimensions.items():
        for b in list(d.evidence) + list(d.escalation_beats):
            if SHORT_LABELS.get(name, name) not in out[b]:
                out[b].append(SHORT_LABELS.get(name, name))
        for c in d.confusions:
            for b in c.get("block_ids", []):
                if "Confusion" not in out[b]:
                    out[b].append("Confusion")
    if j.promo_dna:
        for b in j.promo_dna.supporting_block_ids:
            if "DNA" not in out[b]:
                out[b].append("DNA")
    for r in j.soft_rejections:
        for b in r.block_ids:
            if "Rejection" not in out[b]:
                out[b].append("Rejection")
    return out


def script_panel(clip: Clip, *, hook_id: Optional[str], beat_id: Optional[str], cited: dict[str, list[str]],
                 proposed: bool = False) -> str:
    """The clip's screenplay blocks in time order with track labels; hook, beat and cited lines highlighted.
    `proposed` marks hook/beat as the generator's proposal rather than the judge's choice."""
    if clip.episode is None:
        return '<div class="pp-script"><div class="pp-empty">Parsed screenplay not available.</div></div>'
    if not clip.blocks:
        return '<div class="pp-script"><div class="pp-empty">No screenplay blocks in this clip.</div></div>'
    scenes = {s.scene_id: s for s in clip.episode.scenes}
    hook_tag = "Generator hook (proposed)" if proposed else "Hook"
    beat_tag = "Generator cliffhanger (proposed)" if proposed else "Suspense beat"
    rows, scene = [], object()
    for b in clip.blocks:
        if b.scene_id != scene:
            scene = b.scene_id
            sc = scenes.get(b.scene_id)
            if sc and sc.heading:
                rows.append(f'<div class="pp-scene">{escape(sc.heading)}</div>')
        cls = ["pp-line", f"pp-line--{b.block_type.value.lower()}"]
        tags = []
        if b.block_id == hook_id:
            cls.append("is-hook")
            tags.append(f'<span class="pp-tag pp-tag--hook">{hook_tag}</span>')
        if b.block_id == beat_id:
            cls.append("is-beat")
            tags.append(f'<span class="pp-tag pp-tag--beat">{beat_tag}</span>')
        if cited.get(b.block_id):
            cls.append("is-cited")
            tags.append(f'<span class="pp-tag">Cited: {escape(", ".join(cited[b.block_id]))}</span>')
        if b.start_ms is not None and clip.start_ms is not None and b.start_ms < clip.start_ms:
            tags.append('<span class="pp-tag pp-tag--edge">began before the clip</span>')
        if b.end_ms is not None and clip.end_ms is not None and b.end_ms > clip.end_ms:
            tags.append('<span class="pp-tag pp-tag--edge">continues after the cut</span>')
        who = ""
        if b.block_type.value == "DIALOGUE":
            extra = " ".join(f"({escape(d)})" for d in b.delivery)
            off = f" ({escape(b.offscreen)})" if b.offscreen else ""
            who = (f'<div class="pp-speaker">{escape(b.speaker or b.label_raw)}{off}'
                   + (f' <span class="pp-delivery">{extra}</span>' if extra else "") + '</div>')
        rows.append(
            f'<div id="{anchor(b.block_id)}" class="{" ".join(cls)}">'
            f'<div class="pp-line__time">{escape(b.start_raw or "")}<span>{clip.rel(b.start_ms)}</span></div>'
            f'<div class="pp-line__track">{TRACK_LABELS.get(b.block_type.value, b.block_type.value)}</div>'
            f'<div class="pp-line__body">{who}<div class="pp-text">{escape(b.text)}</div>'
            + (f'<div class="pp-tags">{"".join(tags)}</div>' if tags else "") + '</div></div>')
    head = (f'<div class="pp-script__head"><span>Screenplay</span>'
            f'<span>{format_timestamp(clip.start_ms)} – {format_timestamp(clip.end_ms)} · {len(clip.blocks)} blocks'
            f'</span></div>'
            f'<div class="pp-legend"><span class="pp-key pp-key--hook"></span>{hook_tag}'
            f'<span class="pp-key pp-key--beat"></span>{beat_tag}'
            + ('' if proposed else '<span class="pp-key pp-key--cited"></span>Cited evidence') + '</div>')
    return f'<div class="pp-script">{head}<div class="pp-script__body">{"".join(rows)}</div></div>'
