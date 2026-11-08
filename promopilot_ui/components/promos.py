"""The brief's deliverable, and nothing more: each ranked promo with its episode, segment timestamps, hook,
cliffhanger, one line on why it works and why it holds its rank; then where the tool picks badly. Every value is
read from the stored ranking (`ranking.json` shortlist rows) and decisions; nothing is re-scored."""
from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

from promopilot_ui import audit, data
from promopilot_ui.components import cards
from promopilot_ui.components.layout import badge
from promopilot_ui.status import DIMENSION_LABELS, TRACK_LABELS, gate_label, rank_reason, rule_label

DOWNLOADS = (("shortlist.csv", "text/csv"), ("candidates.csv", "text/csv"), ("catalogue.csv", "text/csv"),
             ("report.md", "text/markdown"), ("ranking.json", "application/json"))
_MEASURED_RULES = ("DURATION_RANGE", "HOOK_WINDOW", "SUSPENSE_TAIL_TOLERANCE", "ENDING_NOT_TRANSITION_ONLY")


def rows(ctx: data.RunData) -> list[dict]:
    return list((ctx.ranking or {}).get("shortlist", []))


def _clip(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _beat(row: dict, prefix: str) -> str:
    text = row.get(f"{prefix}_text") or ""
    if not text:
        return '<span class="pp-none">Not recorded (see the full analysis)</span>'
    when = (row.get(f"{prefix}_timestamp") or "").split("-")[0]
    who = row.get(f"{prefix}_speaker") or TRACK_LABELS.get(row.get(f"{prefix}_block_type") or "", "")
    return (f'<span class="pp-num">{escape(when)}</span> '
            + (f'<b>{escape(who)}</b> ' if who else "") + escape(text))


def promo_card(row: dict, reason: str) -> str:
    segment = (f'<span class="pp-num">{escape(row.get("start", ""))} → {escape(row.get("end", ""))}</span>'
               f' <span class="pp-muted">· {row.get("duration_s", 0):.1f} s</span>')
    flag = badge("Needs review", "review", "The judge's scores contradicted its own explanation; see the full "
                                           "analysis.") if row.get("needs_review") else ""
    fields = [("Segment 1", segment), ("Hook", _beat(row, "hook")), ("Cliffhanger", _beat(row, "cliffhanger")),
              ("Why it works", escape(row.get("explanation") or "") or '<span class="pp-none">Not recorded</span>'),
              ("Why this rank", escape(reason))]
    body = "".join(f'<dt>{escape(k)}</dt><dd>{v}</dd>' for k, v in fields)
    return (f'<article class="pp-promo"><div class="pp-promo__head"><span class="pp-rank">#{row.get("rank")}</span>'
            f'<span class="pp-promo__ep">Episode {int(row["episode_id"][2:])}</span>'
            f'<span class="pp-cid">{escape(row["candidate_id"])}</span>{flag}</div>'
            f'<dl class="pp-promo__dl">{body}</dl></article>')


def promo_list(ctx: data.RunData, link: bool = True) -> None:
    shortlist = rows(ctx)
    if not shortlist:
        st.html('<p class="pp-none">No clip passed every hard rule and quality gate, so nothing is ranked.</p>')
        return
    for i, row in enumerate(shortlist):
        with st.container(key=f"pp-promo-{row['candidate_id']}"):
            st.html(promo_card(row, rank_reason(shortlist, i)))
            if link:
                st.page_link(cards.CANDIDATE_PAGE, label="Full analysis", icon=":material/arrow_forward:",
                             query_params={"cid": row["candidate_id"], "run": ctx.run.name})


# --- where it picks badly -------------------------------------------------------------------------------

def _technical(ctx: data.RunData, v: data.CandidateView) -> str:
    fl = next(iter(audit.rule_failures(ctx, v)), None)
    if fl is None:
        return "broke a hard rule"
    rule = rule_label(fl.rule) if fl.rule else "Hard rule"
    need = f" (required: {fl.required})" if fl.rule in _MEASURED_RULES else ""
    return f"Failed “{rule}”: {fl.observed}{need}."


def _quality(v: data.CandidateView) -> str:
    g = next(iter(audit.gate_failures(v)), None)
    if g is None:
        return "failed a quality gate"
    if g.gate.endswith("_score") and g.dimension:
        out = f"{DIMENSION_LABELS.get(g.dimension, g.dimension)} scored {g.detail}"
    else:
        out = f"Failed “{gate_label(g.gate)}” ({g.detail})"
    return out + (f". Judge: “{_clip(g.reason, 180)}”" if g.reason else ".")


def _review(v: data.CandidateView) -> str:
    it = next(iter(audit.review_items(v)), None)
    if it is None:
        return "; ".join(v.review_flags) or "flagged for review"
    return f"{DIMENSION_LABELS.get(it.dimension, it.dimension)}: {_clip(it.detail, 180)}"


def failures(ctx: data.RunData) -> list[tuple[str, str, list[tuple[data.CandidateView, str]]]]:
    """(title, what it means, [(candidate, the concrete reason)]) per kind of bad pick, from stored outcomes."""
    secs = audit.sections(ctx)
    review = [v for v in secs["review"] if audit.review_status(v) == "needs_review"]
    return [
        ("Broke a hard rule", "Proposed clips with bad boundaries, caught by Python and never ranked.",
         [(v, _technical(ctx, v)) for v in secs["technical"]]),
        ("Too weak to post", "Technically valid clips the judge scored below a quality gate.",
         [(v, _quality(v)) for v in secs["quality"]]),
        ("Judge contradicted itself", "A score disagreed with the judge's own explanation; kept, but flagged for a "
                                      "human.", [(v, _review(v)) for v in review]),
        ("No usable judgement", "The judge request failed or its reply failed checks, so there is no quality result.",
         [(v, (audit.unusable_info(v, ctx.run.name) or audit.Unusable("", "", "", False)).kind)
          for v in secs["unusable"]]),
    ]


def picks_badly(ctx: data.RunData, examples: Optional[int] = 2, link: bool = True) -> None:
    """Counts of every kind of bad pick plus concrete examples (`examples=None` lists them all)."""
    groups = failures(ctx)
    total = len(ctx.candidates)
    bad = sum(len(items) for _, _, items in groups[:2]) + len(groups[3][2])
    items_html = []
    for title, meaning, items in groups:
        if not items and title == "No usable judgement":
            continue
        shown = items if examples is None else items[:examples]
        lis = "".join(f'<li><span class="pp-cid">{escape(v.candidate_id)}</span> {escape(why)}</li>'
                      for v, why in shown)
        more = len(items) - len(shown)
        tail = f'<li class="pp-muted">…and {more} more.</li>' if more > 0 else ""
        items_html.append(f'<div class="pp-bad"><div class="pp-bad__head"><b class="pp-num">{len(items)}</b> '
                          f'<b>{escape(title)}</b><span class="pp-muted"> · {escape(meaning)}</span></div>'
                          + (f'<ul>{lis}{tail}</ul>' if lis else '<p class="pp-none">None in this run.</p>')
                          + '</div>')
    st.html(f'<p>{bad} of the {total} candidate clips the tool proposed were not good enough to rank; the failures '
            'below are the ones it caught itself.</p>' + "".join(items_html)
            + '<p class="pp-disclaimer">What it cannot catch: every score comes from one AI judge, not from viewers. '
              'Without a human review or audience data, a shortlisted clip can still be a weak promo.</p>')
    if link:
        st.page_link(cards.ATTENTION_PAGE, label="Every failure, with evidence", icon=":material/arrow_forward:",
                     query_params={"run": ctx.run.name})


def downloads(ctx: data.RunData) -> None:
    if ctx.judge is None:
        st.html('<p class="pp-none">No ranking files yet.</p>')
        return
    folder = ctx.run.path / "ranking" / ctx.judge.name
    with st.container(horizontal=True):
        for name, mime in DOWNLOADS:
            path = folder / name
            if path.exists():
                st.download_button(name, path.read_bytes(), file_name=name, mime=mime, icon=":material/download:")
