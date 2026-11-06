"""Candidate cards, score pips and status badges. Values come from CandidateView; nothing is recomputed."""
from __future__ import annotations

from html import escape
from typing import Optional
from urllib.parse import quote

import streamlit as st

from promopilot.timecode import format_timestamp
from promopilot_ui import data
from promopilot_ui.components.layout import badge
from promopilot_ui.status import SHORT_LABELS, WEIGHTED, fmt_seconds, humanize, lane

CANDIDATE_PAGE = "views/candidate.py"
ATTENTION_PAGE = "views/attention.py"
REPAIRS_PAGE = "views/repairs.py"


def gate_minimums(ctx: data.RunData) -> dict:
    return ((ctx.ranking or {}).get("settings") or {}).get("gate_minimums") \
        or (ctx.judge_summary or {}).get("gate_minimums") or {}


def pips(score: Optional[int], gate: Optional[int] = None, size: str = "") -> str:
    if score is None:
        return '<span class="pp-pips pp-pips--none" title="Not scored">—</span>'
    cells = []
    for i in range(1, 6):
        cls = "pp-pip is-on" if i <= score else "pp-pip"
        if gate and i == gate:
            cls += " is-gate"
        cells.append(f'<i class="{cls}"></i>')
    title = f"{score} of 5" + (f" · gate minimum {gate}" if gate else "")
    return f'<span class="pp-pips {size}" title="{title}" aria-label="{title}">{"".join(cells)}</span>'


def dims_row(view: data.CandidateView, gates: dict, names=WEIGHTED) -> str:
    if not view.judged:
        return '<div class="pp-dims pp-dims--none">No scores: this candidate has no valid judgement.</div>'
    cells = "".join(f'<div class="pp-dim"><span class="pp-dim__k">{SHORT_LABELS[n]}</span>'
                    f'<span class="pp-dim__v">{pips(view.scores.get(n), gates.get(n))}'
                    f'<b>{view.scores.get(n, "—")}</b></span></div>' for n in names)
    return f'<div class="pp-dims">{cells}</div>'


def dims_inline(view: data.CandidateView, names=WEIGHTED) -> str:
    if not view.judged:
        return '<span class="pp-none">not scored</span>'
    return "".join(f'<span class="pp-mini"><span>{SHORT_LABELS[n]}</span><b>{view.scores.get(n, "—")}</b></span>'
                   for n in names)


def status_badges(view: data.CandidateView, with_lane: bool = True) -> str:
    out = []
    ln = lane(view.lane)
    if with_lane and view.lane != "eligible":
        out.append(badge(ln.label, ln.tone, ln.explanation))
    if view.repaired_from:
        out.append(badge(f"Repaired from {view.repaired_from}", "accent",
                         "Boundaries were adjusted by the deterministic repair step and the new clip was judged "
                         "from scratch."))
    if view.repaired_to:
        out.append(badge(f"Repaired as {view.repaired_to}", "neutral"))
    if view.review_flags:
        out.append(badge("Needs review", "review", " | ".join(view.review_flags)))
    return "".join(out)


DIAGNOSTIC_LABEL = "Diagnostic score — not ranked"
DIAGNOSTIC_HELP = ("Persisted weighted score of a candidate that failed a hard rule or a quality gate. Shown for "
                   "diagnosis only; it does not compete in the ranking.")


def score_inline(view: data.CandidateView) -> str:
    """A weighted score for tables: plain when eligible, muted and labelled when diagnostic, a dash when none."""
    if view.score_role is None:
        return '<span class="pp-none" title="No valid judgement, so no score">—</span>'
    value = f"{view.weighted_score:.2f}"
    if view.score_role == "diagnostic":
        return (f'<span class="pp-diag" title="{escape(DIAGNOSTIC_HELP)}">{value}'
                f'<small>diagnostic</small></span>')
    return f'<b class="pp-num">{value}</b>'


def score_block(view: data.CandidateView) -> str:
    """The large score in a candidate header."""
    if view.score_role is None:
        return '<div class="pp-bigscore pp-bigscore--none"><b>—</b><span>no score</span></div>'
    value = f"{view.weighted_score:.2f}"
    if view.score_role == "diagnostic":
        return (f'<div class="pp-bigscore pp-bigscore--diag" title="{escape(DIAGNOSTIC_HELP)}"><b>{value}</b>'
                f'<span>{DIAGNOSTIC_LABEL}</span></div>')
    return f'<div class="pp-bigscore"><b>{value}</b><span>weighted score</span></div>'


def span_text(view: data.CandidateView) -> str:
    s, e = view.span
    if s is None or e is None:
        return "Boundaries unavailable"
    return f"{format_timestamp(s)} → {format_timestamp(e)} · {fmt_seconds(e - s)}"


def appeal(view: data.CandidateView) -> Optional[str]:
    if view.judged and view.effective.promo_dna and view.effective.promo_dna.appeal:
        return view.effective.promo_dna.appeal
    return None


def shortlist_card(view: data.CandidateView, ctx: data.RunData) -> None:
    gates = gate_minimums(ctx)
    rank = view.shortlist_rank
    ap = appeal(view)
    with st.container(key=f"pp-card-{view.candidate_id}"):
        st.html(
            f'<div class="pp-card">'
            f'<div class="pp-card__head"><span class="pp-rank">#{rank}</span>'
            f'<span class="pp-cid">{escape(view.candidate_id)}</span><span class="pp-ep">{escape(view.episode_id)}</span>'
            f'<span class="pp-score" title="Weighted promotional score (production rubric)">'
            f'{view.weighted_score:.2f}</span></div>'
            f'<div class="pp-meta">{escape(span_text(view))}<span class="pp-dot">·</span>'
            f'<span class="pp-arch">{escape(humanize(view.archetype))}</span></div>'
            + (f'<p class="pp-appeal">{escape(ap)}</p>' if ap else '')
            + dims_row(view, gates)
            + (f'<div class="pp-badges">{status_badges(view)}</div>' if status_badges(view) else '')
            + '</div>')
        st.page_link(CANDIDATE_PAGE, label="Open analysis", icon=":material/arrow_forward:",
                     query_params={"cid": view.candidate_id, "run": ctx.run.name})


def candidate_href(ctx: data.RunData, cid: str, page: str = "candidate", param: str = "cid") -> str:
    """Relative link for HTML tables; page links (`st.page_link`) are used where a widget fits."""
    return f"{page}?run={quote(ctx.run.name)}&{param}={quote(cid)}"


def id_link(ctx: data.RunData, cid: str, page: str = "candidate", param: str = "cid", label: str = "") -> str:
    return (f'<a class="pp-idlink" target="_self" href="{escape(candidate_href(ctx, cid, page, param))}">'
            f'{escape(label or cid)}</a>')


def compact_row(view: data.CandidateView, ctx: data.RunData, note: str) -> None:
    score = f"{view.weighted_score:.2f}" if view.weighted_score is not None else "—"
    with st.container(key=f"pp-row-{view.candidate_id}", horizontal=True, vertical_alignment="center", wrap=False):
        st.html(
            f'<div class="pp-row">'
            f'<div class="pp-row__id"><span class="pp-cid">{escape(view.candidate_id)}</span>'
            f'<span class="pp-meta">{escape(span_text(view))}</span></div>'
            f'<div class="pp-row__score">{score}</div>'
            f'<div class="pp-row__arch">{escape(humanize(view.archetype))}</div>'
            f'<div class="pp-row__mini">{dims_inline(view)}</div>'
            f'<div class="pp-row__note">{escape(note)}{status_badges(view)}</div>'
            f'</div>')
        st.page_link(CANDIDATE_PAGE, label="Open", icon=":material/arrow_forward:",
                     query_params={"cid": view.candidate_id, "run": ctx.run.name})
