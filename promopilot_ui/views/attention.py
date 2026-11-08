"""Where it picks badly: candidates that did not pass, or carry a review flag, grouped by what went wrong."""
from html import escape

import streamlit as st

from promopilot.timecode import format_timestamp
from promopilot_ui import audit
from promopilot_ui.components import cards, layout
from promopilot_ui.status import DIMENSION_LABELS, EPISODE_STATES, gate_label, humanize, lane, rule_label

ctx = layout.context()
run = ctx.run.name
everything = audit.sections(ctx)
pending_eps_all = audit.pending_episodes(ctx)
n_fail = sum(len(everything[k]) for k in ("technical", "quality", "unusable", "pending"))
needs_review = sum(1 for v in everything["review"] if audit.review_status(v) == "needs_review")
layout.page_header("Where it picks badly",
                   f"{n_fail} of {len(ctx.candidates)} candidates did not pass · {needs_review} flagged for review · "
                   f"{len(pending_eps_all)} episode(s) awaiting processing", ctx)

if ctx.judge is None:
    layout.empty_state("No judge output for this run", "Candidates have not been judged yet, so there is nothing to "
                       "audit beyond pending work.", f"python -m promopilot.pipeline judge --run {run}")
    st.stop()

focus = st.query_params.get("focus")
focus = focus if focus in ctx.candidates else None
if focus:
    with st.container(horizontal=True, vertical_alignment="center", key="pp-focus"):
        st.html(f'<div class="pp-banner pp-banner--neutral">Showing entries for <b>{escape(focus)}</b> only.</div>')
        st.page_link(cards.ATTENTION_PAGE, label="Show all", icon=":material/close:", query_params={"run": run})

# --- filters ----------------------------------------------------------------------------------------
views = list(ctx.candidates.values())
c1, c2, c3, c4, c5 = st.columns([1.1, 1.5, 1.4, 1.5, 1.2], vertical_alignment="bottom")
f = audit.Filters(focus=focus)
f.episodes = set(c1.multiselect("Episode", sorted({v.episode_id for v in views} |
                                                  {e.episode_id for e in pending_eps_all}), placeholder="All"))
f.categories = set(c2.multiselect("Category", list(audit.CATEGORIES), format_func=audit.CATEGORIES.get,
                                  placeholder="All categories"))
f.archetypes = set(c3.multiselect("Archetype", sorted({audit.archetype_key(v) for v in views}),
                                  format_func=lambda k: "Not classified" if k == "not_classified" else humanize(k),
                                  placeholder="All archetypes"))
f.repairs = set(c4.multiselect("Repair status", list(audit.REPAIR_KEYS), format_func=audit.REPAIR_KEYS.get,
                               placeholder="Any"))
f.reviews = set(c5.multiselect("Review status", list(audit.REVIEW_STATUSES), format_func=audit.REVIEW_STATUSES.get,
                               placeholder="Any"))
secs = audit.sections(ctx, f)
candidate_only = bool(focus or f.archetypes or f.repairs or f.reviews)
pending_eps = [e for e in pending_eps_all if not candidate_only and (not f.episodes or e.episode_id in f.episodes)
               and (not f.categories or "pending" in f.categories)]
shown = [k for k in audit.CATEGORIES if not f.categories or k in f.categories]

# --- summary strip ----------------------------------------------------------------------------------
TONES = {"technical": "technical", "quality": "quality", "unusable": "neutral", "pending": "neutral",
         "review": "review"}
open_review = [v for v in secs["review"] if audit.review_status(v) == "needs_review"]
resolved_review = len(secs["review"]) - len(open_review)
tiles = []
for k, title in audit.CATEGORIES.items():
    n = len(open_review) if k == "review" else len(secs[k]) + (len(pending_eps) if k == "pending" else 0)
    note = f'<span class="pp-sub">+{resolved_review} resolved by reassessment</span>' \
        if k == "review" and resolved_review else ""
    tiles.append(f'<a class="pp-tile pp-tile--{TONES[k]}{" is-off" if k not in shown else ""}" '
                 f'href="#pp-att-{k}" target="_self"><b>{n}</b><span>{escape(title)}</span>{note}</a>')
st.html(f'<div class="pp-tiles">{"".join(tiles)}</div>')


# --- helpers ----------------------------------------------------------------------------------------
def link(cid):
    return cards.id_link(ctx, cid)


def span(start, end):
    if start is None or end is None:
        return '<span class="pp-none">—</span>'
    return (f'<span class="pp-num">{format_timestamp(start)} – {format_timestamp(end)}</span>'
            f'<small class="pp-sub">{(end - start) / 1000:.1f} s</small>')


def proposed_span(v):
    c = v.candidate
    return span(c.start_ms, c.end_ms) if c else span(*v.span)


def ep(v):
    return f'<span class="pp-ep">{escape(v.episode_id)}</span>'


def outcome(v):
    if v.lane == "eligible":
        if v.ranking_status == "selected":
            return f'<span class="pp-tone--ok">Shortlisted #{v.shortlist_rank}</span>'
        return '<span class="pp-tone--ok">Eligible, not shortlisted</span>' if v.ranking_status else \
            '<span class="pp-tone--ok">Eligible</span>'
    ln = lane(v.lane)
    return f'<span class="pp-tone--{ln.tone}">{escape(ln.label)}</span>'


def section(key, tone, intro, head, rows, empty_all, total_all, count=None):
    if key not in shown:
        return
    if rows:
        body = f'<table class="pp-grid"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    else:
        msg = empty_all if not total_all else "Nothing here matches the current filters."
        body = f'<p class="pp-none">{escape(msg)}</p>'
    count = sum(r.count('class="pp-first"') for r in rows) if count is None else count
    st.html(f'<section id="pp-att-{key}" class="pp-att pp-att--{tone}"><div class="pp-att__head">'
            f'<h2>{escape(audit.CATEGORIES[key])}</h2><span class="pp-count">{escape(str(count))}</span></div>'
            f'<p class="pp-att__intro">{intro}</p>{body}</section>')


def th(*names):
    return "".join(f"<th>{n}</th>" for n in names)


# --- technical failures -----------------------------------------------------------------------------
rows = []
for v in secs["technical"]:
    fails = audit.rule_failures(ctx, v) or [audit.RuleFailure("", "no failed rule recorded", "—", "—", None)]
    rs = audit.repair_state(ctx, v)
    rep = '<span class="pp-none">—</span>'
    if rs:
        rep = f'<b>{escape(rs.label)}</b>'
        if rs.detail:
            rep += f'<small class="pp-sub">{escape(rs.detail)}</small>'
        tv = ctx.candidates.get(rs.target) if rs.target and rs.key != "is_repair" else None
        if tv:
            rep += f'<div class="pp-sub">→ {link(tv.candidate_id)} · {span(*tv.span)}</div>'
        if rs.key not in ("not_attempted", "is_repair"):
            rep += f'<div class="pp-sub">{cards.id_link(ctx, v.candidate_id, "repairs", "focus", "Repair record")}</div>'
    n = len(fails)
    for i, fl in enumerate(fails):
        rule = (f'<b>{escape(rule_label(fl.rule))}</b><small class="pp-sub pp-code">{escape(fl.rule)}</small>'
                if fl.rule else '<span class="pp-none">Unrecognised rule</span>')
        cells = (f'<td>{rule}</td><td>{escape(fl.observed)}</td><td>{escape(fl.required)}'
                 + ('<small class="pp-sub">repairable rule</small>' if fl.repairable else "") + '</td>')
        if i == 0:
            rows.append(f'<tr class="pp-first"><td rowspan="{n}">{link(v.candidate_id)}</td><td rowspan="{n}">{ep(v)}'
                        f'</td><td rowspan="{n}">{proposed_span(v)}</td>{cells}<td rowspan="{n}">{rep}</td></tr>')
        else:
            rows.append(f"<tr>{cells}</tr>")
section("technical", "technical",
        "Broke a deterministic rule checked in Python. These are objective facts about the boundaries; the judge "
        "cannot override them. Values are from the final validation stored with the decision.",
        th("Candidate", "Episode", "Clip as proposed", "Failed rule", "Observed", "Required", "Repair"),
        rows, "No technical failures in this run.", len(everything["technical"]))

# --- quality failures -------------------------------------------------------------------------------
gates = cards.gate_minimums(ctx)


def gate_score(v, name):
    s = v.scores.get(name)
    if s is None:
        return '<span class="pp-none">—</span>'
    lo = gates.get(name)
    below = lo is not None and s < lo
    return (f'<span class="pp-num{" pp-below" if below else ""}">{s}</span>'
            + (f'<small class="pp-sub">min {lo}</small>' if lo is not None else ""))


rows = []
for v in secs["quality"]:
    gf = audit.gate_failures(v)
    why = "".join(
        f'<div class="pp-gate"><b>{escape(gate_label(g.gate))}</b> <span class="pp-muted">({escape(g.detail)})</span>'
        + (f'<div class="pp-why">{escape(DIMENSION_LABELS.get(g.dimension, ""))}: “{escape(g.reason)}”'
           + (f' <span class="pp-code">{escape(", ".join(b.split("_")[-1] for b in g.evidence))}</span>'
              if g.evidence else "") + '</div>' if g.reason else "") + '</div>' for g in gf) \
        or '<span class="pp-none">No failed gate recorded</span>'
    soft = v.effective.soft_rejections if v.judged else []
    if soft:
        why += '<div class="pp-sub">Judge noted: ' + escape("; ".join(
            f"{humanize(r.code)} ({r.severity})" for r in soft)) + '</div>'
    dna = v.effective.promo_dna if v.judged else None
    dna_html = (f'<b>{escape(humanize(dna.archetype.value))}</b><small class="pp-sub">Hook: '
                f'{escape(humanize(dna.hook_mechanism.value))}<br>Cliffhanger: '
                f'{escape(humanize(dna.cliffhanger_mechanism.value))}</small>') if dna else \
        '<span class="pp-none">—</span>'
    rows.append(f'<tr class="pp-first"><td>{link(v.candidate_id)}</td><td>{ep(v)}</td><td>{cards.score_inline(v)}</td>'
                f'<td>{gate_score(v, "hook")}</td><td>{gate_score(v, "cliffhanger")}</td>'
                f'<td>{gate_score(v, "context_independence")}</td><td class="pp-wide">{why}</td><td>{dna_html}</td></tr>')
section("quality", "quality",
        "Passed every technical rule, but the judge's assessment did not meet a promotional-quality gate. Scores are "
        "the effective (post-reassessment) judgement; the weighted score is diagnostic and not ranked.",
        th("Candidate", "Episode", "Score", "Hook", "Cliffhanger", "Context", "Failed gate and the judge's explanation",
           "Promo DNA"),
        rows, "No quality-gate failures in this run.", len(everything["quality"]))

# --- not judged / unusable --------------------------------------------------------------------------
rows = []
for v in secs["unusable"]:
    info = audit.unusable_info(v, run)
    rows.append(f'<tr class="pp-first"><td>{link(v.candidate_id)}</td><td>{ep(v)}</td><td>{proposed_span(v)}</td>'
                f'<td>{outcome(v)}</td><td><b>{escape(info.kind)}</b>'
                + (f'<small class="pp-sub">{escape(info.detail)}</small>' if info.detail else "")
                + f'</td><td>{escape(info.next_step)}</td></tr>')
section("unusable", "neutral",
        "No usable semantic judgement exists, so no scores are shown and these are not quality results.",
        th("Candidate", "Episode", "Clip as proposed", "State", "What happened", "Further processing"),
        rows, "Every candidate in this run has a usable judgement.", len(everything["unusable"]))

# --- pending ----------------------------------------------------------------------------------------
rows = []
for v in secs["pending"]:
    info = audit.unusable_info(v, run)
    rows.append(f'<tr class="pp-first"><td>{link(v.candidate_id)}</td><td>{ep(v)}</td><td>Candidate</td>'
                f'<td>{escape(info.kind)}</td><td>{escape(info.next_step)}</td></tr>')
for e in pending_eps:
    label = EPISODE_STATES.get(e.state, (humanize(e.state), ""))[0]
    num = e.episode_id[2:].lstrip("0") or "0"
    step = (f"Run `python -m promopilot.pipeline generate --run {run} --episodes {num}`."
            if e.state == "generation_pending" else f"Run `python -m promopilot.pipeline judge --run {run}`.")
    reason = escape(e.generation_reason or "no reason recorded")
    rows.append(f'<tr class="pp-first"><td><b>{escape(e.episode_id)}</b></td><td>—</td><td>Episode</td>'
                f'<td>{escape(label)}<small class="pp-sub">{reason}</small></td><td>{escape(step)}</td></tr>')
section("pending", "neutral",
        "Work the pipeline has not finished. An episode awaiting generation has not been evaluated, so it is not a "
        "“no strong promo” result.",
        th("Item", "Episode", "Level", "State", "Next step"),
        rows, "Nothing is pending in this run.", len(everything["pending"]) + len(pending_eps_all))

# --- review flags -----------------------------------------------------------------------------------
rows = []
ordered = sorted(secs["review"], key=lambda v: audit.review_status(v) != "needs_review")
for v in ordered:
    items = audit.review_items(v) or [audit.ReviewItem("", "", "; ".join(v.review_flags), None, None, None,
                                                       "No reassessment record", True)]
    rec = v.reassessment
    status = humanize(rec.status) if rec else "—"
    n = len(items)
    for i, it in enumerate(items):
        flag = (f'<b>{escape(DIMENSION_LABELS.get(it.dimension, it.dimension) or "—")}</b>'
                f'<small class="pp-sub"><span class="pp-code">{escape(it.code)}</span> {escape(it.detail)}</small>')
        still = '<span class="pp-tone--review">Yes</span>' if it.still_flagged else '<span class="pp-muted">No</span>'
        cells = (f'<td class="pp-wide">{flag}</td><td class="pp-num">{"—" if it.original is None else it.original}</td>'
                 f'<td class="pp-num">{"—" if it.revised is None else it.revised}</td>'
                 f'<td class="pp-num">{"—" if it.effective is None else it.effective}</td>'
                 f'<td>{escape(it.outcome)}</td><td>{still}</td>')
        if i == 0:
            rows.append(f'<tr class="pp-first"><td rowspan="{n}">{link(v.candidate_id)}'
                        f'<small class="pp-sub">{escape(audit.REVIEW_STATUSES[audit.review_status(v)])}</small></td>'
                        f'<td rowspan="{n}">{outcome(v)}<small class="pp-sub">reassessment: {escape(status)}</small>'
                        f'</td>{cells}</tr>')
        else:
            rows.append(f"<tr>{cells}</tr>")
section("review", "review",
        "A consistency check found a contradiction between a score and the judge's own explanation, and one targeted "
        "reassessment was requested. A review flag never changes the verdict: the final outcome column is the stored "
        "decision. Resolved cases are listed after open ones for audit.",
        th("Candidate", "Final outcome", "Flagged dimension and contradiction", "Original", "Revised", "Effective",
           "Reassessment result", "Still flagged"),
        rows, "No consistency flags in this run.", len(everything["review"]),
        count=f"{len(open_review)} open · {resolved_review} resolved")
