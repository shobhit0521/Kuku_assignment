"""Repairs: every recorded boundary repair, original against repaired, with the stored provenance."""
from html import escape

import streamlit as st

from promopilot.timecode import format_timestamp
from promopilot_ui import audit
from promopilot_ui.components import cards, layout
from promopilot_ui.status import humanize, lane, rule_label

ctx = layout.context()
run = ctx.run.name
records = audit.repair_records(ctx)
summary = audit.repair_summary(ctx, records)
layout.page_header("Repairs", f"{summary.attempts} repair attempt(s) recorded · {summary.succeeded} revalidated and "
                              f"passing · {summary.became_eligible} became eligible", ctx)

if ctx.judge is None:
    layout.empty_state("No judge output for this run", "Repairs run after judging, so none can exist yet.",
                       f"python -m promopilot.pipeline judge --run {run}")
    st.stop()

focus = st.query_params.get("focus")
focus = focus if any(r.original.candidate_id == focus for r in records) else None
if focus:
    with st.container(horizontal=True, vertical_alignment="center", key="pp-focus"):
        st.html(f'<div class="pp-banner pp-banner--neutral">Showing the repair of <b>{escape(focus)}</b> only.</div>')
        st.page_link(cards.REPAIRS_PAGE, label="Show all", icon=":material/close:", query_params={"run": run})

tiles = [(summary.attempts, "Repair attempts", "accent"), (summary.succeeded, "Revalidated and passing", "ok"),
         (summary.became_eligible, "Became eligible", "ok"), (summary.shortlisted, "Reached the shortlist", "ok"),
         (summary.not_resolved, "Did not resolve the failure", "technical"),
         (summary.technical_without_attempt, "Technical failures with no attempt", "neutral")]
st.html('<div class="pp-tiles">' + "".join(f'<div class="pp-tile pp-tile--{t}"><b>{n}</b><span>{escape(k)}</span></div>'
                                           for n, k, t in tiles) + '</div>'
        '<p class="pp-muted">Counted once per original candidate from the judge decisions. A repair counts as '
        'passing only when the repaired clip\'s own stored decision shows every hard rule passing.</p>')

settings, source = audit.settings_for(ctx, next(iter(ctx.validation_settings), ""))


def secs(key):
    v = settings.get(key)
    return "?" if v is None else f"{v / 1000:g}"


with st.expander("How repair works", expanded=not records):
    st.html(
        '<dl class="pp-dl pp-dl--wide">'
        '<dt>When</dt><dd>After judging, when a candidate with a valid judgement fails a hard rule. The judge\'s hook '
        'and suspense beat are used, so the repair does not rely on the generator\'s pitch.</dd>'
        '<dt>How often</dt><dd>One attempt per candidate. It creates a new candidate <code>&lt;id&gt;_R1</code>; a '
        'repaired candidate is never repaired again.</dd>'
        '<dt>Repairable</dt><dd>Duration too short, a dialogue line cut at a boundary, a transition-only ending, or '
        'too long a tail after the suspense beat. Invalid references, boundary order, a hook outside the window, a '
        'suspense beat outside the clip or not a story beat, and clips over the maximum length are not repaired.</dd>'
        f'<dt>Allowed moves</dt><dd>Trim the end back to the suspense beat (or the last story beat); widen a boundary '
        f'to include a whole dialogue line; move the start earlier only while the hook stays within {secs("hook_window_ms")} s '
        'of the start.</dd>'
        f'<dt>Revalidation</dt><dd>Every rule is re-checked on the new boundaries: {secs("min_clip_ms")}–'
        f'{secs("max_clip_ms")} s, hook within {secs("hook_window_ms")} s, no dialogue cut, suspense beat inside the clip '
        f'and at most {secs("suspense_tail_ms")} s before the end. If any rule still fails, the attempt is recorded as '
        'not possible.</dd>'
        '<dt>Afterwards</dt><dd>The repaired clip is judged from scratch, blind, as a new candidate. The original '
        'candidate, its judgement and its decision are kept unchanged.</dd>'
        '<dt>Preliminary check</dt><dd>The <code>validate</code> command also checks clips before judging, using the '
        'generator\'s beats (provisional). It is shown below for reference and not counted as a repair.</dd>'
        f'</dl><p class="pp-muted">Thresholds: {escape(source)}.</p>')


def ts(ms):
    return '<span class="pp-none">—</span>' if ms is None else f'<span class="pp-num">{format_timestamp(ms)}</span>'


def delta(a, b):
    if a is None or b is None or a == b:
        return '<span class="pp-muted">unchanged</span>' if a is not None and a == b else ""
    d = (b - a) / 1000
    return f'<span class="pp-delta">{d:+.1f} s</span>'


def validation_text(view):
    d = view.decision if view else None
    if d is None or d.validation is None:
        return '<span class="pp-none">not recorded</span>'
    failed = [c for c in d.validation.checks if c.status.value == "fail"]
    if failed:
        return "".join(f'<div class="pp-tone--technical">✕ {escape(rule_label(c.rule))}'
                       f'<small class="pp-sub">{escape(c.message)}</small></div>' for c in failed)
    passed = sum(c.status.value == "pass" for c in d.validation.checks)
    src = "judge-supplied beats" if not d.validation.provisional else "provisional beats"
    return f'<span class="pp-tone--ok">All {passed} rules pass</span><small class="pp-sub">{src}</small>'


def outcome(view):
    if view is None:
        return '<span class="pp-none">—</span>'
    ln = lane(view.lane)
    extra = ""
    if view.lane == "eligible" and view.ranking_status == "selected":
        extra = f" · shortlisted #{view.shortlist_rank}"
    elif view.lane == "eligible" and view.ranking_status:
        extra = " · not shortlisted"
    return f'<span class="pp-tone--{ln.tone}">{escape(ln.label)}{extra}</span>'


def judgement_text(view, is_new):
    if view is None or view.raw is None:
        return '<span class="pp-none">no judgement</span>'
    r = view.raw
    label = "New blind judgement" if is_new else "Judgement"
    return (f'{label}<small class="pp-sub">{escape(layout.local_time(r.judged_at))} · clip '
            f'{escape(r.clip_sha256[:10])}</small>')


def bar(o_s, o_e, r_s, r_e):
    pts = [p for p in (o_s, o_e, r_s, r_e) if p is not None]
    if len(pts) < 4:
        return ""
    lo, hi = min(pts), max(pts)
    width = max(hi - lo, 1)

    def seg(s, e, cls):
        return (f'<div class="pp-span__bar {cls}" style="left:{(s - lo) / width * 100:.2f}%;'
                f'width:{max((e - s) / width * 100, .5):.2f}%"></div>')
    return (f'<div class="pp-span"><div class="pp-span__row"><span>Original</span><div class="pp-span__track">'
            f'{seg(o_s, o_e, "is-orig")}</div></div><div class="pp-span__row"><span>Repaired</span>'
            f'<div class="pp-span__track">{seg(r_s, r_e, "is-rep")}</div></div>'
            f'<div class="pp-span__axis"><span>{format_timestamp(lo)}</span><span>{format_timestamp(hi)}</span></div>'
            f'</div>')


shown = [r for r in records if not focus or r.original.candidate_id == focus]
if not records:
    layout.empty_state("No repairs in this run", "No judged candidate failed a repairable hard rule, so no repair "
                       "was attempted." if not summary.technical_without_attempt else
                       f"{summary.technical_without_attempt} technical failure(s) had no repair attempt; see "
                       "Needs attention for the reasons.")

for rec in shown:
    o, r, prov = rec.original, rec.repaired, rec.provenance
    o_s, o_e = (prov.original_start_ms, prov.original_end_ms) if prov else o.span
    r_s, r_e = r.span if r else ((prov.new_start_ms, prov.new_end_ms) if prov else (None, None))
    o_dur = o_e - o_s if o_s is not None and o_e is not None else None
    r_dur = r_e - r_s if r_s is not None and r_e is not None else None
    triggers = ", ".join(rule_label(t.rule) if t.rule else t.message for t in rec.triggers) or "—"
    state_tone = "ok" if rec.succeeded else "technical"
    title = (f'{cards.id_link(ctx, o.candidate_id)} <span class="pp-arrow">→</span> '
             + (cards.id_link(ctx, r.candidate_id) if r else '<span class="pp-none">no repaired clip</span>'))
    comp = (
        '<table class="pp-grid pp-grid--compare"><thead><tr><th></th><th>Original</th><th>Repaired</th>'
        '<th>Change</th></tr></thead><tbody>'
        f'<tr><th>Candidate</th><td>{cards.id_link(ctx, o.candidate_id)}</td>'
        f'<td>{cards.id_link(ctx, r.candidate_id) if r else "—"}</td><td></td></tr>'
        f'<tr><th>Start</th><td>{ts(o_s)}</td><td>{ts(r_s)}</td><td>{delta(o_s, r_s) if r else ""}</td></tr>'
        f'<tr><th>End</th><td>{ts(o_e)}</td><td>{ts(r_e)}</td><td>{delta(o_e, r_e) if r else ""}</td></tr>'
        f'<tr><th>Duration</th><td class="pp-num">{"—" if o_dur is None else f"{o_dur / 1000:.1f} s"}</td>'
        f'<td class="pp-num">{"—" if r_dur is None else f"{r_dur / 1000:.1f} s"}</td>'
        f'<td>{delta(o_dur, r_dur) if r else ""}</td></tr>'
        f'<tr><th>Final validation</th><td>{validation_text(o)}</td><td>{validation_text(r)}</td><td></td></tr>'
        f'<tr><th>Status</th><td>{outcome(o)}</td><td>{outcome(r)}</td><td></td></tr>'
        f'<tr><th>Weighted score</th><td>{cards.score_inline(o)}</td><td>{cards.score_inline(r) if r else "—"}</td>'
        '<td></td></tr>'
        f'<tr><th>Judgement</th><td>{judgement_text(o, False)}</td><td>{judgement_text(r, rec.new_judgement)}</td>'
        '<td></td></tr></tbody></table>')
    actions = (prov.actions if prov else []) or []
    pre = rec.preliminary
    pre_text = "not run" if pre is None else (
        f"{humanize(pre.status)}{' (provisional)' if pre.provisional else ''}"
        + (f" — failed {', '.join(x.split(':')[0] for x in pre.failed)}" if pre.failed else "")
        + f"; preview repair: {humanize(pre.repair_status)}")
    prov_rows = [
        ("Repair type", "<br>".join(f"{escape(humanize(a.action))} <span class='pp-muted'>({escape(a.detail)})</span>"
                                    for a in actions) or '<span class="pp-none">none recorded</span>'),
        ("Triggered by", escape(triggers)),
        ("Recorded outcome", f'{escape(humanize(o.decision.repair_status))} — {escape(o.decision.repair_reason or "")}'),
        ("Original boundaries", (f'{ts(prov.original_start_ms)} – {ts(prov.original_end_ms)} '
                                 f'<span class="pp-code">{escape(prov.original_start_block_id)} … '
                                 f'{escape(prov.original_end_block_id)}</span>') if prov else "—"),
        ("Final boundaries", f'{ts(prov.new_start_ms)} – {ts(prov.new_end_ms)}' if prov else "—"),
        ("Suspense beat used", (f'<span class="pp-code">{escape(prov.suspense_beat_id or "none")}</span> '
                                f'(source: {escape(prov.suspense_beat_source or "—")})') if prov else "—"),
        ("Revalidated", escape(rec.state.label)),
        ("New judgement", "Yes — judged blind as a new candidate" if rec.new_judgement else
         ("No — no judgement stored for the repaired clip" if r else "No repaired clip")),
        ("Validator", escape(prov.validator_version) if prov else "—"),
        ("Preliminary check", escape(pre_text)),
    ]
    prov_html = '<table class="pp-table">' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in prov_rows) + \
        "</table>"
    st.html(f'<section class="pp-repair"><div class="pp-repair__head"><div class="pp-repair__title">{title}</div>'
            f'<span class="pp-tone--{state_tone}">{escape(rec.state.label)}</span></div>'
            f'<p class="pp-muted">Triggered by: {escape(triggers)}</p>'
            f'<div class="pp-repair__body"><div>{comp}{bar(o_s, o_e, r_s, r_e) if r else ""}</div>'
            f'<div><div class="pp-repair__k">Provenance</div>{prov_html}'
            f'<div class="pp-sub">{cards.id_link(ctx, o.candidate_id, "attention", "focus", "→ Open the Needs attention entry")}'
            f'</div></div></div></section>')
