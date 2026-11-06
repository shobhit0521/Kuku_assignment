"""Candidate intelligence: why a clip works (or why it did not pass), linked to the screenplay lines."""
from html import escape

import streamlit as st

from promopilot.timecode import format_timestamp
from promopilot_ui import audit, data
from promopilot_ui.components import cards, evidence, layout
from promopilot_ui.components.layout import badge
from promopilot_ui.status import (DIMENSION_LABELS, OUTCOME_LABELS, REASSESSMENT_LABELS, fmt_seconds, gate_label,
                                  humanize, lane, rule_label, selection_note)

ctx = layout.context()
layout.page_header("Candidate intelligence", "", ctx)
views = ctx.candidates
if not views:
    layout.empty_state("No candidates in this run", "Candidate generation has not produced any candidates yet.",
                       f"python -m promopilot.pipeline generate --run {ctx.run.name}")
    st.stop()

ids = list(views)
wanted = st.query_params.get("cid")
default = wanted if wanted in views else (ctx.shortlist[0].candidate_id if ctx.shortlist else ids[0])


def option_label(cid):
    v = views[cid]
    rank = f" · shortlist #{v.shortlist_rank}" if v.ranking_status == "selected" else ""
    return f"{cid} · {lane(v.lane).label}{rank}"


choice = st.selectbox("Candidate", ids, index=ids.index(default), format_func=option_label,
                      help="Every candidate in this run, including repaired versions.")
if st.query_params.get("cid") != choice:
    st.query_params["cid"] = choice
view: data.CandidateView = views[choice]
episode, ep_issue = layout.episode(ctx, view.episode_id)
start, end = view.span
clip = evidence.Clip(episode, start, end)
j = view.effective if view.judged else None
dec = view.decision
val = dec.validation if dec else None
gates = cards.gate_minimums(ctx)
dims = j.dimensions if j else {}
dna = j.promo_dna if j else None
cand = view.candidate


def ch(ids_):
    return evidence.chips(ids_ or [], clip)


def quote(block_id):
    b = clip.block(block_id) if block_id else None
    if b is None:
        return '<span class="pp-none">no block selected</span>'
    who = f"<b>{escape(b.speaker)}</b> " if b.speaker else ""
    return (f'<a class="pp-quote" href="#{evidence.anchor(block_id)}"><span class="pp-quote__t">'
            f'{escape(b.start_raw or "")} · {clip.rel(b.start_ms)}</span>{who}{escape(b.text)}</a>')


def field(label, value_html):
    return f'<div class="pp-field"><div class="pp-field__k">{escape(label)}</div><div class="pp-field__v">{value_html}</div></div>'


def text(value):
    return escape(value) if value else '<span class="pp-none">—</span>'


def bullet_list(items):
    items = [i for i in items if i]
    if not items:
        return '<span class="pp-none">none</span>'
    return "<ul class='pp-list'>" + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def score_note(name):
    eff, raw = view.scores.get(name), view.raw_scores.get(name)
    notes = []
    if raw is not None and eff is not None and raw != eff:
        notes.append(f'<span class="pp-revised">Original {raw} → effective {eff} after an accepted reassessment'
                     f'</span>')
    d = dims.get(name)
    if d is not None and d.review:
        notes.append(badge("Needs review", "review", " | ".join(d.review_reasons)))
    return "".join(notes)


def section(name, body, title=None):
    score = view.scores.get(name) if name else None
    head = (f'<div class="pp-sec__head"><h3>{escape(title or DIMENSION_LABELS.get(name, name))}</h3>'
            + (f'{cards.pips(score, gates.get(name), "pp-pips--lg")}<b class="pp-sec__score">{score}</b>'
               if score is not None else "")
            + f'{score_note(name) if name else ""}</div>')
    st.html(f'<section class="pp-sec">{head}{body}</section>')


# --- header ------------------------------------------------------------------------------------------
ln = lane(view.lane)
score_html = cards.score_block(view)
meta = [cards.span_text(view)]
if val and val.hook_offset_ms is not None:
    meta.append(f"hook at +{val.hook_offset_ms / 1000:.1f} s")
if val and val.ending and val.ending.trailing_after_suspense_ms is not None:
    meta.append(f"ends {fmt_seconds(val.ending.trailing_after_suspense_ms)} after the suspense beat")
note = selection_note(view.ordered_row, ctx.ranking) if view.lane == "eligible" else None
st.html(
    f'<div class="pp-cand-head"><div class="pp-cand-head__main">'
    f'<div class="pp-cand-head__ids"><span class="pp-ep">{escape(view.episode_id)}</span>'
    f'<span class="pp-cand-title">{escape(view.candidate_id)}</span>{badge(ln.label, ln.tone, ln.explanation)}'
    f'{cards.status_badges(view, with_lane=False)}</div>'
    f'<div class="pp-meta">{escape(" · ".join(meta))}</div>'
    + (f'<div class="pp-selection">{escape(note)}</div>' if note else "")
    + f'</div>{score_html}</div>')
if ep_issue:
    layout.banner("neutral", f"Screenplay unavailable for {view.episode_id}: {ep_issue.problem}.")

# --- verdict panel for candidates that did not pass -----------------------------------------------------
if view.lane != "eligible":
    body = [f"<p>{escape(ln.explanation)}</p>"]
    if dec and dec.hard_rule_failures:
        body.append(field("Failed rules", bullet_list(dec.hard_rule_failures)))
    if dec and view.lane == "quality_fail":
        failed = [f"{gate_label(g.gate)}: {g.detail}" for g in dec.gates if not g.passed]
        body.append(field("Gates not met", bullet_list(failed)))
    if dec and dec.repair_status:
        target = f" → {escape(view.repaired_to)}" if view.repaired_to else ""
        body.append(field("Repair", f"{escape(humanize(dec.repair_status))}{target}"
                                    f"<div class='pp-muted'>{text(dec.repair_reason)}</div>"))
    if view.lane in ("not_judged", "judge_invalid", "pending") and dec and dec.note:
        body.append(field("Recorded reason", text(data.redact(dec.note, 240))))
    if view.lane == "judge_invalid" and view.raw:
        body.append(field("Why the reply was unusable", bullet_list(view.raw.invalid_reasons)))
    if view.lane == "pending":
        body.append(f'<code class="pp-cmd">python -m promopilot.pipeline judge --run {escape(ctx.run.name)}</code>')
    st.html(f'<div class="pp-verdict pp-verdict--{ln.tone}">{"".join(body)}</div>')
    if view.repaired_to:
        st.page_link("views/candidate.py", label=f"Open the repaired version {view.repaired_to}",
                     icon=":material/build:", query_params={"cid": view.repaired_to, "run": ctx.run.name})

in_attention = audit.category(view) is not None or audit.review_status(view) != "none"
rstate = audit.repair_state(ctx, view)
repair_root = view.repaired_from or (view.candidate_id if rstate and rstate.key != "not_attempted" else None)
if in_attention or repair_root:
    with st.container(horizontal=True, key="pp-cand-links"):
        if in_attention:
            st.page_link(cards.ATTENTION_PAGE, label="View in Needs attention", icon=":material/flag:",
                         query_params={"focus": view.candidate_id, "run": ctx.run.name})
        if repair_root:
            st.page_link(cards.REPAIRS_PAGE, label="View repair record", icon=":material/build:",
                         query_params={"focus": repair_root, "run": ctx.run.name})

left, right = st.columns([1.45, 1], gap="large")

with left:
    if j is None:
        st.html('<div class="pp-sec pp-sec--muted"><p>No valid judgement exists for this candidate, so no '
                'scores or promotional analysis are shown.</p></div>')
    else:
        # 1-3. summary, stakes, archetypes
        arche = escape(humanize(dna.archetype.value)) if dna else "—"
        if dna and dna.archetype_other_label:
            arche += f" <span class='pp-muted'>({escape(dna.archetype_other_label)})</span>"
        secondary = humanize(dna.archetype_secondary) if dna and dna.archetype_secondary else None
        section(None, (f'<p class="pp-lead">{text(dna.appeal if dna else None)}</p>'
                       + '<div class="pp-grid2">'
                       + field("Narrative stakes", text(dna.stakes.value if dna else None)
                               + f'<div class="pp-chips">{ch(dna.stakes.evidence if dna else [])}</div>')
                       + field("Archetype", f"{arche}<div class='pp-muted'>Secondary: {text(secondary)}"
                                            f" · confidence {text(dna.archetype_confidence if dna else None)}</div>"
                                            f'<div class="pp-chips">{ch(dna.archetype.evidence if dna else [])}</div>')
                       + '</div>'), title="Promotional summary")

        # 4. hook
        h = dims["hook"]
        section("hook", field("Opening line", quote(h.block_id))
                + '<div class="pp-grid2">'
                + field("Mechanism", text(humanize(dna.hook_mechanism.value) if dna else None))
                + field("Starts", text(f"+{val.hook_offset_ms / 1000:.1f} s into the clip"
                                       if val and val.hook_offset_ms is not None else None))
                + '</div>' + field("Why", text(h.reason)) + field("Evidence", ch(h.evidence)))

        # 5. tension
        t = dims["narrative_tension"]
        beats = [b for b in t.escalation_beats if clip.block(b)]
        esc = ('<span class="pp-arrow">→</span>'.join(evidence.chip(b, clip) for b in beats) if beats
               else bullet_list(t.escalation_beats))
        section("narrative_tension", field("Escalation", esc) + field("How tension develops", text(t.reason))
                + field("Evidence", ch(t.evidence)))

        # 6. cliffhanger
        c = dims["cliffhanger"]
        outcome = OUTCOME_LABELS.get(c.outcome or "", humanize(c.outcome))
        tone = {"unresolved": "ok", "partly_resolved": "review", "resolved": "quality"}.get(c.outcome or "", "neutral")
        section("cliffhanger", field("Final suspense beat", quote(c.suspense_beat_id))
                + '<div class="pp-grid3">'
                + field("Mechanism", text(humanize(c.mechanism)))
                + field("Outcome at the cut", badge(outcome, tone))
                + field("Ends after beat", text(fmt_seconds(val.ending.trailing_after_suspense_ms)
                                                if val and val.ending else None))
                + '</div>'
                + field("Open question", text(c.open_question)) + field("Why", text(c.reason))
                + field("Evidence", ch(c.evidence)))

        # 7. context independence
        x = dims["context_independence"]
        confusions = "".join(f'<li>{escape(cf.get("issue", ""))} <span class="pp-chips">'
                             f'{ch(cf.get("block_ids", []))}</span></li>' for cf in x.confusions)
        section("context_independence",
                field("Blind summary (what a new viewer understands)", text(x.blind_summary))
                + '<div class="pp-grid2">'
                + field("Confusing references", f"<ul class='pp-list'>{confusions}</ul>" if confusions
                        else '<span class="pp-none">none listed</span>')
                + field("Missing context", bullet_list(dna.missing_context if dna else []))
                + '</div>'
                + field("Dependency on earlier episodes", text(humanize(dna.context_dependency) if dna else None))
                + field("Why", text(x.reason)) + field("Evidence", ch(x.evidence)))

        # 8. curiosity
        q = dims["curiosity_gap"]
        section("curiosity_gap", '<div class="pp-grid2">' + field("Reveals", bullet_list(q.revealed))
                + field("Withholds", bullet_list(q.withheld)) + '</div>'
                + field("Balance", text(humanize(q.balance))) + field("Why", text(q.reason))
                + field("Evidence", ch(q.evidence)))

        # 9. emotion
        m = dims["emotional_impact"]
        section("emotional_impact", '<div class="pp-grid2">' + field("Emotion", text(m.emotion))
                + field("Trigger", text(humanize(dna.emotional_trigger.value) if dna else None)
                        + f'<div class="pp-chips">{ch(dna.emotional_trigger.evidence if dna else [])}</div>')
                + '</div>' + field("Why", text(m.reason)) + field("Supporting evidence", ch(m.evidence)))

        # 10. Promo DNA
        if dna:
            rows = [("Archetype", humanize(dna.archetype.value), dna.archetype.evidence),
                    ("Hook mechanism", humanize(dna.hook_mechanism.value), dna.hook_mechanism.evidence),
                    ("Emotional trigger", humanize(dna.emotional_trigger.value), dna.emotional_trigger.evidence),
                    ("Stakes", dna.stakes.value, dna.stakes.evidence),
                    ("Cliffhanger mechanism", humanize(dna.cliffhanger_mechanism.value),
                     dna.cliffhanger_mechanism.evidence),
                    ("Context dependency", humanize(dna.context_dependency), []),
                    ("Curiosity balance", humanize(dna.curiosity_balance), []),
                    ("Primary appeal", dna.appeal, dna.supporting_block_ids)]
            body = "".join(f'<tr><th>{escape(k)}</th><td>{text(v)}</td><td>{ch(ev) if ev else ""}</td></tr>'
                           for k, v, ev in rows)
            section(None, f'<table class="pp-table pp-table--dna">{body}</table>', title="Promo DNA")

    # 11-12. checks
    if val:
        prov = (' <span class="pp-muted">(provisional: semantic beats not supplied by the judge)</span>'
                if val.provisional else "")
        tone = {"pass": "ok", "fail": "technical", "pending": "neutral", "skipped": "neutral"}
        rows = "".join(f'<tr><th>{escape(rule_label(r.rule))}</th><td>{badge(r.status.value, tone[r.status.value])}'
                       f'</td><td>{escape(r.message)}{" " + ch(r.block_ids) if r.block_ids else ""}'
                       f'{" " + badge("repairable", "accent") if r.repairable and r.status.value == "fail" else ""}'
                       f'</td></tr>' for r in val.checks)
        section(None, f'<p class="pp-muted">Validator {escape(val.validator_version)}{prov}. Python rules are '
                      f'authoritative; the judge cannot override them.</p><table class="pp-table">{rows}</table>',
                title="Technical validation")
    if dec and dec.gates:
        rows = "".join(f'<tr><th>{escape(gate_label(g.gate))}</th>'
                       f'<td>{badge("met" if g.passed else "not met", "ok" if g.passed else "quality")}</td>'
                       f'<td>{escape(g.detail)}</td></tr>' for g in dec.gates)
        soft = "".join(f'<li>{badge(r.severity, "quality" if r.severity == "major" else "neutral")} '
                       f'<b>{escape(humanize(r.code))}</b> {escape(r.explanation)} {ch(r.block_ids)}</li>'
                       for r in (j.soft_rejections if j else []))
        section(None, f'<table class="pp-table">{rows}</table>'
                + field("Soft rejection reasons (judge)", f"<ul class='pp-list'>{soft}</ul>" if soft
                        else '<span class="pp-none">none</span>'), title="Quality gates")

    # 13. reassessment
    rec = view.reassessment
    if rec and view.raw:
        status, accepted, rejected = view.reassessment_outcome()
        flags = "".join(f'<li><b>{escape(DIMENSION_LABELS.get(f.dimension, f.dimension))}</b> · '
                        f'{escape(f.code)}: {escape(f.detail)}</li>' for f in rec.flags)
        changes = []
        for name, orig in rec.original_scores.items():
            new = rec.revised_scores.get(name)
            verdict = ("accepted" if name in accepted else
                       next((f"rejected: {why}" for n, why in rejected if n == name), "not revised"))
            changes.append(f'<tr><th>{escape(DIMENSION_LABELS.get(name, name))}</th><td>{orig}</td>'
                           f'<td>{new if new is not None else "—"}</td>'
                           f'<td>{view.scores.get(name, "—")}</td><td>{escape(verdict)}</td></tr>')
        section(None, f'<p>{escape(REASSESSMENT_LABELS.get(status, humanize(status)))} '
                      f'<span class="pp-muted">(checker {escape(rec.checker_version)}, '
                      f'{escape(layout.local_time(rec.created_at))})</span></p>'
                + field("Contradictions flagged", f"<ul class='pp-list'>{flags}</ul>" if flags else "none")
                + ('<table class="pp-table"><tr><th></th><td><b>Original</b></td><td><b>Revised</b></td>'
                   f'<td><b>Effective</b></td><td><b>Outcome</b></td></tr>{"".join(changes)}</table>'
                   if changes else ""), title="Reassessment history")
    elif view.raw and view.raw.status == "valid":
        section(None, '<p class="pp-muted">No contradictions were flagged; the original judgement is the effective '
                      'one.</p>', title="Reassessment history")

    # generator suggestion
    if cand:
        section(None, '<p class="pp-muted">The generator proposed this clip with the notes below. The judge saw '
                      'only the clip itself, never these notes.</p>'
                + '<div class="pp-grid2">'
                + field("Proposed hook", f"{text(humanize(cand.hook_type))}<div class='pp-chips'>"
                                         f"{ch([cand.hook_block_id] if cand.hook_block_id else [])}</div>")
                + field("Proposed cliffhanger", f"<div class='pp-chips'>"
                        f"{ch([cand.cliffhanger_block_id] if cand.cliffhanger_block_id else [])}</div>")
                + '</div>'
                + field("Why it works (generator)", text(cand.why_it_works))
                + field("Unresolved question (generator)", text(cand.unresolved_question))
                + field("New-viewer context (generator)", text(cand.new_viewer_context)),
                title="Generator suggestion — not shown to the judge")

    # 14. provenance
    with st.expander("Provenance"):
        gen = ctx.generation.get(view.episode_id, {})
        rows = [("Generator", f'{gen.get("provider", "—")} · {gen.get("model", "—")}'),
                ("Generation prompt", f'{gen.get("prompt_version", "—")} · {str(gen.get("prompt_sha256", ""))[:12]}')]
        if view.raw:
            r = view.raw
            rows += [("Judge", f"{r.provider} · {r.model}"),
                     ("Rubric", f"{r.prompt_version} · {r.prompt_sha256[:12]}"),
                     ("Consistency checker", (ctx.judge.checker_version if ctx.judge else None) or "—"),
                     ("Judged at", layout.local_time(r.judged_at)),
                     ("Invocation", r.invocation_id or "—"),
                     ("Clip fingerprint", r.clip_sha256[:12]),
                     ("Masked label terms", str(r.masked_terms))]
        if val:
            rows.append(("Validator", val.validator_version))
        rows.append(("Judge folder", ctx.judge.name if ctx.judge else "—"))
        st.html('<table class="pp-table">' + "".join(f"<tr><th>{escape(k)}</th><td>{escape(str(v))}</td></tr>"
                                                    for k, v in rows) + "</table>")
    if view.raw and view.raw.raw_response_text:
        with st.expander("Raw judge reply"):
            st.code(view.raw.raw_response_text, language="json", wrap_lines=True)

with right:
    hook_id = dims["hook"].block_id if j else (cand.hook_block_id if cand else None)
    beat_id = dims["cliffhanger"].suspense_beat_id if j else (cand.cliffhanger_block_id if cand else None)
    with st.container(key="pp-script"):
        st.html(evidence.script_panel(clip, hook_id=hook_id, beat_id=beat_id, cited=evidence.citations(j),
                                      proposed=j is None))
