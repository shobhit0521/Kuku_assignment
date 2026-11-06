"""Scores and ranking rules: the shortlist with every score, the eligible candidates the ranking did not select,
and how the shortlist is selected."""
from html import escape

import streamlit as st

from promopilot_ui.components import cards, layout
from promopilot_ui.status import DIMENSION_LABELS, humanize, selection_note

ctx = layout.context()
counts = (ctx.ranking or {}).get("counts", {})
n_eps = sum(1 for e in ctx.episodes.values() if e.candidate_ids)
subtitle = (f"{len(ctx.shortlist)} clips selected from {counts.get('passing_quality_gates', 0)} eligible candidates "
            f"across {n_eps} episodes with candidates" if ctx.ranking else "")
layout.page_header("Scores and ranking rules", subtitle, ctx)

if ctx.judge is None or ctx.ranking is None:
    reason = "; ".join(ctx.staleness.reasons) or "no ranking available"
    layout.empty_state("No ranking for this run yet", f"This run cannot be shown as a shortlist: {reason}.",
                       f"python -m promopilot.pipeline rank --run {ctx.run.name}")
    st.stop()

st.html(f'<p class="pp-disclaimer">{escape(ctx.ranking.get("disclaimer", ""))}</p>')

pool = ctx.shortlist + ctx.eligible_not_selected
f1, f2, f3 = st.columns([2, 2, 1.2], vertical_alignment="bottom")
episodes = f1.multiselect("Episode", sorted({v.episode_id for v in pool}), placeholder="All episodes")
archetypes = f2.multiselect("Archetype", sorted({v.archetype or "unclassified" for v in pool}),
                            format_func=humanize, placeholder="All archetypes")
repaired_only = f3.toggle("Repaired only", help="Only candidates whose boundaries were repaired before judging.")


def keep(v):
    return ((not episodes or v.episode_id in episodes) and (not archetypes or (v.archetype or "unclassified")
            in archetypes) and (not repaired_only or v.repaired_from))


shown = [v for v in ctx.shortlist if keep(v)]
hidden = len(ctx.shortlist) - len(shown)
layout.section_title("Shortlist", f"Ranked by weighted score · {hidden} hidden by filters" if hidden else
                     "Ranked by weighted score")
if not ctx.shortlist:
    layout.empty_state("No clip reached the shortlist", "No candidate in this run passed every quality gate.")
elif not shown:
    layout.empty_state("No shortlisted clip matches these filters", "Clear a filter to see the shortlist.")
else:
    for row in [shown[i:i + 2] for i in range(0, len(shown), 2)]:
        cols = st.columns(2, gap="medium")
        for col, view in zip(cols, row):
            with col:
                cards.shortlist_card(view, ctx)

others = [v for v in ctx.eligible_not_selected if keep(v)]
layout.section_title("Also eligible, not selected",
                     "Passed every quality gate; left out by the shortlist's diversity rules. Not rejected.")
if not ctx.eligible_not_selected:
    st.html('<p class="pp-none">Every eligible candidate is on the shortlist.</p>')
elif not others:
    st.html('<p class="pp-none">No eligible-but-unselected candidate matches these filters.</p>')
for view in others:
    cards.compact_row(view, ctx, selection_note(view.ordered_row, ctx.ranking) or "")

settings = ctx.ranking.get("settings", {})
sens = ctx.ranking.get("sensitivity") or {}
with st.expander("How the shortlist is selected"):
    weights = settings.get("weights", {})
    gates = settings.get("gate_minimums", {})
    w = " · ".join(f"{DIMENSION_LABELS.get(k, k)} {v:.0%}" for k, v in weights.items())
    g = " · ".join(f"{DIMENSION_LABELS.get(k, k)} ≥ {v}" for k, v in gates.items())
    same = sens.get("shortlist_approved") == sens.get("shortlist_equal_weight")
    rho = sens.get("spearman_rho")
    st.html(
        '<dl class="pp-dl">'
        f'<dt>Weighted score</dt><dd>{escape(w)}</dd>'
        f'<dt>Quality gates</dt><dd>{escape(g)}; hook inside the opening window; suspense beat inside the clip; '
        'ending not resolved; every deterministic rule passes.</dd>'
        f'<dt>Diversity</dt><dd>At most {settings.get("max_per_episode", 1)} clip per episode and '
        f'{settings.get("max_per_archetype", "?")} per archetype. Clips overlapping a higher-ranked clip from the '
        f'same episode by more than {settings.get("overlap_limit", 0):.0%} of the shorter clip are treated as '
        'near-duplicates.</dd>'
        f'<dt>Sensitivity</dt><dd>With equal weights the shortlist is '
        f'{"unchanged" if same else "different"}'
        + (f"; rank correlation {rho:.2f}" if isinstance(rho, (int, float)) else "") + '.</dd>'
        '</dl>')
