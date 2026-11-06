"""Promos: exactly what the brief asks for. The ranked promos (episode, segment timestamps, hook, cliffhanger, why
it works, why this rank), then where the tool picks badly. Everything else is on the pages in the sidebar."""
from html import escape

import streamlit as st

from promopilot_ui.components import layout, promos

ctx = layout.context()
counts = (ctx.ranking or {}).get("counts", {})
n_eps = len(ctx.episodes)
subtitle = (f"{len(promos.rows(ctx))} promos to post, in order, chosen from {counts.get('total_candidates', 0)} "
            f"candidate clips across {n_eps} episodes" if ctx.ranking else "")
layout.page_header("Promos to post first", subtitle, ctx)

if ctx.judge is None or ctx.ranking is None:
    reason = "; ".join(ctx.staleness.reasons) or "no ranking available"
    layout.empty_state("No ranking for this run yet", f"This run cannot be shown as a shortlist: {reason}.",
                       f"python -m promopilot.pipeline rank --run {ctx.run.name}")
    st.stop()

st.html(f'<p class="pp-disclaimer">{escape(ctx.ranking.get("disclaimer", ""))} Scores, quality checks and every '
        'other candidate are on the pages in the sidebar.</p>')
layout.section_title("Ranked promos", "Post #1 first")
promos.promo_list(ctx)

layout.section_title("Where the tool picks badly")
promos.picks_badly(ctx)
