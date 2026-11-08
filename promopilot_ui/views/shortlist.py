"""Top promos: exactly what the brief asks for. The ranked promos (episode, segment timestamps, hook, cliffhanger,
why it works, why this rank), then where the tool picks badly. Everything else is on the pages in the sidebar."""
from html import escape

import streamlit as st

from promopilot_ui.components import layout, promos

ctx = layout.context()
n_eps = len(ctx.episodes)
shortlist = promos.rows(ctx)

if ctx.judge is None or ctx.ranking is None:
    layout.page_header("Top promos", "", ctx)
    reason = "; ".join(ctx.staleness.reasons) or "no ranking available"
    layout.empty_state("No ranking for this run yet", f"This run cannot be shown as a shortlist: {reason}.",
                       f"python -m promopilot.pipeline rank --run {ctx.run.name}")
    st.stop()

counts = ctx.ranking.get("counts", {})
rejected = sum(len(ctx.by_lane(k)) for k in ("technical_fail", "quality_fail", "not_judged", "judge_invalid"))
stats = [(n_eps, "episodes read"), (counts.get("total_candidates", len(ctx.candidates)), "candidate clips proposed"),
         (rejected, "rejected by the checks"), (len(shortlist), "promos ranked")]

def model_name(slug):
    return " ".join(p if p[:1].isdigit() else p.capitalize() for p in str(slug).split("-")) if slug else "unknown"


generator = model_name(ctx.manifest.get("model"))
judge = model_name(ctx.judge.model)
st.html('<section class="pp-hero">'
        '<h1 class="pp-hero__title">PromoPilot</h1>'
        '<p class="pp-subtitle">An AI-powered tool that reads screenplays, finds high-potential promo moments, and '
        'ranks them using structured quality checks.</p>'
        f'<p class="pp-lede">PromoPilot read all {n_eps} episodes and picked the 20–60 second clips most likely to '
        'stop a scrolling viewer: a hook in the first 3 seconds, a cliffhanger at the cut, and a story that makes '
        'sense to someone who has never seen the show.</p>'
        f'<p class="pp-models">Candidate clips are generated with {escape(generator)}, then evaluated separately by '
        f'{escape(judge)} against a fixed quality rubric, without seeing the generator\'s reasoning. Scores represent '
        'an AI model\'s assessment, not measured audience performance.</p>'
        '<div class="pp-stats">' + "".join(f'<div class="pp-stat"><b>{n}</b><span>{escape(label)}</span></div>'
                                           for n, label in stats) + '</div></section>')
layout.banners(ctx)
if st.button("Try it on your own screenplays", type="primary", icon=":material/upload_file:",
             help="Upload episode files and run the full pipeline on them."):
    st.switch_page("views/custom.py")

layout.section_title("Ranked promos", "Post #1 first")
promos.promo_list(ctx)

layout.section_title("Where the tool picks badly")
promos.picks_badly(ctx)
