"""Full candidate analysis: why a clip works (or why it did not pass), linked to the screenplay lines."""
import streamlit as st

from promopilot_ui import data
from promopilot_ui.components import candidate_detail, layout
from promopilot_ui.status import lane

ctx = layout.context()
layout.page_header("Full candidate analysis", "", ctx)
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
candidate_detail.render(ctx, view)
