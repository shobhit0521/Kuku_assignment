"""Downloads: the ranking files the pipeline wrote for this run, unedited."""
import streamlit as st

from promopilot_ui.components import layout, promos

ctx = layout.context()
layout.page_header("Downloads", "The files the ranking step wrote for this run, exactly as written.", ctx,
                   scores=False)
if ctx.judge is None or ctx.ranking is None:
    layout.empty_state("No ranking for this run yet", "There are no ranking files to download.")
    st.stop()
promos.downloads(ctx)
st.caption("shortlist.csv: the ranked promos · candidates.csv: every candidate and its outcome · catalogue.csv: "
           "every eligible clip in score order · report.md: the run report · ranking.json: all of the above.")
