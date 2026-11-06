"""PromoPilot editorial interface: the 50-episode dashboard (read-only view of persisted pipeline outputs) and the
upload page that runs the pipeline on a visitor's own screenplays. Run with `python -m promopilot_ui`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from promopilot_ui import data  # noqa: E402
from promopilot_ui.components import layout  # noqa: E402

st.set_page_config(page_title="PromoPilot", page_icon=str(layout.ASSETS / "mark.svg"), layout="wide",
                   initial_sidebar_state="expanded")
layout.css()
st.logo(str(layout.ASSETS / "logo.svg"), size="large", icon_image=str(layout.ASSETS / "mark.svg"))

dashboard = [st.Page("views/shortlist.py", title="Promos to post first", icon=":material/format_list_numbered:",
                     default=True)]
details = [
    st.Page("views/candidate.py", title="Full candidate analysis", icon=":material/insights:", url_path="candidate"),
    st.Page("views/attention.py", title="Where it picks badly", icon=":material/flag:", url_path="attention"),
    st.Page("views/scores.py", title="Scores and ranking rules", icon=":material/leaderboard:", url_path="scores"),
    st.Page("views/repairs.py", title="Repairs", icon=":material/build:", url_path="repairs"),
    st.Page("views/downloads.py", title="Downloads", icon=":material/download:", url_path="downloads"),
]
upload = st.Page("views/custom.py", title="Analyse your own screenplays", icon=":material/upload_file:",
                 url_path="analyse")
nav = st.navigation({"Kuku 50-episode analysis": dashboard, "Details": details,
                     "Try it on your screenplays": [upload]}, position="sidebar")

if nav.url_path == upload.url_path:
    # The upload page keeps its own run context and never touches the dashboard's.
    nav.run()
    st.stop()

if st.sidebar.button("Analyse your own screenplays", icon=":material/upload_file:", type="primary",
                     width="stretch", help="Upload episode files and run the full pipeline on them."):
    st.switch_page(upload)

root = data.output_root()
runs = data.discover_runs(root)
if not runs:
    layout.set_context(None)
    layout.page_header("PromoPilot")
    layout.empty_state("No runs found",
                       f"No pipeline runs were found under the configured output directory ({root.name}/runs). "
                       "Set PROMOPILOT_OUTPUT_DIR to the folder that holds the outputs, then reload.")
    warn = data.output_location_warning(root)
    if warn:
        layout.banner("review", warn)
    st.stop()

run = layout.select_run(runs)
ctx = layout.load_run(run, root)
layout.set_context(ctx)
layout.sidebar(ctx)
nav.run()
