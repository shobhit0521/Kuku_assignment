"""PromoPilot editorial interface. Read-only view of persisted pipeline outputs; run with `python -m promopilot_ui`."""
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

pages = [
    st.Page("views/shortlist.py", title="Shortlist", icon=":material/format_list_numbered:", default=True),
    st.Page("views/candidate.py", title="Candidate intelligence", icon=":material/insights:", url_path="candidate"),
    st.Page("views/attention.py", title="Needs attention", icon=":material/flag:", url_path="attention"),
    st.Page("views/repairs.py", title="Repairs", icon=":material/build:", url_path="repairs"),
]
nav = st.navigation(pages, position="sidebar")

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
