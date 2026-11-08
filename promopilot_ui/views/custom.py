"""Analyse your own screenplays: upload episode files, run the full pipeline on them (real Gemini calls), and see
the ranked results on this page. Each upload is an isolated temporary job; the 50-episode outputs are never read
or written here."""
from datetime import datetime, timezone
from html import escape

import streamlit as st

from promopilot import config, live
from promopilot_ui import data
from promopilot_ui.components import candidate_detail, cards, layout, promos
from promopilot_ui.components.layout import badge
from promopilot_ui.status import lane, selection_note

JOB_KEY = "pp_live_job"
RESULT_VIEWS = ("Top promos", "Full candidate analysis", "Where it picks badly", "Scores and ranking",
                "Processing log", "Downloads")
SETTINGS = config.live_settings()
RUNNING = ("queued", "starting", "running")
FORMAT_EXAMPLE = """EPISODE NUMBER: Episode 1
EPISODE LOGLINE: One sentence about the episode.
CONTINUITY NOTES: none

--- SCRIPT START ---

SCENE 01
Scene heading: INT. KITCHEN – NIGHT
Scene start: 00:00.00
Scene end: 00:45.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:06.00)
Close on a phone ringing on the table.

**MAYA (Whispering):**
Start: (00:01.20) | End: (00:03.00)
"Don't answer it."

**SOUND FX:**
Start: (00:03.00) | End: (00:04.00)
The ringing stops.

--- SCRIPT END ---"""

live.sweep()


def current_job():
    jid = st.session_state.get(JOB_KEY) or st.query_params.get("job")
    return jid if jid and live.JOB_ID_RE.match(jid) else None


def set_job(jid):
    if jid:
        st.session_state[JOB_KEY] = jid
        st.query_params["job"] = jid
    else:
        st.session_state.pop(JOB_KEY, None)
        st.query_params.pop("job", None)


def start_over(jid):
    live.delete_job(jid)
    set_job(None)
    for key in ("pp_live_view", "pp_live_cid"):
        st.session_state.pop(key, None)
    st.rerun()


@st.cache_data(show_spinner=False, max_entries=64)
def precheck(name: str, raw: bytes) -> dict:
    return live.precheck(name, raw, SETTINGS)


# --- upload --------------------------------------------------------------------------------------------

def upload_screen(notice=None):
    layout.page_header("Analyse your own screenplays",
                       "Upload one or more episode files. PromoPilot runs the same pipeline used for the "
                       "50-episode analysis and ranks the best promo clips.")
    if notice:
        layout.banner("review", notice)
    st.html(
        '<div class="pp-banner pp-banner--neutral">How it works: each file is parsed into timed blocks, Gemini '
        'proposes candidate clips, Python checks every hard rule (20–60 s, hook in the first 3 s, no cut '
        'mid-line, a suspense beat at the end), a separate blind Gemini judge scores each clip, and the best clips '
        'are ranked. The screenplay text is sent to Google Gemini for this. Scores are one AI judge\'s opinion, '
        'not ground truth. Analysis takes several minutes per file.</div>')
    with st.expander("Expected file format"):
        st.markdown("Plain-text `.txt` files in the same structure as the PromoPilot episode pack: a header with "
                    "the episode number, then scenes, then blocks. Each block is a `**LABEL:**` line (`CAMERA`, "
                    "`ACTION`, `SOUND FX`, or a speaker name with optional delivery notes), a "
                    "`Start: (mm:ss.cc) | End: (mm:ss.cc)` line and the text.")
        st.code(FORMAT_EXAMPLE, language=None)
    st.caption(f"Limits: up to {SETTINGS['max_files']} files per analysis, {SETTINGS['max_file_kb']} KB each. "
               "Uploads are temporary and deleted afterwards.")

    files = st.file_uploader("Episode files", type=["txt"], accept_multiple_files=True,
                             help="Drag and drop one or more .txt episode files, or browse.")
    if not files:
        layout.empty_state("No files yet", "Drop one or more .txt episode files above to check them.")
        return
    raws = [(f.name, f.getvalue()) for f in files]
    rows = [precheck(n, r) for n, r in raws]
    ok = [(n, r) for (n, r), row in zip(raws, rows) if row["ok"]]
    dupes = live.duplicate_episodes(rows)

    st.dataframe(rows, hide_index=True, width="stretch",
                 column_order=("file", "ok", "episode_id", "scenes", "blocks", "duration", "problem"),
                 column_config={"file": "File", "ok": st.column_config.CheckboxColumn("Ready"),
                                "episode_id": "Episode", "scenes": "Scenes", "blocks": "Blocks",
                                "duration": "Duration",
                                "problem": st.column_config.TextColumn("Notes", width="large")})
    problems = []
    if dupes:
        problems.append("Two or more files have the same episode number: "
                        + "; ".join(f"{eid} in {', '.join(n)}" for eid, n in dupes.items())
                        + ". Remove the duplicates or renumber them.")
    if len(ok) > SETTINGS["max_files"]:
        problems.append(f"Too many files: {len(ok)} ready, the limit is {SETTINGS['max_files']}.")
    if not ok:
        problems.append("None of the files can be analysed yet; see the notes above.")
    blocker = None if problems else live.start_blocker(len(ok))
    for p in problems + ([blocker] if blocker else []):
        layout.banner("review", p)
    skipped = len(rows) - len(ok)
    if ok and skipped and not problems:
        st.caption(f"{skipped} file(s) with problems will be skipped; {len(ok)} will be analysed.")
    budget = live.budgets(len(ok))
    if st.button(f"Start analysis of {len(ok)} file(s)", type="primary", icon=":material/play_arrow:",
                 disabled=bool(problems or blocker),
                 help=f"Uses up to {budget['generate']} generation and {budget['judge']} judge requests."):
        try:
            jid = live.create_job(ok)
            live.start_job(jid)
        except live.JobError as exc:
            layout.banner("review", str(exc))
            return
        set_job(jid)
        st.rerun()


# --- processing ----------------------------------------------------------------------------------------

STAGE_TONES = {"pending": ("Waiting", "neutral"), "running": ("In progress", "accent"), "done": ("Done", "ok"),
               "partial": ("Stopped early", "review"), "failed": ("Failed", "technical")}


def _elapsed(start, finish=None):
    if not start:
        return ""
    try:
        a = datetime.fromisoformat(start)
        b = datetime.fromisoformat(finish) if finish else datetime.now(timezone.utc)
    except ValueError:
        return ""
    secs = int((b - a).total_seconds())
    return f"{secs // 60} min {secs % 60:02d} s" if secs >= 60 else f"{secs} s"


def stage_detail(stage, info, status):
    d = info.get("detail") or {}
    if not d:
        return ""
    if stage == "parse":
        return f"{d.get('episodes', 0)} episode(s), {d.get('scenes', 0)} scenes, {d.get('blocks', 0)} blocks"
    if stage == "generate":
        out = (f"{d.get('candidates', 0)} candidate clip(s) from {d.get('episodes_complete', 0)} of "
               f"{d.get('episodes', 0)} episode(s) · {status['requests']['generate']} of "
               f"{status['budget']['generate']} requests")
        return out + (f" · {d['stopped_early']}" if d.get("stopped_early") else "")
    if stage == "validate":
        parts = ", ".join(f"{v} {k}" for k, v in sorted((d.get("statuses") or {}).items()))
        return f"{d.get('checked', 0)} candidate(s) checked" + (f": {parts}" if parts else "")
    if stage == "judge":
        parts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted((d.get("verdicts") or {}).items()))
        out = (f"{d.get('decisions', 0)} decision(s)" + (f": {parts}" if parts else "")
               + f" · {status['requests']['judge']} of {status['budget']['judge']} requests")
        return out + (f" · {d['stopped_early']}" if d.get("stopped_early") else "")
    return f"{d.get('shortlisted', 0)} clip(s) shortlisted from {d.get('eligible', 0)} eligible"


def stages_html(status):
    items = []
    for stage in live.STAGES:
        info = status["stages"].get(stage, {})
        label, tone = STAGE_TONES.get(info.get("state", "pending"), ("Waiting", "neutral"))
        time = _elapsed(info.get("started_at"), info.get("finished_at")) if info.get("started_at") else ""
        detail = stage_detail(stage, info, status)
        msg = info.get("message") if info.get("state") in ("partial", "failed") else ""
        items.append(f'<li class="pp-stage"><div class="pp-stage__head">{badge(label, tone)} '
                     f'<b>{escape(live.STAGE_LABELS[stage])}</b>'
                     f'<span class="pp-muted"> {escape(time)}</span></div>'
                     + (f'<div class="pp-muted">{escape(detail)}</div>' if detail else "")
                     + (f'<pre class="pp-stage__msg">{escape(msg)}</pre>' if msg else "") + '</li>')
    return f'<ol class="pp-stages">{"".join(items)}</ol>'


@st.fragment(run_every=3)
def progress(jid):
    status = live.job_state(jid)
    if status is None or status.get("state") not in RUNNING:
        st.rerun()
    st.html(f'<p class="pp-muted">{len(status["files"])} file(s): {escape(", ".join(status["files"]))} · '
            f'running for {escape(_elapsed(status.get("started_at") or status.get("created_at")))}. Each stage is '
            'marked done only when its output has been written.</p>' + stages_html(status))


def processing_screen(jid):
    layout.page_header("Analysing your screenplays", "The full pipeline is running. You can leave this page open; "
                                                     "it updates by itself.")
    progress(jid)
    if st.button("Cancel analysis", icon=":material/close:"):
        start_over(jid)


# --- results -------------------------------------------------------------------------------------------

def results_screen(jid, status):
    job = live.job_dir(jid)
    runs = data.discover_runs(job)
    ctx = layout.load_run(runs[0], job) if runs else None
    title = "Your results"
    if ctx is None or ctx.ranking is None:
        layout.page_header(title)
        layout.banner("review", status.get("error") or "The analysis finished without a ranking.")
        st.html(stages_html(status))
        if st.button("Start over", type="primary", icon=":material/restart_alt:"):
            start_over(jid)
        return

    shortlist = ctx.shortlist
    section = st.sidebar.radio("Your results", RESULT_VIEWS, key="pp_live_view",
                               help="The promos come first; everything else is here if you want it.")
    counts = ctx.ranking.get("counts", {})
    layout.page_header(title if section == RESULT_VIEWS[0] else section,
                       f"{len(shortlist)} promo(s) to post, in order, chosen from {counts.get('total_candidates', 0)} "
                       f"candidate clip(s) across {len(status['files'])} uploaded file(s).", ctx)
    if status.get("state") == "partial":
        layout.banner("review", "Some stages stopped early (request limit or provider error), so not every "
                                "candidate was generated or judged. The results below cover what was completed.")

    if section == RESULT_VIEWS[0]:
        st.html(f'<p class="pp-disclaimer">{escape(ctx.ranking.get("disclaimer", ""))} Scores, quality checks and '
                'every other candidate are in the sidebar under "Your results".</p>')
        layout.section_title("Ranked promos", "Post #1 first")
        promos.promo_list(ctx, link=False)
        layout.section_title("Where the tool picks badly")
        promos.picks_badly(ctx, link=False)
    elif section == RESULT_VIEWS[1]:
        views = ctx.candidates
        if not views:
            layout.empty_state("No candidates", "The generator proposed no candidate clips for these files.")
        else:
            ids = list(views)
            wanted = st.session_state.get("pp_live_cid")
            default = wanted if wanted in views else (shortlist[0].candidate_id if shortlist else ids[0])

            def option_label(cid):
                v = views[cid]
                rank = f" · promo #{v.shortlist_rank}" if v.ranking_status == "selected" else ""
                return f"{cid} · {lane(v.lane).label}{rank}"

            choice = st.selectbox("Candidate", ids, index=ids.index(default), format_func=option_label)
            st.session_state["pp_live_cid"] = choice
            candidate_detail.render(ctx, views[choice], links=False)
    elif section == RESULT_VIEWS[2]:
        promos.picks_badly(ctx, examples=None, link=False)
    elif section == RESULT_VIEWS[3]:
        layout.section_title("Shortlist with scores", "Ranked by weighted score")
        for row in [shortlist[i:i + 2] for i in range(0, len(shortlist), 2)]:
            cols = st.columns(2, gap="medium")
            for col, view in zip(cols, row):
                with col:
                    cards.shortlist_card(view, ctx, link=False)
        others = ctx.eligible_not_selected
        if others:
            layout.section_title("Also eligible, not selected", "Passed every gate; left out by the diversity rules")
            for view in others:
                st.html(f'<p><span class="pp-cid">{escape(view.candidate_id)}</span> '
                        f'{escape(selection_note(view.ordered_row, ctx.ranking) or "")}</p>')
    elif section == RESULT_VIEWS[4]:
        st.html(stages_html(status))
    else:
        promos.downloads(ctx)

    st.caption("Your files and results are deleted when you start over, or automatically after "
               f"{SETTINGS['job_ttl_min']} minutes.")
    if st.button("Start over with new files", type="primary", icon=":material/restart_alt:"):
        start_over(jid)


# --- page ----------------------------------------------------------------------------------------------

st.sidebar.page_link("views/shortlist.py", label="Back to the 50-episode analysis", icon=":material/arrow_back:")
jid = current_job()
status = live.job_state(jid) if jid else None
if jid and status is None:
    set_job(None)
    upload_screen("That analysis is no longer available: it was deleted, expired, or the server restarted. "
                  "Please upload your files again.")
elif status is None:
    upload_screen()
elif status.get("state") in RUNNING:
    processing_screen(jid)
else:
    results_screen(jid, status)
