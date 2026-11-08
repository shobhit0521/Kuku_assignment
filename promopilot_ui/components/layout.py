"""App-wide state, cached loading and shared page furniture (header, badges, banners, empty states)."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

import streamlit as st

from promopilot_ui import data
from promopilot_ui.status import EPISODE_STATES

ASSETS = Path(__file__).resolve().parent.parent / "assets"
_CTX_KEY = "_pp_ctx"


# --- cached loading ---------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=6)
def _load_run(run_name: str, judge_name: Optional[str], root: str, fp: str) -> data.RunData:
    run = next(r for r in data.discover_runs(Path(root)) if r.name == run_name)
    judge = next((j for j in run.judges if j.name == judge_name), None)
    return data.load_run(run, judge, Path(root))


@st.cache_resource(show_spinner=False, max_entries=64)
def _load_episode(episode_id: str, root: str, sha: str):
    return data.load_episode(episode_id, Path(root))


def load_run(run: data.RunRef, root: Path) -> data.RunData:
    """Cached by a content fingerprint of every file the view reads, so any edit, addition or removal reloads."""
    judge = data.production_judge(run)
    return _load_run(run.name, judge.name if judge else None, str(root), data.run_fingerprint(run, judge, root))


def episode(ctx: data.RunData, episode_id: str):
    path = data.episode_path(episode_id, ctx.output_root)
    return _load_episode(episode_id, str(ctx.output_root), data.file_sha256(path))


def reload_all() -> None:
    _load_run.clear()
    _load_episode.clear()


def set_context(ctx: Optional[data.RunData]) -> None:
    st.session_state[_CTX_KEY] = ctx


def context() -> Optional[data.RunData]:
    return st.session_state.get(_CTX_KEY)


# --- formatting -------------------------------------------------------------------------------------

def local_time(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y, %H:%M")
    except ValueError:
        return iso


def css() -> None:
    st.html(f"<style>{(ASSETS / 'styles.css').read_text(encoding='utf-8')}</style>")


def badge(text: str, tone: str = "neutral", title: str = "") -> str:
    t = f' title="{escape(title)}"' if title else ""
    return f'<span class="pp-badge pp-badge--{tone}"{t}>{escape(text)}</span>'


def rubric_badge(ctx: data.RunData) -> str:
    j = ctx.judge
    if j is None:
        return badge("No judge output", "neutral")
    parts = [j.prompt_version or "unknown rubric", j.model or j.name]
    if j.checker_version:
        parts.append(f"checker {j.checker_version}")
    if ctx.rubric_is_production:
        return (f'<span class="pp-rubric" title="Production rubric. Scores are one LLM judge\'s opinion.">'
                f'<span class="pp-rubric__k">Rubric</span>{escape(" · ".join(parts))}</span>')
    return (f'<span class="pp-rubric pp-rubric--exp" title="Not the production rubric">'
            f'<span class="pp-rubric__k">Experimental</span>{escape(" · ".join(parts))}</span>')


def freshness(ctx: data.RunData) -> str:
    s = ctx.staleness
    if s.status == "current":
        pill = badge("Ranking current", "ok")
    elif s.status == "stale":
        pill = badge("Ranking out of date", "review", "; ".join(s.reasons))
    else:
        pill = badge("Not ranked", "neutral", "; ".join(s.reasons))
    return f'{pill}<div class="pp-side-note">Ranking generated {escape(local_time(s.generated_at))}</div>'


# --- page furniture ---------------------------------------------------------------------------------

def page_header(title: str, subtitle: str = "", ctx: Optional[data.RunData] = None, scores: bool = True) -> None:
    right = rubric_badge(ctx) if ctx is not None and scores else ""
    sub = f'<p class="pp-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    st.html(f'<header class="pp-header"><div><h1 class="pp-title">{escape(title)}</h1>{sub}</div>'
            f'<div class="pp-header__right">{right}</div></header>')
    if ctx is not None:
        banners(ctx)


def banners(ctx: data.RunData) -> None:
    if ctx.judge is not None and not ctx.rubric_is_production:
        banner("review", "Experimental rubric. These scores are not production results and are never mixed with "
                         "the production ranking.")
    if ctx.staleness.status == "stale":
        banner("review", "The ranking is older than the judge decisions: " + "; ".join(ctx.staleness.reasons)
               + ". Figures below are as of the last ranking. Re-run the `rank` command to refresh it.")
    if ctx.issues:
        banner("neutral", f"{len(ctx.issues)} file(s) could not be read and were skipped. "
                          "Details are listed in the sidebar under Data issues.")


def banner(tone: str, text: str) -> None:
    st.html(f'<div class="pp-banner pp-banner--{tone}">{escape(text)}</div>')


def empty_state(title: str, body: str, command: Optional[str] = None) -> None:
    cmd = f'<code class="pp-cmd">{escape(command)}</code>' if command else ""
    st.html(f'<div class="pp-empty"><div class="pp-empty__title">{escape(title)}</div>'
            f'<p>{escape(body)}</p>{cmd}</div>')


def section_title(text: str, aside: str = "") -> None:
    st.html(f'<div class="pp-section-title"><h2>{escape(text)}</h2><span>{aside}</span></div>')


def episode_state_badge(state: str) -> str:
    label, tone = EPISODE_STATES.get(state, (state, "neutral"))
    return badge(label, tone)


# --- sidebar ----------------------------------------------------------------------------------------

def _default_run(runs: list[data.RunRef]) -> data.RunRef:
    def ranked_at(r):
        j = data.production_judge(r)
        p = r.path / "ranking" / j.name / "ranking.json" if j else None
        return p.stat().st_mtime if p and p.exists() else 0.0
    return max(runs, key=ranked_at)


def select_run(runs: list[data.RunRef], sb=None) -> data.RunRef:
    names = [r.name for r in runs]
    wanted = st.query_params.get("run")
    default = wanted if wanted in names else _default_run(runs).name
    if len(names) == 1:
        choice = names[0]
    else:
        choice = (sb or st.sidebar).selectbox("Run", names, index=names.index(default),
                                              help="Folders under the configured output directory's runs/ folder.")
    if st.query_params.get("run") != choice:
        st.query_params["run"] = choice
    return runs[names.index(choice)]


def sidebar(ctx: data.RunData, sb=None) -> None:
    """Dataset, rubric and freshness details, written into `sb` (the sidebar by default)."""
    sb = sb or st.sidebar
    pack = ctx.pack or {}
    dataset = escape(pack.get("pack_name") or "Unknown dataset")
    eps = pack.get("episode_count")
    meta = f"{eps} episodes parsed · parser {escape(str(pack.get('parser_version', '?')))}" if eps else ""
    exp = [j for j in ctx.run.experimental_judges]
    exp_html = ""
    if exp:
        names = ", ".join(escape(j.prompt_version or j.name) for j in exp)
        exp_html = (f'<div class="pp-side-block"><div class="pp-side-k">Experimental outputs</div>'
                    f'<div class="pp-side-note">{names} on disk; not used on these pages.</div></div>')
    sb.html(f'<div class="pp-side-block"><div class="pp-side-k">Dataset</div><div class="pp-side-v">{dataset}</div>'
            f'<div class="pp-side-note">{meta}</div></div>'
            f'<div class="pp-side-block"><div class="pp-side-k">Production rubric</div>{rubric_badge(ctx)}</div>'
            f'<div class="pp-side-block"><div class="pp-side-k">Data freshness</div>{freshness(ctx)}'
            f'<div class="pp-side-note">Loaded {escape(local_time(ctx.loaded_at))} · '
            f'fingerprint {ctx.fingerprint[:10]}</div></div>{exp_html}')
    if sb.button("Reload data", icon=":material/refresh:", width="stretch",
                 help="Clear cached data and re-read every file."):
        reload_all()
        st.rerun()
    warn = data.output_location_warning(ctx.output_root)
    if warn:
        st.sidebar.warning(warn)
    if ctx.issues:
        with st.sidebar.expander(f"Data issues ({len(ctx.issues)})"):
            for i in ctx.issues:
                st.markdown(f"- `{i.path}`: {i.problem}")
