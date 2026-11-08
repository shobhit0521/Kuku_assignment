"""The upload page in each state (empty, processing, results, expired) and the dashboard's link to it. AppTest only;
jobs are run in-process with the fake provider from test_live."""
import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_fixtures as fx  # noqa: E402
from promopilot import live  # noqa: E402
from test_live import SETTINGS, env, upload  # noqa: E402,F401

APP = str(Path(__file__).resolve().parents[1] / "promopilot_ui" / "app.py")
PAGE = "views/custom.py"


def html(at):
    return "\n".join(b for b in (h.proto.body for h in at.get("html")) if not b.startswith("<style>"))


def open_page(job=None):
    at = AppTest.from_file(APP, default_timeout=60)
    if job:
        at.query_params["job"] = job
    at.run()
    at.switch_page(PAGE)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_empty_upload_page_explains_the_format_and_limits(env):
    at = open_page()
    page = html(at)
    assert "Analyse your own screenplays" in page and "No files yet" in page
    assert "sent to Google Gemini" in page
    assert any("up to 5 files per analysis, 100 KB each" in c.value for c in at.caption)
    assert "No runs found" not in page


def test_processing_page_shows_real_stage_states(env):
    jid = live.create_job(upload(1), settings=SETTINGS)
    status = live.read_status(live.job_dir(jid))
    status.update(state="running", pid=os.getpid(), started_at=status["created_at"])
    status["stages"]["parse"].update(state="done", detail={"episodes": 1, "scenes": 1, "blocks": 12})
    status["stages"]["generate"].update(state="running", started_at=status["created_at"])
    live._write_json(live.status_path(live.job_dir(jid)), status)
    page = html(open_page(jid))
    assert "Analysing your screenplays" in page
    assert "1 episode(s), 1 scenes, 12 blocks" in page
    assert page.count(">Done<") == 1 and page.count(">In progress<") == 1 and page.count(">Waiting<") == 3


def test_results_page_shows_ranked_clips_and_full_analysis_without_dashboard_links(env):
    jid = live.create_job(upload(1, 2), settings=SETTINGS)
    live.run_job(live.job_dir(jid))
    at = open_page(jid)
    page = html(at)
    assert "Your results" in page and "Ranked promos" in page and "Where the tool picks badly" in page
    assert '<article class="pp-promo">' in page and "pp-cand-title" not in page
    assert "Start over with new files" in [b.label for b in at.button]
    radio = at.sidebar.radio[0]
    assert list(radio.options) == list(custom_views())
    radio.set_value("Full candidate analysis").run()
    detail = html(at)
    assert "pp-cand-title" in detail and "pp-script" in detail and "Technical validation" in detail
    at.sidebar.radio[0].set_value("Downloads").run()
    assert not any(p.proto.label in ("Open analysis", "Full analysis", "View in Where it picks badly",
                                     "Every failure, with evidence") for p in at.get("page_link"))
    assert [d.proto.label for d in at.get("download_button")]


def custom_views():
    return ("Top promos", "Full candidate analysis", "Where it picks badly", "Scores and ranking",
            "Processing log", "Downloads")


def test_expired_job_returns_to_the_upload_screen(env):
    page = html(open_page("0123456789ab"))
    assert "no longer available" in page and "No files yet" in page


def test_dashboard_offers_the_upload_page(built_fixture, monkeypatch):
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(built_fixture))
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"] = fx.RUN
    at.run()
    assert not at.exception
    assert "Try it on your own screenplays" in [b.label for b in at.main.button]
    assert "Analyse your own screenplays" not in [b.label for b in at.sidebar.button]
    next(b for b in at.main.button if b.label == "Try it on your own screenplays").click().run()
    assert not at.exception and "No files yet" in html(at)


@pytest.fixture(scope="module")
def built_fixture(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("ui_upload"))
