"""Streamlit rendering of the shell, Shortlist and Candidate pages against fixture outputs (AppTest, no browser,
no network)."""
import hashlib
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import ui_fixtures as fx  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "promopilot_ui" / "app.py")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return fx.build(tmp_path_factory.mktemp("ui_pages"))


def html(at):
    return "\n".join(h.proto.body for h in at.get("html"))


def render(out, monkeypatch, page=None, cid=None):
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(out))
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"] = fx.RUN
    if cid:
        at.query_params["cid"] = cid
    at.run()
    if page:
        at.switch_page(page)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_shell_shows_dataset_rubric_and_freshness(built, monkeypatch):
    at = render(built, monkeypatch)
    side = "\n".join(h.proto.body for h in at.sidebar.get("html"))
    assert "Fixture pack" in side and "judge-v1" in side and "checker c1" in side
    assert "Ranking current" in side and "judge-v2 on disk; not used" in side


def test_default_page_shows_only_the_briefs_fields(built, monkeypatch):
    at = render(built, monkeypatch)
    page = "\n".join(b for b in (h.proto.body for h in at.get("html")) if not b.startswith("<style>"))
    assert "PromoPilot</h1>" in page and "An AI-powered tool that reads screenplays" in page
    assert "Ranked promos" in page and "pp-disclaimer" not in page.split("Where the tool picks badly", 1)[0]
    assert re.search(r"Candidate clips are generated with .+, then evaluated separately by .+ against a fixed "
                     r"quality rubric, without seeing the generator's reasoning\. Scores represent an AI model's "
                     r"assessment, not measured audience performance\.", page)
    assert re.search(r'<b>\d+</b><span>promos ranked</span>', page)
    assert "pp-rubric" not in page.split("Ranked promos", 1)[0]
    promo = page.split('<article class="pp-promo">', 1)[1].split("</article>", 1)[0]
    labels = re.findall(r"<dt>([^<]+)</dt>", promo)
    assert labels == ["Segment 1", "Hook", "Cliffhanger", "Why it works", "Why this rank"]
    assert "Episode 9" in promo and "Weighted score" in promo and " → " in promo
    assert "Where the tool picks badly" in page and "Broke a hard rule" in page and "Too weak to post" in page
    assert "EP009_C01" in page and "not from viewers" in page
    for detail in ("pp-dims", "pp-pips", "Also eligible", "How the shortlist is selected", "Technical validation"):
        assert detail not in page
    assert "Full analysis" in [p.proto.label for p in at.get("page_link")]


def test_detail_pages_are_in_the_sidebar(built, monkeypatch):
    at = render(built, monkeypatch, "views/downloads.py")
    assert {"shortlist.csv", "ranking.json"} <= {d.proto.label for d in at.get("download_button")}


def test_scores_page_lists_ranked_and_unselected_eligible(built, monkeypatch):
    page = html(render(built, monkeypatch, "views/scores.py"))
    assert "Scores and ranking rules" in page and "judge-v1" in page
    assert "EP009_C01_R1" in page and "Repaired from EP009_C01" in page
    assert "Also eligible, not selected" in page and "Near-duplicate" in page and "per episode" in page
    assert "Needs review" in page
    for absent in ("EP009_C04", "EP009_C05", "EP009_C06", "EP009_C07"):
        assert absent not in page
    assert re.search(r'class="pp-badge[^"]*"[^>]*>[^<]*reject', page, re.I) is None
    assert "Not rejected" in page


@pytest.mark.parametrize("cid", list(fx.EXPECTED_LANES))
def test_candidate_page_renders_every_state(built, monkeypatch, cid):
    page = html(render(built, monkeypatch, "views/candidate.py", cid))
    assert f'<span class="pp-cand-title">{cid}</span>' in page
    assert "pp-script" in page and 'id="blk-EP009_B' in page
    assert fx.FAKE_KEY not in page
    assert "Generator suggestion — not shown to the judge" in page


def test_candidate_without_valid_judgement_shows_no_scores(built, monkeypatch):
    for cid in ("EP009_C05", "EP009_C06", "EP009_C07"):
        page = body(render(built, monkeypatch, "views/candidate.py", cid))
        assert "No valid judgement exists" in page and "weighted score" not in page
        assert "Promotional summary" not in page and "pp-pips--lg" not in page
        assert "Generator hook (proposed)" in page


def test_candidate_page_shows_raw_and_effective_scores_and_evidence_links(built, monkeypatch):
    page = html(render(built, monkeypatch, "views/candidate.py", "EP009_C04"))
    assert "Original 3 → effective 2" in page and "Reassessment history" in page
    assert "Quality gate not met" in page and 'href="#blk-EP009_B' in page
    review = html(render(built, monkeypatch, "views/candidate.py", "EP009_C03"))
    assert "Revision rejected" in review and "Needs review" in review


def test_technical_failure_links_to_repaired_version(built, monkeypatch):
    at = render(built, monkeypatch, "views/candidate.py", "EP009_C01")
    page = html(at)
    assert "Technical failure" in page and "DIALOGUE_BOUNDARY" in page
    header = page.split('<span class="pp-cand-title">', 1)[1].split("</div>", 1)[0]
    assert header.count("Technical failure</span>") == 1
    assert any("EP009_C01_R1" in (p.proto.label or "") for p in at.get("page_link"))


def test_missing_ranking_shows_empty_state(built, monkeypatch, tmp_path):
    out = shutil.copytree(built, tmp_path / "outputs")
    (out / "runs" / fx.RUN / "ranking" / fx.V1 / "ranking.json").unlink()
    page = html(render(out, monkeypatch))
    assert "No ranking for this run yet" in page and "rank --run fixture" in page


def test_no_runs_shows_setup_message(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception and "No runs found" in html(at)


def test_no_legacy_pages_directory():
    # A pages/ folder beside app.py makes Streamlit run page files directly, skipping app.py, until "/" is loaded.
    assert not (Path(APP).parent / "pages").exists()


def test_rendering_never_modifies_source_files(built, monkeypatch):
    def hashes():
        return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(built.rglob("*")) if p.is_file()}
    before = hashes()
    render(built, monkeypatch)
    render(built, monkeypatch, "views/scores.py")
    render(built, monkeypatch, "views/downloads.py")
    render(built, monkeypatch, "views/attention.py")
    render(built, monkeypatch, "views/repairs.py")
    for cid in fx.EXPECTED_LANES:
        render(built, monkeypatch, "views/candidate.py", cid)
    assert hashes() == before


# --- Needs attention, Repairs and diagnostic scores ------------------------------------------------------

def body(at):
    """Rendered HTML without the injected stylesheet."""
    return "\n".join(b for b in (h.proto.body for h in at.get("html")) if not b.startswith("<style>"))


def section_html(page, key):
    start = page.index(f'id="pp-att-{key}"')
    end = page.find("</section>", start)
    return page[start:end]


def test_attention_page_separates_failure_types(built, monkeypatch):
    page = body(render(built, monkeypatch, "views/attention.py"))
    tech, qual = section_html(page, "technical"), section_html(page, "quality")
    unus, pend, rev = section_html(page, "unusable"), section_html(page, "pending"), section_html(page, "review")
    assert ">EP009_C01<" in tech and ">EP009_C08<" in tech and "10.0 s" in tech and "20 s to 60 s" in tech
    assert "Repaired; every rule passes after revalidation" in tech and "Repair attempted; not possible" in tech
    assert ">EP009_C04<" in qual and 'pp-below">2<' in qual and "An emotional button" in qual
    assert 'class="pp-diag"' in qual
    for cid in ("EP009_C05", "EP009_C06"):
        assert f">{cid}<" in unus and f">{cid}<" not in qual
    assert "pp-diag" not in unus and "Provider request failed" in unus and "Malformed judge reply" in unus
    assert ">EP009_C07<" in pend and "EP010" in pend and "Generation pending" in pend
    assert ">EP009_C03<" in rev and "Eligible, not shortlisted" in rev and "1 open · 1 resolved" in rev
    assert ">EP009_C03<" not in tech + qual + unus + pend
    assert fx.FAKE_KEY not in page


def test_attention_filters_and_focus(built, monkeypatch):
    at = render(built, monkeypatch, "views/attention.py")
    next(m for m in at.multiselect if m.label == "Category").set_value(["technical"]).run()
    page = body(at)
    assert 'id="pp-att-technical"' in page and 'id="pp-att-quality"' not in page
    monkeypatch.setenv("PROMOPILOT_OUTPUT_DIR", str(built))
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["run"], at.query_params["focus"] = fx.RUN, "EP009_C04"
    at.run()
    at.switch_page("views/attention.py")
    at.run()
    page = body(at)
    assert "Showing entries for <b>EP009_C04</b> only" in page
    assert ">EP009_C04<" in page and ">EP009_C01<" not in page and ">EP009_C05<" not in page


def test_repairs_page_compares_original_and_repaired(built, monkeypatch):
    page = body(render(built, monkeypatch, "views/repairs.py"))
    assert re.search(r"<b>2</b><span>Repair attempts", page) and re.search(r"<b>1</b><span>Became eligible", page)
    assert ">EP009_C01_R1<" in page and "+1.0 s" in page and "Expand end for dialogue" in page
    assert "New blind judgement" in page and "Repair attempted; not possible" in page
    assert "no conservative boundary change is available" in page and "One attempt per candidate" in page
    assert 'href="attention?run=fixture&amp;focus=EP009_C01"' in page


def test_failed_candidates_show_a_muted_diagnostic_score(built, monkeypatch):
    for cid in ("EP009_C01", "EP009_C04", "EP009_C08"):
        page = body(render(built, monkeypatch, "views/candidate.py", cid))
        assert "pp-bigscore--diag" in page and "Diagnostic score — not ranked" in page
        assert "<span>weighted score</span>" not in page
    eligible = body(render(built, monkeypatch, "views/candidate.py", "EP009_C01_R1"))
    assert "<span>weighted score</span>" in eligible and "pp-bigscore--diag" not in eligible
    none = body(render(built, monkeypatch, "views/candidate.py", "EP009_C05"))
    assert "<span>no score</span>" in none and "pp-diag" not in none
    shortlist = body(render(built, monkeypatch))
    assert "pp-diag" not in shortlist


def test_candidate_page_links_to_attention_and_repair_record(built, monkeypatch):
    at = render(built, monkeypatch, "views/candidate.py", "EP009_C01")
    labels = {p.proto.label: p.proto for p in at.get("page_link")}
    assert {"View in Where it picks badly", "View repair record"} <= set(labels)
    at = render(built, monkeypatch, "views/candidate.py", "EP009_C02")
    assert not {"View in Where it picks badly", "View repair record"} & {p.proto.label for p in at.get("page_link")}
