"""Complete ranked catalogue and deliverable export fields (hook, suspense beat, explanation). Synthetic data only."""
import csv
import io
import json

from promopilot import ranking
from promopilot.ranking import build_report, deliverable, rank, write_report
from promopilot.schemas import EpisodeGeneration
from test_judge import candidate, default_reply, episode, judgement_from
from test_ranking import dec


def catalogue_mix():
    """Two episodes with several eligible clips each, one near-duplicate, plus ineligible clips."""
    return [dec("EP001_C01", 0, 30, hook=5), dec("EP001_C02", 5, 30, hook=4),          # C02 duplicates C01
            dec("EP001_C03", 100, 130, hook=3), dec("EP002_C01", 0, 30, hook=4, arche="reversal"),
            dec("EP002_C02", 200, 230, hook=2, cliff=4), dec("EP003_C01", 0, 30, hook=4),
            dec("EP003_C02", 50, 80, verdict="quality_fail"), dec("EP003_C03", 90, 120, verdict="not_judged"),
            dec("EP004_C01", 0, 30, verdict="hard_fail", hard="fail"),
            dec("EP004_C02", 40, 70, verdict="judge_invalid")]


ELIGIBLE = {"EP001_C01", "EP001_C02", "EP001_C03", "EP002_C01", "EP002_C02", "EP003_C01"}


# --- Fix A: complete catalogue ---------------------------------------------------------------------

def test_all_eligible_candidates_appear_exactly_once_with_sequential_ranks():
    r = rank(catalogue_mix())
    ids = [x["candidate_id"] for x in r["ordered"]]
    assert set(ids) == ELIGIBLE and len(ids) == len(set(ids))
    assert [x["catalogue_rank"] for x in r["ordered"]] == list(range(1, len(ELIGIBLE) + 1))


def test_ineligible_candidates_never_enter_the_catalogue():
    ids = {x["candidate_id"] for x in rank(catalogue_mix())["ordered"]}
    assert not ids & {"EP003_C02", "EP003_C03", "EP004_C01", "EP004_C02"}


def test_catalogue_order_is_score_then_tie_breaks_and_ignores_caps():
    r = rank(catalogue_mix(), max_per_archetype=1)
    scores = [x["score"] for x in r["ordered"]]
    assert scores == sorted(scores, reverse=True)
    assert len(r["ordered"]) == len(ELIGIBLE)
    loose = rank(catalogue_mix(), max_per_archetype=99)
    assert [x["candidate_id"] for x in loose["ordered"]] == [x["candidate_id"] for x in r["ordered"]]
    assert {x["status"] for x in r["ordered"]} >= {"not_selected_episode_cap", "not_selected_archetype_cap"}


def test_near_duplicates_keep_a_rank_and_name_the_clip_they_overlap():
    row = next(x for x in rank(catalogue_mix())["ordered"] if x["candidate_id"] == "EP001_C02")
    assert row["status"] == "suppressed_duplicate" and row["catalogue_rank"] is not None
    assert row["suppressed_by"] == "EP001_C01" and row["overlap"] == 1.0


def test_shortlist_rows_reference_their_catalogue_rank():
    r = rank(catalogue_mix())
    by_id = {x["candidate_id"]: x["catalogue_rank"] for x in r["ordered"]}
    assert r["shortlist"]
    for s in r["shortlist"]:
        assert s["catalogue_rank"] == by_id[s["candidate_id"]]
    assert [s["catalogue_rank"] for s in r["shortlist"]] == sorted(s["catalogue_rank"] for s in r["shortlist"])


def test_shortlist_selection_is_unchanged_by_the_catalogue():
    r = rank(catalogue_mix())
    assert [(s["rank"], s["candidate_id"]) for s in r["shortlist"]] == \
        [(1, "EP001_C01"), (2, "EP002_C01"), (3, "EP003_C01")]
    status = {x["candidate_id"]: x["status"] for x in r["ordered"]}
    assert status["EP001_C03"] == status["EP002_C02"] == "not_selected_episode_cap"


def write_judge_dir(tmp_path, decisions):
    run = tmp_path / "run"
    jdir = run / "judge" / "fake__m"
    (jdir / "decisions").mkdir(parents=True)
    by_ep = {}
    for d in decisions:
        by_ep.setdefault(d.episode_id, []).append(d)
    for eid, ds in by_ep.items():
        (jdir / "decisions" / f"{eid}.json").write_text(
            json.dumps({"decisions": [d.model_dump(mode="json") for d in ds]}), encoding="utf-8")
    return run, jdir


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def test_catalogue_csv_lists_every_eligible_clip_with_required_columns(tmp_path):
    run, jdir = write_judge_dir(tmp_path, catalogue_mix())
    out = write_report(run, jdir)
    rows = read_csv(out / "catalogue.csv")
    assert {r["candidate_id"] for r in rows} == ELIGIBLE
    assert [int(r["catalogue_rank"]) for r in rows] == list(range(1, len(ELIGIBLE) + 1))
    for col in ("catalogue_rank", "candidate_id", "episode_id", "start", "end", "duration_s", "score", "hook_score",
                "cliffhanger_score", "context_independence_score", "emotional_impact_score", "archetype",
                "ranking_status", "duplicate_of", "duplicate_overlap", "hook_text", "hook_timestamp",
                "hook_block_id", "cliffhanger_text", "cliffhanger_timestamp", "cliffhanger_block_id",
                "explanation", "explanation_source", "evidence_issues"):
        assert col in rows[0]
    dup = next(r for r in rows if r["candidate_id"] == "EP001_C02")
    assert dup["duplicate_of"] == "EP001_C01" and dup["ranking_status"] == "suppressed_duplicate"
    short = read_csv(out / "shortlist.csv")
    assert [r["candidate_id"] for r in short] == ["EP001_C01", "EP002_C01", "EP003_C01"]
    assert all(r["catalogue_rank"] and "hook_text" in r and "explanation" in r for r in short)
    rep = json.loads((out / "ranking.json").read_text(encoding="utf-8"))
    assert rep["counts"]["catalogue_size"] == len(ELIGIBLE)


def test_report_presents_shortlist_and_complete_catalogue_separately(tmp_path):
    run, jdir = write_judge_dir(tmp_path, catalogue_mix())
    md = (write_report(run, jdir) / "report.md").read_text(encoding="utf-8")
    assert "## Editorial shortlist" in md and "## Complete ranked catalogue" in md
    catalogue = md.split("## Complete ranked catalogue", 1)[1].split("## Weight sensitivity", 1)[0]
    assert all(cid in catalogue for cid in ELIGIBLE)
    assert "EP001_C01 (1.0)" in catalogue
    assert "explanations are model-written" in md


# --- Fix B: deliverable fields -----------------------------------------------------------------------

def judged(**override):
    ep, n = episode()
    c = candidate(ep, n)
    j, view = judgement_from(ep, c, lambda view: json.loads(default_reply(view.text, **override)))
    d = dec(c.candidate_id, c.start_ms / 1000, c.end_ms / 1000)
    return ep, c, j, d


def test_deliverable_quotes_the_judge_selected_source_blocks_verbatim():
    ep, c, j, d = judged()
    row = deliverable(d, j, ep, c)
    hook = ep.get_block(j.dimensions["hook"].block_id)
    beat = ep.get_block(j.dimensions["cliffhanger"].suspense_beat_id)
    assert row["hook_block_id"] == hook.block_id and row["hook_text"] == hook.text
    assert row["cliffhanger_block_id"] == beat.block_id and row["cliffhanger_text"] == beat.text
    assert row["hook_timestamp"].count(":") == 2 and "-" in row["hook_timestamp"]
    assert row["hook_offset_s"] == round((hook.start_ms - c.start_ms) / 1000, 2)
    assert row["explanation"] == "a betrayal confronted" and row["explanation_source"] == "judge_appeal"
    assert row["open_question"] == "What is behind the door?"
    assert row["evidence_issues"] == ""


def test_missing_hook_evidence_leaves_hook_empty_and_is_reported():
    ep, c, j, d = judged()
    j.dimensions["hook"] = j.dimensions["hook"].model_copy(update={"block_id": None})
    row = deliverable(d, j, ep, c)
    assert row["hook_text"] == "" and row["hook_block_id"] == "" and row["hook_offset_s"] == ""
    assert "no hook evidence" in row["evidence_issues"]
    assert row["cliffhanger_text"]


def test_missing_suspense_evidence_leaves_cliffhanger_empty_and_is_reported():
    ep, c, j, d = judged()
    j.dimensions["cliffhanger"] = j.dimensions["cliffhanger"].model_copy(update={"suspense_beat_id": None})
    row = deliverable(d, j, ep, c)
    assert row["cliffhanger_text"] == "" and row["cliffhanger_end_gap_s"] == ""
    assert "no suspense beat evidence" in row["evidence_issues"]


def test_invalid_block_reference_is_reported_not_quoted():
    ep, c, j, d = judged()
    j.dimensions["hook"] = j.dimensions["hook"].model_copy(update={"block_id": "EP009_B999"})
    row = deliverable(d, j, ep, c)
    assert row["hook_text"] == "" and row["hook_block_id"] == "EP009_B999"
    assert "hook block EP009_B999 is not in EP009" in row["evidence_issues"]


def test_without_parsed_episode_or_judgement_nothing_is_inferred():
    ep, c, j, d = judged()
    row = deliverable(d, j, None, c)
    assert row["hook_text"] == row["cliffhanger_text"] == ""
    assert "parsed episode not available" in row["evidence_issues"]
    row = deliverable(d, None, ep, c)
    assert row["hook_text"] == "" and "no judgement file" in row["evidence_issues"]
    assert row["explanation"] == "GEN_WHY_X" and row["explanation_source"] == "generator_why_it_works"


def test_explanation_falls_back_to_generator_and_is_labelled():
    ep, c, j, d = judged(promo_dna__appeal="")
    row = deliverable(d, j, ep, c)
    assert row["explanation"] == "GEN_WHY_X" and row["explanation_source"] == "generator_why_it_works"
    assert row["judge_appeal"] == ""
    row = deliverable(d, j, ep, c.model_copy(update={"why_it_works": ""}))
    assert row["explanation"] == "" and "no explanation recorded" in row["evidence_issues"]
    assert set(ranking.EXPLANATION_SOURCES) == {"judge_appeal", "generator_why_it_works"}
    assert all("not human-reviewed" in v for v in ranking.EXPLANATION_SOURCES.values())


def test_csv_escapes_commas_quotes_and_newlines():
    rows = [{"a": 'He said, "run"\nnow', "b": "x|y", "c": ["l"], "d": None}]
    parsed = list(csv.DictReader(io.StringIO(ranking._csv(rows))))
    assert parsed == [{"a": 'He said, "run"\nnow', "b": "x|y", "c": '["l"]', "d": ""}]
    assert ranking._quote("A", "x|y") == "A: x\\|y"


def test_report_quotes_hook_and_cliffhanger_from_parsed_episodes(tmp_path):
    ep, c, j, d = judged()
    run, jdir = write_judge_dir(tmp_path, [d])
    (jdir / "judgements").mkdir()
    (jdir / "judgements" / f"{c.candidate_id}.json").write_text(j.model_dump_json(), encoding="utf-8")
    (run / "candidates").mkdir()
    gen = EpisodeGeneration(episode_id="EP009", status="complete", provider="g", model="g", prompt_version="v",
                            prompt_sha256="0", candidates=[c])
    (run / "candidates" / "EP009.json").write_text(gen.model_dump_json(), encoding="utf-8")
    out = write_report(run, jdir, {9: ep})
    [row] = read_csv(out / "catalogue.csv")
    assert row["hook_text"] == ep.get_block(j.dimensions["hook"].block_id).text
    assert row["cliffhanger_text"] == ep.get_block(j.dimensions["cliffhanger"].suspense_beat_id).text
    assert row["explanation_source"] == "judge_appeal"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "- Hook (" in md and "- Cliffhanger (" in md and "a betrayal confronted" in md
    rep = build_report(run, jdir)
    assert "parsed episode not available" in rep["catalogue"][0]["evidence_issues"]
