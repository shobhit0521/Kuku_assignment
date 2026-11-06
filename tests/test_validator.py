"""Deterministic validator and repair engine on synthetic episodes (invented text, no screenplay content)."""
import hashlib
import json

import pytest

from promopilot import config
from promopilot.schemas import (BeatInput, BeatSource, Block, BlockType, Candidate, CandidateStatus, Episode,
                                EpisodeGeneration, Reason, RuleStatus, SemanticInputs, TimelineStats)
from promopilot.timecode import format_timestamp
from promopilot.validator import (advisory_inputs, beat_class, candidate_from_blocks, is_score_cue, is_transition,
                                  repair_candidate, run_validation, validate_and_repair, validate_candidate)

SETTINGS = {"min_clip_ms": 20_000, "max_clip_ms": 60_000, "hook_window_ms": 3_000, "suspense_tail_ms": 2_000,
            "ending_window_ms": 8_000}
TYPES = {"CAMERA": BlockType.CAMERA, "ACTION": BlockType.ACTION, "DIALOGUE": BlockType.DIALOGUE,
         "SOUND_FX": BlockType.SOUND_FX}


def make_episode(extra=(), track_end=80, eid="EP009"):
    """A 2-second camera track from 0 to track_end, plus extra (name, type, start_s, end_s, text) blocks.
    Returns the episode and a name -> block ID map; camera shot starting at t seconds is named 'c<t>'."""
    specs = [(f"c{t}", "CAMERA", t, t + 2, f"Shot {t}.") for t in range(0, track_end, 2)] + list(extra)
    blocks, names = [], {}
    for i, (name, kind, s, e, text) in enumerate(specs, 1):
        bid = f"{eid}_B{i:03d}"
        names[name] = bid
        s_ms, e_ms = round(s * 1000), round(e * 1000)
        blocks.append(Block(block_id=bid, episode_id=eid, file_index=i, scene_id=f"{eid}_S01",
                            block_type=TYPES[kind], label_raw="MARA (Low)" if kind == "DIALOGUE" else kind,
                            speaker="MARA" if kind == "DIALOGUE" else None, start_raw=format_timestamp(s_ms),
                            end_raw=format_timestamp(e_ms), start_ms=s_ms, end_ms=e_ms, text=text, line_number=i))
    duration = max(b.end_ms for b in blocks)
    ep = Episode(episode_id=eid, episode_number=int(eid[2:]), source_file="synthetic.txt", source_sha256="0",
                 parser_version="test", blocks=blocks,
                 stats=TimelineStats(duration_ms=duration, camera_coverage_ratio=1.0, max_camera_gap_ms=0))
    return ep, names


def cand(ep, n, start, end, hook, cliff=None, cid="EP009_C01"):
    return candidate_from_blocks(ep, cid, n[start], n[end], n[hook], n[cliff or end])


def beats(n, suspense=None, hook=None, source=BeatSource.TEST):
    return SemanticInputs(hook=BeatInput(block_id=n[hook], source=source) if hook else None,
                          suspense_beat=BeatInput(block_id=n[suspense], source=source) if suspense else None)


def check(result, rule):
    return next(c for c in result.checks if c.rule == rule)


# --- classification --------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Cut to black.", "CUT TO BLACK", "Fade out.", "Slow fade to black.",
                                  "Smash cut to black!", "Screen fades to black.", "Fade to white.", "Black screen.",
                                  "Title card.", "To be continued..."])
def test_transition_only_texts(text):
    ep, n = make_episode([("t", "CAMERA", 0, 1, text)], track_end=0)
    assert is_transition(ep.get_block(n["t"]))


@pytest.mark.parametrize("text", [
    "Dramatic hold on Mara's grim profile as the score swells into the final cut to black.",
    "Mara turns away. Cut to black.",
    "Close-up of Mara's face hardening.",
    "Blackwood Manor at dusk."])
def test_shots_that_mention_transitions_are_story_beats(text):
    ep, n = make_episode([("t", "CAMERA", 0, 1, text)], track_end=0)
    assert not is_transition(ep.get_block(n["t"])) and beat_class(ep.get_block(n["t"])) == "story"


@pytest.mark.parametrize("text, cue", [("Dramatic musical chord swelling.", True),
                                       ("Low string drone swells.", True),
                                       ("Heavy doors creak open, followed by a musical swell.", False),
                                       ("A wooden beam crashes to the floor.", False),
                                       ("Glass shatters.", False)])
def test_score_cue_detection(text, cue):
    ep, n = make_episode([("s", "SOUND_FX", 0, 1, text)], track_end=0)
    assert is_score_cue(ep.get_block(n["s"])) is cue


# --- inclusive limits ------------------------------------------------------------------------------

@pytest.mark.parametrize("end_s, ok", [(30, True), (29.99, False), (70, True), (70.01, False)])
def test_duration_bounds_are_inclusive(end_s, ok):
    ep, n = make_episode([("end", "ACTION", 25, end_s, "Mara stops.")])
    v = validate_candidate(ep, cand(ep, n, "c10", "end", "c10"), beats(n, "end"), SETTINGS)
    assert check(v, "DURATION_RANGE").status == (RuleStatus.PASS if ok else RuleStatus.FAIL)


@pytest.mark.parametrize("hook_s, ok", [(13.0, True), (13.01, False), (10.0, True)])
def test_hook_window_is_inclusive_at_three_seconds(hook_s, ok):
    ep, n = make_episode([("hook", "ACTION", hook_s, hook_s + 1, "Mara slams the door.")])
    v = validate_candidate(ep, cand(ep, n, "c10", "c38", "hook"), beats(n, "c38"), SETTINGS)
    assert check(v, "HOOK_WINDOW").status == (RuleStatus.PASS if ok else RuleStatus.FAIL)
    assert v.hook_offset_ms == round((hook_s - 10) * 1000)


def test_hook_starting_before_the_clip_fails():
    ep, n = make_episode([("hook", "ACTION", 9, 12, "Mara slams the door.")])
    v = validate_candidate(ep, cand(ep, n, "c10", "c38", "hook"), beats(n, "c38"), SETTINGS)
    assert check(v, "HOOK_WINDOW").status == RuleStatus.FAIL


@pytest.mark.parametrize("beat_end, ok", [(28.0, True), (27.99, False)])
def test_suspense_tail_is_inclusive_at_two_seconds(beat_end, ok):
    ep, n = make_episode([("beat", "ACTION", 20, beat_end, "Mara reaches for the letter.")])
    v = validate_candidate(ep, cand(ep, n, "c10", "c28", "c10"), beats(n, "beat"), SETTINGS)
    tail = check(v, "SUSPENSE_TAIL_TOLERANCE")
    assert tail.status == (RuleStatus.PASS if ok else RuleStatus.FAIL)
    assert tail.value_ms == 30_000 - round(beat_end * 1000)


# --- semantic inputs ------------------------------------------------------------------------------

def test_generator_cliffhanger_is_not_used_as_the_suspense_beat_by_default():
    ep, n = make_episode([("early", "ACTION", 14, 15, "Mara gasps.")])
    c = cand(ep, n, "c10", "c38", "c10", cliff="early")
    v = validate_candidate(ep, c, None, SETTINGS)
    assert v.status == RuleStatus.PENDING and v.provisional
    assert all(check(v, r).status == RuleStatus.PENDING
               for r in ("SUSPENSE_BEAT_IN_CLIP", "SUSPENSE_BEAT_IS_STORY_BEAT", "SUSPENSE_TAIL_TOLERANCE"))
    preview = validate_candidate(ep, c, advisory_inputs(c, include_generator_cliffhanger=True), SETTINGS)
    assert preview.status == RuleStatus.FAIL and preview.provisional
    assert check(preview, "SUSPENSE_TAIL_TOLERANCE").status == RuleStatus.FAIL


def test_only_judge_or_human_beats_make_a_result_final():
    ep, n = make_episode()
    c = cand(ep, n, "c10", "c38", "c10")
    assert validate_candidate(ep, c, beats(n, "c38", "c10", BeatSource.TEST), SETTINGS).provisional
    final = validate_candidate(ep, c, beats(n, "c38", "c10", BeatSource.HUMAN), SETTINGS)
    assert final.status == RuleStatus.PASS and not final.provisional


def test_suspense_beat_outside_clip_or_unknown_fails():
    ep, n = make_episode()
    c = cand(ep, n, "c10", "c38", "c10")
    assert check(validate_candidate(ep, c, beats(n, "c50"), SETTINGS), "SUSPENSE_BEAT_IN_CLIP").status == RuleStatus.FAIL
    bad = SemanticInputs(suspense_beat=BeatInput(block_id="EP007_B001", source=BeatSource.JUDGE),
                         hook=BeatInput(block_id=n["c10"], source=BeatSource.JUDGE))
    assert check(validate_candidate(ep, c, bad, SETTINGS), "SUSPENSE_BEAT_IN_CLIP").status == RuleStatus.FAIL


# --- endings --------------------------------------------------------------------------------------

def test_transition_only_ending_fails_and_is_trimmed():
    ep, n = make_episode([("black", "CAMERA", 30, 33, "Cut to black.")], track_end=30)
    c = cand(ep, n, "c10", "black", "c10")
    rec = validate_and_repair(ep, c, None, SETTINGS)
    rule = check(rec.validation, "ENDING_NOT_TRANSITION_ONLY")
    assert rule.status == RuleStatus.FAIL and rule.repairable and rule.value_ms == 3000
    assert rec.repair.status == "repaired"
    assert rec.repair.actions[0].action == "trim_transition_tail"
    assert (rec.repair.repaired_candidate.start_ms, rec.repair.repaired_candidate.end_ms) == (10_000, 30_000)
    assert rec.repair.repaired_validation.status == RuleStatus.PENDING


def test_transition_cannot_be_the_suspense_beat_but_a_reaction_hold_can():
    ep, n = make_episode([("black", "CAMERA", 30, 31, "Cut to black."),
                          ("hold", "CAMERA", 30, 31, "Dramatic hold on Mara's face as the music cuts to black.")],
                         track_end=30)
    c = cand(ep, n, "c10", "black", "c10")
    v = validate_candidate(ep, c, beats(n, "black"), SETTINGS)
    assert check(v, "SUSPENSE_BEAT_IS_STORY_BEAT").status == RuleStatus.FAIL
    v2 = validate_candidate(ep, c, beats(n, "hold"), SETTINGS)
    assert v2.status == RuleStatus.PASS
    assert v2.ending.transition_block_ids == [n["black"]] and n["hold"] in v2.ending.suspense_beat_options


def test_ending_analysis_reports_last_dialogue_and_story_beat():
    ep, n = make_episode([("line", "DIALOGUE", 31, 33, '"Not yet."'), ("cue", "SOUND_FX", 36, 40, "Music swells.")])
    v = validate_candidate(ep, cand(ep, n, "c10", "c38", "c10"), beats(n, "c38"), SETTINGS)
    assert v.ending.last_dialogue_block_id == n["line"]
    assert v.ending.last_story_beat_id == n["c38"] and v.ending.trailing_after_last_story_ms == 0
    assert v.ending.score_cue_block_ids == [n["cue"]]


# --- dialogue boundaries and repair ---------------------------------------------------------------

def test_dialogue_expansion_succeeds_and_is_revalidated():
    ep, n = make_episode([("line", "DIALOGUE", 9, 11, '"You came back."')])
    c = cand(ep, n, "c10", "c38", "c10")
    rec = validate_and_repair(ep, c, beats(n, "c38", "c10"), SETTINGS)
    assert check(rec.validation, "DIALOGUE_BOUNDARY").block_ids == [n["line"]]
    assert rec.repair.status == "repaired"
    r = rec.repair.repaired_candidate
    assert r.start_block_id == n["line"] and (r.start_ms, r.end_ms) == (9_000, 40_000)
    assert rec.repair.repaired_validation.status == RuleStatus.PASS
    assert [a.action for a in rec.repair.actions] == ["expand_start_for_dialogue"]


def test_dialogue_expansion_that_breaks_duration_is_unrepairable():
    ep, n = make_episode([("line", "DIALOGUE", 69, 71, '"Wait."')])
    rec = validate_and_repair(ep, cand(ep, n, "c10", "c68", "c10"), beats(n, "c68", "c10"), SETTINGS)
    assert rec.repair.status == "unrepairable" and rec.repair.repaired_candidate is None
    assert "DURATION_RANGE" in rec.repair.reason
    assert rec.repair.repaired_validation.duration_ms == 61_000


def test_dialogue_expansion_that_breaks_hook_window_is_unrepairable():
    ep, n = make_episode([("line", "DIALOGUE", 8, 10.5, '"Listen."'), ("hook", "ACTION", 12.5, 13, "A gunshot.")])
    rec = validate_and_repair(ep, cand(ep, n, "c10", "c38", "hook"), beats(n, "c38", "hook"), SETTINGS)
    assert rec.repair.status == "unrepairable" and "HOOK_WINDOW" in rec.repair.reason


def test_trim_then_extend_start_skips_points_that_cut_dialogue():
    ep, n = make_episode([("hook", "ACTION", 20.5, 21, "Mara drops the key."),
                          ("beat", "ACTION", 30, 38, "The lock turns."),
                          ("line", "DIALOGUE", 17.5, 19, '"Who is there?"')])
    c = cand(ep, n, "c20", "c44", "hook")
    rec = validate_and_repair(ep, c, beats(n, "beat", "hook"), SETTINGS)
    assert check(rec.validation, "SUSPENSE_TAIL_TOLERANCE").value_ms == 8_000
    assert rec.repair.status == "repaired"
    r = rec.repair.repaired_candidate
    assert (r.start_ms, r.end_ms) == (17_500, 38_000) and r.start_block_id == n["line"]
    assert [a.action for a in rec.repair.actions] == ["trim_end_to_suspense_beat", "extend_start_backwards"]
    assert rec.repair.repaired_validation.status == RuleStatus.PASS


def test_too_short_without_room_for_the_hook_is_unrepairable():
    ep, n = make_episode([("hook", "ACTION", 4.5, 5, "A scream.")])
    rec = validate_and_repair(ep, cand(ep, n, "c4", "c14", "hook"), beats(n, "c14", "hook"), SETTINGS)
    assert check(rec.validation, "DURATION_RANGE").status == RuleStatus.FAIL
    assert rec.repair.status == "unrepairable"


def test_too_long_is_not_shortened():
    ep, n = make_episode()
    rec = validate_and_repair(ep, cand(ep, n, "c0", "c62", "c0"), beats(n, "c62", "c0"), SETTINGS)
    assert rec.repair.status == "unrepairable" and "longer than the maximum" in rec.repair.reason


def test_non_repairable_failures_are_not_repaired():
    ep, n = make_episode([("hook", "ACTION", 16, 17, "Thunder.")])
    rec = validate_and_repair(ep, cand(ep, n, "c10", "c38", "hook"), beats(n, "c38", "hook"), SETTINGS)
    assert rec.repair.status == "unrepairable" and "HOOK_WINDOW" in rec.repair.reason


def test_grounding_rejected_candidate_is_technically_invalid():
    ep, n = make_episode()
    c = Candidate(candidate_id="EP009_C02", episode_id="EP009", status=CandidateStatus.REJECTED,
                  rejection_reasons=[Reason(code="QUOTE_MISMATCH", message="x")], start_block_id=n["c10"],
                  end_block_id=n["c38"], hook_block_id=n["c10"])
    rec = validate_and_repair(ep, c, None, SETTINGS)
    assert check(rec.validation, "REFERENCES_VALID").status == RuleStatus.FAIL
    assert "QUOTE_MISMATCH" in check(rec.validation, "REFERENCES_VALID").message
    assert all(check(rec.validation, r).status == RuleStatus.SKIPPED for r in ("DURATION_RANGE", "HOOK_WINDOW"))
    assert rec.repair.status == "unrepairable"


def test_cross_episode_reference_fails():
    ep, n = make_episode()
    c = cand(ep, n, "c10", "c38", "c10")
    inputs = SemanticInputs(hook=BeatInput(block_id="EP010_B001", source=BeatSource.JUDGE))
    v = validate_candidate(ep, c, inputs, SETTINGS)
    assert check(v, "REFERENCES_VALID").status == RuleStatus.FAIL and "EP010_B001" in v.checks[0].message


# --- immutability and provenance ------------------------------------------------------------------

def test_original_candidate_is_never_modified_and_repair_has_provenance():
    ep, n = make_episode([("line", "DIALOGUE", 9, 11, '"You came back."')])
    c = cand(ep, n, "c10", "c38", "c10")
    before = c.model_dump()
    rec = validate_and_repair(ep, c, beats(n, "c38", "c10", BeatSource.HUMAN), SETTINGS)
    assert c.model_dump() == before and c.repair is None
    r = rec.repair.repaired_candidate
    assert r.candidate_id == "EP009_C01_R1" and f"REPAIRED_FROM:EP009_C01" in r.flags
    p = r.repair
    assert (p.original_candidate_id, p.original_start_block_id, p.original_end_block_id) == \
        ("EP009_C01", n["c10"], n["c38"])
    assert (p.original_start_ms, p.original_end_ms, p.new_start_ms, p.new_end_ms) == (10_000, 40_000, 9_000, 40_000)
    assert p.suspense_beat_id == n["c38"] and p.suspense_beat_source == "human"
    assert p.validator_version == config.VALIDATOR_VERSION and p.actions
    assert r.hard_rule_validation.startswith("pass")
    assert r.why_it_works == c.why_it_works and r.hook_block_id == c.hook_block_id


def test_repair_is_a_single_attempt():
    ep, n = make_episode([("line", "DIALOGUE", 9, 11, '"You came back."')])
    rec = validate_and_repair(ep, cand(ep, n, "c10", "c38", "c10"), beats(n, "c38", "c10"), SETTINGS)
    again = repair_candidate(ep, rec.repair.repaired_candidate, rec.repair.repaired_validation,
                             beats(n, "c38", "c10"), SETTINGS)
    assert again.status == "not_needed"


def test_run_validation_reads_but_never_writes_candidate_files(tmp_path):
    ep, n = make_episode([("line", "DIALOGUE", 9, 11, '"You came back."')])
    gen = EpisodeGeneration(episode_id="EP009", status="complete", provider="p", model="m", prompt_version="gen-v1",
                            prompt_sha256="x", candidates=[cand(ep, n, "c10", "c38", "c10")])
    legacy = json.loads(gen.model_dump_json())
    for c in legacy["candidates"]:
        c.pop("repair")
    path = tmp_path / "candidates" / "EP009.json"
    path.parent.mkdir()
    path.write_text(json.dumps(legacy), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    beat_map = {"EP009_C01": beats(n, "c38", source=BeatSource.HUMAN)}
    summary = run_validation(tmp_path, {9: ep}, beats=beat_map, settings=SETTINGS)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    rec = summary["episodes"]["EP009"][0]
    assert rec["status"] == "fail" and rec["repair"] == "repaired" and rec["repaired_status"] == "pass"
    saved = json.loads((tmp_path / "validation" / "EP009.json").read_text(encoding="utf-8"))
    assert saved["source_sha256"] == digest
    assert saved["records"][0]["repair"]["repaired_candidate"]["candidate_id"] == "EP009_C01_R1"
