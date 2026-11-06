"""Parser tests on synthetic screenplays (invented text, same format as the input pack)."""
from promopilot.parser import ParseError, parse_episode_text, parse_label, parse_pack
from promopilot.schemas import BlockType

import pytest

STANDARD = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A baker finds a letter in the flour.
CONTINUITY NOTES: Follows Episode 6.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. BAKERY – NIGHT
Scene start: 00:00.00
Scene end: 00:10.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:04.00)
Wide shot of an empty bakery.

**ACTION:**
Start: (00:01.00) | End: (00:04.00)
Mara lifts a sealed envelope from a flour sack.

**MARA (Whispering, Afraid):**
Start: (00:02.50) | End: (00:03.80)
"Who left this here?"

**TOMAS (O.S.) (Calm):**
Start: (00:03.90) | End: (00:05.00)
"I did."

**CAMERA:**
Start: (00:04.00) | End: (00:10.00)
Close-up on Mara turning.

SCENE 02
Scene heading: EXT. STREET – CONTINUOUS
Scene start: 00:10.00
Scene end: 00:20.00

Actions and dialogue:

**SOUND FX:**
Start: (00:10.00) | End: (00:12.00)
Thunder rolls.

**HOUSEKEEPER (Sharp, Stern, O.S.):**
Start: (00:12.00) | End: (00:14.00)
"Close the door!"

**CAMERA:**
Start: (00:12.00) | End: (00:20.00)
Wide shot of the rainy street.

--- SCRIPT END ---
"""


def build(body: str, number: int = 7) -> str:
    return (f"EPISODE NUMBER: Episode {number}\nEPISODE LOGLINE: x\nCONTINUITY NOTES: y\n\n"
            f"--- SCRIPT START ---\n\n{body}\n--- SCRIPT END ---\n")


def codes(episode):
    return [a.code for a in episode.anomalies]


def test_standard_episode_structure():
    ep = parse_episode_text(STANDARD, "episode-007-Episode 999.txt")
    assert ep.episode_id == "EP007" and ep.episode_number == 7
    assert ep.logline == "A baker finds a letter in the flour."
    assert [s.scene_id for s in ep.scenes] == ["EP007_S01", "EP007_S02"]
    assert ep.scenes[1].start_ms == 10000 and ep.scenes[1].end_ms == 20000
    assert ep.scenes[0].heading == "INT. BAKERY – NIGHT"
    assert [b.block_id for b in ep.blocks] == [f"EP007_B{i:03d}" for i in range(1, 9)]
    assert [b.block_type for b in ep.blocks] == [
        BlockType.CAMERA, BlockType.ACTION, BlockType.DIALOGUE, BlockType.DIALOGUE, BlockType.CAMERA,
        BlockType.SOUND_FX, BlockType.DIALOGUE, BlockType.CAMERA]
    assert ep.scenes[0].block_ids == [f"EP007_B{i:03d}" for i in range(1, 6)]
    assert codes(ep) == []
    assert ep.stats.duration_ms == 20000


def test_dialogue_block_fields_and_text_preserved():
    ep = parse_episode_text(STANDARD, "episode-007.txt")
    mara = ep.get_block("EP007_B003")
    assert mara.speaker == "MARA" and mara.delivery == ["Whispering", "Afraid"] and mara.offscreen is None
    assert (mara.start_raw, mara.end_raw, mara.start_ms, mara.end_ms) == ("00:02.50", "00:03.80", 2500, 3800)
    assert mara.text == '"Who left this here?"'
    assert mara.duration_ms == 1300
    assert ep.block_timestamps("EP007_B003") == ("00:02.50", "00:03.80")
    assert mara.line_number == STANDARD.splitlines().index("**MARA (Whispering, Afraid):**") + 1


def test_offscreen_tag_in_own_or_shared_parentheses():
    ep = parse_episode_text(STANDARD, "episode-007.txt")
    tomas, keeper = ep.get_block("EP007_B004"), ep.get_block("EP007_B007")
    assert (tomas.speaker, tomas.offscreen, tomas.delivery) == ("TOMAS", "O.S.", ["Calm"])
    assert (keeper.speaker, keeper.offscreen, keeper.delivery) == ("HOUSEKEEPER", "O.S.", ["Sharp", "Stern"])


@pytest.mark.parametrize("label, expected", [
    ("CAMERA", (BlockType.CAMERA, None, [], None)),
    ("SOUND FX", (BlockType.SOUND_FX, None, [], None)),
    ("MARA'S MOTHER (Weeping)", (BlockType.DIALOGUE, "MARA'S MOTHER", ["Weeping"], None)),
    ("NARRATOR (V.O.)", (BlockType.DIALOGUE, "NARRATOR", [], "V.O.")),
    ("OFF-SCREEN VOICE (Muffled)", (BlockType.DIALOGUE, "OFF-SCREEN VOICE", ["Muffled"], None)),
])
def test_parse_label(label, expected):
    assert parse_label(label) == expected


def test_uppercase_scene_keys_parse_and_are_flagged():
    variant = STANDARD.replace("Scene heading:", "SCENE HEADING:").replace("Scene start:", "SCENE START:") \
        .replace("Scene end:", "SCENE END:").replace("Actions and dialogue:", "ACTIONS AND DIALOGUE:")
    ep = parse_episode_text(variant, "episode-007.txt")
    assert [(s.start_ms, s.end_ms) for s in ep.scenes] == [(0, 10000), (10000, 20000)]
    assert ep.scenes[0].heading == "INT. BAKERY – NIGHT"
    assert codes(ep) == ["HEADER_CASE_VARIANT"]
    assert len(ep.blocks) == 8


def test_out_of_order_blocks_keep_file_order_and_are_flagged():
    body = """SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:10.00

**ACTION:**
Start: (00:05.50) | End: (00:07.00)
A door slams.

**SOUND FX:**
Start: (00:05.00) | End: (00:06.00)
Wind howls.
"""
    ep = parse_episode_text(build(body), "episode-007.txt")
    assert [b.block_id for b in ep.blocks] == ["EP007_B001", "EP007_B002"]
    assert [b.block_id for b in ep.time_ordered_blocks()] == ["EP007_B002", "EP007_B001"]
    flagged = [a for a in ep.anomalies if a.code == "FILE_ORDER_NOT_TIME_ORDER"]
    assert len(flagged) == 1 and flagged[0].block_id == "EP007_B002"
    assert ep.get_block("EP007_B001").start_ms == 5500


def test_block_overrunning_scene_is_flagged_not_clipped():
    body = """SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:10.00

**SOUND FX:**
Start: (00:09.00) | End: (00:10.30)
A bell rings.

SCENE 02
Scene heading: INT. ROOM – DAY
Scene start: 00:10.00
Scene end: 00:15.00

**CAMERA:**
Start: (00:10.00) | End: (00:15.00)
Close-up.
"""
    ep = parse_episode_text(build(body), "episode-007.txt")
    overrun = [a for a in ep.anomalies if a.code == "BLOCK_OVERRUNS_SCENE"]
    assert len(overrun) == 1 and overrun[0].block_id == "EP007_B001"
    assert ep.get_block("EP007_B001").end_ms == 10300
    assert ep.get_block("EP007_B001").scene_id == "EP007_S01"


def test_track_overlaps_and_gaps():
    body = """SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:20.00

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Shot one.

**A (Calm):**
Start: (00:00.50) | End: (00:02.00)
"Line one."

**B (Calm):**
Start: (00:01.50) | End: (00:02.50)
"Line two."

**CAMERA:**
Start: (00:02.80) | End: (00:04.00)
Shot two.

**ACTION:**
Start: (00:04.00) | End: (00:12.00)
A long walk.

**CAMERA:**
Start: (00:12.00) | End: (00:14.00)
Shot three.

**CAMERA:**
Start: (00:16.00) | End: (00:20.00)
Shot four.
"""
    ep = parse_episode_text(build(body), "episode-007.txt")
    c = codes(ep)
    assert c.count("OVERLAPPING_DIALOGUE") == 1 and c.count("OVERLAPPING_CAMERA") == 1
    gap = [a for a in ep.anomalies if a.code == "LARGE_CAMERA_GAP"]
    assert [(a.details["start_ms"], a.details["end_ms"]) for a in gap] == [(4000, 12000)]
    dead = [a for a in ep.anomalies if a.code == "UNCOVERED_TIME"]
    assert [(a.details["start_ms"], a.details["end_ms"]) for a in dead] == [(14000, 16000)]
    assert ep.stats.uncovered_intervals_ms == [(14000, 16000)]
    assert ep.stats.max_camera_gap_ms == 8000
    assert ep.stats.camera_coverage_ratio == 0.5


def test_interval_lookup_uses_time_and_all_tracks():
    ep = parse_episode_text(STANDARD, "episode-007.txt")
    ids = [b.block_id for b in ep.blocks_intersecting(3000, 4500)]
    assert ids == ["EP007_B001", "EP007_B002", "EP007_B003", "EP007_B004", "EP007_B005"]
    assert [b.block_id for b in ep.blocks_intersecting(4000, 4000)] == []


def test_malformed_blocks_are_reported():
    body = """SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:10.00

**CAMERA:**
Wide shot without a time line.

**ACTION:**
Start: (00:05.00) | End: (00:04.00)
Time runs backwards.

**ACTION:**
Start: (00:05.00) | End: (00:61.00)
Bad seconds value.

**ACTION:**
Start: (00:06.00) | End: (00:06.00)
Instant.

**ACTION:**
Start: 00:07.00 | End: 00:08.00
Bare time format.
continued on a second line.

stray text after a blank line
"""
    ep = parse_episode_text(build(body), "episode-007.txt")
    c = codes(ep)
    for code in ("BLOCK_MISSING_TIME", "END_BEFORE_START", "INVALID_TIMESTAMP", "ZERO_DURATION",
                 "TIME_FORMAT_VARIANT", "MULTILINE_BLOCK_TEXT", "UNPARSED_LINE"):
        assert code in c, code
    missing = ep.get_block("EP007_B001")
    assert missing.start_ms is None and not missing.has_valid_time
    assert missing.text == "Wide shot without a time line."
    assert ep.get_block("EP007_B003").end_ms is None
    assert ep.get_block("EP007_B005").text == "Bare time format.\ncontinued on a second line."
    assert ep.get_block("EP007_B005").start_ms == 7000
    assert all(b.has_valid_time for b in ep.time_ordered_blocks())


def test_missing_markers_and_header_fields():
    text = "EPISODE NUMBER: Episode 3\n\nSCENE 01\nScene start: 00:00.00\n\n**ACTION:**\nStart: (00:00.00) | End: (00:01.00)\nX.\n"
    ep = parse_episode_text(text, "episode-003.txt")
    c = codes(ep)
    for code in ("MISSING_SCRIPT_START", "MISSING_SCRIPT_END", "MISSING_HEADER_FIELD", "SCENE_MISSING_TIME"):
        assert code in c, code
    assert len(ep.blocks) == 1


def test_episode_number_sources():
    ep = parse_episode_text(STANDARD, "episode-008-Episode 1.txt")
    assert ep.episode_number == 7 and "EPISODE_NUMBER_MISMATCH" in codes(ep)
    headerless = STANDARD.replace("EPISODE NUMBER: Episode 7\n", "")
    assert parse_episode_text(headerless, "episode-012.txt").episode_number == 12
    with pytest.raises(ParseError):
        parse_episode_text(headerless, "notes.txt")


def test_parsing_is_deterministic():
    a = parse_episode_text(STANDARD, "episode-007.txt").model_dump_json()
    b = parse_episode_text(STANDARD, "episode-007.txt").model_dump_json()
    assert a == b


def test_json_round_trip_keeps_lookups():
    from promopilot.schemas import Episode
    ep = parse_episode_text(STANDARD, "episode-007.txt")
    again = Episode.model_validate_json(ep.model_dump_json())
    assert again.get_block("EP007_B003").text == ep.get_block("EP007_B003").text
    assert again.model_dump() == ep.model_dump()


def test_pack_level_duplicates_and_gaps(tmp_path):
    (tmp_path / "episode-001.txt").write_text(build("SCENE 01\n", 1), encoding="utf-8")
    (tmp_path / "episode-001-copy.txt").write_text(build("SCENE 01\n", 1), encoding="utf-8")
    (tmp_path / "episode-004.txt").write_text(build("SCENE 01\n", 4), encoding="utf-8")
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe\x00bad")
    episodes, summary = parse_pack(tmp_path)
    pack_codes = [a.code for a in summary.pack_anomalies]
    assert "DUPLICATE_EPISODE_NUMBER" in pack_codes
    assert "EPISODE_NUMBER_GAP" in pack_codes
    assert "UNREADABLE_FILE" in pack_codes
    assert [e.episode_number for e in episodes] == [1, 1, 4]


def test_missing_input_folder(tmp_path):
    with pytest.raises(ParseError):
        parse_pack(tmp_path / "nope")
    with pytest.raises(ParseError):
        parse_pack(tmp_path)
