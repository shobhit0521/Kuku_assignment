import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from promopilot import config  # noqa: E402
from promopilot.parser import parse_pack  # noqa: E402


@pytest.fixture(scope="session")
def pack_dir() -> Path:
    path = config.input_dir_from_env()
    if path is None or not path.is_dir():
        pytest.skip("PROMOPILOT_INPUT_DIR not set or missing; real-pack tests skipped")
    return path


@pytest.fixture(scope="session")
def parsed_pack(pack_dir):
    return parse_pack(pack_dir)
