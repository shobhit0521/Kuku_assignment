import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_runs", Path(__file__).resolve().parent.parent / "scripts" / "compare_runs.py")
compare_runs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_runs)


def test_timing_checks_inclusive_boundaries():
    ok = compare_runs.timing_checks(0, 20_000, 3_000, 18_000)
    assert ok == {"duration_20_60": True, "hook_within_3s": True, "cliff_within_tolerance": True}
    bad = compare_runs.timing_checks(0, 19_990, 3_010, 17_980)
    assert bad == {"duration_20_60": False, "hook_within_3s": False, "cliff_within_tolerance": False}


def test_timing_checks_reject_hook_before_start_and_cliff_after_end():
    res = compare_runs.timing_checks(5_000, 40_000, 4_000, 41_000)
    assert not res["hook_within_3s"] and not res["cliff_within_tolerance"]
