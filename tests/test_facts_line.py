"""RESULTS 61d repeat-stop (findpics.facts_line): the golden fixture the Swift port is tested against, plus semantics."""
import json
from pathlib import Path

from findpics.facts_line import apply_repeat_stop, repeat_scan

FIX = Path(__file__).resolve().parents[1] / "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/facts_line.json"


def test_fixture_matches_python():
    for c in json.loads(FIX.read_text())["cases"]:
        s = repeat_scan(c["text"], c["version"])
        assert (None if s is None else list(s)) == c["scan"], c
        assert apply_repeat_stop(c["text"], c["version"]) == (c["kept"], c["stopped"]), c


def test_streaming_prefix_gives_the_same_cut():
    t = "scene: x; objects: a; b; a; b; a; c; d"
    full = apply_repeat_stop(t)
    first = next(n for n in range(len(t) + 1) if repeat_scan(t[:n]) is not None)
    assert apply_repeat_stop(t[:first]) == full == ("scene: x; objects: a; b; a; b", True)


def test_v2_default_keeps_legitimate_pairs():
    t = "people: 3 - man in red, person in black walking, person in black walking; activity: walking"
    assert apply_repeat_stop(t) == (t, False)
    assert apply_repeat_stop(t, 1)[1] is True
