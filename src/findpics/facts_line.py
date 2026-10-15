"""The short facts line the phone's facts writer emits (RESULTS 61): "scene: ..; objects: a, b; colours: ..; ...".

REPEAT-STOP (RESULTS 61d): a small greedy writer can loop ("objects: kilt, kilt, kilt, ..." or, mostly, a collapsed
";"-list "mask; gloves; shoes; mask; gloves; ..." to the token cap). Decoding stops as soon as a COMPLETED list item (one
followed by "," or ";") reaches its LIMIT-th occurrence within its field; the line is cut just before that item. Items
compare lower-cased with whitespace runs collapsed and surrounding spaces / trailing periods stripped. The decision
depends only on the text generated so far, so stopping while streaming gives exactly the cut of the full greedy output.
Version 1 (pre-registered): every ";" segment is a field, limit 2. Version 2 (ADOPTED, the default): a new field starts
only at a "name:" segment, limit 3. See VERSIONS.
Swift twin: FindPicsCore/FactsLine.swift (golden parity test FactsLineTests).
"""
import re

_SPACES = re.compile(r"\s+")
_FIELD = re.compile(r"\s*[A-Za-z][A-Za-z ]*:\s*(.*)$", re.S)


def _norm(s: str) -> str:
    return _SPACES.sub(" ", s.strip().lower()).strip().rstrip(".").strip()


def repeat_cut(text: str, version: int = 2):
    """Character offset to cut `text` at (the kept line is text[:offset]), or None when no COMPLETED item repeats an
    earlier completed item of its field. Prefix-monotone: on a streamed prefix it returns the same cut as on the full
    text once the repeated item is completed, so decoding can stop at the first non-None."""
    r = repeat_scan(text, version)
    return None if r is None else r[0]


# version 1 = the rule pre-registered for RESULTS 61d (every ";" segment is a field; stop at the 2nd occurrence).
# version 2 = pre-registered AFTER v1's loop coverage was seen (JOURNAL 10-15 61d v2), BEFORE any v2 number: a new field
#   starts only at a "name:" segment (";"-separated lists without names - the 2B's collapsed format - stay one field),
#   and decoding stops at the 3rd occurrence of an item in its field (two equal items are often legitimate:
#   "person in black walking, person in black walking"); the cut is before that 3rd occurrence.
# ADOPTED (RESULTS 61d): version 2 is the default (both passed the bar; v2 loop rate .054 vs v1 .075, 70,580 photos).
VERSIONS = {1: dict(named_fields=False, limit=2), 2: dict(named_fields=True, limit=3)}


def repeat_scan(text: str, version: int = 2):
    """(cut offset, end offset) or None. end = offset just past the "," / ";" that completed the repeated item, i.e.
    the generated text a streaming decoder has seen when it stops (the tokens decoded up to text[:end])."""
    named, limit = VERSIONS[version]["named_fields"], VERSIONS[version]["limit"]
    seen = {}                       # item -> occurrences in the current field
    item_start = 0                  # offset where the current item begins
    first_in_seg = True
    for i, ch in enumerate(text):
        if ch != "," and ch != ";":
            continue
        item = text[item_start:i]
        if first_in_seg:            # "objects: kilt" -> "kilt"; the field name is not an item
            m = _FIELD.match(item)
            if m:
                item = m.group(1)
                if named:
                    seen = {}
        key = _norm(item)
        if key and seen.get(key, 0) + 1 >= limit:
            return (item_start - 1 if item_start > 0 and text[item_start - 1] in ",;" else item_start), i + 1
        if key:
            seen[key] = seen.get(key, 0) + 1
        item_start = i + 1
        first_in_seg = ch == ";"
        if ch == ";" and not named:
            seen = {}
    return None


def apply_repeat_stop(text: str, version: int = 2):
    """(line after repeat-stop, stopped?)"""
    c = repeat_cut(text, version)
    if c is None:
        return text, False
    return text[:c].rstrip(" ,;"), True
