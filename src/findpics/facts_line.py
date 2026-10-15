"""The short facts line the phone's facts writer emits (RESULTS 61): "scene: ..; objects: a, b; colours: ..; ...".

REPEAT-STOP (RESULTS 61d): a small greedy writer can loop ("objects: kilt, kilt, kilt, ..." to the token cap). Decoding
stops as soon as a COMPLETED list item repeats an earlier completed item of the SAME field; the line is cut just before
the repeated item (its leading ", " dropped). An item is completed when the next "," or ";" follows it. Items compare
lower-cased with runs of spaces collapsed and surrounding spaces / a trailing period stripped. Fields are the
"name: value" segments separated by ";"; text before the first "name:" is its own field. The decision depends only on the
text generated so far, so stopping while streaming gives exactly the cut of the full greedy output.
Swift twin: FindPicsCore/FactsLine.swift (golden parity test FactsLineTests).
"""
import re

_SPACES = re.compile(r"\s+")
_FIELD = re.compile(r"\s*[A-Za-z][A-Za-z ]*:\s*(.*)$", re.S)


def _norm(s: str) -> str:
    return _SPACES.sub(" ", s.strip().lower()).strip().rstrip(".").strip()


def repeat_cut(text: str):
    """Character offset to cut `text` at (the kept line is text[:offset]), or None when no COMPLETED item repeats an
    earlier completed item of its field. Prefix-monotone: on a streamed prefix it returns the same cut as on the full
    text once the repeated item is completed, so decoding can stop at the first non-None."""
    seen = set()
    item_start = 0                  # offset where the current item begins
    first_in_field = True
    for i, ch in enumerate(text):
        if ch != "," and ch != ";":
            continue
        item = text[item_start:i]
        if first_in_field:          # "objects: kilt" -> "kilt"; the field name is not an item
            m = _FIELD.match(item)
            if m:
                item = m.group(1)
        key = _norm(item)
        if key and key in seen:
            return item_start - 1 if item_start > 0 and text[item_start - 1] == "," else item_start
        if key:
            seen.add(key)
        item_start = i + 1
        first_in_field = ch == ";"
        if ch == ";":
            seen = set()
    return None


def apply_repeat_stop(text: str):
    """(line after repeat-stop, stopped?)"""
    c = repeat_cut(text)
    if c is None:
        return text, False
    return text[:c].rstrip(" ,;"), True
