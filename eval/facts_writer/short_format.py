"""RESULTS 61: the COMPACT facts format (target 40-80 output tokens) and the 27B's text-only rewrite prompt.
One line, fixed field order, empty fields omitted:
  scene: ...; objects: ...; colours: <colour thing>, ...; people: N - <person>, ...; activity: ...; text: ...; look: ...; event: ...
Colours are bound to the thing they belong to (RESULTS 59: unbound colour lists made the scorer say "red jacket" for a
red sock puppet); people carry per-person clothing colours (the c category); look = photo-level look words."""
import json
import re

FIELDS = ["scene", "objects", "colours", "people", "activity", "text", "look", "event"]

REWRITE = """Rewrite these photo facts as ONE compact line for a photo-search index. At most 70 tokens. Use this exact field order and leave out a field that would be empty:
scene: <1-3 words>; objects: <up to 12 visible things, the most distinctive first, small ones too>; colours: <up to 5, each a colour WITH the thing it belongs to, e.g. red jacket, blue car>; people: <count> - <up to 3 prominent people: sex or age, clothing colours, what they do>; activity: <up to 3>; text: <visible text, up to 6 words>; look: <photo-look words only: black and white, sepia, close-up, blurry, overexposed, dark, night, silhouette, reflection, aerial, panorama, long exposure>; event: <occasion>
Use only what the facts state. No sentences, no other words.
FACTS: {facts}"""

STUDENT_PROMPT = "Search facts for this photo, one line."


def facts_json(f) -> str:
    """The 27B facts as compact JSON without empty fields (the rewrite input)."""
    if not isinstance(f, dict):
        return "{}"
    keep = {k: v for k, v in f.items() if v not in (None, "", [], {}) and k != "raw"}
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":"))


def clean(line: str) -> str:
    line = " ".join((line or "").split())
    line = re.sub(r"^```\w*|```$", "", line).strip()
    return line[:1200]


def parse(line: str) -> dict:
    """Short line -> {field: text}; unknown pieces are kept under 'other'."""
    out = {}
    for part in re.split(r";\s*", line or ""):
        m = re.match(r"\s*([a-z]+)\s*:\s*(.*)$", part, re.I)
        if m and m.group(1).lower() in FIELDS:
            out[m.group(1).lower()] = m.group(2).strip()
        elif part.strip():
            out["other"] = (out.get("other", "") + " " + part.strip()).strip()
    return out
