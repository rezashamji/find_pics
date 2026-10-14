"""RESULTS 59b: which judge questions still need PIXELS at query time (-> the 9B judge) and which go to the facts
scorer cascade. Decided from the question text the engine passes to the judge (+ whether the engine draws a red box),
written down BEFORE any cascade was scored. The kinds are RESULTS 59's measured weak spots of the facts reader:
  red-box person look (per-photo F1 .74), count (.47), colour (.76), person attribute (.77), text / sign (.83),
  photo-level look (.81), plus the kinds read off its worst end-to-end requests: relations between things, small /
  background details, follow-up OR compositions.
Everything else (objects, scenes / places / trips, activities without a person word) -> scorer cascade.
"""
import re

COLOURS = ("red|orange|yellow|green|blue|purple|pink|magenta|teal|turquoise|brown|beige|grey|gray|black|white|golden|"
           "gold|silver|navy|violet|maroon|cyan|colou?r|colou?rful|neon|pastel")
RULES = [
    ("red-box person", None),       # crop_person == "box" or "red box" in the question
    ("count", re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|several|group of|crowd|many|multiple|\d+|"
                         r"pair of|alone|only one|single|couple)\b", re.I)),
    ("text / sign", re.compile(r"\b(text|sign|signs|signage|written|writing|reads?|says|words?|letters|logo|menu|banner|"
                               r"label|poster|number plate)\b", re.I)),
    ("photo look", re.compile(r"\b(blurr?y|blurred|out of focus|motion blur|black[- ]and[- ]white|monochrome|sepia|"
                              r"overexposed|underexposed|washed out|close-?up|long exposure|silhouette|reflections?|aerial|"
                              r"panorama|bokeh|macro|wide|tilted|grainy|dark|bright|hazy|smog|foggy|fog|lit|lighting|light|"
                              r"lights|sunlight|sunset|night)\b", re.I)),
    ("colour", re.compile(rf"\b({COLOURS})\b", re.I)),
    ("person attribute", re.compile(r"\b(person|man|woman|men|women|boy|girl|child|children|kid|kids|people|someone|"
                                    r"bride|groom|baby|he|she|guy|lady|player|players)\b", re.I)),
    ("relation", re.compile(r"\b(next to|beside|behind|in front of|on top of|underneath|under|above|below|between|near|"
                            r"nearby|against|leaning|surrounded by|attached to|hanging|on (his|her|their|its))\b", re.I)),
    ("small / background detail", re.compile(r"\b(background|in the distance|distant|far away|tiny|small|little|corner|"
                                             r"edge|faint|partially|barely)\b", re.I)),
    ("or-composition", re.compile(r"\bor\b", re.I)),
]
# the grounding tail itself contains "or" ("... toy, model or picture of one") - not a composition
_TAIL = re.compile(r"\s*\(not a drawing, painting, statue, toy, model or picture of one\)", re.I)


def route(question: str, crop) -> str:
    """'' = scorer cascade; else the name of the rule that sends the question to the judge."""
    if str(crop) == "box" or re.search(r"(?i)\bred box\b", question):
        return "red-box person"
    q = _TAIL.sub("", question)
    for name, rx in RULES[1:]:
        if rx.search(q):
            return name
    return ""
