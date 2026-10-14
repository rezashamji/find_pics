"""RESULTS 56 step 1: b (attribute / colour / style) truth-program error kinds per truth version (all splits; truth programs
only, no system output). Word heuristics, so indicative. Per request:
  hidden_time  : an album has dates/days but the words carry no time cue (build.TIME_CUE)
  hidden_place : an album has a place and the words name no place (no word of it, no address/landmark name, no
                 "the winery/hotel/..." phrase)
  extra_need   : a "need"/"need_any" question shares no content word with the request (build._content; 4-letter stems):
                 a condition the owner never stated ("Is there a stage?" for "the black and white photos from the London
                 gigs", "Is there a tree or bush?" for "those hazy mountain shots")
  one_day      : the album is pinned to ONE day while the words name no day/date (a remembered look made one occasion)
Usage: python eval/complex_bench/b56/diag.py <queries folder> [...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build import PLACE_NOUN, TIME_CUE, Lib, _content, _names, _nrm  # noqa: E402
from common import ROOT, load_items, norm  # noqa: E402

DAYWORD = re.compile(r"\d|\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                     r"christmas|halloween|thanksgiving|easter|new year|yesterday|today|tonight|that day|that night)\b", re.I)


def diag(folder, it, libs):
    kinds = defaultdict(set); n = defaultdict(int)
    for s in ("test", "dev", "train"):
        f = ROOT / folder / f"{s}.json"
        if not f.exists():
            continue
        for q in json.loads(f.read_text())["queries"]:
            if q["cat"] != "b":
                continue
            n[s] += 1
            words = " ".join(q["turns"]); wn = _nrm(words)
            u = q["user"]
            if u not in libs:
                libs[u] = Lib(u, it)
            named = libs[u].named_places(words) or PLACE_NOUN.search(words)
            W = {norm(w) for w in re.findall(r"[a-zA-Z]{3,}", words.lower())}; W4 = {w[:4] for w in W}
            for a in [a for a in q["program"] if isinstance(a, dict)]:
                if (a.get("date_from") or a.get("date_to") or a.get("days")) and not TIME_CUE.search(words):
                    kinds["hidden_time"].add(q["qid"])
                for nm in _names(a.get("place")) + _names(a.get("place_any")):
                    toks = [t for t in re.findall(r"\w+", _nrm(nm)) if len(t) >= 3]
                    if not any(re.search(rf"\b{re.escape(t)}\b", wn) for t in toks) and not named:
                        kinds["hidden_place"].add(q["qid"])
                for k in ("need", "need_any"):
                    for x in a.get(k) or []:
                        if isinstance(x, str) and not {c for c in _content(x) if c in W or c[:4] in W4}:
                            kinds["extra_need"].add(q["qid"])
                one = (len(a.get("days") or []) == 1) or (a.get("date_from") and a.get("date_to") and
                       (__import__("pandas").Timestamp(a["date_to"]) - __import__("pandas").Timestamp(a["date_from"])).days <= 1)
                if one and not DAYWORD.search(words):
                    kinds["one_day"].add(q["qid"])
    return n, kinds


def main():
    it = load_items(); libs = {}
    K = ["hidden_time", "hidden_place", "one_day", "extra_need"]
    print("folder | b requests (test/dev/train) | " + " | ".join(K))
    for folder in sys.argv[1:]:
        n, k = diag(folder, it, libs)
        print(f"{folder} | {n['test']}/{n['dev']}/{n['train']} | " + " | ".join(str(len(k[x])) for x in K))


if __name__ == "__main__":
    main()
