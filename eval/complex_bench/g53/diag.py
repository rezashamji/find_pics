"""RESULTS 53 step 1: how often each g (two albums) truth-program error kind occurs, per truth version (all splits; truth
programs only, no system output). Word heuristics, so indicative. Per request:
  proxy_and     : the words name an event/activity and an album ANDs >= 2 photo questions none of which names it
                  ("roller derby" = helmet AND rink)
  event_proxy   : the words name an event/activity and no album question names it (tested through objects)
  hidden_time   : an album has dates/days but the words carry no time cue (build.TIME_CUE)
  hidden_place  : an album has a place the words never name (none of its words in the request, no named place)
  meta_question : a photo question asks about an address/date ("Is the address in ...?")
  small_album   : an album holds < 5 photos while the other holds >= 5 x more
Usage: python eval/complex_bench/g53/diag.py <queries folder> [...]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build import TIME_CUE, Lib, _names, _nrm  # noqa: E402
from common import ROOT, load_items  # noqa: E402

EVENT = re.compile(r"\b(concerts?|weddings?|parades?|games?|matche?s?|races?|(?:the|a|that) shows?|festivals?|ceremon(?:y|ies)|"
                   r"part(?:y|ies)|derby|football|rugby|hockey|soccer|tennis|wrestling|judo|cycling|reenactment|"
                   r"air ?show|car show|rodeo|classes)\b", re.I)
META = re.compile(r"\b(address|located in|taken in (?:19|20)\d\d|county|gps)\b", re.I)


def qs(a):
    return [q for k in ("need", "need_any", "not", "person_need") for q in (a.get(k) or []) if isinstance(q, str)]


def diag(folder, it, libs):
    kinds = defaultdict(set); n = defaultdict(int)
    for s in ("test", "dev", "train"):
        f = ROOT / folder / f"{s}.json"
        if not f.exists():
            continue
        for q in json.loads(f.read_text())["queries"]:
            if q["cat"] != "g":
                continue
            n[s] += 1
            words = " ".join(q["turns"]); wn = _nrm(words)
            u = q["user"]
            if u not in libs:
                libs[u] = Lib(u, it)
            named = libs[u].named_places(words)
            als = [a for a in q["program"] if isinstance(a, dict)]
            stems = {m.group(1).lower().split()[-1][:5] for m in EVENT.finditer(words)}
            if stems:
                if not any(st in x.lower() for a in als for x in qs(a) for st in stems):
                    kinds["event_proxy"].add(q["qid"])
                if any(len(a.get("need") or []) >= 2 and not any(st in x.lower() for x in a["need"] for st in stems) for a in als):
                    kinds["proxy_and"].add(q["qid"])
            if not TIME_CUE.search(words) and any(a.get("date_from") or a.get("date_to") or a.get("days") for a in als):
                kinds["hidden_time"].add(q["qid"])
            for a in als:
                for nm in _names(a.get("place")) + _names(a.get("place_any")):
                    toks = [t for t in re.findall(r"\w+", _nrm(nm)) if len(t) >= 3]
                    if not any(re.search(rf"\b{re.escape(t)}\b", wn) for t in toks) and not named:
                        kinds["hidden_place"].add(q["qid"])
            if any(META.search(x) for a in als for x in qs(a)):
                kinds["meta_question"].add(q["qid"])
            sz = [len(x) for x in q["answers"][-1]]
            if len(sz) == 2 and min(sz) < 5 and max(sz) >= 5 * max(1, min(sz)):
                kinds["small_album"].add(q["qid"])
    return n, kinds


def main():
    it = load_items(); libs = {}
    K = ["event_proxy", "proxy_and", "hidden_time", "hidden_place", "meta_question", "small_album"]
    print("folder | g requests (test/dev/train) | " + " | ".join(K))
    for folder in sys.argv[1:]:
        n, k = diag(folder, it, libs)
        print(f"{folder} | {n['test']}/{n['dev']}/{n['train']} | " + " | ".join(str(len(k[x])) for x in K))


if __name__ == "__main__":
    main()
