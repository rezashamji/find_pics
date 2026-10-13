"""RESULTS 52 step 1: how often each systematic h truth-program error occurs, per truth version (all splits; truth
programs only - no system output is read). Kinds, counted per request over its turns (final compiled programs of a
queries folder, plus the raw generated programs):
  whole_library : a turn's answer holds >= 25% of the library (the words never ask for that)
  strip_named   : the 10-12 strip removed a place/date constraint while the words NAME a place (an address name of the
                  library at any level, a landmark of the photo facts, or "the winery/hospital/zoo...")
  edit_narrow / edit_widen : a follow-up whose words only narrow returns > 10% photos not in the previous turn / one
                  that only widens loses > 10% of the previous turn (build.edit_problems)
  event_window  : the words name an event (parade, concert, wedding, game, party, festival, show, ceremony, race,
                  graduation, birthday, fair, carnival, rodeo, gig, match, tournament, recital, performance) and the
                  turn's program has no question at all (the event = a whole day/window)
  event_proxy   : the words name such an event, the program has questions, and none of them names the event (or its
                  word stem): the event is tested through objects it may contain (a stage, a microphone, a flag)
  places_or     : the words name >= 2 places of the library's addresses and the turn's program covers fewer of them
Usage: python eval/complex_bench/h52/diag.py <queries folder> [<queries folder> ...] -> printed table
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from build import PLACE_NOUN, Lib, _names, _nrm, edit_problems  # noqa: E402
from common import ROOT, load_items  # noqa: E402

EVENT = re.compile(r"\b(parade|concert|wedding|game|games|party|festival|(?:the|a|that) show|ceremony|race|graduation|birthday|"
                   r"fair|carnival|rodeo|gig|match|tournament|recital|performance)\b(?! car| day| trip| weekend)", re.I)


def questions(a):
    qs = []
    for k in ("need", "need_any", "not", "person_need"):
        qs += [q for q in (a.get(k) or []) if isinstance(q, str)]
    anc = a.get("anchor")
    if isinstance(anc, dict):
        qs += [q for q in anc.get("need") or [] if isinstance(q, str)]
    return qs


def diag(folder, it, libs):
    kinds = defaultdict(set)
    n = Counter()
    for s in ("test", "dev", "train"):
        f = ROOT / folder / f"{s}.json"
        if not f.exists():
            continue
        for q in json.loads(f.read_text())["queries"]:
            if q["cat"] != "h":
                continue
            n[s] += 1
            u = q["user"]
            if u not in libs:
                libs[u] = Lib(u, it)
            lib = libs[u]
            N = q["library_size"]
            for k, (albums, ans) in enumerate(zip(q["program"], q["answers"])):
                words = " ".join(q["turns"][:k + 1])
                flat = set().union(*map(set, ans)) if ans else set()
                if len(flat) >= 0.25 * N:
                    kinds["whole_library"].add(q["qid"])
                ev = EVENT.search(words)
                als = [a for a in albums or [] if isinstance(a, dict)]
                qs = [x for a in als for x in questions(a)]
                if ev:
                    if not qs:
                        kinds["event_window"].add(q["qid"])
                    else:
                        stems = {m.group(1).lower().split()[-1][:5] for m in EVENT.finditer(words)}
                        if not any(st in x.lower() for x in qs for st in stems):
                            kinds["event_proxy"].add(q["qid"])
                named = {x for x in lib.named_places(words) if lib.place_mask(x).sum() >= 2}
                addr_named = {x for x in named if lib.place_mask(x).any()}
                if len(addr_named) >= 2:
                    cov = {x for x in addr_named for a in als
                           for nm in _names(a.get("place")) + _names(a.get("place_any"))
                           if re.sub(r"\W+", "", x) in re.sub(r"\W+", "", _nrm(nm))}
                    if len(cov) < len(addr_named):
                        kinds["places_or"].add(q["qid"])
            gone = [g for g in q.get("stripped") or []]
            if gone:   # something was stripped and a place the words name is covered by no album's place fields
                named = lib.named_places(" ".join(q["turns"]))
                pl = [_nrm(nm) for albums in q["program"] for a in albums or [] if isinstance(a, dict)
                      for nm in _names(a.get("place")) + _names(a.get("place_any"))]
                unc = {x for x in named if not any(re.sub(r"\W+", "", x) in re.sub(r"\W+", "", y) for y in pl)}
                if unc or (PLACE_NOUN.search(" ".join(q["turns"])) and not pl):
                    kinds["strip_named"].add(q["qid"])
            for p in edit_problems(q["turns"], q["answers"]):
                kinds["edit_narrow" if "narrows" in p else "edit_widen"].add(q["qid"])
    return n, kinds


def main():
    it = load_items()
    libs = {}
    rows = []
    for folder in sys.argv[1:]:
        n, kinds = diag(folder, it, libs)
        rows.append((folder, n, kinds))
    K = ["whole_library", "strip_named", "edit_narrow", "edit_widen", "event_window", "event_proxy", "places_or"]
    print("folder | h requests (test/dev/train) | " + " | ".join(K))
    for folder, n, kinds in rows:
        print(f"{folder} | {n['test']}/{n['dev']}/{n['train']} | " + " | ".join(str(len(kinds[k])) for k in K))
    for folder, n, kinds in rows:
        print(f"[{folder}] " + "; ".join(f"{k}: " + " ".join(sorted(q for q in kinds[k] if not q.startswith("x"))[:0]) + str(len(kinds[k])) for k in K))

if __name__ == "__main__":
    main()
