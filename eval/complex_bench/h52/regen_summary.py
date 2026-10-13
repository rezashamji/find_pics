"""RESULTS 52: what the 27B rewrite of the h truth programs changed (programs/ vs programs_h2/; all splits, programs only).
Usage: python eval/complex_bench/h52/regen_summary.py"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ROOT  # noqa: E402

QK = ("need", "need_any", "not", "person_need")
MK = ("place", "place_any", "date_from", "date_to", "days", "hours")


def qs(albums):
    return {q for a in albums or [] if isinstance(a, dict) for k in QK for q in (a.get(k) or []) if isinstance(q, str)}


def main():
    split = json.loads((ROOT / "splits.json").read_text())
    so = {u: s for s, us in split.items() for u in us}
    c = Counter()
    for f in sorted((ROOT / "programs_h2").glob("*.json")):
        u = f.stem
        P = json.loads((ROOT / "programs" / f"{u}.json").read_text())
        for qi, x in json.loads(f.read_text()).items():
            r = P["requests"][int(qi)]
            c["requests"] += 1; c[f"split_{so.get(u, 'train')}"] += 1
            if not x.get("ok"):
                c["rewrite failed (original kept)"] += 1; continue
            c["rewrite ok"] += 1
            old, new = r.get("turn_programs") or [], x["turn_programs"]
            strip = lambda tp: json.dumps([{k: v for k, v in a.items() if k != "name" and v not in (None, [], "", {})} for a in tp if isinstance(a, dict)], sort_keys=True)
            changed = [k for k in range(len(new)) if k >= len(old) or strip(old[k]) != strip(new[k])]
            c["requests with >= 1 turn changed"] += bool(changed)
            c["turns changed"] += len(changed); c["turns"] += len(new)
            c["turns with several albums (union)"] += sum(len(tp) > 1 for tp in new)
            c["turns using place_any"] += sum(any(isinstance(a, dict) and a.get("place_any") for a in tp) for tp in new)
            for k in changed:
                o = old[k] if k < len(old) else []
                n = new[k]
                c["  ... questions changed"] += qs(o) != qs(n)
                meta = lambda al: json.dumps(sorted((m, json.dumps(a.get(m), sort_keys=True)) for a in al if isinstance(a, dict) for m in MK if a.get(m)))
                c["  ... time/place changed"] += meta(o) != meta(n)
            c["new question texts"] += len(qs([a for tp in new for a in tp]) - qs([a for tp in old for a in tp]))
    for k, v in c.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
