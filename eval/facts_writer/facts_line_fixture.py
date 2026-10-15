"""RESULTS 61d: golden fixture for the Swift port of findpics.facts_line (FindPicsCore FactsLine.swift, FactsLineTests).
Cases = hand-written edge cases + real 2B-student lines on public DEV photos (61b final, cross-fit; every line the rule
cuts in a seeded sample + as many it leaves alone). Each case: text, expected scan (cut, end in Unicode scalars / Python
str indices) or null, kept line, stopped flag, and the streamed stop: the first prefix length at which the rule fires.
  python eval/facts_writer/facts_line_fixture.py -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/facts_line.json
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
from findpics.facts_line import apply_repeat_stop, repeat_scan  # noqa: E402

HAND = [
    "", "objects: kilt", "objects: kilt, kilt", "objects: kilt, kilt,", "objects: kilt, Kilt.;", "objects: kilt,  KILT ,x",
    "scene: x; objects: kilt, kilt, kilt, kil", "scene: x; objects: a, b; colours: a, b, a; people: 2",
    "scene: beach, beach; objects: a", "objects: a, b; objects: a, b;", "objects: a,, b, , a, c", "a, b, a, c",
    "text: 'Time: 10, Time: 10, x'", "people: 2 - man in red, man in red, child", "objects: chair; chair; chair;",
    "objects: café, CAFÉ, x", "objects: tree.., tree, x", "objects: tree . ., tree ., x", "Objects:kilt,kilt,",
    "objects: kilt\n, kilt\t,", "1 objects: a, a, b", "objects : a, a,", "scene: a b: c, a b: c,",
]


def case(t):
    s = repeat_scan(t)
    kept, stopped = apply_repeat_stop(t)
    first = next((n for n in range(len(t) + 1) if repeat_scan(t[:n]) is not None), None)
    return dict(text=t, scan=None if s is None else list(s), kept=kept, stopped=stopped, streamStop=first)


def main():
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    lines = []
    for u in sorted(sp["dev"]):
        f = Path("eval/facts_writer/facts/t2b_final_xfit") / f"{u}.jsonl"
        lines += [json.loads(x)["facts"]["caption"] for x in open(f)]
    rnd = random.Random(0); rnd.shuffle(lines)
    cut = [t for t in lines if repeat_scan(t) is not None][:150]
    kept = [t for t in lines if repeat_scan(t) is None][:150]
    cases = [case(t) for t in HAND + cut + kept]
    out = Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/facts_line.json")
    out.write_text(json.dumps(dict(cases=cases), ensure_ascii=False, indent=0) + "\n")
    print("FIXTURE", len(cases), "cases,", sum(c["stopped"] for c in cases), "stopped ->", out)


if __name__ == "__main__":
    main()
