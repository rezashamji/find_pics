"""RESULTS 68 (2): golden fixtures for FindPicsCore CascadeRoute.swift from the recorder's logs (eval/cascade_port/record65.py).
  python eval/cascade_port/fixture.py
  -> data/public/cascade_port/full/<split>/<library>.json   every recorded (request, config) of that library (not in git;
                                                            the Swift test reads them when FP_CASCADE_FULL=<that dir>)
  -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/cascade.json   a small stratified subset (always tested) + factsPrefix
     cases (place, taken -> prefix, built exactly as eval/facts_scorer/data.py --meta=1)
Also prints the recorder's reproduction check: returned sets vs RESULTS 65's runs of the same configs."""
import gzip
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

R = Path("data/public/cascade_port/record")
FULL = Path("data/public/cascade_port/full")
FIX = Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/cascade.json")
CFGS = ("0_N_V", "25_N_V", "100_N_A")


def compact(r):
    jp = [dict(r=a, q=b, c=c, f=d, ref=e, p=p) for a, b, c, d, e, p in r["jp"]]
    return dict(qid=r["qid"], user=r["user"], cat=r["cat"], cfg=r["cfg"], error=r["error"], subs=r["subs"], A=r["A"],
                B=r["B"], tv=r["tv"], vec=r["vec"], hasf=r["hasf"], jp=jp,
                judged=[dict(r=a, q=b, c=c) for a, b, c in r["judged"]], ext=r["ext"],
                scope={k: [dict(r=a, ms=b, f=c) for a, b, c in v] for k, v in r["scope"].items()})


def features(r):
    f = set()
    if r["ext"] and any(len(r["scope"].get(k, [])) > 0 and v > max(50, -(-int(.05 * len(r["scope"][k]) * 1000) // 1000))
                        for k, v in r["ext"].items()):
        f.add("extension")
    if any(c["ref"] for c in r["B"]):
        f.add("reference")
    if any(c["excl"] for c in r["B"]):
        f.add("exclude")
    if any(c["sub"] >= 0 for c in r["B"]):
        f.add("subset")
    if any(re.search(r"(?i)\btrip|vacation|taken (in|at|during|on)\b", c["q"]) for c in r["B"]):
        f.add("trip")
    if any(re.search(r"(?i)\bor\b|\band also\b|\?\s+[A-Z]", c["q"]) for c in r["B"]):
        f.add("parts")
    if r["error"]:
        f.add("error")
    f.add("cat_" + r["cat"]); f.add("cfg_" + r["cfg"])
    return f


def main():
    recs = {}
    libs = {}
    for split in ("dev", "train"):
        for f in sorted((R / split).glob("lib_*.json")):
            L = json.loads(f.read_text()); libs[L["user"]] = L
        for cfg in CFGS:
            for f in sorted((R / split / cfg).glob("rec*.jsonl.gz")):
                for line in gzip.open(f, "rt"):
                    r = json.loads(line); r["split"] = split
                    recs[(r["qid"], cfg)] = r     # requeued shards re-append: the last line wins
    print(len(recs), "recorded (request, config);", len(libs), "libraries")
    # reproduction check vs RESULTS 65 (same configs, same frozen code)
    same = tot = 0
    for split in ("dev", "train"):
        for cfg in CFGS:
            old = {}
            for f in (Path("eval/partial65/replay") / split / cfg).glob("part*.json"):
                old.update({x["qid"]: x for x in json.loads(f.read_text())})
            for f in (R / split / cfg).glob("part*.json"):
                for x in json.loads(f.read_text()):
                    o = old.get(x["qid"]); tot += 1
                    fin = lambda y: [t["final"]["albums"] if t["final"] else None for t in y["turns"]]  # noqa: E731
                    same += o is not None and fin(o) == fin(x) and bool(o["error"]) == bool(x["error"])
    print(f"reproduction: returned sets identical to RESULTS 65 for {same}/{tot} (request, config)")
    by_lib = defaultdict(list)
    for (q, cfg), r in sorted(recs.items()):
        by_lib[(r["split"], r["user"])].append(compact(r))
    for (split, u), rs in by_lib.items():
        d = FULL / split; d.mkdir(parents=True, exist_ok=True)
        (d / f"{u}.json").write_text(json.dumps(dict(user=u, place=libs[u]["place"], requests=rs)))
    # small stratified subset for the bundle: cover every feature, smallest requests first, <= 2 per feature
    rng = random.Random(68)
    keys = sorted(recs); rng.shuffle(keys)
    size = {k: len(json.dumps(recs[k])) for k in keys}
    keys.sort(key=lambda k: size[k])
    pick, cover = [], defaultdict(int)
    for k in keys:
        fs = features(recs[k])
        if size[k] < 400_000 and any(cover[f] < 2 for f in fs):
            pick.append(k)
            for f in fs:
                cover[f] += 1
    used = sorted({recs[k]["user"] for k in pick})
    # factsPrefix cases: 1,500 photos of the recorded libraries (seeded) + hand cases
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path", "place", "taken"])
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    it = it[it.user.isin(list(libs))].sample(1500, random_state=68)
    t_ = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    pre = []
    for pl, tk in zip(it.place, t_):
        parts = [p_.strip() for p_ in str(pl or "").split(",") if p_.strip() and not re.search(r"\d", p_)]
        parts = list(dict.fromkeys(parts))
        m = tk.strftime("%B %Y") if not pd.isna(tk) else None
        s = (f"place: {', '.join(parts)}. " if parts else "") + (f"taken: {m}. " if m else "")
        pre.append(dict(place=pl, taken=None if pd.isna(tk) else tk.timestamp(), prefix=s))
    for pl in ["", "12 Main St, Springfield, Springfield, USA", "Zürich, Zürich, Schweiz", " , ,", "Kyōto, 京都府, 日本",
               "Paris 75001, Paris, France"]:
        pre.append(dict(place=pl, taken=None, prefix=(lambda parts: f"place: {', '.join(parts)}. " if parts else "")(
            list(dict.fromkeys(p_.strip() for p_ in pl.split(",") if p_.strip() and not re.search(r"\d", p_))))))
    FIX.write_text(json.dumps(dict(libs={u: libs[u]["place"] for u in used}, requests=[compact(recs[k]) for k in pick],
                                   prefix=pre)))
    print(f"bundle fixture: {len(pick)} (request, config) from {len(used)} libraries, {FIX.stat().st_size / 1e6:.1f} MB; "
          f"features {dict(sorted(cover.items()))}")


if __name__ == "__main__":
    main()
