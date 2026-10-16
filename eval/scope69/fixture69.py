"""RESULTS 69: golden fixtures for the DRY / ON scope sources in FindPicsCore CascadeRoute.swift (dryAnswer, online) from
the R69 harness recordings (eval/scope69/cascade69.py --rec). Same format as eval/cascade_port/fixture.py.
  python eval/scope69/fixture69.py [--bundle-only]
  -> data/public/scope69/full/<split>/<library>.json   every recorded (request, config) (not in git; Swift reads it with
                                                       FP_CASCADE69_FULL=<that dir>)
  -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/cascade69.json   small stratified subset (always tested)"""
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cascade_port"))
from fixture import compact, features  # noqa: E402

R = Path("data/public/scope69/record")
FULL = Path("data/public/scope69/full")
FIX = Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/cascade69.json")
CFGS = ("AJ_bf16", "DRY_bf16", "ON_bf16", "DRY_fp32", "ON_fp32")


def main():
    recs, libs = {}, {}
    for split in ("dev", "train"):
        for f in sorted((R / split).glob("lib_*.json")):
            L = json.loads(f.read_text()); libs[L["user"]] = L
        for cfg in CFGS:
            for f in sorted((R / split / cfg).glob("rec*.jsonl.gz")):
                for line in gzip.open(f, "rt"):
                    r = json.loads(line); r["split"] = split
                    recs[(r["qid"], cfg)] = r     # requeued shards re-append: the last line wins
    print(len(recs), "recorded (request, config);", len(libs), "libraries")
    by_lib = defaultdict(list)
    for (q, cfg), r in sorted(recs.items()):
        by_lib[(r["split"], r["user"])].append(compact(r))
    for (split, u), rs in ([] if "--bundle-only" in sys.argv else by_lib.items()):
        d = FULL / split; d.mkdir(parents=True, exist_ok=True)
        (d / f"{u}.json").write_text(json.dumps(dict(user=u, place=libs[u]["place"], requests=rs)))
    rng = random.Random(69)
    keys = sorted(k for k in recs if not k[1].startswith("AJ")); rng.shuffle(keys)
    size = {k: len(json.dumps(recs[k])) for k in keys}
    keys.sort(key=lambda k: size[k])
    pick, cover = [], defaultdict(int)
    for k in keys:
        fs = features(recs[k]) | {"cfg_" + k[1]}
        if recs[k]["scope"]:     # a judge leaf under this scope source (and for DRY, its dry pass A)
            fs.add("judge_leaf_" + k[1])
        if size[k] < 300_000 and any(cover[f] < 2 for f in fs):
            pick.append(k)
            for f in fs:
                cover[f] += 1
    used = sorted({recs[k]["user"] for k in pick})
    FIX.write_text(json.dumps(dict(libs={u: libs[u]["place"] for u in used}, requests=[compact(recs[k]) for k in pick],
                                   prefix=[])))
    print(f"bundle fixture: {len(pick)} (request, config) from {len(used)} libraries, {FIX.stat().st_size / 1e6:.1f} MB; "
          f"features {dict(sorted(cover.items()))}")


if __name__ == "__main__":
    main()
