"""Does a stricter oracle cut fix precision? For every labelled photo (audit/ + audit2/), the program's visual score
under v3 = min over need P(yes), max over need_any, min over not of 1-P, min over person_need; metadata/anchor/person
constraints must hold. Sweep the cut; count labelled matches kept vs labelled non-matches kept, per category.
Usage: python eval/complex_bench/cut_sweep.py"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from build import Lib
from common import ROOT, load_items
from audit_score import labels, load

def score_vec(lib, a, cat):
    m = lib.meta_mask(a).astype(float)
    sc = np.ones(len(lib.ids))
    P = lambda q: np.array([lib.O.get(q.strip(), {}).get(i, 0.0) for i in lib.ids])
    for q in a.get("need") or []: sc = np.minimum(sc, P(q))
    if a.get("need_any"): sc = np.minimum(sc, np.max([P(q) for q in a["need_any"]], axis=0))
    for q in a.get("not") or []: sc = np.minimum(sc, 1 - P(q))
    if a.get("person"):
        ok = np.isin(lib.ids, list(lib.persons.get(a["person"], {})))
        m *= ok
        for q in a.get("person_need") or []:
            sc = np.minimum(sc, np.array([lib.O.get(f"PERSON[{a['person']}]::{q.strip()}", {}).get(i, 0.0) for i in lib.ids]))
    anc = a.get("anchor")
    if isinstance(anc, dict) and anc.get("window"):
        rows = [lib.pos[str(i)] for i in anc.get("ids") or [] if str(i) in lib.pos]
        if rows: m *= lib.window(rows, anc["window"])
    if cat == "f" and a.get("ids"):
        sc = np.where(np.isin(lib.ids, [str(i) for i in a["ids"]]), 1.0, sc)
    return np.where(m > 0, sc, -1.0)

def main():
    it = load_items(); V = load("queries")
    L = {**{("1",) + k: v for k, v in labels("audit").items()}, **{("2",) + k: v for k, v in labels("audit2").items()}}
    keys = {"1": json.loads((ROOT / "audit/key.json").read_text())}
    if (ROOT / "audit2/key.json").exists(): keys["2"] = json.loads((ROOT / "audit2/key.json").read_text())
    rows = []
    libs = {}
    for a_, K in keys.items():
        for name, k in K.items():
            q = V.get(k["qid"])
            if q is None or k["cat"] == "g": continue
            u = k["user"]
            if u not in libs: libs.clear(); libs[u] = Lib(u, it)
            lib = libs[u]
            progs = q["program"][-1] if q["cat"] == "h" else q["program"]
            S = np.max([score_vec(lib, a, q["cat"]) for a in progs if isinstance(a, dict)], axis=0)
            for x in k["items"]:
                lab = L.get((a_, name, str(x["n"])))
                if lab in ("match", "no_match"):
                    rows.append((k["cat"], float(S[lib.pos[x["item_id"]]]), lab == "match"))
    out = ["cut | " + " | ".join(f"{c}: matches kept / non-matches kept" for c in "abcdefh")]
    for cut in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99):
        r = []
        for c in "abcdefh":
            R = [x for x in rows if x[0] == c]
            tm = sum(x[2] for x in R); tn = len(R) - tm
            km = sum(1 for x in R if x[2] and x[1] >= cut); kn = sum(1 for x in R if not x[2] and x[1] >= cut)
            r.append(f"{km}/{tm} {kn}/{tn}")
        out.append(f"{cut} | " + " | ".join(r))
    (ROOT / "audit" / "cut_sweep.txt").write_text("\n".join(out)); print("\n".join(out))

if __name__ == "__main__":
    main()
