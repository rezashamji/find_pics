"""RESULTS 64: write the facts line ONCE per near-duplicate group and share it (index-cost cut). Public bench only.

RULE G(s) (pre-registered, see JOURNAL 10-16 "RESULTS 64 PRE-REGISTRATION"): per library, photos in capture-time order;
a photo i may join the group of an earlier REPRESENTATIVE r iff
    both dated, 0 <= t_i - t_r <= 10 min (findpics.bursts.BURST_MINUTES), cosine(PE-Core-L-14-336 image vectors) >= s,
    and, when both have GPS, distance <= 1 km;
i joins the eligible representative with the highest cosine; if none is eligible i becomes a representative itself.
Leader grouping (every member is directly similar to the photo whose line it receives - no chaining), streamable at
index time in capture order. Undated photos always get their own line. A member keeps its OWN metadata prefix (place,
month: data.py --meta=1) and its own image vector; only the writer's line (the facts part) is copied from its rep.

  python eval/dedup64/group.py --s=0.96[,0.94,...]       -> eval/dedup64/data/groups_s<s>.parquet (user, split,
       item_id, rep_id, cos) + facts_shared_s<s>.parquet (same columns as the data.py facts parquet: user, split,
       item_id, text, has_facts), built from BOTH per-photo parquets (TRAIN rows from the cross-fit 61b 2B + v2 lines,
       dev rows from the product p2b_all + v2 lines), and prints writer calls saved per library.
  s = inf -> no groups (reproduction check: must reproduce the per-photo baseline).
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/dedup64/data"
D = ROOT / "eval/facts_scorer/data"
SRC = {"train": D / "facts_w_t2b_final_rs2_xfitm.parquet", "dev": D / "facts_w_p2b_devmix_rs2m.parquet"}
MINUTES = 10.0
KM = 1.0


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def load_items():
    sys.path.insert(0, str(ROOT / "src"))
    from findpics import store
    idx = store.load(ROOT / "data/public/index_disbench")
    C = idx.clip.astype(np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    first = pd.Series(np.arange(len(idx.units)), index=idx.units["item_row"].to_numpy()).groupby(level=0).first()
    V = np.zeros((len(idx.items), C.shape[1]), np.float32)
    V[first.index.to_numpy()] = C[first.to_numpy()]
    it = idx.items.copy()
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    t = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    it["tmin"] = ((t - pd.Timestamp(0, tz="UTC")).dt.total_seconds() / 60.0)
    return it, V


def km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def group_library(sub: pd.DataFrame, V: np.ndarray, s: float):
    """sub: one library's items (index rows in sub.index). Returns rep item_id per item and the cosine to it."""
    rep = dict(zip(sub.item_id, sub.item_id)); cs = dict(zip(sub.item_id, [1.0] * len(sub)))
    if not math.isfinite(s):
        return rep, cs
    d = sub[sub.tmin.notna()].sort_values(["tmin", "item_id"])
    reps = []          # (row, tmin, lat, lon, item_id) of representatives, in time order
    lo = 0
    for row, r in zip(d.index, d.itertuples()):
        while lo < len(reps) and r.tmin - reps[lo][1] > MINUTES:
            lo += 1
        best, bc = None, -1.0
        if lo < len(reps):
            act = reps[lo:]
            c = V[[a[0] for a in act]] @ V[row]
            for a, cv in zip(act, c):
                if cv < s or cv <= bc:
                    continue
                if all(isinstance(x, float) and math.isfinite(x) for x in (r.lat, r.lon, a[2], a[3])) and \
                        km(r.lat, r.lon, a[2], a[3]) > KM:
                    continue
                best, bc = a, float(cv)
        if best is None:
            reps.append((row, r.tmin, float(r.lat) if r.lat is not None else float("nan"),
                         float(r.lon) if r.lon is not None else float("nan"), r.item_id))
        else:
            rep[r.item_id] = best[4]; cs[r.item_id] = bc
    return rep, cs


def prefix_of(it: pd.DataFrame) -> dict:
    """data.py --meta=1's per-photo metadata prefix, recomputed exactly (checked against the parquet text)."""
    t_ = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    month = dict(zip(it.item_id, t_.dt.strftime("%B %Y")))
    out = {}
    for iid, pl in zip(it.item_id, it.place):
        parts = [p_.strip() for p_ in str(pl or "").split(",") if p_.strip() and not re.search(r"\d", p_)]
        parts = list(dict.fromkeys(parts))
        out[iid] = (f"place: {', '.join(parts)}. " if parts else "") + (
            f"taken: {month[iid]}. " if isinstance(month.get(iid), str) else "")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sp = json.loads((ROOT / "eval/complex_bench/splits.json").read_text())
    split_of = {u: k for k in ("train", "dev") for u in sp[k]}          # TEST libraries never touched
    it, V = load_items()
    it = it[it.user.isin(split_of)]
    pre = prefix_of(it)
    base = pd.concat([pd.read_parquet(SRC[k]).query("split == @k") for k in ("train", "dev")], ignore_index=True)
    base["item_id"] = base.item_id.astype(str)
    assert set(base.item_id) == set(it.item_id), "facts parquet and index disagree on the bench photos"
    bad = [i for i, t in zip(base.item_id, base.text) if not t.startswith(pre[i])]
    assert not bad, f"{len(bad)} prefixes do not reproduce data.py"
    body = {i: t[len(pre[i]):] for i, t in zip(base.item_id, base.text)}
    hasf = dict(zip(base.item_id, base.has_facts))
    for s in [float(x) for x in arg("s", "0.96").split(",")]:
        tag = "inf" if not math.isfinite(s) else f"{s:.2f}"
        G = []
        for u, sub in it.groupby("user"):
            rep, cs = group_library(sub, V, s)
            G += [(u, split_of[u], i, rep[i], cs[i]) for i in sub.item_id]
        G = pd.DataFrame(G, columns=["user", "split", "item_id", "rep_id", "cos"])
        G.to_parquet(OUT / f"groups_s{tag}.parquet")
        rep = dict(zip(G.item_id, G.rep_id))
        T = base.copy()
        T["text"] = [pre[i] + body[rep[i]] for i in T.item_id]
        T["has_facts"] = [bool(hasf[rep[i]]) for i in T.item_id]
        T.to_parquet(OUT / f"facts_shared_s{tag}.parquet")
        L = [f"G(s={tag}), {MINUTES:.0f} min, GPS <= {KM:.0f} km: writer calls saved per library = members / photos"]
        for k in ("train", "dev"):
            g = G[G.split == k]
            per = g.groupby("user").apply(lambda x: (len(x), int((x.item_id != x.rep_id).sum())), include_groups=False)
            n, m = sum(a for a, _ in per), sum(b for _, b in per)
            fr = sorted(b / a for a, b in per)
            L.append(f"  {k}: {m:,}/{n:,} = {m / n:.4f} saved pooled; per library median {np.median(fr):.4f}, "
                     f"min {fr[0]:.4f}, max {fr[-1]:.4f} ({len(per)} libraries)")
        print("\n".join(L))
        (OUT / f"saved_s{tag}.txt").write_text("\n".join(L) + "\n" + "\n".join(
            f"{u} {k} {a} {b} {b / a:.4f}" for k in ("train", "dev") for u, (a, b) in
            G[G.split == k].groupby("user").apply(lambda x: (len(x), int((x.item_id != x.rep_id).sum())),
                                                  include_groups=False).items()) + "\n")


if __name__ == "__main__":
    main()
