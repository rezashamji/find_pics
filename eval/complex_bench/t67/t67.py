"""RESULTS 67: category c truth rewrite - compile versions, blind eye sample, score (rules: t67/PREREG.txt, fec56c8).

  python eval/complex_bench/t67/t67.py build    # after regen.py: versions T/T9/R5/R9 per (request, photo), drops
  python eval/complex_bench/t67/t67.py sample   # strata, seed-67 draw, sheets, requests.txt, key.json (NOT read while labelling)
  python eval/complex_bench/t67/t67.py score    # after labels_main.txt is committed: estimates, decision, t67/queries/
Writes only under eval/complex_bench/t67/. Dev + TRAIN only.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "t63"))
from common import ROOT, city, load_items  # noqa: E402

T67 = ROOT / "t67"
VERS = ("T", "T9", "R5", "R9")
TAKE_PAT, TAKE_OUT = 12, dict(OD=20, O1=12, O2=8, O3=8)
MAX_FRAC = 0.25


def c_truth():
    out = {}
    for s in ("dev", "train"):
        for q in json.loads((ROOT / "t63" / "queries" / f"{s}.json").read_text())["queries"]:
            if q["cat"] == "c":
                out[q["qid"]] = q
    return out


def cmd_build():
    import build
    it = load_items()
    Q = c_truth()
    by_user = defaultdict(list)
    for q in Q.values():
        by_user[q["user"]].append(q)
    rows, drops, progs = [], [], {}
    for u, qs in sorted(by_user.items()):
        lib = build.Lib(u, it)
        f = T67 / "oracle" / f"{u}.parquet"
        if f.exists():
            O2 = pd.read_parquet(f)
            for qq, g in O2.groupby("q"):
                lib.O[qq] = dict(zip(g.item_id.astype(str), g.p))
        pf = T67 / "oracle" / f"{u}_persons.json"
        if pf.exists():
            for nm, d in json.loads(pf.read_text()).items():
                lib.persons.setdefault(nm, d)
        X = json.loads((T67 / "programs_c2" / f"{u}.json").read_text())
        n = len(lib.ids)
        for q in qs:
            qid = q["qid"]
            T = set(map(str, q["answers"][-1][0]))
            lib.cut = 0.9
            m9, _ = lib.album(q["program"][0], "c")      # old program, strict place matcher as built (cat c)
            T9 = T & set(lib.ids[m9])
            sets = dict(T=T, T9=T9)
            x = X.get(qid) or {}
            for v, cut in (("R5", 0.5), ("R9", 0.9)):
                why = None
                if not x.get("ok"):
                    why = "rewrite failed to parse"
                elif x.get("well_defined") is False:
                    why = "well_defined=false: " + str(x.get("why"))[:160]
                else:
                    lib.cut = cut
                    try:
                        # album semantics of build.py; cat "b" only to get the RESULTS 52/56 place matcher (place_any)
                        mk, pb = lib.album(x["albums"][0], "b")
                    except KeyError as e:
                        mk, pb = None, [f"error: {e}"]
                    pb = [p for p in pb if p.startswith(("person", "error"))]
                    s = set(lib.ids[mk]) if mk is not None else set()
                    if pb:
                        why = "; ".join(pb)
                    elif not s:
                        why = "empty album"
                    elif len(s) > MAX_FRAC * n:
                        why = f"answer too large ({len(s)} of {n})"
                    sets[v] = s
                if why:
                    drops.append(dict(qid=qid, version=v, reason=why))
                    sets[v] = set()
                progs[qid] = x.get("albums")
            for i in set().union(*sets.values()):
                rows.append((qid, i, *[int(i in sets[v]) for v in VERS]))
    V = pd.DataFrame(rows, columns=["qid", "item_id", *VERS])
    V.to_parquet(T67 / "versions.parquet")
    (T67 / "drops.json").write_text(json.dumps(drops, indent=1))
    (T67 / "programs_c2_flat.json").write_text(json.dumps(progs, indent=1))
    g = V.merge(pd.DataFrame([(k, q["split"]) for k, q in Q.items()], columns=["qid", "split"]), on="qid")
    lines = [f"requests {len(Q)}; drops: " + ", ".join(f"{v} {sum(d['version'] == v for d in drops)}" for v in ("R5", "R9"))]
    for sp, gg in g.groupby("split"):
        lines.append(f"{sp}: units " + ", ".join(f"{v} {int(gg[v].sum())}" for v in VERS))
    for d in drops:
        lines.append(f"  DROP {d['version']} {d['qid']} ({' || '.join(Q[d['qid']]['turns'])}): {d['reason']}")
    (T67 / "build.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def units():
    """Universe (t63 direct-judge units of each c request, album 0) + version units, with strata."""
    from t63 import load_direct
    Q = c_truth()
    D = load_direct("d27")
    D = D[D.qid.isin(set(Q)) & (D.album == 0)][["qid", "item_id", "p"]].rename(columns={"p": "pD"})
    V = pd.read_parquet(T67 / "versions.parquet")
    U = D.merge(V, on=["qid", "item_id"], how="outer")
    for v in VERS:
        U[v] = U[v].fillna(0).astype(int)
    inany = U[list(VERS)].sum(axis=1) > 0
    pat = "P" + U["T"].astype(str) + U["T9"].astype(str) + U["R5"].astype(str) + U["R9"].astype(str)
    out = np.where(U.pD >= 0.5, "OD", np.where(U.pD >= 0.2, "O1", np.where(U.pD >= 0.05, "O2", "O3")))
    U["s"] = np.where(inany, pat, out)
    U["split"] = U.qid.map(lambda k: Q[k]["split"]); U["user"] = U.qid.map(lambda k: Q[k]["user"])
    return U


def person_of(qid, Q, P2):
    a = Q[qid]["program"][0]
    r = (P2.get(qid) or [{}])[0] or {}
    return r.get("person") or a.get("person")


def cmd_sample():
    from audit import sheet, tile, font   # noqa: E402
    from PIL import Image, ImageDraw
    U = units()
    Q = c_truth()
    P2 = json.loads((T67 / "programs_c2_flat.json").read_text())
    seen = {(k["qid"], str(k["item_id"])) for k in json.loads((ROOT / "t63" / "key.json").read_text())}
    rng = random.Random(67)
    pick = []
    for s in sorted(U.s.unique()):
        pool = [j for j in U.index[U.s == s] if (U.at[j, "qid"], U.at[j, "item_id"]) not in seen]
        rng.shuffle(pool)
        pick += pool[: (TAKE_OUT[s] if s in TAKE_OUT else TAKE_PAT)]
    U["eye"] = 0; U.loc[pick, "eye"] = 1
    U.to_parquet(T67 / "units.parquet")
    E = U[U.eye == 1].copy()
    qorder = sorted(E.qid.unique()); random.Random(7).shuffle(qorder)
    it = load_items().set_index("item_id")
    # faces for named-person units + reference crops
    need = defaultdict(set)
    for r in E.itertuples():
        nm = person_of(r.qid, Q, P2)
        if nm:
            need[Q[r.qid]["user"]].add(nm)
    boxed, refs = {}, {}
    if need:
        from findpics import store
        from findpics.engine import _boxed, reference_crop
        from oracle import person_matches
        idx = store.load("data/public/index_disbench", clip_model="hf-hub:timm/PE-Core-L-14-336")
        user_of = idx.items.path.str.extract(r"/images/images/([^/]+)/")[0].to_numpy()
        for u, names in need.items():
            sub = store.subset(idx, np.where(user_of == u)[0])
            ids = sub.items.item_id.astype(str).to_numpy()
            meta = json.loads((ROOT / "libraries" / f"{u}.json").read_text())
            pm = person_matches(sub, meta)
            G = [g for g in meta.get("face_groups", []) if g["n_items"] >= 25][:2]
            want = set(E[E.user == u].item_id)
            for nm in names:
                if nm not in pm:
                    continue
                prow, frow, _ = pm[nm]
                for r_, f_ in zip(prow, frow):
                    if ids[r_] in want:
                        boxed[(nm, ids[r_])] = _boxed(sub, Image.open(sub.items.path.iloc[r_]).convert("RGB"), int(f_))
                gi = ["Sam", "Alex"].index(nm)
                refs[(u, nm)] = [reference_crop(sub, int(fr)).convert("RGB") for fr in [G[gi]["rep"]] + G[gi]["face_rows"][:2]]
    (T67 / "sheets").mkdir(exist_ok=True); (T67 / "boxed").mkdir(exist_ok=True)
    TILE = 500
    key, req_lines, n, sh = [], [], 0, 0
    for qid in qorder:
        q = Q[qid]
        rows = list(E[E.qid == qid].index); random.Random(hash(qid) % 1000).shuffle(rows)
        nm = person_of(qid, Q, P2)
        tiles = []
        if nm and (q["user"], nm) in refs:
            for im in refs[(q["user"], nm)]:
                k_ = 300 / max(im.size)
                im = im.resize((max(1, int(im.width * k_)), max(1, int(im.height * k_))), Image.LANCZOS)
                c = Image.new("RGB", (TILE, TILE + 46), "white"); c.paste(im, ((TILE - im.width) // 2, 0))
                d = ImageDraw.Draw(c); d.rectangle([0, 0, 44, 30], fill=(255, 120, 0))
                d.text((6, 4), "R", fill="black", font=font(20)); d.text((6, TILE + 4), f"reference: {nm}", fill="black", font=font(15))
                tiles.append(c)
        first = n + 1
        for j in rows:
            n += 1
            iid = U.at[j, "item_id"]; p = it.loc[iid]
            path = p.path
            if nm and (nm, iid) in boxed:
                path = T67 / "boxed" / f"{iid}_{nm}.jpg"; boxed[(nm, iid)].save(path, quality=95)
            tiles.append(tile(path, str(n), f"{p.t:%Y-%m-%d %a %H:%M} | {city(p.place)}"))
            key.append(dict(n=n, qid=qid, split=q["split"], item_id=iid, stratum=U.at[j, "s"],
                            **{v: int(U.at[j, v]) for v in VERS}, pD=None if pd.isna(U.at[j, "pD"]) else float(U.at[j, "pD"])))
        names = []
        for s in range(0, len(tiles), 4):   # 2 x 2 per sheet: each photo at its native pixels
            f = f"s{sh:03d}.jpg"; sheet(tiles[s:s + 4], cols=2).save(T67 / "sheets" / f, quality=92); names.append(f); sh += 1
        prs = f" [{nm} = the person in the red box; tiles R = reference crops of {nm}]" if nm else ""
        req_lines.append(f"photos {first}-{n} on {', '.join(names)} (today {q['today']}): {' || '.join(q['turns'])}{prs}")
    (T67 / "key.json").write_text(json.dumps(key, indent=0))
    (T67 / "requests.txt").write_text("\n".join(req_lines) + "\n")
    print(len(key), "eye units on", sh, "sheets;", len(qorder), "requests")
    print(E.groupby("s").size().to_string())
    print(U.groupby("s").size().to_string())


def read_labels():
    L = {}
    for line in open(T67 / "labels_main.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split(None, 2)
        L[int(p[0])] = p[1].strip().lower()
    return L


def cmd_score():
    from t63 import estimate, boot
    U = pd.read_parquet(T67 / "units.parquet")
    key = json.loads((T67 / "key.json").read_text())
    lab = read_labels()
    miss = [k["n"] for k in key if k["n"] not in lab]
    if miss:
        raise SystemExit(f"unlabelled {miss[:10]} ({len(miss)})")
    Q = c_truth()
    drops = json.loads((T67 / "drops.json").read_text())
    dropped = {v: {d["qid"] for d in drops if d["version"] == v} for v in ("R5", "R9")}
    out = ["RESULTS 67: category c truth rewrite (rules t67/PREREG.txt, pre-registered fec56c8)", ""]
    unsure = sum(lab[k["n"]] == "unsure" for k in key)
    out.append(f"eye units {len(key)}, unsure {unsure} (excluded)")

    def est(vers, keep):
        Uk = U[U.qid.isin(keep)]
        N = Uk.groupby("s").size().to_dict()
        recs = [(k["stratum"], int(lab[k["n"]] == "match")) for k in key if k["qid"] in keep and lab[k["n"]] != "unsure"]
        M = defaultdict(lambda: [0, 0])
        for s, m in recs:
            M[s][0] += m; M[s][1] += 1
        M = {s: tuple(v) for s, v in M.items()}
        ins = {s for s in N if s.startswith("P") and s[1 + VERS.index(vers)] == "1"}
        nolab = sum(n for s, n in N.items() if s not in M)
        P, R, TP, FP, FN = estimate(N, M, ins)
        (pl, ph), (rl, rh) = boot(N, recs, ins)
        f1 = 2 * P * R / (P + R) if P + R else 0
        return dict(P=P, R=R, F1=f1, Pci=(pl, ph), Rci=(rl, rh), TP=TP, FP=FP, FN=FN, nolab=nolab, M=M, N=N)

    def fmt(name, e):
        return (f"  {name:4s} precision {e['P']:.2f} [{e['Pci'][0]:.2f}, {e['Pci'][1]:.2f}]  recall {e['R']:.2f} "
                f"[{e['Rci'][0]:.2f}, {e['Rci'][1]:.2f}]  F1 {e['F1']:.3f}  est right/wrong/missed {e['TP']:.0f}/{e['FP']:.0f}/{e['FN']:.0f}"
                + (f"  (units in unlabelled strata {e['nolab']})" if e["nolab"] else ""))
    allq = set(Q)
    out.append(f"\n(i) all {len(allq)} requests:")
    E_all = {v: est(v, allq) for v in ("T", "T9")}
    for v, e in E_all.items():
        out.append(fmt(v, e))
    e0 = E_all["T"]
    out.append("  strata (N, eye match/labelled): " + "; ".join(f"{s} {e0['N'][s]} {e0['M'].get(s, (0, 0))[0]}/{e0['M'].get(s, (0, 0))[1]}"
                                                         for s in sorted(e0["N"])))
    res = {}
    for cand in ("R5", "R9"):
        keep = allq - dropped[cand]
        out.append(f"\n(ii) the {len(keep)} requests {cand} keeps ({len(dropped[cand])} dropped):")
        res[cand] = {v: est(v, keep) for v in VERS}
        for v, e in res[cand].items():
            out.append(fmt(v, e))
    c = "R9" if res["R9"]["R9"]["F1"] >= res["R5"]["R5"]["F1"] else "R5"
    e = res[c][c]; eT = res[c]["T"]
    ver = e["P"] >= 0.80 and e["R"] >= 0.70
    out.append(f"\nDECISION: candidate = {c} (F1 {e['F1']:.3f} vs {('R5' if c == 'R9' else 'R9')} "
               f"{res['R5' if c == 'R9' else 'R9']['R5' if c == 'R9' else 'R9']['F1']:.3f}); VERIFIED: "
               f"{'YES' if ver else 'no (stays PROVISIONAL)'} (P {e['P']:.2f}, R {e['R']:.2f}); "
               f"RECOMMEND ADOPT: {'YES' if e['F1'] > eT['F1'] else 'NO'} (candidate F1 {e['F1']:.3f} vs T {eT['F1']:.3f} on the same requests)")
    # eye agreement per request kind
    named = {k for k, q in Q.items() if q["program"][0].get("person")}
    for nmset, title in ((named, "named-person"), (allq - named, "other")):
        kk = [k for k in key if k["qid"] in nmset and lab[k["n"]] != "unsure"]
        for v in VERS:
            inn = [k for k in kk if k[v] == 1]
            out.append(f"  {title} requests, {v}: labelled in-version photos right {sum(lab[k['n']] == 'match' for k in inn)}/{len(inn)}")
    # write t67/queries
    V = pd.read_parquet(T67 / "versions.parquet")
    S = {qid: sorted(g.item_id[g[c] == 1]) for qid, g in V.groupby("qid")}
    P2 = json.loads((T67 / "programs_c2_flat.json").read_text())
    (T67 / "queries").mkdir(exist_ok=True)
    for sp in ("dev", "train"):
        F = json.loads((ROOT / "queries" / f"{sp}.json").read_text())
        keep, nd, nr, unk = [], 0, 0, []
        for q in F["queries"]:
            if q["cat"] == "c":
                if q["qid"] not in Q:
                    unk.append(q["qid"]); keep.append(q); continue
                if q["qid"] in dropped[c]:
                    nd += 1; continue
                q = dict(q, program=P2[q["qid"]], answers=[[S.get(q["qid"], [])]], verified=ver,
                         cut=0.9 if c == "R9" else 0.5, program_version="c2")
                nr += 1
            keep.append(q)
        F["queries"] = keep
        (T67 / "queries" / f"{sp}.json").write_text(json.dumps(F))
        out.append(f"t67/queries/{sp}.json: c rewritten {nr}, dropped {nd}, c not in t63 (kept as is) {unk}")
    (T67 / "report.txt").write_text("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    {"build": cmd_build, "sample": cmd_sample, "score": cmd_score}[sys.argv[1]]()
