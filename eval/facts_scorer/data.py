"""RESULTS 59, data: per-photo FACTS text + (question, photo, oracle P) pairs for TRAIN and DEV libraries only.

The complex_bench TEST split is never read here (its users are skipped before any file is opened).
  python eval/facts_scorer/data.py [--facts=eval/complex_bench/facts] [--tag=27b]
  -> eval/facts_scorer/data/facts_<tag>.parquet (user, split, item_id, text)
     eval/facts_scorer/data/pairs_{train,dev}.parquet (user, item_id, q, qs, p, person): q = oracle key, qs = the text
     the scorer reads ("PERSON[x]::" removed, "the person in the red box" -> "a person": the facts have no box).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

CB = Path("eval/complex_bench")
OUT = Path("eval/facts_scorer/data")


def splits():
    sp = json.loads((CB / "splits.json").read_text())
    return {u: s for s in ("train", "dev") for u in sp[s]}     # test deliberately absent


def _s(x):
    if isinstance(x, dict):
        return ", ".join(f"{v}" for v in x.values() if isinstance(v, (str, int, float)) and str(v).strip())
    return str(x).strip()


def _lst(v):
    if v is None:
        return []
    if isinstance(v, (str, dict)):
        v = [v]
    return [_s(x) for x in v if _s(x) and _s(x).lower() not in ("none", "n/a")]


def facts_text(f) -> str:
    """One photo's facts as plain text, fixed field order (what the scorer reads; ~150-250 word pieces)."""
    if not f:
        return "no description"
    parts = []
    if f.get("caption"):
        parts.append(_s(f["caption"]))
    for k, name in (("image_type", "type"), ("main_subject", "subject"), ("setting", "setting")):
        if f.get(k):
            parts.append(f"{name}: {_s(f[k]).replace('_', ' ')}")
    for k in ("scene", "objects", "animals", "depictions"):
        L = _lst(f.get(k))
        if L:
            parts.append(f"{k}: " + ", ".join(L))
    pc = f.get("people_count")
    P = _lst(f.get("people"))
    if isinstance(pc, (int, float)) or P:
        parts.append(f"people: {pc if isinstance(pc, (int, float)) else '?'}" + ("; " + "; ".join(P) if P else ""))
    for k in ("activities",):
        L = _lst(f.get(k))
        if L:
            parts.append(f"{k}: " + ", ".join(L))
    if f.get("text"):
        parts.append(f"text: {_s(f['text'])}")
    L = _lst(f.get("colours"))
    if L:
        parts.append("colours: " + ", ".join(L))
    for k, name in (("notable_colour", "notable colour"), ("lighting", "lighting")):
        if f.get(k):
            parts.append(f"{name}: {_s(f[k]).replace('_', ' ')}")
    for k in ("weather", "quality", "style"):
        L = [x.replace("_", " ") for x in _lst(f.get(k))]
        if L:
            parts.append(f"{k}: " + ", ".join(L))
    for k in ("event", "landmark"):
        if f.get(k):
            parts.append(f"{k}: {_s(f[k])}")
    return ". ".join(parts)


def scorer_question(q: str) -> str:
    q = q.split("::", 1)[1] if q.startswith("PERSON[") else q
    q = re.sub(r"\b[Tt]he person in the red box\b", "a person", q)
    q = re.sub(r"\s*\bin the red box\b", "", q)
    if __import__("os").environ.get("FP_R59_QNORM") == "1":
        # (10-14, from the TRAIN-library eye sets) the product's grounding rewrites every plain object question to
        # "Is there a real X anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?",
        # a form no training question has (0 of 1,189; "Is there a real X in this photo?" 33): read it in that form.
        q = re.sub(r"\s*\(not a drawing, painting, statue, toy, model or picture of one\)", "", q)
        q = re.sub(r"\banywhere (in|visible in)\b", r"\1", q)
    return q.strip()


LIST_KEYS = ("scene", "objects", "animals", "depictions", "people", "activities", "colours", "weather", "quality", "style")
STR_KEYS = ("caption", "image_type", "main_subject", "setting", "text", "notable_colour", "lighting", "event", "landmark")


def _dedupe_phrase(s: str, max_parts=12) -> str:
    """'a cap, a cap, a cap ...' (a looping small model) -> 'a cap'."""
    parts, seen = [], set()
    for x in re.split(r",\s*", s):
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k); parts.append(x.strip())
    return ", ".join(parts[:max_parts])


def salvage(raw):
    """Facts from a truncated / slightly invalid JSON answer (small models: loops cut at the token cap, trailing
    commas): every field read on its own, list items de-duplicated (max 30), looping strings collapsed."""
    if not raw or "{" not in raw:
        return None
    out = {}
    for k in STR_KEYS:
        m = re.search(rf'"{k}"\s*:\s*"([^"]*)', raw)
        if m:
            out[k] = _dedupe_phrase(m.group(1))
    for k in LIST_KEYS:
        m = re.search(rf'"{k}"\s*:\s*(\[[^\]]*\]?|"[^"]*"?)', raw, re.S)
        if m:
            items = re.findall(r'"([^"]+)"', m.group(1))
            seen, keep = set(), []
            for x in items:
                x = _dedupe_phrase(x)
                if x.lower() not in seen:
                    seen.add(x.lower()); keep.append(x)
            out[k] = keep[:30]
    m = re.search(r'"people_count"\s*:\s*(\d+)', raw)
    if m:
        out["people_count"] = int(m.group(1))
    return out or None


def load_facts_dir(d: Path, user: str, salvage_bad: bool = False) -> dict:
    out = {}
    f = d / f"{user}.jsonl"
    if not f.exists():
        return out
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fx = r.get("facts")
        if fx is None and salvage_bad:
            fx = salvage(r.get("raw"))
        if fx is not None or r["item_id"] not in out:
            out[str(r["item_id"])] = fx
    return out


def main():
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    fdir = Path(arg("facts", str(CB / "facts")))
    tag = arg("tag", "27b")
    OUT.mkdir(parents=True, exist_ok=True)
    S = splits()
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path"])
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    rows = []
    for u, s in S.items():
        F = load_facts_dir(fdir, u, salvage_bad=arg("salvage", "0") == "1")
        for iid in it.item_id[it.user == u]:
            rows.append((u, s, iid, facts_text(F.get(iid)), F.get(iid) is not None))
    T = pd.DataFrame(rows, columns=["user", "split", "item_id", "text", "has_facts"])
    T.to_parquet(OUT / f"facts_{tag}.parquet")
    print(f"facts_{tag}: {len(T)} photos, {T.has_facts.sum()} with facts; by split {T.groupby('split').size().to_dict()}")
    if arg("pairs", "1") == "1":
        for s in ("train", "dev"):
            P = []
            for u in [u for u, x in S.items() if x == s]:
                O = pd.read_parquet(CB / "oracle" / f"{u}.parquet")
                O["item_id"] = O.item_id.astype(str)
                O["user"] = u
                P.append(O)
            P = pd.concat(P, ignore_index=True)
            P["person"] = P.q.str.startswith("PERSON[")
            uq = {q: scorer_question(q) for q in P.q.unique()}
            P["qs"] = P.q.map(uq)
            P.to_parquet(OUT / f"pairs_{s}.parquet")
            print(f"pairs_{s}: {len(P)} pairs, {P.q.nunique()} questions, {(P.p >= .8).sum()} >= .8, {(P.p < .2).sum()} < .2")


if __name__ == "__main__":
    main()
