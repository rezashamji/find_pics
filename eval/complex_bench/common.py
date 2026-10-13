"""Shared helpers for the complex-query benchmark (public DISBench libraries only)."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("eval/complex_bench")
LISTS = ["scene", "objects", "animals", "depictions", "activities", "colours", "weather", "quality", "style"]
SCALARS = ["image_type", "main_subject", "setting", "lighting", "event", "landmark", "notable_colour"]


KEEP_S = {"jeans", "shorts", "pants", "glasses", "sunglasses", "christmas", "trousers", "scissors", "binoculars",
          "stairs", "clothes", "goggles", "headphones", "earphones", "leggings", "tights", "news", "bus", "gas", "grass",
          "glass", "dress", "cross", "moss", "chess", "lens", "series", "species", "canvas", "atlas", "cactus", "octopus",
          "walrus", "bonus", "status", "virus", "campus", "circus", "iris", "tennis", "chassis", "overalls"}


def norm(w) -> str:
    if isinstance(w, dict):
        w = " ".join(str(v) for v in w.values() if isinstance(v, str))
    w = re.sub(r"\s+", " ", str(w).lower().strip().replace("_", " "))
    last = w.split(" ")[-1] if w else w
    if not last or last in KEEP_S or len(last) <= 3:
        return w
    if last.endswith("ies"):
        last = last[:-3] + "y"
    elif last.endswith("ves") and last not in ("eaves",):
        last = last[:-3] + "f"            # leaves -> leaf, shelves -> shelf
    elif last.endswith(("ches", "shes", "xes", "sses", "zes")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith(("ss", "us", "is")):
        last = last[:-1]
    return " ".join(w.split(" ")[:-1] + [last])


def whole_photo_q(q: str) -> str:
    """A truth program's `person_need` question asked of the WHOLE photo, for an album with no named person
    ("the guy in the red jacket", "the women in saris": person null). With no red box "the person" names nobody, so it
    becomes "a person": "Is the person wearing a sari?" -> "Is a person wearing a sari?". (RESULTS 49c: build.py used to
    skip person_need when person was null, so 6/50 dev c truths were every photo of a hidden day/place.)"""
    q = q.strip()
    q = re.sub(r"\bThe person\b", "A person", q)
    return re.sub(r"\bthe person\b", "a person", q)


def load_items():
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "path", "taken", "lat", "lon", "place"])
    it["item_id"] = it.item_id.astype(str)
    it["user"] = it.path.str.extract(r"/images/images/([^/]+)/")[0]
    # Flickr 'taken' is the camera's wall clock stored as UTC (date_source metadata_json): drop the tz -> local clock
    it["t"] = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601").dt.tz_localize(None)
    return it


def load_facts(user: str) -> dict:
    f = ROOT / "facts" / f"{user}.jsonl"
    out = {}
    if f.exists():
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("facts") is not None or r["item_id"] not in out:
                out[r["item_id"]] = r.get("facts")
    return out


def place_tail(addr) -> str:
    if not isinstance(addr, str) or not addr:
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip() and not re.search(r"\d", p)]
    return ", ".join(parts[-4:])


def city(addr) -> str:
    parts = [p.strip() for p in str(addr or "").split(",") if p.strip() and not re.search(r"\d", p)]
    return ", ".join(parts[-4:-2]) if len(parts) >= 4 else ", ".join(parts[:2])


def terms(f: dict) -> set:
    """Normalized searchable terms of one photo's facts (field:term)."""
    if not f:
        return set()
    out = set()
    for k in LISTS:
        v = f.get(k) or []
        if isinstance(v, str):
            v = [v]
        for x in v:
            if x:
                out.add(f"{k}:{norm(x)}")
    for k in SCALARS:
        v = f.get(k)
        if v and isinstance(v, str) and v.lower() not in ("none", "n/a", ""):
            out.add(f"{k}:{norm(v)}")
    pc = f.get("people_count")
    if isinstance(pc, (int, float)):
        out.add("people:" + ("0" if pc == 0 else "1" if pc == 1 else "2" if pc == 2 else "3-5" if pc <= 5 else "6+"))
    return out


def library_table(user: str, it=None):
    it = load_items() if it is None else it
    lib = it[it.user == user].copy().sort_values("t").reset_index(drop=True)
    F = load_facts(user)
    lib["facts"] = lib.item_id.map(F)
    lib["terms"] = lib.facts.map(terms)
    lib["day"] = lib.t.dt.date.astype(str)
    lib["city"] = lib.place.map(city)
    return lib


H2_DIR = __import__("os").environ.get("FP_CB_H2", "programs_h2")   # "" = programs as generated (before RESULTS 52)
G2_DIR = __import__("os").environ.get("FP_CB_G2", "programs_g2")   # "" = g programs as generated (before RESULTS 53)


def load_programs(user: str) -> dict:
    """programs/<user>.json with category h's turn_programs replaced by the RESULTS 52 rewrite (programs_h2/<user>.json,
    same words; the original is kept as turn_programs_v1) and category g's albums by the RESULTS 53 rewrite
    (programs_g2/, original kept as albums_v1). A rewrite that failed to parse keeps the original."""
    P = json.loads((ROOT / "programs" / f"{user}.json").read_text())
    f = ROOT / H2_DIR / f"{user}.json" if H2_DIR else None
    if f is not None and f.exists():
        H2 = json.loads(f.read_text())
        for qi, r in enumerate(P["requests"]):
            x = H2.get(str(qi))
            if r.get("cat") == "h" and x and x.get("ok") and x.get("turn_programs"):
                r["turn_programs_v1"] = r.get("turn_programs")
                r["turn_programs"] = x["turn_programs"]
                r["why_v1"], r["why"] = r.get("why"), x.get("why") or r.get("why")
    f = ROOT / G2_DIR / f"{user}.json" if G2_DIR else None
    if f is not None and f.exists():   # RESULTS 53: g's two albums rewritten (same words); original kept as albums_v1
        G2 = json.loads(f.read_text())
        for qi, r in enumerate(P["requests"]):
            x = G2.get(str(qi))
            if r.get("cat") == "g" and x and x.get("ok") and x.get("albums"):
                r["albums_v1"], r["albums"] = r.get("albums"), x["albums"]
                r["why_v1"], r["why"] = r.get("why"), x.get("why") or r.get("why")
    return P
