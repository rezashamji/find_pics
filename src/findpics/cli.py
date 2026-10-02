"""findpics CLI.

  findpics scan   <library_dir> <index_dir> [--metadata library_metadata.json]
  findpics index  <index_dir> [--shards K]            (local; on a cluster use slurm/templates/index_array.sbatch)
  findpics ask    <index_dir> "<request>" --out <albums_dir> [--me NAME] [--apple-apply]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def _refs_for(idx, person: str, me: str | None):
    """Reference faces for a person: items Apple (or the user) tagged with that name."""
    from .people import refs_from_items, expand_refs
    name = me if person.lower() in ("me", "myself", "i", "owner") and me else person
    known = sorted({p for ps in idx.items["apple_persons"] if ps is not None for p in ps})
    if name not in known:  # planner wrote "Reza", Photos says "Reza Shamji" (or the reverse)
        cands = [k for k in known if name.lower() in k.lower() or k.lower() in name.lower()]
        if len(cands) == 1:
            name = cands[0]
        elif me and me in known and (name.lower() in me.lower() or me.lower() in name.lower()):
            name = me
    rows = [i for i, ps in enumerate(idx.items["apple_persons"]) if ps is not None and name in list(ps)]
    if not rows:
        # never silently fall back to "anyone who looks X": that would answer a different question
        raise SystemExit(f"No reference photos for '{person}' (resolved to '{name}'). Known people: {known[:30]}. "
                         f"Tag this person in Apple Photos' People album, or pass --me with the exact name.")
    refs, face_rows = refs_from_items(idx, rows, return_rows=True)
    # query-time clustering: confident matches (>=0.55) become references for 3 rounds, chaining across eras.
    # Test library, refs from each person's most recent half only: oldest-third recall 0.80-0.97 -> 0.86-0.98, wrong matches unchanged.
    refs = expand_refs(idx, refs, accept=0.55, rounds=3)
    return name, refs, (int(face_rows[0]) if len(face_rows) else None), len(rows)


def cmd_scan(a):
    from .ingest import scan, to_frame
    out = Path(a.index_dir); out.mkdir(parents=True, exist_ok=True)
    df = to_frame(scan(a.library, a.metadata))
    df.to_parquet(out / "items.parquet")
    print(f"{len(df):,} items ({df['media'].value_counts().to_dict()}) -> {out/'items.parquet'}")


def cmd_index(a):
    from .index import index_shard
    for k in range(a.shards):
        index_shard(Path(a.index_dir), k, a.shards, None, None, workers=a.workers)


def cmd_ask(a):
    from . import store
    from .models import ImageTextEncoder
    from .vlm import VLLMJudge, MLXJudge
    from .planner import plan
    from .engine import run_album, make_exclusive
    from .albums import write_folder_album, write_apple_album
    from .report import write_review_page

    idx = store.load(a.index_dir)
    people = sorted({p for ps in idx.items["apple_persons"] if ps is not None for p in ps})
    import torch
    judge = VLLMJudge() if torch.cuda.is_available() else MLXJudge()  # Linux GPU vs Apple Silicon (MLX path untested)
    P = plan(a.request, judge.text, owner=a.me or "me", people=people,
             today=date.fromisoformat(a.today) if a.today else None)
    print("PLAN:", P.model_dump_json(indent=1))
    if a.plan_only:
        print("--plan-only: stopping before any search. Check the albums, dates and conditions above, then rerun without it.")
        return
    enc = ImageTextEncoder()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary, pages, results = [], [], []
    for spec in P.albums:
        refs, ref_face, n_ref, name = None, None, 0, None
        if spec.person:
            name, refs, ref_face, n_ref = _refs_for(idx, spec.person, a.me)
            print(f"person '{spec.person}' -> '{name}': {n_ref} tagged items, {0 if refs is None else len(refs)} reference faces")
        from .engine import Thresholds
        results.append(run_album(idx, spec, enc, judge, refs, ref_face_row=ref_face, th=Thresholds(tail_budget=a.audit)))
    make_exclusive(results)
    for res in results:
        spec = res.spec
        d = write_folder_album(spec.name, res.returned, out, res.report)
        res.judged.to_parquet(d / "judged.parquet")
        msg = write_apple_album(spec.name, list(res.returned.item_id), apply=a.apple_apply)
        print(res.report); print(msg)
        summary.append(dict(album=spec.name, n=len(res.returned), report=res.report, certificate=res.cert, apple=msg))
        lab = lambda r: f"{str(idx.items.taken.iloc[int(r.item_row)])[:10]} p={r.p_attr:.2f}"
        aud = res.judged[res.judged["where"].isin(["tail_sample", "human_audit_sample"]) & (~res.judged.y)].head(200)
        pages.append(dict(name=spec.name, report=res.report,
                          items=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in res.returned.itertuples()],
                          audit=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in aud.itertuples()]))
    page = write_review_page(out, a.request, P.model_dump(), pages)
    print(f"Review page: {page}")
    (out / "summary.json").write_text(json.dumps(dict(request=a.request, plan=P.model_dump(), albums=summary), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(prog="findpics")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("library"); s.add_argument("index_dir"); s.add_argument("--metadata"); s.set_defaults(f=cmd_scan)
    s = sp.add_parser("index"); s.add_argument("index_dir"); s.add_argument("--shards", type=int, default=1); s.add_argument("--workers", type=int, default=4); s.set_defaults(f=cmd_index)
    s = sp.add_parser("ask"); s.add_argument("index_dir"); s.add_argument("request"); s.add_argument("--out", required=True)
    s.add_argument("--me"); s.add_argument("--today"); s.add_argument("--apple-apply", action="store_true")
    s.add_argument("--plan-only", action="store_true", help="print how the sentence was understood, then stop")
    s.add_argument("--audit", type=int, default=1000, help="random photos the judge checks among the rest; more = tighter "
                   "completeness bound (to prove at most m misses among N unchecked, you need about 3N/m)")
    s.set_defaults(f=cmd_ask)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
