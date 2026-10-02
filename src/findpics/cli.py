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
    name = me if person in ("me", "Me", "myself", "I") and me else person
    rows = [i for i, ps in enumerate(idx.items["apple_persons"]) if ps is not None and name in list(ps)]
    if not rows:
        return name, None, 0
    refs = refs_from_items(idx, rows)
    return name, refs, len(rows)


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
    from .vlm import VLLMJudge
    from .planner import plan
    from .engine import run_album
    from .albums import write_folder_album, write_apple_album

    idx = store.load(a.index_dir)
    people = sorted({p for ps in idx.items["apple_persons"] if ps is not None for p in ps})
    judge = VLLMJudge()
    P = plan(a.request, judge.text, owner=a.me or "me", people=people,
             today=date.fromisoformat(a.today) if a.today else None)
    print("PLAN:", P.model_dump_json(indent=1))
    enc = ImageTextEncoder()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in P.albums:
        refs, n_ref, name = None, 0, None
        if spec.person:
            name, refs, n_ref = _refs_for(idx, spec.person, a.me)
            print(f"person '{spec.person}' -> '{name}': {n_ref} tagged items, {0 if refs is None else len(refs)} reference faces")
        res = run_album(idx, spec, enc, judge, refs)
        d = write_folder_album(spec.name, res.returned, out, res.report)
        res.candidates.to_parquet(d / "candidates.parquet")
        msg = write_apple_album(spec.name, list(res.returned.item_id), apply=a.apple_apply)
        print(res.report); print(msg)
        summary.append(dict(album=spec.name, n=len(res.returned), report=res.report, audit=res.audit, apple=msg))
    (out / "summary.json").write_text(json.dumps(dict(request=a.request, plan=P.model_dump(), albums=summary), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser(prog="findpics")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("library"); s.add_argument("index_dir"); s.add_argument("--metadata"); s.set_defaults(f=cmd_scan)
    s = sp.add_parser("index"); s.add_argument("index_dir"); s.add_argument("--shards", type=int, default=1); s.add_argument("--workers", type=int, default=4); s.set_defaults(f=cmd_index)
    s = sp.add_parser("ask"); s.add_argument("index_dir"); s.add_argument("request"); s.add_argument("--out", required=True)
    s.add_argument("--me"); s.add_argument("--today"); s.add_argument("--apple-apply", action="store_true"); s.set_defaults(f=cmd_ask)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
