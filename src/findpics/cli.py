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


def _parse_refs(specs):
    """--ref "Name=a.jpg,b.jpg" (repeatable) -> {name: [paths]}"""
    out = {}
    for sp in specs or []:
        name, _, paths = sp.partition("=")
        out.setdefault(name.strip(), []).extend(p.strip() for p in paths.split(",") if p.strip())
    return out


def _faces_from_photos(paths):
    """Largest face in each reference photo -> identity vectors. Empty if the photos contain no faces."""
    from .media import load_image
    from .models import FaceEncoder
    fe = FaceEncoder()
    vecs = []
    for p in paths:
        fs = fe.faces(load_image(p))
        if fs:
            big = max(fs, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
            vecs.append(big["emb"])
    return np.stack(vecs).astype(np.float16) if vecs else np.zeros((0, 512), np.float16)


def _refs_for(idx, person: str, me: str | None, user_refs: dict | None = None):
    """Reference faces for a person: photos the user passed with --ref, else items Apple tagged with that name."""
    from .people import refs_from_items, expand_refs
    name = me if person.lower() in ("me", "myself", "i", "owner") and me else person
    user_refs = user_refs or {}
    match = [k for k in user_refs if k.lower() == name.lower() or k.lower() in name.lower() or name.lower() in k.lower()]
    if match:
        refs = _faces_from_photos(user_refs[match[0]])
        if len(refs) == 0:
            raise SystemExit(f"No face found in the --ref photos for '{match[0]}'. Non-face subjects (pets, objects) are "
                             f"not supported yet: the general reference path is being measured (eval/eval_pet_identity.py).")
        refs = expand_refs(idx, refs, accept=0.55, rounds=3)
        return match[0], refs, None, len(user_refs[match[0]])
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
    user_refs = _parse_refs(a.ref)
    people = sorted({p for ps in idx.items["apple_persons"] if ps is not None for p in ps} | set(user_refs))
    import torch
    judge = VLLMJudge() if torch.cuda.is_available() else MLXJudge()  # Linux GPU vs Apple Silicon (MLX path untested)
    if a.multistep:   # two-step: anchor moment -> window -> target (+ exclusion); single album
        from .agent import execute, make_plan
        MP = make_plan(a.request, judge.text, today=date.fromisoformat(a.today) if a.today else None)
        print("MULTI-STEP PLAN:", MP.model_dump_json(indent=1))
        if a.plan_only:
            return
        from .albums import write_folder_album as _wfa
        res = execute(idx, MP, ImageTextEncoder(), judge)
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        _wfa("results", res["returned"], out, res["report"] + "\n" + json.dumps(res["trace"], default=str))
        print(res["report"]); print("trace:", json.dumps(res["trace"], default=str))
        (out / "summary.json").write_text(json.dumps(dict(request=a.request, multistep_plan=MP.model_dump(), trace=res["trace"],
                                                          index_dir=str(Path(a.index_dir).resolve())), indent=1, default=str))
        return
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
            name, refs, ref_face, n_ref = _refs_for(idx, spec.person, a.me, user_refs)
            print(f"person '{spec.person}' -> '{name}': {n_ref} tagged items, {0 if refs is None else len(refs)} reference faces")
        from .engine import Thresholds
        th = Thresholds(tail_budget=a.audit)
        if a.exhaustive:   # the judge looks at EVERY in-scope item: slow, nothing lost to the cheap first stage
            th.head_size = th.head_max = idx.n_items
        results.append(run_album(idx, spec, enc, judge, refs, ref_face_row=ref_face, th=th))
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
    (out / "summary.json").write_text(json.dumps(dict(request=a.request, plan=P.model_dump(), albums=summary,
                                                      index_dir=str(Path(a.index_dir).resolve())), indent=1, default=str))


def cmd_apply_reviews(a):
    """Apply review-page clicks: drop items marked wrong from each album and record the human check counts.
    Albums are folders of symlinks, so this only removes links; originals are never touched."""
    rev = json.loads(Path(a.reviews).read_text())
    out = Path(a.albums_dir)
    for man in sorted(out.glob("*/manifest.json")):
        m = json.loads(man.read_text())
        items = m["items"]
        ids = [str(it["item_id"]) for it in items]
        ok = sum(rev.get(i) == "ok" for i in ids); bad = [i for i in ids if rev.get(i) == "bad"]
        keep = [it for it in items if rev.get(str(it["item_id"])) != "bad"]
        for it in items:
            if rev.get(str(it["item_id"])) == "bad":
                from .albums import remove_album_link
                for link in man.parent.glob(f"*_{Path(it['path']).name}"):
                    if link.is_symlink():
                        remove_album_link(link)
        m["items"] = keep
        m["human_check"] = dict(checked=ok + len(bad), correct=ok, wrong=len(bad))
        man.write_text(json.dumps(m, indent=1, default=str))
        print(f"{m['album']}: you checked {ok + len(bad)} of {len(items)}; {ok} correct, {len(bad)} wrong (removed). "
              f"Album now {len(keep)} items.")


def cmd_refine(a):
    """Follow-up edit of an album: "remove the blurry ones", "get rid of ones like these", "add more like this"."""
    import pandas as pd
    from . import store
    from .refine import apply_ops, plan_edits, write_edit
    out = Path(a.albums_dir)
    summ = json.loads((out / "summary.json").read_text())
    idx = store.load(summ["index_dir"])
    albums = [d.name for d in out.iterdir() if (d / "manifest.json").exists()]
    import torch
    from .vlm import VLLMJudge, MLXJudge
    judge = VLLMJudge() if torch.cuda.is_available() else MLXJudge()
    selected = [x for x in (a.selected or "").split(",") if x]
    P = plan_edits(a.instruction, judge.text, albums, selected)
    name = a.album or P.album or (albums[0] if len(albums) == 1 else None)
    if name not in albums:
        raise SystemExit(f"Which album? Choose one of {albums} with --album.")
    print("EDIT PLAN:", P.model_dump_json(indent=1))
    d = out / name
    items = pd.DataFrame(json.loads((d / "manifest.json").read_text())["items"])
    q = next((x.get("judge_question") for x in summ["plan"]["albums"] if x.get("name") == name), None)
    new, log = apply_ops(idx, items, P, judge, q)
    print("\n".join(log))
    if a.dry_run:
        print(f"--dry-run: album '{name}' would go from {len(items)} to {len(new)} items. Nothing written.")
        return
    write_edit(d, new, log)
    print(f"Album '{name}': {len(items)} -> {len(new)} items (previous version kept as manifest.v*.json).")


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
    s.add_argument("--ref", action="append", help='reference photos for a subject, e.g. --ref "Reza=me1.jpg,me2.jpg" '
                   "(no Apple tags needed). Repeat for several subjects.")
    s.add_argument("--multistep", action="store_true", help='two-step requests, e.g. "photos from the day I saw X, without Y"')
    s.add_argument("--exhaustive", action="store_true", help="judge every photo (slow; nothing missed by the fast first "
                   "stage). Fast mode judges the top candidates + a random sample and states a completeness bound.")
    s.set_defaults(f=cmd_ask)
    s = sp.add_parser("refine", help='follow-up edit, e.g. "remove the blurry ones" or "add more like these"')
    s.add_argument("albums_dir"); s.add_argument("instruction"); s.add_argument("--album")
    s.add_argument("--selected", help="comma-separated item ids picked on the review page")
    s.add_argument("--dry-run", action="store_true"); s.set_defaults(f=cmd_refine)
    s = sp.add_parser("apply-reviews", help="apply review-page clicks (reviews.json) to the albums")
    s.add_argument("albums_dir"); s.add_argument("reviews"); s.set_defaults(f=cmd_apply_reviews)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
