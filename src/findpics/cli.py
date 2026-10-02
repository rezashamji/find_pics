"""findpics CLI.

  findpics scan   <library_dir> <index_dir> [--metadata library_metadata.json]
  findpics index  <index_dir> [--shards K]            (local; on a cluster use slurm/templates/index_array.sbatch)
  findpics ask    "<message>" --out <conversation_dir> [--index <index_dir>] [--me NAME] [--apple-apply]
                  (same --out again = follow-up: "only the ones at night", "also 2019", "look harder")
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
    """One message of the conversation. First message in a folder = new search; later messages = follow-ups that
    change the plan ("only the ones at night", "also 2019", "look harder"), which is then rerun."""
    from . import store
    from .models import ImageTextEncoder
    from .vlm import VLLMJudge, MLXJudge
    from .converse import CachedJudge, Session, plan_turn, run_plan
    from .engine import Thresholds
    from .albums import write_folder_album, write_apple_album
    from .report import write_review_page

    S = Session(a.out)
    index_dir = a.index_dir or S.state.get("index_dir")
    if not index_dir:
        raise SystemExit("First message in a new folder: pass the index folder with --index.")
    S.state["index_dir"] = str(Path(index_dir).resolve())
    if a.reviews:      # taps from the review page: photos marked wrong stay out of every later answer
        S.add_reviews(json.loads(Path(a.reviews).read_text()))
    idx = store.load(index_dir)
    user_refs = _parse_refs(a.ref)
    people = sorted({p for ps in idx.items["apple_persons"] if ps is not None for p in ps} | set(user_refs))
    import torch
    base = VLLMJudge() if torch.cuda.is_available() else MLXJudge()  # Linux GPU vs Apple Silicon (MLX path untested)
    judge = CachedJudge(base, S.dir / "judge_cache.json")
    P = plan_turn(a.message, judge.text, history=S.state["messages"], current=S.current, owner=a.me or "me",
                  people=people, today=date.fromisoformat(a.today) if a.today else None)
    print("PLAN:", P.model_dump_json(indent=1))
    if a.plan_only:
        print("--plan-only: stopping before any search.")
        return
    S.add_turn(a.message, P)
    out = S.dir / f"turn_{S.turn}"; out.mkdir(parents=True, exist_ok=True)

    def refs_for(person):
        name, refs, ref_face, n = _refs_for(idx, person, a.me, user_refs)
        print(f"person '{person}' -> '{name}': {n} tagged items, {0 if refs is None else len(refs)} reference faces")
        return name, refs, ref_face, n

    results = run_plan(idx, P, ImageTextEncoder(), judge, refs_for, th=Thresholds(tail_budget=a.audit),
                       exclude_ids=set(S.state["exclude_ids"]))
    judge.save()
    summary, pages = [], []
    for res in results:
        spec = res.spec
        d = write_folder_album(spec.name, res.returned, out, res.report)
        res.judged.to_parquet(d / "judged.parquet")
        apple_name = spec.name if S.turn == 1 else f"{spec.name} (v{S.turn})"   # Apple albums are only ever added to
        msg = write_apple_album(apple_name, list(res.returned.item_id), apply=a.apple_apply)
        print(res.report); print(msg)
        summary.append(dict(album=spec.name, n=len(res.returned), report=res.report, certificate=res.cert, apple=msg))
        lab = lambda r: f"{str(idx.items.taken.iloc[int(r.item_row)])[:10]} p={r.p_attr:.2f}"
        aud = res.judged[res.judged["where"].isin(["tail_sample", "human_audit_sample"]) & (~res.judged.y)].head(200)
        pages.append(dict(name=spec.name, report=res.report,
                          items=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in res.returned.itertuples()],
                          audit=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in aud.itertuples()]))
    page = write_review_page(out, " / ".join(S.state["messages"]), P.model_dump(), pages, session_dir=S.dir)
    print(f"Judge answers reused from earlier turns: {judge.hits}; new: {judge.misses}")
    print(f"Review page: {page}")
    (out / "summary.json").write_text(json.dumps(dict(messages=S.state["messages"], plan=P.model_dump(), albums=summary,
                                                      index_dir=S.state["index_dir"]), indent=1, default=str))
    S.save()


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


def main():
    ap = argparse.ArgumentParser(prog="findpics")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan"); s.add_argument("library"); s.add_argument("index_dir"); s.add_argument("--metadata"); s.set_defaults(f=cmd_scan)
    s = sp.add_parser("index"); s.add_argument("index_dir"); s.add_argument("--shards", type=int, default=1); s.add_argument("--workers", type=int, default=4); s.set_defaults(f=cmd_index)
    s = sp.add_parser("ask", help="one message of a conversation: a new request, or a follow-up in the same --out folder")
    s.add_argument("message"); s.add_argument("--out", required=True, help="conversation folder (created on the first message)")
    s.add_argument("--index", dest="index_dir", help="index folder (first message only; remembered after)")
    s.add_argument("--me"); s.add_argument("--today"); s.add_argument("--apple-apply", action="store_true")
    s.add_argument("--plan-only", action="store_true", help="print how the sentence was understood, then stop")
    s.add_argument("--reviews", help="reviews.json exported from the review page: photos marked wrong stay out")
    s.add_argument("--audit", type=int, default=1000, help="random photos the judge checks among the rest; more = tighter "
                   "completeness bound (to prove at most m misses among N unchecked, you need about 3N/m)")
    s.add_argument("--ref", action="append", help='reference photos for a subject, e.g. --ref "Reza=me1.jpg,me2.jpg" '
                   "(no Apple tags needed). Repeat for several subjects.")
    s.set_defaults(f=cmd_ask)
    s = sp.add_parser("apply-reviews", help="apply review-page clicks (reviews.json) to the albums")
    s.add_argument("albums_dir"); s.add_argument("reviews"); s.set_defaults(f=cmd_apply_reviews)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
