"""findpics CLI.

  findpics scan   <library_dir> <index_dir> [--metadata library_metadata.json]
  findpics index  <index_dir> [--shards K]            (local; on a cluster use slurm/templates/index_array.sbatch)
  findpics ask    "<message>" --out <conversation_dir> [--index <index_dir>] [--me NAME] [--apple-apply]
                  (same --out again = follow-up: "only the ones at night", "also 2019", "drop the group shots")
  findpics chat   --out <conversation_dir> [--index <index_dir>]   (models load once; each line typed is a message)
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


def cmd_apple_copy(a):
    """Apple's privacy.apple.com iCloud Photos copy (zips) -> shrunk library + sidecars, one zip at a time."""
    from .apple_copy import ingest_all
    rep = ingest_all(a.zips_dir, a.out_dir)
    print(f"{len(rep)} new zip(s) ingested -> {a.out_dir}/library (then: findpics scan {a.out_dir}/library <index_dir>)")


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
    change the plan ("only the ones at night", "also 2019", "drop the group shots"), which is then rerun. Each album keeps
    improving in rounds until every photo is checked; answers cached from earlier turns are reused."""
    ctx = _load(a)
    _turn(ctx, a.message, a)


def cmd_chat(a):
    """The chat box: models load once, then every line you type is a message (same as `ask` with the same --out)."""
    ctx = _load(a)
    print("Type what you want; each line is a message. Ctrl-C stops the current search (keeps what it found); "
          "Ctrl-D quits.", flush=True)
    while True:
        try:
            msg = input("> ").strip()
        except EOFError:
            break
        if msg:
            _turn(ctx, msg, a)


def _load(a):
    from . import store
    from .models import ImageTextEncoder
    from .vlm import VLLMJudge, MLXJudge
    from .converse import CachedJudge, Session
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
    return dict(S=S, idx=idx, user_refs=user_refs, people=people, judge=CachedJudge(base, S.dir / "judge_cache.json"),
                enc=ImageTextEncoder())


def _turn(ctx, message, a):
    from .converse import plan_turn, stream_plan
    from .engine import Thresholds
    import time
    S, idx, judge = ctx["S"], ctx["idx"], ctx["judge"]
    h0, m0 = judge.hits, judge.misses
    P = plan_turn(message, judge.text, history=S.state["messages"], current=S.current, owner=a.me or "me",
                  people=ctx["people"], today=date.fromisoformat(a.today) if a.today else None)
    print("PLAN:", P.model_dump_json(indent=1))
    if a.plan_only:
        print("--plan-only: stopping before any search.")
        return
    S.add_turn(message, P); S.save()
    out = S.dir / f"turn_{S.turn}"; out.mkdir(parents=True, exist_ok=True)

    def refs_for(person):
        name, refs, ref_face, n = _refs_for(idx, person, a.me, ctx["user_refs"])
        print(f"person '{person}' -> '{name}': {n} tagged items, {0 if refs is None else len(refs)} reference faces")
        return name, refs, ref_face, n

    t0, rnd, results = time.time(), 0, None
    try:   # every round is written out, so stopping at any point (Ctrl-C, --minutes) keeps the latest answer
        for results in stream_plan(idx, P, ctx["enc"], judge, refs_for, th=Thresholds(tail_budget=a.audit, stream=True),
                                   exclude_ids=set(S.state["exclude_ids"])):
            rnd += 1
            judge.save(); _write_turn(idx, S, P, out, results, a, final=False)
            print(f"[round {rnd}, {time.time() - t0:.0f}s] " + "; ".join(
                f"{r.spec.name}: {len(r.returned)}" + (f" (at least {r.cert['recall_lower']:.0%} found)" if r.cert else "")
                for r in results), flush=True)
            if a.minutes and time.time() - t0 > 60 * a.minutes:
                print(f"Stopped after {a.minutes} min; the bound above holds at this point. Send another message to continue.")
                break
    except KeyboardInterrupt:
        print("Stopped; the albums and bounds from the last finished round are saved. Send another message to continue.")
    judge.save()
    if results is None:
        print("Stopped before the first round finished; nothing written for this turn.")
        return
    _write_turn(idx, S, P, out, results, a, final=True)
    print(f"Judge answers reused: {judge.hits - h0}; new: {judge.misses - m0}")
    S.save()


def _write_turn(idx, S, P, out, results, a, final):
    from .albums import write_folder_album, write_apple_album
    from .report import write_review_page
    summary, pages = [], []
    for res in results:
        spec = res.spec
        d = write_folder_album(spec.name, res.returned, out, res.report)
        res.judged.to_parquet(d / "judged.parquet")
        apple = "(written when the search finishes or stops)"
        if final:   # Apple albums are only ever added to, so write them once, at the end
            apple = write_apple_album(spec.name if S.turn == 1 else f"{spec.name} (v{S.turn})", list(res.returned.item_id),
                                      apply=a.apple_apply)
            print(res.report); print(apple)
        summary.append(dict(album=spec.name, n=len(res.returned), report=res.report, certificate=res.cert, apple=apple))
        lab = lambda r: f"{str(idx.items.taken.iloc[int(r.item_row)])[:10]} p={r.p_attr:.2f}"
        aud = res.judged[res.judged["where"].isin(["tail_sample", "human_audit_sample"]) & (~res.judged.y)].head(200)
        pages.append(dict(name=spec.name, report=res.report,
                          items=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in res.returned.itertuples()],
                          audit=[dict(item_id=r.item_id, path=r.path, label=lab(r)) for r in aud.itertuples()]))
    page = write_review_page(out, " / ".join(S.state["messages"]), P.model_dump(), pages, session_dir=S.dir)
    if final:
        print(f"Review page: {page}")
    (out / "summary.json").write_text(json.dumps(dict(messages=S.state["messages"], plan=P.model_dump(), albums=summary,
                                                      finished=final, index_dir=S.state["index_dir"]), indent=1, default=str))


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
    s = sp.add_parser("apple-copy", help="ingest Apple's 'copy of your data' iCloud Photos zips (never deletes them)")
    s.add_argument("zips_dir"); s.add_argument("out_dir"); s.set_defaults(f=cmd_apple_copy)
    s = sp.add_parser("scan"); s.add_argument("library"); s.add_argument("index_dir"); s.add_argument("--metadata"); s.set_defaults(f=cmd_scan)
    s = sp.add_parser("index"); s.add_argument("index_dir"); s.add_argument("--shards", type=int, default=1); s.add_argument("--workers", type=int, default=4); s.set_defaults(f=cmd_index)
    for cmd, f in (("ask", cmd_ask), ("chat", cmd_chat)):
        s = sp.add_parser(cmd, help="one message (ask) or an interactive chat (chat); same --out folder = same conversation")
        if cmd == "ask":
            s.add_argument("message")
        s.add_argument("--out", required=True, help="conversation folder (created on the first message)")
        s.add_argument("--index", dest="index_dir", help="index folder (first message only; remembered after)")
        s.add_argument("--me"); s.add_argument("--today"); s.add_argument("--apple-apply", action="store_true")
        s.add_argument("--plan-only", action="store_true", help="print how the sentence was understood, then stop")
        s.add_argument("--minutes", type=float, help="stop each search after this many minutes (default: keep going "
                       "until every photo is checked; each round is saved, Ctrl-C also stops)")
        s.add_argument("--reviews", help="reviews.json exported from the review page: photos marked wrong stay out")
        s.add_argument("--audit", type=int, default=1000, help="random photos the judge checks in the first round; later "
                       "rounds check more (to prove at most m misses among N unchecked, you need about 3N/m)")
        s.add_argument("--ref", action="append", help='reference photos for a subject, e.g. --ref "Reza=me1.jpg,me2.jpg" '
                       "(no Apple tags needed). Repeat for several subjects.")
        s.set_defaults(f=f)
    s = sp.add_parser("apply-reviews", help="apply review-page clicks (reviews.json) to the albums")
    s.add_argument("albums_dir"); s.add_argument("reviews"); s.set_defaults(f=cmd_apply_reviews)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
