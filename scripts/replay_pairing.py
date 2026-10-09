"""Offline replay of the person-album PAIRING step ("me heavier" vs "me fit") from saved demo runs, no GPU.

Every saved run (data/private/sample_runs/mode_*) holds, per album, judged.parquet (the judged pool: item_id, p_attr =
judge P(yes), rel = within-person rank) and the session's judge_cache.json (P(yes) keyed like CachedJudge: md5 of tag +
question + crop pixels). This script
  keys    : renders every pooled photo's judge crop once (engine._frames_for + person_crop_boxed, exactly what
            _judge_rows hands the judge), hashes it the way CachedJudge does for every question/tag the runs used, and
            saves item_id -> key (data/private/audits/pair_replay_keys.json). CPU only, ~1-2 min: run in background.
  replay  : per run, P comes FROM THE JUDGE CACHE via those keys (a stub judge: cached P by key; checked equal to the
            pool's p_attr), make_exclusive runs on the pools with the run's events, and the albums are scored against
            Reza's labels. Also checks the replayed albums equal the saved album folders, and prints the split's
            internals (logit differences, mixture components, which branch fired).
Counts only on stdout. Usage:
  python scripts/replay_pairing.py keys
  python scripts/replay_pairing.py replay [--engine-rev REV] [--out table.csv] [run ...]
    --engine-rev: replay with engine.py as of a git revision (e.g. the commit before the 10-09 fix) for a before table
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from findpics import engine as E          # noqa: E402
from findpics.planner import AlbumSpec    # noqa: E402

RUNS = ROOT / "data/private/sample_runs"
INDEX = ROOT / "data/private/index_sample"
KEYS = ROOT / "data/private/audits/pair_replay_keys.json"
LABELS = ROOT / "data/private/audits/reza_labels_demo6.json"
RED = re.compile(r"(?i)\bthe person in (?:this|the) (?:photo|image|picture|video)\b|\bthe person\b(?! in the red box)")


def runs(names=None):
    out = []
    for d in sorted(RUNS.glob("mode_*")):
        if d.is_dir() and (d / "judge_cache.json").exists() and (d / "session.json").exists():
            if not names or d.name in names:
                out.append(d)
    return out


def albums_of(run):
    s = json.loads((run / "session.json").read_text())
    return [a for a in s["plans"][-1]["albums"]]


def pools_of(run):
    out = []
    for a in albums_of(run):
        j = pd.read_parquet(run / "turn_1" / a["name"] / "judged.parquet")
        out.append((a, j))
    return out


def red_q(q):
    return RED.sub("the person in the red box", q)


def bipolar_q(qa, qb):   # converse.stream_plan's FP_PAIR_BIPOLAR question (original, un-rewritten questions)
    return f'Which describes this photo better? A: "{qa}" B: "{qb}" Is it A rather than B?'


def questions_used():
    qs = set()
    for r in runs():
        al = albums_of(r)
        qs |= {red_q(a["judge_question"]) for a in al}
        qs.add(bipolar_q(al[0]["judge_question"], al[1]["judge_question"]))
    return sorted(qs)


def cmd_keys(_):
    from findpics import store
    idx = store.load(str(INDEX))
    pool = pd.read_parquet(runs()[0] / "turn_1" / albums_of(runs()[0])[0]["name"] / "judged.parquet")
    pool = pool[pool["where"] == "identity_match"]
    qs = questions_used(); tags = ["", "first|"]
    def one(a):
        iid, r, fr = a
        ims = [E.person_crop_boxed(idx, im, int(fr)) for im in E._frames_for(idx, int(r), int(fr))]
        return iid, {t + "\x1f" + q: [hashlib.md5(t.encode() + q.encode() + im.tobytes() + str(im.size).encode()).hexdigest()
                                     for im in ims] for t in tags for q in qs}
    from concurrent.futures import ThreadPoolExecutor
    keys = {}
    with ThreadPoolExecutor(min(16, len(os.sched_getaffinity(0)))) as ex:   # decoding dominates (HEIC / video frames); same work as _judge_rows' prefetch
        for k, (iid, d) in enumerate(ex.map(one, zip(pool.item_id, pool.item_row, pool.face_row))):
            keys[iid] = d
            if k % 50 == 0:
                print(f"{k}/{len(pool)}", flush=True)
    KEYS.write_text(json.dumps(keys)); os.chmod(KEYS, 0o600)
    print("KEYS", len(keys), "items x", len(qs), "questions x", len(tags), "tags ->", KEYS)


class StubJudge:
    """Cached P by key: what CachedJudge would have answered, without the model."""
    def __init__(self, cache, keys):
        self.cache, self.keys, self.miss = cache, keys, 0

    def p(self, iid, tag, q):
        # vote runs: the person look asks judge.first (tag "first|") since 10-06; earlier vote runs used the plain key
        for t in (tag, "" if tag else "first|"):
            ks = self.keys[iid][t + "\x1f" + q]
            got = [self.cache[k] for k in ks if k in self.cache]
            if ks and len(got) == len(ks):
                return max(got)  # _judge_rows: a video's answer = max over its frames
        self.miss += 1
        return None


def gmm_report(x, groups=None):
    """The mixture the split saw (same EM as engine._two_groups) + one-group BIC."""
    info = {}
    try:
        post, clear, mid = E._two_groups(x, info, groups)
    except TypeError:          # engine before the 10-09 fix: no diagnostics
        post, clear, mid = E._two_groups(x)
    info.pop("n_degenerate_starts", None)
    return dict(n=len(x), branch="split" if clear else "rank-margin", **info,
                n_sure_up=int(((post >= E.PAIR_SURE) & (x > mid)).sum()),
                n_sure_lo=int(((post <= 1 - E.PAIR_SURE) & (x < mid)).sum()))


def load_engine_rev(rev):
    """findpics.engine as of git revision `rev` (same package, so its relative imports resolve)."""
    import importlib.util
    import subprocess
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:src/findpics/engine.py"], capture_output=True,
                         text=True, check=True).stdout
    path = ROOT / ".cache" / f"engine_{rev}.py"; path.parent.mkdir(exist_ok=True); path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"findpics.engine_{rev}", path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


def cmd_replay(a):
    global E
    if a.engine_rev:
        E = load_engine_rev(a.engine_rev)
    lab = json.loads(LABELS.read_text())
    keys = json.loads(KEYS.read_text()) if KEYS.exists() else None
    from findpics.agent import events
    items = pd.read_parquet(INDEX / "items.parquet")
    dated = pd.to_datetime(items["taken"], utc=True, errors="coerce", format="ISO8601").notna().to_numpy()
    ev = {str(i): int(x) for i, x, ok in zip(items["item_id"], events(items["taken"]), dated) if ok}
    rows = []
    for run in runs(a.runs):
        cache = json.loads((run / "judge_cache.json").read_text())
        pl = pools_of(run)
        tag = "first|" if run.name.startswith("mode_ens") else ""
        stub = StubJudge(cache, keys) if keys else None
        results, maxdiff = [], 0.0
        for spec_d, j in pl:
            j = j.copy()
            ident = j["where"] == "identity_match"
            if stub is not None:   # P from the judge cache by key (stub judge), checked against the saved pool
                q = red_q(spec_d["judge_question"])
                pc = np.array([stub.p(i, tag, q) if m else np.nan for i, m in zip(j.item_id, ident)], dtype=float)
                ok = ident & ~np.isnan(pc)
                maxdiff = max(maxdiff, float(np.nanmax(np.abs(pc[ok] - j.loc[ok, "p_attr"].to_numpy()))) if ok.any() else 0)
                j.loc[ok, "p_attr"] = pc[ok]
                srt = np.sort(j.loc[ident, "p_attr"].to_numpy())
                j.loc[ident, "rel"] = np.searchsorted(srt, j.loc[ident, "p_attr"].to_numpy(), side="right") / len(srt)
            spec = AlbumSpec(**{k: v for k, v in spec_d.items() if k in AlbumSpec.model_fields})
            ret = j[j.y & ident].copy(); ret["reason"] = "x"
            results.append(NS(spec=spec, judged=j, returned=ret, report=f"Album '{spec.name}': {len(ret)} items."))
        bi = run.name.endswith("_bi")
        E.PAIR_REJUDGE = None
        if bi:
            qa, qb = (s["judge_question"] for s, _ in pl)
            qbi = bipolar_q(qa, qb)
            if stub is None:
                print(run.name, "SKIP: bipolar run needs keys"); continue
            s_bi = {i: stub.p(i, "", qbi) for i in results[0].judged.item_id[results[0].judged["where"] == "identity_match"]}
            E.PAIR_REJUDGE = lambda pool, _qa, _qb, s=s_bi: {i: v for i, v in s.items() if v is not None}
        # split internals (same inputs _split_pair builds)
        A_, B_ = [r.judged[r.judged["where"] == "identity_match"] for r in results]
        pa = dict(zip(A_.item_id, A_.p_attr)); pb = dict(zip(B_.item_id, B_.p_attr))
        if E.PAIR_REJUDGE is not None:
            s = E.PAIR_REJUDGE(None, None, None); pa = {i: s[i] for i in pa if i in s}; pb = {i: 1 - s[i] for i in pa}
        ids = [i for i in pa if i in pb]
        x = pd.Series(E._logit([pa[i] for i in ids]) - E._logit([pb[i] for i in ids]), index=ids)
        g = pd.Series([ev.get(i, f"_{i}") for i in ids], index=ids)
        x = x.groupby(g).transform("median").to_numpy()
        info = gmm_report(x, g.to_numpy())
        E.make_exclusive(results, event_of=ev)
        E.PAIR_REJUDGE = None
        cnt = {}
        for r in results:
            names = [Path(p).name for p in r.returned.path]
            cnt[r.spec.name.split()[-1]] = collections.Counter(lab.get(n, "unl") for n in names)
        unclear = int(re.search(r"(\d+) photo\(s\) are not clearly either", results[0].report).group(1)) \
            if "not clearly either" in results[0].report else -1
        # equal to what the run saved on disk?
        same = all({re.sub(r"^[0-9a-f]{8}_", "", f.name) for f in (run / "turn_1" / r.spec.name).iterdir()
                    if f.is_symlink()} == {Path(p).name for p in r.returned.path} for r in results)
        h, f = cnt.get("heavier", {}), cnt.get("fit", {})
        rows.append(dict(run=run.name.replace("mode_", ""), heavier=sum(h.values()), fit=sum(f.values()), unclear=unclear,
                         hH=h.get("H", 0), hF=h.get("F", 0), fF=f.get("F", 0), fH=f.get("H", 0),
                         same_as_saved=same, cache_vs_pool=round(maxdiff, 6) if stub else None,
                         misses=stub.miss if stub else None, **{k: v for k, v in info.items()}))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False); os.chmod(a.out, 0o600)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("keys")
    r = sp.add_parser("replay"); r.add_argument("runs", nargs="*"); r.add_argument("--out")
    r.add_argument("--engine-rev")
    a = ap.parse_args()
    {"keys": cmd_keys, "replay": cmd_replay}[a.cmd](a)
