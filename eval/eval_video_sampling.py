"""Video frame sampling sweep (MAC_INBOX M20; JOURNAL 10-08): how findable is a video at each sampling rule, and at what
cost (frames per video = image vectors to compute on the phone)?

Nothing chose the phone's VideoFrames.sampleEach default (every 2 s, at most 40, = media.sample_video_frames); the
earlier video eval (eval_video.py) held it fixed. This sweep varies it, including poster-only (index each video from
one still image, what the phone's first pass now does).

Mechanism. One dense GRID of frames per video: t = 0, 1, 2, ... s (every second, NO cap), decoded like the server's
sampler (first decoded frame at or after the target; display rotation applied), saved at <= 1280 px (the phone samples
at 1280; the judge sees 896 either way). Every rule is a set of target times, each snapped to the nearest grid frame
(<= 0.5 s away, the phone's AVAssetImageGenerator tolerance is +-0.5 s). So all rules are judged on the same frames.
  frames  : durations + grid frames -> data/public/pexels_grid/<item>_<sec>.jpg, eval/video_sampling/grid.parquet
  embed   : PE-Core-L-14-336 (the phone's image model) vector of every grid frame -> eval/video_sampling/grid_vecs.npy
  oracle  : per query (eval/video/plan_<k>.json: the same plans and judge questions as eval_video.py): the 9B judge on
            EVERY grid frame -> eval/video_sampling/oracle_<k>.parquet (shard by query)
  analyze : per rule: the phone's search picks each video's best frame by the text-vector score among the rule's frames
            and the judge decides on that one frame (Search.swift bestT; VIDEO_FRAMES = 1). Found = that p >= 0.7.
            TRUTH = videos where some grid frame is judged yes (lenient; inflated by single-frame judge flukes,
            JOURNAL 10-02 15:00) and, stricter, where >= 2 grid frames are yes (the content lasts ~2 s or more).
            Recall pooled over the 24 queries, split by clip length; plus top-50-by-score recall (fast-mode proxy).
  sheets  : raw look at judge resolution (896 px): videos one rule finds and another misses, both judged frames.
Usage (vLLM env, GPU for embed/oracle): python eval/eval_video_sampling.py frames|embed|analyze|sheets A B|oracle <shard> <n>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IDX = "data/public/index_pexels"
GRID = Path("data/public/pexels_grid")
OUT = Path("eval/video_sampling"); OUT.mkdir(parents=True, exist_ok=True)
SIDE = 1280
BUCKETS = [("<10s", 0, 10), ("10-30s", 10, 30), ("30-80s", 30, 80), (">=80s", 80, 1e9)]


def bucket(d):
    return next(b for b, lo, hi in BUCKETS if lo <= d < hi)


def _decode(args):
    item, path = args
    import av
    from PIL import Image
    rows = []
    with av.open(path) as c:
        s = c.streams.video[0]; s.thread_type = "AUTO"
        dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else 0.0)
        if dur <= 0:
            dur = 1.0
        targets = list(range(0, int(np.floor(max(dur - 0.05, 0))) + 1))
        ti = 0
        for frame in c.decode(s):
            if frame.time is None:
                continue
            if frame.time + 1e-3 >= targets[ti]:
                f = GRID / f"{item}_{targets[ti]}.jpg"
                if not f.exists():
                    im = frame.to_image()
                    rot = getattr(frame, "rotation", 0) or 0
                    if rot:
                        im = im.rotate(rot, expand=True)
                    if max(im.size) > SIDE:
                        im.thumbnail((SIDE, SIDE), Image.BICUBIC)
                    im.save(f, quality=90)
                rows.append(dict(item_row=item, sec=targets[ti], t=float(frame.time), dur=dur))
                ti += 1
                if ti >= len(targets):
                    break
    return rows


def frames():
    from concurrent.futures import ProcessPoolExecutor
    items = pd.read_parquet(Path(IDX) / "items.parquet"); GRID.mkdir(parents=True, exist_ok=True)
    jobs = [(i, p) for i, p in enumerate(items.path)]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(_decode, jobs, chunksize=4)):
            rows += r
            if k % 50 == 0:
                print("decoded", k, "of", len(jobs), "frames", len(rows), flush=True)
    g = pd.DataFrame(rows).reset_index(drop=True)
    g.to_parquet(OUT / "grid.parquet")
    print("GRID", len(g), "frames of", g.item_row.nunique(), "videos; total duration", round(g.groupby("item_row").dur.first().sum()))


def embed():
    from PIL import Image
    from findpics.models import DEFAULT_CLIP, ImageTextEncoder
    g = pd.read_parquet(OUT / "grid.parquet"); enc = ImageTextEncoder(DEFAULT_CLIP); vs = []
    for i in range(0, len(g), 128):
        ims = [Image.open(GRID / f"{r.item_row}_{r.sec}.jpg").convert("RGB") for r in g.iloc[i:i + 128].itertuples()]
        vs.append(enc.images(ims))
    np.save(OUT / "grid_vecs.npy", np.concatenate(vs)); json.dump(dict(model=DEFAULT_CLIP), open(OUT / "embed.json", "w"))
    print("EMBED", len(g), DEFAULT_CLIP, flush=True)


def oracle(shard, n):
    from PIL import Image
    from findpics.models import DEFAULT_CLIP, ImageTextEncoder
    from findpics.vlm import VLLMJudge
    import time
    enc = ImageTextEncoder(DEFAULT_CLIP); J = VLLMJudge(gpu_mem=0.7)
    for _ in range(180):                       # the prep job may still be embedding
        if (OUT / "embed.json").exists():
            break
        time.sleep(10)
    g = pd.read_parquet(OUT / "grid.parquet"); V = np.load(OUT / "grid_vecs.npy").astype(np.float32)
    for q in json.load(open("eval/video/queries.json"))[shard::n]:
        f = OUT / f"oracle_{q['k']}.parquet"
        if f.exists():
            continue
        pl = json.load(open(f"eval/video/plan_{q['k']}.json"))
        jq = pl["judge_question"] or q["query"]; looks = pl["plan"].get("looks") or [q["query"]]
        T = enc.texts(looks).astype(np.float32).mean(0); T /= np.linalg.norm(T)
        p = []
        for i in range(0, len(g), 96):
            ims = [Image.open(GRID / f"{r.item_row}_{r.sec}.jpg").convert("RGB") for r in g.iloc[i:i + 96].itertuples()]
            p += J.p_yes(ims, jq)
        pd.DataFrame(dict(item_row=g.item_row, sec=g.sec, look=V @ T, p=np.array(p))).to_parquet(f)
        print(f"q{q['k']} '{q['query']}' -> '{jq}': grid frames yes {int((np.array(p) >= .7).sum())}", flush=True)


# ---- rules: target times for a clip of `dur` seconds (the phone's formula: VideoFrames.sampleEach) ----

def even(dur, n):
    return [0.0] if n == 1 else list(np.linspace(0, max(dur - 0.05, 0), n))


def every(s, cap=40):
    return lambda d: even(d, max(1, min(cap, int(d // s) + 1)))


def thirds(d):
    """Frames at 1/3 and 2/3 beside the cover (asked for on 10-09, measured worse than "fixed 3": not shipped)."""
    return [d / 3, 2 * d / 3] if d >= 3 else [d / 2] if d >= 1 else []


def sweep1(d):
    """FindPicsCore.videoSweep1Times: decoded frames beside the cover = "fixed 3" without its t = 0 (middle and end of
    [0, d - 0.05]). Frames >= 1 s apart (2x the generator's 0.5 s tolerance) are distinct: middle + end once the middle
    is >= 1 s in, the middle alone from 1 s, none below (the cover is the clip)."""
    end = max(d - 0.05, 0)
    return [end / 2, end] if end / 2 >= 1 else [d / 2] if d >= 1 else []


RULES = {
    "poster (t=0)": lambda d: [0.0],
    "middle": lambda d: [d / 2],
    "fixed 2": lambda d: even(d, 2),
    "fixed 3": lambda d: even(d, 3),
    "fixed 5": lambda d: even(d, 5),
    "every 8 s": every(8),
    "every 4 s": every(4),
    "every 2 s (current)": every(2),
    "every 1 s": every(1),
    # length-dependent candidate (added after the first 12 queries: 3 frames per clip did as well as 2 s): at least 3
    # frames, else every 4 s (cap 40)
    "every 4 s, >= 3": lambda d: even(d, max(3, min(40, int(d // 4) + 1))),
    # the phone's two-sweep frames pass (10-09, FindPicsCore videoSweep1Times / videoSweep2Times): sweep 1 = the cover
    # (t=0) + the middle and end frames ("fixed 3" layout); sweep 2 ships as exactly "every 4 s" (sweep-1 frames reused
    # only where they fall on its times). Measured, not shipped: "cover + thirds" (the 1/3 + 2/3 layout first asked
    # for) and "every 4 s + sweep 1" (keeping every sweep-1 frame beside the grid).
    "cover + thirds": lambda d: [0.0] + thirds(d),
    "cover + mid + end (sweep 1)": lambda d: [0.0] + sweep1(d),
    "every 4 s + sweep 1 (sweep 2)": lambda d: sorted(set(every(4)(d)) | set(sweep1(d))),
}


def rule_frames(rule, durs):
    """item -> sorted unique grid seconds the rule's targets snap to; and the rule's own frame count (cost)."""
    sel, cost = {}, {}
    for item, (dur, last) in durs.items():
        ts = RULES[rule](dur)
        cost[item] = len(ts)
        sel[item] = sorted({int(min(max(round(t), 0), last)) for t in ts})
    return sel, cost


def load():
    g = pd.read_parquet(OUT / "grid.parquet")
    durs = {i: (d.dur.iloc[0], int(d.sec.max())) for i, d in g.groupby("item_row")}
    Q = {q["k"]: q for q in json.load(open("eval/video/queries.json"))}
    O = {int(f.stem.split("_")[1]): pd.read_parquet(f) for f in sorted(OUT.glob("oracle_*.parquet"))}
    return g, durs, Q, O


def picks(o, sel):
    """Per video: the rule's frame with the best text score, its p, and the video's score (max over the rule's frames)."""
    o = o.set_index(["item_row", "sec"])
    out = {}
    for item, secs in sel.items():
        sub = o.loc[[(item, s) for s in secs]]
        j = int(np.argmax(sub.look.to_numpy()))
        out[item] = (secs[j], float(sub.p.iloc[j]), float(sub.look.iloc[j]))
    return out


def analyze():
    g, durs, Q, O = load()
    print("queries with oracle:", len(O), "of", len(Q))
    bk = {i: bucket(d) for i, (d, _) in durs.items()}
    nb = pd.Series(bk).value_counts()
    rows, cmp = [], {}
    for rule in RULES:
        sel, cost = rule_frames(rule, durs)
        for k, o in O.items():
            yes = o[o.p >= 0.7].groupby("item_row").size()
            P = picks(o, sel)
            top = set(sorted(P, key=lambda i: -P[i][2])[:50])
            for item, (sec, p, s) in P.items():
                cmp[(rule, k, item)] = (sec, p)
                rows.append(dict(rule=rule, k=k, item=item, bucket=bk[item], frames=cost[item],
                                 truth=int(yes.get(item, 0) >= 1), strict=int(yes.get(item, 0) >= 2),
                                 found=int(p >= 0.7), top50=int(p >= 0.7 and item in top)))
    df = pd.DataFrame(rows)
    order = list(RULES)
    lines = []
    def table(truth_col, title):
        lines.append(f"\n{title}  (recall = found / truth videos, pooled over {len(O)} queries)")
        hdr = f"{'rule':22s} {'fr/video':>8s} {'all':>14s}" + "".join(f" {b:>14s}" for b, _, _ in BUCKETS) + f" {'top50 all':>12s}"
        lines.append(hdr)
        for rule in order:
            d = df[df.rule == rule]; t = d[d[truth_col] == 1]
            fpv = d.drop_duplicates("item").frames.mean()
            cell = lambda x: f"{x.found.sum():>4d}/{len(x):<4d} {x.found.mean() if len(x) else float('nan'):.3f}"
            s = f"{rule:22s} {fpv:8.2f} {cell(t):>14s}"
            for b, _, _ in BUCKETS:
                s += f" {cell(t[t.bucket == b]):>14s}"
            s += f" {t.top50.sum():>4d}/{len(t):<4d}{t.top50.mean():.3f}"
            lines.append(s)
    lines.append(f"videos per length bucket: {nb.to_dict()}; grid frames {len(g)}")
    lines.append("frames per video by bucket (cost): " + "; ".join(
        f"{rule}: " + ", ".join(f"{b} {df[(df.rule == rule) & (df.bucket == b)].drop_duplicates('item').frames.mean():.1f}"
                                for b, _, _ in BUCKETS) for rule in order))
    table("truth", "LENIENT truth: >= 1 grid frame (every 1 s) judged yes")
    table("strict", "STRICT truth: >= 2 grid frames judged yes")
    # paired: the same (query, truth video) under two rules; only the discordant ones carry information (sign test)
    from math import comb
    lines.append("\nPAIRED vs 'every 2 s (current)' (strict truth): videos only the other rule finds / only 2 s finds; "
                 "two-sided sign-test p")
    base = df[(df.rule == "every 2 s (current)") & (df.strict == 1)].set_index(["k", "item"]).found
    for rule in order:
        if rule == "every 2 s (current)":
            continue
        o = df[(df.rule == rule) & (df.strict == 1)].set_index(["k", "item"]).found.reindex(base.index)
        for b, lo, hi in [("all", 0, 1e9)] + BUCKETS:
            m = df[(df.rule == rule) & (df.strict == 1)].set_index(["k", "item"]).bucket.reindex(base.index)
            sel = (m == b) if b != "all" else m.notna()
            a = int(((o == 1) & (base == 0))[sel].sum()); c = int(((o == 0) & (base == 1))[sel].sum()); n = a + c
            pv = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(a, c) + 1)) / 2 ** n) if n else 1.0
            if b == "all":
                line = f"{rule:22s} all: +{a} -{c} p={pv:.2f}"
            else:
                line += f" | {b}: +{a} -{c}"
        lines.append(line)
    txt = "\n".join(lines); print(txt)
    (OUT / "report.txt").write_text(txt + "\n")
    df.to_parquet(OUT / "per_video.parquet")


def sheets(a, b, limit=24):
    """Videos rule `a` finds and rule `b` misses (strict truth): both judged frames side by side at the judge's 896 px."""
    from PIL import Image, ImageDraw
    df = pd.read_parquet(OUT / "per_video.parquet"); g, durs, Q, O = load()
    A = df[df.rule == a].set_index(["k", "item"]); B = df[df.rule == b].set_index(["k", "item"])
    diff = [(k, i) for (k, i), r in A.iterrows() if r.found and r.strict and not B.loc[(k, i)].found]
    print(len(diff), f"videos found by '{a}' and missed by '{b}' (strict truth)")
    sa, _ = rule_frames(a, durs); sb, _ = rule_frames(b, durs)
    sd = OUT / f"sheets_{a.split()[0]}_{a.split()[1] if len(a.split()) > 1 else ''}_vs_{b.split()[0]}_{b.split()[1] if len(b.split()) > 1 else ''}".replace("(", "").replace(")", "").replace("=", "")
    sd.mkdir(exist_ok=True)
    for n, (k, i) in enumerate(diff[:limit]):
        o = O[k].set_index(["item_row", "sec"])
        pa = picks(O[k], {i: sa[i]})[i]; pb = picks(O[k], {i: sb[i]})[i]
        ims = []
        for lab, (sec, p, _) in ((a, pa), (b, pb)):
            im = Image.open(GRID / f"{i}_{sec}.jpg").convert("RGB"); im.thumbnail((896, 896))
            c = Image.new("RGB", (896, im.height + 40), "white"); c.paste(im, (0, 40))
            ImageDraw.Draw(c).text((6, 6), f"{lab}: t={sec}s p={p:.2f}", fill="black"); ims.append(c)
        h = max(x.height for x in ims); sheet = Image.new("RGB", (896 * 2 + 10, h + 40), "white")
        for j, x in enumerate(ims):
            sheet.paste(x, (j * 906, 40))
        ImageDraw.Draw(sheet).text((6, 6), f"q{k}: {O[k].shape[0] and json.load(open(f'eval/video/plan_{k}.json'))['judge_question']}  "
                                   f"video {i} ({durs[i][0]:.0f} s)", fill="black")
        sheet.save(sd / f"{n:02d}_q{k}_v{i}.jpg", quality=88)
    print("wrote", min(len(diff), limit), "sheets to", sd)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "oracle":
        oracle(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "sheets":
        sheets(sys.argv[2], sys.argv[3])
    else:
        {"frames": frames, "embed": embed, "analyze": analyze, "prep": lambda: (frames(), embed())}[c]()
