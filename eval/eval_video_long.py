"""LONG videos (>= 80 s): does a 4-s frame grid (phone sweep 2) find more than 3 frames (sweep 1: start/middle/end)?
(10-12; question from M49: RESULTS 37/37b measured 522 Pexels clips, NONE >= 80 s; Reza's library has 2,143 videos >= 80 s.)

Same harness as eval/eval_video_sampling.py (section 37), on a new public set:
  harvest  : Wikimedia Commons videos (free licences: CC BY / CC BY-SA / CC0 / PD), 80-900 s, searched with ~70
             everyday-life words (walk, dog, beach, birthday, train ride, ...); archive films, lectures, broadcasts,
             animations dropped by title + recorded date (>= 2000); <= 3 per uploader; stratified by length.
             -> data/public/raw/commons_long_videos/{candidates.json, posters/}
  posters  : contact sheets of the candidates' posters for an eye screen (personal-video-like or not)
  download : the kept videos (720p transcode, else 1080p, else original <= 600 MB, else 480p), per-file timeout
  frames   : 1-s grid (eval_video_sampling._decode, unchanged: first decoded frame at/after each second, <= 1280 px)
  embed    : grid vectors with PE-Core-B-16 (the phone's model) and PE-Core-L-14-336 (section 37's)
  gen      : 40 queries (seed 0): a RANDOM grid frame of a random video -> 9B describes it -> 9B writes one search
             query (eval_video.gen's prompts) -> product planner (plan_turn) -> judge question + looks
  oracle   : the 9B judge on EVERY grid frame for each query (shard by query)
  analyze  : per rule, the phone's search: each video's best frame by text score among the rule's frames, the judge's
             p on that frame >= 0.7 = found. Truth STRICT = >= 2 grid frames yes. Paired sign tests vs sweep 1.
  sheets   : videos rule A finds and rule B misses, both judged frames at 896 px (eye check)
Usage: python eval/eval_video_long.py harvest|posters|download|frames|embed|gen|oracle <shard> <n>|analyze|sheets A B
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_video_sampling as S  # noqa: E402  (reuse: _decode, even, every, sweep1, picks)

RAW = Path("data/public/raw/commons_long_videos")
VID = RAW / "videos"
GRID = Path("data/public/commons_long_grid")
OUT = Path("eval/video_sampling/long")
S.GRID = GRID; S.OUT = OUT                       # _decode / picks read these module globals
MODELS = {"B16": "hf-hub:timm/PE-Core-B-16", "L336": "hf-hub:timm/PE-Core-L-14-336"}
BUCKETS = [("80-180s", 80, 180), ("3-6min", 180, 360), ("6+min", 360, 1e9)]
NQ = 40
API = "https://commons.wikimedia.org/w/api.php"
UA = "find_pics-research/0.1 (video sampling eval; rezamshamji@gmail.com) python-requests"

TERMS = ["walk", "walking tour", "hike", "hiking trail", "beach", "waves", "dog", "puppy", "cat", "kitten", "birds",
         "zoo", "horse", "park", "playground", "children playing", "birthday", "wedding", "party", "concert",
         "festival", "parade", "fireworks", "carnival", "market", "street", "city walk", "train ride", "tram",
         "bus ride", "driving", "car ride", "cycling", "bicycle", "boat", "ferry", "river", "lake", "waterfall",
         "snow", "skiing", "garden", "flowers", "cooking", "kitchen", "restaurant", "food", "museum", "church",
         "temple", "football", "swimming", "fishing", "farm", "cows", "sheep", "airplane", "airport", "aquarium",
         "rain", "sunset", "mountain", "forest", "camping", "fair", "christmas", "holiday", "family", "vacation",
         "picnic", "harbour", "village", "dance", "procession", "skateboard", "ducks", "goats", "chickens"]
BAD = re.compile(r"\b(18\d\d|19[0-8]\d)\b|lecture|interview|conference|presentation|webinar|speech|\btalk\b|podcast|"
                 r"trailer|tutorial|animat|\bcgi\b|news|documentary|episode|fox sports|nasa|c-span|hearing|press|"
                 r"briefing|debate|session|sermon|lesson|course|seminar|music video|official|wikimania|wikipedia|"
                 r"meetup|panel|keynote|campaign|commercial|propaganda|\bfilm\b|movie|cartoon|screencast|slideshow|"
                 r"gameplay|minecraft|simulation|render|newsreel|archive|archival|reel\b|silent|vlog #|webcast|"
                 r"senate|congress|parliament|minister|president|council|lyric|karaoke|audiobook", re.I)


def _session():
    import requests
    s = requests.Session(); s.headers["User-Agent"] = UA
    return s


_last = [0.0]


def get(s, params, tries=10, post=False):
    params = {**params, "format": "json", "maxlag": 5}
    for k in range(tries):
        try:
            time.sleep(max(0.0, _last[0] + 0.4 - time.time())); _last[0] = time.time()
            r = s.post(API, data=params, timeout=60) if post else s.get(API, params=params, timeout=60)
            if r.status_code == 200:
                d = r.json()
                if d.get("error", {}).get("code") == "maxlag":
                    time.sleep(5 + 5 * k); continue
                return d
            ra = r.headers.get("Retry-After"); print("http", r.status_code, "retry-after", ra, flush=True)
            time.sleep(float(ra) if ra and ra.isdigit() else min(120, 5 * 2 ** k))
        except Exception as e:
            print("retry", k, e, flush=True); time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"API failed: {params}")


def harvest():
    """Search -> filter -> videoinfo (licence, recorded date, uploader, derivatives, poster) -> stratified candidates."""
    s = _session(); (RAW / "meta").mkdir(parents=True, exist_ok=True)
    seen = {}
    t0 = time.time()
    cache = RAW / "meta" / "search.json"
    if cache.exists():
        seen = json.load(open(cache))
    for term in ([] if seen else TERMS):
        n = 0
        for off in range(0, 200, 50):
            d = get(s, dict(action="query", generator="search", gsrsearch=f"filetype:video {term}", gsrnamespace=6,
                            gsrlimit=50, gsroffset=off, prop="imageinfo", iiprop="size|mime|user"))
            for p in d.get("query", {}).get("pages", {}).values():
                ii = (p.get("imageinfo") or [{}])[0]; dur = float(ii.get("duration") or 0)
                if p["title"] in seen or not (80 <= dur <= 900) or min(ii.get("width", 0), ii.get("height", 0)) < 480:
                    continue
                if BAD.search(p["title"]):
                    continue
                seen[p["title"]] = dict(title=p["title"], dur=dur, w=ii["width"], h=ii["height"], bytes=ii["size"],
                                        user=ii.get("user"), term=term); n += 1
            if "continue" not in d:
                break
        print(f"{time.time() - t0:6.0f}s {term!r}: +{n}, total {len(seen)}", flush=True)
    json.dump(seen, open(cache, "w"))
    # videoinfo in batches of 25, POST (non-Latin titles overflow a GET URL)
    titles = list(seen)
    for i in range(0, len(titles), 25):
        d = get(s, dict(action="query", titles="|".join(titles[i:i + 25]), prop="videoinfo",
                        viprop="derivatives|extmetadata|url", viurlwidth=480), post=True)
        if i % 500 == 0:
            print("videoinfo", i, "of", len(titles), flush=True)
        for p in d.get("query", {}).get("pages", {}).values():
            v = (p.get("videoinfo") or [{}])[0]; em = v.get("extmetadata", {})
            c = seen.get(p["title"])
            if c is None:
                continue
            c["licence"] = em.get("LicenseShortName", {}).get("value", "")
            c["date"] = re.sub("<[^>]+>", "", str(em.get("DateTimeOriginal", {}).get("value", "")))[:40]
            c["poster"] = v.get("thumburl"); c["original"] = v.get("url")
            c["derivatives"] = [dict(key=x.get("transcodekey"), w=x.get("width"), h=x.get("height"), src=x["src"])
                                for x in v.get("derivatives", [])]
    json.dump(list(seen.values()), open(RAW / "meta" / "all.json", "w"), indent=0)
    keep = []
    for c in seen.values():
        y = re.search(r"(19|20)\d\d", c.get("date", ""))
        lic = c.get("licence", "").lower()
        if not y or int(y.group(0)) < 2000:
            continue                                    # unknown or archival recording date
        if not any(k in lic for k in ("cc by", "cc-by", "cc0", "public domain", "pd")):
            continue
        keep.append(c)
    rng = np.random.default_rng(0); rng.shuffle(keep)
    per_user, out = {}, []
    for c in keep:
        if per_user.get(c["user"], 0) >= 3:
            continue
        per_user[c["user"]] = per_user.get(c["user"], 0) + 1; out.append(c)
    by_b = {b: [c for c in out if lo <= c["dur"] < hi] for b, lo, hi in BUCKETS}
    print("after filters:", len(keep), "| <= 3 per uploader:", len(out), {b: len(v) for b, v in by_b.items()})
    cand = sum((v[:110] for v in by_b.values()), [])
    for k, c in enumerate(cand):
        c["cid"] = k
    json.dump(cand, open(RAW / "candidates.json", "w"), indent=0)
    print("CANDIDATES", len(cand))


def posters():
    """Download each candidate's poster; contact sheets (5 x 4 at 480 px) with cid + duration + title for the screen."""
    from PIL import Image, ImageDraw
    s = _session(); P = RAW / "posters"; P.mkdir(exist_ok=True)
    cand = json.load(open(RAW / "candidates.json"))
    for c in cand:
        f = P / f"{c['cid']}.jpg"
        if f.exists() or not c.get("poster"):
            continue
        for k in range(4):
            try:
                r = s.get(c["poster"], timeout=60)
                if r.status_code == 200:
                    f.write_bytes(r.content); break
                time.sleep(2 ** k)
            except Exception as e:
                print("poster retry", c["cid"], e); time.sleep(2 ** k)
        time.sleep(0.2)
    sd = OUT / "poster_sheets"; sd.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(cand), 20):
        sheet = Image.new("RGB", (5 * 490, 4 * 330), "white"); dr = ImageDraw.Draw(sheet)
        for j, c in enumerate(cand[i:i + 20]):
            x, y = (j % 5) * 490, (j // 5) * 330
            f = P / f"{c['cid']}.jpg"
            if f.exists():
                im = Image.open(f).convert("RGB"); im.thumbnail((480, 290)); sheet.paste(im, (x + 5, y + 36))
            dr.text((x + 5, y + 2), f"#{c['cid']} {c['dur']:.0f}s {c['date'][:10]}", fill="black")
            dr.text((x + 5, y + 18), c["title"][5:70], fill="black")
        sheet.save(sd / f"posters_{i // 20:02d}.jpg", quality=85)
    print("POSTERS", len(list(P.glob("*.jpg"))), "of", len(cand), "sheets", sd)


def _pick_src(c):
    d = {x["key"]: x for x in c["derivatives"] if x["key"]}
    for k in ("720p.vp9.webm", "720p.webm", "1080p.vp9.webm", "1080p.webm"):
        if k in d:
            return d[k]["src"], k
    if c["bytes"] <= 600e6:
        return c["original"], "original"
    for k in ("480p.vp9.webm", "480p.webm"):
        if k in d:
            return d[k]["src"], k
    return None, None


def download():
    """Kept candidates (candidates.json minus RAW/rejected.txt) -> VID/<cid>.<ext>; 15 min timeout per file."""
    s = _session(); VID.mkdir(parents=True, exist_ok=True)
    cand = json.load(open(RAW / "candidates.json"))
    rej = set()
    if (RAW / "rejected.txt").exists():            # eye screen of the posters; '#' lines are comments
        rej = {int(x) for l in (RAW / "rejected.txt").read_text().splitlines() if not l.startswith("#")
               for x in re.findall(r"\d+", l)}
    rows, t0, total = [], time.time(), 0
    for c in cand:
        if c["cid"] in rej:
            continue
        src, key = _pick_src(c)
        if src is None:
            print("no source", c["cid"]); continue
        ext = ".webm" if key != "original" else Path(c["original"]).suffix
        f = VID / f"{c['cid']}{ext}"
        if not f.exists():
            tmp = f.with_suffix(".part"); ok = False
            for k in range(3):
                try:
                    st = time.time()
                    with s.get(src, stream=True, timeout=60) as r:
                        if r.status_code != 200:
                            print("http", r.status_code, c["cid"]); time.sleep(5 * 2 ** k); continue
                        with open(tmp, "wb") as fh:
                            for ch in r.iter_content(1 << 20):
                                fh.write(ch)
                                if time.time() - st > 900:
                                    raise TimeoutError("15 min per file")
                    tmp.rename(f); ok = True; break
                except Exception as e:
                    print("dl retry", c["cid"], e, flush=True); time.sleep(5 * 2 ** k)
            if not ok:
                continue
        total += f.stat().st_size
        rows.append(dict(cid=c["cid"], path=str(f), title=c["title"], dur=c["dur"], src=key, licence=c["licence"],
                         user=c["user"], date=c["date"]))
        if len(rows) % 10 == 0:
            print(f"{time.time() - t0:6.0f}s {len(rows)} videos {total / 1e9:.1f} GB", flush=True)
    pd.DataFrame(rows).to_parquet(RAW / "items.parquet")
    print("DOWNLOADED", len(rows), f"{total / 1e9:.1f} GB")


def _safe_decode(job):
    """S._decode unchanged; a video the decoder rejects mid-stream is DROPPED (reported), not half-gridded."""
    try:
        return S._decode(job)
    except Exception as e:
        print("DECODE FAILED, dropped:", job, repr(e), flush=True)
        return []


def frames(shard=0, n=1):
    """shard k of n (videos k, k+n, ...) -> grid_part_k.parquet; 'frames_merge' joins the parts in items order."""
    from concurrent.futures import ProcessPoolExecutor
    items = pd.read_parquet(RAW / "items.parquet"); GRID.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(int(r.cid), r.path) for r in items.itertuples()][shard::n]
    rows = []
    with ProcessPoolExecutor(int(os.environ.get("NPROC", 8))) as ex:
        for k, r in enumerate(ex.map(_safe_decode, jobs, chunksize=1)):
            rows += r
            if k % 10 == 0:
                print("decoded", k, "of", len(jobs), "frames", len(rows), flush=True)
    g = pd.DataFrame(rows).reset_index(drop=True)
    g.to_parquet(OUT / f"grid_part_{shard}.parquet")
    print("GRID PART", shard, len(g), "frames of", g.item_row.nunique(), "videos")


def frames_merge():
    items = pd.read_parquet(RAW / "items.parquet")
    g = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob("grid_part_*.parquet"))])
    order = {int(c): i for i, c in enumerate(items.cid)}
    g = g.assign(o=g.item_row.map(order)).sort_values(["o", "sec"]).drop(columns="o").reset_index(drop=True)
    g.to_parquet(OUT / "grid.parquet")
    print("GRID", len(g), "frames of", g.item_row.nunique(), "videos of", len(items))


PACK = Path("data/public/commons_long_grid_pack")


def pack(nparts=8):
    """The 1-s grid's JPEGs (bytes unchanged) packed into nparts parquet files in grid order. Why: on 10-12 the lab
    filesystem took ~185 ms per small-file open (200 random frames in 37 s), i.e. ~2.4 h to read the grid once."""
    from concurrent.futures import ThreadPoolExecutor
    g = pd.read_parquet(OUT / "grid.parquet"); PACK.mkdir(parents=True, exist_ok=True)
    rd = lambda r: (GRID / f"{r[0]}_{r[1]}.jpg").read_bytes()
    keys = list(zip(g.item_row, g.sec)); t0 = time.time()
    for k, idx in enumerate(np.array_split(np.arange(len(g)), nparts)):
        f = PACK / f"part_{k}.parquet"
        if f.exists():
            continue
        with ThreadPoolExecutor(64) as ex:
            b = list(ex.map(rd, [keys[i] for i in idx]))
        pd.DataFrame(dict(item_row=g.item_row.to_numpy()[idx], sec=g.sec.to_numpy()[idx], jpg=b)).to_parquet(f)
        print(f"{time.time() - t0:6.0f}s part {k}: {len(idx)} frames", flush=True)
    print("PACKED", len(g))


def grid_images():
    """Grid frames as JPEG bytes in grid.parquet order (from the pack; falls back to the files)."""
    g = pd.read_parquet(OUT / "grid.parquet")
    if not (PACK / "part_0.parquet").exists():
        return [(GRID / f"{r.item_row}_{r.sec}.jpg").read_bytes() for r in g.itertuples()]
    P = pd.concat([pd.read_parquet(f) for f in sorted(PACK.glob("part_*.parquet"), key=lambda f: int(f.stem.split("_")[1]))])
    assert (P.item_row.to_numpy() == g.item_row.to_numpy()).all() and (P.sec.to_numpy() == g.sec.to_numpy()).all()
    return list(P.jpg)


def _img(b):
    import io
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def embed():
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from findpics.models import ImageTextEncoder
    g = pd.read_parquet(OUT / "grid.parquet"); ex = ThreadPoolExecutor(8); B = grid_images()
    for tag, name in MODELS.items():
        if (OUT / f"grid_vecs_{tag}.npy").exists():
            continue
        enc = ImageTextEncoder(name); vs = []
        for i in range(0, len(g), 256):
            vs.append(enc.images(list(ex.map(_img, B[i:i + 256]))))
            if i % 5120 == 0:
                print(tag, i, "of", len(g), flush=True)
        np.save(OUT / f"grid_vecs_{tag}.npy", np.concatenate(vs)); del enc
    json.dump(MODELS, open(OUT / "embed.json", "w")); print("EMBED done", len(g))


def gen():
    """eval_video.gen's prompts on a random 1-s grid frame of each of NQ random videos, then the product planner."""
    from datetime import date
    from PIL import Image
    from findpics.converse import plan_turn
    from findpics.vlm import VLLMJudge, _data_url
    if (OUT / "queries.json").exists():
        print("queries.json exists; gen not rerun"); return
    g = pd.read_parquet(OUT / "grid.parquet"); J = VLLMJudge(gpu_mem=0.8)
    rng = np.random.default_rng(0); qs = []
    vids = sorted(g.item_row.unique())
    for k, item in enumerate(rng.choice(vids, NQ, replace=False)):
        secs = g.sec[g.item_row == item].to_numpy(); sec = int(rng.choice(secs))
        im = Image.open(GRID / f"{item}_{sec}.jpg").convert("RGB")
        cap = J._chat([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": _data_url(im)}},
                       {"type": "text", "text": "Describe what is visible in this video frame in 2 sentences."}]}]],
                      J.SP(temperature=0.0, max_tokens=120))[0].outputs[0].text.strip()
        q = J.text("Write ONE short search query (under 12 words) a person might type into their photo app to find a "
                   f"video showing this scene. Scene: {cap}\nReturn only the query.", max_tokens=40).strip().splitlines()[0].strip('"')
        P = plan_turn(q, J.text, today=date(2026, 10, 2)); a = P.albums[0]
        jq = a.judge_question or q
        qs.append(dict(k=k, item_row=int(item), sec=sec, caption=cap, query=q, judge_question=jq,
                       looks=a.looks or [q], plan=a.model_dump()))
        print(k, item, sec, repr(q), "->", repr(jq), a.looks, flush=True)
    if (OUT / "queries.json").exists():          # never replace queries an oracle may already be using
        json.dump(qs, open(OUT / "queries_rerun.json", "w"), indent=1); return
    json.dump(qs, open(OUT / "queries.json", "w"), indent=1)


def oracle(shard, n):
    from PIL import Image
    from concurrent.futures import ThreadPoolExecutor
    from findpics.vlm import VLLMJudge
    J = VLLMJudge(gpu_mem=0.85); ex = ThreadPoolExecutor(8)
    g = pd.read_parquet(OUT / "grid.parquet"); B = grid_images(); print("grid bytes loaded", len(B), flush=True)
    while not (OUT / "queries.json").exists():   # submitted beside the gen job: imports take ~1.5 h on the slow FS
        time.sleep(30)
    for q in json.load(open(OUT / "queries.json"))[shard::n]:
        f = OUT / f"oracle_{q['k']}.parquet"
        if f.exists():
            continue
        p, t0 = [], time.time()
        for i in range(0, len(g), 128):
            p += J.p_yes(list(ex.map(_img, B[i:i + 128])), q["judge_question"])
        pd.DataFrame(dict(item_row=g.item_row, sec=g.sec, p=np.array(p, dtype=np.float32))).to_parquet(f)
        print(f"q{q['k']} {q['judge_question']!r}: grid frames yes {int((np.array(p) >= .7).sum())} "
              f"({time.time() - t0:.0f} s for {len(g)} frames)", flush=True)


# ---- rules (target times for a clip of d seconds) ----
def sweep1(d):
    return [0.0] + S.sweep1(d)            # shipped sweep 1: cover + middle + end (= "fixed 3" layout)


def hybrid(rule, L):
    """sweep 1 below L seconds; `rule` (which replaces sweep 1's frames, as the phone's sweep 2 does) from L up."""
    return lambda d: sweep1(d) if d < L else rule(d)


UNCAP = 10 ** 6
SWEEP2 = S.every(4, cap=40)    # the phone's sweep 2 AS SHIPPED: videoSampleTimes(every 4 s, maxFrames 40), LazyVideo.swift
RULES = {
    "sweep1: start+mid+end": sweep1,
    "fixed 5": lambda d: S.even(d, 5),
    "fixed 8": lambda d: S.even(d, 8),
    "fixed 16": lambda d: S.even(d, 16),
    "every 8 s": S.every(8, cap=UNCAP),
    "every 4 s (no cap)": S.every(4, cap=UNCAP),
    "sweep2 shipped: 4 s, cap 40": SWEEP2,
    "every 1 s (grid)": S.every(1, cap=UNCAP),
    "s1; sweep2 if >=180s": hybrid(SWEEP2, 180),
    "s1; sweep2 if >=360s": hybrid(SWEEP2, 360),
    "s1; every 8 s if >=180s": hybrid(S.every(8, cap=UNCAP), 180),
    "s1 + 1 per 30 s": lambda d: sorted(set(sweep1(d)) | set(S.even(d, max(3, int(d // 30) + 1)))),
}
BASE = "sweep1: start+mid+end"


def bucket(d):
    return next(b for b, lo, hi in BUCKETS if lo <= d < hi)


def rule_frames(rule, durs):
    sel, cost = {}, {}
    for item, (dur, last) in durs.items():
        ts = RULES[rule](dur)
        sel[item] = sorted({int(min(max(round(t), 0), last)) for t in ts})
        cost[item] = len(sel[item])
    return sel, cost


def _text_vecs(model_tag, Q):
    f = OUT / f"text_vecs_{model_tag}.npy"
    if f.exists():
        return np.load(f)
    from findpics.models import ImageTextEncoder
    enc = ImageTextEncoder(MODELS[model_tag], device="cpu")
    T = []
    for q in Q:
        t = enc.texts(q["looks"]).astype(np.float32).mean(0); T.append(t / np.linalg.norm(t))
    np.save(f, np.stack(T)); return np.stack(T)


def analyze(model_tag="B16"):
    from math import comb
    g = pd.read_parquet(OUT / "grid.parquet")
    durs = {i: (d.dur.iloc[0], int(d.sec.max())) for i, d in g.groupby("item_row")}
    Qs = json.load(open(OUT / "queries.json")); Q = {q["k"]: q for q in Qs}
    V = np.load(OUT / f"grid_vecs_{model_tag}.npy").astype(np.float32); T = _text_vecs(model_tag, Qs)
    O = {}
    for f in sorted(OUT.glob("oracle_*.parquet")):
        k = int(f.stem.split("_")[1]); o = pd.read_parquet(f)
        assert (o.item_row.to_numpy() == g.item_row.to_numpy()).all()
        o["look"] = V @ T[k]; O[k] = o
    bk = {i: bucket(d) for i, (d, _) in durs.items()}
    rows = []
    for rule in RULES:
        sel, cost = rule_frames(rule, durs)
        for k, o in O.items():
            yes = o[o.p >= 0.7].groupby("item_row").size()
            P = S.picks(o, sel)
            for item, (sec, p, s) in P.items():
                rows.append(dict(rule=rule, k=k, item=item, bucket=bk[item], dur=durs[item][0], frames=cost[item],
                                 sec=sec, p=p, truth=int(yes.get(item, 0) >= 1), strict=int(yes.get(item, 0) >= 2),
                                 found=int(p >= 0.7), seed=int(Q[k]["item_row"] == item)))
    df = pd.DataFrame(rows); df.to_parquet(OUT / f"per_video_{model_tag}.parquet")
    L = [f"ranking model {MODELS[model_tag]}; {len(O)} queries; {len(durs)} videos; grid frames {len(g)}",
         "videos per bucket: " + ", ".join(f"{b} {sum(v == b for v in bk.values())}" for b, _, _ in BUCKETS)]

    def sign(a, c):
        n = a + c
        return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(a, c) + 1)) / 2 ** n) if n else 1.0

    for truth_col, title in (("strict", "STRICT truth (>= 2 grid frames yes)"), ("truth", "LENIENT truth (>= 1 grid frame yes)")):
        L.append(f"\n{title}: found/truth (query, video) pairs; paired vs '{BASE}': +only this rule / -only sweep 1, sign-test p")
        L.append(f"{'rule':26s} {'fr/vid all':>10s} " + " ".join(f"{b + ' fr':>9s}" for b, _, _ in BUCKETS) + "  "
                 + "  ".join(f"{b:>28s}" for b in ["all"] + [b for b, _, _ in BUCKETS]))
        base = df[(df.rule == BASE) & (df[truth_col] == 1)].set_index(["k", "item"])
        for rule in RULES:
            d = df[df.rule == rule]; t = d[d[truth_col] == 1].set_index(["k", "item"])
            u = d.drop_duplicates("item")
            s = f"{rule:26s} {u.frames.mean():10.1f} " + " ".join(f"{u[u.bucket == b].frames.mean():9.1f}" for b, _, _ in BUCKETS) + "  "
            for b in ["all"] + [b for b, _, _ in BUCKETS]:
                tt = t if b == "all" else t[t.bucket == b]; bb = base.loc[tt.index]
                a = int(((tt.found == 1) & (bb.found == 0)).sum()); c = int(((tt.found == 0) & (bb.found == 1)).sum())
                s += f"  {int(tt.found.sum()):>4d}/{len(tt):<4d} +{a:<3d}-{c:<3d} p={sign(a, c):.2f}"
            L.append(s)
    # seed video excluded (the query was written from one of its frames: the most-favourable case for any rule)
    t = df[(df.strict == 1) & (df.seed == 0)]
    L.append(f"\nSTRICT truth without the query's own seed video ({t[t.rule == BASE].shape[0]} pairs): found per rule: "
             + "; ".join(f"{r} {int(t[t.rule == r].found.sum())}" for r in RULES))
    t = df[(df.strict == 1)]
    L.append("truth pairs per query (strict): " + str(t[t.rule == BASE].groupby("k").size().to_dict()))
    txt = "\n".join(L); print(txt)
    (OUT / f"report_{model_tag}.txt").write_text(txt + "\n")


def sheets(a, b, model_tag="B16", limit=40):
    """Videos rule a finds and rule b misses (strict): a's judged frame | b's judged frame, each at 896 px, + query."""
    from PIL import Image, ImageDraw
    df = pd.read_parquet(OUT / f"per_video_{model_tag}.parquet"); Q = {q["k"]: q for q in json.load(open(OUT / "queries.json"))}
    A = df[df.rule == a].set_index(["k", "item"]); B = df[df.rule == b].set_index(["k", "item"])
    diff = [(k, i) for (k, i), r in A.iterrows() if r.found and r.strict and not B.loc[(k, i)].found]
    print(len(diff), f"(query, video) pairs found by '{a}' and missed by '{b}' (strict truth)")
    rng = np.random.default_rng(1); rng.shuffle(diff)
    sd = OUT / ("sheets_" + re.sub(r"[^a-z0-9]+", "_", f"{a} vs {b}".lower())); sd.mkdir(exist_ok=True)
    info = []
    for n, (k, i) in enumerate(diff[:int(limit)]):
        ims = []
        for lab, r in ((a, A.loc[(k, i)]), (b, B.loc[(k, i)])):
            im = Image.open(GRID / f"{i}_{int(r.sec)}.jpg").convert("RGB"); im.thumbnail((896, 896))
            c = Image.new("RGB", (896, im.height + 30), "white"); c.paste(im, (0, 30))
            ImageDraw.Draw(c).text((6, 8), f"{lab}: t={int(r.sec)}s p={r.p:.2f}", fill="black"); ims.append(c)
        h = max(x.height for x in ims); sheet = Image.new("RGB", (896 * 2 + 10, h + 30), "white")
        for j, x in enumerate(ims):
            sheet.paste(x, (j * 906, 30))
        ImageDraw.Draw(sheet).text((6, 8), f"q{k}: {Q[k]['judge_question']}  | video {i} ({A.loc[(k, i)].dur:.0f} s)", fill="black")
        fn = sd / f"{n:02d}_q{k}_v{i}.jpg"; sheet.save(fn, quality=90)
        info.append(dict(n=n, k=k, item=i, sheet=str(fn), q=Q[k]["judge_question"]))
    json.dump(info, open(sd / "index.json", "w"), indent=1)
    print("wrote", len(info), "sheets to", sd)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "frames":
        frames(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "frames_merge":
        frames_merge()
    elif c == "oracle":
        oracle(int(sys.argv[2]), int(sys.argv[3]))
    elif c == "sheets":
        sheets(*sys.argv[2:])
    elif c == "analyze":
        analyze(*sys.argv[2:])
    else:
        {"pack": pack, "harvest": harvest, "posters": posters, "download": download, "frames": frames, "embed": embed, "gen": gen}[c]()
