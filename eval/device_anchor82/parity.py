"""M82 (B) WRITER PARITY SET: 20 public Wikimedia Commons iPhone photos by URL + the cluster's MLX lines (base 2B and the
trained writer p2b_all_final_mlx) from the exact downloaded file, for the phone's #writerparity.

URLS. Commons now serves thumbnails ONLY at its standard widths (tested 11-05: 336/400/448/450/480/640/800 px -> HTTP 400
"Use thumbnail sizes listed on https://w.wiki/GHai"; 250/330/500/960 -> 200). So no URL can serve a 448 px long side. Rule:
per photo the standard width in {250, 330, 500, 960} whose long side is closest to 448 (4:3 landscape -> 500 x 375, 3:4
portrait -> 330 x 440, 9:16 portrait -> 250 x 444). The phone's FactsWriter path then applies processing.resize =
448 x 448 (MediaProcessing.apply: affine scale min(448/w, 448/h), up OR down) before the Qwen3-VL processor.
CLUSTER INPUT = the downloaded file scaled by that same factor with Pillow BICUBIC to (round(w*s), round(h*s)), saved
lossless (PNG), then mlx-vlm's own processor (smart_resize to multiples of 32, bicubic) - the closest CPU stand-in for
CoreImage's affine + bicubic resample; the residual pixel difference is part of what parity measures.

SELECTION (label-independent, from the parity pool commons1k): 5 photos where the BASE writer loops on its vLLM line:
repeat-stop v2 fired (seeded random), topped up with lines that ran to the 120-token cap without a stop (the base 2B's
loops are mostly not comma lists, so v2 rarely fires on them); then 15 for diversity = k-means (k=15, seed 82) on the remaining photos' PE-Core-B-16
vectors, the photo nearest each centre.
  python eval/device_anchor82/parity.py pick        -> data/public/device_anchor82/parity/{img/,in/,photos.json}
  (GPU) bash eval/device_anchor82/launch.sh parity20 mlx -> data/public/device_anchor82/out/writer_parity20/{base,trained}_0.jsonl
  python eval/device_anchor82/parity.py assemble    -> eval/device_anchor82/writer_parity.json
"""
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from findpics.facts_line import apply_repeat_stop, repeat_scan  # noqa: E402

D = Path("data/public/device_anchor82")
P = D / "parity"
STD = [250, 330, 500, 960]
UA = "find_pics-research/0.1 (device anchor eval; rezamshamji@gmail.com) python-requests"


def base_lines():
    """the base writer's vLLM lines on the parity pool (commons1k) - used only to find the photos where it loops"""
    out = {}
    for l in open(D / "out" / "vllm_pool6k" / "base_vllm.jsonl"):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r["key"].startswith("c"):
            out[r["key"]] = r
    return out


def pick():
    import requests
    from PIL import Image
    C = pd.read_parquet(D / "commons_manifest.parquet").set_index("key")
    base = base_lines()
    z = np.load(D / "out" / "pe_pool6k.npz")
    vec = dict(zip(z["keys"].tolist(), z["img"].astype(np.float32)))
    pool = set(pd.read_parquet(D / "manifest_commons1k.parquet").key)
    keys = sorted(k for k in base if k in vec and k in C.index and k in pool)
    fired = [k for k in keys if apply_repeat_stop(base[k]["text"], 2)[1]]
    capped = [k for k in keys if k not in fired and len(base[k]["tokens"]) >= 120]     # loops repeat-stop does not catch
    rng = random.Random(82)
    sel = rng.sample(fired, min(5, len(fired)))
    sel += rng.sample(capped, min(5 - len(sel), len(capped)))
    print(f"pool {len(keys)}: base repeat-stop fired {len(fired)}, base at the 120-token cap without a stop {len(capped)}")
    rest = [k for k in keys if k not in sel]
    from sklearn.cluster import KMeans
    X = np.stack([vec[k] for k in rest])
    km = KMeans(n_clusters=20 - len(sel), random_state=82, n_init=10).fit(X)
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        sel.append(rest[idx[np.argmin(((X[idx] - km.cluster_centers_[c]) ** 2).sum(1))]])
    (P / "img").mkdir(parents=True, exist_ok=True); (P / "in").mkdir(parents=True, exist_ok=True)
    S = requests.Session(); S.headers["User-Agent"] = UA
    rows = []
    for k in sel:
        r = C.loc[k]
        W, H = int(r.w), int(r.h)
        width = min(STD, key=lambda s: (abs(max(s, round(s * H / W)) - 448), -s))
        url = re.sub(r"/1280px-", f"/{width}px-", r.thumb1280.split("?")[0])
        dst = P / "img" / f"{k}.jpg"
        for att in range(5):
            x = S.get(url, timeout=60)
            if x.status_code == 200 and x.headers.get("content-type", "").startswith("image/jpeg"):
                dst.write_bytes(x.content); break
            time.sleep(2 + 3 * att)
        else:
            raise RuntimeError(f"download failed {x.status_code} {url}")
        import hashlib
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        im = Image.open(dst); im.load()
        w, h = im.size
        s = min(448 / w, 448 / h)
        im2 = im.convert("RGB").resize((round(w * s), round(h * s)), Image.BICUBIC)
        inp = P / "in" / f"{k}.png"
        im2.save(inp)
        rows.append(dict(key=k, image=str(inp.resolve()), url=url, page=r.descurl, title=r.title,
                         license=r.license, author=r.artist, w=w, h=h, sha256=sha, fit_scale=round(s, 6), cluster_input_wh=list(im2.size),
                         base_repeat_stop_fired_in_pool=k in fired, base_at_cap_in_pool=k in capped))
        time.sleep(0.5)
    (P / "photos.json").write_text(json.dumps(rows, indent=1))
    pd.DataFrame([dict(key=r["key"], r448=r["image"]) for r in rows]).to_parquet(D / "manifest_parity20.parquet")
    (D / "manifest_parity20.json").write_text(json.dumps([dict(key=r["key"], r448=r["image"]) for r in rows]))
    print("PARITY_PICK", len(rows), "photos;", sum(r["base_repeat_stop_fired_in_pool"] for r in rows), "base repeat-stop,",
          sum(r["base_at_cap_in_pool"] for r in rows), "base at cap")


def tokens_to_stop(r, stopped):
    if not stopped:
        return len(r["tokens"])
    end = repeat_scan(r["text"])[1]; acc = 0
    for i, s in enumerate(r["segs"]):
        acc += len(s)
        if acc >= end:
            return i + 1
    return len(r["tokens"])


def assemble():
    photos = json.loads((P / "photos.json").read_text())
    out = {}
    for name in ("base", "trained"):
        for f in glob.glob(str(D / "out" / "writer_parity20" / f"{name}_*.jsonl")):
            for l in open(f):
                r = json.loads(l); out[(name, r["key"])] = r
    from tokenizers import Tokenizer        # the checkpoint's own tokenizer.json (light import; no transformers)
    tok = Tokenizer.from_file("eval/facts_writer/models/p2b_all_final_mlx/tokenizer.json")
    items = []
    for p in photos:
        it = dict(url=p["url"], sha256=p["sha256"], page=p["page"], title=p["title"], license=p["license"], author=p["author"],
                  w=p["w"], h=p["h"], resized_wh_fed=p["cluster_input_wh"],
                  selected_because=("base writer's vLLM line ran to the 120-token cap (a loop)" if p.get("base_at_cap_in_pool")
                                    else "base repeat-stop fired" if p.get("base_repeat_stop_fired_in_pool")
                                    else "k-means diversity"))
        for name in ("base", "trained"):
            r = out[(name, p["key"])]
            line, st = apply_repeat_stop(r["text"], 2)
            it[f"line_{name}"] = line
            it[f"tokens_{name}"] = tokens_to_stop(r, st)
            it[f"tokens_{name}_without_stop"] = len(r["tokens"])
            it[f"repeat_stop_fired_{name}"] = st
            it[f"raw_{name}"] = r["text"]
        # K = 48 (the phone's cap since RESULTS 70 / M77): the first 48 tokens of the SAME greedy run (greedy decoding
        # with max 48 new tokens produces exactly this prefix), detokenised, then repeat-stop v2
        r = out[("trained", p["key"])]
        t48 = tok.decode(r["tokens"][:48], skip_special_tokens=True)
        line48, st48 = apply_repeat_stop(t48, 2)
        it["line_trained_k48"] = line48
        it["tokens_trained_k48"] = min(48, it["tokens_trained"])
        it["repeat_stop_fired_trained_k48"] = st48
        it["k48_cut_inside_a_character"] = "\ufffd" in t48
        items.append(it)
    hdr = dict(
        purpose="M82(5) #writerparity: run FactsWriter's generate on each downloaded file; compare with line_base / "
                "line_trained (exact match + field F1). Cluster reference noise: RESULTS 61d MLX vs vLLM 19/50 exact.",
        prompt="Search facts for this photo, one line.",
        sampling="greedy (temperature 0), max 120 new tokens, repetition penalty 1.05 over a 4,096-token context "
                 "(prompt + output), stop at EOS; then FactsLine.applyRepeatStop v2 (line_* = after the stop; raw_* = the "
                 "full 120-token greedy text; tokens_* = tokens the phone generates when it stops streaming at the repeat)",
        image="the file at url, then processing.resize = 448 x 448 (MediaProcessing.apply: scale min(448/w, 448/h)); the "
              "cluster scaled the same file with Pillow BICUBIC by that factor and saved it lossless before mlx-vlm",
        url_note="Commons serves only standard thumbnail widths (448 px and the computed portrait widths return HTTP 400, "
                 "tested 11-05); width = the standard width (250/330/500/960) with long side closest to 448",
        engines="mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4 on an H100 (RESULTS 61d parity stack), chat template via "
                "mlx_vlm.prompt_utils.apply_chat_template (image first, then text)",
        k48="line_trained_k48 = the trained writer at max 48 new tokens (the phone's K = 48): the first 48 tokens of "
            "the same greedy run, decoded with the checkpoint's tokenizer (skip special tokens), then repeat-stop v2",
        writers=dict(base="mlx-community/Qwen3-VL-2B-Instruct-4bit (local HF snapshot)",
                     trained="eval/facts_writer/models/p2b_all_final_mlx (= rezashamji/findpics-writer-2b per the coordinator)"),
        selection="parity pool = commons1k (new Commons iPhone photos, in no training set); 5 where the base writer loops "
                  "on its vLLM line (repeat-stop v2 fired, else ran to the 120-token cap), seeded; + 15 = k-means(15, seed "
                  "82) on the other photos' PE-Core-B-16 vectors, the photo nearest each centre")
    (Path("eval/device_anchor82") / "writer_parity.json").write_text(json.dumps(dict(settings=hdr, photos=items), indent=1,
                                                                              ensure_ascii=False))
    print("PARITY_ASSEMBLED", len(items))


if __name__ == "__main__":
    {"pick": pick, "assemble": assemble}[sys.argv[1]]()
