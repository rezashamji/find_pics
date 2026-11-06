"""M82 (B') PARITY-EXACT: build the 20 exact-size PNGs (parity_exact_recipe.build_png), check that mlx-vlm's Qwen3-VL
processor leaves each one unresized, and write the TRAINED writer's K = 48 line for each (real MLX, the RESULTS 61d
stack). Runs inside the container (parity_exact_job.sh).
  -> data/public/device_anchor82/parity_exact/<key>.png, eval/device_anchor82/writer_parity_exact.json
Target size per photo = the processor's own _resolved_size (smart_resize, factor 32, its min/max pixels) of the 448
best-fit input the earlier parity run fed (writer_parity.json resized_wh_fed).
"""
import hashlib
import inspect
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "eval/device_anchor82")
sys.path.insert(0, "src")
import parity_exact_recipe as R  # noqa: E402
from findpics.facts_line import apply_repeat_stop  # noqa: E402

D = Path("data/public/device_anchor82")
MODEL = "eval/facts_writer/models/p2b_all_final_mlx"
PROMPT = "Search facts for this photo, one line."


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def word_f1(a, b):
    ca, cb = Counter(words(a)), Counter(words(b))
    if not ca and not cb:
        return 1.0
    ov = sum((ca & cb).values())
    if not ov:
        return 0.0
    p, r = ov / sum(ca.values()), ov / sum(cb.values())
    return 2 * p * r / (p + r)


def icc_name(b):
    from PIL import Image
    icc = Image.open(io.BytesIO(b)).info.get("icc_profile")
    if not icc:
        return None
    try:
        from PIL import ImageCms
        return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:
        return f"embedded ({len(icc)} bytes)"


def main():
    import PIL
    import mlx_vlm
    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.models.qwen3_vl import processing_qwen3_vl as PQ
    old = json.loads(Path("eval/device_anchor82/writer_parity.json").read_text())["photos"]
    keys = [p["key"] for p in json.loads((D / "parity" / "photos.json").read_text())]
    assert len(keys) == len(old) == 20
    out = D / "parity_exact"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model, processor = load(MODEL)
    ip = processor.image_processor
    print(f"loaded in {time.time() - t0:.0f}s; image processor {type(ip).__name__} min_pixels {ip.min_pixels} "
          f"max_pixels {ip.max_pixels} patch {ip.patch_size} merge {ip.merge_size}", flush=True)
    prompt = apply_chat_template(processor, model.config, PROMPT, num_images=1)
    # resumable: MLX's CUDA backend segfaulted (exit 139) after 11 photos on the first try; one jsonl line per photo
    part = D / "parity_exact" / "rows.jsonl"
    done = {}
    if part.exists():
        for x in open(part):
            try:
                r = json.loads(x); done[r["key"]] = r
            except json.JSONDecodeError:
                pass
    fo = open(part, "a")
    rows = []
    for key, o in zip(keys, old):
        if key in done:
            rows.append(done[key]); continue
        b = (D / "parity" / "img" / f"{key}.jpg").read_bytes()
        thumb_sha = hashlib.sha256(b).hexdigest()
        assert thumb_sha == o["sha256"], f"{key}: thumbnail sha256 differs from writer_parity.json"
        fw, fh = o["resized_wh_fed"]
        H, W = ip._resolved_size(fh, fw)                     # what the processor made of the earlier 448-fit input
        png, pix = R.build_png(b, W, H)
        f = out / f"{key}.png"; f.write_bytes(png)
        arr = PQ._to_numpy_image(str(f))                     # (C, H, W) uint8, exactly as the processor reads the file
        same_size = tuple(ip._resolved_size(arr.shape[1], arr.shape[2])) == (arr.shape[1], arr.shape[2]) == (H, W)
        frame = PQ._resize_video_frames(arr[None, ...], H, W)[0]
        unchanged = bool(np.array_equal(frame, arr))
        grid = ip([str(f)])["image_grid_thw"][0].tolist()
        grid_old = ip([str(D / "parity" / "in" / f"{key}.png")])["image_grid_thw"][0].tolist()
        t1 = time.time(); segs, toks = [], []
        for r in stream_generate(model, processor, prompt, image=[str(f)], max_tokens=48, temperature=0.0,
                                 repetition_penalty=1.05, repetition_context_size=4096):
            segs.append(r.text)
            if getattr(r, "token", None) is not None:
                toks.append(int(r.token))
        # mlx-vlm 0.7.4 yields max_tokens + 1 tokens (121 at the 120 cap in dev500); the phone stops at exactly K = 48,
        # so the line is the streamed text of the first 48 tokens
        segs, toks = segs[:48], toks[:48]
        text = "".join(segs)
        line, stopped = apply_repeat_stop(text, 2)
        rows.append(dict(key=key, url=o["url"], thumb_sha256=thumb_sha, thumb_wh=[o["w"], o["h"]], thumb_icc_profile=icc_name(b),
                         W=W, H=H, png_sha256=hashlib.sha256(png).hexdigest(), pixel_sha256=pix,
                         processor_grid_thw=grid, processor_left_unresized=bool(same_size and unchanged),
                         previous_input_wh=[fw, fh], previous_grid_thw=grid_old,
                         line_trained_k48=line, tokens=len(toks), token_ids=toks, repeat_stop_fired=stopped,
                         raw_trained_k48=text, previous_line_trained_k48=o["line_trained_k48"],
                         word_f1_vs_previous_k48=round(word_f1(line, o["line_trained_k48"]), 4),
                         s=round(time.time() - t1, 2)))
        fo.write(json.dumps(rows[-1], ensure_ascii=False) + "\n"); fo.flush()
        print(f"{key} {W}x{H} grid {grid} (previous input {fw}x{fh} -> grid {grid_old}) unresized "
              f"{same_size and unchanged} tokens {len(toks)} F1 vs previous {rows[-1]['word_f1_vs_previous_k48']:.3f}",
              flush=True)
    recipe = dict(
        build_png_source=inspect.getsource(R.build_png),
        steps="thumbnail = the file at url (sha256 must match); (H, W) = the processor's smart_resize of the earlier 448 "
              "best-fit input (factor 32, round half to even, min/max pixels below); png = build_png(thumbnail, W, H)",
        pillow=PIL.__version__, numpy=np.__version__, mlx_vlm=mlx_vlm.__version__,
        processor=dict(cls=type(ip).__name__, min_pixels=ip.min_pixels, max_pixels=ip.max_pixels, patch_size=ip.patch_size,
                       merge_size=ip.merge_size, resample="PIL BICUBIC (skipped by Pillow when the size is already the "
                       "target: Image.resize returns a copy)"),
        writer=MODEL + " (= rezashamji/findpics-writer-2b per the coordinator)", prompt=PROMPT,
        sampling="greedy, max 48 new tokens (K = 48), repetition penalty 1.05 over a 4,096-token context, then "
                 "FactsLine.applyRepeatStop v2; line = the streamed text of the first 48 tokens (mlx-vlm 0.7.4 emits "
                 "max_tokens + 1 tokens, so the 49th is dropped)",
        chat_template=prompt,
        word_f1="multiset F1 of lower-cased [a-z0-9]+ words (the cluster's definition; the phone's may differ)",
        rebuild="python3 eval/device_anchor82/parity_exact_recipe.py  (downloads the urls, rebuilds, compares hashes)")
    Path("eval/device_anchor82/writer_parity_exact.json").write_text(
        json.dumps(dict(recipe=recipe, photos=rows), indent=1, ensure_ascii=False))
    n_un = sum(r["processor_left_unresized"] for r in rows)
    print(f"PARITY_EXACT_DONE {len(rows)} photos; processor left {n_un}/{len(rows)} unresized; mean word F1 vs previous "
          f"k48 lines {np.mean([r['word_f1_vs_previous_k48'] for r in rows]):.3f}", flush=True)


if __name__ == "__main__":
    main()
