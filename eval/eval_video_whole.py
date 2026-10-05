"""Does the judge do better on a VIDEO when it sees more of it? Reza's labeled videos (62: 42 heavier-era, 20 fit-era,
labels by Reza 10-04). Every variant judges Reza (red box from the face match), questions "Does the person in the red
box look heavier?" and "... look fit?":
  A one frame   : the best face frame (what the product does now)
  B frames mean : up to 6 face frames, each judged alone, P averaged (and max)
  C frames joint: the same up to 6 boxed frames in ONE prompt ("frames from one video of the same person")
  D video       : the face frames (+ neighbours, 2 fps, <= 16) encoded as a short video, the model's video input
Metric: AUC heavier-era vs fit-era of P(heavier), of -P(fit) and of logit(heavier) - logit(fit), per variant.
n = 62 is small: differences under ~0.05 AUC are noise. Private data: outputs in data/private/audits only.
Usage (vLLM env, GPU): python eval/eval_video_whole.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/private/audits/video_whole"); OUT.mkdir(parents=True, exist_ok=True)
QS = {"heavier": "Does the person in the red box look heavier?", "fit": "Does the person in the red box look fit?"}


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def url(im):
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main():
    from vllm import LLM, SamplingParams
    from findpics.store import load
    from findpics.people import face_sims, expand_refs
    from findpics.media import sample_video_frames
    from findpics.engine import person_crop_boxed
    from findpics.vlm import draw_box
    import av
    idx = load("data/private/index_sample")
    lab = json.load(open("data/private/audits/reza_labels_demo6.json"))
    refs = expand_refs(idx, np.load("data/private/index_sample/named_Reza.npy"), accept=0.55, rounds=3)
    s = face_sims(idx, refs); f = idx.faces.copy(); f["s"] = s; f["row"] = np.arange(len(f))
    vids = []
    for r, it in idx.items[idx.items.media == "video"].iterrows():
        l = lab.get(str(it.item_id))
        ff = f[(f.item_row == r) & (f.s >= 0.40)].sort_values("s", ascending=False)
        if l in ("H", "F") and len(ff):
            vids.append((it, l, ff.drop_duplicates("frame_t")))
    llm = LLM(model="Qwen/Qwen3.5-9B", max_model_len=16384, gpu_memory_utilization=0.85, trust_remote_code=True,
              limit_mm_per_prompt={"image": 6, "video": 1}, allowed_local_media_path=str(OUT.resolve()))
    tok = llm.get_tokenizer()
    yes = {tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")}
    no = {tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")}
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    def p_yes(msgs):
        outs = llm.chat(msgs, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = sum(np.exp(v.logprob) for k, v in lp.items() if k in yes); pn = sum(np.exp(v.logprob) for k, v in lp.items() if k in no)
            res.append(py / (py + pn) if py + pn > 0 else 0.5)
        return res

    rows = []
    for it, l, ff in vids:
        frames = sample_video_frames(it.path)
        if not frames:
            continue
        def at(t):
            return min(frames, key=lambda x: abs(x[0] - t))[1]
        top = ff.head(6)
        crops = [person_crop_boxed(idx, at(float(r.frame_t)), int(r.row)) for r in top.itertuples()]
        boxed = []                                  # full frames with the box, for the video variant
        for r in top.sort_values("frame_t").itertuples():
            im = at(float(r.frame_t)); fr = idx.faces.iloc[int(r.row)]
            sx, sy = im.width / float(fr["img_w"]), im.height / float(fr["img_h"])
            boxed.append(draw_box(im, (fr["x1"] * sx, fr["y1"] * sy, fr["x2"] * sx, fr["y2"] * sy)))
        while len(boxed) < 4:                       # the video input needs >= 2 sampled frames (1-frame videos crashed)
            boxed = boxed + boxed
        vp = OUT / f"{len(rows):03d}.mp4"
        with av.open(str(vp), "w") as c:
            st = c.add_stream("libx264", rate=2); w0 = min(im.width for im in boxed) // 2 * 2; h0 = min(im.height for im in boxed) // 2 * 2
            st.width, st.height, st.pix_fmt = w0, h0, "yuv420p"
            for im in boxed:
                for pkt in st.encode(av.VideoFrame.from_image(im.resize((w0, h0)))):
                    c.mux(pkt)
            for pkt in st.encode():
                c.mux(pkt)
        row = dict(id=str(it.item_id), label=l, n_frames=len(crops))
        for qn, q in QS.items():
            row[f"A_{qn}"] = p_yes([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(crops[0])}},
                                                                  {"type": "text", "text": q + " Answer with one word: yes or no."}]}]])[0]
            each = p_yes([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(c)}},
                                                        {"type": "text", "text": q + " Answer with one word: yes or no."}]}] for c in crops])
            row[f"B_{qn}"] = float(np.mean(each)); row[f"Bmax_{qn}"] = float(np.max(each))
            row[f"C_{qn}"] = p_yes([[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url(c)}} for c in crops] +
                                     [{"type": "text", "text": "These are frames from one video of the same person (the person in the "
                                                               "red box). " + q + " Answer with one word: yes or no."}]}]])[0]
            row[f"D_{qn}"] = p_yes([[{"role": "user", "content": [{"type": "video_url", "video_url": {"url": f"file://{vp.resolve()}"}},
                                                                  {"type": "text", "text": "This video shows one person (in the red box). " + q +
                                                                   " Answer with one word: yes or no."}]}]])[0]
        rows.append(row); print(row, flush=True)
    d = pd.DataFrame(rows); d.to_csv(OUT / "scores.csv", index=False)
    y = (d.label == "H").to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    res = {}
    for v in ["A", "B", "Bmax", "C", "D"]:
        res[v] = dict(heavier=round(auc(d[f"{v}_heavier"].to_numpy(), y), 3), fit=round(auc(-d[f"{v}_fit"].to_numpy(), y), 3),
                      diff=round(auc((lg(d[f"{v}_heavier"]) - lg(d[f"{v}_fit"])).to_numpy(), y), 3))
    print("AUC heavier-era vs fit-era (n =", len(d), ", H", int(y.sum()), "):", json.dumps(res), flush=True)
    json.dump(res, open(OUT / "auc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
