"""RESULTS 48: ship the learned head (M48-scale JDonly) as fast mode's RANKING, blended with the phone's cosine.

What runs where:
  python eval/learned_head/head_ship.py rescore RUN [RUN ...]   (torch, GPU job)
      Text vectors of the RESULTS 47 beach question ("Is a beach clearly visible in this photo?", converse.BEACH_QUESTION)
      and every A4 / EV16 judge question, PE-Core-B-16 text tower exactly as embed.py made texts.npy, plus the same with
      the text weights int8-simulated (the phone's pe_core_text is the int8 Core ML package: symmetric per output
      channel on every weight tensor > 2048 elements, as eval/coreml_preproc_parity.int8_copy does for the image side).
      Then the head's P for the A4 / EV16 rows of scale/scores_{RUN}.parquet with the beach question replaced
      (-> scores_{RUN}_r47.parquet) and with every question's text from the int8-simulated tower
      (-> scores_{RUN}_r47i8.parquet). Every mlp seed whose weights exist (weights_{RUN}_mlp_s{k}.pt, or for the original
      JDonly run only seed 0 = weights_JDonly_mlp.pt). Check: non-beach rows reproduce the stored P.
  python eval/learned_head/head_ship.py export RUN SEED OUT      (torch) the phone file (layout below) + checks against
      the torch model on every A4 row (fp32 model vs the fp16 file read back by the numpy reference).
  python eval/learned_head/head_ship.py fixture HEADFILE OUT     (numpy) Swift golden fixture: 3 audit4 searches x 500
      audit4 photos; expected cosine (lookScores), head logit, blended score and top-200 order.

PHONE FILE (".fphead", little-endian; FindPicsCore/HeadRanker.swift reads it; this file's HeadFile is the reference):
  0   8   magic "FPHEAD01"
  8   4   u32 version = 1
  12  4   u32 D   embedding dim (1024, PE-Core-B-16 image/text vectors, L2-normalised)
  16  4   u32 B   feature blocks: 3 = [img, txt, img*txt]; 4 = [img, txt, img*txt, |img - txt|]
  20  4   u32 H1  first hidden width (1024)
  24  4   u32 H2  second hidden width (256)
  28  4   u32 metaLen; then metaLen bytes of UTF-8 JSON (provenance, for people; the reader skips it); zero padding to a
          multiple of 16 bytes
  W1  f16 [H1][B*D] row-major; column block k multiplies feature block k; the training standardisation is folded in
          (W1 = W_train / sd, column-wise)
  b1  f32 [H1]      (b_train - W1 . mu, computed with the fp16-rounded W1)
  W2  f16 [H2][H1]
  b2  f32 [H2]
  w3  f32 [H2]
  b3  f32 [1]
  logit(img, txt) = w3 . relu(W2 relu(W1 [img, txt, img*txt(, |img - txt|)] + b1) + b2) + b3
Per search, with txt fixed: M = W1[:, img] + W1[:, img*txt] * txt (column-scaled), c = W1[:, txt] txt + b1, so the
first layer is one H1 x D matrix per photo (M img + c), plus, for B = 4 only, W1[:, |img-txt|] |img - txt| per photo.

BLEND (fast mode's order; per search, over that search's in-scope photos of the one library):
  cos  = lookScores (the shipped fast score), head = logit clipped to +-LOGIT_CLIP (= logit of 1e-6, as blend.py)
  z(x) = (x - mean(x)) / std(x), population std (divide by n), std 0 -> 1, over the finite values only
  order key = z(cos) + z(head)  (equal weights, fixed before any test); a photo without a vector (cos = -inf) stays last.
  Videos (several unit vectors): head logit = max over the item's units, like lookScores (untested: DISBench is photos).
"""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval"); sys.path.insert(0, "eval/learned_head")
OUT = Path("eval/learned_head"); S = OUT / "scale"
BEACH_QUESTION = "Is a beach clearly visible in this photo?"      # = findpics.converse.BEACH_QUESTION (RESULTS 47)
LOGIT_CLIP = float(np.log((1 - 1e-6) / 1e-6))                     # 13.8155: blend.py clips P to [1e-6, 1 - 1e-6]
MAGIC = b"FPHEAD01"


# ------------------------------------------------------------------------------------------------ numpy reference
class HeadFile:
    """The phone file, read back exactly as HeadRanker.swift reads it (float64 math here = the golden reference)."""

    def __init__(self, path):
        b = Path(path).read_bytes()
        assert b[:8] == MAGIC, "not an .fphead file"
        ver, D, B, H1, H2, ml = struct.unpack_from("<6I", b, 8)
        assert ver == 1 and B in (3, 4)
        self.D, self.B, self.H1, self.H2 = D, B, H1, H2
        self.meta = json.loads(b[32:32 + ml].decode())
        o = 32 + ml; o += (-o) % 16

        def take(dt, n):
            nonlocal o
            a = np.frombuffer(b, dt, n, o); o += a.nbytes; return a
        self.W1 = take("<f2", H1 * B * D).reshape(H1, B * D).astype(np.float64)
        self.b1 = take("<f4", H1).astype(np.float64)
        self.W2 = take("<f2", H2 * H1).reshape(H2, H1).astype(np.float64)
        self.b2 = take("<f4", H2).astype(np.float64)
        self.w3 = take("<f4", H2).astype(np.float64)
        self.b3 = float(take("<f4", 1)[0])
        assert o == len(b), (o, len(b))
        self.sha256 = hashlib.sha256(b).hexdigest()

    def fold(self, t):
        D = self.D; t = np.asarray(t, np.float64)
        M = self.W1[:, :D] + self.W1[:, 2 * D:3 * D] * t[None, :]
        c = self.W1[:, D:2 * D] @ t + self.b1
        return M, c, t

    def logits(self, X, folded, dtype=np.float64):
        """float64 = the golden reference; float32 = what HeadRanker.swift computes (fold in float32 too)."""
        M, c, t = (np.asarray(a, dtype) for a in folded)
        X = np.asarray(X, dtype)
        h = X @ M.T + c
        if self.B == 4:
            h = h + np.abs(X - t[None, :]) @ self.W1[:, 3 * self.D:].astype(dtype).T
        h = np.maximum(h, 0)
        h2 = np.maximum(h @ self.W2.astype(dtype).T + self.b2.astype(dtype), 0)
        return h2 @ self.w3.astype(dtype) + dtype(self.b3)


def zscore(x):
    x = np.asarray(x, np.float64); f = np.isfinite(x)
    out = np.full(len(x), -np.inf)
    if f.any():
        m = x[f].mean(); s = x[f].std()
        out[f] = (x[f] - m) / (s if s > 0 else 1.0)
    return out


def blend(cos, logit):
    lg = np.clip(np.asarray(logit, np.float64), -LOGIT_CLIP, LOGIT_CLIP)
    lg = np.where(np.isfinite(np.asarray(cos, np.float64)), lg, np.nan)       # no vector: no head either
    zc, zh = zscore(cos), zscore(np.where(np.isnan(lg), -np.inf, lg))
    return np.where(np.isfinite(zc), zc + zh, -np.inf)


# ------------------------------------------------------------------------------------------------ torch helpers
def _torch():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def load_head(path):
    torch, dev = _torch()
    import train_scale as TS
    sd = torch.load(path, map_location=dev)
    din = sd["mu"].shape[0]
    kind = {4096: "mlp", 3072: "mlp_na"}[din]
    m = TS.Head(kind, din).to(dev); m.load_state_dict(sd); m.eval()
    return m, kind


def feats(i, t, kind):
    torch, _ = _torch()
    if kind == "mlp_na":
        return torch.cat([i, t, i * t], 1)
    return torch.cat([i, t, i * t, (i - t).abs()], 1)


def vec_keys():
    """= train_scale.vec_keys (row of each DISBench item in vecs.npy; first occurrence wins), without importing torch."""
    keys = Path("eval/tiles_phone/ids.txt").read_text().split()
    keys += Path("eval/tiles_phone/ev16_ids.txt").read_text().split()
    for f in sorted(OUT.glob("img_part*_ids.txt")):
        keys += f.read_text().split()
    first = {}
    for k, key in enumerate(keys):
        first.setdefault(key, k)
    return first


def image_vectors(items):
    """Whole-photo vector (unit 0) of each item, normalised as train_scale.Data does."""
    first = vec_keys()
    rows = np.array([first[str(i)] for i in items])
    V = np.load(OUT / "vecs.npy", mmap_mode="r")
    u, inv = np.unique(rows, return_inverse=True)
    X = np.asarray(V[u, 0], np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-6)
    return X, inv


def head_p(m, kind, X, inv, T, tix, B=1 << 15, raw=False):
    """P(yes) (raw=True: the logit) for rows (image row inv[r], text row tix[r]); X, T float32 unit rows."""
    torch, dev = _torch()
    Xg = torch.tensor(X, dtype=torch.float16, device=dev)       # train_scale keeps image vectors in fp16 on the GPU
    Tg = torch.tensor(T, dtype=torch.float32, device=dev)
    out = []
    with torch.no_grad():
        for b in range(0, len(inv), B):
            i = Xg[torch.as_tensor(inv[b:b + B], device=dev)].float()
            t = Tg[torch.as_tensor(tix[b:b + B], device=dev)]
            z = m(feats(i, t, kind))
            out.append((z if raw else torch.sigmoid(z)).cpu().numpy())
    return np.concatenate(out)


def int8_text_copy(model):
    """The phone's text tower is pe_core_text_PE_Core_B_16_int8 (scripts/quantize_coreml.py: linear_symmetric, weight
    threshold 2048). Simulated: every non-visual weight tensor with >= 2 dims and > 2048 elements, int8 per output
    channel, scale = max|w| / 127 (as coreml_preproc_parity.int8_copy does for the image tower)."""
    import copy
    torch, _ = _torch()
    q = copy.deepcopy(model); n = 0
    with torch.no_grad():
        for name, p in q.named_parameters():
            if name.startswith("visual.") or p.dim() < 2 or p.numel() <= 2048:
                continue
            w = p.data.float(); s = w.abs().flatten(1).amax(1).clamp_min(1e-12) / 127
            s = s.view(-1, *[1] * (w.dim() - 1)); p.data.copy_(((w / s).round().clamp(-127, 127) * s).to(p.dtype)); n += 1
    print("int8-simulated text weight tensors:", n, flush=True)
    return q


def rescore(runs):
    from findpics.models import ImageTextEncoder
    texts = json.load(open(OUT / "texts.json"))
    T0 = np.load(OUT / "texts.npy").astype(np.float32)
    first = pd.read_parquet(S / f"scores_{runs[0]}.parquet", columns=["src", "query", "tid"])
    first = first[first.src.isin(["A4", "EV16"])]
    tids = sorted(first.tid.unique().tolist())
    names = [texts[t] for t in tids] + [BEACH_QUESTION]
    enc = ImageTextEncoder("hf-hub:timm/PE-Core-B-16")
    fp = enc.texts(names)
    enc.model = int8_text_copy(enc.model)
    i8 = enc.texts(names)

    def unit16(a):                         # texts.npy is fp16; train_scale normalises after the cast
        a = np.asarray(a, np.float16).astype(np.float32)
        return a / np.linalg.norm(a, axis=-1, keepdims=True)
    fp, i8 = unit16(fp), unit16(i8)
    stored = T0[tids] / np.linalg.norm(T0[tids], axis=-1, keepdims=True)
    cs = (fp[:-1] * stored).sum(1); ci = (fp * i8).sum(1)
    rep = ["TEXT VECTORS (PE-Core-B-16 text tower)",
           f"  re-embedded vs texts.npy (the training texts): min cosine {cs.min():.5f} over {len(cs)} eval questions",
           f"  int8-simulated tower vs fp: cosine min {ci.min():.5f} / median {np.median(ci):.5f} over {len(ci)} texts",
           f"  new beach question vs old ('{texts[1368]}'): cosine {float(fp[-1] @ stored[tids.index(1368)]):.4f}"]
    for n, c in zip(names, ci):
        rep.append(f"    int8 cos {c:.5f}  {n}")
    np.savez(S / "ship_texts.npz", names=np.array(names), fp=fp.astype(np.float16), i8=i8.astype(np.float16))
    ti = {t: k for k, t in enumerate(tids)}; beach_k = len(names) - 1
    for run in runs:
        X = pd.read_parquet(S / f"scores_{run}.parquet").reset_index(drop=True)
        ev = X.src.isin(["A4", "EV16"]).to_numpy()
        beach = ev & (X["query"] == "beach photos").to_numpy()
        Xv, inv = image_vectors(X.item[ev].astype(str).to_numpy())
        tix_old = np.array([ti[t] for t in X.tid[ev]])
        tix_new = np.where(beach[ev], beach_k, tix_old)
        files = {k: S / f"weights_{run}_mlp_s{k}.pt" for k in range(3)}
        files = {k: f for k, f in files.items() if f.exists()} or {0: S / f"weights_{run}_mlp.pt"}
        R47, R8 = X.copy(), X.copy()
        for k, f in files.items():
            m, kind = load_head(f)
            p_old = head_p(m, kind, Xv, inv, stored, tix_old)
            d = np.abs(p_old - X[f"mlp_s{k}"].to_numpy()[ev])
            rep.append(f"[{run}] seed {k} ({kind}, {f.name}): stored-text P reproduces the parquet: max |dP| {d.max():.2e} "
                       f"over {ev.sum()} A4/EV16 rows")
            p_new = head_p(m, kind, Xv, inv, np.vstack([stored, fp[-1:]]), tix_new)
            p_i8 = head_p(m, kind, Xv, inv, i8, tix_new)
            c = R47[f"mlp_s{k}"].to_numpy().copy(); c[ev] = p_new; R47[f"mlp_s{k}"] = c.astype(np.float32)
            c = R8[f"mlp_s{k}"].to_numpy().copy(); c[ev] = p_i8; R8[f"mlp_s{k}"] = c.astype(np.float32)
            lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
            for q in sorted(X["query"][ev].unique()):
                mq = (X["query"][ev] == q).to_numpy()
                from scipy.stats import spearmanr
                rho = spearmanr(lg(p_new[mq]), lg(p_i8[mq])).correlation
                rep.append(f"    {q:24s} int8-text vs fp-text head logit: Spearman {rho:.4f}, max |dlogit| "
                           f"{np.abs(lg(p_new[mq]) - lg(p_i8[mq])).max():.3f}")
        for D_, suf in ((R47, "_r47"), (R8, "_r47i8")):
            cols = [f"mlp_s{k}" for k in files]
            D_ = D_.drop(columns=[c for c in ("mlp_s0", "mlp_s1", "mlp_s2") if c not in cols])
            D_["mlp"] = D_[cols].mean(1).astype(np.float32)
            D_["head_text"] = np.where(beach, BEACH_QUESTION, "")
            D_.to_parquet(S / f"scores_{run}{suf}.parquet")
        rep.append(f"[{run}] wrote scores_{run}_r47.parquet, scores_{run}_r47i8.parquet (seeds {sorted(files)})")
        print("\n".join(rep[-3:]), flush=True)
    (S / "ship_rescore.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep)); print("RESCORE_DONE", flush=True)


# ------------------------------------------------------------------------------------------------ export
def write_head(out, W1, b1, W2, b2, w3, b3, meta, D=None):
    """The .fphead layout (module doc). W1: H1 x B*D with the standardisation already folded in."""
    H1, BD = W1.shape; H2 = W2.shape[0]; D = D or 1024; B = BD // D
    assert BD == B * D and B in (3, 4) and W2.shape == (H2, H1)
    mb = json.dumps(meta, separators=(",", ":")).encode()
    hdr = MAGIC + struct.pack("<6I", 1, D, B, H1, H2, len(mb)) + mb
    hdr += b"\0" * ((-len(hdr)) % 16)
    body = (np.asarray(W1, "<f2").tobytes() + np.asarray(b1, "<f4").tobytes() + np.asarray(W2, "<f2").tobytes() +
            np.asarray(b2, "<f4").tobytes() + np.asarray(w3, "<f4").tobytes() + np.asarray(b3, "<f4").reshape(1).tobytes())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(hdr + body)


def export(run, seed, out):
    torch, dev = _torch()
    f = S / (f"weights_{run}_mlp_s{seed}.pt" if (S / f"weights_{run}_mlp_s{seed}.pt").exists() else f"weights_{run}_mlp.pt")
    sd = {k: v.detach().double().cpu().numpy() for k, v in torch.load(f, map_location="cpu").items()}
    mu, sdv = sd["mu"], sd["sd"]
    W1, b1 = sd["net.0.weight"], sd["net.0.bias"]
    W2, b2 = sd["net.2.weight"], sd["net.2.bias"]
    w3, b3 = sd["net.4.weight"][0], sd["net.4.bias"]
    H1, din = W1.shape; H2 = W2.shape[0]; D = 1024; B = din // D
    assert din == B * D and B in (3, 4)
    W1f = W1 / sdv[None, :]
    assert np.abs(W1f).max() < 6e4, np.abs(W1f).max()
    W1h = W1f.astype(np.float16)
    b1f = b1 - W1h.astype(np.float64) @ mu
    meta = dict(name="find_pics learned head (fast-mode ranking blend)", results="eval/RESULTS.md 48", run=run, seed=seed,
                source=f.name, features=["img", "txt", "img*txt", "|img-txt|"][:B], dim=D, h1=H1, h2=H2,
                embedding="PE-Core-B-16 (timm/open_clip), L2-normalised image and text vectors",
                text="the album's judge question (the planner's yes/no question), embedded by the same text tower",
                image="the stored whole-photo vector (unit 0; videos: each frame, max over frames)",
                blend="per search over in-scope photos: z(lookScores) + z(clip(logit, +-13.8155)); population std",
                train="M48-scale JDonly: 2,412 real planner questions x 48 DISBench photos (Qwen3.5-9B P) + 15 Open "
                      "Images concepts; teacher soft labels; eval/learned_head/train_scale.py")
    write_head(out, W1h, b1f, W2, b2, w3, b3, meta)
    H = HeadFile(out)
    # check: the file (numpy reference, float64) vs the torch fp32 model on every A4 row
    m, kind = load_head(f)
    X = pd.read_parquet(S / f"scores_{run if (S / f'scores_{run}.parquet').exists() else 'JDonly'}.parquet")
    X = X[X.src == "A4"].reset_index(drop=True)
    texts = np.load(OUT / "texts.npy").astype(np.float32); texts /= np.linalg.norm(texts, axis=-1, keepdims=True)
    Xv, inv = image_vectors(X.item.astype(str).to_numpy())
    tids = sorted(X.tid.unique()); ti = {t: k for k, t in enumerate(tids)}
    tix = np.array([ti[t] for t in X.tid])
    lt = head_p(m, kind, Xv, inv, texts[tids], tix, raw=True).astype(np.float64)
    Xh = Xv.astype(np.float16).astype(np.float64)        # the phone's stored vectors are fp16 (torch path too)
    lines = [f"EXPORT {out}: {Path(out).stat().st_size:,} bytes, sha256 {H.sha256}, B={B} H1={H1} H2={H2}, from {f.name}"]
    worst = 0.0
    for q in sorted(X["query"].unique()):
        mq = (X["query"] == q).to_numpy()
        tq = texts[X.tid[mq].iloc[0]]
        lr = H.logits(Xh[inv[mq]], H.fold(tq))
        d = np.abs(lr - lt[mq]); worst = max(worst, float(d.max()))
        top = lambda a: set(np.argsort(-a, kind="stable")[:200])  # noqa: E731
        from scipy.stats import spearmanr
        lines.append(f"  {q:24s} fp16 file vs torch fp32: max |dlogit| {d.max():.4f}, Spearman "
                     f"{spearmanr(lr, lt[mq]).correlation:.6f}, top-200 overlap {len(top(lr) & top(lt[mq]))}/200")
    lines.append(f"  worst |dlogit| {worst:.4f} (fp16 weights + fold vs the fp32 torch model)")
    print("\n".join(lines), flush=True)
    with open(S / "ship_export.txt", "a") as fh:
        fh.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------ Swift fixture
FIX_QUERIES = ["photos with a car", "photos with a bicycle", "beach photos"]


def fixture(headfile, out, n=500, seed=0):
    """3 audit4 searches x the same 500 audit4 photos (the 120 best by cosine per search + random fill), phone vectors
    (whole photo, fp16 as the phone stores them). Beach asks the RESULTS 47 question."""
    import base64
    H = HeadFile(headfile)
    texts = json.load(open(OUT / "texts.json")); tid = {t: k for k, t in enumerate(texts)}
    T = np.load(OUT / "texts.npy").astype(np.float32); T /= np.linalg.norm(T, axis=-1, keepdims=True)
    st = np.load(S / "ship_texts.npz")
    beach_v = st["fp"][list(st["names"]).index(BEACH_QUESTION)].astype(np.float32)
    beach_v /= np.linalg.norm(beach_v)
    looks = json.load(open("eval/tiles_phone/texts.json"))["looks"]
    P = pd.read_parquet(OUT / "pairs.parquet", columns=["src", "query", "item", "text", "look"])
    P = P[P.src == "A4"]
    items = sorted(P.item.astype(str).unique())
    rng = np.random.default_rng(seed)
    pick = set()
    for q in FIX_QUERIES:
        g = P[P["query"] == q].sort_values("look", ascending=False)
        pick |= set(g.item.astype(str).iloc[:120])
    rest = sorted(set(items) - pick)
    pick = sorted(pick | set(rng.choice(rest, n - len(pick), replace=False)))
    Xall, inv = image_vectors(items)
    vec_of = {it: Xall[inv[k]] for k, it in enumerate(items)}
    qtext = {q: (BEACH_QUESTION if q == "beach photos" else P[P["query"] == q].text.iloc[0]) for q in FIX_QUERIES}
    qvec = {q: (beach_v if q == "beach photos" else T[tid[qtext[q]]]).astype(np.float64) for q in FIX_QUERIES}
    folded = {q: H.fold(qvec[q]) for q in FIX_QUERIES}
    # NEAR-TIES: two photos whose blended scores differ by < 2e-4 can swap between float32 (Swift) and float64 (here);
    # near-duplicate photos do that. Replace the lower one of any such pair in a top-201 with a fresh random photo.
    for it_ in range(200):
        Xd = np.stack([vec_of[i] for i in pick]).astype(np.float16).astype(np.float64)
        bad = None
        for q in FIX_QUERIES:
            L = np.stack([T[tid[l]] for l in looks[q]]).astype(np.float64)
            bl = blend((Xd @ L.T).mean(1), H.logits(Xd, folded[q]))
            o = np.argsort(-bl, kind="stable")[:201]
            g = bl[o[:-1]] - bl[o[1:]]
            if (g < 2e-4).any():
                bad = o[1:][np.argmax(g < 2e-4)]; break
        if bad is None:
            break
        rest = sorted(set(items) - set(pick))
        pick = sorted((set(pick) - {pick[bad]}) | {rng.choice(rest)})
    else:
        raise SystemExit("could not remove near-ties")
    print(f"near-tie replacements: {it_}", flush=True)
    Xh = np.stack([vec_of[i] for i in pick]).astype(np.float16)
    Xd = Xh.astype(np.float64)
    fx = dict(note="RESULTS 48 golden fixture: eval/learned_head/head_ship.py fixture. Public DISBench audit4 photos, "
                   "PE-Core-B-16 phone-simulation whole-photo vectors (fp16, as the index stores them).",
              head_file=os.environ.get("HEAD_NAME", Path(headfile).name), head_bytes=Path(headfile).stat().st_size,
              head_sha256=H.sha256, n=len(pick), dim=H.D, logit_clip=LOGIT_CLIP,
              vectors_f16_b64=base64.b64encode(Xh.astype("<f2").tobytes()).decode(), queries=[])
    gaps = []
    for q in FIX_QUERIES:
        question, t = qtext[q], qvec[q].astype(np.float32)
        L = np.stack([T[tid[l]] for l in looks[q]]).astype(np.float64)
        cos = (Xd @ L.T).mean(1)                                   # lookScores: mean look similarity, no avoid
        lg = H.logits(Xd, H.fold(t.astype(np.float64)))
        bl = blend(cos, lg)
        order = np.argsort(-bl, kind="stable")
        b = bl[order]; gaps.append(float(np.min(b[:200] - b[1:201])))
        # the same in float32 (what the Swift test computes) must give the same top-200 order
        lg32 = H.logits(Xh.astype(np.float32), H.fold(t.astype(np.float64)), dtype=np.float32)
        bl32 = blend(cos.astype(np.float32), lg32)
        o32 = np.argsort(-bl32, kind="stable")
        assert (o32[:200] == order[:200]).all(), q
        fx["queries"].append(dict(name=q, question=question, text=[float(v) for v in t.astype(np.float32)],
                                  looks=[[float(v) for v in T[tid[l]]] for l in looks[q]], avoid=[],
                                  cosine=[round(float(v), 7) for v in cos], head_logit=[round(float(v), 6) for v in lg],
                                  blend=[round(float(v), 6) for v in bl], top200=[int(i) for i in order[:200]],
                                  max_abs_f32_vs_f64_blend=float(np.abs(bl32 - bl).max()),
                                  max_abs_f32_vs_f64_logit=float(np.abs(lg32 - lg).max())))
        print(f"{q}: min gap in top-200 blended scores {gaps[-1]:.2e}; f32 vs f64 max |d| blend "
              f"{fx['queries'][-1]['max_abs_f32_vs_f64_blend']:.2e} logit {fx['queries'][-1]['max_abs_f32_vs_f64_logit']:.2e}; "
              f"logit range {lg.min():.2f}..{lg.max():.2f}", flush=True)
    Path(out).write_text(json.dumps(fx, separators=(",", ":")))
    print(f"FIXTURE {out}: {Path(out).stat().st_size:,} bytes, {len(pick)} photos, min top-200 gap {min(gaps):.2e}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "rescore":
        rescore(sys.argv[2:])
    elif cmd == "export":
        export(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif cmd == "fixture":
        fixture(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
