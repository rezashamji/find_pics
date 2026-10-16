"""RESULTS 68 (3): the millisecond scorers' HEADS for pure-Swift inference (FindPicsCore FactsScorer.swift).
  python eval/cascade_port/heads.py export      -> eval/cascade_port/models/{towervec_head,vecB}.fphead (+ tiny random heads
                                                   with inputs/outputs -> Tests/.../Fixtures/heads.json)
  CUDA_VISIBLE_DEVICES= python eval/cascade_port/heads.py inputs <user> [...]
                                                -> data/public/cascade_port/heads/<user>.json + .bin: every (photo,
                                                   scorer question) the dev recordings of that library scored, with the
                                                   head inputs (f, q from the towervec encoder in fp32 on CPU; v = the
                                                   stored PE-Core-B-16 image vector; t = PE-Core-B-16 text in fp32 on CPU)
                                                   and PyTorch fp32 P of towervec and vecB.
.fphead = b"FPHEAD01" + uint32 LE header length + header JSON (kind, dims, tensor names/shapes in order) + float32 LE data.
"""
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "eval/facts_scorer")
M = Path("eval/facts_scorer/models")
OUT = Path("eval/cascade_port/models")
TOWERVEC, VECB = M / "tvw_t2b_final_rs2_xfit_full", M / "vecB_full"
FACTS_DEV = "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"


def head_tensors(sd, use_txt):
    names = (["ln_f.weight", "ln_f.bias", "ln_q.weight", "ln_q.bias"] if use_txt else []) + \
        ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]
    return [(n, sd["head." + n].float().numpy()) for n in names]


def write_head(path, kind, d_txt, d_img, tensors):
    hdr = dict(kind=kind, d_txt=d_txt, d_img=d_img, hid=int(tensors[-6][1].shape[0]), hid2=int(tensors[-4][1].shape[0]),
               img_scale=32.0, ln_eps=1e-5, activation="gelu_erf",
               layout="x = [LN_f(f), LN_q(q), LN_f(f)*LN_q(q)] (txt) + [32v, 32t, 32v*32t] (img); Linear(W row-major [out, in])",
               tensors=[[n, list(a.shape)] for n, a in tensors])
    h = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(b"FPHEAD01" + struct.pack("<I", len(h)) + h)
        for _, a in tensors:
            f.write(np.ascontiguousarray(a, "<f4").tobytes())


def export():
    import base64
    from scorers import Head
    OUT.mkdir(parents=True, exist_ok=True)
    sd = torch.load(TOWERVEC / "weights.pt", map_location="cpu")
    write_head(OUT / "towervec_head.fphead", "towervec", 384, 1024, head_tensors(sd, True))
    sd = torch.load(VECB / "weights.pt", map_location="cpu")
    write_head(OUT / "vecB.fphead", "vec", 0, 1024, head_tensors(sd, False))
    for f in OUT.glob("*.fphead"):
        print(f, f.stat().st_size)
    # tiny random heads + expected outputs (always-on Swift test)
    torch.manual_seed(68)
    cases = []
    for kind, dt, di in (("towervec", 8, 16), ("vec", 0, 16)):
        h = Head(dt or 1, di, hid=12, use_txt=dt > 0, use_img=True).eval()
        for p in h.parameters():
            torch.nn.init.normal_(p, std=0.3)
        n = 40
        f, q = torch.randn(n, dt or 1) * 2 + 0.5, torch.randn(n, dt or 1)
        v, t = torch.nn.functional.normalize(torch.randn(n, di), dim=-1), torch.nn.functional.normalize(torch.randn(n, di), dim=-1)
        with torch.no_grad():
            lg = h(f=f, q=q, v=v, t=t) if dt else h(v=v, t=t)
        sd = {"head." + k: x for k, x in h.state_dict().items()}
        path = OUT / f"tiny_{kind}.fphead"
        write_head(path, kind, dt, di, head_tensors(sd, dt > 0))
        cases.append(dict(kind=kind, head=base64.b64encode(path.read_bytes()).decode(), f=f.tolist() if dt else None,
                          q=q.tolist() if dt else None, v=v.tolist(), t=t.tolist(), logit=lg.tolist(),
                          p=torch.sigmoid(lg).tolist()))
    Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/heads.json").write_text(json.dumps(dict(cases=cases)))


def inputs(users):
    import pandas as pd
    from scorers import Scorer
    from train import b16_image_vectors, b16_text
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (fp32 CPU reference)"
    torch.set_num_threads(8)
    FT = pd.read_parquet(FACTS_DEV)
    facts = dict(zip(FT.item_id.astype(str), FT.text))
    d = Path("data/public/cascade_port/heads"); d.mkdir(parents=True, exist_ok=True)
    for u in users:
        lib = json.loads(Path(f"data/public/cascade_port/record/dev/lib_{u}.json").read_text())
        ids = lib["ids"]
        need = {}     # scorer question -> set of library rows (towervec and vecB both, every config)
        for cfg in ("0_N_V", "25_N_V", "100_N_A"):
            for f in Path(f"data/public/cascade_port/record/dev/{cfg}").glob("rec*.jsonl.gz"):
                for line in gzip.open(f, "rt"):
                    r = json.loads(line)
                    if r["user"] != u:
                        continue
                    for kk in ("tv", "vec"):
                        for qs, m in r[kk].items():
                            need.setdefault(qs, set()).update(int(x) for x in m)
        rows = sorted(set().union(*need.values()))
        qs = sorted(need)
        pid = [ids[r] for r in rows]
        IMG = b16_image_vectors(set(pid))
        TV = Scorer(TOWERVEC, facts=facts, img=IMG, qvec_fn=b16_text)
        VB = Scorer(VECB, img=IMG, qvec_fn=b16_text)
        Fv = TV._fvecs(pid).cpu().numpy().astype(np.float32)
        Qv = TV._qvecs(qs).cpu().numpy().astype(np.float32)
        Tv = TV._tvecs(qs).cpu().numpy().astype(np.float32)
        Vv = TV._ivecs(pid).cpu().numpy().astype(np.float32)
        pairs = []
        rix = {r: k for k, r in enumerate(rows)}
        for j, q in enumerate(qs):
            rr = sorted(need[q])
            ptv = TV.p([ids[r] for r in rr], [q] * len(rr))
            pvb = VB.p([ids[r] for r in rr], [q] * len(rr))
            pairs += [[rix[r], j, float(a), float(b)] for r, a, b in zip(rr, ptv, pvb)]
        hdr = dict(user=u, rows=rows, qs=qs, n=len(rows), m=len(qs), d_txt=384, d_img=1024,
                   order=["F n x 384", "V n x 1024", "Q m x 384", "T m x 1024"], pairs=pairs,
                   note="pairs = [photo index, question index, towervec P fp32 torch CPU, vecB P fp32 torch CPU]")
        (d / f"{u}.json").write_text(json.dumps(hdr))
        with open(d / f"{u}.bin", "wb") as fh:
            for a in (Fv, Vv, Qv, Tv):
                fh.write(np.ascontiguousarray(a, "<f4").tobytes())
        print(u, len(rows), "photos", len(qs), "questions", len(pairs), "pairs", flush=True)


if __name__ == "__main__":
    {"export": lambda: export(), "inputs": lambda: inputs(sys.argv[2:])}[sys.argv[1]]()
