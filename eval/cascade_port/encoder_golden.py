"""11-05 (RESULTS 68 (4) follow-up): golden fixture for the PHONE's facts text encoder, read by the Mac's #factsparity
(device MLX vs this file) and by FindPicsCoreTests/FactsEncoderGoldenTests.swift on Linux (everything that needs no MLX).
  python eval/cascade_port/encoder_golden.py [--out=<json>] [--mlx=0] [--drift_photos=200] [--mlx_procs=1]
     (repo root, CPU. The fixture itself takes ~1 min after imports (imports alone took 12 min on the login node, 11-05).
      --drift_photos=N (0 = off) also runs MLX fp16 on N seeded photos per library + every recorded question (470 texts at
      N=200, 1.15 s each: the login node gives 1 CPU per user) to measure how far fp16 alone moves P, which sets p_bar;
      --mlx_procs splits that over processes, useful only on a node with more than one CPU)
  -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/facts_encoder_golden.json
What the product does (ios/FindPicsApp.swiftpm/Sources/FactsWriter.swift `write` / `encode`):
  text = factsPrefix(place, taken) + writer line (after repeat-stop v2); ids = BertTokenizer.factsInput(text) =
  [CLS] + wordpiece[:254] + [SEP] (a scorer question: questionInput = [CLS] + [:64] + [SEP]); MLXEmbedders BertModel(ids,
  tokenTypeIds 0, attention mask 1); vector = MEAN over ALL tokens (incl. CLS/SEP) of the last hidden state, NOT normalised,
  384-d. P = towervec head(f, q, v, t) (FindPicsCore FactsScorer.swift, towervec_head.fphead).
Texts (all PUBLIC): dev libraries 14798958@N08 and 23090753@N06 (the two whose head inputs v, t exist in
  data/public/cascade_port/heads, from heads.py inputs); writer lines = eval/facts_writer/facts/p2b_all_dev_rs2 (product
  p2b_all + repeat-stop v2 = the dev lines of facts_w_p2b_devmix_rs2m.parquet that RESULTS 61d-70 scored); place / taken =
  data/public/index_disbench/items.parquet, prefix built as eval/facts_scorer/data.py --meta=1; + named edge cases.
  Questions: recorded dev scorer questions of those two libraries (4 each, seeded) + 1 synthetic over-long one (dev query
  turns joined) for questionInput's 64-token cap.
References:
  fp32     = the ORIGINAL towervec weights the cluster scores with (eval/facts_scorer/models/tvw_t2b_final_rs2_xfit_full/
             weights.pt) through scorers.Scorer in PyTorch fp32 on CPU, ONE unpadded sequence at a time (= the phone's input).
  mlx_fp16 = the exported fp16 safetensors (eval/cascade_port/models/towervec_encoder_fp16 = the app's bundled copy) in
             real MLX (Python, CPU backend, tools/mlxpy container) built as MLXEmbedders builds BertModel (encoder_mlx.py),
             fp16 compute, mean pool in fp16 then cast to fp32 (as FactsWriter.encode).
  P        = scorers.Head (towervec) in fp32 PyTorch on CPU, fed the fixture's ROUNDED vectors (what a reader parses).
"""
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/facts_encoder_golden.json")
SRC = Path("eval/facts_scorer/models/tvw_t2b_final_rs2_xfit_full")
ENC16 = Path("eval/cascade_port/models/towervec_encoder_fp16")
HEAD = Path("eval/cascade_port/models/towervec_head.fphead")
LINES = Path("eval/facts_writer/facts/p2b_all_dev_rs2")      # product writer + repeat-stop v2 (what the phone stores)
RAW = Path("eval/facts_writer/facts/p2b_all_dev")            # the same writer before repeat-stop
FACTS_PQ = "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"
HEADS = Path("data/public/cascade_port/heads")
LIBS = ["14798958@N08", "23090753@N06"]
QUERIES = Path("eval/complex_bench/queries/dev.json")
WORK = Path(".cache/encgold")
SEED = 1105
MLXPY = ["singularity", "exec", "--cleanenv", "-B", "{root}:{root}", "tools/mlxpy/py312.sif", "tools/mlxpy/venv/bin/python"]


def arg(k, d=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)


def r7(a):
    return [float(f"{float(x):.7g}") for x in a]


def r9(a):     # float32-exact (9 significant digits)
    return [float(f"{float(x):.9g}") for x in a]


def prefix(place, taken):
    """eval/facts_scorer/data.py --meta=1 (taken = unix seconds, UTC month) = FindPicsCore factsPrefix(place:taken:)"""
    import pandas as pd
    parts = [p_.strip() for p_ in str(place or "").split(",") if p_.strip() and not re.search(r"\d", p_)]
    parts = list(dict.fromkeys(parts))
    m = None if taken is None else pd.Timestamp(taken, unit="s", tz="UTC").strftime("%B %Y")
    return (f"place: {', '.join(parts)}. " if parts else "") + (f"taken: {m}. " if m else "")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))


def mlx_main(inp, outp):
    """inside tools/mlxpy: id sequences -> mean-pooled vectors from the fp16 export, the way the phone computes them"""
    import mlx.core as mx
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from encoder_mlx import Bert, sanitize
    mx.set_default_device(mx.cpu)
    J = json.loads(Path(inp).read_text())
    d = Path(J["model"])
    c = json.loads((d / "config.json").read_text())
    w = sanitize(mx.load(str(d / "model.safetensors")))
    assert all(v.dtype == mx.float16 for v in w.values()), "expected the fp16 export"
    m = Bert(c)
    m.load_weights(list(w.items()), strict=True)
    m.set_dtype(mx.float16)
    out = dict(mlx=mx.__version__)
    import time
    t0, n = time.time(), 0
    for k in ("facts", "questions"):
        vs = []
        for s in J[k]:
            ids = mx.array([s])
            h = m(ids, mx.zeros_like(ids))
            out["hidden_dtype"] = str(h.dtype)
            vs.append(np.array(h.mean(axis=1).astype(mx.float32)[0]).tolist())
            n += 1
        out[k] = vs
    Path(outp).write_text(json.dumps(out))
    print("mlx", out["mlx"], "hidden", out["hidden_dtype"], {k: len(out[k]) for k in ("facts", "questions")},
          f"{(time.time() - t0) / max(1, n):.2f} s per sequence", flush=True)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import pandas as pd
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, "eval/facts_scorer")
    from scorers import Scorer
    out_path = Path(arg("out", str(OUT)))
    rng = random.Random(SEED)

    # ---- inputs -------------------------------------------------------------------------------------------------
    cap = lambda r: (r.get("facts") or {}).get("caption") or ""      # noqa: E731  (no line -> "", never used)
    lines, raw = {}, {}
    for u in LIBS:
        lines[u] = {}
        for ln in open(LINES / f"{u}.jsonl"):
            r = json.loads(ln); lines[u][str(r["item_id"])] = (cap(r), bool(r.get("stopped")))
        raw[u] = {}
        for ln in open(RAW / f"{u}.jsonl"):
            r = json.loads(ln); raw[u][str(r["item_id"])] = cap(r)
    it = pd.read_parquet("data/public/index_disbench/items.parquet", columns=["item_id", "place", "taken"])
    it["item_id"] = it.item_id.astype(str)
    mine = {p for u in LIBS for p in lines[u]}
    it = it[it.item_id.isin(mine)]
    meta = {i: (pl, pd.Timestamp(tk).tz_convert("UTC").timestamp()) for i, pl, tk in zip(it.item_id, it.place, it.taken)}
    FT = pd.read_parquet(FACTS_PQ)
    cluster_text = dict(zip(FT.item_id.astype(str), FT.text))
    H, V, T, Fb, Qb, rowpid = {}, {}, {}, {}, {}, {}
    for u in LIBS:
        h = json.loads((HEADS / f"{u}.json").read_text()); H[u] = h
        n, m = h["n"], h["m"]
        a = np.fromfile(HEADS / f"{u}.bin", dtype="<f4")
        assert a.size == n * 1408 + m * 1408
        Fb[u] = a[:n * 384].reshape(n, 384); V[u] = a[n * 384:n * 1408].reshape(n, 1024)
        Qb[u] = a[n * 1408:n * 1408 + m * 384].reshape(m, 384); T[u] = a[n * 1408 + m * 384:].reshape(m, 1024)
        ids = json.loads(Path(f"data/public/cascade_port/record/dev/lib_{u}.json").read_text())["ids"]
        rowpid[u] = [str(ids[r]) for r in h["rows"]]

    facts, questions, pairs, used = [], [], [], set()

    def add_fact(name, kind, u, pid, place, taken, line, stopped=None, row=None, note=None):
        pre = prefix(place, taken)
        f = dict(name=name, kind=kind, user=u, item_id=pid, place=place, taken=taken, prefix=pre, line=line,
                 stopped=stopped, text=pre + line, note=note, _row=row)
        if kind == "real":     # the cluster's own text for that photo (data.py --meta=1 strips the line; tokens identical)
            assert pre + line.strip() == cluster_text[pid], (pid, f["text"], cluster_text[pid])
            f["cluster_text_equal"] = f["text"] == cluster_text[pid]
        facts.append(f); used.add(pid)
        return len(facts) - 1

    # ---- real (question, photo) pairs: 4 recorded scorer questions per library, a mid-P and a high-P photo each ------
    for u in LIBS:
        h = H[u]
        by_q = defaultdict(list)
        for r, j, ptv, _ in h["pairs"]:
            pid = rowpid[u][int(r)]
            if lines[u].get(pid, ("",))[0].strip():
                by_q[int(j)].append((int(r), float(ptv)))
        cands = [j for j, lst in sorted(by_q.items()) if any(.25 <= p <= .75 for _, p in lst) and any(p >= .85 for _, p in lst)]
        for j in sorted(rng.sample(cands, 4)):
            qi = len(questions)
            questions.append(dict(name=f"{u} recorded scorer question", kind="real", user=u, question=h["qs"][j],
                                  _t=T[u][j], _qb=Qb[u][j]))
            lst = sorted(by_q[j], key=lambda x: (abs(x[1] - .5), x[0]))
            for want in ("mid", "high"):
                if want == "high":
                    lst = sorted(by_q[j], key=lambda x: (-x[1], x[0]))
                r, p = next((r, p) for r, p in lst if rowpid[u][r] not in used)
                pid = rowpid[u][r]
                line, st = lines[u][pid]
                fi = add_fact(f"{u} {pid} ({want} recorded P)", "real", u, pid, meta[pid][0], meta[pid][1], line, st, row=r)
                pairs.append(dict(fact=fi, question=qi, p_cluster=p))

    # ---- edge cases -------------------------------------------------------------------------------------------------
    A, B = LIBS
    qcyc = {u: [i for i, q in enumerate(questions) if q["user"] == u] for u in LIBS}
    qn = defaultdict(int)

    def q_next(u):     # edge pairs cycle through that library's questions
        qn[u] += 1
        return qcyc[u][(qn[u] - 1) % len(qcyc[u])]
    rix = {u: {pid: r for r, pid in enumerate(rowpid[u])} for u in LIBS}

    def pick(u, cond, key=None):
        c = sorted(pid for pid, (ln, st) in lines[u].items() if pid not in used and pid in rix[u] and ln.strip() and cond(pid, ln, st))
        if key:
            c = sorted(c, key=key)
        return c[0] if c else None

    edge = []
    # a line repeat-stop cut (the longest such line in the two libraries)
    st_pids = [(len(lines[u][p][0]), u, p) for u in LIBS for p, (ln, st) in lines[u].items() if st and p not in used and p in rix[u]]
    if st_pids:
        _, u, pid = max(st_pids)
        edge.append(("repeat-stop v2 cut this line (stopped=true)", "real", u, pid, None, None, None))
    else:
        print("WARNING: no repeat-stop cut line in", LIBS)
    # a place in Chinese characters (Hong Kong library)
    pid = pick(A, lambda p, ln, st: bool(re.search("[\u4e00-\u9fff]", meta[p][0] or "")))
    edge.append(("place with CJK characters", "real", A, pid, None, None, None))
    # a place with digit parts dropped AND a duplicate part de-duplicated
    def dup_digit(p, ln, st):
        parts = [x.strip() for x in (meta[p][0] or "").split(",") if x.strip()]
        return any(re.search(r"\d", x) for x in parts) and len(set(parts)) < len(parts)
    pid = pick(B, dup_digit) or pick(A, dup_digit) or pick(B, lambda p, ln, st: bool(re.search(r"\d", meta[p][0] or "")))
    edge.append(("place with digit parts (dropped) and a repeated part (de-duplicated)", "real",
                 A if pid in lines[A] else B, pid, None, None, None))
    # an empty place (date only)
    for u in LIBS:
        pid = pick(u, lambda p, ln, st: not (meta[p][0] or "").strip())
        if pid:
            edge.append(("empty place: date only", "real", u, pid, None, None, None)); break
    # natural non-ASCII inside the writer line
    for u in LIBS:
        pid = pick(u, lambda p, ln, st: any(ord(ch) > 127 for ch in ln))
        if pid:
            edge.append(("non-ASCII characters in the writer line itself", "real", u, pid, None, None, None)); break
    for e in edge:
        name, kind, u, pid, *_ = e
        if pid is None:
            print("WARNING: no photo for edge case", name); continue
        line, st = lines[u][pid]
        fi = add_fact(f"edge: {name}", kind, u, pid, meta[pid][0], meta[pid][1], line, st, row=rix[u][pid])
        pairs.append(dict(fact=fi, question=q_next(u), p_cluster=None))
    # synthetic: no place and no date (the phone has neither for this photo)
    pid = pick(B, lambda p, ln, st: True)
    fi = add_fact("edge: no place and no date (meta withheld)", "synthetic", B, pid, None, None, lines[B][pid][0],
                  lines[B][pid][1], row=rix[B][pid], note="real line, place and taken set to null")
    pairs.append(dict(fact=fi, question=q_next(B), p_cluster=None))
    # synthetic: empty writer line (prefix only), and nothing at all ("" -> [CLS] [SEP])
    pid = pick(A, lambda p, ln, st: bool((meta[p][0] or "").strip()))
    fi = add_fact("edge: empty writer line (prefix only)", "synthetic", A, pid, meta[pid][0], meta[pid][1], "", None,
                  row=rix[A][pid], note="real place/taken, line set to empty")
    pairs.append(dict(fact=fi, question=q_next(A), p_cluster=None))
    fi = add_fact("edge: empty text (no place, no date, empty line)", "synthetic", A, None, None, None, "", None,
                  row=rix[A][pid], note="v of the previous photo")
    pairs.append(dict(fact=fi, question=q_next(A), p_cluster=None))
    # truncation: longer than 254 word pieces. The longest RAW (before repeat-stop) looping line of the two libraries,
    # repeated until it is > 300 word pieces if it is not already (the phone caps at 120 writer tokens, so a natural
    # line rarely gets there; the cap must still be [:254]).
    tk = None
    lr = max(((len(raw[u][p]), u, p) for u in LIBS for p in raw[u] if p in rix[u] and p not in used))
    _, u, pid = lr
    line = raw[u][pid]
    S = Scorer(SRC, facts={})
    tk = S.tok
    n0 = len(tk.ids([prefix(*meta[pid]) + line], 10 ** 6)[0])
    note = f"the longest raw (pre-repeat-stop) line of the two libraries: {n0} word pieces with its prefix"
    if n0 <= 300:
        k = 300 // max(1, n0) + 1
        line = " ".join([line] * k)
        note += f"; repeated x{k} (synthetic) to pass the 254 cap"
    fi = add_fact("edge: longer than 254 word pieces (truncation)", "synthetic", u, pid, meta[pid][0], meta[pid][1],
                  line, None, row=rix[u][pid], note=note)
    pairs.append(dict(fact=fi, question=q_next(u), p_cluster=None))
    # synthetic: non-ASCII incl. U+FFFD (a byte-level BPE decode cut inside a CJK character gives U+FFFD)
    pid = pick(A, lambda p, ln, st: True)
    line = lines[A][pid][0] + "; text: CAFÉ naïve 香港茶餐廳 4,50€ \ufffd"
    fi = add_fact("edge: non-ASCII text incl. U+FFFD (BertNormalizer drops U+FFFD)", "synthetic", A, pid, meta[pid][0],
                  meta[pid][1], line, None, row=rix[A][pid], note="real line + a synthetic text field")
    pairs.append(dict(fact=fi, question=q_next(A), p_cluster=None))
    # synthetic: UTC month boundary (taken 1 s before / exactly at 2014-01-01 00:00:00 UTC)
    for ts in (1388534399.0, 1388534400.0):
        pid = pick(B, lambda p, ln, st: True)
        add_fact(f"edge: taken {ts:.0f} (UTC month boundary)", "synthetic", B, pid, meta[pid][0], ts, lines[B][pid][0],
                 lines[B][pid][1], note="real line and place, synthetic taken")
    # synthetic over-long question: two dev queries' turns joined (> 64 word pieces)
    Qd = json.loads(QUERIES.read_text())["queries"]
    longest = sorted(Qd, key=lambda q: (-len(" ".join(q["turns"])), q["qid"]))[:2]
    lq = " ".join(t for q in longest for t in q["turns"])
    questions.append(dict(name="synthetic: turns of dev queries " + ", ".join(q["qid"] for q in longest) + " joined (> 64 word pieces)",
                          kind="synthetic", user=None, question=lq, _t=None, _qb=None))

    # ---- fp32 reference ---------------------------------------------------------------------------------------------
    texts = [f["text"] for f in facts]
    qs = [q["question"] for q in questions]
    fids = tk.ids(texts, 256)                 # Scorer._facts_tok: ids(texts, 256) then single(.., 256) -> [:254]
    qids = tk.ids(qs, 64)                     # Scorer._qvecs: ids(qs, 64) then single(.., 66)
    fseq = [[tk.cls] + a[:254] + [tk.sep] for a in fids]
    qseq = [[tk.cls] + a[:64] + [tk.sep] for a in qids]
    full = tk.ids(texts, 10 ** 6)
    with torch.no_grad():
        fv = [S.model.embed(*tk.single([a], 256))[0].numpy() for a in fids]
        qv = [S.model.embed(*tk.single([a], 66))[0].numpy() for a in qids]
    for f, s, v, n in zip(facts, fseq, fv, full):
        f["ids"] = s; f["fp32"] = r7(v); f["wordpieces_before_cap"] = len(n)
    for q, s, v in zip(questions, qseq, qv):
        q["ids"] = s; q["fp32"] = r7(v)
    # cross-check vs the vectors heads.py stored (batched + padded, CPU fp32) for the same photos / questions
    cf = [cos(f["fp32"], Fb[f["user"]][f["_row"]]) for f in facts if f["kind"] == "real"]
    cq = [cos(q["fp32"], q["_qb"]) for q in questions if q["_qb"] is not None]
    print(f"fp32 vs heads.py stored vectors: facts n {len(cf)} cos min {min(cf):.8f}; questions n {len(cq)} cos min {min(cq):.8f}")

    # ---- MLX fp16 (second reference) --------------------------------------------------------------------------------
    def run_mlx(tag, fs, qs_, procs=1):
        """id sequences -> (facts vectors, question vectors, info) from the fp16 export in real MLX (container; fp16 on
        an x86 CPU runs 1.15 s per sequence, so long lists go to `procs` parallel processes); an identical earlier request
        is reused"""
        WORK.mkdir(parents=True, exist_ok=True)
        k_ = max(1, min(procs, len(fs)))
        chunks = [fs[i * len(fs) // k_:(i + 1) * len(fs) // k_] for i in range(k_)]
        jobs, outs = [], []
        for i, ch in enumerate(chunks):
            inp, outp = Path.cwd() / WORK / f"mlx_in_{tag}_{i}.json", Path.cwd() / WORK / f"mlx_out_{tag}_{i}.json"
            req = json.dumps(dict(model=str(Path.cwd() / ENC16), facts=ch, questions=qs_ if i == 0 else [],
                                  sha=sha(ENC16 / "model.safetensors")))
            if not (outp.exists() and inp.exists() and inp.read_text() == req):
                inp.write_text(req)
                outp.unlink(missing_ok=True)
                cmd = [x.format(root=str(Path.cwd())) for x in MLXPY] + [str(Path(__file__).resolve()), "mlx", str(inp), str(outp)]
                jobs.append(subprocess.Popen(cmd))
            outs.append(outp)
        for j in jobs:
            if j.wait(timeout=7200) != 0:
                raise RuntimeError(f"MLX process exited {j.returncode}")
        M = [json.loads(o.read_text()) for o in outs]
        return ([v for m_ in M for v in m_["facts"]], M[0]["questions"],
                f"MLX {M[0]['mlx']} (Python, CPU backend), hidden states {M[0]['hidden_dtype']}")

    mlx_info = None
    if arg("mlx", "1") == "1":
        try:
            mf, mq, mlx_info = run_mlx("fixture", fseq, qseq)
            for f, v in zip(facts, mf):
                f["mlx_fp16"] = r7(v)
            for q, v in zip(questions, mq):
                q["mlx_fp16"] = r7(v)
        except Exception as e:      # noqa: BLE001
            print("MLX step failed:", repr(e))
    if mlx_info is None:
        for x in facts + questions:
            x["mlx_fp16"] = None

    # ---- how far fp16 alone moves P: a seeded sample of photos of the two libraries x EVERY question heads.py recorded
    # for them (fp32 side = heads.py's stored vectors and recorded P; fp16 side = the same texts through MLX fp16, same
    # head). The sample keeps it to ~10 min: the login node gives one CPU per user and MLX fp16 on it is 1.15 s/sequence.
    drift = None
    n_drift = int(arg("drift_photos", "200"))
    if mlx_info and n_drift > 0:
        rs = {u: sorted(random.Random(SEED + k).sample(range(H[u]["n"]), min(n_drift, H[u]["n"]))) for k, u in enumerate(LIBS)}
        at, aq = [], []
        for u in LIBS:
            at += [cluster_text[rowpid[u][r]] for r in rs[u]]
            aq += H[u]["qs"]
        dfs, dqs, _ = run_mlx("drift", [[tk.cls] + a[:254] + [tk.sep] for a in tk.ids(at, 256)],
                              [[tk.cls] + a[:64] + [tk.sep] for a in tk.ids(aq, 64)], procs=int(arg("mlx_procs", "1")))
        dfs, dqs = np.asarray(dfs, np.float32), np.asarray(dqs, np.float32)
        P32, P16, P32re, cs = [], [], [], []
        of, oq = 0, 0
        for u in LIBS:
            m = H[u]["m"]
            loc = {r: i for i, r in enumerate(rs[u])}
            F16, Q16 = dfs[of:of + len(rs[u])], dqs[oq:oq + m]
            cs += [cos(F16[loc[r]], Fb[u][r]) for r in rs[u]] + [cos(a, b) for a, b in zip(Q16, Qb[u])]
            pr = np.array([x for x in H[u]["pairs"] if int(x[0]) in loc])
            r_, j_ = pr[:, 0].astype(int), pr[:, 1].astype(int)
            with torch.no_grad():
                for s0 in range(0, len(pr), 8192):
                    rr, jj = r_[s0:s0 + 8192], j_[s0:s0 + 8192]
                    vt = dict(v=torch.from_numpy(V[u][rr]), t=torch.from_numpy(T[u][jj]))
                    f16 = torch.from_numpy(F16[[loc[r] for r in rr]])
                    P16 += torch.sigmoid(S.model.head(f=f16, q=torch.from_numpy(Q16[jj]), **vt)).tolist()
                    P32re += torch.sigmoid(S.model.head(f=torch.from_numpy(Fb[u][rr]), q=torch.from_numpy(Qb[u][jj]), **vt)).tolist()
            P32 += pr[:, 2].tolist()
            of += len(rs[u]); oq += m
        P32, P16, P32re = np.array(P32), np.array(P16), np.array(P32re)
        d = np.abs(P16 - P32)
        drift = dict(photos_per_library=n_drift, pairs=int(len(d)), texts=int(len(cs)), cos_min=float(min(cs)), p_max_abs=float(d.max()),
                     p_q999_abs=float(np.quantile(d, .999)), p_q99_abs=float(np.quantile(d, .99)),
                     pairs_over_1e3=int((d > 1e-3).sum()), pairs_over_5e3=int((d > 5e-3).sum()),
                     pairs_across_07=int(((P16 >= .7) != (P32 >= .7)).sum()),
                     recompute_vs_recorded_p_max_abs=float(np.abs(P32re - P32).max()))
        print(f"fp16 drift, recorded pairs of {n_drift} seeded photos per library of", LIBS, json.dumps(drift))

    # ---- head P from the ROUNDED vectors ----------------------------------------------------------------------------
    head = S.model.head.eval()
    t32 = lambda a: torch.tensor(np.asarray(a, np.float32))[None]      # noqa: E731
    for f in facts:
        f["v"] = None
    for q in questions:
        q["t"] = r9(q["_t"]) if q["_t"] is not None else None
    for p in pairs:
        f, q = facts[p["fact"]], questions[p["question"]]
        if f["v"] is None:
            f["v"] = r9(V[f["user"]][f["_row"]])
        with torch.no_grad():
            lg = head(f=t32(f["fp32"]), q=t32(q["fp32"]), v=t32(f["v"]), t=t32(q["t"]))
            p["logit"] = float(f"{float(lg[0]):.9g}"); p["p"] = float(f"{float(torch.sigmoid(lg)[0]):.9g}")
            if mlx_info:
                lm = head(f=t32(f["mlx_fp16"]), q=t32(q["mlx_fp16"]), v=t32(f["v"]), t=t32(q["t"]))
                p["p_mlx"] = float(f"{float(torch.sigmoid(lm)[0]):.9g}")
            else:
                p["p_mlx"] = None
    pc = [abs(p["p"] - p["p_cluster"]) for p in pairs if p["p_cluster"] is not None]
    print(f"P (fixture fp32) vs heads.py recorded P (same photo, question): n {len(pc)} max |diff| {max(pc):.2e}")

    checks = dict(fp16_drift_sampled_pairs=drift, fp32_vs_heads_py_facts_cos_min=min(cf), fp32_vs_heads_py_questions_cos_min=min(cq),
                  p_vs_heads_py_max_abs=max(pc), real_texts_equal_cluster_parquet=sum(bool(f.get("cluster_text_equal")) for f in facts),
                  real_texts=sum(f["kind"] == "real" for f in facts))
    if mlx_info:
        cmf = [cos(f["fp32"], f["mlx_fp16"]) for f in facts]
        cmq = [cos(q["fp32"], q["mlx_fp16"]) for q in questions]
        dp = [abs(p["p"] - p["p_mlx"]) for p in pairs]
        checks.update(mlx_vs_fp32_facts_cos_min=min(cmf), mlx_vs_fp32_facts_cos_mean=float(np.mean(cmf)),
                      mlx_vs_fp32_questions_cos_min=min(cmq), mlx_vs_fp32_max_abs_diff=float(max(
                          np.abs(np.array(x["fp32"]) - np.array(x["mlx_fp16"])).max() for x in facts + questions)),
                      p_mlx_vs_p_max_abs=max(dp))
        print(f"MLX fp16 vs fp32: facts cos min {min(cmf):.7f} mean {np.mean(cmf):.7f}; questions cos min {min(cmq):.7f}; "
              f"max |P diff| {max(dp):.2e}")
    for x in facts + questions:
        for k in [k for k in x if k.startswith("_")]:
            del x[k]
    # P bar: the requested 1e-3, unless fp16 ALONE (MLX's own output vs fp32) already moves P further; then 2x the largest
    # such move, rounded up to one significant digit
    p_bar_req = 1e-3
    fx = max(abs(p["p"] - p["p_mlx"]) for p in pairs) if mlx_info else None
    worst = max([x for x in (fx, drift["p_max_abs"] if drift else None) if x is not None], default=0.0)
    p_bar = p_bar_req
    if 2 * worst > p_bar_req:
        e = math.floor(math.log10(2 * worst))
        p_bar = float(f"{math.ceil(2 * worst / 10 ** e) * 10 ** e:.1g}")
    why = (f"p_bar = {p_bar:g}: the requested {p_bar_req:g} is smaller than what fp16 alone does to P (MLX fp16 on CPU vs the "
           f"fp32 reference: max {fx:.2e} on this file's {len(pairs)} pairs" +
           (f", max {drift['p_max_abs']:.2e} over {drift['pairs']} (photo, question) pairs heads.py recorded for "
            f"{drift['photos_per_library']} seeded photos of each of these two libraries, {drift['pairs_over_1e3']} of them over 1e-3, {drift['pairs_across_07']} on the other "
            f"side of P = .7" if drift else "") + "), so p_bar = 2 x that maximum, rounded up"
           ) if p_bar > p_bar_req else f"p_bar = {p_bar:g} (fp16 alone moves P by at most {worst:.2e})"
    if not mlx_info:
        why = f"p_bar = {p_bar:g} as requested; MLX did not run, so how far fp16 alone moves P is NOT measured"
    doc = dict(
        note="Golden fixture for the phone's facts text encoder (towervec MiniLM-L12, 384-d) and the towervec head. Made by "
             "eval/cascade_port/encoder_golden.py from PUBLIC dev data only (dev libraries " + ", ".join(LIBS) + "; writer lines "
             "p2b_all + repeat-stop v2; recorded dev scorer questions). kind=real: the photo's own place/taken and line, text "
             "verified equal to the cluster's facts parquet; kind=synthetic: an edge case, see name/note.",
        how_to_check="For every facts entry: ids_device = BertTokenizer(vocab.txt of towervec_encoder_fp16).factsInput(text) must "
                     "equal ids (questions: questionInput(question)); vector_device = mean over all tokens of BertModel(ids, "
                     "tokenTypeIds zeros, attention mask ones).hiddenStates (no normalisation). PASS if cosine(vector_device, "
                     "fp32) >= 0.9999 for EVERY text (mlx_fp16 = what MLX fp16 gave on CPU; the device runs fp16 on Metal, so "
                     "expect it about as far from fp32 as mlx_fp16 is, not bit-equal to mlx_fp16; omitting tokenTypeIds gave "
                     "cosine min .72 / mean .93 in RESULTS 68). Then for every pair: P_device = "
                     "ScorerHead(towervec_head.fphead).fold(q: device vector of questions[question], t: questions[question].t)"
                     ".p(f: device vector of facts[fact], v: facts[fact].v). PASS if |P_device - p| <= p_bar for EVERY pair; "
                     "report the max. " + why + ". Linux (FactsEncoderGoldenTests.swift) checks ids, prefixes and the head on "
                     "this file's own vectors.",
        cos_bar=0.9999, p_bar=p_bar, p_bar_requested=p_bar_req,
        pooling="mean over ALL tokens incl. [CLS]/[SEP] of the last hidden state, not normalised; token_type_ids 0; mask 1; "
                "facts [CLS]+ids[:254]+[SEP], question [CLS]+ids[:64]+[SEP]; no padding",
        reference_fp32="ORIGINAL weights " + str(SRC / "weights.pt") + " (enc.*) via eval/facts_scorer/scorers.py Scorer, "
                       "PyTorch " + torch.__version__ + " fp32 CPU, one unpadded sequence at a time",
        reference_mlx_fp16=(mlx_info + "; weights " + str(ENC16 / "model.safetensors") + " (= the app's bundled copy), fp16 "
                            "compute, mean in fp16 then fp32") if mlx_info else None,
        tokenizer="Hugging Face fast tokenizer of sentence-transformers/all-MiniLM-L12-v2 (what the scorers used)",
        encoder_safetensors_bytes=(ENC16 / "model.safetensors").stat().st_size, encoder_safetensors_sha256=sha(ENC16 / "model.safetensors"),
        head_file="facts/towervec_head.fphead", head_bytes=HEAD.stat().st_size, head_sha256=sha(HEAD),
        p_reference="scorers.Head (towervec, from the same weights.pt as the .fphead) in fp32 PyTorch CPU on the ROUNDED "
                    "fp32 vectors of this file; p_mlx = the same head on mlx_fp16 vectors; p_cluster = heads.py's recorded "
                    "fp32 P for that (photo, question) (batched vectors), null for edge cases",
        float_digits="vectors 7 significant digits; v, t (head inputs) 9 (float32-exact); p 9",
        checks=checks, facts=facts, questions=questions, pairs=pairs)
    out_path.write_text(json.dumps(doc, ensure_ascii=False))
    print(f"{out_path}: {len(facts)} facts texts ({sum(f['kind'] == 'real' for f in facts)} real), {len(questions)} questions, "
          f"{len(pairs)} pairs, {out_path.stat().st_size / 1e3:.0f} kB; checks {json.dumps(checks)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mlx":
        mlx_main(sys.argv[2], sys.argv[3])
    else:
        main()
