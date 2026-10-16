"""RESULTS 68 (4): towervec's TEXT ENCODER (its own fine-tuned all-MiniLM-L12-v2, weights.pt `enc.*`) in a phone format.
  python eval/cascade_port/encoder.py export
     -> eval/cascade_port/models/towervec_encoder/{config.json, model.safetensors (fp32), vocab.txt, tokenizer.json,
        tokenizer_config.json, special_tokens_map.json, modules.json, sentence_bert_config.json, 1_Pooling/config.json}
        (= a sentence-transformers / Hugging Face BERT directory: MLXEmbedders' BertModel loads it as is, its sanitize()
        renames the HF keys) and towervec_encoder_fp16/ (same, fp16 weights)
     -> data/public/cascade_port/encoder/lines.json: 1,000 dev facts lines (seeded) + every recorded dev scorer question,
        their token ids exactly as Scorer builds them, and the ORIGINAL PyTorch fp32 (CPU) mean-pooled vectors
     -> ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/bert_tokens.json + bert_vocab.txt: WordPiece golden ids (HF
        fast tokenizer, add_special_tokens=False) for the Swift BertTokenizer, on those lines + hand cases
  python eval/cascade_port/encoder.py hfcheck    re-load the exported dirs with transformers (fp32, fp16) -> cosines
  (MLX check: eval/cascade_port/encoder_mlx.py inside tools/mlxpy, see that file)
"""
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "eval/facts_scorer")
SRC = Path("eval/facts_scorer/models/tvw_t2b_final_rs2_xfit_full")
OUT = Path("eval/cascade_port/models")
D = Path("data/public/cascade_port/encoder")
FIX = Path("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures")
FACTS_DEV = "eval/facts_scorer/data/facts_w_p2b_devmix_rs2m.parquet"
HAND = ["place: Zürich, Schweiz. taken: May 2014. scene: café; text: CRÈME BRÛLÉE 4,50€", "Ærø Ø ß İstanbul Œuvre naïve",
        "東京タワー 夜景 tokyo tower", "emoji 🎉🎂 party!!", "tab\there\nnewline\r\x00null �replacement",
        "supercalifragilisticexpialidocious" * 4, "a-b_c.d,e;f:g/h\\i(j)k[l]m{n}o<p>q'r\"s?t!u@v#w$x%y^z&*+=|~`",
        "Is the person in the red box wearing a hat?", "", "   ", "naïve café résumé", "Ｆｕｌｌｗｉｄｔｈ ＡＢＣ １２３",
        "ﬁ ligature ﬂ", "Hello​world zero-width", "मराठी हिन्दी", "Привет мир", "x" * 120, "don't won't 1990s 4x4 3.5mm"]


def base_dir():
    from huggingface_hub import snapshot_download
    return Path(snapshot_download("sentence-transformers/all-MiniLM-L12-v2"))


def load_original():
    from scorers import Scorer
    S = Scorer(SRC, facts={})
    return S


def export():
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, BertConfig
    import pandas as pd
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (fp32 CPU reference)"
    b = base_dir()
    sd = torch.load(SRC / "weights.pt", map_location="cpu")
    enc = {k[4:]: v.float().contiguous() for k, v in sd.items() if k.startswith("enc.")}
    cfg = json.loads((b / "config.json").read_text())
    for name, dt in (("towervec_encoder", torch.float32), ("towervec_encoder_fp16", torch.float16)):
        d = OUT / name
        (d / "1_Pooling").mkdir(parents=True, exist_ok=True)
        save_file({k: v.to(dt) for k, v in enc.items()}, str(d / "model.safetensors"), metadata={"format": "pt"})
        c = dict(cfg); c["torch_dtype"] = "float32" if dt == torch.float32 else "float16"
        c["_find_pics"] = ("towervec12B encoder of eval/facts_scorer/models/tvw_t2b_final_rs2_xfit_full (RESULTS 61d/65): "
                           "mean pool over ALL tokens incl. [CLS]/[SEP], no normalisation; facts: [CLS]+ids[:254]+[SEP]; "
                           "question: [CLS]+ids[:64]+[SEP]; token_type 0")
        (d / "config.json").write_text(json.dumps(c, indent=1))
        for f in ("vocab.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "modules.json"):
            shutil.copy(b / f, d / f)
        (d / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 256, "do_lower_case": False}))
        (d / "1_Pooling" / "config.json").write_text(json.dumps({
            "word_embedding_dimension": 384, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False, "pooling_mode_mean_sqrt_len_tokens": False}))
        print(d, sum(f.stat().st_size for f in d.rglob("*") if f.is_file()))
    # lines: 1,000 dev facts lines + every recorded dev scorer question
    S = load_original()
    FT = pd.read_parquet(FACTS_DEV)
    rng = random.Random(68)
    lines = rng.sample(FT.text.tolist(), 1000)
    qs = set()
    for f in Path("data/public/cascade_port/full/dev").glob("*.json"):
        for r in json.loads(f.read_text())["requests"]:
            qs.update(r["tv"]); qs.update(r["vec"])
    qs = sorted(qs)
    fid = S.tok.ids(lines, 256)                       # Scorer._facts_tok: ids(texts, max_len) then single(.., 256)
    qid = S.tok.ids(qs, 64)                           # Scorer._qvecs: ids(miss, 64) then single(.., 66)
    cls, sep = S.tok.cls, S.tok.sep
    fseq = [[cls] + a[:254] + [sep] for a in fid]
    qseq = [[cls] + a[:64] + [sep] for a in qid]
    with torch.no_grad():
        fv = torch.cat([S.model.embed(*S.tok.single(fid[i:i + 64], 256)) for i in range(0, len(fid), 64)]).numpy()
        qv = torch.cat([S.model.embed(*S.tok.single(qid[i:i + 64], 66)) for i in range(0, len(qid), 64)]).numpy()
    D.mkdir(parents=True, exist_ok=True)
    (D / "lines.json").write_text(json.dumps(dict(facts=lines, facts_ids=fseq, facts_vec=fv.tolist(), questions=qs,
                                                  question_ids=qseq, question_vec=qv.tolist())))
    print(len(lines), "facts lines,", len(qs), "questions; tokens per facts line mean", np.mean([len(s) for s in fseq]))
    # WordPiece golden for Swift
    tk = AutoTokenizer.from_pretrained(str(b))
    texts = lines + qs + HAND
    ids = tk(texts, add_special_tokens=False)["input_ids"]
    shutil.copy(b / "vocab.txt", FIX / "bert_vocab.txt")
    (FIX / "bert_tokens.json").write_text(json.dumps(dict(cases=[dict(text=t, ids=i) for t, i in zip(texts, ids)]),
                                                     ensure_ascii=False))
    print("bert_tokens.json", len(texts), "cases;", (FIX / "bert_tokens.json").stat().st_size)


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def hfcheck():
    from transformers import AutoModel
    L = json.loads((D / "lines.json").read_text())
    for name in ("towervec_encoder", "towervec_encoder_fp16"):
        m = AutoModel.from_pretrained(str(OUT / name), torch_dtype=torch.float32).eval()
        for kind in ("facts", "question"):
            seqs = L[f"{kind}_ids"]
            out = []
            with torch.no_grad():
                for s in seqs:
                    x = torch.tensor([s])
                    h = m(input_ids=x, token_type_ids=torch.zeros_like(x), attention_mask=torch.ones_like(x)).last_hidden_state
                    out.append(h.mean(1)[0].numpy())
            c = cos(out, L[f"{kind}_vec"])
            print(f"HF reload {name} (computed fp32) {kind}: n {len(c)} cosine min {c.min():.6f} mean {c.mean():.6f}; "
                  f"max |diff| {np.abs(np.array(out) - np.array(L[f'{kind}_vec'])).max():.2e}")


if __name__ == "__main__":
    {"export": export, "hfcheck": hfcheck}[sys.argv[1]]()
