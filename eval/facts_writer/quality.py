"""RESULTS 61: per-photo quality of a writer's short facts line vs the 27B's (eval/facts_writer/short27b), on DEV photos
and on TRAIN photos the writer never trained on (cross-fit), plus output tokens and repetition.
Per field (parsed by short_format.parse): set-of-terms precision / recall / F1 for objects (singular nouns, common.norm),
colours (each "colour thing" phrase; and colour words alone), scene / activity / look / event (words); people count
within 1 (when the 27B gives one); text: shares a word with the 27B's visible text (when the 27B has text).
Repetition = output at the 120-token cap, or any word 3-gram occurring >= 3 times in the line.
  python eval/facts_writer/quality.py <tag> ... [--out=file]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "eval/complex_bench")
from common import norm  # noqa: E402
from short_format import parse  # noqa: E402

COLW = set("red orange yellow green blue purple pink magenta teal turquoise brown beige grey gray black white golden gold "
           "silver navy violet maroon cyan cream tan khaki olive".split())


def load(tag):
    d = Path("eval/facts_writer/facts") / tag
    out = {}
    for f in d.glob("*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[str(r["item_id"])] = (r["facts"]["caption"], r.get("n_out", 0), f.stem)
    return out


def items(s):
    return [x.strip() for x in re.split(r",\s*", s or "") if x.strip()]


def prf(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    tp = len(a & b)
    p = tp / len(a) if a else 0.0; r = tp / len(b) if b else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rep(line, n_out):
    w = re.findall(r"\w+", line.lower())
    g = Counter(zip(w, w[1:], w[2:]))
    return n_out >= 120 or (max(g.values()) >= 3 if g else False)


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text())
    dev = set(sp["dev"])
    ref = load("short27b")
    lines = ["writer | set | photos | out tokens mean / p90 | repetition | objects P / R / F1 | colour phrases F1 | colour "
             "words F1 | scene F1 | activity F1 | look F1 | people count within 1 | text shared"]
    for tag in ["short27b"] + tags:
        W = load(tag)
        for name, keep in (("dev", lambda u: u in dev), ("TRAIN out-of-fold", lambda u: u not in dev)):
            ks = [k for k, v in W.items() if keep(v[2]) and k in ref]
            if not ks:
                continue
            agg = {f: [] for f in ("obj", "colp", "colw", "scene", "act", "look")}
            pc, tx, ntok, reps = [], [], [], []
            for k in ks:
                a, b = parse(W[k][0]), parse(ref[k][0])
                ntok.append(W[k][1]); reps.append(rep(W[k][0], W[k][1]))
                for f, fa, fb in (("obj", [norm(x) for x in items(a.get("objects"))], [norm(x) for x in items(b.get("objects"))]),
                                  ("colp", [x.lower() for x in items(a.get("colours"))], [x.lower() for x in items(b.get("colours"))]),
                                  ("colw", [w for w in re.findall(r"\w+", a.get("colours", "").lower()) if w in COLW],
                                   [w for w in re.findall(r"\w+", b.get("colours", "").lower()) if w in COLW]),
                                  ("scene", re.findall(r"\w+", a.get("scene", "").lower()), re.findall(r"\w+", b.get("scene", "").lower())),
                                  ("act", re.findall(r"\w+", a.get("activity", "").lower()), re.findall(r"\w+", b.get("activity", "").lower())),
                                  ("look", re.findall(r"\w+", a.get("look", "").lower()), re.findall(r"\w+", b.get("look", "").lower()))):
                    x = prf(fa, fb)
                    if x is not None:
                        agg[f].append(x)
                nb = re.match(r"\s*(\d+)", b.get("people", "")); na = re.match(r"\s*(\d+)", a.get("people", "") or "0")
                if nb:
                    pc.append(abs(int(nb.group(1)) - (int(na.group(1)) if na else 0)) <= 1)
                if b.get("text"):
                    tx.append(bool(set(re.findall(r"\w+", a.get("text", "").lower())) & set(re.findall(r"\w+", b["text"].lower()))))
            m = lambda f, j: np.mean([x[j] for x in agg[f]]) if agg[f] else float("nan")
            ntok = np.array(ntok)
            lines.append(f"{tag} | {name} | {len(ks):,} | {ntok.mean():.0f} / {np.percentile(ntok, 90):.0f} | {np.mean(reps):.3f} | "
                         f"{m('obj', 0):.2f} / {m('obj', 1):.2f} / {m('obj', 2):.2f} | {m('colp', 2):.2f} | {m('colw', 2):.2f} | "
                         f"{m('scene', 2):.2f} | {m('act', 2):.2f} | {m('look', 2):.2f} | {np.mean(pc):.2f} | {np.mean(tx):.2f}")
    txt = "\n".join(lines)
    print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
