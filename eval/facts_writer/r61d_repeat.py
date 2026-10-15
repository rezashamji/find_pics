"""RESULTS 61d (3): REPEAT-STOP applied to a writer's existing greedy outputs (exact for greedy decoding: the decision
depends only on the text so far, so stopping while streaming = cutting the full output; findpics.facts_line).
Writes eval/facts_writer/facts/<out_tag>/ (same layout; facts cut; n_out = tokens decoded until the stop, i.e. the
tokens of the line up to and including the "," / ";" that completed the repeated item; unchanged lines keep n_out)
and a summary: lines stopped, among looping / non-looping lines (loop = quality.rep), tokens before / after.
  python eval/facts_writer/r61d_repeat.py <in_tag> <out_tag> [--version=1|2] [--tok=models/qwen3vl_2b_mlx4sim] [--out=summary.txt]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from findpics.facts_line import apply_repeat_stop, repeat_scan  # noqa: E402
from quality import rep  # noqa: E402


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    arg = lambda k, d=None: next((x.split("=", 1)[1] for x in sys.argv if x.startswith(f"--{k}=")), d)
    src, dst = Path("eval/facts_writer/facts") / a[0], Path("eval/facts_writer/facts") / a[1]
    from transformers import AutoTokenizer
    ver = int(arg("version", "1"))
    tok = AutoTokenizer.from_pretrained(arg("tok", "models/qwen3vl_2b_mlx4sim"))
    sp = json.loads(Path("eval/complex_bench/splits.json").read_text()); dev = set(sp["dev"])
    dst.mkdir(parents=True, exist_ok=True)
    rows = []   # (split, loop_before, stopped, n_before, n_after, loop_after)
    for f in sorted(src.glob("*.jsonl")):
        out = []
        for line in open(f):
            r = json.loads(line)
            t = r["facts"]["caption"]; n0 = r.get("n_out", 0)
            s = repeat_scan(t, ver)
            if s is None:
                t1, n1 = t, n0
            else:
                t1, _ = apply_repeat_stop(t, ver)
                n1 = len(tok(t[:s[1]], add_special_tokens=False).input_ids)
            rows.append(("dev" if f.stem in dev else "train", rep(t, n0), s is not None, n0, n1, rep(t1, n1)))
            r2 = dict(r); r2["facts"] = {"caption": t1}; r2["n_out"] = n1; r2["n_out_full"] = n0; r2["stopped"] = s is not None
            out.append(json.dumps(r2))
        (dst / f.name).write_text("\n".join(out) + "\n")
    L = [f"repeat-stop version {ver}", "set | photos | loop before | loop after | at 120 cap before / after | stopped (all) | stopped among looping | stopped among NON-looping "
         "| tokens mean before / after | p90 before / after"]
    for name in ("dev", "train", "all"):
        R = [x for x in rows if name == "all" or x[0] == name]
        lb = np.array([x[1] for x in R]); st = np.array([x[2] for x in R]); la = np.array([x[5] for x in R])
        n0 = np.array([x[3] for x in R]); n1 = np.array([x[4] for x in R])
        L.append(f"{name} | {len(R):,} | {lb.sum():,} ({lb.mean():.3f}) | {la.sum():,} ({la.mean():.3f}) | {(n0 >= 120).sum():,} / {(n1 >= 120).sum():,} | {st.sum():,} "
                 f"({st.mean():.3f}) | {(st & lb).sum():,}/{lb.sum():,} | {(st & ~lb).sum():,}/{(~lb).sum():,} | "
                 f"{n0.mean():.1f} / {n1.mean():.1f} | {np.percentile(n0, 90):.0f} / {np.percentile(n1, 90):.0f}")
    txt = "\n".join(L); print(txt)
    if arg("out"):
        Path(arg("out")).write_text(txt + "\n")


if __name__ == "__main__":
    main()
