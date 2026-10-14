"""RESULTS 59 step 2: index cost of writing facts on the phone. Token counts are MEASURED (vLLM, the phone's simulated
4-bit weights, TRAIN+DEV photos); seconds on the phone are an ESTIMATE:
  per photo = t0 + n_out / r
  t0 = vision tower + prompt prefill: the phone's measured judge call (RESULTS 44: 35 photos/min cool = 1.7 s, 12-19/min
       hot = 3.2-5.0 s; the facts prompt's ~550 text tokens are the same for every photo -> prefix-cached if placed
       before the image), r = decode tokens/s: NOT measured on the phone (no measurement exists); assumed 4B 4-bit 15
       or 30 tok/s cool, halved hot (RESULTS 44's throttling), 2B x1.8 and 0.8B x3.5 (bandwidth-bound decode scales
       with weight bytes read per token).
  python eval/facts_scorer/index_cost.py [--n=187084] -> stdout
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "eval/facts_scorer")
from data import salvage  # noqa: E402

TAGS = {"q3vl4bq4r": ("Qwen3-VL-4B 4-bit, rep 1.05", 1.0), "q3vl4bq4s": ("Qwen3-VL-4B 4-bit, SHORT keyword line, rep 1.05", 1.0), "q3vl2bq4r": ("Qwen3-VL-2B 4-bit, rep 1.1", 1.8),
        "q35_08bq4r": ("Qwen3.5-0.8B 4-bit, rep 1.1", 3.5), "q3vl4bq4": ("Qwen3-VL-4B 4-bit, greedy", 1.0),
        "q3vl2bq4": ("Qwen3-VL-2B 4-bit, greedy", 1.8)}


def main():
    N = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--n=")), "187084"))
    print(f"Index cost of facts on the phone, {N:,} photos (Reza's library). Tokens measured; seconds ESTIMATED (see docstring).")
    print("writer | photos | strict JSON | salvaged | empty | out tokens mean / median / p90 | at cap | prompt tokens | "
          "s/photo cool r=15 / r=30 | hours cool r=15 / r=30 | hours hot r=15 / r=30")
    for tag, (name, speed) in TAGS.items():
        R = []
        for f in glob.glob(f"eval/facts_scorer/phone_facts/{tag}/*.jsonl"):
            for line in open(f):
                try:
                    R.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if not R:
            continue
        n_out = np.array([r["n_out"] for r in R]); n_in = np.array([r["n_prompt"] for r in R])
        strict = np.mean([r["facts"] is not None for r in R])
        salv = np.mean([r["facts"] is None and salvage(r.get("raw")) is not None for r in R])
        m = n_out.mean()
        cells = []
        for hot in (False, True):
            t0 = 4.0 if hot else 1.7
            for r in (15, 30):
                rr = r * speed / (2 if hot else 1)
                cells.append((t0 + m / rr, N * (t0 + m / rr) / 3600))
        print(f"{name} | {len(R):,} | {strict:.3f} | {salv:.3f} | {1 - strict - salv:.3f} | {m:.0f} / {np.median(n_out):.0f} / "
              f"{np.percentile(n_out, 90):.0f} | {(n_out >= (160 if tag.endswith('s') else 700)).mean():.3f} | {n_in.mean():.0f} | "
              f"{cells[0][0]:.1f} / {cells[1][0]:.1f} | {cells[0][1]:,.0f} / {cells[1][1]:,.0f} | {cells[2][1]:,.0f} / {cells[3][1]:,.0f}")
    print("For scale: the judge alone at the measured cool rate (35/min) = 89 h for one pass over 187,084 photos; "
          "the 27B facts are ~243 tokens/photo (compact JSON).")


if __name__ == "__main__":
    main()
