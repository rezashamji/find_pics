"""M48-scale step 4: the learned head as fast mode's RANKING (instead of cosine), same judge, same shipped stop rule.
No model runs: the judge's answers are the stored ones (audit4: phone real weights at the phone cuts; ev16: Qwen3-VL-4B
16-bit yes set at 0.7), only the ORDER in which fast mode asks changes.

  cosine = the shipped order (tune_stop_rule.audit4 / ev16: the phone's look score)
  head   = the head's P(yes) from eval/learned_head/scale/scores_{RUN}.parquet (unseen query, unseen library)
Measures, pooled over every search x library (judge-relative, = found / judge-yes):
  (a) pure ranking: judge calls in rank order until 50 / 80 / 90 / 95% of the judge's yes are found
  (b) the SHIPPED fast-mode rule (fastmode_cut_replay.SHIPPED, tune_stop_time.sim_timed, 20 seeds): first answer's
      recall and calls; and "calls to match": with the head's order, the calls at which each case reaches the number
      cosine's first answer found (same seed).
  python eval/learned_head/rank_replay.py RUN [RUN ...] -> eval/learned_head/scale/rank_replay.txt
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "eval")
import tune_stop_rule as T  # noqa: E402
import tune_stop_time as TT  # noqa: E402

S = Path("eval/learned_head/scale")
SHIPPED = "timed round_calls=120 inter=0 topup=30 head=150 a1=0.2 fast_stop=480 stop_topup=150"
L = []


def say(s):
    print(s, flush=True); L.append(s)


def reorder(cases, score):
    out = []
    for c in cases:
        s = np.array([score.get((c["qkey"], str(i)), np.nan) for i in c["ids"]], float)
        assert not np.isnan(s).any(), (c["key"], int(np.isnan(s).sum()))
        o = np.argsort(-s, kind="stable")
        out.append(dict(c, y=c["y"][o], ids=c["ids"][o]))
    return out


def calls_to(cases, fr):
    tot = 0
    for c in cases:
        cum = np.cumsum(c["y"]); need = int(np.ceil(fr * c["y"].sum()))
        tot += int(np.argmax(cum >= need)) + 1 if need > 0 else 0
    return tot


def shipped(cases, base=None):
    kind, r = TT.parse(SHIPPED)
    found = calls = kept = match_calls = 0
    per = {}
    for s in range(T.SEEDS):
        for c in cases:
            order, _ = TT.sim_timed(c["y"], r, s)
            f = int(c["y"][order].sum())
            found += f; calls += len(order); kept += int(c["y"].sum())
            per[(c["key"], s)] = (f, len(order))
            if base is not None:
                bf, bc = base[(c["key"], s)]
                cum = np.cumsum(c["y"][order])
                if bf == 0:
                    mc = 0
                elif cum[-1] >= bf:
                    mc = int(np.argmax(cum >= bf)) + 1
                else:                              # head's first answer found fewer: continue down its ranking
                    done = np.zeros(len(c["y"]), bool); done[order] = True
                    rest = np.where(~done)[0]
                    cum2 = cum[-1] + np.cumsum(c["y"][rest])
                    mc = len(order) + int(np.argmax(cum2 >= bf)) + 1
                match_calls += mc
    return dict(recall=found / kept, calls=calls / T.SEEDS, match_calls=match_calls / T.SEEDS), per


def main():
    runs = sys.argv[1:]
    a4 = T.audit4()
    for c in a4:
        c["qkey"] = c["key"]
    ev = T.ev16()
    for c in ev:
        c["qkey"] = (c["query"], c["user"])
    for run in runs:
        X = pd.read_parquet(run if run.endswith(".parquet") else S / f"scores_{run}.parquet")
        say(f"\n=== head = {run} mlp (3-seed mean), unseen queries and libraries ===")
        for name, cases, src in (("audit4 (6 searches x 4 libraries, phone real-weight judge at 0.7/0.999)", a4, "A4"),
                                 ("ev16 (12 searches x 16 libraries, Qwen3-VL-4B 16-bit at 0.7)", ev, "EV16")):
            d = X[X.src == src]
            if src == "A4":
                sc = dict(zip(zip(d["query"], d.item.astype(str)), d.mlp))
            else:
                sc = dict(zip(zip(zip(d["query"], d.lib), d.item.astype(str)), d.mlp))
            hc = reorder(cases, sc)
            yes = sum(int(c["y"].sum()) for c in cases); n = sum(len(c["y"]) for c in cases)
            say(f"-- {name}: {len(cases)} cases, {n} photos, {yes} judge-yes")
            say("   (a) calls in rank order to find  50% / 80% / 90% / 95% of the judge's yes (pooled sum)")
            for lab, cc in (("cosine (shipped look)", cases), (f"head {run}", hc)):
                say(f"       {lab:26s} " + " / ".join(f"{calls_to(cc, f):7d}" for f in (.5, .8, .9, .95)))
            b, per = shipped(cases)
            h, _ = shipped(hc, base=per)
            say(f"   (b) shipped fast mode, first answer (20 seeds): cosine recall {b['recall']:.3f} with {b['calls']:.0f} calls; "
                f"head recall {h['recall']:.3f} with {h['calls']:.0f} calls; head reaches cosine's found with "
                f"{h['match_calls']:.0f} calls ({100 * (1 - h['match_calls'] / b['calls']):+.1f}% saved)")
            for q in sorted({c["query"] if src == "EV16" else c["key"] for c in cases}):
                cq = [c for c in cases if (c.get("query") or c["key"]) == q]
                hq = [c for c in hc if (c.get("query") or c["key"]) == q]
                bq, pq = shipped(cq)
                hq_, _ = shipped(hq, base=pq)
                say(f"       {q:24s} 80%: cos {calls_to(cq, .8):6d} head {calls_to(hq, .8):6d} | first answer recall "
                    f"cos {bq['recall']:.3f} head {hq_['recall']:.3f}; calls to match {bq['calls']:.0f} -> {hq_['match_calls']:.0f}")
    (S / os.environ.get("RR_OUT", "rank_replay.txt")).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
