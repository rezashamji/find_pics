"""RESULTS 55: the pixel veto on DEV, replayed exactly from the per-photo dump (dump.py: the m54 dev S1 run's cached 9B
answers; every dev b plan is one album with no filter/exclude question, so final round = judged photos with P >= 0.7).
A request changes only if its album question names a photo-level look (pixel_rule.photo_looks); then a judge yes
also needs the pixels to agree (pixel_rule.pixel_veto). Writes runs/cmp/r55_veto/scored.json (S1's scored.json with the
changed requests re-scored) for compare.py, and b55/veto_eval.txt (per request, photos removed by truth membership).
Usage: PYTHONPATH=src python eval/complex_bench/b55/veto_eval.py
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402
from score import score_turn  # noqa: E402


def main():
    from pixel_rule import photo_looks, pixel_veto
    S1 = ROOT / "runs" / "m54" / "m54_dev_s1"
    rows = json.loads((S1 / "scored.json").read_text())
    Q = {q["qid"]: q for q in json.loads((ROOT / "queries" / "dev.json").read_text())["queries"]}
    pix = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT / "b55" / "dump_*" / "pix_*.parquet"))])
    pix = pix.assign(item_id=pix.item_id.astype(str)).drop_duplicates("item_id").set_index("item_id")
    dumps = {json.loads(Path(f).read_text())["qid"]: f for f in glob.glob(str(ROOT / "b55" / "dump_*" / "*@*.json"))}
    out, changed = [], 0
    for r in rows:
        qid = r["qid"]
        if qid not in dumps:
            continue
        meta = json.loads(Path(dumps[qid]).read_text())
        if len(meta["plan"]["albums"]) != 1 or len(Q[qid]["turns"]) != 1:
            continue
        a = meta["plan"]["albums"][0]
        looks = photo_looks(a["judge_question"]) | photo_looks(a.get("filter_question"))
        if not looks:
            continue
        assert not a.get("filter_question") and not a.get("exclude_question"), qid
        d = pd.read_parquet(dumps[qid].replace(".json", ".parquet")).set_index("item_id").join(pix)
        assert set(d.index[d.ret_final]) == set(d.index[d.judged0 & (d.p_main0 >= 0.7)]), qid
        keep = d.ret_final & ~d.apply(lambda x: pixel_veto(looks, x.to_dict()), axis=1)
        s = score_turn(r["cat"], [sorted(d.index[keep])], Q[qid]["answers"][-1])
        rem = d[d.ret_final & ~keep]
        out.append(f"{qid} [{r['cat']}] {sorted(looks)} F1 {r['final']['f1']:.3f} -> {s['f1']:.3f}  returned {int(d.ret_final.sum())} -> "
                   f"{int(keep.sum())} (removed {len(rem)}: truth {int(rem.in_truth.sum())}, not truth {int((~rem.in_truth).sum())})"
                   f"  | {Q[qid]['turns'][-1][:60]}")
        out.append("    removed truth: " + " ".join(rem.index[rem.in_truth][:12]) + " | removed other (first 12): "
                   + " ".join(rem.index[~rem.in_truth][:12]))
        r["final"] = {**r["final"], **{k: s[k] for k in ("p", "r", "f1", "n_got")}}
        changed += 1
    od = ROOT / "runs" / "cmp" / "r55_veto"
    od.mkdir(parents=True, exist_ok=True)
    (od / "scored.json").write_text(json.dumps(rows, indent=1))
    txt = f"requests whose question names a photo-level look: {changed}\n" + "\n".join(out)
    (ROOT / "b55" / "veto_eval.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
