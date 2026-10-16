"""RESULTS 61d table: R59d cascade ALL difference vs the all-judge stack (exact bootstrap = r61b_table.diff) for the 2B
writer variants: 61c (no stop), repeat-stop v1 / v2 (scorers retrained), the scorer-seed placebo, and the PRODUCT adapter
(dev only; 61c scorer unchanged; plain and with repeat-stop v2), plus tokens / loop rate where the facts exist.
  python eval/facts_writer/r61d_table.py [--out=file]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from r61b_table import diff, toks  # noqa: E402

R = "eval/facts_writer/replay"
ROWS = [("2B 61b final, no stop (61c)", "t2b_final_xfit", f"{R}/c61c_train_t2b_final_xfit/r59d", f"{R}/c61c_dev_t2b_final_xfit/r59d"),
        ("2B 61b final + repeat-stop v1 (scorers retrained)", "t2b_final_rs_xfit", f"{R}/c61d_train_t2b_final_rs_xfit/r59d", f"{R}/c61d_dev_t2b_final_rs_xfit/r59d"),
        ("2B 61b final + repeat-stop v2 (scorers retrained) ADOPTED", "t2b_final_rs2_xfit", f"{R}/c61d_train_t2b_final_rs2_xfit/r59d", f"{R}/c61d_dev_t2b_final_rs2_xfit/r59d"),
        ("PLACEBO: 2B 61b final, no stop, scorers retrained at seed 1", "t2b_final_xfit", f"{R}/c61d_train_t2b_final_xfit_seed1/r59d", f"{R}/c61d_dev_t2b_final_xfit_seed1/r59d"),
        ("PRODUCT 2B (all TRAIN + extra), no stop, 61c scorer", "p2b_all_dev", None, f"{R}/c61d_dev_p2b_devmix_plain/r59d"),
        ("PRODUCT 2B + repeat-stop v2, v2 scorer", "p2b_all_dev_rs2", None, f"{R}/c61d_dev_p2b_devmix_rs2/r59d")]


def main():
    out_f = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    f = lambda x: "-" if x is None else f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}] (n={x[3]})"
    L = ["writer | facts tag | tokens mean (p90) | loop rate (quality.rep) | TRAIN R59d diff vs all-judge [95%] | dev R59d diff vs all-judge [95%] | bar (lower >= -.03 on both)"]
    for name, tag, tr, dv in ROWS:
        n, rep = toks(tag)
        t = diff("train", tr) if tr else None
        d = diff("dev", dv) if Path(dv).exists() else None
        ok = "-" if d is None else ("dev only: " if t is None else "") + ("PASS" if d[1] >= -0.03 and (t is None or t[1] >= -0.03) else "FAIL")
        tk = "-" if n is None else f"{n.mean():.1f} ({sorted(n)[int(.9 * len(n))]:.0f})"
        L.append(f"{name} | {tag} | {tk} | {'-' if rep is None else f'{rep:.3f}'} | {f(t)} | {f(d)} | {ok}")
    txt = "\n".join(L); print(txt)
    if out_f:
        Path(out_f).write_text(txt + "\n")


if __name__ == "__main__":
    main()
