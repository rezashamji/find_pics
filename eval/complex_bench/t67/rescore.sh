#!/bin/bash
# RESULTS 67: re-score EXISTING runs (no new product run) on the current truth (queries/) and the t67 c rewrite.
# Copies the run parts (original run dirs are not written to), scores each copy, compares.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics; source env.sh > /dev/null
R=eval/complex_bench/t67/rescore
declare -A SRC=( [ship_dev]=eval/facts_writer/replay/c61d_dev_t2b_final_rs2_xfit/r59d
                 [ship_train]=eval/facts_writer/replay/c61d_train_t2b_final_rs2_xfit/r59d
                 [alljudge_dev]=eval/complex_bench/runs/cmp/r60v2_base_search
                 [alljudge_train]=eval/facts_scorer/replay/train_alljudge )
for truth in cur t67; do
  for n in "${!SRC[@]}"; do
    sp=${n#*_}; d=$R/$truth/$n; mkdir -p $d; cp ${SRC[$n]}/*part*.json $d/
    Qf=eval/complex_bench/queries/$sp.json; [ $truth = t67 ] && Qf=eval/complex_bench/t67/queries/$sp.json
    python eval/complex_bench/score.py $Qf $d > /dev/null
  done
  for sp in dev train; do
    python eval/complex_bench/compare.py alljudge=$R/$truth/alljudge_$sp shipped=$R/$truth/ship_$sp --out=$R/compare_${truth}_$sp.txt > /dev/null
  done
done
for sp in dev train; do for n in ship alljudge; do
  python eval/complex_bench/compare.py cur_truth=$R/cur/${n}_$sp t67_truth=$R/t67/${n}_$sp --out=$R/truth_${n}_$sp.txt > /dev/null
done; done
echo done
