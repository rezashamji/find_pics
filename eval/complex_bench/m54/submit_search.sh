#!/bin/bash
# RESULTS 54: submit the search stage after both plan stages finished: test in 10 library shards (2 libraries each),
# dev in 4 (judge answers mostly cached). Each job runs s1 then s0 on its libraries (search_job.sh).
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
for s in s0 s1; do for sp in test dev; do
  [ -f eval/complex_bench/runs/m54/${sp}_${s}_plan/plans.json ] || { echo "missing plans $sp $s"; exit 1; }
done; done
for k in $(seq 0 9); do bash scripts/race_sbatch.sh fp_m54_test_$k 5:00:00 bash eval/complex_bench/m54/search_job.sh test $k 10; done
for k in $(seq 0 3); do bash scripts/race_sbatch.sh fp_m54_dev_$k 3:00:00 bash eval/complex_bench/m54/search_job.sh dev $k 4; done
