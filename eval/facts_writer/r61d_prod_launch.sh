#!/bin/bash
# RESULTS 61d (1): wait for the product adapter, write its dev facts, then the R59d dev cascade with the 61c scorer
# (tvw_t2b_final_xfit_full, unchanged, as pre-registered); TRAIN libraries in the facts parquet are the 61c cross-fit
# lines (not scored: TRAIN is in the product adapter's training data). Also the same with repeat-stop v2 applied
# (scorer tvw_t2b_final_rs2_xfit_full) for information. Background, login node.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
M=eval/facts_scorer/models; D=eval/facts_scorer/data
until [ -f eval/facts_writer/models/p2b_all/final/adapter_model.safetensors ] && grep -q TRAIN_DONE eval/facts_writer/logs/p2b_all.log; do sleep 60; done
DEVU=$(python -c "import json; print(','.join(json.load(open('eval/complex_bench/splits.json'))['dev']))")
export SPECS="kempner_requeue:gpu:nvidia_h100_80gb_hbm3:1 kempner_rtx:gpu:1 kempner_h100:gpu:1 kempner_h200:gpu:1"   # no A100-40GB: the lazily loaded 9B judge fails engine init there (61d: Mamba cache blocks)
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
NDEV=$(echo $DEVU | tr , '\n' | wc -l)
if [ $(ls eval/facts_writer/facts/p2b_all_dev/*.jsonl 2>/dev/null | wc -l) -lt $NDEV ]; then
  inq fp_r61d_pw || MEM=160G bash scripts/race_sbatch.sh fp_r61d_pw 05:00:00 bash eval/facts_writer/r61d_prod_job.sh $DEVU > /dev/null
  until [ $(ls eval/facts_writer/facts/p2b_all_dev/*.jsonl 2>/dev/null | wc -l) -ge $NDEV ]; do sleep 60; done
fi
for V in plain rs2; do
  T=p2b_devmix_$V; FD=eval/facts_writer/facts/$T; mkdir -p $FD
  SRC=t2b_final_xfit; [ $V = rs2 ] && SRC=t2b_final_rs2_xfit
  for f in eval/facts_writer/facts/$SRC/*.jsonl; do ln -sf $(realpath $f) $FD/$(basename $f); done   # TRAIN (+ old dev, replaced next)
  if [ $V = plain ]; then for f in eval/facts_writer/facts/p2b_all_dev/*.jsonl; do rm -f $FD/$(basename $f); cp $f $FD/; done
  else
    mkdir -p eval/facts_writer/facts/p2b_all_dev_rs2
    python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from findpics.facts_line import apply_repeat_stop
for f in Path("eval/facts_writer/facts/p2b_all_dev").glob("*.jsonl"):
    out = []
    for line in open(f):
        r = json.loads(line); t, st = apply_repeat_stop(r["facts"]["caption"], 2); r["facts"] = {"caption": t}; r["stopped"] = st
        out.append(json.dumps(r))
    Path("$FD", f.name).unlink(missing_ok=True); Path("$FD", f.name).write_text("\n".join(out) + "\n")
PY
  fi
  [ -f $D/facts_w_${T}m.parquet ] || python eval/facts_scorer/data.py --facts=$FD --tag=w_${T}m --meta=1 --salvage=0 --pairs=0
  F=$D/facts_w_${T}m.parquet; TV=$M/tvw_t2b_final_xfit_full; [ $V = rs2 ] && TV=$M/tvw_t2b_final_rs2_xfit_full
  for k in $(seq 0 12); do
    { grep -q CASCADE_DONE eval/facts_writer/replay/c61d_dev_$T/cascade_$k.log 2>/dev/null || inq fp_r61d_d_${T}_$k; } && continue
    MEM=128G bash scripts/race_sbatch.sh fp_r61d_d_${T}_$k 04:00:00 bash eval/facts_scorer/cascade_job.sh \
      eval/facts_writer/replay/c61d_dev_$T $k 13 $TV $M/ce12B_full $F r59d --save_sub=r61d_d_$T > /dev/null
  done
done
echo "R61D_PROD_SUBMITTED $(date)"
