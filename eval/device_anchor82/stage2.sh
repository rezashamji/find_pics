#!/bin/bash
# M82 stage 2 (one GPU job, main env): PE vectors of the extension, the extension size N (the smallest prefix of devext_2000,
# in its seeded order, that brings the 4B judge's dog judge-yes to >= 50), then scoring: dev5k with ablations (both
# writers), dev5k + devext_N (main table), and the n=300 device-statistic simulation.
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
D=data/public/device_anchor82; PY=envs/fp/bin/python
[ -f $D/out/pe_devext_2000.npz ] || $PY eval/device_anchor82/pe.py $PWD/$D/manifest_devext_2000.parquet $D/out/pe_devext_2000.npz
N=$($PY - <<'PY'
import pandas as pd
D = "data/public/device_anchor82"
q = "Is there a real dog anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
a = pd.read_parquet(f"{D}/out/judge_dev5k/j4_0.parquet"); a = a[a.q == q]
b = pd.read_parquet(f"{D}/out/judge_devext_2000/j4_0.parquet"); b = b[b.q == q].set_index("key")
have = int((a.p >= 0.7).sum())
order = pd.read_parquet(f"{D}/manifest_devext_2000.parquet").key.tolist()
n = 0
for i, k in enumerate(order):
    if have >= 50:
        break
    have += int(b.p.loc[k] >= 0.7); n = i + 1
print(n)
PY
)
echo "extension N=$N"
$PY eval/device_anchor82/subset.py --ext=$N
$PY eval/device_anchor82/score.py --manifest=dev5k --tag=dev5k --sources=base_vllm,trained_vllm61d \
  --ablate=trained_vllm61d,base_vllm --pairs=trained_vllm61d:abl_shuffle_trained_vllm61d,base_vllm:abl_shuffle_base_vllm,trained_vllm61d:abl_zero_v_trained_vllm61d,base_vllm:abl_zero_v_base_vllm,abl_zero_v_trained_vllm61d:abl_zero_v_shuffle_trained_vllm61d,abl_zero_v_base_vllm:abl_zero_v_shuffle_base_vllm
$PY eval/device_anchor82/device_stat.py --tag=dev5k
$PY eval/device_anchor82/score.py --manifest=dev5k,devext_$N --tag=dev5kext --sources=base_vllm,trained_vllm61d
echo STAGE2_DONE
