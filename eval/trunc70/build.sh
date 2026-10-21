#!/bin/bash
# RESULTS 70 inputs (login node, CPU, background): capped / reordered writer lines -> facts parquets (data.py --meta=1, the
# exact path of the R61d parquets) -> check70.py. Usage: bash eval/trunc70/build.sh
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
python eval/trunc70/truncate.py --order=scene,objects,colours,event,look,activity,people,text --rks=48,64 || exit 1
for src in t2b_final_rs2_xfit p2b_devmix_rs2; do
  for tag in k16 k24 k32 k48 k64 r48 r64 rfull; do
    python eval/facts_scorer/data.py --facts=data/public/trunc70/facts/${src}_$tag --tag=w_r70_${src}_${tag}m --meta=1 --salvage=0 --pairs=0 || exit 1
  done
done
python eval/trunc70/check70.py
echo BUILD_DONE
