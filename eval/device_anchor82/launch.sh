#!/bin/bash
# M82 cluster anchor launcher (sbatch only; idempotent; every job restartable: scripts/race_sbatch.sh now sends every copy
# to kempner_requeue, A100-40GB / RTX PRO 6000 by the GPU type named here; H100/H200 are not used - Reza 11-05).
# Usage: bash eval/device_anchor82/launch.sh <set> <what>
#   set  = dev5k (the AUC set) | devext_<n> (extension, only if a question has < 50 judge-yes) | commons1k (parity pool)
#          | dev500 (MLX-vs-vLLM subset) | parity20
#   what = j4 | j9 | pe | vb (vLLM base writer) | mlx (MLX base + trained writers, ONE job over the manifest)
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
source env.sh
SET=$1; WHAT=$2
D=data/public/device_anchor82; MAN=$PWD/$D/manifest_$SET.parquet; O=$D/out
mkdir -p $O slurm/logs
inq() { squeue -u $USER -h -o %j | grep -qx "$1"; }
A="kempner:gpu:1 kempner_rtx:gpu:1"; RTX="kempner_rtx:gpu:1"
R="bash eval/device_anchor82/run.sh"
Q9=$(ls -d $PWD/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*/ | head -1); PE=$(ls -d $PWD/.cache/huggingface/hub/models--timm--PE-Core-B-16/snapshots/*/ | head -1)
J=fp_da82_${WHAT}_$SET
inq $J && { echo "$J already queued"; exit 0; }
case $WHAT in
  j4) [ -f $O/judge_$SET/j4_0.parquet ] || SPECS="$A" bash scripts/race_sbatch.sh $J 02:00:00 $R vllm $PWD/models/q3vl4b_real python eval/device_anchor82/judge.py \
        --model=$PWD/models/q3vl4b_real --tag=j4 --manifest=$MAN --shard=0/1 --out=$O/judge_$SET ;;
  j9) [ -f $O/judge_$SET/j9_0.parquet ] || SPECS="$RTX" bash scripts/race_sbatch.sh $J 02:00:00 $R vllm $Q9 python eval/device_anchor82/judge.py \
        --model=Qwen/Qwen3.5-9B --tag=j9 --manifest=$MAN --shard=0/1 --out=$O/judge_$SET ;;   # the 9B cannot init on A100-40GB
  pe) [ -f $O/pe_$SET.npz ] || SPECS="$A" bash scripts/race_sbatch.sh $J 01:00:00 $R fp $PE envs/fp/bin/python eval/device_anchor82/pe.py \
        $MAN $O/pe_$SET.npz ;;
  vb) SPECS="$A" bash scripts/race_sbatch.sh $J 02:00:00 $R vllm $PWD/models/qwen3vl_2b_mlx4sim python eval/device_anchor82/writer_vllm.py \
        --model=$PWD/models/qwen3vl_2b_mlx4sim --tag=base --manifest=$MAN --out=$O/vllm_$SET ;;
  mlx) SPECS="$A" bash scripts/race_sbatch.sh $J 02:00:00 bash eval/device_anchor82/writer_job.sh 0 1 $O/writer_$SET $MAN ;;
  score) SPECS="$A" bash scripts/race_sbatch.sh $J 01:00:00 $R fp - envs/fp/bin/python eval/device_anchor82/score.py \
        --manifest=$SET --tag=$SET $SCORE_ARGS ;;
esac
echo "DA82_SUBMITTED $J $(date)"
