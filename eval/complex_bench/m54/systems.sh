# RESULTS 54 (milestone on the complex-bench TEST split, run once): the two systems. Sourced by plan_job.sh / search_job.sh.
#   s0 "phone on 10-12 morning": BASE Qwen3.5-4B on the phone's 4-bit weights (models/qwen35_4b_mlx4sim), product code
#      a95b214 = origin/main at 10-12 14:00 (src/ identical; before track A a67b1f7; old person rule), plus ONE patch: the
#      RESULTS 49b parity fix (null field = default, as Plan.swift always decoded it; .cache/m54/parity_49b.patch).
#   s1 "shipped tonight": the 10-05 planner adapter merged into the same 4-bit weights (models/planner_4b27ball_q4merged),
#      product code 64709ef (track A + RESULTS 49c person rule + RESULTS 51 fixes).
# Code = sparse git worktrees .cache/wt_m54_s0 (patched) and .cache/wt_m54_s1. Judge = Qwen3.5-9B for both (shared jcache).
model_of() { case $1 in s0) echo $FP_ROOT/models/qwen35_4b_mlx4sim;; s1) echo $FP_ROOT/models/planner_4b27ball_q4merged;; esac; }
src_of() { echo $FP_ROOT/.cache/wt_m54_$1/src; }
OUT=eval/complex_bench/runs/m54
