"""Raw planner output on the conversations a distilled planner fails (is the miss in the model or in grounding?).
Usage (vLLM env, GPU): FP_VLM_MODEL=<model> python eval/debug_planner_fails.py"""
import sys
sys.path.insert(0, "eval")
import eval_planners as E
from findpics.converse import plan_turn
from findpics.vlm import VLLMJudge

FAILS = {("photos of Dad at the beach", "only the ones from 2019"), ("all my photos with bread", "drop the sandwiches"),
         ("photos from the day of my graduation",),
         ("all my photos with bread", "drop the sandwiches and burgers", "actually keep the sandwiches")}


def main():
    J = VLLMJudge(gpu_mem=0.8)
    for msgs, _ in E.CONVS:
        if tuple(msgs) not in FAILS:
            continue
        P, hist = None, []
        for m in msgs:
            raw = []
            P = plan_turn(m, lambda p, **k: raw.append(J.text(p, **k)) or raw[-1], history=hist, current=P, owner="Reza",
                          people=E.PEOPLE, today=E.T)
            hist.append(m)
            print("MSG", m, "\nRAW", raw[-1][:900], "\nGROUNDED", P.model_dump_json(exclude_defaults=True)[:900], "\n", flush=True)


if __name__ == "__main__":   # vLLM spawns workers that re-import this file
    main()
