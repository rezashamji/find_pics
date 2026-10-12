"""Complex-query benchmark: the PLAN stage alone (text only, no photos), for comparing planners and grounding code.

Same chat path as run_system.py: plan_turn on the owner's words, follow-ups get the history + the current (grounded)
plan, people = the benchmark's named face groups (Sam, Alex) when the library has them, a person not in that list is
dropped after each turn ("me"/unknown: Flickr libraries carry no identity tags). One vLLM batch per turn index; a retry
(plan_turn's invalid-output path) goes one prompt at a time. run_system.py --plans=<file> then runs the 9B judge on
these plans, so every planner meets the same search stage and plan changes can be counted directly.
Usage (vLLM env, GPU): python eval/complex_bench/plan_system.py --split=dev --out=<dir> [--model=<planner dir or HF id>]
        [--reuse=<raw.json of an earlier run with the SAME model>]
  -> <dir>/plans.json   ({qid: {"turns": [{"message", "plan"}], "error"}}) and <dir>/raw.json (prompt -> model output)
--reuse: identical prompts get the earlier run's raw output (vLLM output depends on batch composition and GPU type:
62/350 final 9B plans differed between two runs of the same code on 10-12), so two code versions are compared on the
same model answers wherever the prompt did not change.
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402

NAMES = ["Sam", "Alex"]


def people_of(user):
    lib = json.loads((ROOT / "libraries" / f"{user}.json").read_text())
    G = [g for g in lib.get("face_groups", []) if g["n_items"] >= 25][:2]
    return NAMES[:len(G)]


def main():
    import os
    from findpics.converse import build_prompt, plan_turn
    from findpics.vlm import DEFAULT_VLM, VLLMJudge
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    split, out = arg("split"), Path(arg("out"))
    model = arg("model", os.environ.get("FP_PLANNER_MODEL") or DEFAULT_VLM)
    Q = json.loads((ROOT / "queries" / f"{split}.json").read_text())["queries"]
    out.mkdir(parents=True, exist_ok=True)
    J = VLLMJudge(model=model, gpu_mem=0.85)
    cache = json.loads(Path(arg("reuse")).read_text()) if arg("reuse") else {}
    n_reused = len(cache)

    def batch(prompts):
        todo = [p for p in dict.fromkeys(prompts) if p not in cache]
        if todo:
            outs = J._chat([[{"role": "user", "content": p}] for p in todo], J.SP(temperature=0.0, max_tokens=1024))
            cache.update({p: o.outputs[0].text for p, o in zip(todo, outs)})

    def rec(prompt, max_tokens=1024):
        o = cache.get(prompt)
        if o is None:
            o = cache[prompt] = J.text(prompt, max_tokens=max_tokens)
        return o

    st = {q["qid"]: dict(history=[], P=None, turns=[], error=None, people=people_of(q["user"]),
                         today=date.fromisoformat(q["today"])) for q in Q}
    t0 = time.time()
    for t in range(max(len(q["turns"]) for q in Q)):
        alive = [q for q in Q if len(q["turns"]) > t and st[q["qid"]]["error"] is None]
        batch([build_prompt(q["turns"][t], st[q["qid"]]["history"], st[q["qid"]]["P"], "me", st[q["qid"]]["people"],
                            st[q["qid"]]["today"]) for q in alive])
        for q in alive:
            s, msg = st[q["qid"]], q["turns"][t]
            try:
                P = plan_turn(msg, rec, history=s["history"], current=s["P"], people=s["people"], today=s["today"])
                s["history"].append(msg)
                for a in P.albums:
                    if a.person and a.person not in s["people"]:
                        a.person = None
                    a.with_people = [w for w in a.with_people if w in s["people"]]
                s["P"] = P
                s["turns"].append(dict(message=msg, plan=json.loads(P.model_dump_json())))
            except Exception as e:
                s["error"] = f"{type(e).__name__}: {e}"[:400]
        print(f"turn {t + 1}: {len(alive)} planned, {time.time() - t0:.0f}s", flush=True)
    res = {qid: dict(turns=s["turns"], error=s["error"]) for qid, s in st.items()}
    (out / "plans.json").write_text(json.dumps(dict(model=str(model), split=split, plans=res)))
    (out / "raw.json").write_text(json.dumps(cache))
    print("raw outputs", len(cache), "reused from --reuse", n_reused, flush=True)
    print("PLANS_DONE", len(res), "errors", sum(bool(r["error"]) for r in res.values()), flush=True)


if __name__ == "__main__":
    main()
