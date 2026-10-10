"""Can the photo judge (Qwen3-VL-4B-Instruct, phone weights models/q3vl4b_real) also be the planner, so the phone holds
ONE model? Runs the 30 scripted conversations of eval/eval_planners.py (CONVS, same checks, same owner/people/today)
through converse.plan_turn exactly as the app's Planner.swift does (one user message = build_prompt, temperature 0,
max 1024 tokens, retry with the problem fed back), in two variants:
  raw      = converse.ground and keep_partial_undo replaced by identity (the model's own plan; retries still on)
  grounded = the app's path (ground + keep_partial_undo after every turn)
and two chat-template modes: default (no kwargs) and off (enable_thinking=False, what the app sends).
Also records per-call latency + token counts, and per-turn Swift-grounding cases (raw plan in, Python plan out) for
eval/planner_vl/swift_ground (the app's FindPicsCore port must give the same grounded plan).
Usage (vLLM env, GPU): python eval/eval_planner_vl.py <model_dir> <tag> <modes: off,default>
Output: eval/planner_vl/<tag>_<mode>_<variant>.json, <tag>_<mode>_swiftcases.json, printed summary.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "eval")
from eval_planners import CONVS, PEOPLE, T  # noqa: E402
from findpics import converse as C  # noqa: E402
from findpics.vlm import VLLMJudge  # noqa: E402

OUT = Path("eval/planner_vl")


def main():
    model, tag, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    J = VLLMJudge(model=model, gpu_mem=0.8)
    tok = J.llm.get_tokenizer()
    m = [{"role": "user", "content": "x"}]
    r_def = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    r_off = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"TEMPLATE {tag}: default == thinking-off render: {r_def == r_off}\n  default: {r_def!r}\n  off: {r_off!r}", flush=True)
    orig_ground, orig_kpu = C.ground, C.keep_partial_undo
    summary = {}
    for mode in modes:
        kw = {} if mode == "default" else {"chat_template_kwargs": {"enable_thinking": False}}
        calls = []

        def llm(prompt, max_tokens=1024):
            t0 = time.time()
            o = J.llm.chat([[{"role": "user", "content": prompt}]], J.SP(temperature=0.0, max_tokens=max_tokens),
                           use_tqdm=False, **kw)[0]
            calls.append(dict(sec=time.time() - t0, prompt_tokens=len(o.prompt_token_ids),
                              out_tokens=len(o.outputs[0].token_ids), text=o.outputs[0].text))
            return o.outputs[0].text

        for variant in ("raw", "grounded"):
            rec = {}
            if variant == "raw":
                C.ground = lambda P, *a, **k: P
                C.keep_partial_undo = lambda P, *a, **k: P
            else:
                def g(P, message, history, today=None, owner="me", people=None):
                    rec["raw"] = json.loads(P.model_dump_json())
                    return orig_ground(P, message, history, today, owner, people)
                C.ground, C.keep_partial_undo = g, orig_kpu
            convs, cases = [], []
            for msgs, check in CONVS:
                P, hist, err, turns = None, [], None, []
                try:
                    for msg in msgs:
                        n0 = len(calls)
                        cur = json.loads(P.model_dump_json()) if P else None
                        prompt = C.build_prompt(msg, hist, P, "Reza", PEOPLE, T)
                        t0 = time.time()
                        P = C.plan_turn(msg, llm, history=hist, current=P, owner="Reza", people=PEOPLE, today=T)
                        turns.append(dict(message=msg, sec=time.time() - t0, calls=calls[n0:],
                                          plan=json.loads(P.model_dump_json())))
                        if variant == "grounded":
                            cases.append(dict(conv=len(convs), message=msg, history=list(hist), owner="Reza",
                                              people=PEOPLE, today=T.isoformat(), prompt=prompt, current=cur,
                                              raw=rec.pop("raw"), grounded=json.loads(P.model_dump_json())))
                        hist.append(msg)
                    ok = bool(P.albums) and bool(check(P))
                except Exception as e:
                    ok, err = False, f"{type(e).__name__}: {e}"[:300]
                convs.append(dict(messages=msgs, ok=ok, error=err, turns=turns,
                                  plan=json.loads(P.model_dump_json()) if P else None))
                print(f"{tag} {mode} {variant}", "PASS" if ok else "FAIL", msgs, err or "", flush=True)
                if not ok and P is not None:
                    print("   final plan:", P.model_dump_json(exclude={"notes"}), flush=True)
            json.dump(convs, open(OUT / f"{tag}_{mode}_{variant}.json", "w"), indent=1)
            if variant == "grounded":
                json.dump(cases, open(OUT / f"{tag}_{mode}_swiftcases.json", "w"), indent=1)
            n = sum(c["ok"] for c in convs)
            tc = [c for cv in convs for t in cv["turns"] for c in t["calls"]]
            ts = [t["sec"] for cv in convs for t in cv["turns"]]
            s = dict(passed=n, total=len(convs), calls=len(tc), turns=len(ts),
                     retries=len(tc) - len(ts), turn_sec_median=statistics.median(ts), turn_sec_mean=statistics.mean(ts),
                     call_sec_median=statistics.median(c["sec"] for c in tc),
                     out_tokens_mean=statistics.mean(c["out_tokens"] for c in tc),
                     prompt_tokens_mean=statistics.mean(c["prompt_tokens"] for c in tc),
                     decode_tok_per_s=sum(c["out_tokens"] for c in tc) / sum(c["sec"] for c in tc))
            summary[f"{mode}_{variant}"] = s
            print(f"SUMMARY {tag} {mode} {variant}: {n}/{len(convs)} conversations pass; {json.dumps(s)}", flush=True)
    C.ground, C.keep_partial_undo = orig_ground, orig_kpu
    json.dump(summary, open(OUT / f"{tag}_summary.json", "w"), indent=1)


def swiftscore(cases_path, swift_path):
    """Score the conversations on the plans the app's Swift grounding (eval/planner_vl/swift_ground) produced."""
    cases, sw = json.load(open(cases_path)), json.load(open(swift_path))
    last = {}
    for c, s in zip(cases, sw):
        last[c["conv"]] = s                       # the final turn of each conversation
    ok = 0
    for i, (msgs, check) in enumerate(CONVS):
        P = C.Plan.model_validate_json(last[i]) if i in last else None
        r = bool(P and P.albums and check(P))
        ok += r
        print("SWIFT", "PASS" if r else "FAIL", msgs)
    print(f"SWIFT-GROUNDED {cases_path}: {ok}/{len(CONVS)} conversations pass")


if __name__ == "__main__":
    if sys.argv[1] == "swiftscore":
        swiftscore(sys.argv[2], sys.argv[3])
    else:
        main()
