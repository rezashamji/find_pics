"""RESULTS 57: code fixes for the systematic conversion errors found in the B3 SFT targets BEFORE training (eye check of
25 random targets + every target of the B1 error kinds; inspect_targets.py). Each fix edits the teacher's raw JSON (the
training target) and is logged; rows are never invented.
  F1 UNSTATED EXCLUSION: an exclude_question the conversation never asks for (no negation / "alone" cue in the words).
     Cause: the check demands exclusion parity with the key, and the g truth programs make two albums mutually exclusive
     by construction (RESULTS 53 rule "no overlap"; "dog photos versus cat photos" -> the cat album excludes dogs).
     -> exclude_question = null. Only where no later turn of the conversation was trained on this plan.
  F2 ANCHOR THE KEY DOES NOT HAVE, next to the album's own question (turn 1): "the waterfall shots from the jungle trip"
     -> anchor "Is this a photo from a jungle trip?" + same_event + "Is there a waterfall?". The B1 "trip as a window"
     kind; sft_from_truth.check lets it pass whenever the key has days or a place. v2 rule: an anchor only when the key
     has one or the words ask for everything around a moment. Dev evidence (shipped adapter, m54 dev S1): such plans in
     a/b/f score .00-.67 (8 requests). -> anchor = window = null; the album's own question stays.
  F3 MOMENT RESTATED: the key wants EVERYTHING in a window (anchor or days, no need / need_any / person_need / not) but the
     target also asks a judge_question ("photos from the same night as the curling match" -> "Is this a photo from a
     curling match?"). B2 fixed 5 such targets by hand; RESULTS 51's ground() rule D catches only some phrasings.
     -> judge_question = null, looks = [].
  DROP: a target whose notes mention the answer key.
Usage: python eval/complex_bench/b57/fix_targets.py <dir with rows.jsonl + checked.json>  -> <dir>/rows_fixed.jsonl, fixes.log
"""
import json
import re
import sys
from pathlib import Path

NEG = re.compile(r"(?i)\b(not|no|without|except|exclude|excluding|minus|remove|take out|skip|leave out|other than|none|"
                 r"get rid|drop|hide|ditch|alone|by (him|her|them)sel(f|ves)|only (him|her|them))\b|n't\b")
EVERY = re.compile(r"(?i)\b(everything|all (of )?(the |my |our )?(photos|pictures|pics|shots)|what else|anything else|"
                   r"(the )?other (photos|pictures|pics|shots))\b")


def needs(k):
    return any(k.get(f) for f in ("need", "need_any", "person_need", "not"))


def main():
    d = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(d / "rows.jsonl")]
    C = {(c["qid"], c["turn"]): c for c in json.load(open(d / "checked.json"))}
    last = {}
    for r in rows:
        last[r["qid"]] = max(last.get(r["qid"], 0), r["turn"])
    out, log, n = [], [], {"F1": 0, "F2": 0, "F3": 0, "drop": 0}
    for r in rows:
        c = C[(r["qid"], r["turn"])]
        if re.search(r"(?i)answer key", r["output"]):
            n["drop"] += 1
            log.append(f"DROP {r['qid']} t{r['turn']}: notes mention the answer key")
            continue
        P = json.loads(re.search(r"\{.*\}", r["output"], re.S).group(0))
        K = [k for k in c["key"] if isinstance(k, dict)]
        said = " ".join((r.get("history") or []) + [r["request"]])
        changed = []
        for i, a in enumerate(P["albums"]):
            k = K[i] if len(K) == len(P["albums"]) else (K[0] if len(K) == 1 else {})
            if a.get("exclude_question") and not NEG.search(said) and r["turn"] == last[r["qid"]]:
                changed.append(f"F1 album {i}: exclude {a['exclude_question']!r} dropped")
                a["exclude_question"] = None
                n["F1"] += 1
            key_anchor = isinstance(k.get("anchor"), dict)
            if (r["turn"] == 1 and a.get("anchor") and a.get("judge_question") and not key_anchor and needs(k)
                    and not EVERY.search(said) and a.get("window") in ("same_event", "same_day")):
                changed.append(f"F2 album {i}: anchor {a['anchor'].get('judge_question')!r} ({a['window']}) dropped")
                a["anchor"], a["window"] = None, None
                n["F2"] += 1
            if a.get("anchor") and a.get("judge_question") and not needs(k) and (key_anchor or k.get("days") or
                                                                              k.get("date_from")):
                changed.append(f"F3 album {i}: restated question {a['judge_question']!r} dropped")
                a["judge_question"], a["looks"] = None, []
                n["F3"] += 1
        if changed:
            log.append(f"{r['qid']} t{r['turn']} [{r['cat']}] {r['request'][:100]}")
            log += [f"    {x}" for x in changed]
            r = dict(r, output=json.dumps(P, indent=2, ensure_ascii=False), fixed=changed)
        out.append(r)
    # every fixed target must still be a valid plan that the product grounds
    from datetime import date
    from findpics.converse import Plan, ground
    for r in out:
        if r.get("fixed"):
            P = Plan.model_validate(json.loads(r["output"]))
            today = date.fromisoformat(re.search(r"Today's date is (\d{4}-\d\d-\d\d)", r["prompt"]).group(1))
            ground(P, r["request"], r.get("history") or [], today, "me", [])
    with open(d / "rows_fixed.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary = f"rows {len(rows)} -> {len(out)}; fixes {n}; rows changed {sum(1 for r in out if r.get('fixed'))}"
    (d / "fixes.log").write_text(summary + "\n" + "\n".join(log) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
