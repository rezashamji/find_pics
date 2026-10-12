"""Planner SFT data from the complex benchmark's TRAIN split truth programs (never dev/test).

A truth program says which photos are right, but it was written by a model that saw the library (exact days, full
address names, anchor photo ids): copying it would teach the planner to invent dates and places. So the target is
written by the teacher (Qwen3.5-27B-FP8, the same teacher as the distillation data) from the PLANNER'S OWN PROMPT
(converse.build_prompt: the owner's words, today, people, the current plan for follow-ups) plus the truth program as a
hidden answer key, with rules for what may be taken from the key (structure, question wording) and what may not
(dates, days and places the words do not state). The training row is the planner prompt WITHOUT the key -> the
teacher's JSON.
Each target is then checked in code against the key after grounding (converse.ground + follow_up_edits, exactly what
the product applies): album count, person, anchor + window, exclusion, OR for need_any, dates that contain the key's
dates (or none when the words state no time), place words. Rows failing a check are dropped (and listed); a follow-up
conversation stops at its first failing turn (later turns would see a wrong current plan).
Usage (vLLM env, GPU with FP8):
  python eval/complex_bench/sft_from_truth.py --queries=eval/complex_bench/queries/train.json --out=data/public/distill/cb_train
  -> <out>/rows.jsonl (training rows, distillation format), <out>/checked.json (every turn, with the check result)
  --rules=v2: the revised target rules (KEY_V2) after v1's dev result.
"""
import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT  # noqa: E402
from plan_system import people_of  # noqa: E402

WINDOW = {"before_in_event": "before", "after_in_event": "after"}
TIME_WORDS = re.compile(r"(?i)\b((19|20)\d\d|january|february|march|april|june|july|august|september|october|november|"
                        r"december|spring|summer|autumn|winter|christmas|thanksgiving|halloween|easter|yesterday|ago|"
                        r"(last|this|past) (week|weekend|month|year|summer|winter|spring|fall|autumn))\b")

KEY = """
ANSWER KEY for THIS message (hidden from the person; written by someone who has seen the library). It defines which
photos are correct:
{key}
Key semantics: "need" = every question must be yes; "need_any" = at least one yes; "not" = every question must be no;
"person_need" = questions about the named person; "date_from"/"date_to" (end exclusive), "days", "hours", "place" =
photo metadata; "anchor" = a moment found by its "need" questions, "window" = how far around that moment to look.
Write the plan a careful planner would write from the conversation alone, with the structure the key has:
- judge_question = the key's first "need" question (for a person album: the "person_need" question, asked about "the
  person in the red box"); a second "need" -> filter_question; "need_any" -> ONE question that joins the alternatives
  with "or" ("Is there a statue or a church in this photo?"); "not" -> exclude_question (alternatives joined with "or").
  Use the key's wording; looks = short visual phrases for the same things. Ask nothing the conversation does not ask.
- "anchor" in the key -> "anchor" (judge_question = the key anchor's question) and the same "window"
  ("before_in_event" -> "before", "after_in_event" -> "after").
- Dates, days, hours and places in the key that the conversation does NOT state: leave them out (the planner cannot
  know them). When the conversation names an event or moment instead ("photos from the wedding", "the day we hiked in
  Denali", "from that night"), make it the "anchor" (what photos of that moment show) with window "same_event" ("the
  day ..."/"that night" -> "same_day"), and no judge_question unless something inside the moment is asked for.
- "time_phrase" = the exact words of the conversation that state the time; dates computed from them and today.
- "place" = the place words exactly as the conversation says them (key "Monterey County", words "Monterey" -> "Monterey").
- Follow-ups: "add/also/too/include ..." makes the new thing an ALTERNATIVE ("or") to the latest condition, never one
  more condition; "actually show me all ..." removes earlier narrowing; "not X but Y" swaps X for Y.
- Never mention the key in "notes".
"""


# v2 (10-12, after v1's dev result, aggregate plan shapes only): v1 targets copied the key's second "need" into
# filter_question even when the words state ONE thing (filter rate on train targets b .39, c .56, f .87 vs .19 in the old
# distillation data; dev plans b .03 -> .44) and made every named event an anchor (d .44): dev F1 a -.050, d -.057 vs
# the old adapter. v2: one question for one described thing; filter only for a separately stated condition; an event
# named only by the key's dates is asked about, an anchor only when the key has one or the words ask for everything
# around a moment.
KEY_V2 = KEY.replace("""- judge_question = the key's first "need" question (for a person album: the "person_need" question, asked about "the
  person in the red box"); a second "need" -> filter_question; "need_any" -> ONE question that joins the alternatives
  with "or" ("Is there a statue or a church in this photo?"); "not" -> exclude_question (alternatives joined with "or").
  Use the key's wording; looks = short visual phrases for the same things. Ask nothing the conversation does not ask.""",
"""- judge_question = ONE question that asks for everything the conversation describes about the photo (for a person
  album: about "the person in the red box"). When the key splits ONE described thing into several "need" questions
  ("a person wearing a large headdress" + "the headdress is made of feathers"), write one question ("Is there a person
  wearing a large feather headdress?"). filter_question ONLY for a second condition the conversation states separately
  ("only the ones where ...", a follow-up that narrows). "need_any" -> ONE question that joins the alternatives with
  "or" ("Is there a statue or a church in this photo?"); "not" -> exclude_question (alternatives joined with "or").
  Use the key's wording where it fits the words; looks = short visual phrases. Ask nothing the conversation does not ask.""").replace(
"""- Dates, days, hours and places in the key that the conversation does NOT state: leave them out (the planner cannot
  know them). When the conversation names an event or moment instead ("photos from the wedding", "the day we hiked in
  Denali", "from that night"), make it the "anchor" (what photos of that moment show) with window "same_event" ("the
  day ..."/"that night" -> "same_day"), and no judge_question unless something inside the moment is asked for.""",
"""- Dates, days, hours and places in the key that the conversation does NOT state: leave them out (the planner cannot
  know them). An event the key gives only as dates ("photos from the air show", "the concert in Nottingham") is asked
  about directly (judge_question "Is this a photo of an air show?"), no anchor. Use an "anchor" only when the key has
  one, or when the words ask for everything AROUND a moment ("everything from the day of the wedding", "all the photos
  from that night") -> anchor = what photos of that moment show, window "same_day" (or "same_event" for a trip).""")
assert KEY_V2 != KEY and KEY_V2.count("ONE question that asks") == 1 and "no anchor" in KEY_V2


def key_text(albums):
    out = []
    for a in albums or []:
        if not isinstance(a, dict):
            continue
        k = {f: a[f] for f in ("name", "person", "need", "need_any", "not", "person_need", "date_from", "date_to", "days",
                               "hours", "place") if a.get(f) not in (None, [], "")}
        anc = a.get("anchor")
        if isinstance(anc, dict) and anc.get("window"):
            k["anchor"] = {"need": anc.get("need") or [], "window": anc["window"]}
        out.append(k)
    return json.dumps(out, indent=1)


RULES = KEY_V2 if "--rules=v2" in sys.argv else KEY


def with_key(prompt, albums):
    i = prompt.rfind("JSON:")
    return prompt[:i] + RULES.format(key=key_text(albums)) + prompt[i:]


def _words(s):
    return set(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def check(G, albums, said, today):
    """Grounded plan vs the key -> list of problems ([] = pass)."""
    K = [a for a in albums or [] if isinstance(a, dict)]
    probs = []
    if len(G.albums) != len(K):
        return [f"{len(G.albums)} albums vs key {len(K)}"]
    for g, k in zip(G.albums, K):
        if (g.person or None) != (k.get("person") or None):
            probs.append(f"person {g.person} vs {k.get('person')}")
        anc = k.get("anchor") if isinstance(k.get("anchor"), dict) else None
        if anc and anc.get("window"):
            want = WINDOW.get(anc["window"], anc["window"])
            if not g.anchor:
                probs.append("key anchor missing")
            elif g.window != want:
                probs.append(f"window {g.window} vs {want}")
        elif g.anchor and not (k.get("days") or k.get("date_from") or k.get("date_to") or k.get("place")):
            probs.append("anchor the key does not have")
        if bool(k.get("not")) != bool(g.exclude_question):
            probs.append(f"exclusion {g.exclude_question!r} vs key {k.get('not')}")
        if len(k.get("need_any") or []) >= 2 and not any(re.search(r"\bor\b", q or "") for q in
                                                         (g.judge_question, g.filter_question)):
            probs.append("need_any not asked as one 'or' question")
        kd = sorted(k.get("days") or []) or ([k["date_from"]] if k.get("date_from") else [])
        ke = (date.fromisoformat(max(k["days"])) + timedelta(days=1)).isoformat() if k.get("days") else k.get("date_to")
        if g.date_from or g.date_to:
            if not (kd or ke):
                probs.append(f"dates {g.date_from}..{g.date_to} the key does not have")
            elif (g.date_from and kd and g.date_from > kd[0]) or (g.date_to and ke and g.date_to < ke):
                probs.append(f"dates {g.date_from}..{g.date_to} do not contain the key's {kd[:1]}..{ke}")
        if (kd or ke) and not (g.date_from or g.date_to) and TIME_WORDS.search(said):
            probs.append("the words state a time but the plan has no dates")
        if g.place:
            kp = " ".join(k["place"]) if isinstance(k.get("place"), list) else (k.get("place") or "")
            if not _words(g.place) <= _words(kp) | {"city", "county", "state", "region"}:
                probs.append(f"place {g.place!r} vs key {kp!r}")
            if not _words(g.place) <= _words(said):
                probs.append(f"place {g.place!r} not in the words")
    return probs


def main():
    import os
    from findpics.converse import Plan, _unanswerable, build_prompt, follow_up_edits, ground, strip_identity_conditions
    from findpics.vlm import VLLMJudge
    arg = lambda k, d=None: next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{k}=")), d)
    qf, out = Path(arg("queries")), Path(arg("out"))
    Q = json.loads(qf.read_text())["queries"]
    assert all(q["split"] == "train" for q in Q), "TRAIN split only"
    out.mkdir(parents=True, exist_ok=True)
    J = VLLMJudge(model=arg("model", "Qwen/Qwen3.5-27B-FP8"), gpu_mem=0.9)
    st = {q["qid"]: dict(history=[], P=None, alive=True, people=people_of(q["user"]), today=date.fromisoformat(q["today"]))
          for q in Q}
    rows, checked = [], []
    t0 = time.time()
    for t in range(max(len(q["turns"]) for q in Q)):
        alive = [q for q in Q if len(q["turns"]) > t and st[q["qid"]]["alive"]]
        progs = {q["qid"]: (q["program"][t] if q["cat"] == "h" else q["program"]) for q in alive}
        base = {q["qid"]: build_prompt(q["turns"][t], st[q["qid"]]["history"], st[q["qid"]]["P"], "me", st[q["qid"]]["people"],
                                       st[q["qid"]]["today"]) for q in alive}
        pending = {qid: with_key(p, progs[qid]) for qid, p in base.items()}
        got = {}
        for attempt in range(3):
            if not pending:
                break
            outs = J._chat([[{"role": "user", "content": p}] for p in pending.values()], J.SP(temperature=0.0, max_tokens=1024))
            nxt = {}
            for (qid, p), o in zip(pending.items(), outs):
                txt = o.outputs[0].text
                q = next(x for x in alive if x["qid"] == qid)
                s = st[qid]
                said = " \n ".join(s["history"] + [q["turns"][t]])
                try:
                    m = re.search(r"\{.*\}", txt, re.S)
                    raw = json.loads(m.group(0))
                    P = Plan.model_validate(raw)
                    if not P.albums:
                        raise ValueError("zero albums")
                    strip_identity_conditions(P, "me")
                    prob = _unanswerable(P, list(s["people"]) + ["me"], said)
                    if prob:
                        raise ValueError(prob)
                    got[qid] = (txt, P)
                except Exception as e:
                    nxt[qid] = base[qid] + f"\n(Previous output was invalid: {str(e)[:300]}. Return valid JSON only.)\nJSON:"
                    nxt[qid] = with_key(nxt[qid], progs[qid])
            pending = nxt
        for q in alive:
            qid, s, msg = q["qid"], st[q["qid"]], q["turns"][t]
            if qid not in got:
                s["alive"] = False
                checked.append(dict(qid=qid, cat=q["cat"], turn=t + 1, message=msg, ok=False, problems=["no valid output"]))
                continue
            txt, P = got[qid]
            G = follow_up_edits(ground(Plan.model_validate(json.loads(P.model_dump_json())), msg, s["history"], s["today"],
                                       "me", s["people"]), s["P"], msg)
            for a in G.albums:
                if a.person and a.person not in s["people"]:
                    a.person = None
                a.with_people = [w for w in a.with_people if w in s["people"]]
            said = " ".join(s["history"] + [msg])
            probs = check(G, progs[qid], said, s["today"])
            checked.append(dict(qid=qid, cat=q["cat"], turn=t + 1, message=msg, history=list(s["history"]), ok=not probs,
                                problems=probs, key=progs[qid], output=txt, grounded=json.loads(G.model_dump_json())))
            if probs:
                s["alive"] = False
                continue
            rows.append(dict(split="train", source="complex_bench_train", qid=qid, cat=q["cat"], turn=t + 1, request=msg,
                             history=list(s["history"]), prompt=base[qid], output=txt))
            s["history"].append(msg)
            s["P"] = G
        print(f"turn {t + 1}: {len(alive)} prompts, {sum(1 for c in checked if c['turn'] == t + 1 and c['ok'])} pass, "
              f"{time.time() - t0:.0f}s", flush=True)
    with open(out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out / "checked.json").write_text(json.dumps(checked, indent=1))
    from collections import Counter
    print("SFT_DONE rows", len(rows), "of", len(checked), "turns;", dict(Counter((c["cat"], c["ok"]) for c in checked)), flush=True)


if __name__ == "__main__":
    main()
