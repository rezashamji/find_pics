"""Golden cases for the Swift port of findpics.subjects (when a request names ONE specific pet / thing, what to ask for,
and the rewritten album): every grounded plan of the 2,300 planner cases in Fixtures/ground.json (most must give no
ask) + hand-written cases (named pets, saved subjects, negations, compounds, several albums)."""
import json
import sys

sys.path.insert(0, "src")
from findpics.converse import Plan
from findpics.subjects import named_subjects, resolve_subject, subject_plan

FIX = "ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/"


def album(**kw):
    a = dict(name="x", person=None, looks=[], judge_question=None)
    a.update(kw)
    return a


HAND = [
    ("my dog Max at the beach", [album(name="Max at the beach", person="Max", looks=["a dog on a beach"],
                                       judge_question="Is Max on a beach?")], ["Max"], []),
    ("my dog max at the beach", [album(name="dog at beach", person="Max", looks=["a dog on a beach"],
                                       judge_question="Is there a dog on a beach in this photo?")], ["Max"], []),
    ("my blue car", [album(name="blue car", looks=["a blue car"], judge_question="Is there a blue car in this photo?")], [], []),
    ("photos of Luna the cat sleeping", [album(name="Luna sleeping", looks=["a sleeping cat"],
                                               judge_question="Is the cat sleeping?")], [], []),
    ("the cat named Luna on the sofa", [album(name="Luna", looks=["a cat on a sofa"],
                                              judge_question="Is a cat named Luna on a sofa?")], [], []),
    ("me and my dog Max hiking", [album(name="me and Max", person="me", looks=["hiking"],
                                        judge_question="Is the person in the red box hiking?", with_people=["Max"])],
     [], ["Max"]),
    ("my car keys", [album(name="keys", looks=["car keys"], judge_question="Are there car keys?")], [], []),
    ("my dog's bowl", [album(name="bowl", looks=["a dog bowl"], judge_question="Is there a dog bowl?")], [], []),
    ("photos without my dog", [album(name="no dog", looks=["photos"], exclude_question="Is there a dog?")], [], []),
    ("my dogs at the park", [album(name="dogs", looks=["dogs in a park"], judge_question="Are there dogs in a park?")], [], []),
    ("selfies in my car", [album(name="car selfies", person="me", looks=["a selfie in a car"], judge_question="Is this a selfie?")], [], []),
    ("my dog vs my cat", [album(name="dog", looks=["a dog"], judge_question="Is there a dog?"),
                          album(name="cat", looks=["a cat"], judge_question="Is there a cat?")], [], []),
    ("Max at the beach", [album(name="Max at the beach", person="Max", looks=["a beach"], judge_question="Is Max at the beach?")],
     [], [], [dict(name="Max", kind="dog", words="dog")]),
    ("Max at the beach", [album(name="Max at the beach", person="Max", looks=["a beach"], judge_question="Is this a beach?")],
     [], [], []),
    ("my sister's red bike in Paris", [album(name="bike", looks=["a red bike"], judge_question="Is there a red bike?",
                                             place="Paris")], [], []),
    ("our old teddy bear", [album(name="teddy", looks=["a teddy bear"], judge_question="Is there a teddy bear?")], [], []),
    ("my guitar on stage, not my piano", [album(name="guitar", looks=["a guitar on stage"],
                                                judge_question="Is there a guitar on a stage?")], [], []),
    ("photos of my wife", [album(name="wife", looks=[], judge_question=None)], [], ["my wife"]),
    ("Show my cat Mochi in the snow", [album(name="Mochi snow", looks=["a cat in snow"],
                                            judge_question="Is a cat in the snow?",
                                            filter_question="Is Mochi outdoors?")], [], []),
]


def ask_json(a):
    return dict(albums=a.albums, name=a.name, kind=a.kind, words=a.words, key=a.key, display=a.display, prompt=a.prompt,
                judgeName=a.judge_name, fromSaved=a.from_saved)


out = []
for c in json.load(open(FIX + "ground.json")):
    P = Plan.model_validate(c["grounded"])
    asks = named_subjects(P, c["message"], c["history"])
    out.append(dict(message=c["message"], history=c["history"], saved=[], plan=json.loads(P.model_dump_json()),
                    asks=[ask_json(a) for a in asks], planned=json.loads(subject_plan(P, asks).model_dump_json())))
for h in HAND:
    msg, albums, _, unknown = h[:4]
    saved = h[4] if len(h) > 4 else []
    P = Plan.model_validate(dict(albums=albums, notes="", unknown_people=unknown))
    asks = named_subjects(P, msg, [], saved)
    out.append(dict(message=msg, history=[], saved=saved, plan=json.loads(P.model_dump_json()),
                    asks=[ask_json(a) for a in asks], planned=json.loads(subject_plan(P, asks).model_dump_json())))
    print(f"{msg!r:45} -> {[(a.display, a.albums, a.prompt) for a in asks]}")
saved = [dict(name="Max", kind="dog", words="dog"), dict(name=None, kind="car", words="blue car"),
         dict(name=None, kind="cat", words="cat")]
res = []
for msg in ["my dog Max", "my blue car", "my car", "my cat", "my red car", "my dog"]:
    P = Plan.model_validate(dict(albums=[album(name="x")]))
    for a in named_subjects(P, msg, []):
        res.append(dict(message=msg, saved=saved, index=resolve_subject(a, saved)))
json.dump(dict(cases=out, resolve=res), open(FIX + "subjects.json", "w"))
print("SUBJECT_ASK_FIXTURES", len(out), "cases,", sum(1 for o in out if o["asks"]), "with an ask;", len(res), "resolve")
print("resolve:", [(r["message"], r["index"]) for r in res])
