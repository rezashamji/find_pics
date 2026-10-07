"""A request that names ONE specific pet or thing ("my dog Max at the beach", "my blue car", "Luna the cat"): the plan
alone would search the category ("a dog"), which answers a different question. These rules decide when to ask for
1-3 example photos instead (the way an unknown person gets the face sheet), and rewrite the album so the subject
search (converse.SubjectRefs: image-vector ranking + side-by-side judge) runs with the rest of the plan as its scope.

Pure text rules, no model. The phone runs the identical port (ios/FindPicsCore/Sources/FindPicsCore/Subjects.swift),
checked against this file on eval/subjects_fixtures.py's cases."""
from __future__ import annotations

import re
from dataclasses import dataclass

PETS = ["dog", "puppy", "pup", "cat", "kitten", "kitty", "horse", "pony", "bird", "parrot", "budgie", "rabbit", "bunny",
        "hamster", "guinea pig", "ferret", "goldfish", "turtle", "tortoise", "lizard", "gecko", "snake", "hen", "goat"]
THINGS = ["car", "truck", "van", "jeep", "bike", "bicycle", "motorcycle", "motorbike", "scooter", "boat", "kayak", "canoe",
          "guitar", "piano", "violin", "ukulele", "backpack", "bag", "purse", "handbag", "suitcase", "watch", "ring",
          "necklace", "bracelet", "shoes", "sneakers", "boots", "jacket", "coat", "hat", "cap", "dress", "sweater", "scarf",
          "glasses", "sunglasses", "teddy bear", "teddy", "stuffed animal", "plush", "toy", "doll", "mug", "couch", "sofa",
          "chair", "desk", "tent", "skateboard", "surfboard", "snowboard", "stroller", "plant", "camper"]
KINDS = set(PETS) | set(THINGS)

# words that end the "my <modifiers> <kind>" phrase, and that are never a name
STOP = set("""photo photos picture pictures pic pics image images video videos clip clips shot shots selfie selfies of with
and or the a an at in on from to for by near under over is are was were be been this that these those all every some
any me my our your his her their its i we you he she they it them him not no without but only just also when where
while who which what how why as if than then so very s one ones""".split())
NEG = {"not", "without", "except", "excluding", "exclude", "drop", "remove", "minus", "nor"}
# "selfies in my car", "the dog on my couch": a place the photo was taken, not the subject
WHERE = {"in", "inside", "into", "on", "onto", "from", "under", "off", "out"}
# "my car keys", "my dog's bowl" (the token "dog's" is not a kind): the kind is only a modifier of another noun
COMPOUND_AFTER = set("""keys key seat seats ride rides trip trips wash show shows accident food bowl toys bed park parking
leash collar house door window case strap band charger cover lesson lessons class classes walker sitter groomer vet race
racing dealer repair insurance lot garage rack tire tires wheel engine hair fur treats treat cage tank stand string strings
shop store box""".split())
NOT_NAMES = set("""january february march april may june july august september october november december monday tuesday
wednesday thursday friday saturday sunday christmas xmas halloween easter thanksgiving""".split())
# a condition question that only restates the subject ("Is there a real dog anywhere in this photo (not a drawing...)?")
TRIVIAL = set("""is are there this that a an the any some real photo photos image images picture pictures video videos clip
frame of in visible anywhere not drawing painting statue toy model or one does do show shown showing it its specific named
called with contain contains containing s my your our person""".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()


@dataclass
class SubjectAsk:
    albums: list[int]
    name: str | None          # "Max" when the request names it
    kind: str                 # "dog", "car"
    words: str                # modifiers + kind as said, lowercased: "blue car", "sister's dog", "dog"
    key: str = ""             # where its example photos are remembered: "name:max" / "my:blue car"
    display: str = ""         # "Max" / "your blue car"
    prompt: str = ""          # "Show me Max: pick 1-3 clear photos of Max."
    judge_name: str = ""      # for the side-by-side question: "Max" / "this car"
    from_saved: bool = False  # recognised from a name whose photos were picked earlier

    def finish(self):
        self.key = f"name:{_norm(self.name)}" if self.name else f"my:{_norm(self.words)}"
        self.display = self.name or f"your {self.words}"
        self.prompt = (f"Show me {self.name}: pick 1-3 clear photos of {self.name}." if self.name else
                       f"Show me {self.display}: pick 1-3 clear photos of it.")
        self.judge_name = self.name or f"this {self.kind}"
        return self


def _sentence_start(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?\n"


def _capitalized(tok: str) -> bool:
    return tok[:1].isupper() and tok[:1].isalpha()


def _name_ok(tok: str) -> bool:
    lo = tok.lower()
    return tok.isalpha() and len(tok) >= 2 and lo not in STOP and lo not in KINDS and lo not in NOT_NAMES


def _albums_for(P, kind: str, name: str | None) -> list[int]:
    if len(P.albums) == 1:
        return [0]
    out = []
    for k, a in enumerate(P.albums):
        if name and _norm(a.person or "") == _norm(name):
            out.append(k); continue
        words = set(_norm(" ".join([a.name, *a.looks, a.judge_question or "", a.filter_question or ""])).split())
        kt = _norm(kind).split()
        if all(t in words or t + "s" in words for t in kt) or (name and _norm(name) in words):
            out.append(k)
    return out


def _scan(text: str, plan_names: set[str]) -> list[tuple[str | None, str, str]]:
    """(name, kind, words) for every 'my/our <0-2 modifiers> <kind> [Name | named X]', '<kind> named X', 'X the <kind>'."""
    text = text.replace("’", "'")
    ms = list(_TOKEN.finditer(text))
    tok = [m.group(0) for m in ms]
    lo = [t.lower() for t in tok]
    n = len(tok)

    def kind_at(j):
        if j + 1 < n and f"{lo[j]} {lo[j + 1]}" in KINDS:
            return f"{lo[j]} {lo[j + 1]}", j + 2
        if lo[j] in KINDS:
            return lo[j], j + 1
        return None, j

    def name_after(end):
        if end >= n:
            return None
        if lo[end] in ("named", "called") and end + 1 < n and _name_ok(tok[end + 1]):
            return tok[end + 1][:1].upper() + tok[end + 1][1:]
        if _name_ok(tok[end]) and (_capitalized(tok[end]) or lo[end] in plan_names):
            return tok[end][:1].upper() + tok[end][1:]
        return None

    out = []
    for i in range(n):
        if lo[i] in ("my", "our"):
            if any(lo[j] in NEG for j in range(max(0, i - 5), i)) or (i >= 1 and lo[i - 1] in WHERE):
                continue
            j, mods, kind, end = i + 1, [], None, i + 1
            while j < n:
                kind, end = kind_at(j)
                if kind or lo[j] in STOP or len(mods) == 2:
                    break
                mods.append(lo[j]); j += 1
            if not kind or (end < n and lo[end] in COMPOUND_AFTER):
                continue
            out.append((name_after(end), kind, " ".join(mods + [kind])))
            continue
        kind, end = kind_at(i)
        if kind and i >= 1 and lo[i - 1] in ("the", "a", "an") and end + 1 < n and lo[end] in ("named", "called") \
                and _name_ok(tok[end + 1]) and not any(lo[j] in NEG for j in range(max(0, i - 4), i)) \
                and not (i >= 2 and lo[i - 2] in ("my", "our")):
            out.append((tok[end + 1][:1].upper() + tok[end + 1][1:], kind, kind))
        if kind and i >= 2 and lo[i - 1] == "the" and _capitalized(tok[i - 2]) and _name_ok(tok[i - 2]) \
                and not _sentence_start(text, ms[i - 2].start()) and not (end < n and lo[end] in COMPOUND_AFTER) \
                and not any(lo[j] in NEG for j in range(max(0, i - 5), i - 2)):
            out.append((tok[i - 2], kind, kind))
    return out


def named_subjects(P, message: str, history: list[str] | None = None, saved: list[dict] | None = None) -> list[SubjectAsk]:
    """Which albums are about ONE specific pet / thing, and what to ask for. `saved`: subjects whose photos were picked
    before, [{"name": "Max" | None, "kind": "dog", "words": "dog"}] (a saved name is recognised without 'my dog')."""
    msgs = list(history or []) + [message]
    plan_names = {_norm(x) for a in P.albums for x in [a.person or "", *a.with_people] if _norm(x)} | \
                 {_norm(x) for x in P.unknown_people if _norm(x)}
    found: dict[str, SubjectAsk] = {}
    for m in msgs:
        for name, kind, words in _scan(m, plan_names):
            ask = SubjectAsk([], name, kind, words).finish()
            ask.albums = _albums_for(P, kind, name)
            if ask.albums and ask.key not in found:
                found[ask.key] = ask
    for s in saved or []:     # "Max at the beach" once Max's photos were picked: the name alone is enough
        nm = s.get("name")
        if not nm:
            continue
        said_it = any(t == nm for m in msgs for t in _TOKEN.findall(m.replace("’", "'")))
        albums = [k for k, a in enumerate(P.albums) if _norm(a.person or "") == _norm(nm)]
        if not albums and said_it:
            albums = _albums_for(P, s["kind"], nm)
        ask = SubjectAsk(albums, nm, s["kind"], s.get("words") or s["kind"], from_saved=True).finish()
        if albums and ask.key not in found:
            found[ask.key] = ask
    return list(found.values())


def resolve_subject(ask: SubjectAsk, saved: list[dict]) -> int | None:
    """Index of the saved subject this ask means, or None (then ask for photos). Exact key first; 'my dog' with no
    modifiers also means the one saved dog when there is exactly one."""
    keys = [(f"name:{_norm(s['name'])}" if s.get("name") else f"my:{_norm(s.get('words') or s['kind'])}") for s in saved]
    if ask.key in keys:
        return keys.index(ask.key)
    if not ask.name and ask.words == ask.kind:
        same = [i for i, s in enumerate(saved) if s["kind"] == ask.kind]
        if len(same) == 1:
            return same[0]
    return None


def _rename(q: str | None, ask: SubjectAsk) -> str | None:
    if not q:
        return q
    the = f"the {ask.kind}"
    q = re.sub(r"(?i)\bthe person in the red box\b", the, q)
    if ask.name:
        nm = re.escape(ask.name)
        q = re.sub(r"(?i)\b(?:an?|the|my|our) (?:\w+ )?" + re.escape(ask.kind) + r" (?:named|called) " + nm + r"\b", the, q)
        q = re.sub(r"(?i)\b" + nm + r"\b", the, q)
        q = re.sub(r"(?i)\bthe the\b", "the", q)
    return q


def subject_condition(q: str | None, ask: SubjectAsk) -> str | None:
    """The album's own condition to ask about the subject's photos ("... at the beach"), or None when the question only
    restates the subject ("Is there a real dog anywhere in this photo?")."""
    q = _rename(q, ask)
    if not q:
        return None
    skip = TRIVIAL | set(_norm(ask.words).split()) | {t + "s" for t in _norm(ask.kind).split()}
    if ask.name:
        skip |= set(_norm(ask.name).split())
    return q if [t for t in _norm(q).split() if t not in skip] else None


def subject_plan(P, asks: list[SubjectAsk]):
    """The plan with each subject album rewritten for the subject search: the subject is not a person (any person on
    the album must ALSO be in the photo: with_people), and questions say 'the dog', never its name (the judge cannot
    know it)."""
    P = P.model_copy(deep=True)
    for ask in asks:
        nn = _norm(ask.name or "")
        for k in ask.albums:
            a = P.albums[k]
            if a.person and _norm(a.person) != nn and _norm(a.person) not in (_norm(ask.words), _norm("my " + ask.words)):
                a.with_people = [a.person] + [w for w in a.with_people if _norm(w) != _norm(a.person)]
            a.person = None
            a.with_people = [w for w in a.with_people if _norm(w) != nn or not nn]
            a.judge_question = subject_condition(a.judge_question, ask)
            a.filter_question = subject_condition(a.filter_question, ask)
            a.exclude_question = _rename(a.exclude_question, ask)
        drop = {nn, _norm(ask.words), _norm("my " + ask.words), _norm("our " + ask.words)} - {""}
        P.unknown_people = [u for u in P.unknown_people if _norm(u) not in drop]
    return P
