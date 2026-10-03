"""Natural-language request -> structured plan, via a fixed prompt template with the request injected.

The LLM never sees photos here; it only rewrites the request into fields the engine executes exactly
(dates, media type, person) plus short visual descriptions that the image-text model and the judge use.
"""
from __future__ import annotations

import json
import re
from datetime import date

from pydantic import BaseModel, Field, ValidationError


class AlbumSpec(BaseModel):
    name: str
    person: str | None = None                       # who must be in it ("me", "Dad", a name) or None
    looks: list[str] = Field(default_factory=list)  # short visual descriptions to rank by, e.g. "overweight man"
    avoid: list[str] = Field(default_factory=list)  # descriptions to push down
    judge_question: str | None = None                # yes/no question about the requested CONDITION (never identity)
    date_from: str | None = None                     # ISO date, inclusive
    date_to: str | None = None                       # ISO date, exclusive
    time_phrase: str | None = None                   # exact words from the request that set the dates (grounding)
    place: str | None = None                         # place named in the request ("Tokyo", "Cape Cod"), exact words
    time_of_day: str | None = None                   # "HH:MM-HH:MM" local clock (may wrap midnight), set by code from the words
    media: str = "any"                               # "photo" | "video" | "any"
    want: str = "all"                                # "all" = find every match; "best" = top-ranked only
    max_items: int | None = None


class Plan(BaseModel):
    albums: list[AlbumSpec]
    notes: str = ""


TEMPLATE = """You convert a person's request about their own photo library into a JSON search plan.
Today's date is {today}. The library owner is {owner}. Known people in the library: {people}.

Return ONLY JSON matching this schema:
{{"albums": [{{"name": str, "person": str|null, "looks": [str], "avoid": [str], "judge_question": str,
  "place": str|null, "time_phrase": str|null, "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, "media": "photo"|"video"|"any",
  "want": "all"|"best", "max_items": int|null}}], "notes": str}}

Rules:
- One album per group the person asks for. If they ask for two categories, make two albums.
- "person" must be one of the known people, "me" for the owner, or null if no specific person.
- "looks": 1-4 short, concrete VISUAL descriptions of the CONDITION the person asked for (e.g. "a man with a heavy build
  and round face", "a slice of bread"). Identity is handled separately by face matching: never describe what the person
  looks like in general (no hair color, eye color, "looks like <name>"). If the request names a person but gives no
  condition (e.g. "every photo of Dad"), "looks" is [] and "judge_question" is null.
- "judge_question": a yes/no question about ONE image, mentioning the person as "the person in the red box" when a
  person is specified, e.g. "Does the person in the red box look overweight in this photo?"
- Dates: "time_phrase" = the exact words of the request that constrain THIS album's time (e.g. "past 6 months"), or
  null if the request gives no time for this album. A time phrase attached to one album does not apply to the other.
  Convert it using today's date ("past 6 months" -> date_from = today minus 6 months). If time_phrase is null, both dates are null.
  date_to is EXCLUSIVE: "the 1990s" -> date_from "1990-01-01", date_to "2000-01-01"; "in 2019" -> "2019-01-01".."2020-01-01".
- "place": the exact words naming a geographic place in the request (city, region, country, landmark area), else null.
  Not a scene type ("beach" is a look, not a place; "Cape Cod" is a place).
- media: "video" only if they ask only for videos; "any" if they say photos and videos.
- want: "best" if they ask for the best/top items, else "all".

Request: {request}
JSON:"""


def build_prompt(request: str, owner: str = "me", people: list[str] | None = None, today: date | None = None) -> str:
    return TEMPLATE.format(today=(today or date.today()).isoformat(), owner=owner,
                           people=", ".join(people or []) or "unknown", request=request.strip())


def parse_plan(text: str, request: str | None = None) -> Plan:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in planner output: {text[:200]!r}")
    P = Plan.model_validate(json.loads(m.group(0)))
    if request is not None:
        ground_dates(P, request)
        ground_place(P, request)
    strip_identity_conditions(P)
    fix_red_box(P)
    return P


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def ground_place(P: Plan, request: str) -> Plan:
    """Code-enforced like dates: a place is kept only if those exact words occur in the request."""
    req = _norm(request)
    for a in P.albums:
        if a.place and _norm(a.place) not in req:
            P.notes = (P.notes + f" [place '{a.place}' removed from '{a.name}': not in the request]").strip()
            a.place = None
    return P


def fix_red_box(P: Plan) -> Plan:
    """Code-enforced: a red box is drawn only around a matched PERSON. If the album has no person, the judge would be
    asked about a box that does not exist and answers 'no' (measured: 15/22 source-photo failures in the 72-query
    generalization test, e.g. 'Does the person in the red box look like an elephant seal?')."""
    for a in P.albums:
        if not a.person and a.judge_question and "red box" in a.judge_question.lower():
            q = re.sub(r"(?i)\b(the|a) person in (the|a) red box\b", "someone", a.judge_question)
            q = re.sub(r"(?i)\s*\bin (the|a) red box\b", "", q)
            a.judge_question = re.sub(r"\s{2,}", " ", q).strip()
    return P


# a question that ONLY tests who someone is (NAME = the album's person); anything else ("Is Mom laughing at dinner?") is a
# condition to keep, with the name replaced by the boxed person (fuzz 10-03: deleting it lost "laughing", "birthday party")
_IDENTITY_ONLY = re.compile(r"(is|are) (this|that|it|the person|the person in the red box|this person) NAME|"
                            r"(does|do|is) (the person|the person in the red box|this person|he|she|they) (look|looks|resemble|resembles) (like )?NAME|"
                            r"(does|do) (this|the) (photo|image|picture|video) (show|contain|include|feature|have) NAME|"
                            r"(is|are) NAME (in|present in|visible in|shown in|pictured in) (this|the) (photo|image|picture|video)|"
                            r"(is|are) NAME (in|present|visible|there|pictured)( here)?|"
                            r"(does|do) NAME appear( in (this|the) (photo|image|picture|video))?")


def strip_identity_conditions(P: Plan) -> Plan:
    """Code-enforced: identity comes from faces, so a question that only tests identity ("Does the person look like Drew
    Barrymore?") is removed (the planner did exactly this in the 03:2x demo). A real condition that names the person
    ("Is Mom laughing at dinner?") is kept, with the name replaced by the boxed person."""
    for a in P.albums:
        if a.person:   # the person's name in filter/exclude questions too (fuzz round 6)
            toks = [t for t in _norm(a.person).split() if len(t) > 2]
            for f in ("filter_question", "exclude_question"):
                q = getattr(a, f, None)   # (the one-shot planner's albums have no filter/exclude fields)
                if q and toks:
                    alt = "|".join(re.escape(t) for t in toks)
                    setattr(a, f, re.sub(r"(?i)\b(?:" + alt + r")(?:\s+(?:" + alt + r"))*('s)?(?!\w)",
                                         lambda m: "the person in the red box" + (m.group(1) or ""), q))
        if a.person and a.judge_question and re.fullmatch(
                r"(is|are) there (a |any )?(person|people|someone|a face|faces)( visible)?( in (the|this) (photo|image|picture|video))?",
                _norm(a.judge_question)):
            a.judge_question = None   # "every photo of me" + "Is there a person in the photo?": faces already decide that
            continue
        if a.person and a.judge_question:
            toks = [t for t in _norm(a.person).split() if len(t) > 2]
            q = _norm(a.judge_question)
            if not any(t in q.split() for t in toks):
                continue
            name = r"(?:" + "|".join(re.escape(t) for t in toks) + r")(?: s)?"
            qn = re.sub(r"\b" + name + r"(?: " + name + r")*\b", "NAME", q)
            if re.fullmatch(_IDENTITY_ONLY.pattern, qn):
                P.notes = (P.notes + f" [identity-style condition removed from '{a.name}': identity uses face matching]").strip()
                a.judge_question = None; a.looks = []; a.avoid = []
            else:
                alt = "|".join(re.escape(t) for t in toks)
                a.judge_question = re.sub(r"(?i)\b(?:" + alt + r")(?:\s+(?:" + alt + r"))*('s)?(?!\w)",
                                          lambda m: "the person in the red box" + (m.group(1) or ""), a.judge_question)
    return P


def ground_dates(P: Plan, request: str) -> Plan:
    """Code-enforced rule: an album keeps dates only if it quotes a time phrase that actually occurs in the request.
    (The LLM once copied 'past 6 months' from the 'fit' album onto the 'heavier' album, silently dropping every
    old photo. A prompt rule can be ignored; this check cannot.)"""
    req = _norm(request)
    for a in P.albums:
        tp = _norm(a.time_phrase or "")
        if not tp or tp not in req:
            if a.date_from or a.date_to:
                P.notes = (P.notes + f" [dates removed from '{a.name}': no time phrase in the request supports them]").strip()
            a.date_from = a.date_to = None; a.time_phrase = None
    return P


def plan(request: str, llm, owner="me", people=None, today=None, retries: int = 2) -> Plan:
    """`llm(prompt) -> str`. Retries on malformed JSON, feeding the validation error back."""
    prompt = build_prompt(request, owner, people, today)
    last = None
    for _ in range(retries + 1):
        out = llm(prompt if last is None else prompt + f"\n(Previous output was invalid: {last}. Return valid JSON only.)\nJSON:")
        try:
            return parse_plan(out, request)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    raise ValueError(f"planner failed: {last}")
