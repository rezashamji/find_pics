"""One path for everything the person types: the first request and every follow-up.

Each message -> the planner sees the conversation so far + the current plan -> returns the WHOLE updated plan
(same schema every time) -> the engine reruns it. There is no separate "edit" system and no mode flags:
  - two-step requests ("photos from the day I ..., without ...") are an optional anchor/window/exclude per album,
    filled in by the planner when the sentence needs them and left null otherwise;
  - there are no modes: every album keeps improving in rounds (most likely photos first) until the judge has looked at
    every photo; the person can stop at any round and the stated bound still holds (engine.stream_album);
  - follow-ups ("only the ones at night", "also add 2019", "drop the group shots") change the plan, then it reruns.
Reruns are cheap because the judge's answers are cached per (image, question) in the session folder.
Taps on the review page (this photo is wrong) are not language: they are stored as per-item overrides and applied
after every rerun.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from . import store
from .agent import Step, window_rows
from .engine import Thresholds, _judge_rows, make_exclusive, run_album, stream_album
from .planner import AlbumSpec, _norm, fix_red_box, ground_dates, ground_place, strip_identity_conditions


class SubjectRefs:
    """Reference photos of a subject that is not a face ("my dog Max", "my bike"): found by image-vector similarity,
    confirmed by the judge comparing [reference | candidate] side by side (DogFaceNet look-alike pairs: AUC 0.883 vs
    image vectors 0.566; strict question, cut ~0.5)."""

    def __init__(self, images, name: str = "it", kind: str = "subject"):
        self.images, self.name, self.kind = list(images), name, kind


SUBJECT_Q = ("The left panel shows {name}, one specific {kind}. Compare individual features: colour pattern and markings, "
             "shape, scars, distinctive details. Is the {kind} in the right panel the SAME individual {kind}, not just a "
             "similar-looking one? If you are not sure, answer no.")


class Album(AlbumSpec):
    anchor: Step | None = None              # what identifies the moment, when the album is defined by one
    window: str | None = None               # same_day | same_week | same_event | same_place
    exclude_question: str | None = None     # items the judge says YES to are dropped
    filter_question: str | None = None      # "only the ones where ...": items must ALSO get a YES to this


class Plan(BaseModel):
    albums: list[Album]
    notes: str = ""


TEMPLATE = """You maintain a JSON search plan over a person's own photo library during a conversation.
Today's date is {today} ({weekday}). The library owner is {owner}. Known people in the library: {people}.

The library can contain anything: people, pets, objects, places, screenshots, documents, pictures of pictures.
Never refuse and never return zero albums. "notes" describe what the plan searches for; nothing has been searched
yet, so never claim results ("Found ...").

Return ONLY JSON:
{{"albums": [{{"name": str, "person": str|null, "looks": [str], "avoid": [str], "judge_question": str|null,
   "anchor": {{"looks": [str], "judge_question": str}}|null,
   "window": "same_day"|"same_week"|"same_month"|"same_year"|"same_event"|"same_place"|"days_before:N"|"days_after:N"|null,
   "exclude_question": str|null, "filter_question": str|null, "place": str|null, "time_phrase": str|null,
   "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, "media": "photo"|"video"|"any",
   "want": "all"|"best", "max_items": int|null}}], "notes": str}}

Rules:
- One album per group the person asks for. If they ask for two categories, make two albums.
- "person": one of the known people, "me" for the owner, or null if no specific person.
- "looks": 1-4 short, concrete VISUAL descriptions of the CONDITION asked for (e.g. "a man with a heavy build and round
  face", "a slice of bread"). Identity is handled by face matching: never describe what a person looks like in general.
  If the request names a person but gives no condition ("every photo of Dad"), "looks" is [] and "judge_question" null.
- "judge_question": a yes/no question about ONE image. Never "I/me/my/we/us/our/you/the user/the owner" in it: the judge
  does not know the owner ("Is this the house we bought?" -> "Is this the outside of a house?").
  Say "the person in the red box" only when "person" is set.
- person "me" when the owner should be IN the photo ("photos of me", "me at the beach", "where I'm smiling", "the
  selfie I took", "my feet in the water"). "the sushi I ate", "videos I took of the sunset" are about the owner's
  library, not the owner's face: person null. Anyone else: a known person whose name is said ("Dad", "Mom", "Sara" when
  they are known people), or null. "my sister"/"my daughter" with no known person of that name -> person null, the notes
  ask who, and NEVER invent what they look like (no "a woman with long hair"): keep only the rest of the request.
- Indirect moments ("the day when...", "the week when...", "during the trip where...", "at the place where..."):
  "anchor" describes what is visible in photos of that moment, "window" how far around it to look, and looks/
  judge_question describe what to find inside that window. Otherwise "anchor" and "window" are null.
  "the day ..." -> same_day; "the week ..." -> same_week; "the month/year ..." -> same_month/same_year;
  "the trip/party/wedding where ..." -> same_event; "the city/place where ..." -> same_place;
  "the day after ..." -> days_after:1; "3 days before ..." -> days_before:3; "the week before ..." -> days_before:7. The moment's
  description is NOT a time_phrase and NOT a place: "the week I went to the Grand Canyon" -> anchor (Grand Canyon),
  window same_week, time_phrase null, place null.
- Exclusions ("without...", "excluding...", "no ..."): a yes/no question about the excluded thing in
  "exclude_question"; do not mention it in looks or judge_question.
- Extra conditions ("only the ones where/at/with ...") on an album that already has a judge_question or a person:
  a yes/no question in "filter_question" (photos must also pass it); keep judge_question as it is.
- Dates: "time_phrase" = the exact words that constrain THIS album's time, else null (then both dates null). A time
  phrase attached to one album does not apply to the other: "me heavier vs me fit in the past 6 months" -> only the
  "fit" album gets "past 6 months"; "heavier" has no dates. Convert
  with today's date. date_to is EXCLUSIVE: "the 1990s" -> 1990-01-01..2000-01-01; "in 2019" -> 2019-01-01..2020-01-01.
- "place": exact words naming a geographic place (city, region, country, landmark area), else null. "beach" is a look.
- media: "video" only if they ask only for videos. want: "best" if they ask for the best/top items, else "all".
{conversation}
JSON:"""


def build_prompt(message: str, history: list[str], current: Plan | None, owner="me", people=None, today=None) -> str:
    if current is None:
        conv = f"\nRequest: {message.strip()}"
    else:
        conv = ("\nThis is a follow-up. Earlier messages:\n" + "\n".join(f"- {m}" for m in history) +
                f"\nCurrent plan:\n{current.model_dump_json(exclude={'notes'})}\n"
                "Return the WHOLE updated plan. A follow-up EDITS the existing album(s); keep everything it does not change.\n"
                "- \"only ...\" narrows them: dates/place/media if it is about those, otherwise filter_question.\n"
                "- \"also ...\" widens them (e.g. \"also videos\" -> media any; \"also 2019\" -> widen the dates).\n"
                "- \"drop/remove/without ...\" -> exclude_question.\n"
                "- Undoing PART of an earlier change edits that field and keeps the rest: exclude_question \"sandwich or "
                "burger?\" + \"actually keep the sandwiches\" -> exclude_question \"burger?\".\n"
                "- \"he/she/her/him/it/them\" refers to the subject of the current album(s), never a new person.\n"
                "- Add a new album ONLY if the message clearly asks for a separate, additional group.\n"
                "- Change only the album(s) the message is about; leave the others exactly as they are."
                f"\nNew message: {message.strip()}")
    t = today or date.today()
    return TEMPLATE.format(today=t.isoformat(), weekday=t.strftime("%A"), owner=owner,
                           people=", ".join(people or []) or "unknown", conversation=conv)


_CAL = re.compile(r"\d|\b(today|tonight|yesterday|ago|last|past|this|next|recent|recently|decade|century|"
                  r"jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|"
                  r"september|october|november|december|spring|summer|fall|autumn|winter|christmas|thanksgiving|"
                  r"halloween|easter|new year|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
# "7 days before the photo of X" is relative to another photo (a moment), not to the calendar
_RELATIVE = re.compile(r"\b(before|after|since|until|prior to)\b(?!.*\b(19|20)\d\d\b)(?!.*\b(jan|feb|mar|apr|may|jun|jul|aug|"
                       r"sep|oct|nov|dec)[a-z]*\b)", re.I)
# a yes/no question about ONE photo cannot compare it with other photos or moments
_RELATIONAL = re.compile(r"\b(same|identical)\b[^?]*\bas (the|in|that|a)\b|\breference (photo|image|picture)\b|"
                         r"\b(previous|earlier|other|another|first|anchor|later) (photo|image|picture|event|day|trip|time)\b|"
                         r"\bsame (year|day|week|month|trip) as\b|\b(also|again|later)\b[^?]*\b(appear|appears|seen|shown|"
                         r"photographed|docking|docked)\b|\bsame (one|top|shirt|jacket|scarf|hat|person|dog|car|outfit|"
                         r"clothes)\b[^?]*\b(as|worn|in \d{4})\b|\b(twice|three times|consecutive)\b", re.I)
_WINDOW_WORDS = [("same_day", r"\b(the|that) day\b"), ("same_week", r"\b(the|that) week\b"),
                 ("same_month", r"\b(the|that) month\b"), ("same_year", r"\b(the|that) year\b"),
                 ("same_event", r"\b(trip|vacation|holiday|party|wedding|concert|game|event)\b")]


_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_REL_PHRASE = re.compile(r"(?i)\b(?:(?:last|this|past)\s+(?:summer|winter|spring|fall|autumn|week|weekend|month|year|night|"
                         r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|yesterday|today|tonight|this morning|"
                         r"(?:the\s+)?(?:week|weekend) before last)\b")
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_OCCASION = re.compile(r"\b(?:on |at |for )?(?:my|our|his|her|their|\w+ s) (?:birthday|bday|anniversary|wedding day|graduation)\b")


def resolve_relative(phrase: str, today: date) -> tuple[str, str] | None:
    """Relative calendar phrases resolved in code, not by the LLM (fuzz 10-03: it thought Saturday was a Thursday, made
    'last weekend' 8 days and 'this year' the past 12 months). Returns (date_from, date_to exclusive) or None when the
    phrase is not one of these simple forms (the planner's dates are then kept). Calendar conventions: last week/month/
    year = from the previous Monday / the previous calendar month / the previous calendar year."""
    t = re.sub(r"^(?:(?:from|in|on|during|taken|over|within|for|of|since|at)\s+)*(?:the\s+)?", "", _norm(phrase))
    t = re.sub(r"\s+(?:only|too)$", "", t)
    d1 = timedelta(days=1)
    def iso(a, b):
        return a.isoformat(), b.isoformat()
    if t in ("today", "this morning", "this afternoon", "this evening", "tonight", "earlier today"):
        return iso(today, today + d1)
    if t in ("yesterday", "last night", "yesterday morning", "yesterday afternoon", "yesterday evening"):
        return iso(today - d1, today)
    if t in ("last weekend", "past weekend"):    # the most recent Saturday-Sunday that has already ended
        sat = today - timedelta(days=(today.weekday() - 5) % 7 or 7)
        if sat + d1 >= today:
            sat -= timedelta(days=7)
        return iso(sat, sat + 2 * d1)
    if t in ("weekend before last", "the weekend before last"):
        a, b = resolve_relative("last weekend", today)
        return iso(date.fromisoformat(a) - timedelta(days=7), date.fromisoformat(b) - timedelta(days=7))
    if t == "this weekend":
        sat = today + timedelta(days=(5 - today.weekday()) % 7) if today.weekday() < 5 else today - timedelta(days=today.weekday() - 5)
        return iso(sat, sat + 2 * d1)
    m = re.fullmatch(r"(?:last |past |this past )?(" + "|".join(_DAYS) + r")", t)
    if m:                                        # "Tuesday" / "last Tuesday": the most recent one before today
        back = (today.weekday() - _DAYS.index(m.group(1))) % 7 or 7
        return iso(today - timedelta(days=back), today - timedelta(days=back) + d1)
    mon = today - timedelta(days=today.weekday())
    if t == "this week":
        return iso(mon, today + d1)
    if t == "week before last":
        return iso(mon - timedelta(days=14), mon - timedelta(days=7))
    if t in ("last week", "past week"):   # previous Monday .. today: people say "last week" for the last ~7-13 days
        return iso(mon - timedelta(days=7), today + d1)
    first = today.replace(day=1)
    if t == "this month":
        return iso(first, today + d1)
    if t == "last month":
        prev = (first - d1).replace(day=1)
        return iso(prev, first)
    if t == "this year":
        return iso(today.replace(month=1, day=1), today + d1)
    if t == "last year":
        return iso(date(today.year - 1, 1, 1), date(today.year, 1, 1))
    m = re.fullmatch(r"(?:last|past|previous) (\d+|" + "|".join(_NUM) + r"|a|one) ?(day|week|month|year)s?", t) or \
        re.fullmatch(r"(?:last|past|previous) ()(day|week|month|year)", t)
    if m and m.group(2) and not (m.group(1) == "" and m.group(2) in ("week", "month", "year")):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM.get(m.group(1), 1)
        if m.group(2) == "day":
            start = today - timedelta(days=n)
        elif m.group(2) == "week":
            start = today - timedelta(days=7 * n)
        else:
            k = n * (12 if m.group(2) == "year" else 1)
            y, mo = divmod(today.year * 12 + today.month - 1 - k, 12)
            mo += 1
            import calendar
            start = date(y, mo, min(today.day, calendar.monthrange(y, mo)[1]))
        return iso(start, today + d1)
    return None


def ground(P: Plan, message: str, history: list[str], today: date | None = None, owner: str = "me") -> Plan:
    """Code-enforced checks (same as the one-shot planner), against EVERYTHING the person has typed so far, so an
    album's dates/place from message 1 survive message 3."""
    said = " \n ".join(history + [message])
    for a in P.albums:   # a time phrase must name calendar time ("the week I went to X" is a moment -> anchor, not dates)
        if a.time_phrase and resolve_relative(a.time_phrase, today or date.today()) is None and \
                (not _CAL.search(a.time_phrase) or _RELATIVE.search(a.time_phrase)):
            P.notes = (P.notes + f" [dates removed from '{a.name}': '{a.time_phrase}' names no calendar time]").strip()
            a.time_phrase = None
    said_tok = set(_norm(said).split())
    span = re.findall(r"(?i)\b((?:from |between |in )?(?:19|20)\d\d(?:\s*(?:-|to|and|through|until)\s*(?:19|20)\d\d)?)\b", message) \
        or re.findall(r"(?i)\b((?:from |between |in )?(?:19|20)\d\d(?:\s*(?:-|to|and|through|until)\s*(?:19|20)\d\d)?)\b", said)
    rel = sorted(set(m.group(0) for m in _REL_PHRASE.finditer(message)))
    for a in P.albums:
        if not a.time_phrase and (a.date_from or a.date_to) and len(span) == 1:
            a.time_phrase = span[0]       # planner set dates but no phrase (planner test: "Dad from 2015 to 2018")
        elif not a.time_phrase and (a.date_from or a.date_to) and len(rel) == 1 and \
                sum(bool(b.date_from or b.date_to) for b in P.albums) == 1:
            a.time_phrase = rel[0]        # "the concert we went to last month": dates set, phrase left null (fuzz 10-03: 5/117)
        tp = a.time_phrase
        if not tp:
            continue
        years = sorted(int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", tp))
        if years and not re.search(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|spring|summer|fall|autumn|"
                                   r"winter|christmas|week|month|day|before|after|since|until|prior|s\b)", tp) \
                and not re.search(r"\d0s\b", tp):
            # years only ("2019 and 2020", "2015 to 2018"): whole years, end exclusive (the planner wrote 2020-01-01 once)
            a.date_from, a.date_to = f"{years[0]}-01-01", f"{years[-1] + 1}-01-01"
        if _norm(tp) not in _norm(said):
            # planner reformatted the phrase ("from 2015 to 2018" -> "2015-2018"): accept if every word/number in it was said
            toks = [t for t in _norm(tp).split() if t not in ("to", "and", "from", "through", "between", "in", "the")]
            if toks and all(t in said_tok for t in toks):
                a.time_phrase = said    # grounded by its words; ground_dates below checks a substring of what was said
    ground_dates(P, said); ground_place(P, said)
    today = today or date.today()
    for a in P.albums:
        tp = a.time_phrase
        if not tp:
            continue
        rest = _OCCASION.sub("", _norm(tp)).strip()
        if rest != _norm(tp):      # "on my birthday this year": the planner cannot know the date (it guessed today)
            P.notes = (P.notes + f" [I don't know the date of the occasion in '{tp}'; "
                                 f"{'searching ' + rest if rest else 'no date limit'}]").strip()
            r = resolve_relative(rest, today) if rest else None
            if r is None and not re.search(r"\d", rest):
                a.date_from = a.date_to = a.time_phrase = None; continue
        else:
            r = resolve_relative(tp, today)
        if r and (a.date_from, a.date_to) != r:
            P.notes = (P.notes + f" [dates for '{tp}': {r[0]} to {(date.fromisoformat(r[1]) - timedelta(days=1)).isoformat()}"
                                 f" (computed from today, {today.strftime('%A')} {today.isoformat()})]").strip()
        if r:
            a.date_from, a.date_to = r
        elif a.date_to and re.search(r"(?i)\b(and|to|through|thru|until|till|-)\s*(" + _MONTHS + r")[a-z]*\.?\s+(\d{1,2})(st|nd|rd|th)?\b", tp):
            # "between July 1st and July 15th": the named end day is included (the planner wrote date_to = the 15th,
            # exclusive, dropping that day: fuzz 10-03)
            m = re.findall(r"(?i)\b(" + _MONTHS + r")[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", tp)[-1]
            try:
                end = date(int(a.date_to[:4]), _MONTHS.split("|").index(m[0][:3].lower()) + 1, int(m[1]))
                if end.isoformat() == a.date_to:
                    a.date_to = (end + timedelta(days=1)).isoformat()
            except ValueError:
                pass
    said_words = _norm(said).split()
    first_person = bool(re.search(r"(?i)\b(i|me|my|mine|myself|i'm|im|i've|ive|i'd|we|us|our)\b", said))
    for a in P.albums:   # a person must be NAMED in the conversation ("my sister" became Sara, "we" became Dad: fuzz 10-03)
        is_owner = _norm(a.person or "") in ("me", "i", "myself") or set(_norm(a.person or "x").split()) <= set(_norm(owner or "me").split())
        if a.person and is_owner and not first_person and not any(
                t in said_words for t in _norm(owner or "").split() if len(t) > 2):
            P.notes = (P.notes + f" [person removed from '{a.name}': the request does not mention you]").strip()
            a.person = None
        if a.person and not is_owner and not any(
                w.startswith(t) or t.startswith(w) for t in _norm(a.person).split() if len(t) > 2
                for w in said_words if len(w) > 2):
            P.notes = (P.notes + f" [person '{a.person}' removed from '{a.name}': not named in the request. Who is it? Say "
                                 f"their name, or add reference photos]").strip()
            a.person = None
    strip_identity_conditions(P); fix_red_box(P)
    for a in P.albums:
        if a.judge_question and a.exclude_question and _norm(a.judge_question) == _norm(a.exclude_question):
            a.judge_question = None; a.looks = []   # "all photos that week, excluding X" (DISBench q4 asked X twice)
        if a.anchor and a.window not in ("same_day", "same_week", "same_month", "same_year", "same_event", "same_place") \
                and not re.fullmatch(r"days_(before|after):\d+", a.window or ""):
            a.window = "same_event"     # an invented window ("same_year") would otherwise mean "the whole library"
        if a.anchor:     # the window is what the words say ("the week ..." -> same_week), when they say it
            if not re.fullmatch(r"days_(before|after):\d+", a.window or ""):   # explicit offsets win over word rules
                for w, pat in _WINDOW_WORDS:
                    if re.search(pat, said, re.I):
                        a.window = w; break
            if a.place and a.anchor and _norm(a.place) in _norm(" ".join(a.anchor.looks + [a.anchor.judge_question or ""])):
                a.place = None   # the place IS the anchor moment, not a filter on what to find
        if a.window and not a.anchor:
            a.window = None
        if a.anchor and a.anchor.judge_question:     # the anchor is about a moment, never a boxed person
            q = re.sub(r"(?i)\b(the|a) person in the red box\b", "someone", a.anchor.judge_question)
            a.anchor.judge_question = re.sub(r"(?i)\s*\bin the red box\b", "", q).strip()
        for f in ("exclude_question", "filter_question"):    # asked about the whole photo, never a boxed person
            q = getattr(a, f)
            if q:
                setattr(a, f, re.sub(r"\s{2,}", " ", re.sub(r"(?i)\s*\bin the red box\b", "", q.replace(
                    "the person in the red box", "the person"))).strip())
    return P


def plan_turn(message: str, llm, history: list[str] | None = None, current: Plan | None = None, owner="me",
              people=None, today=None, retries: int = 2) -> Plan:
    history = history or []
    prompt = build_prompt(message, history, current, owner, people, today)
    last, fallback = None, None
    for _ in range(retries + 1):
        out = llm(prompt if last is None else prompt + f"\n(Previous output was invalid: {last}. Return valid JSON only.)\nJSON:")
        try:
            m = re.search(r"\{.*\}", out, re.S)
            if not m:
                raise ValueError(f"no JSON object in planner output: {out[:200]!r}")
            P = Plan.model_validate(json.loads(m.group(0)))
            if not P.albums:
                raise ValueError("zero albums; the library can contain anything, return at least one album")
            problem = _unanswerable(P)
            if problem:
                fallback = P          # well-formed: usable if every retry repeats the problem
                raise ValueError(problem)
            return ground(P, message, history, today, owner)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    if fallback is not None:      # degrade instead of failing (DISBench q30): drop the part one photo cannot answer
        return ground(_drop_unanswerable(fallback), message, history, today, owner)
    raise ValueError(f"planner failed: {last}")


# the judge is a stranger looking at one photo: "the house we bought", "the concert we went to", "you and Reza" (fuzz 10-03:
# ~9/117 plans) need knowledge it does not have
def _personal(q: str) -> bool:
    return bool(re.search(r"\bI\b", q) or re.search(r"(?i)\b(me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours)\b|"
                                                     r"\bthe (user|owner)\b", q))


_NAMED = re.compile(r"(?i)\b(person|man|woman|boy|girl|child|kid|baby|guy|lady|someone|friend)\s+(named|called)\b")


def _bad_q(q: str | None) -> bool:
    return bool(q) and (bool(_RELATIONAL.search(q)) or _personal(q) or bool(_NAMED.search(q)))


def _unanswerable(P: Plan) -> str | None:
    for a in P.albums:   # questions the judge cannot answer from ONE photo (DISBench q3/q17/q30): ask again
        for q in [a.judge_question] + ([a.anchor.judge_question] if a.anchor else []) + [a.exclude_question, a.filter_question]:
            if q and _RELATIONAL.search(q):
                return (f"'{q}' refers to another photo or moment; the judge sees ONE photo at a time. "
                        "Put the moment in 'anchor'/'window' and ask only about what is visible in this photo")
            if q and _NAMED.search(q):
                return (f"'{q}' asks for a name; the judge cannot know names. Put a known person in 'person' or ask "
                        "only about what is visible")
            if q and _personal(q):
                return (f"'{q}' needs to know the owner (I/me/my/we/our/you); the judge is a stranger seeing ONE photo. "
                        "Ask only about what is visible (use 'the person in the red box' when 'person' is set)")
        if a.anchor and a.judge_question and _norm(a.judge_question) == _norm(a.anchor.judge_question):
            return (f"album '{a.name}': judge_question repeats the anchor question; judge_question must "
                    "describe what to find INSIDE the moment, not the moment itself")
    return None


def _drop_unanswerable(P: Plan) -> Plan:
    """Last resort: replace a question one photo cannot answer by a plain visual question built from 'looks', and
    say so in the notes (shown to the person), instead of refusing the whole request."""
    for a in P.albums:
        def plain(q, looks):
            return f"Does this photo show {looks[0]}?" if looks else None
        if a.judge_question and a.anchor and _norm(a.judge_question) == _norm(a.anchor.judge_question or ""):
            # "photos from the wedding": the moment itself was asked twice -> everything inside it (the old fallback,
            # "Does this photo show <first look>?", kept only bride-and-groom photos: fuzz 10-03, ~12/117)
            P.notes = (P.notes + " [showing everything from that moment]").strip()
            a.judge_question = None; a.looks = []
        if a.judge_question and _bad_q(a.judge_question):
            P.notes = (P.notes + f" [could not express '{a.judge_question}' as a question about one photo; "
                                 f"searching for what it looks like instead]").strip()
            a.judge_question = plain(a.judge_question, a.looks)
        if a.anchor and _bad_q(a.anchor.judge_question):
            a.anchor.judge_question = plain(a.anchor.judge_question, a.anchor.looks) or a.anchor.judge_question
        if _bad_q(a.exclude_question):
            P.notes = (P.notes + f" [dropped exclusion '{a.exclude_question}': not answerable from one photo]").strip()
            a.exclude_question = None
        if _bad_q(a.filter_question):   # forgotten at first (DISBench v4: 10/14 losses)
            P.notes = (P.notes + f" [dropped condition '{a.filter_question}': not answerable from one photo]").strip()
            a.filter_question = None
    return P


class CachedJudge:
    """Wraps a judge; answers are cached per (image pixels, question), so rerunning a changed plan only pays for
    images/questions not seen before. Persisted as JSON in the session folder."""

    def __init__(self, judge, path: Path | None = None):
        self.judge, self.path = judge, path
        self.cache = json.loads(path.read_text()) if path and path.exists() else {}
        self.hits = self.misses = 0

    def text(self, prompt, **kw):
        return self.judge.text(prompt, **kw)

    def p_yes(self, images, question):
        keys = [hashlib.md5(question.encode() + im.tobytes() + str(im.size).encode()).hexdigest() for im in images]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        if todo:
            for i, p in zip(todo, self.judge.p_yes([images[i] for i in todo], question)):
                self.cache[keys[i]] = float(p)
        self.hits += len(keys) - len(todo); self.misses += len(todo)
        return [self.cache[k] for k in keys]

    def save(self):
        if self.path:
            self.path.write_text(json.dumps(self.cache))


def run_plan(idx, P: Plan, enc, judge, refs_for=None, th: Thresholds = Thresholds(), max_anchor: int | None = None,
             exclude_ids: set | None = None) -> list:
    """The final answer (the first answer unless th.stream). See stream_plan."""
    out = None
    for out in stream_plan(idx, P, enc, judge, refs_for, th, max_anchor, exclude_ids):
        pass
    return out


def stream_plan(idx, P: Plan, enc, judge, refs_for=None, th: Thresholds = Thresholds(), max_anchor: int | None = None,
                exclude_ids: set | None = None):
    """Yields the list of album results after each round; albums advance round-robin so all of them improve together.
    Rows in `returned`/`judged` point at the FULL index. refs_for(person) -> (name, refs, ref_face_row, n_tagged)."""
    gens = [_album_stream(idx, a, enc, judge, refs_for, th, max_anchor, exclude_ids or set()) for a in P.albums]
    cur = [next(g) for g in gens]
    yield make_exclusive(cur)
    live = list(range(len(gens)))
    while live:
        moved = False
        for i in list(live):
            try:
                cur[i] = next(gens[i]); moved = True
            except StopIteration:
                live.remove(i)
        if moved:
            yield make_exclusive(cur)


def _album_stream(idx, a, enc, judge, refs_for, th, max_anchor, exclude_ids):
    a = place_or_look(idx, filter_to_place(idx, a))
    refs = ref_face = subject = None
    if a.person:
        if refs_for is None:
            raise ValueError(f"album '{a.name}' needs reference faces for '{a.person}'")
        _, refs, ref_face, _ = refs_for(a.person)
        if isinstance(refs, SubjectRefs):
            subject, refs = refs, None
    sub, trace = idx, {}
    if a.anchor:
        anc = Album(name=f"{a.name} (anchor)", looks=a.anchor.looks, judge_question=a.anchor.judge_question,
                    date_from=a.date_from, date_to=a.date_to, place=a.place, media=a.media)
        ar = run_album(idx, anc, enc, judge, None, th=Thresholds(**{**th.__dict__, "stream": False}))
        # EVERY confident anchor hit opens a window (not just the top few): if the planner adds an anchor that merely
        # repeats the target ("pics at a car show"), all car shows stay in scope; a real anchor is a specific moment
        # and still narrows the search. (Word rules to drop such anchors removed 35/85 legitimate DISBench two-steps.)
        top = ar.returned.sort_values("p_attr", ascending=False)
        top = (top.head(max_anchor) if max_anchor else top).item_row.to_numpy()
        # anchor found nothing -> the moment was not found: say so and return nothing. (Old agent fell back to the
        # WHOLE library: DISBench q99/q32 returned 440/405 photos for a moment that was never found.)
        scope = window_rows(idx, top, a.window) if len(top) else np.zeros(0, int)
        trace = dict(anchor_found=len(ar.returned), anchor_used=len(top), window=a.window, window_items=len(scope))
        if not len(scope):
            r = ar
            r.returned = ar.returned.iloc[0:0]; r.judged = ar.judged.iloc[0:0]; r.cert = None; r.spec = a
            r.report = (f"Album '{a.name}': 0 items.\n  Could not find the moment this album is anchored to "
                        f"(\"{a.anchor.judge_question}\"): no photo passed that question, so nothing was searched "
                        f"around it. Describe the moment differently, or say \"search everywhere\" to look for "
                        f"\"{a.judge_question}\" across the whole library instead.")
            r.trace = trace
            yield r
            return
        sub = store.subset(idx, scope) if len(scope) < idx.n_items else idx
        spec = a.model_copy(update=dict(date_from=None, date_to=None, time_phrase=None, place=None))
        ref_face = None if sub is not idx else ref_face
    else:
        spec = a
    row_of = {iid: i for i, iid in enumerate(idx.items.item_id)} if sub is not idx else None
    pex: dict = {}                       # exclusion answers, judged once per item
    pfil: dict = {}                      # filter answers ("only the ones where ..."), judged once per item
    pcond: dict = {}                     # subject albums: the album's own condition, judged on identity matches
    if subject is not None:
        kind = subject.kind if subject.kind != "subject" else (a.looks[0] if a.looks else "subject")
        # the album's own condition ("Max outdoors") is asked about the whole photo; the planner phrases it for a person
        cond_q = a.judge_question and re.sub(r"\s{2,}", " ", re.sub(r"(?i)\b(the|a) person in the red box\b", f"the {kind}",
                                                                      a.judge_question).replace(" in the red box", "")).strip()
        V = enc.images(subject.images).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
        # mean over the reference photos, not max: R-precision things 0.695 -> 0.731, places 0.680 -> 0.693 (1,703 / 1,500
        # identities, eval/instance_combos.py). (Faces keep max: references of a person can span very different eras.)
        fast = store.per_item_max((sub.clip.astype(np.float32) @ V.T).mean(1), sub.units["item_row"].to_numpy(), sub.n_items)
        spec = spec.model_copy(update=dict(person=None, looks=a.looks or [kind],
                                           judge_question=SUBJECT_Q.format(name=a.person, kind=kind)))
        # measured (eval_pet_search): image-vector similarity RANKS best (mixed library: top-3 precision 114/120, all-dogs
        # library 87/120); the side-by-side judge is a safe VETO (cut 0.2 keeps 140/147 targets) but a poor filter at 0.5
        gen = stream_album(sub, spec, enc, judge, None, th=Thresholds(**{**th.__dict__, "judge_accept": 0.2}),
                           fast_override=fast, ref_img=subject.images[0])
    else:
        cond_q = None
        gen = stream_album(sub, spec, enc, judge, refs, ref_face_row=ref_face, th=th)
    for r in gen:
        t = dict(trace)
        if cond_q and len(r.returned):    # "Max at the beach": identity first, then the condition on those photos
            new = r.returned[~r.returned.item_id.isin(pcond)]
            if len(new):
                pcond.update(zip(new.item_id, _judge_rows(sub, judge, new.item_row.to_numpy(), np.full(len(new), -1), cond_q)))
            keep = r.returned.item_id.map(pcond).to_numpy() >= th.judge_accept
            t["filtered_out"] = int((~keep).sum()); r.returned = r.returned[keep]
        if a.filter_question and len(r.returned):    # "only the ones where I'm outdoors": keep the YES
            new = r.returned[~r.returned.item_id.isin(pfil)]
            if len(new):
                pfil.update(zip(new.item_id, _judge_rows(sub, judge, new.item_row.to_numpy(), new.face_row.to_numpy(),
                                                         a.filter_question, crop_person=False)))
            keep = r.returned.item_id.map(pfil).to_numpy() >= th.judge_accept
            t["filtered_out"] = int((~keep).sum()); r.returned = r.returned[keep]
        if a.exclude_question and len(r.returned):
            new = r.returned[~r.returned.item_id.isin(pex)]
            if len(new):
                pex.update(zip(new.item_id, _judge_rows(sub, judge, new.item_row.to_numpy(), new.face_row.to_numpy(),
                                                        a.exclude_question, crop_person=bool(a.person))))
            drop = r.returned.item_id.map(pex).to_numpy() >= th.judge_accept
            t["excluded"] = int(drop.sum()); r.returned = r.returned[~drop]
        if row_of is not None:   # map rows back to the full index
            for df in (r.returned, r.judged):
                if len(df) and "item_id" in df:
                    df["item_row"] = [row_of[i] for i in df.item_id]
        if exclude_ids:
            n0 = len(r.returned); r.returned = r.returned[~r.returned.item_id.astype(str).isin(exclude_ids)]
            if n0 - len(r.returned):
                t["removed_by_you"] = n0 - len(r.returned)
        c = r.cert or {}
        if c.get("tail_sampled", 0) >= 50 and c["tail_hits"] / c["tail_sampled"] > 0.5 and "too broad" not in r.report:
            # the judge says yes to most RANDOM photos: the question is probably always-true (DISBench: "Is this photo
            # of the building in real life or non-real form?" matched 1,923 of 1,948). Warn; do not silently filter.
            r.report += (f"\n  Warning: the judge said yes to {c['tail_hits']} of {c['tail_sampled']} randomly chosen "
                         f"photos, so this question may be too broad to mean what you asked: \"{spec.judge_question}\".")
        if subject is not None:   # best matches first; identity of a pet/object is NOT certified -> no bound shown
            r.cert = None
            r.report = re.sub(r"\n  Scored all .*?(?=\n|$)", "", r.report, flags=re.S)
        if subject is not None and len(r.returned):
            r.returned = r.returned.sort_values("fast", ascending=False)
            if "not certified" not in r.report:
                r.report += ("\n  Found by similarity to your reference photos, best matches first; the judge only removed "
                             "clear mismatches. Unlike object search, which photos show this exact individual is not "
                             "certified: check the list (on a hard test, 72-95% of each search's top 3 were right).")
        n_drop = t.get("excluded", 0) + t.get("removed_by_you", 0) + t.get("filtered_out", 0)
        if n_drop:
            _after_removal(r, n_drop)
        if t:
            r.report += "\n  Steps: " + json.dumps(t)
        r.trace = t
        yield r


def _after_removal(r, n_drop: int):
    """Exclusions / taps removed items AFTER the search: fix the count in the report and restate the bound for what is
    left. Every match of the narrower request is also a match of the search, so the search's upper bound on missed
    matches still bounds the misses: recall >= kept / (kept + missed_upper)."""
    kept = len(r.returned)
    r.report = r.report.replace(f"': {kept + n_drop} items.", f"': {kept} items.", 1)
    c = r.cert
    if c and c.get("missed_upper") is not None:
        mu, mp = c["missed_upper"], c["missed_point"]
        c = r.cert = {**c, "found": kept, "recall_lower": kept / (kept + mu) if kept + mu > 0 else 0.0,
                      "recall_point": kept / (kept + mp) if kept + mp > 0 else float("nan")}
        r.report += (f"\n  After removing {n_drop} item(s) you asked to leave out: {kept} items; at least "
                     f"{c['recall_lower']:.0%} of matching items found (same bound on missed matches as above).")


def filter_to_place(idx, a: Album) -> Album:
    """'only from Paris' written as filter_question 'Is the location Paris?' (planner test, 10-03): a judge cannot see
    which city a photo is from; GPS/place metadata can. If a capitalized name in the filter matches this library's place
    names, use it as the place filter instead."""
    if not a.filter_question or "place" not in idx.items:
        return a
    from .engine import scope_mask
    names = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", a.filter_question)
    for n in names[::-1]:
        if n.split()[0] in ("Is", "Does", "Are", "Was", "Do", "The"):
            continue
        if scope_mask(idx, Album(name="p", place=n)).any():
            return a.model_copy(update=dict(place=n, filter_question=None))
    return a


def place_or_look(idx, a: Album) -> Album:
    """A 'place' that matches no item's place name in THIS library ("beach", "gym") is a kind of scene, not a geographic
    filter: turn it into a visual condition instead of silently returning nothing (planner test: 'Dad at the beach'
    became place='beach' with no condition)."""
    if not a.place:
        return a
    from .engine import scope_mask     # same word match the engine's place filter uses
    if "place" in idx.items and scope_mask(idx, Album(name="p", place=a.place)).any():
        return a
    q = f"Is the person in the red box at a {a.place}?" if a.person else f"Was this photo taken at a {a.place}?"
    jq = f"{a.judge_question.rstrip(' ?')}, and {q[0].lower()}{q[1:]}" if a.judge_question else q
    return a.model_copy(update=dict(place=None, looks=a.looks + [a.place], judge_question=jq))


class Session:
    """A folder holding the conversation: session.json (messages, plans, your per-item overrides), the judge cache,
    and one subfolder per turn (turn_1/, turn_2/, ...; earlier turns are kept, so any answer can be gone back to)."""

    def __init__(self, folder):
        self.dir = Path(folder); self.dir.mkdir(parents=True, exist_ok=True)
        f = self.dir / "session.json"
        self.state = json.loads(f.read_text()) if f.exists() else dict(index_dir=None, messages=[], plans=[], exclude_ids=[])

    @property
    def current(self) -> Plan | None:
        return Plan.model_validate(self.state["plans"][-1]) if self.state["plans"] else None

    @property
    def turn(self) -> int:
        return len(self.state["messages"])

    def add_turn(self, message: str, P: Plan):
        self.state["messages"].append(message); self.state["plans"].append(P.model_dump())

    def add_reviews(self, reviews: dict):
        bad = {str(k) for k, v in reviews.items() if v == "bad"}
        self.state["exclude_ids"] = sorted(set(self.state["exclude_ids"]) | bad)

    def save(self):
        (self.dir / "session.json").write_text(json.dumps(self.state, indent=1, default=str))
