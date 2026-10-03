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
import pandas as pd
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
    with_people: list[str] = []             # other people who must ALSO be in the photo ("me with Pierce"): face-matched
    until: Step | None = None               # a second moment that ENDS the span: "after A but before B" (anchor A, until B)


class Plan(BaseModel):
    albums: list[Album]
    notes: str = ""
    unknown_people: list[str] = []      # people the request names who are not known yet ("Jay", "my sister"): the chat
                                        # offers the face sheet so they can be named once


TEMPLATE = """You maintain a JSON search plan over a person's own photo library during a conversation.
Today's date is {today} ({weekday}). The library owner is {owner}. Known people in the library: {people}.

The library can contain anything: people, pets, objects, places, screenshots, documents, pictures of pictures.
Never refuse and never return zero albums. "notes" describe what the plan searches for; nothing has been searched
yet, so never claim results ("Found ...").

Return ONLY JSON:
{{"albums": [{{"name": str, "person": str|null, "looks": [str], "avoid": [str], "judge_question": str|null,
   "anchor": {{"looks": [str], "judge_question": str}}|null,
   "window": "same_day"|"same_week"|"same_month"|"same_year"|"same_event"|"same_place"|"days_before:N"|"days_after:N"|
             "before"|"after"|"since"|"until"|"minutes_before:N"|"minutes_after:N"|null,
   "exclude_question": str|null, "filter_question": str|null, "with_people": [str],
   "until": {{"looks": [str], "judge_question": str}}|null, "place": str|null, "time_phrase": str|null,
   "time_of_day": str|null,
   "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, "media": "photo"|"video"|"any",
   "want": "all"|"best", "max_items": int|null}}], "notes": str, "unknown_people": [str]}}

Rules:
- One album per group the person asks for. If they ask for two categories, make two albums. A date range ("March
  through June") is ONE album with one date span, never one album per month.
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
- "until": a SECOND moment that ends the span, {{"looks": [...], "judge_question": ...}}: "after we photographed the driver
  but before we reached the Citadel" -> anchor = the driver photo, window "after" (same trip) or "since", until = the
  Citadel. Otherwise null.
- "with_people": other known people who must ALSO be in the photo: "me with Dad" -> person "me", with_people ["Dad"];
  "Mom and Dad together" -> person "Mom", with_people ["Dad"]. Two people as two separate albums only if they ask for
  two albums.
- "unknown_people": people the request mentions who are not known people (a name like "Jay", or "my sister"), else [].
- Indirect moments ("the day when...", "the week when...", "during the trip where...", "at the place where..."):
  "anchor" describes what is visible in photos of that moment, "window" how far around it to look, and looks/
  judge_question describe what to find inside that window. Otherwise "anchor" and "window" are null.
  "the day ..." -> same_day; "the week ..." -> same_week; "the month/year ..." -> same_month/same_year;
  "the trip/party/wedding where ..." -> same_event; "the city/place where ..." -> same_place;
  "the day after ..." -> days_after:1; "3 days before ..." -> days_before:3; "the week before ..." -> days_before:7.
  Earlier/later IN THE SAME trip or day: "during the trip, before the first photo of X" / "before I saw X" -> anchor X,
  window "before"; "later during the same trip/day than X" -> window "after". Any later/earlier time, not the same
  trip ("cats on the windowsill after the cat tree was assembled", "later hanging on a backpack") -> "since"/"until".
  Minutes or hours: "30 minutes before X" -> minutes_before:30, "right/immediately after X" -> minutes_after:30, "an
  hour after X" -> minutes_after:60; a range ("30 minutes to 1 hour before") -> its larger end (minutes_before:60). The moment's
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
- "time_of_day": "yes" if THIS album is limited to a time of day the person said ("between 8 and 11pm", "in the
  morning", "at night", "around 3pm"), else null. The exact hours are computed by code from the words.
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
_MONTH_RANGE = re.compile(r"(?i)\b(?:between )?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?: \d{4})? (?:to|till|til|"
                          r"until|through|thru|and|-) (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?: \d{4})?"
                          r"(?: (?:of )?(?:last|this) year)?\b")
_REL_PHRASE = re.compile(r"(?i)\b(?:(?:last|this|past)\s+(?:summer|winter|spring|fall|autumn|week|weekend|month|year|night|"
                         r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|yesterday|today|tonight|this morning|"
                         r"(?:the\s+)?(?:week|weekend) before last)\b")
_MONTH_NUM = {m: i % 12 + 1 for i, m in enumerate("january february march april may june july august september october "
                                                  "november december jan feb mar apr may jun jul aug sep oct nov dec".split())}
_MONTH_NUM["sept"] = 9
_MONTH_NAMES = sorted(_MONTH_NUM, key=len, reverse=True)
_HOLIDAYS = [(r"christmas eve", (12, 24)), (r"christmas|xmas", (12, 25)), (r"new year s eve|nye", (12, 31)),
             (r"new year s day|new year s|new years|new year", (1, 1)), (r"(?:fourth|4th) of july|july (?:4th|fourth|4)", (7, 4)),
             (r"halloween", (10, 31)), (r"valentine s day|valentines day|valentine s", (2, 14)), (r"thanksgiving", (11, 0))]
def _hm(h: int, m: int = 0) -> str:
    return f"{h % 24:02d}:{m:02d}"


def resolve_time_of_day(text: str) -> str | None:
    """Clock range from words, computed in code (never by the model): "between 8 and 11pm" -> 20:00-23:00,
    "after 10pm" -> 22:00-04:00, "before 9am" -> 04:00-09:00, "around 3pm" -> 14:00-16:00, "in the morning" ->
    05:00-12:00, "at night" -> 20:00-04:00. "last night"/"tonight"/"Friday night" are DATES, not hours (after-midnight
    photos of a party carry the next day's date), so they set no hours. None if no time of day is said."""
    t = text.lower()
    num = r"(\d{1,2})(?::(\d\d))?\s*(am|pm|a\.m\.|p\.m\.)?"
    m = re.search(r"\b(?:between|from)\s+" + num + r"\s*(?:and|to|-|till|until|til)\s*" + num, t)
    if m and (m.group(3) or m.group(6)):
        h1, m1, ap1, h2, m2, ap2 = int(m.group(1)), int(m.group(2) or 0), m.group(3), int(m.group(4)), int(m.group(5) or 0), m.group(6)
        def to24(h, ap):
            return h % 12 + (12 if ap and ap.startswith("p") else 0)
        e = to24(h2, ap2 or ap1)
        if ap1:
            s0 = to24(h1, ap1)
        else:   # "8 to 11pm" / "11 to 2am": the start's am/pm is the one that gives the shorter span
            s0 = min((to24(h1, "am"), to24(h1, "pm")), key=lambda x: (e - x) % 24 or 24)
        return f"{_hm(s0, m1)}-{_hm(e, m2)}"
    m = re.search(r"\b(after|past|later than)\s+" + num, t)
    if m and m.group(4):
        h = int(m.group(2)) % 12 + (12 if m.group(4).startswith("p") else 0)
        return f"{_hm(h, int(m.group(3) or 0))}-{_hm(4) if h >= 12 or h < 4 else _hm(12)}"
    m = re.search(r"\b(before|earlier than)\s+" + num, t)
    if m and m.group(4):
        h = int(m.group(2)) % 12 + (12 if m.group(4).startswith("p") else 0)
        return f"{_hm(4) if h > 4 else _hm(0)}-{_hm(h, int(m.group(3) or 0))}"
    m = re.search(r"\b(?:at|around|about|near)\s+" + num, t)
    if m and m.group(3):
        h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
        return f"{_hm(h - 1, int(m.group(2) or 0))}-{_hm(h + 1, int(m.group(2) or 0))}"
    if re.search(r"\bmidnight\b", t):
        return "23:00-01:00"
    if re.search(r"\b(noon|midday|lunch ?time)\b", t):
        return "11:30-14:00"
    if re.search(r"\b(late at night|late night|middle of the night)\b", t):
        return "23:00-04:00"
    if re.search(r"\b(at night|night ?time|nighttime|in the night)\b", t):
        return "20:00-04:00"
    if re.search(r"\b(in the (early )?morning|this morning|morning|mornings)\b", t) and not re.search(r"\bmorning (after|of)\b", t):
        return "05:00-12:00"
    if re.search(r"\b(in the afternoon|this afternoon|afternoon|afternoons)\b", t):
        return "12:00-17:00"
    if re.search(r"\b(in the evening|this evening|evening|evenings)\b", t):
        return "17:00-21:00"
    return None


_DAY_ABBR = {**{d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])},
             "mon": 0, "tue": 1, "tues": 1, "wed": 2, "weds": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_OCCASION = re.compile(r"\b(?:on |at |for |from )?(?:(?:my|our|his|her|their|your|the|\w+ s) )?(?:\d+(?:st|nd|rd|th)? |first |last )?"
                       r"(?:birthday|bday|anniversary|wedding day|wedding|graduation)(?: party| celebration| dinner)?\b")


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
    if t in ("last", "latest", "most recent", "recent", "newest", "last one"):
        return None                              # "the last video I took": an order, not a date range (handled in ground)
    mo = re.fullmatch(r"(?:last |this past |past |this )?(" + "|".join(_MONTH_NAMES) + r")(?: (\d{4}))?", t)
    if mo:                                       # "August" / "last August": the most recent August that has started
        m_i = _MONTH_NUM[mo.group(1)]
        y = int(mo.group(2)) if mo.group(2) else (today.year if m_i <= today.month else today.year - 1)
        if not mo.group(2) and t.startswith("last ") and m_i == today.month:
            y -= 1
        return iso(date(y, m_i, 1), date(y + (m_i == 12), m_i % 12 + 1, 1))
    rng = re.fullmatch(r"(?:between )?(" + "|".join(_MONTH_NAMES) + r")(?: (\d{4}))? (?:to|till|til|until|through|thru|and|"
                       r"-) (" + "|".join(_MONTH_NAMES) + r")(?: (\d{4}))?(?: (?:of )?(last|this) year)?", t)
    if rng:                                      # "may till july last year", "january and march of this year"
        m1, m2 = _MONTH_NUM[rng.group(1)], _MONTH_NUM[rng.group(3)]
        y2 = int(rng.group(4) or rng.group(2) or 0) or (today.year - 1 if rng.group(5) == "last" else today.year if
                                                         rng.group(5) == "this" else today.year - (m2 > today.month))
        y1 = int(rng.group(2)) if rng.group(2) else (y2 if m1 <= m2 else y2 - 1)
        return iso(date(y1, m1, 1), date(y2 + (m2 == 12), m2 % 12 + 1, 1))
    wk = re.fullmatch(r"week (?:of|around) (.+)", t)
    if wk:                                       # "the week of Christmas": Monday-Sunday around that day
        r0 = resolve_relative(wk.group(1), today)
        if r0:
            d0 = date.fromisoformat(r0[0]); m0 = d0 - timedelta(days=d0.weekday())
            return iso(m0, m0 + timedelta(days=7))
    for names, (hm, hd) in _HOLIDAYS:
        if re.fullmatch(r"(?:last |this past |this )?(?:" + names + r")(?: (?:morning|day|eve|night|party|dinner|celebration|"
                        r"fireworks|holidays?))*", t):
            y = today.year if (hm, hd) <= (today.month, today.day) else today.year - 1
            if hm == 11 and hd == 0:             # Thanksgiving: 4th Thursday of November
                d0 = date(y, 11, 1); th = d0 + timedelta(days=(3 - d0.weekday()) % 7 + 21)
                if th > today:
                    d0 = date(y - 1, 11, 1); th = d0 + timedelta(days=(3 - d0.weekday()) % 7 + 21)
                return iso(th, th + d1)
            return iso(date(y, hm, hd), date(y, hm, hd) + d1)
    ago = re.fullmatch(r"(\d+|" + "|".join(_NUM) + r"|a) (day|week|month|year)s? ago", t)
    if ago:                                      # "two years ago" -> that calendar year (not one day)
        n = int(ago.group(1)) if ago.group(1).isdigit() else _NUM.get(ago.group(1), 1)
        if ago.group(2) == "day":
            return iso(today - timedelta(days=n), today - timedelta(days=n) + d1)
        if ago.group(2) == "week":
            m0 = today - timedelta(days=today.weekday() + 7 * n)
            return iso(m0, m0 + timedelta(days=7))
        if ago.group(2) == "month":
            y, m_ = divmod(today.year * 12 + today.month - 1 - n, 12)
            return iso(date(y, m_ + 1, 1), date(y + (m_ == 11), (m_ + 1) % 12 + 1, 1))
        return iso(date(today.year - n, 1, 1), date(today.year - n + 1, 1, 1))
    se = re.fullmatch(r"this (summer|spring|fall|autumn)", t)
    if se:                                       # "this summer" said in October = this year's (planner gave last year's)
        m1 = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9}[se.group(1)]
        y = today.year if (m1, 1) <= (today.month, today.day) else today.year - 1
        return iso(date(y, m1, 1), date(y, m1 + 3, 1))
    if t in ("last quarter", "previous quarter", "this quarter"):
        q0 = date(today.year, 3 * ((today.month - 1) // 3) + 1, 1)
        if t == "this quarter":
            return iso(q0, today + d1)
        y, m_ = (q0.year, q0.month - 3) if q0.month > 3 else (q0.year - 1, 10)
        return iso(date(y, m_, 1), q0)
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
    m = re.fullmatch(r"(?:last |past |this past |on )?(" + "|".join(sorted(_DAY_ABBR, key=len, reverse=True)) +
                     r")(?: (?:morning|afternoon|evening|night))?", t)
    if m:                                        # "Tuesday" / "last thurs" / "last Friday night": the most recent one before today
        back = (today.weekday() - _DAY_ABBR[m.group(1)]) % 7 or 7
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


def ground(P: Plan, message: str, history: list[str], today: date | None = None, owner: str = "me", people=None) -> Plan:
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
    rel = sorted(set(m.group(0) for m in _REL_PHRASE.finditer(message)) |
                 set(m.group(0) for m in _MONTH_RANGE.finditer(message)))
    rel = [x for x in rel if not any(x != y and x.lower() in y.lower() for y in rel)]   # keep the longest phrases
    for a in P.albums:
        if not a.time_phrase and (a.date_from or a.date_to) and len(span) == 1:
            a.time_phrase = span[0]       # planner set dates but no phrase (planner test: "Dad from 2015 to 2018")
        elif (not a.time_phrase or _norm(a.time_phrase) not in _norm(said)) and (a.date_from or a.date_to) and len(rel) == 1 and \
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
        if re.search(r"(?i)\b" + re.escape(_norm(tp)) + r"\s+(tree|trees|lights|decorations|ornaments|costumes?|eggs?|cards?|"
                     r"sweaters?|markets?|movies?|songs?|cookies|wreath|stockings?|presents|gifts)\b", _norm(said)) and \
                re.fullmatch(r"(?i)(christmas|xmas|halloween|easter|thanksgiving|valentine s|valentines)", _norm(tp)):
            P.notes = (P.notes + f" [no dates: '{tp}' names a thing here, not a time]").strip()
            a.date_from = a.date_to = a.time_phrase = None      # "the christmas tree pics" (fuzz set 1)
            continue
        if re.search(r"(?i)\bthe last (night|day|morning|evening|afternoon)\b", said) and \
                re.fullmatch(r"(?:on )?(?:the )?last (night|day|morning|evening|afternoon)", _norm(tp)):
            P.notes = (P.notes + f" [no dates: 'the last {_norm(tp).split()[-1]}' is a point in a trip, not yesterday]").strip()
            a.date_from = a.date_to = a.time_phrase = None      # "the spicy ramen I ate on the last night" (fuzz set 8)
            continue
        if _norm(tp) in ("last", "latest", "most recent", "recent", "newest", "last one"):
            a.date_from = a.date_to = a.time_phrase = None      # "the last video I took": an order, not a date
            continue
        rest = re.sub(r"\s+", " ", _OCCASION.sub(" ", _norm(tp))).strip()
        if rest != _norm(tp) and not (rest and _CAL.search(rest)):
            # "on my birthday", "the 50th anniversary celebration": the planner cannot know the date (it guessed today)
            P.notes = (P.notes + f" [I don't know the date of '{tp}'; no date limit]").strip()
            a.date_from = a.date_to = a.time_phrase = None; continue
        r = resolve_relative(rest, today) if rest != _norm(tp) else resolve_relative(tp, today)
        if r and rest != _norm(tp):
            P.notes = (P.notes + f" [I don't know the date of the occasion in '{tp}'; searching {rest}]").strip()
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
    first_person = bool(re.search(r"(?i)\b(i|me|my|mine|myself|i'm|im|i've|ive|i'd|we|us|our|selfie|selfies)\b", said))
    for a in P.albums:   # a person must be NAMED in the conversation ("my sister" became Sara, "we" became Dad: fuzz 10-03)
        is_owner = _norm(a.person or "") in ("me", "i", "myself") or set(_norm(a.person or "x").split()) <= set(_norm(owner or "me").split())
        if a.person and is_owner and not first_person and not any(
                t in said_words for t in _norm(owner or "").split() if len(t) > 2):
            P.notes = (P.notes + f" [person removed from '{a.name}': the request does not mention you]").strip()
            _drop_name_questions(a, a.person, owner); a.person = None
        if a.person and not is_owner and not any(
                w.startswith(t) or t.startswith(w) for t in _norm(a.person).split() if len(t) > 2
                for w in said_words if len(w) > 2):
            P.notes = (P.notes + f" [person '{a.person}' removed from '{a.name}': not named in the request. Who is it? Say "
                                 f"their name, or add reference photos]").strip()
            _drop_name_questions(a, a.person, owner); a.person = None
    known = {_norm(x) for x in (people or [])} | {_norm(owner or "me"), "me", "i", "myself"}
    for a in P.albums:   # "me with Pierce": Pierce must be a known person to be face-matched; else ask (unknown_people)
        keep = []
        sw = _norm(said).split()
        for w in a.with_people:
            if not any(x.startswith(t) or t.startswith(x) for t in _norm(w).split() if len(t) > 2 for x in sw if len(x) > 2):
                P.notes = (P.notes + f" ['{w}' dropped from '{a.name}': not named in the request]").strip()
                continue          # "us all together" became with Dad, Mom, Sara and Ali (fuzz 10-03)
            if _norm(w) in known or any(_norm(w) in k or k in _norm(w) for k in known if len(k) > 2):
                keep.append(w)
            elif _norm(w):
                P.unknown_people.append(w)
                P.notes = (P.notes + f" ['{w}' is not known yet, so this album does not require them]").strip()
        a.with_people = [w for w in keep if _norm(w) != _norm(a.person or "")]
        if not a.person and a.with_people:
            a.person = a.with_people.pop(0)
    P.unknown_people = [u for u in dict.fromkeys(P.unknown_people)   # only people actually mentioned and not known yet
                        if _norm(re.sub(r"(?i)^my\s+", "", u)) and _norm(u) not in known
                        and _norm(re.sub(r"(?i)^my\s+", "", u)) not in known
                        and all(t in _norm(said).split() for t in _norm(re.sub(r"(?i)^my\s+", "", u)).split())]
    tod = resolve_time_of_day(message) or resolve_time_of_day(said)
    for a in P.albums:   # hours come from the words, never from the model; the model only says WHICH album they apply to
        if tod and (a.time_of_day or len(P.albums) == 1):
            if a.time_of_day != tod:
                P.notes = (P.notes + f" [time of day {tod}, local time where each photo was taken]").strip()
            a.time_of_day = tod
            if tod[:5] > tod[6:] and a.date_to:   # wraps midnight: "11pm to 2am last night" also needs the next morning
                a.date_to = (date.fromisoformat(a.date_to) + timedelta(days=1)).isoformat()
        elif not tod:
            a.time_of_day = None
    strip_identity_conditions(P); fix_red_box(P)
    for a in P.albums:   # "Is this photo taken in Japan?" with Japan in the request: GPS answers that, the judge guesses
        for f in ("judge_question", "filter_question"):    # (planner eval 10-03: "my 10 best photos from Japan")
            m = re.fullmatch(r"(?:Is|Was) (?:this|the) (?:photo|image|picture|video|clip|item)(?: taken| shot| from)? "
                             r"(?:in|at|from) ([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)\??", (getattr(a, f) or "").strip())
            if m and _norm(m.group(1)) in _norm(said) and (not a.place or _norm(a.place) == _norm(m.group(1))):
                a.place = m.group(1); setattr(a, f, None)
    for a in P.albums:
        if a.anchor and a.anchor.judge_question and a.window in (None, "same_place", "same_event"):
            # "Is this Hawaii?" as the moment: GPS answers it. Not for "the WEEK I went to the Grand Canyon" (food that
            # week, anywhere): planner eval 10-03 caught my first version turning that into "food at the Grand Canyon"
            m = re.fullmatch(r"(?:Is|Was) (?:this|the) (?:(?:photo|image|picture|video)(?: taken| shot)? (?:in|at|from) )?"
                             r"(?:the )?([A-Z][\w'-]*(?: [A-Z][\w'-]*)*)\??", a.anchor.judge_question.strip())
            if m and _norm(m.group(1)) in _norm(said) and m.group(1).split()[0] not in ("A", "An", "The") and \
                    (not a.place or _norm(a.place) == _norm(m.group(1))):
                a.place = m.group(1); a.anchor = None; a.window = None
        for f in ("judge_question", "filter_question", "exclude_question"):
            # "Is this a video?": the judge sees ONE frame and may say no; media is decided by the file type
            if re.fullmatch(r"(?i)(?:is|was) (?:this|it) (?:a |an )?(?:video|photo|picture|image|clip|video clip|video file|"
                            r"photo file|file|recording|home video|movie)\??", (getattr(a, f) or "").strip()):
                setattr(a, f, None)
    wm = re.search(r"(?i)\b(?:the|that) (day|week|weekend|month|year) (?:when )?(?:i|we) (?:went|was|were|visited|flew|drove|"
                   r"traveled|travelled|stayed|got) (?:to|at|in) (?:the )?([\w' -]+?)(?=[,.!?]| and | no | without | but |$)", said)
    for a in P.albums:   # "food from the WEEK I went to the Grand Canyon" planned as place=Grand Canyon (= food AT the
        # canyon): the place is the moment, the window is the week (planner eval 10-03, round 7)
        if wm and a.place and not a.anchor and _norm(a.place) == _norm(wm.group(2)):
            w = {"day": "same_day", "week": "same_week", "weekend": "same_week", "month": "same_month",
                 "year": "same_year"}[wm.group(1).lower()]
            a.anchor = Step(looks=[a.place], judge_question=f"Does this photo show {a.place}?"); a.window = w
            a.place = None
    if len(P.albums) > 1:   # "march through june" as 4 month albums: April/May lost their dates -> the whole library
        keep = [a for a in P.albums if a.person or a.judge_question or a.looks or a.date_from or a.date_to or a.place
                or a.anchor or a.media != "any" or a.filter_question]
        if keep and len(keep) < len(P.albums):
            P.notes = (P.notes + f" [{len(P.albums) - len(keep)} album(s) with no condition left out]").strip()
            P.albums = keep
    merged = []   # the same album twice, once for photos and once for videos ("also videos of her") -> one, media any
    for a in P.albums:
        twin = next((b for b in merged if {a.media, b.media} == {"photo", "video"} and
                     a.model_dump(exclude={"name", "media", "judge_question", "filter_question", "exclude_question"}) ==
                     b.model_dump(exclude={"name", "media", "judge_question", "filter_question", "exclude_question"}) and
                     all(_norm(re.sub(r"(?i)\b(photo|video|image|picture|clip)\b", "x", getattr(a, f) or "")) ==
                         _norm(re.sub(r"(?i)\b(photo|video|image|picture|clip)\b", "x", getattr(b, f) or ""))
                         for f in ("judge_question", "filter_question", "exclude_question"))), None)
        if twin is not None:
            twin.media = "any"
            P.notes = (P.notes + f" ['{a.name}' merged into '{twin.name}': same search, photos and videos]").strip()
        else:
            merged.append(a)
    P.albums = merged
    for a in P.albums:
        if a.judge_question and a.exclude_question and _norm(a.judge_question) == _norm(a.exclude_question):
            a.judge_question = None; a.looks = []   # "all photos that week, excluding X" (DISBench q4 asked X twice)
        if a.anchor and a.window not in ("same_day", "same_week", "same_month", "same_year", "same_event", "same_place",
                                         "before", "after", "since", "until") \
                and not re.fullmatch(r"(days|minutes)_(before|after):\d+", a.window or ""):
            a.window = "same_event"     # an invented window ("same_year") would otherwise mean "the whole library"
        if a.anchor:     # the window is what the words say ("the week ..." -> same_week), when they say it
            if not re.fullmatch(r"(days|minutes)_(before|after):\d+|before|after|since|until", a.window or ""):   # explicit
                for w, pat in _WINDOW_WORDS:
                    if re.search(pat, said, re.I):
                        a.window = w; break
            if a.place and a.anchor and _norm(a.place) in _norm(" ".join(a.anchor.looks + [a.anchor.judge_question or ""])):
                a.place = None   # the place IS the anchor moment, not a filter on what to find
        if a.window and not a.anchor:
            a.window = None
        if a.until and not a.anchor:   # "before B" alone is anchor B + window "before"
            a.anchor, a.window, a.until = a.until, "before", None
        if a.until and a.window not in ("after", "since"):
            a.window = "after" if a.window in (None, "same_event", "same_day") else "since"
        if a.anchor and re.fullmatch(r"(days|minutes)_(before|after):\d+|before|after|since|until", a.window or "") and \
                not re.search(r"(?i)\b(before|after|prior|earlier|later|following|leading up|since|until)\b", said):
            a.window = "same_event"   # "the family reunion last weekend" got days_before:7 (fuzz round 6)
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
            raw = json.loads(m.group(0))
            for al in raw.get("albums", []) if isinstance(raw, dict) else []:
                an = al.get("anchor") if isinstance(al, dict) else None
                if isinstance(an, dict) and not an.get("judge_question"):   # moment with no question (fuzz 10-03: crash)
                    lk = [x for x in an.get("looks") or [] if x]
                    al["anchor"] = dict(an, judge_question=f"Does this photo show {lk[0]}?") if lk else None
            P = Plan.model_validate(raw)
            if not P.albums:
                raise ValueError("zero albums; the library can contain anything, return at least one album")
            problem = _unanswerable(P, list(people or []) + [owner or ""], " \n ".join(history + [message]))
            if problem:
                fallback = P          # well-formed: usable if every retry repeats the problem
                raise ValueError(problem)
            return ground(P, message, history, today, owner, people)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    if fallback is not None:      # degrade instead of failing (DISBench q30): drop the part one photo cannot answer
        return ground(_drop_unanswerable(fallback, " \n ".join(history + [message]), list(people or []) + [owner or ""]),
                      message, history, today, owner, people)
    raise ValueError(f"planner failed: {last}")


# the judge is a stranger looking at one photo: "the house we bought", "the concert we went to", "you and Reza" (fuzz 10-03:
# ~9/117 plans) need knowledge it does not have
def _personal(q: str) -> bool:
    return bool(re.search(r"\bI\b", q) or re.search(r"(?i)\b(me|my|mine|myself|we|us|our|ours|ourselves|you|your|yours)\b|"
                                                     r"\bthe (user|owner)\b", q))


_NAMED = re.compile(r"(?i)\b(person|man|woman|boy|girl|child|kid|baby|guy|lady|someone|friend)\s+(named|called)\b")


# file/camera details are not in the pixels (fuzz set 4, a photographer: ~10/117 questions about RAW, fps, lens, audio,
# location tags, ratings, filters, 4K); the judge would answer them at random
_METADATA = re.compile(r"(?i)\b(raw (file|version|format)|\d+ ?fps|frames per second|\d+ ?mm\b|telephoto|wide[- ]angle lens|"
                       r"lens\b(?! flare)|audio|sound track|music track|location tag|geotag|tagged|rated|rating|favou?rited|"
                       r"filter applied|backup|file\b|\d ?k\b|1080p|720p|high[- ]res|resolution|timestamp|"
                       r"seconds long|minutes long|duration|trimmed|time (is )?between|time of day|between \d{1,2} ?(am|pm)|shot on (a|my) phone|edited|unedited|exif)\b")


# events no single photo shows: "the moment the person passed away", "before he got sick", "after it was sold"
_INVISIBLE_EVENT = re.compile(r"(?i)\b(pass(?:ed|es|ing)? away|died|dies|dying|death|(?:got|get|gets|getting|became|becoming)"
                              r" (?:sick|ill)|fell ill|falling ill|(?:was|got|being|getting) sold|broke up|breaking up|"
                              r"got divorced|getting divorced|moved out|moving out|retired|retiring)\b")


def _bad_q(q: str | None) -> bool:
    return bool(q) and (bool(_RELATIONAL.search(q)) or _personal(q) or bool(_NAMED.search(q)) or bool(_METADATA.search(q))
                        or bool(_INVISIBLE_EVENT.search(q))
                        or bool(_IS_NAME.search(q)))


# "Is this person Jay?", "Is this Uncle Harry?", "Is this the brother?": who someone is, which the judge cannot know
# (fuzz sets 3, 10)
_REL = (r"brother|sister|mom|mother|dad|father|grandma|grandmother|grandpa|grandfather|uncle|aunt|cousin|son|daughter|"
        r"wife|husband|niece|nephew|grandson|granddaughter|friend|boyfriend|girlfriend|partner")
_IS_NAME = re.compile(r"\b(?:[Ii]s|[Aa]re) (?:this|that|the) (?:person|man|woman|guy|girl|boy|kid|child)\s+[A-Z][a-z]+\b|"
                      r"\b(?:[Ii]s|[Aa]re) (?:this|that|it) (?:(?:my|our|the|your) )?(?:(?:" + _REL + r")(?: [A-Z][a-z]+)?|"
                      r"the person in the (?:photo|image|picture|video))\??$|"
                      r"(?i:\b(?:uncle|aunt|grandma|grandpa|cousin)\s+)[A-Z][a-z]+")
# hair/looks the request never mentioned, written for an unknown relative ("my sister" -> "a woman with long hair")
_INVENTED_LOOK = re.compile(r"(?i)\b(long|short|gray|grey|white|blonde|blond|brown|black|dark|red|curly|straight) hair\b")


# phrases from the prompt's own examples; seen copied into plans for unrelated requests ("food we ate at that little
# Italian place" -> "Is this a slice of bread?" 3/117; "you and me at the beach" -> "a man with a heavy build": fuzz 10-03)
_EXAMPLE_LEAKS = [("heavy build", r"heav|weight|fat|big|overweight|chubby|build"), ("round face", r"round|face|heav"),
                  ("slice of bread", r"bread|toast|sandwich|loaf"), ("grand canyon", r"grand canyon"),
                  ("burger", r"burger"), ("sandwich", r"sandwich")]


def _leak(P: Plan, said: str) -> str | None:
    for a in P.albums:
        texts = a.looks + [a.judge_question or "", a.filter_question or "", a.exclude_question or ""] + \
            ((a.anchor.looks + [a.anchor.judge_question or ""]) if a.anchor else [])
        for t in texts:
            for phrase, ok in _EXAMPLE_LEAKS:
                if phrase in t.lower() and not re.search(ok, said, re.I):
                    return phrase
    return None


def _unanswerable(P: Plan, names: list[str] | None = None, said: str = "") -> str | None:
    if said and not re.search(r"(?i)hair", said):
        for a in P.albums:
            for q in [a.judge_question or "", a.filter_question or ""] + a.looks:
                if _INVENTED_LOOK.search(q):
                    return (f"'{q}' describes hair the request never mentioned: never invent what a person looks like; "
                            "ask only about what the request says")
    leak = _leak(P, said) if said else None
    if leak:
        return (f"'{leak}' comes from the instructions' examples, not from the request: describe only what THIS request "
                "asks for")
    for a in P.albums:   # questions the judge cannot answer from ONE photo (DISBench q3/q17/q30): ask again
        own = set(_norm(a.person or "").split())
        toks = {t for n in (names or []) for t in _norm(n).split() if len(t) > 2 and t not in own and t != "me"}
        for q in (a.judge_question, a.filter_question, a.exclude_question):
            hit = [t for t in toks if q and re.search(r"\b" + re.escape(t) + r"\b", _norm(q))]
            if hit:   # "Is this a family member (Reza, Dad, Mom, ...)?": the judge cannot recognize people (fuzz 10-03)
                return (f"'{q}' names {', '.join(sorted(hit))}; the judge cannot recognize people. Put one person in "
                        "'person' (faces are matched separately) and ask only about what is visible")
        for q in [a.judge_question] + ([a.anchor.judge_question] if a.anchor else []) + \
                ([a.until.judge_question] if a.until else []) + [a.exclude_question, a.filter_question]:
            if q and _INVISIBLE_EVENT.search(q):
                return (f"'{q}' asks about an event no photo shows (someone dying, getting sick, a sale): leave it out; "
                        "search only for what is visible")
            if q and _RELATIONAL.search(q):
                return (f"'{q}' refers to another photo or moment; the judge sees ONE photo at a time. "
                        "Put the moment in 'anchor'/'window' and ask only about what is visible in this photo")
            if q and _METADATA.search(q):
                return (f"'{q}' asks about the file or camera (RAW, lens, fps, audio, tags, resolution); the judge sees only "
                        "the picture. Ask only about what is visible")
            if q and (_NAMED.search(q) or _IS_NAME.search(q)):
                return (f"'{q}' asks for a name; the judge cannot know names. Put a known person in 'person' or ask "
                        "only about what is visible")
            if q and _personal(q):
                return (f"'{q}' needs to know the owner (I/me/my/we/our/you); the judge is a stranger seeing ONE photo. "
                        "Ask only about what is visible (use 'the person in the red box' when 'person' is set)")
        if a.anchor and a.judge_question and _norm(a.judge_question) == _norm(a.anchor.judge_question):
            return (f"album '{a.name}': judge_question repeats the anchor question; judge_question must "
                    "describe what to find INSIDE the moment, not the moment itself")
    return None


def _drop_name_questions(a, person, owner="me"):
    """The album's person was removed: a question naming them ("Is this Reza?") would now go to the judge, which cannot
    recognize anyone (fuzz round 6). Drop those questions."""
    names = [person] + ([owner] if _norm(person or "") in ("me", "i", "myself") else [])
    for f in ("judge_question", "filter_question", "exclude_question"):
        if _names_in(getattr(a, f) or "", names):
            setattr(a, f, None)


def _names_in(q: str, names, person=None) -> list[str]:
    if q and re.search(r"(?i)\b(screenshot|chat|message|text|texts|written|says|reads|label|sign|caption|contact|"
                       r"conversation|email|name)\b", q):
        return []     # a name you can READ in the picture is visible ("a chat with Mom": fuzz round 6)
    own = set(_norm(person or "").split())
    toks = {t for n in (names or []) for t in _norm(n).split() if len(t) > 2 and t not in own and t != "me"}
    return [t for t in toks if q and re.search(r"\b" + re.escape(t) + r"\b", _norm(q))]


_EVERYTHING = re.compile(r"(?i)\b(photos|pics|pictures|videos|clips|images|everything|anything|all|memories|footage|album|"
                         r"shots)\b[^.?!]{0,40}?\b(from|at|during|on)\b|\b(trip|vacation|holiday|weekend|day|night)\b")


def _drop_unanswerable(P: Plan, said: str = "", names: list[str] | None = None) -> Plan:
    for a in P.albums:   # copied prompt examples go first, so nothing below rebuilds a question from them
        if not re.search(r"(?i)hair", said):
            a.looks = [x for x in a.looks if not _INVENTED_LOOK.search(x)]
            if _INVENTED_LOOK.search(a.judge_question or ""):
                P.notes = (P.notes + f" [dropped '{a.judge_question}': it guessed what the person looks like]").strip()
                a.judge_question = None
        bad = [ph for ph, ok in _EXAMPLE_LEAKS if not re.search(ok, said, re.I)]
        a.looks = [x for x in a.looks if not any(ph in x.lower() for ph in bad)]
        if a.anchor:
            a.anchor.looks = [x for x in a.anchor.looks if not any(ph in x.lower() for ph in bad)]
        for f in ("judge_question", "filter_question", "exclude_question"):
            if any(ph in (getattr(a, f) or "").lower() for ph in bad):
                setattr(a, f, None)
    """Last resort: replace a question one photo cannot answer by a plain visual question built from 'looks', and
    say so in the notes (shown to the person), instead of refusing the whole request."""
    for a in P.albums:
        def plain(q, looks):
            lk = [re.sub(r"(?i)\s+(named|called)\s+\w+", "", x) for x in looks or []]
            lk = [x for x in lk if not _bad_q(f"Does this photo show {x}?") and not _names_in(x, names, a.person)]
            return f"Does this photo show {lk[0]}?" if lk else None
        for f in ("judge_question", "filter_question", "exclude_question"):
            if _names_in(getattr(a, f) or "", names, a.person):
                P.notes = (P.notes + f" [dropped '{getattr(a, f)}': the judge cannot recognize people by name]").strip()
                # the main question falls back to the looks (None would mean "everything in scope")
                setattr(a, f, plain(None, a.looks) if f == "judge_question" else None)
        if a.judge_question and a.anchor and _norm(a.judge_question) == _norm(a.anchor.judge_question or "") \
                and not (said and _EVERYTHING.search(said)):
            # "that pizza we ate last friday": the thing itself is wanted -> keep the question, drop the moment
            P.notes = (P.notes + " [searching for the thing itself, not everything around it]").strip()
            a.anchor = None; a.window = None
        if a.judge_question and a.anchor and _norm(a.judge_question) == _norm(a.anchor.judge_question or ""):
            # "photos from the wedding": the moment itself was asked twice -> everything inside it (the old fallback,
            # "Does this photo show <first look>?", kept only bride-and-groom photos: fuzz 10-03, ~12/117)
            P.notes = (P.notes + " [showing everything from that moment]").strip()
            a.judge_question = None; a.looks = []
        if a.judge_question and _bad_q(a.judge_question):
            P.notes = (P.notes + f" [could not express '{a.judge_question}' as a question about one photo; "
                                 f"searching for what it looks like instead]").strip()
            a.judge_question = plain(a.judge_question, a.looks)
        if a.until and (_bad_q(a.until.judge_question) or _names_in(a.until.judge_question or "", names, None)):
            P.notes = (P.notes + f" [dropped the end moment '{a.until.judge_question}': no photo can show it]").strip()
            a.until = None
        if a.anchor and _INVISIBLE_EVENT.search(a.anchor.judge_question or ""):
            P.notes = (P.notes + f" [dropped the moment '{a.anchor.judge_question}': no photo can show it]").strip()
            a.anchor = None; a.window = None
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
        if a.until and len(scope):
            # "after the driver photo but before we reached the Citadel": keep the span up to the first B photo AFTER
            # the first A photo (DISBench q45)
            ua = Album(name=f"{a.name} (until)", looks=a.until.looks, judge_question=a.until.judge_question,
                       date_from=a.date_from, date_to=a.date_to, place=a.place)
            ur = run_album(idx, ua, enc, judge, None, th=Thresholds(**{**th.__dict__, "stream": False}))
            tt = pd.to_datetime(idx.items.taken, utc=True, errors="coerce", format="ISO8601")
            start = tt.iloc[top].min()
            ends = tt.iloc[ur.returned.item_row.to_numpy()] if len(ur.returned) else tt.iloc[0:0]
            ends = ends[ends > start]
            if len(ends):
                cut = ends.min()
                scope = scope[(tt.iloc[scope] < cut).to_numpy()]
                trace.update(until_found=len(ur.returned), until_cut=str(cut), window_items=len(scope))
            else:
                trace.update(until_found=len(ur.returned), until_cut=None)
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
    with_ok = None        # "me with Pierce": items where each other person's face also matches
    for w in a.with_people:
        _, wrefs, _, _ = refs_for(w)
        if isinstance(wrefs, SubjectRefs) or wrefs is None or not len(wrefs):
            continue
        from .people import item_person_scores
        ws, _ = item_person_scores(sub, wrefs)
        ok = set(sub.items.item_id.to_numpy()[ws >= th.person_accept])
        with_ok = ok if with_ok is None else with_ok & ok
    for r in gen:
        t = dict(trace)
        if with_ok is not None and len(r.returned):
            keep = r.returned.item_id.isin(with_ok).to_numpy()
            t["without_the_other_people"] = int((~keep).sum()); r.returned = r.returned[keep]
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
        n_drop = t.get("excluded", 0) + t.get("removed_by_you", 0) + t.get("filtered_out", 0) + \
            t.get("without_the_other_people", 0)
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
