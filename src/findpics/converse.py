"""One path for everything the person types: the first request and every follow-up.

Each message -> the planner sees the conversation so far + the current plan -> returns the WHOLE updated plan
(same schema every time) -> the engine reruns it. There is no separate "edit" system and no mode flags:
  - two-step requests ("photos from the day I ..., without ...") are an optional anchor/window/exclude per album,
    filled in by the planner when the sentence needs them and left null otherwise;
  - "look harder" / "check every photo" is grounded words (effort_phrase) that switch on exhaustive judging;
  - follow-ups ("only the ones at night", "also add 2019", "drop the group shots") change the plan, then it reruns.
Reruns are cheap because the judge's answers are cached per (image, question) in the session folder.
Taps on the review page (this photo is wrong) are not language: they are stored as per-item overrides and applied
after every rerun.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from . import store
from .agent import Step, window_rows
from .engine import Thresholds, _judge_rows, make_exclusive, run_album
from .planner import AlbumSpec, _norm, fix_red_box, ground_dates, ground_place, strip_identity_conditions


class Album(AlbumSpec):
    anchor: Step | None = None              # what identifies the moment, when the album is defined by one
    window: str | None = None               # same_day | same_week | same_event | same_place
    exclude_question: str | None = None     # items the judge says YES to are dropped


class Plan(BaseModel):
    albums: list[Album]
    effort_phrase: str | None = None        # exact words asking for a thorough search ("look harder")
    notes: str = ""


TEMPLATE = """You maintain a JSON search plan over a person's own photo library during a conversation.
Today's date is {today}. The library owner is {owner}. Known people in the library: {people}.

Return ONLY JSON:
{{"albums": [{{"name": str, "person": str|null, "looks": [str], "avoid": [str], "judge_question": str|null,
   "anchor": {{"looks": [str], "judge_question": str}}|null, "window": "same_day"|"same_week"|"same_event"|"same_place"|null,
   "exclude_question": str|null, "place": str|null, "time_phrase": str|null,
   "date_from": "YYYY-MM-DD"|null, "date_to": "YYYY-MM-DD"|null, "media": "photo"|"video"|"any",
   "want": "all"|"best", "max_items": int|null}}], "effort_phrase": str|null, "notes": str}}

Rules:
- One album per group the person asks for. If they ask for two categories, make two albums.
- "person": one of the known people, "me" for the owner, or null if no specific person.
- "looks": 1-4 short, concrete VISUAL descriptions of the CONDITION asked for (e.g. "a man with a heavy build and round
  face", "a slice of bread"). Identity is handled by face matching: never describe what a person looks like in general.
  If the request names a person but gives no condition ("every photo of Dad"), "looks" is [] and "judge_question" null.
- "judge_question": a yes/no question about ONE image. Say "the person in the red box" only when "person" is set.
- Indirect moments ("the day when...", "the week when...", "during the trip where...", "at the place where..."):
  "anchor" describes what is visible in photos of that moment, "window" how far around it to look, and looks/
  judge_question describe what to find inside that window. Otherwise "anchor" and "window" are null.
- Exclusions ("without...", "excluding...", "no ..."): a yes/no question about the excluded thing in
  "exclude_question"; do not mention it in looks or judge_question.
- Dates: "time_phrase" = the exact words that constrain THIS album's time, else null (then both dates null). Convert
  with today's date. date_to is EXCLUSIVE: "the 1990s" -> 1990-01-01..2000-01-01; "in 2019" -> 2019-01-01..2020-01-01.
- "place": exact words naming a geographic place (city, region, country, landmark area), else null. "beach" is a look.
- media: "video" only if they ask only for videos. want: "best" if they ask for the best/top items, else "all".
- "effort_phrase": the exact words of the NEW message if it asks to search harder or check everything ("look harder",
  "check every photo", "you missed some"), else null.
{conversation}
JSON:"""


def build_prompt(message: str, history: list[str], current: Plan | None, owner="me", people=None, today=None) -> str:
    if current is None:
        conv = f"\nRequest: {message.strip()}"
    else:
        conv = ("\nThis is a follow-up. Earlier messages:\n" + "\n".join(f"- {m}" for m in history) +
                f"\nCurrent plan:\n{current.model_dump_json(exclude={'notes'})}\n"
                "Return the WHOLE updated plan: keep everything the new message does not change; change only what it asks."
                f"\nNew message: {message.strip()}")
    return TEMPLATE.format(today=(today or date.today()).isoformat(), owner=owner,
                           people=", ".join(people or []) or "unknown", conversation=conv)


def ground(P: Plan, message: str, history: list[str]) -> Plan:
    """Code-enforced checks (same as the one-shot planner), against EVERYTHING the person has typed so far, so an
    album's dates/place from message 1 survive message 3; effort only from the new message (it is per-turn)."""
    said = " \n ".join(history + [message])
    ground_dates(P, said); ground_place(P, said); strip_identity_conditions(P); fix_red_box(P)
    for a in P.albums:
        if a.window and not a.anchor:
            a.window = None
        if a.anchor and a.anchor.judge_question:     # the anchor is about a moment, never a boxed person
            q = re.sub(r"(?i)\b(the|a) person in the red box\b", "someone", a.anchor.judge_question)
            a.anchor.judge_question = re.sub(r"(?i)\s*\bin the red box\b", "", q).strip()
        if a.exclude_question and not a.person:
            a.exclude_question = re.sub(r"(?i)\s*\bin the red box\b", "", a.exclude_question).strip()
    if P.effort_phrase and _norm(P.effort_phrase) not in _norm(message):
        P.effort_phrase = None
    return P


def plan_turn(message: str, llm, history: list[str] | None = None, current: Plan | None = None, owner="me",
              people=None, today=None, retries: int = 2) -> Plan:
    history = history or []
    prompt = build_prompt(message, history, current, owner, people, today)
    last = None
    for _ in range(retries + 1):
        out = llm(prompt if last is None else prompt + f"\n(Previous output was invalid: {last}. Return valid JSON only.)\nJSON:")
        try:
            m = re.search(r"\{.*\}", out, re.S)
            if not m:
                raise ValueError(f"no JSON object in planner output: {out[:200]!r}")
            return ground(Plan.model_validate(json.loads(m.group(0))), message, history)
        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            last = str(e)[:300]
    raise ValueError(f"planner failed: {last}")


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


def run_plan(idx, P: Plan, enc, judge, refs_for=None, th: Thresholds = Thresholds(), max_anchor: int = 5,
             exclude_ids: set | None = None) -> list:
    """Run every album; returns engine AlbumResults whose `returned`/`judged` rows point at the FULL index.
    refs_for(person) -> (name, refs, ref_face_row, n_tagged)."""
    if P.effort_phrase:      # thorough: the judge looks at every in-scope item
        th = Thresholds(**{**th.__dict__, "head_size": idx.n_items, "head_max": idx.n_items})
    results = []
    for a in P.albums:
        refs = ref_face = None
        if a.person:
            if refs_for is None:
                raise ValueError(f"album '{a.name}' needs reference faces for '{a.person}'")
            _, refs, ref_face, _ = refs_for(a.person)
        sub, trace = idx, {}
        if a.anchor:
            anc = Album(name=f"{a.name} (anchor)", looks=a.anchor.looks, judge_question=a.anchor.judge_question,
                        date_from=a.date_from, date_to=a.date_to, place=a.place, media=a.media)
            ar = run_album(idx, anc, enc, judge, None, th=th)
            top = ar.returned.sort_values("p_attr", ascending=False).head(max_anchor).item_row.to_numpy()
            scope = window_rows(idx, top, a.window)
            trace = dict(anchor_found=len(ar.returned), anchor_used=len(top), window=a.window, window_items=len(scope))
            sub = store.subset(idx, scope) if len(scope) < idx.n_items else idx
            spec = a.model_copy(update=dict(date_from=None, date_to=None, time_phrase=None, place=None))
            ref_face = None if sub is not idx else ref_face
        else:
            spec = a
        r = run_album(sub, spec, enc, judge, refs, ref_face_row=ref_face, th=th)
        if a.exclude_question and len(r.returned):
            pex = _judge_rows(sub, judge, r.returned.item_row.to_numpy(), r.returned.face_row.to_numpy(),
                              a.exclude_question, crop_person=bool(a.person))
            trace["excluded"] = int((pex >= th.judge_accept).sum())
            r.returned = r.returned[pex < th.judge_accept]
        if sub is not idx:   # map rows back to the full index
            row_of = {iid: i for i, iid in enumerate(idx.items.item_id)}
            for df in (r.returned, r.judged):
                if len(df) and "item_id" in df:
                    df["item_row"] = [row_of[i] for i in df.item_id]
        if exclude_ids:
            n0 = len(r.returned); r.returned = r.returned[~r.returned.item_id.astype(str).isin(exclude_ids)]
            if n0 - len(r.returned):
                trace["removed_by_you"] = n0 - len(r.returned)
        if trace:
            r.report += "\n  Steps: " + json.dumps(trace)
        r.trace = trace
        results.append(r)
    return make_exclusive(results)


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
