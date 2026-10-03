"""Multi-step search: anchor moment -> window -> target (+ exclusion). Same machinery will run refinement edits.

Real queries are often two-hop: "photos from the DAY/WEEK/EVENT when <anchor> happened, that show <target>,
excluding <thing>" (DISBench). A 9B local model is not a reliable free-form agent, so the plan has a fixed shape:
  anchor   : what identifies the moment (looks + yes/no question)          -> run a normal search, keep confident hits
  window   : "same_day" | "same_week" | "same_event" | "same_place" | null -> turn anchor hits into a set of items
  target   : what the user actually wants inside that window              -> normal search restricted to the window
  exclude  : optional yes/no question; items the judge says YES to are dropped
Events = bursts of photos with no gap longer than EVENT_GAP_H hours (standard time-gap segmentation).
"""
from __future__ import annotations

import json
import re
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import store
from .engine import Thresholds, _judge_rows, run_album
from .planner import AlbumSpec, _norm

EVENT_GAP_H = 3.0


class Step(BaseModel):
    looks: list[str] = Field(default_factory=list)
    judge_question: str


class MultiPlan(BaseModel):
    anchor: Step | None = None
    window: str | None = None            # same_day | same_week | same_event | same_place
    target: Step
    exclude_question: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    time_phrase: str | None = None
    place: str | None = None
    want: str = "all"


TEMPLATE = """You turn a request about someone's own photo library into a JSON search plan with up to two steps.
Today's date is {today}.

If the request refers to a moment indirectly ("the day when...", "the week when...", "during the trip where...",
"at the place where..."), split it:
  "anchor": what identifies that moment (something visible in the photos taken then),
  "window": "same_day" | "same_week" | "same_event" | "same_place",
  "target": what the person wants to find within that moment.
Otherwise "anchor" and "window" are null and everything goes in "target".
If the request excludes something ("excluding...", "without...", "no ..."), put a yes/no question about the excluded
thing in "exclude_question" and do NOT mention it in target.

Return ONLY JSON:
{{"anchor": {{"looks": [str], "judge_question": str}} | null, "window": str | null,
  "target": {{"looks": [str], "judge_question": str}}, "exclude_question": str | null,
  "time_phrase": str | null, "date_from": "YYYY-MM-DD" | null, "date_to": "YYYY-MM-DD" | null, "place": str | null,
  "want": "all" | "best"}}
"looks": 1-3 short visual descriptions. "judge_question": one yes/no question about ONE photo.
time_phrase/place: exact words from the request, else null (dates exclusive end, as in "the 1990s" -> 1990-01-01..2000-01-01).

Request: {request}
JSON:"""


def make_plan(request: str, llm, today: date | None = None, retries: int = 2) -> MultiPlan:
    prompt = TEMPLATE.format(today=(today or date.today()).isoformat(), request=request.strip())
    last = None
    for _ in range(retries + 1):
        out = llm(prompt if last is None else prompt + f"\n(Previous output invalid: {last}. JSON only.)\nJSON:")
        try:
            m = re.search(r"\{.*\}", out, re.S)
            P = MultiPlan.model_validate(json.loads(m.group(0)))
            req = _norm(request)   # same code-enforced grounding as the one-step planner
            if not P.time_phrase or _norm(P.time_phrase) not in req:
                P.date_from = P.date_to = P.time_phrase = None
            if P.place and _norm(P.place) not in req:
                P.place = None
            if P.window and not P.anchor:
                P.window = None
            for st in (P.anchor, P.target):        # no person/red box in multi-step plans (see planner.fix_red_box)
                if st and st.judge_question:
                    st.judge_question = re.sub(r"(?i)\s*\bin the red box\b", "", st.judge_question.replace("the person in the red box", "someone"))
            return P
        except Exception as e:
            last = str(e)[:200]
    raise ValueError(f"multi-step planner failed: {last}")


def events(taken: pd.Series, gap_h: float = EVENT_GAP_H) -> np.ndarray:
    """Event id per item: sort by time, start a new event after a gap > gap_h hours."""
    t = pd.to_datetime(taken, utc=True, errors="coerce", format="ISO8601")
    order = np.argsort(t.values)
    ts = t.values[order]
    gaps = np.r_[True, (np.diff(ts) / np.timedelta64(1, "h")) > gap_h]
    ev = np.empty(len(t), np.int64); ev[order] = np.cumsum(gaps)
    return ev


def window_rows(idx, anchor_rows, window: str | None) -> np.ndarray:
    it = idx.items
    if window is None or len(anchor_rows) == 0:
        return np.arange(idx.n_items)
    t = pd.to_datetime(it.taken, utc=True, errors="coerce", format="ISO8601")
    if window == "same_day":
        days = set(t.iloc[anchor_rows].dt.date)
        return np.where(t.dt.date.isin(days))[0]
    if window == "same_week":
        wk = t.dt.isocalendar(); key = list(zip(wk.year, wk.week))
        keys = {key[i] for i in anchor_rows}
        return np.array([i for i, k in enumerate(key) if k in keys])
    if window in ("same_month", "same_year"):
        key = t.dt.to_period("M" if window == "same_month" else "Y")
        keys = set(key.iloc[anchor_rows].dropna())
        return np.where(key.isin(keys))[0]
    m = re.fullmatch(r"days_(before|after):(\d+)", window or "")
    if m:   # offset windows: "the day after the wedding" = days_after:1; "the week before I moved" = days_before:7
        n = int(m.group(2)); days = pd.Series(sorted(set(t.iloc[anchor_rows].dt.normalize().dropna())))
        d = t.dt.normalize(); keep = np.zeros(len(t), bool)
        for a in days:
            lo, hi = (a - pd.Timedelta(days=n), a - pd.Timedelta(days=1)) if m.group(1) == "before" else \
                     (a + pd.Timedelta(days=1), a + pd.Timedelta(days=n))
            keep |= ((d >= lo) & (d <= hi)).to_numpy()
        return np.where(keep)[0]
    if window == "same_event":
        ev = events(it.taken)
        return np.where(np.isin(ev, ev[anchor_rows]))[0]
    if window in ("before", "after"):
        # inside the anchor's event, earlier / later than its FIRST photo there ("during the Paris trip, before the first
        # photo of a Van Gogh painting"; "photos of the cats after the cat tree was assembled"). Day windows cannot
        # express this: DISBench q20/q64/q85 scored F1 0.00-0.07 with same_event/days_* (10-03).
        ev = events(it.taken); keep = np.zeros(len(t), bool); tv = t.values
        for e in set(ev[anchor_rows]):
            first = tv[[r for r in anchor_rows if ev[r] == e]].min()
            same = ev == e
            keep |= same & ((tv < first) if window == "before" else (tv > first))
        return np.where(keep)[0]
    m = re.fullmatch(r"minutes_(before|after):(\d+)", window or "")
    if m:   # "30 minutes to an hour before the torch performance", "right after the photo of X" (minutes_after:30)
        n = pd.Timedelta(minutes=int(m.group(2))); keep = np.zeros(len(t), bool); tv = t
        for r in anchor_rows:
            a = t.iloc[r]
            if pd.isna(a):
                continue
            keep |= (((tv >= a - n) & (tv < a)) if m.group(1) == "before" else ((tv > a) & (tv <= a + n))).to_numpy()
        return np.where(keep)[0]
    if window == "same_place" and "place" in it:
        places = set(it.place.iloc[anchor_rows]) - {""}
        return np.where(it.place.isin(places))[0]
    return np.arange(idx.n_items)


def execute(idx, P: MultiPlan, enc, judge, th: Thresholds = Thresholds(), max_anchor: int = 5) -> dict:
    trace = {}
    scope = np.arange(idx.n_items)
    if P.anchor:
        a = run_album(idx, AlbumSpec(name="anchor", looks=P.anchor.looks, judge_question=P.anchor.judge_question,
                                     date_from=P.date_from, date_to=P.date_to, place=P.place), enc, judge, None, th=th)
        top = a.returned.sort_values("p_attr", ascending=False).head(max_anchor)
        anchor_rows = top.item_row.to_numpy()
        scope = window_rows(idx, anchor_rows, P.window)
        trace["anchor"] = dict(found=len(a.returned), used=anchor_rows.tolist(), window=P.window, window_items=len(scope))
    sub = store.subset(idx, scope) if len(scope) < idx.n_items else idx
    spec = AlbumSpec(name="target", looks=P.target.looks, judge_question=P.target.judge_question,
                     date_from=None if P.anchor else P.date_from, date_to=None if P.anchor else P.date_to,
                     place=None if P.anchor else P.place, want=P.want)
    r = run_album(sub, spec, enc, judge, None, th=th)
    ret = r.returned.copy()
    if P.exclude_question and len(ret):
        pex = _judge_rows(sub, judge, ret.item_row.to_numpy(), np.full(len(ret), -1), P.exclude_question)
        trace["excluded"] = int((pex >= th.judge_accept).sum())
        ret = ret[pex < th.judge_accept]
    trace["target"] = dict(in_scope=r.n_in_scope, returned=len(ret))
    return dict(returned=ret, trace=trace, report=r.report)
