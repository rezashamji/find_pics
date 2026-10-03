"""Multi-step planner pieces: event segmentation, windows, plan grounding (fake LLM; no GPU)."""
import json
from datetime import date

import numpy as np
import pandas as pd

from findpics.agent import events, make_plan, window_rows
from findpics.store import Index


def _idx(times, places=None):
    items = pd.DataFrame(dict(item_id=[str(i) for i in range(len(times))], path="", media="photo", taken=times,
                              place=places or [""] * len(times)))
    return Index(None, items, pd.DataFrame(), np.zeros((0, 2)), pd.DataFrame(), np.zeros((0, 512)), pd.DataFrame())


T = ["2020-05-01T10:00:00+00:00", "2020-05-01T11:30:00+00:00", "2020-05-01T20:00:00+00:00",   # day 1: two events
     "2020-05-03T09:00:00+00:00", "2020-05-20T09:00:00+00:00"]                                # same week / later


def test_events_split_on_gaps():
    ev = events(pd.Series(T))
    assert ev[0] == ev[1] != ev[2] and len(set(ev)) == 4


def test_windows():
    idx = _idx(T, ["Paris, FR", "Paris, FR", "Lyon, FR", "Paris, FR", ""])
    assert window_rows(idx, np.array([0]), "same_day").tolist() == [0, 1, 2]
    assert window_rows(idx, np.array([0]), "same_event").tolist() == [0, 1]
    assert window_rows(idx, np.array([0]), "same_week").tolist() == [0, 1, 2, 3]   # 2020-05-01 (Fri) .. 05-03 (Sun)
    assert window_rows(idx, np.array([2]), "same_place").tolist() == [2]
    assert window_rows(idx, np.array([0]), "same_month").tolist() == [0, 1, 2, 3, 4]
    assert window_rows(idx, np.array([0]), "same_year").tolist() == [0, 1, 2, 3, 4]


def test_make_plan_two_steps_and_grounding():
    reply = json.dumps({"anchor": {"looks": ["foggy city at dusk"], "judge_question": "Is this a foggy cityscape at dusk?"},
                        "window": "same_week", "target": {"looks": ["dinner table"], "judge_question": "Is there a dinner table?"},
                        "exclude_question": "Are there wine bottles?", "time_phrase": "last summer", "date_from": "2025-06-01",
                        "date_to": "2025-09-01", "place": "Seattle", "want": "all"})
    P = make_plan("photos from the week I saw a foggy city at dusk, excluding wine bottles", lambda p: reply, today=date(2026, 10, 2))
    assert P.anchor and P.window == "same_week" and P.exclude_question
    assert P.date_from is None and P.place is None     # invented time/place removed (not in the request)


def test_offset_windows():
    idx = _idx(T)   # day 05-01 (x3), 05-03, 05-20
    assert window_rows(idx, np.array([3]), "days_before:2").tolist() == [0, 1, 2]     # 05-01..05-02 before 05-03
    assert window_rows(idx, np.array([0]), "days_after:2").tolist() == [3]            # 05-02..05-03 after 05-01
    assert window_rows(idx, np.array([0]), "days_after:1").tolist() == []
