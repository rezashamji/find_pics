"""Planner JSON parsing/validation and exact date/media scoping."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from findpics.planner import build_prompt, parse_plan, plan, AlbumSpec
from findpics.engine import scope_mask
from findpics.store import Index


def test_prompt_injects_today_and_request():
    p = build_prompt("find bread", owner="Reza", people=["Reza", "Dad"], today=date(2026, 10, 2))
    assert "2026-10-02" in p and "find bread" in p and "Reza, Dad" in p


def test_parse_plan_extracts_json_from_chatter():
    txt = 'Sure! {"albums":[{"name":"Bread","judge_question":"Is there bread?","looks":["bread"]}],"notes":""} done'
    P = parse_plan(txt)
    assert P.albums[0].name == "Bread" and P.albums[0].media == "any" and P.albums[0].want == "all"


def test_plan_retries_then_succeeds():
    outs = iter(["not json", '{"albums":[{"name":"A","judge_question":"q"}]}'])
    P = plan("x", lambda prompt: next(outs))
    assert P.albums[0].name == "A"


def test_plan_gives_up():
    with pytest.raises(ValueError):
        plan("x", lambda prompt: "nope", retries=1)


def _idx():
    items = pd.DataFrame(dict(item_id=list("abcd"), path=list("abcd"), media=["photo", "video", "photo", "photo"],
                              taken=["2016-05-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00",
                                     "2026-04-02T00:00:00+00:00", "2026-10-01T00:00:00+00:00"]))
    return Index(None, items, pd.DataFrame(), np.zeros((0, 4)), pd.DataFrame(), np.zeros((0, 512)), pd.DataFrame())


def test_scope_dates_inclusive_exclusive_and_media():
    idx = _idx()
    s = AlbumSpec(name="x", judge_question="q", date_from="2026-04-02", date_to="2026-10-01")
    assert scope_mask(idx, s).tolist() == [False, True, True, False]
    s = AlbumSpec(name="x", judge_question="q", media="video")
    assert scope_mask(idx, s).tolist() == [False, True, False, False]


def test_dates_without_supporting_phrase_are_removed():
    req = "photos of me looking heavier, and the best fit photos from the past 6 months"
    txt = ('{"albums":[{"name":"heavier","judge_question":"q","time_phrase":null,"date_from":"2025-04-02"},'
           '{"name":"fit","judge_question":"q","time_phrase":"past 6 months","date_from":"2025-04-02"},'
           '{"name":"x","judge_question":"q","time_phrase":"last summer","date_from":"2025-06-01"}]}')
    P = parse_plan(txt, req)
    assert P.albums[0].date_from is None
    assert P.albums[1].date_from == "2025-04-02"
    assert P.albums[2].date_from is None  # phrase not in request


def test_identity_style_condition_removed():
    txt = ('{"albums":[{"name":"Drew 90s","person":"Drew Barrymore","looks":["a woman with blonde hair"],'
           '"judge_question":"Does the person in the red box look like Drew Barrymore?"},'
           '{"name":"Reza heavy","person":"Reza","looks":["a heavy man"],"judge_question":"Does the person in the red box look overweight?"}]}')
    P = parse_plan(txt, "every photo of Drew Barrymore, and Reza looking heavy")
    assert P.albums[0].judge_question is None and P.albums[0].looks == []
    assert P.albums[1].judge_question is not None


def test_parse_refs():
    from findpics.cli import _parse_refs
    r = _parse_refs(["Reza=a.jpg, b.jpg", "Jelly=c.jpg", "Reza=d.jpg"])
    assert r == {"Reza": ["a.jpg", "b.jpg", "d.jpg"], "Jelly": ["c.jpg"]}


def test_red_box_removed_when_no_person():
    txt = ('{"albums":[{"name":"a","judge_question":"Does the person in the red box look like an elephant seal in this photo?"},'
           '{"name":"b","judge_question":"Does the spider in the red box look like it is in its web?"},'
           '{"name":"c","person":"Reza","judge_question":"Does the person in the red box look heavier?"}]}')
    P = parse_plan(txt, "x")
    assert P.albums[0].judge_question == "Does someone look like an elephant seal in this photo?"
    assert P.albums[1].judge_question == "Does the spider look like it is in its web?"
    assert "red box" in P.albums[2].judge_question          # person albums keep it (a box/crop is drawn)


def test_identity_name_replaced_not_condition_deleted():
    txt = ('{"albums":[{"name":"m","person":"Mom","looks":["a woman laughing"],"judge_question":"Is Mom laughing at dinner?"},'
           '{"name":"d","person":"Dad","judge_question":"Is Dad in this photo?"}]}')
    P = parse_plan(txt, "mom laughing at dinner, and every photo of dad")
    assert P.albums[0].judge_question == "Is the person in the red box laughing at dinner?"
    assert P.albums[1].judge_question is None
