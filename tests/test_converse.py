"""The single conversational path: planning a turn (fake LLM), grounding across turns, the anchor/window/exclude
execution, per-item overrides, and the judge cache (no GPU, no images decoded)."""
import json
from datetime import date

import numpy as np
import pandas as pd
from PIL import Image

from findpics import engine as E
from findpics.converse import CachedJudge, Plan, Session, build_prompt, plan_turn, run_plan
from findpics.store import Index


def _reply(**album):
    a = dict(name="x", person=None, looks=["thing"], judge_question="Is there a thing?")
    a.update(album)
    return lambda prompt: json.dumps({"albums": [a], "notes": ""})


def test_single_step_request_has_no_anchor():
    P = plan_turn("all my photos with bread", _reply(), today=date(2026, 10, 2))
    a = P.albums[0]
    assert a.anchor is None and a.window is None


def test_two_step_request_and_grounding():
    llm = _reply(anchor={"looks": ["birthday cake"], "judge_question": "Does the person in the red box hold a cake?"},
                 window="same_day", exclude_question="Is there a dog?", place="Paris", time_phrase="last summer",
                 date_from="2025-06-01", date_to="2025-09-01")
    P = plan_turn("photos from the day of my birthday party, without the dog", llm, today=date(2026, 10, 2))
    a = P.albums[0]
    assert a.anchor and a.window == "same_day" and a.exclude_question
    assert "red box" not in a.anchor.judge_question            # an anchor is a moment, never a boxed person
    assert a.place is None and a.date_from is None              # invented place/time removed: not in what was said


def test_window_without_anchor_is_dropped():
    P = plan_turn("cats", _reply(window="same_week"), today=date(2026, 10, 2))
    assert P.albums[0].window is None


def test_followup_sees_history_and_keeps_earlier_grounding():
    first = Plan.model_validate({"albums": [dict(name="fit", person="me", looks=["lean"], judge_question="lean?",
                                                 time_phrase="past 6 months", date_from="2026-04-02")]})
    prompt = build_prompt("only the ones at the beach", ["me looking fit in the past 6 months"], first, today=date(2026, 10, 2))
    assert "past 6 months" in prompt and "only the ones at the beach" in prompt and "Return the WHOLE updated plan" in prompt
    llm = _reply(name="fit", person="me", looks=["lean man on a beach"], judge_question="Is the person in the red box lean and on a beach?",
                 time_phrase="past 6 months", date_from="2026-04-02")
    P = plan_turn("only the ones at the beach", llm, history=["me looking fit in the past 6 months"], current=first,
                  today=date(2026, 10, 2))
    assert P.albums[0].date_from == "2026-04-02"     # time phrase came from message 1: still grounded in message 2


class Enc:
    name = "fake"

    def texts(self, t):
        v = np.zeros((len(t), 4), np.float32); v[:, 0] = 1
        return v


class QJudge:
    """The 'image' is the item id; answers depend on which question is asked."""

    def __init__(self, yes):     # {question substring: set of item ids that are yes}
        self.yes = yes

    def p_yes(self, ims, q):
        ids = next((v for k, v in self.yes.items() if k in q), set())
        return [1.0 if im in ids else 0.0 for im in ims]


def _lib(monkeypatch, n=30):
    days = [f"2020-05-{1 + i // 5:02d}T{8 + i % 5}:00:00+00:00" for i in range(n)]       # 5 photos per day
    items = pd.DataFrame(dict(item_id=[f"i{i}" for i in range(n)], path=[f"/x/{i}.jpg" for i in range(n)],
                              media="photo", taken=days, place=""))
    units = pd.DataFrame(dict(item_id=items.item_id, frame_t=-1.0, item_row=np.arange(n)))
    clip = np.ones((n, 4), np.float16)
    idx = Index(None, items, units, clip, pd.DataFrame(columns=["item_row", "unit_row", "frame_t"]),
                np.zeros((0, 512), np.float16), pd.DataFrame())
    monkeypatch.setattr(E, "_frame_for", lambda idx, r, fr: idx.items.item_id.iloc[r])
    monkeypatch.setattr(E, "_boxed", lambda idx, im, fr: im)
    return idx


def test_anchor_window_exclude_and_overrides(monkeypatch):
    idx = _lib(monkeypatch)
    # anchor (cake) is i6 on day 2 (items i5..i9); targets i7, i8 on day 2 and i20 on another day; i8 has a dog
    J = QJudge({"cake": {"i6"}, "people dancing": {"i7", "i8", "i20"}, "dog": {"i8"}})
    P = Plan.model_validate({"albums": [dict(name="party", looks=["dancing"], judge_question="Are people dancing?",
                                             anchor={"looks": ["cake"], "judge_question": "Is there a birthday cake?"},
                                             window="same_day", exclude_question="Is there a dog?")]})
    r = run_plan(idx, P, Enc(), J)[0]
    assert set(r.returned.item_id) == {"i7"}                              # i20: outside the day; i8: excluded
    assert (idx.items.item_id.to_numpy()[r.returned.item_row] == r.returned.item_id).all()   # rows point at the FULL index
    assert r.trace["window_items"] == 5 and r.trace["excluded"] == 1
    r2 = run_plan(idx, P, Enc(), J, exclude_ids={"i7"})[0]                # tapped "wrong" on the review page
    assert len(r2.returned) == 0


def test_streaming_plan_reaches_every_photo(monkeypatch):
    from findpics.converse import stream_plan
    from findpics.engine import Thresholds
    idx = _lib(monkeypatch, n=900)
    J = QJudge({"thing": {"i850", "i3"}})
    P = Plan.model_validate({"albums": [dict(name="x", looks=["thing"], judge_question="Is there a thing?")]})
    rounds = list(stream_plan(idx, P, Enc(), J, th=Thresholds(head_size=100, head_max=100, tail_budget=0, stream=True)))
    assert len(rounds) > 1 and set(rounds[-1][0].returned.item_id) == {"i850", "i3"}
    assert (rounds[-1][0].judged["where"] == "head").sum() == 900


def test_judge_cache(tmp_path):
    class Count:
        n = 0

        def p_yes(self, ims, q):
            Count.n += len(ims); return [0.5] * len(ims)

    ims = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    J = CachedJudge(Count(), tmp_path / "c.json")
    J.p_yes(ims, "q?"); J.p_yes(ims, "q?"); J.p_yes(ims[:1], "other?")
    assert Count.n == 3 and J.hits == 2
    J.save(); assert CachedJudge(Count(), tmp_path / "c.json").p_yes(ims, "q?") == [0.5, 0.5] and Count.n == 3


def test_session_roundtrip(tmp_path):
    S = Session(tmp_path / "s")
    S.add_turn("bread", Plan.model_validate({"albums": [dict(name="bread")]})); S.add_reviews({"i1": "bad", "i2": "ok"})
    S.save()
    T = Session(tmp_path / "s")
    assert T.turn == 1 and T.current.albums[0].name == "bread" and T.state["exclude_ids"] == ["i1"]


# --- regressions from the 9B planner test (eval/planners/conversations.json, job 49950289): its actual outputs ---

def test_invented_dates_from_a_moment_are_removed_and_window_follows_words():
    llm = _reply(name="food", looks=["food"], judge_question="is there food in the photo?",
                 anchor={"looks": ["a canyon"], "judge_question": "is this the Grand Canyon?"}, window="same_event",
                 exclude_question="is there a burger?", place="Grand Canyon",
                 time_phrase="the week I went to the Grand Canyon", date_from="2026-09-24", date_to="2026-10-01")
    a = plan_turn("food photos from the week I went to the Grand Canyon, no burgers", llm, today=date(2026, 10, 2)).albums[0]
    assert a.date_from is None and a.time_phrase is None      # "the week I went to X" names no calendar time
    assert a.window == "same_week"                            # the words say week
    assert a.place is None                                    # the place is the anchor moment, not a filter


def test_zero_albums_is_retried():
    replies = iter([json.dumps({"albums": [], "notes": "library has only people"}),
                    json.dumps({"albums": [dict(name="s", looks=["a screenshot"], judge_question="Is this a screenshot?")]})])
    P = plan_turn("screenshots of text messages", lambda p: next(replies), today=date(2026, 10, 2))
    assert len(P.albums) == 1


def test_scene_word_used_as_place_becomes_a_condition(monkeypatch):
    from findpics.converse import Album, place_or_look
    idx = _lib(monkeypatch); idx.items["place"] = ["Paris, FR"] * 15 + [""] * 15
    a = place_or_look(idx, Album(name="Dad at the beach", person="Dad", place="beach"))
    assert a.place is None and "beach" in a.looks and "beach" in a.judge_question and "red box" in a.judge_question
    assert place_or_look(idx, Album(name="p", place="Paris")).place == "Paris"     # real place in this library: kept
    b = place_or_look(idx, Album(name="fit", person="me", looks=["lean"], judge_question="Is the person in the red box lean?", place="gym"))
    assert b.judge_question.startswith("Is the person in the red box lean") and "gym" in b.judge_question


def test_followup_prompt_says_edit_not_new_album():
    cur = Plan.model_validate({"albums": [dict(name="cat", looks=["a cat"], judge_question="Is there a cat?", media="photo")]})
    p = build_prompt("also videos of her", ["photos of my cat"], cur, today=date(2026, 10, 2))
    assert "EDITS the existing album" in p and "never a new person" in p


def test_time_relative_to_another_photo_is_not_calendar_time():
    llm = _reply(time_phrase="7 days before the photo of the elderly Maasai woman", date_from="2026-09-25", date_to="2026-10-02")
    P = plan_turn("Find photos taken 7 days before the photo of the elderly Maasai woman", llm, today=date(2026, 10, 2))
    assert P.albums[0].date_from is None
    llm2 = _reply(time_phrase="before 2019", date_to="2019-01-01")
    assert plan_turn("photos of bread before 2019", llm2, today=date(2026, 10, 2)).albums[0].date_to == "2019-01-01"


def test_every_anchor_hit_opens_a_window(monkeypatch):
    idx = _lib(monkeypatch)
    # "pics at a car show": anchor = car show (i6 on day 2, i16 on day 4); targets on both days must survive
    J = QJudge({"car show": {"i6", "i16"}, "classic cars": {"i7", "i17", "i25"}})
    P = Plan.model_validate({"albums": [dict(name="cars", looks=["classic cars"], judge_question="Are there classic cars?",
                                             anchor={"looks": ["car show"], "judge_question": "Is this a car show?"},
                                             window="same_day")]})
    assert set(run_plan(idx, P, Enc(), J)[0].returned.item_id) == {"i7", "i17"}     # i25: a different day
