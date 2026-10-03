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


def test_exclusion_fixes_count_and_restates_bound(monkeypatch):
    idx = _lib(monkeypatch)
    J = QJudge({"bread": {"i1", "i2", "i3", "i4"}, "sandwich": {"i1", "i2"}})
    P = Plan.model_validate({"albums": [dict(name="bread", looks=["bread"], judge_question="Is there bread?",
                                             exclude_question="Is there a sandwich?")]})
    r = run_plan(idx, P, Enc(), J)[0]
    assert set(r.returned.item_id) == {"i3", "i4"} and "'bread': 2 items." in r.report and r.cert["found"] == 2


def test_anchor_not_found_returns_nothing_and_says_so(monkeypatch):
    idx = _lib(monkeypatch)
    J = QJudge({"cake": set(), "dancing": {"i7", "i20"}})
    P = Plan.model_validate({"albums": [dict(name="party", looks=["dancing"], judge_question="Are people dancing?",
                                             anchor={"looks": ["cake"], "judge_question": "Is there a birthday cake?"},
                                             window="same_day")]})
    r = run_plan(idx, P, Enc(), J)[0]
    assert len(r.returned) == 0 and "Could not find the moment" in r.report


def test_invented_window_becomes_same_event_and_year_is_supported():
    llm = _reply(anchor={"looks": ["a medal"], "judge_question": "Is there a finisher medal?"}, window="same_decade")
    assert plan_turn("race photos around when I got the medal", llm, today=date(2026, 10, 2)).albums[0].window == "same_event"
    llm2 = _reply(anchor={"looks": ["a medal"], "judge_question": "Is there a finisher medal?"}, window="same_event")
    assert plan_turn("race photos from the year I got the medal", llm2, today=date(2026, 10, 2)).albums[0].window == "same_year"


def test_always_true_question_is_flagged(monkeypatch):
    idx = _lib(monkeypatch, n=900)
    J = QJudge({"real life or": {f"i{i}" for i in range(900)}})
    P = Plan.model_validate({"albums": [dict(name="b", looks=["building"], judge_question="Is this building in real life or not?")]})
    from findpics.engine import Thresholds
    r = run_plan(idx, P, Enc(), J, th=Thresholds(head_size=100, head_max=100, tail_budget=200))[0]
    assert "too broad" in r.report


def test_unanswerable_or_copied_questions_are_sent_back():
    bad = [json.dumps({"albums": [dict(name="t", looks=["tiger"], judge_question="Is this a tiger taken in the same year as the yawning tiger?",
                                       anchor={"looks": ["yawning tiger"], "judge_question": "Is a tiger yawning?"}, window="same_year")]}),
           json.dumps({"albums": [dict(name="t", looks=["wall"], judge_question="Is there a wall with drawings?",
                                       anchor={"looks": ["wall"], "judge_question": "Is there a wall with drawings?"}, window="same_day")]}),
           json.dumps({"albums": [dict(name="t", looks=["chairs"], judge_question="Are there metal chairs?",
                                       anchor={"looks": ["wall"], "judge_question": "Is there a wall with drawings?"}, window="same_day")]})]
    prompts = []
    it = iter(bad)
    P = plan_turn("photo on the day the drawing wall was seen, with metal chairs", lambda p: (prompts.append(p), next(it))[1],
                  today=date(2026, 10, 2))
    assert P.albums[0].judge_question == "Are there metal chairs?" and len(prompts) == 3
    assert "ONE photo" in prompts[1] and "repeats the anchor" in prompts[2]


def test_repeated_unanswerable_question_degrades_instead_of_failing():
    bad = json.dumps({"albums": [dict(name="shoes", looks=["several men wearing shoes"],
                                      judge_question="Is the person wearing the same shoes as the person in the reference photo?")]})
    P = plan_turn("men wearing the same shoes on my Europe trip", lambda p: bad, today=date(2026, 10, 2))
    assert P.albums[0].judge_question == "Does this photo show several men wearing shoes?" and "could not express" in P.notes


def test_offset_window_kept_over_word_rules():
    llm = _reply(anchor={"looks": ["wedding"], "judge_question": "Is this a wedding?"}, window="days_after:1")
    assert plan_turn("photos from the day after the wedding", llm, today=date(2026, 10, 2)).albums[0].window == "days_after:1"


def test_subject_reference_photos_for_a_pet(monkeypatch):
    """'my dog Max' from 3 photos: no faces -> image-vector candidates + side-by-side judge; condition applied after."""
    from findpics.converse import SubjectRefs
    idx = _lib(monkeypatch, n=30)
    v = np.zeros((30, 4), np.float16); v[:, 1] = 1; v[[3, 4, 9], 0] = 1; v[[3, 4, 9], 1] = 0   # i3, i4, i9 look like the refs
    idx.clip = v
    monkeypatch.setattr(E, "side_by_side", lambda ref, im: im)

    class SubjEnc(Enc):
        def images(self, ims):
            x = np.zeros((len(ims), 4), np.float32); x[:, 0] = 1
            return x

    J = QJudge({"SAME individual": {"i3", "i4"}, "beach": {"i4", "i20"}})      # i9: a look-alike the judge rejects
    refs_for = lambda person: (person, SubjectRefs([Image.new("RGB", (8, 8))] * 3, person), None, 3)
    P = Plan.model_validate({"albums": [dict(name="Max", person="Max", looks=["dog"])]})
    assert set(run_plan(idx, P, SubjEnc(), J, refs_for)[0].returned.item_id) == {"i3", "i4"}
    P2 = Plan.model_validate({"albums": [dict(name="Max at the beach", person="Max", looks=["dog"],
                                              judge_question="Is the dog on a beach?")]})
    assert set(run_plan(idx, P2, SubjEnc(), J, refs_for)[0].returned.item_id) == {"i4"}


def test_filter_question_narrows_an_album_that_already_has_a_question(monkeypatch):
    idx = _lib(monkeypatch)
    J = QJudge({"heavier": {"i1", "i2", "i3"}, "outdoors": {"i2", "i3", "i7"}})
    P = Plan.model_validate({"albums": [dict(name="heavier", looks=["heavy build"], judge_question="Does someone look heavier?",
                                             filter_question="Is this photo taken outdoors?")]})
    r = run_plan(idx, P, Enc(), J)[0]
    assert set(r.returned.item_id) == {"i2", "i3"} and r.trace["filtered_out"] == 1


def test_place_written_as_filter_becomes_a_place_filter(monkeypatch):
    from findpics.converse import Album, filter_to_place
    idx = _lib(monkeypatch); idx.items["place"] = ["Paris, France"] * 10 + ["Boston, US"] * 20
    a = filter_to_place(idx, Album(name="cat", looks=["a cat"], judge_question="Is this a cat?",
                                   filter_question="Is the location Paris?"))
    assert a.place == "Paris" and a.filter_question is None
    b = filter_to_place(idx, Album(name="x", filter_question="Is the person at the gym?"))
    assert b.filter_question and b.place is None


def test_no_condition_returns_everything_in_scope_without_judging(monkeypatch):
    idx = _lib(monkeypatch)
    class NoCalls:
        def p_yes(self, ims, q):
            raise AssertionError("judge must not be called")
    P = Plan.model_validate({"albums": [dict(name="may 2", date_from="2020-05-02", date_to="2020-05-03",
                                             time_phrase="May 2 2020")]})
    r = run_plan(idx, P, Enc(), NoCalls())[0]
    assert set(r.returned.item_id) == {f"i{i}" for i in range(5, 10)} and r.cert["recall_lower"] == 1.0


def test_relational_wordings_and_copied_exclusion():
    for q in ["Is the ship later seen docking in another country?", "Does the woman also appear in another photo?",
              "does this photo show a later event than the 2004 performance?"]:
        from findpics.converse import _RELATIONAL
        assert _RELATIONAL.search(q), q
    llm = _reply(judge_question="is there a wine bottle?", exclude_question="Is there a wine bottle?",
                 anchor={"looks": ["fog"], "judge_question": "Is this a foggy city?"}, window="same_week")
    a = plan_turn("all photos from the week I saw a foggy city, excluding wine bottles", llm, today=date(2026, 10, 2)).albums[0]
    assert a.judge_question is None and a.exclude_question


def test_unanswerable_filter_is_dropped_after_retries():
    from findpics.converse import _RELATIONAL
    for q in ["Is the jacket the same one worn in 2007?", "Was this photo taken twice within the last six months?",
              "Is the person wearing the same top as in the photo at Puffing Billy?"]:
        assert _RELATIONAL.search(q), q
    bad = json.dumps({"albums": [dict(name="x", looks=["a girl"], judge_question="Is there a girl?",
                                      filter_question="Does the girl also appear in another photo wearing a scarf?")]})
    P = plan_turn("photos of the girl", lambda p: bad, today=date(2026, 10, 2))
    assert P.albums[0].filter_question is None and P.albums[0].judge_question == "Is there a girl?"


def test_reformatted_year_phrases_are_grounded_and_span_whole_years():
    llm = _reply(person="Dad", looks=[], judge_question=None, time_phrase="2015-2018", date_from="2015-01-01", date_to="2018-12-31")
    a = plan_turn("photos and videos of Dad from 2015 to 2018", llm, today=date(2026, 10, 2)).albums[0]
    assert (a.date_from, a.date_to) == ("2015-01-01", "2019-01-01")
    llm2 = _reply(person="me", judge_question="Is this a wedding?", time_phrase="2019 and 2020", date_from="2019-01-01", date_to="2020-01-01")
    b = plan_turn("just 2019 and 2020", llm2, history=["photos of me at weddings"], current=Plan(albums=[]), today=date(2026, 10, 2)).albums[0]
    assert (b.date_from, b.date_to) == ("2019-01-01", "2021-01-01")
    llm3 = _reply(time_phrase="the 1990s", date_from="1990-01-01", date_to="2000-01-01")
    assert plan_turn("bread in the 1990s", llm3, today=date(2026, 10, 2)).albums[0].date_to == "2000-01-01"   # decades untouched


def test_dates_without_time_phrase_recovered_from_the_message():
    llm = _reply(person="Dad", looks=[], judge_question=None, time_phrase=None, date_from="2015-01-01", date_to="2018-12-31")
    a = plan_turn("photos and videos of Dad from 2015 to 2018", llm, today=date(2026, 10, 2)).albums[0]
    assert (a.date_from, a.date_to) == ("2015-01-01", "2019-01-01")
    llm2 = _reply(time_phrase=None, date_from="2019-01-01", date_to="2020-01-01")
    b = plan_turn("just 2019 and 2020", llm2, history=["photos of me at weddings"], current=Plan(albums=[]), today=date(2026, 10, 2)).albums[0]
    assert (b.date_from, b.date_to) == ("2019-01-01", "2021-01-01")
    llm3 = _reply(time_phrase=None, date_from="2019-01-01", date_to="2020-01-01")
    assert plan_turn("bread", llm3, today=date(2026, 10, 2)).albums[0].date_from is None   # no year said: still removed


def test_reference_kind_decided_by_image_text_model_not_face_detector():
    from findpics.cli import _KINDS, refs_show_a_person

    class KindEnc:
        def __init__(self, kinds):
            self.kinds = kinds                      # which prompt each reference image is closest to
        def images(self, ims):
            V = np.zeros((len(ims), len(_KINDS)), np.float32)
            for i, k in enumerate(self.kinds):
                V[i, k] = 1
            return V
        def texts(self, t):
            return np.eye(len(_KINDS), dtype=np.float32)
    img = [Image.new("RGB", (4, 4))] * 3
    assert refs_show_a_person(img, KindEnc([0, 0, 1]))          # 2 of 3 look like a person
    assert not refs_show_a_person(img, KindEnc([1, 1, 0]))      # dog photos (a 'face' may be detected anyway)


def test_resolve_relative_dates():
    from datetime import date
    from findpics.converse import resolve_relative
    T = date(2026, 10, 3)   # a Saturday
    assert resolve_relative("last weekend", T) == ("2026-09-26", "2026-09-28")
    assert resolve_relative("last weekend", date(2026, 10, 5)) == ("2026-10-03", "2026-10-05")   # Monday after
    assert resolve_relative("last Tuesday", T) == ("2026-09-29", "2026-09-30")
    assert resolve_relative("on Tuesday", T) == ("2026-09-29", "2026-09-30")
    assert resolve_relative("yesterday", T) == ("2026-10-02", "2026-10-03")
    assert resolve_relative("this morning", T) == ("2026-10-03", "2026-10-04")
    assert resolve_relative("this year", T) == ("2026-01-01", "2026-10-04")
    assert resolve_relative("last year", T) == ("2025-01-01", "2026-01-01")
    assert resolve_relative("last month", T) == ("2026-09-01", "2026-10-01")
    assert resolve_relative("last week", T) == ("2026-09-21", "2026-10-04")   # previous Monday .. today
    assert resolve_relative("the last 30 days", T) == ("2026-09-03", "2026-10-04")
    assert resolve_relative("past 6 months", T) == ("2026-04-03", "2026-10-04")
    assert resolve_relative("last summer", T) is None and resolve_relative("2019", T) is None


def _plan(llm_json, msg, **kw):
    from findpics.converse import plan_turn
    outs = list(llm_json) if isinstance(llm_json, list) else [llm_json]
    return plan_turn(msg, lambda p: outs.pop(0) if len(outs) > 1 else outs[0], **kw)


def test_planner_dates_fixed_in_code_and_occasion_dropped():
    from datetime import date
    T = date(2026, 10, 3)
    P = _plan('{"albums":[{"name":"a","looks":["a mug"],"judge_question":"Is there a mug?","time_phrase":"last weekend",'
              '"date_from":"2026-09-26","date_to":"2026-10-04"}]}', "the mug from last weekend", today=T)
    assert (P.albums[0].date_from, P.albums[0].date_to) == ("2026-09-26", "2026-09-28")
    P = _plan('{"albums":[{"name":"a","time_phrase":"on my birthday this year","date_from":"2026-10-03","date_to":"2026-10-04"}]}',
              "photos taken on my birthday this year", today=T)
    assert (P.albums[0].date_from, P.albums[0].date_to) == ("2026-01-01", "2026-10-04")


def test_unnamed_person_removed():
    P = _plan('{"albums":[{"name":"a","person":"Sara","time_phrase":null}]}', "photos of my sister", people=["Sara", "Dad"])
    assert P.albums[0].person is None and "Who is it" in P.notes
    P = _plan('{"albums":[{"name":"a","person":"Sara"}]}', "photos of Sarah at the lake", people=["Sara"])
    assert P.albums[0].person == "Sara"
    P = _plan('{"albums":[{"name":"a","person":"Dad"}]}', "dad's birthday", people=["Dad"])
    assert P.albums[0].person == "Dad"
    P = _plan('{"albums":[{"name":"a","person":"Reza"}]}', "me at the gym", owner="Reza Shamji")
    assert P.albums[0].person == "Reza"


def test_first_person_question_retried_then_degraded():
    bad = '{"albums":[{"name":"a","looks":["a house exterior"],"judge_question":"Is this the new house we bought?"}]}'
    good = '{"albums":[{"name":"a","looks":["a house exterior"],"judge_question":"Is this the outside of a house?"}]}'
    assert _plan([bad, good], "the new house we bought").albums[0].judge_question == "Is this the outside of a house?"
    P = _plan(bad, "the new house we bought")
    assert P.albums[0].judge_question == "Does this photo show a house exterior?"


def test_fuzz3_fixes():
    from datetime import date
    T = date(2026, 10, 3)
    # owner album with no reference to the owner ("photos from" -> "Reza: Is this a man?")
    P = _plan('{"albums":[{"name":"a","person":"Reza","judge_question":"Is this a photo of a man?"}]}', "photos from",
              owner="Reza", people=["Reza"])
    assert P.albums[0].person is None
    P = _plan('{"albums":[{"name":"a","person":"me"}]}', "the selfie I took on the plane", owner="Reza")
    assert P.albums[0].person == "me"
    # a red box written with "a"
    P = _plan('{"albums":[{"name":"a","judge_question":"Does this photo show a person in a red box?"}]}', "videos of Italy")
    assert "red box" not in P.albums[0].judge_question
    # names are not visible
    bad = '{"albums":[{"name":"a","looks":["a road trip"],"judge_question":"Is this a woman named Sarah on a road trip?"}]}'
    assert "named" not in _plan(bad, "my friend Sarah on the road trip").albums[0].judge_question
    # the named end day is included; code-changed dates are stated in the notes
    P = _plan('{"albums":[{"name":"a","time_phrase":"between July 1st and July 15th","date_from":"2026-07-01",'
              '"date_to":"2026-07-15","media":"video"}]}', "videos taken between July 1st and July 15th", today=T)
    assert P.albums[0].date_to == "2026-07-16"
    P = _plan('{"albums":[{"name":"a","time_phrase":"last weekend","date_from":"2026-09-26","date_to":"2026-10-04"}]}',
              "photos from last weekend", today=T)
    assert "2026-09-26 to 2026-09-27" in P.notes


def test_fuzz4_fixes():
    from datetime import date
    from findpics.converse import resolve_relative
    T = date(2026, 10, 3)
    assert resolve_relative("the weekend before last", T) == ("2026-09-19", "2026-09-21")
    # lowercase "i" refers to the owner
    P = _plan('{"albums":[{"name":"a","person":"me","judge_question":"Is this a selfie on a plane?"}]}',
              "the selfie i took on the plane", owner="Reza")
    assert P.albums[0].person == "me"
    # "Tuesday" is calendar time
    P = _plan('{"albums":[{"name":"a","looks":["a receipt"],"judge_question":"Is this a receipt?","time_phrase":"on Tuesday",'
              '"date_from":"2026-10-06","date_to":"2026-10-07"}]}', "the receipt from the grocery run on Tuesday", today=T)
    assert (P.albums[0].date_from, P.albums[0].date_to) == ("2026-09-29", "2026-09-30")
    # dates set, phrase left null: recovered from the message, then computed
    P = _plan('{"albums":[{"name":"a","anchor":{"looks":["a stage"],"judge_question":"Is this a concert?"},"window":"same_event",'
              '"judge_question":"Is there a stage?","date_from":"2026-09-03","date_to":"2026-10-03"}]}',
              "pictures from the concert we went to last month", today=T)
    assert (P.albums[0].date_from, P.albums[0].date_to) == ("2026-09-01", "2026-10-01")
    # the moment asked twice -> everything in it
    same = ('{"albums":[{"name":"a","looks":["a bride"],"judge_question":"Is this a wedding?",'
            '"anchor":{"looks":["a bride"],"judge_question":"Is this a wedding?"},"window":"same_event"}]}')
    P = _plan(same, "photos from the wedding")
    assert P.albums[0].judge_question is None and P.albums[0].anchor is not None
    # "the user's brother" needs the owner's knowledge
    from findpics.converse import _bad_q
    assert _bad_q("Is the person in the photo the user's brother?") and _bad_q("Did the owner take this photo?")


def test_place_question_and_media_twins():
    P = _plan('{"albums":[{"name":"a","looks":["Japan"],"judge_question":"Is this photo taken in Japan?","want":"best","max_items":10}]}',
              "my 10 best photos from Japan")
    assert P.albums[0].place == "Japan" and P.albums[0].judge_question is None
    P = _plan('{"albums":[{"name":"p","looks":["a cat"],"judge_question":"Is there a cat in this photo?","media":"photo"},'
              '{"name":"v","looks":["a cat"],"judge_question":"Is there a cat in this video?","media":"video"}]}', "my cat, also videos")
    assert len(P.albums) == 1 and P.albums[0].media == "any"
    P = _plan('{"albums":[{"name":"p","looks":["a cat"],"judge_question":"Is there a cat?","media":"photo"},'
              '{"name":"v","looks":["a dog"],"judge_question":"Is there a dog?","media":"video"}]}', "cat photos and dog videos")
    assert len(P.albums) == 2


def test_heldout_fixes():
    from datetime import date
    from findpics.converse import resolve_relative as r
    T = date(2026, 10, 3)
    assert r("last August", T) == ("2026-08-01", "2026-09-01") and r("sept", T) == ("2026-09-01", "2026-10-01")
    assert r("Fourth of July", T) == ("2026-07-04", "2026-07-05") and r("Thanksgiving", T) == ("2025-11-27", "2025-11-28")
    assert r("two years ago", T) == ("2024-01-01", "2025-01-01")
    # an occasion with no calendar part: no date (the planner guessed today); with one: that part
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a celebration?","time_phrase":"50th anniversary celebration",'
              '"date_from":"2026-10-03","date_to":"2026-10-04"}]}', "the 50th anniversary celebration", today=T)
    assert P.albums[0].date_from is None
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a wedding?","time_phrase":"the wedding last summer",'
              '"date_from":"2025-06-01","date_to":"2025-09-01"}]}', "photos from the wedding last summer", today=T)
    assert P.albums[0].date_from == "2025-06-01"
    # "the last video": an order, not a date
    P = _plan('{"albums":[{"name":"a","judge_question":"Is a baby sleeping?","time_phrase":"last","date_from":"2026-10-03",'
              '"date_to":"2026-10-04","media":"video"}]}', "the last video i took of the baby sleeping", today=T)
    assert P.albums[0].date_from is None
    # "Is this a video?" removed; "Is this Hawaii?" anchor -> place
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there a cake?","filter_question":"Is this a video file?","media":"video"},'
              '{"name":"b","judge_question":"Is this a sunset?","anchor":{"looks":["Hawaii"],"judge_question":"Is this Hawaii?"},'
              '"window":"same_place"}]}', "the cake video, and the sunset in Hawaii")
    assert P.albums[0].filter_question is None and P.albums[1].place == "Hawaii" and P.albums[1].anchor is None
    # month-per-album split: condition-less albums dropped
    P = _plan('{"albums":[{"name":"March","time_phrase":"March","date_from":"2026-03-01","date_to":"2026-04-01"},'
              '{"name":"April","time_phrase":"April","date_from":"2026-04-01","date_to":"2026-05-01"}]}', "memories from March", today=T)
    assert len(P.albums) == 1
    # known names in a question -> retried
    bad = '{"albums":[{"name":"a","judge_question":"Is this a family member (Reza, Dad, Mom)?"}]}'
    good = '{"albums":[{"name":"a","judge_question":"Is this a family photo?"}]}'
    assert _plan([bad, good], "family ones", people=["Reza", "Dad", "Mom"]).albums[0].judge_question == "Is this a family photo?"
    # the thing itself vs everything from the moment
    same = ('{"albums":[{"name":"a","looks":["a pizza"],"judge_question":"Is this a pizza?",'
            '"anchor":{"looks":["a pizza"],"judge_question":"Is this a pizza?"},"window":"same_day"}]}')
    P = _plan(same, "show me that pizza we ate")
    assert P.albums[0].judge_question == "Is this a pizza?" and P.albums[0].anchor is None


def test_heldout_34_fixes():
    from datetime import date
    from findpics.converse import resolve_relative as r, _bad_q
    T = date(2026, 10, 3)
    assert r("last thurs", T) == ("2026-10-01", "2026-10-02") and r("last Friday night", T) == ("2026-10-02", "2026-10-03")
    # place already set + anchor that only names it -> place filter, no anchor
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a group at a concert?","place":"London",'
              '"anchor":{"looks":["a skyline"],"judge_question":"Is this London?"},"window":"same_event"}]}',
              "the group picture at the concert in London")
    assert P.albums[0].place == "London" and P.albums[0].anchor is None
    assert _bad_q("Was this photo taken with a telephoto lens?") and _bad_q("Is this video recorded at 60 fps?")
    assert _bad_q("Is this a RAW file?") and _bad_q("Is this person Jay?") and not _bad_q("Is there a dog on a beach?")
    # a selfie is the owner even without "I"
    P = _plan('{"albums":[{"name":"a","person":"me","judge_question":"Is this a selfie in a dorm?"}]}',
              "the selfie with the bad lighting at the dorm", owner="Reza")
    assert P.albums[0].person == "me"


def test_round5_fixes():
    from datetime import date
    from findpics.converse import resolve_relative as r, _bad_q
    T = date(2026, 10, 3)
    assert r("may till july last year", T) == ("2025-05-01", "2025-08-01")
    assert r("between january and march of this year", T) == ("2026-01-01", "2026-04-01")
    assert r("the week of Christmas", T) == ("2025-12-22", "2025-12-29")
    assert not _bad_q("Does this photo show a lens flare?") and _bad_q("Was this taken with a wide angle lens?")
    # anchor with no question: filled from its looks (was a validation crash)
    P = _plan('{"albums":[{"name":"a","anchor":{"looks":["a Christmas tree"],"judge_question":null},"window":"same_week"}]}',
              "images from the week of Christmas", today=T)
    assert P.albums[0].anchor.judge_question == "Does this photo show a Christmas tree?"
    # planner's own phrase not said -> the said phrase is recovered and computed
    P = _plan('{"albums":[{"name":"a","time_phrase":"May through July 2025","date_from":"2025-05-01","date_to":"2026-05-01"}]}',
              "anything from may till july last year", today=T)
    assert (P.albums[0].date_from, P.albums[0].date_to) == ("2025-05-01", "2025-08-01")
    # fallback never keeps a name question
    bad = '{"albums":[{"name":"a","person":null,"judge_question":"Is this Reza?"}]}'
    assert _plan(bad, "photos from", people=["Reza"], owner="Reza").albums[0].judge_question is None


def test_round5b_fixes():
    from datetime import date
    from findpics.converse import _bad_q
    T = date(2026, 10, 3)
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there a Christmas tree?","time_phrase":"christmas",'
              '"date_from":"2025-12-25","date_to":"2025-12-26"}]}', "make the lighting brighter on the christmas tree pics", today=T)
    assert P.albums[0].date_from is None
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there a Christmas tree?","time_phrase":"christmas",'
              '"date_from":"2025-12-25","date_to":"2025-12-26"}]}', "photos from christmas", today=T)
    assert P.albums[0].date_from == "2025-12-25"
    assert _bad_q("Is the time of day between 6pm and 8pm?")
    P = _plan('{"albums":[{"name":"a","judge_question":"Is a drone flying?","anchor":{"looks":["canyon"],'
              '"judge_question":"Is this the Grand Canyon?"},"window":"same_place"}]}', "the drone video over the Grand Canyon")
    assert P.albums[0].place == "Grand Canyon" and P.albums[0].anchor is None


def test_week_anchor_stays_anchor():
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there food?","anchor":{"looks":["canyon"],'
              '"judge_question":"Is this the Grand Canyon?"},"window":"same_week","exclude_question":"Is there a burger?"}]}',
              "food photos from the week I went to the Grand Canyon, no burgers")
    assert P.albums[0].anchor is not None and P.albums[0].window == "same_week" and not P.albums[0].place


def test_removed_person_name_question_dropped():
    P = _plan('{"albums":[{"name":"a","person":"Reza","judge_question":"Is this Reza?"}]}', "photos from",
              people=["Reza"], owner="Reza")
    assert P.albums[0].person is None and P.albums[0].judge_question is None


def test_readable_name_kept():
    from findpics.converse import _names_in
    assert _names_in("Is this a screenshot of a chat with Mom?", ["Mom"]) == []
    assert _names_in("Is Mom in this photo?", ["Mom"]) == ["mom"]


def test_round6_fixes():
    from datetime import date
    from findpics.converse import resolve_relative as r, _bad_q
    T = date(2026, 10, 3)
    assert r("this summer", T) == ("2026-06-01", "2026-09-01")
    assert _bad_q("Is this video exactly 15 seconds long?") and _bad_q("Is the beginning of the video trimmed?")
    P = _plan('{"albums":[{"name":"a","anchor":{"looks":["family"],"judge_question":"Is this a family gathering?"},'
              '"window":"days_before:7"}]}', "photos from the family reunion last weekend", today=T)
    assert P.albums[0].window == "same_event"
    P = _plan('{"albums":[{"name":"a","anchor":{"looks":["a repaired vase"],"judge_question":"Is the vase repaired?"},'
              '"window":"days_before:1","judge_question":"Is the vase broken?"}]}', "the broken vase before I glued it", today=T)
    assert P.albums[0].window == "days_before:1"
    P = _plan('{"albums":[{"name":"a","person":"Reza","judge_question":"Is the person in the red box standing?",'
              '"exclude_question":"Is there a person standing behind Reza?"}]}', "remove the person standing behind me", owner="Reza")
    assert "Reza" not in P.albums[0].exclude_question


def test_week_i_went_to_place_is_anchor():
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there food?","place":"Grand Canyon",'
              '"exclude_question":"Is there a burger?"}]}', "food photos from the week I went to the Grand Canyon, no burgers")
    assert P.albums[0].anchor is not None and P.albums[0].window == "same_week" and not P.albums[0].place


def test_round7_fixes():
    from datetime import date
    from findpics.converse import resolve_relative as r
    T = date(2026, 10, 3)
    assert r("last quarter", T) == ("2026-07-01", "2026-10-01")
    leak = '{"albums":[{"name":"a","looks":["a slice of bread"],"judge_question":"Is this a slice of bread?"}]}'
    good = '{"albums":[{"name":"a","looks":["a plate of pasta"],"judge_question":"Is there food on a table?"}]}'
    assert _plan([leak, good], "the food we ate at that little Italian place").albums[0].judge_question == "Is there food on a table?"
    P = _plan(leak, "the food we ate at that little Italian place")
    assert "bread" not in (P.albums[0].judge_question or "") and not P.albums[0].looks
    assert _plan(leak, "all my photos with bread").albums[0].judge_question == "Is this a slice of bread?"
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this ramen?","time_phrase":"last night","date_from":"2026-10-02",'
              '"date_to":"2026-10-03"}]}', "the spicy ramen I ate on the last night", today=T)
    assert P.albums[0].date_from is None
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this ramen?","time_phrase":"last night","date_from":"2026-10-02",'
              '"date_to":"2026-10-03"}]}', "the ramen from last night", today=T)
    assert P.albums[0].date_from == "2026-10-02"


def test_round8_fixes():
    from findpics.converse import _bad_q
    assert _bad_q("Is this Uncle Harry?") and _bad_q("Is this the brother?") and _bad_q("Is this the person in the photo?")
    assert not _bad_q("Is this a birthday party?") and not _bad_q("Is this a man wearing a uniform?")
    bad = '{"albums":[{"name":"a","looks":["a woman with long hair"],"judge_question":"Is this a woman with long hair?"}]}'
    good = '{"albums":[{"name":"a","looks":["a person"],"judge_question":"Is there a person in this photo?"}]}'
    assert "hair" not in _plan([bad, good], "videos of my sister from the summer").albums[0].judge_question
    assert _plan(bad, "my sister with long hair at the beach").albums[0].judge_question == "Is this a woman with long hair?"


def test_time_of_day_from_words():
    from datetime import date
    from findpics.converse import resolve_time_of_day as r
    assert r("between 8 and 11pm") == "20:00-23:00" and r("between 11 and 2am") == "23:00-02:00"
    assert r("after 10pm") == "22:00-04:00" and r("in the morning") == "05:00-12:00"
    assert r("last night's party") is None and r("Friday night") is None and r("the morning after the wedding") is None
    T = date(2026, 10, 3)
    P = _plan('{"albums":[{"name":"a","person":"me","judge_question":"Is this a selfie?","time_phrase":"last night",'
              '"date_from":"2026-10-02","date_to":"2026-10-03","time_of_day":"yes"}]}',
              "selfies taken between 11pm and 2am last night", today=T, owner="Reza")
    assert P.albums[0].time_of_day == "23:00-02:00" and P.albums[0].date_to == "2026-10-04"
    P = _plan('{"albums":[{"name":"a","judge_question":"Is there a dog?","time_of_day":"20:00-23:00"}]}', "my dog photos")
    assert P.albums[0].time_of_day is None      # hours the person never said are removed


def test_unknown_people_grounded():
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a beach?"}],"unknown_people":["Jay","Bob","Sara"]}',
              "photos of me and jay at the beach", people=["Sara"], owner="Reza")
    assert P.unknown_people == ["Jay"]          # Bob was never said; Sara is already known
    P = _plan('{"albums":[{"name":"a"}],"unknown_people":["my sister"]}', "photos of my sister")
    assert P.unknown_people == ["my sister"]


def test_name_it_reply():
    from findpics.cli import _NAME_IT
    m = _NAME_IT.fullmatch("Jay is 4")
    assert m and m.group(1) == "Jay" and m.group(2) == "4"
    assert _NAME_IT.fullmatch("my sister is group 7").group(1) == "sister"
    assert _NAME_IT.fullmatch("photos of the dog") is None


def test_with_people_grounding():
    P = _plan('{"albums":[{"name":"a","person":"me","with_people":["Dad","Pierce"]}]}', "photos of me with Dad and Pierce",
              people=["Dad"], owner="Reza")
    assert P.albums[0].with_people == ["Dad"] and "Pierce" in P.unknown_people
    P = _plan('{"albums":[{"name":"a","with_people":["Mom","Dad"]}]}', "Mom and Dad together", people=["Mom", "Dad"])
    assert P.albums[0].person == "Mom" and P.albums[0].with_people == ["Dad"]


def test_until_grounding():
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a landscape on a road?","anchor":{"looks":["a driver"],'
              '"judge_question":"Is there a driver?"},"window":"same_event","until":{"looks":["a citadel"],'
              '"judge_question":"Is this a citadel?"}}]}', "the landscape shot after we photographed the driver but before the citadel")
    assert P.albums[0].window == "after" and P.albums[0].until is not None
    P = _plan('{"albums":[{"name":"a","judge_question":"Is this a statue?","until":{"looks":["a painting"],'
              '"judge_question":"Is this a painting?"}}]}', "statues before the first painting photo")
    assert P.albums[0].anchor.judge_question == "Is this a painting?" and P.albums[0].window == "before" and P.albums[0].until is None


def test_with_people_must_be_said_and_invisible_events():
    P = _plan('{"albums":[{"name":"a","person":"me","with_people":["Dad","Mom","Sara"],"judge_question":"Is this a lake house?"}]}',
              "find the picture of us all together at the lake house", people=["Dad", "Mom", "Sara"], owner="Reza")
    assert P.albums[0].with_people == []
    bad = ('{"albums":[{"name":"a","looks":["a man"],"judge_question":"Is there a man?","anchor":{"looks":["a man"],'
           '"judge_question":"Is this a photo of a man?"},"window":"since","until":{"looks":["a hospital"],'
           '"judge_question":"Is this the moment the person passed away?"}}]}')
    P = _plan(bad, "photos of my brother before he passed away")
    assert P.albums[0].until is None


def test_invisible_event_forms():
    from findpics.converse import _bad_q
    assert _bad_q("Is this the brother passing away?") and _bad_q("Is this the person getting sick?")
    assert not _bad_q("Is the person in the red box smiling?")
