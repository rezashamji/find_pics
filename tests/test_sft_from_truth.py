"""RESULTS 53: sft_from_truth.check understands the RESULTS 52/53 truth programs (place_any; several albums = the
UNION for one message of a conversation), so TRAIN truths stay usable for planner training."""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "complex_bench"))

from findpics.converse import Plan  # noqa: E402
from sft_from_truth import check, key_text  # noqa: E402

T = date(2014, 7, 9)


def P(*albums):
    return Plan.model_validate({"albums": [dict(name="a", **a) for a in albums]})


def test_key_text_keeps_place_any():
    assert "place_any" in key_text([{"name": "x", "place_any": ["Dubrovnik", "Zagreb"]}])


def test_place_any_key_accepts_a_place_list():
    K = [{"name": "a", "place_any": ["Burgos", "León"], "need": ["Is there a church in this photo?"]}]
    said = "photos from the Spain trip in Burgos and León only the church photos"
    assert check(P(dict(place="Burgos and León", judge_question="Is there a church?")), K, said, T, "h") == []
    assert check(P(dict(place="Madrid", judge_question="Is there a church?")), K, said + " Madrid", T, "h")


def test_union_of_conditions_as_one_or_album():
    K = [{"name": "lions", "need": ["Is there a lion in this photo?"], "place": "Phoenix"},
         {"name": "bobcats", "need": ["Is there a bobcat in this photo?", "Is the bobcat sleeping?"], "place": "Phoenix"}]
    said = "the zoo trip with the big cats; just the lions; include the bobcats too, the ones sleeping"
    assert check(P(dict(judge_question="Is there a lion or a sleeping bobcat in this photo?")), K, said, T, "h") == []
    probs = check(P(dict(judge_question="Is there a lion in this photo?")), K, said, T, "h")
    assert any("'or' question" in p for p in probs)
    # two albums for a two-album union: checked pairwise as before
    assert check(P(dict(judge_question="Is there a lion?"), dict(judge_question="Is there a sleeping bobcat?")),
                 K, said, T, "h") == []
    # three albums for two: refused
    assert check(P(dict(judge_question="a"), dict(judge_question="b"), dict(judge_question="c")), K, said, T, "h")


def test_union_of_places_needs_a_place_list():
    K = [{"name": "d", "place": "Dubrovnik"}, {"name": "z", "place": "Zagreb"}]
    said = "photos from the Europe trip; only the ones in Dubrovnik; also add the ones from Zagreb"
    assert check(P(dict(place="Dubrovnik and Zagreb")), K, said, T, "h") == []
    assert any("place list" in p for p in check(P(dict(place="Dubrovnik")), K, said, T, "h"))


def test_union_dates_must_cover_every_part():
    K = [{"name": "jan", "need": ["Is there a Google Glass?"], "date_from": "2014-01-01", "date_to": "2014-02-01"},
         {"name": "oct", "need": ["Is there a Google Glass?"], "date_from": "2013-10-01", "date_to": "2013-11-01"}]
    said = "photos with the Google Glass; only January; also include the ones from last October"
    ok = P(dict(judge_question="Is there a Google Glass?", date_from="2013-10-01", date_to="2014-02-01"))
    assert check(ok, K, said, T, "h") == []
    short = P(dict(judge_question="Is there a Google Glass?", date_from="2014-01-01", date_to="2014-02-01"))
    assert any("do not contain" in p for p in check(short, K, said, T, "h"))


def test_g_is_still_pairwise():
    K = [{"name": "a", "need": ["Is this photo taken at a concert?"]}, {"name": "b", "need": ["Is there a dog?"]}]
    assert check(P(dict(judge_question="Is this a concert?")), K, "concerts vs dogs", T, "g") == ["1 albums vs key 2"]
