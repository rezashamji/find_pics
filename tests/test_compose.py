"""RESULTS 59d (adopted 10-15): composed judge questions - trip/place questions from place metadata (+ the rest judged
at the place), "A or B" by parts. findpics.compose is the product copy of the pre-registered eval/facts_scorer/route.py
rules: both must decompose every question the same way."""
import sys
from pathlib import Path

import numpy as np

from findpics import compose as C

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "facts_scorer"))

PLACES = ["Viñales, Pinar del Río, Cuba", "Pinagsanjan, Laguna, Philippines", "Machupicchu, Urubamba, Cusco, Peru",
          "León, Castilla y León, España", "Madrid, España", "Olympic Park, Stratford, London, United Kingdom",
          "Soho, London, United Kingdom", "San Juan, Puerto Rico", "Pompei, Napoli, Italia", ""]
PI = C.PlaceIndex(PLACES)


def rows(m):
    return [int(i) for i in np.where(m)[0]]


def test_tolerant_place_matching():
    assert rows(PI.mask("Leon")) == [3]                      # accents folded ("León" is not "le n")
    assert rows(PI.mask("Machu Picchu")) == [2]              # spaces removed = one address token
    assert rows(PI.mask("Pagsanjan Falls")) == [1]           # edit distance 2 + natural-feature word optional
    assert rows(PI.mask("Pompeii")) == [8]                   # edit distance 1, >= 6 characters
    assert rows(PI.mask("London Olympics")) == []            # "london" matches alone: no fuzzy "olympics" ~ "olympic"
    assert rows(PI.mask("Madrid or León")) == [3, 4]


def test_trip_question_nothing_left_is_metadata_only():
    n = C.decompose("Is this a photo taken during a trip to Puerto Rico?", PI)
    assert n[0] == "meta" and rows(n[1]) == [7] and n[2] is None
    n = C.decompose("Is this a photo from a hiking trip in Madrid in July 2010?", PI)
    assert n[0] == "meta" and n[2] is None                   # a trip's kind and dates are not pixels


def test_trip_question_with_a_visual_condition_keeps_it():
    n = C.decompose("Is this a photo with a sepia tone taken in Madrid?", PI)
    assert n[0] == "meta" and rows(n[1]) == [4] and n[2] == ("leaf", "Is this a photo with a sepia tone?")
    q = "Is there a real little boy anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one), and was this photo taken at a Pompeii?"
    n = C.decompose(q, PI)
    assert n[0] == "meta" and n[2][1].startswith("Is there a real little boy anywhere in this photo (not a drawing")


def test_or_split_and_shared_head():
    assert C.split_parts("Is there a statue or a church in this photo?") == \
        ("or", ["Is there a statue in this photo?", "Is there a church in this photo?"])
    assert C.split_parts("Is there a player wearing a maroon or white jersey in this photo?") == \
        ("or", ["Is there a player wearing a maroon jersey in this photo?", "Is there a player wearing a white jersey in this photo?"])
    assert C.split_parts("Is there a real jaguar or tiger anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?")[1][1] == \
        "Is there a tiger anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?"
    assert C.split_parts("Is there a red car, or are there people in this photo?") == \
        ("or", ["Is there a red car?", "Are there people in this photo?"])
    assert C.split_parts("Is this a photo taken with or showing a Google Glass device?") is None
    assert C.split_parts("Is this a black and white photo?") is None
    assert C.split_parts("Is this building in real life or not?") is None       # one yes/no question
    assert C.split_parts("Is this a real dog or a toy dog?")[0] == "or"
    assert C.split_parts("Is there a crowd visible in this photo? Is the ground muddy?")[0] == "and"


def test_evaluate_union_meta_and():
    node = ("meta", np.array([True, False, True]), ("or", [("leaf", "A?"), ("leaf", "B?")]))
    P = {"A?": {0: .9, 2: .1}, "B?": {0: .2, 2: .8}}
    out = C.evaluate(node, np.array([0, 1, 2]), lambda rr, q: [P[q][int(r)] for r in rr])
    assert np.allclose(out, [.9, .01, .8])


def test_product_copy_equals_the_preregistered_rule():
    import route
    qs = ["Is there a statue or a church in this photo?",  # (the product's "or not" guard is not in route.py) "Is this a photo of a small or miniature bottle?",
          "Is the person in the photo flying or in the air, or on the water (boat or swimming)?",
          "Is this image blurry or out of focus?", "Is there a dog and also a cat in this photo?"]
    for q in qs:
        assert C.split_parts(q) == route.split_parts(q)
        assert C.remainder(q + " taken in Madrid", "Madrid") == route.remainder(q + " taken in Madrid", "Madrid")
    R = route.PlaceIndex(PLACES)
    for name in ["Leon", "Machu Picchu", "Pagsanjan Falls", "London Olympics", "Pompeii", "Puerto Rico"]:
        assert rows(PI.mask(name)) == rows(R.mask(name))
