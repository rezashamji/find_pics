"""When a request names ONE specific pet / thing, ask for its photos (findpics.subjects; the phone's Subjects.swift
is checked against these rules on eval/subjects_fixtures.py's cases)."""
from findpics.converse import Plan
from findpics.subjects import named_subjects, resolve_subject, subject_condition, subject_plan


def _plan(*albums, unknown=()):
    return Plan.model_validate(dict(albums=[dict(dict(name="x", person=None, looks=[], judge_question=None), **a)
                                            for a in albums], unknown_people=list(unknown)))


def test_named_pet_asks_and_keeps_the_condition():
    P = _plan(dict(name="Max at the beach", person="Max", judge_question="Is Max on a beach?"))
    asks = named_subjects(P, "my dog Max at the beach")
    assert [(a.name, a.kind, a.albums) for a in asks] == [("Max", "dog", [0])]
    assert asks[0].prompt == "Show me Max: pick 1-3 clear photos of Max."
    Q = subject_plan(P, asks)
    assert Q.albums[0].person is None and Q.albums[0].judge_question == "Is the dog on a beach?"


def test_owned_thing_and_category_only_question():
    P = _plan(dict(name="car", judge_question="Is there a real blue car anywhere in this photo (not a drawing, painting, "
                                              "statue, toy, model or picture of one)?"))
    asks = named_subjects(P, "my blue car")
    assert [(a.display, a.key) for a in asks] == [("your blue car", "my:blue car")]
    assert subject_plan(P, asks).albums[0].judge_question is None     # only restates the subject


def test_no_ask_for_categories_negations_compounds_places():
    for msg in ["a red car", "dogs at the park", "my dogs", "photos without my dog", "my car keys", "my dog's bowl",
                "selfies in my car", "photos of my wife"]:
        assert named_subjects(_plan(dict(name="x")), msg) == [], msg


def test_person_on_a_subject_album_must_also_be_in_the_photo():
    P = _plan(dict(name="me and Max", person="me", with_people=["Max"]), unknown=["Max"])
    asks = named_subjects(P, "me and my dog Max hiking")
    Q = subject_plan(P, asks)
    assert Q.albums[0].person is None and Q.albums[0].with_people == ["me"] and Q.unknown_people == []


def test_saved_subject_by_name_and_resolution():
    P = _plan(dict(name="Max", person="Max", judge_question="Is Max at the beach?"))
    saved = [dict(name="Max", kind="dog", words="dog"), dict(name=None, kind="car", words="blue car")]
    asks = named_subjects(P, "Max at the beach", saved=saved)
    assert asks and asks[0].from_saved and resolve_subject(asks[0], saved) == 0
    assert named_subjects(P, "Max at the beach") == []       # never saved: Max stays a person (face sheet)
    car = named_subjects(_plan(dict(name="x")), "my car")[0]
    assert resolve_subject(car, saved) == 1                  # the one saved car
    assert subject_condition("Is this a photo of a car?", car) is None
