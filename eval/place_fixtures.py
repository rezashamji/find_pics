"""Golden cases for the Swift ports of converse.filter_to_place and converse.place_or_look (synthetic library)."""
import json
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, "src")
from findpics.converse import Album, filter_to_place, place_or_look

items = pd.DataFrame(dict(item_id=[f"i{k}" for k in range(6)], media=["photo"] * 5 + ["video"],
                          taken=["2020-05-01T10:00:00+00:00"] * 6, taken_local=[None] * 6, date_source=["exif"] * 6,
                          place=["Paris, Ile-de-France, FR, France", "New York City, New York, US, United States", "",
                                 "Kyoto, Kyoto, JP, Japan", "Paris, Texas, US, United States", ""]))
idx = SimpleNamespace(items=items)
cases = []
for kw in [dict(filter_question="Is the location Paris?"), dict(filter_question="Was this taken in New York City?"),
           dict(filter_question="Is this in Atlantis?"), dict(filter_question="Is the sky blue?"),
           dict(place="beach"), dict(place="beach", person="Dad", judge_question="Is the person in the red box smiling?"),
           dict(place="Kyoto"), dict(place="gym", judge_question="Is there a treadmill?")]:
    a = Album(name="a", **kw)
    cases.append(dict(album=a.model_dump(), filtered=filter_to_place(idx, a).model_dump(), placed=place_or_look(idx, a).model_dump()))
json.dump(dict(items=[dict(id=r.item_id, media=r.media, taken=None, localMinutes=None, place=r.place) for r in items.itertuples()],
               cases=cases), open("ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/places.json", "w"), indent=1, default=str)
print("PLACE_FIXTURES", len(cases))
