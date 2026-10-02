"""Google Takeout (Android path): dates come from sidecar photoTakenTime, under all three sidecar naming schemes."""
import json

from PIL import Image

from findpics.ingest import scan


def test_takeout_sidecar_variants(tmp_path):
    d = tmp_path / "Takeout" / "Google Photos" / "Photos from 2019"; d.mkdir(parents=True)
    names = {"IMG_0001.jpg": ".json",                                    # pre-2024 exports
             "IMG_0002.jpg": ".supplemental-metadata.json",              # late-2024+ exports
             "PXL_20190704_123456789.MP.jpg": ".supplemental-metadat.json"}  # clipped at 46 chars
    ts = {"IMG_0001.jpg": 1546300800, "IMG_0002.jpg": 1556668800, "PXL_20190704_123456789.MP.jpg": 1562243696}
    for n, suf in names.items():
        Image.new("RGB", (8, 8)).save(d / n)
        (d / (n + suf)).write_text(json.dumps({"title": n, "photoTakenTime": {"timestamp": str(ts[n])}}))
    items = {i.path.split("/")[-1]: i for i in scan(tmp_path / "Takeout")}
    assert set(items) == set(names)
    assert items["IMG_0001.jpg"].taken.startswith("2019-01-01")
    assert items["IMG_0002.jpg"].taken.startswith("2019-05-01")
    assert items["PXL_20190704_123456789.MP.jpg"].taken.startswith("2019-07-04")
    assert all(i.date_source == "takeout" for i in items.values())


def test_place_from_takeout_gps_and_osxphotos(tmp_path):
    from findpics.ingest import scan
    d = tmp_path / "lib"; d.mkdir()
    Image.new("RGB", (8, 8)).save(d / "a.jpg")
    (d / "a.jpg.supplemental-metadata.json").write_text(json.dumps(
        {"photoTakenTime": {"timestamp": "1700000000"}, "geoData": {"latitude": 35.6762, "longitude": 139.6503}}))
    Image.new("RGB", (8, 8)).save(d / "U1.jpeg")
    (tmp_path / "meta.json").write_text(json.dumps([{"uuid": "U1", "date": "2020-01-01T00:00:00+00:00", "latitude": 41.66,
        "longitude": -70.3, "place": {"name": "Hyannis, MA", "names": {"city": ["Barnstable"], "country": ["United States"]}}}]))
    items = {i.path.split("/")[-1]: i for i in scan(d, tmp_path / "meta.json")}
    assert "Tokyo" in items["a.jpg"].place                      # offline reverse geocode of Takeout GPS
    assert "Hyannis" in items["U1.jpeg"].place and "Barnstable" in items["U1.jpeg"].place   # Apple place names


def test_place_scope_and_grounding():
    import numpy as np, pandas as pd
    from findpics.engine import scope_mask
    from findpics.planner import AlbumSpec, parse_plan
    from findpics.store import Index
    items = pd.DataFrame(dict(item_id=list("abc"), path=list("abc"), media="photo", taken="2020-01-01T00:00:00+00:00",
                              place=["Shibuya, Tokyo, JP", "Cambridge, Massachusetts, US", ""]))
    idx = Index(None, items, pd.DataFrame(), np.zeros((0, 4)), pd.DataFrame(), np.zeros((0, 512)), pd.DataFrame())
    assert scope_mask(idx, AlbumSpec(name="x", judge_question="q", place="Tokyo")).tolist() == [True, False, False]
    P = parse_plan('{"albums":[{"name":"a","judge_question":"q","place":"Paris"}]}', "photos of my dog in Tokyo")
    assert P.albums[0].place is None   # invented place removed
