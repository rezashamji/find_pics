"""Apple export path: files named {uuid}.jpeg / {uuid}_preview.jpeg + `osxphotos query --json` records
(format checked against osxphotos source: list of PhotoInfo.json(shallow=False); date isoformat with tz; '_UNKNOWN_')."""
import json

from PIL import Image

from findpics.ingest import scan


def test_osxphotos_export(tmp_path):
    lib = tmp_path / "exp"; (lib / "2016").mkdir(parents=True); (lib / "2026").mkdir()
    Image.new("RGB", (8, 8)).save(lib / "2016" / "AAAA-1111.jpeg")
    Image.new("RGB", (8, 8)).save(lib / "2026" / "BBBB-2222_preview.jpeg")   # iCloud-only original -> preview
    (lib / "2026" / "CCCC-3333.mov").write_bytes(b"\0")
    meta = [
        {"uuid": "AAAA-1111", "date": "2016-07-04T12:34:56.123456-04:00", "persons": ["Reza Shamji", "_UNKNOWN_"],
         "labels": ["Food", "Bread"], "ismovie": False, "isphoto": True},
        {"uuid": "BBBB-2222", "date": "2026-09-01T08:00:00-04:00", "persons": [], "labels": [], "ismovie": False},
        {"uuid": "CCCC-3333", "date": "2026-08-01T08:00:00-04:00", "persons": ["Reza Shamji"], "labels": [], "ismovie": True},
    ]
    (tmp_path / "library_metadata.json").write_text(json.dumps(meta))
    items = {i.item_id: i for i in scan(lib, tmp_path / "library_metadata.json")}
    assert set(items) == {"AAAA-1111", "BBBB-2222", "CCCC-3333"}
    a = items["AAAA-1111"]
    assert a.apple_persons == ["Reza Shamji"] and a.apple_labels == ["Food", "Bread"]
    assert a.taken.startswith("2016-07-04T12:34:56") and a.date_source == "metadata_json"
    assert items["CCCC-3333"].media == "video"
    assert items["BBBB-2222"].taken.startswith("2026-09-01")
