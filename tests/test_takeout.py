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
