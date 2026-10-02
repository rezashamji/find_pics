"""Read-only guarantee: product code never deletes, moves or overwrites library files."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "findpics"
FORBIDDEN = [r"\bos\.remove\(", r"\bos\.unlink\(", r"\bshutil\.rmtree\(", r"\bshutil\.move\(", r"\bos\.rename\(",
             r"\.unlink\(", r"\bdelete_photo", r"\.delete\(", r"\bremove_from_album", r"\bos\.replace\("]


def test_no_destructive_calls():
    hits = []
    for p in SRC.rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if p.name == "albums.py" and "SAFE-LINK-ONLY" in line:
                continue  # the single guarded symlink remover (see remove_album_link + test below)
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, "destructive calls found:\n" + "\n".join(hits)


def test_apple_writer_is_dry_run_by_default():
    from findpics.albums import write_apple_album
    msg = write_apple_album("x", ["a", "b"])
    assert msg.startswith("[dry-run]")


def test_remove_album_link_refuses_real_files(tmp_path):
    import pytest
    from findpics.albums import remove_album_link, write_folder_album
    import pandas as pd
    photo = tmp_path / "photo.jpg"; photo.write_bytes(b"x")
    d = write_folder_album("A", pd.DataFrame(dict(item_id=["1"], path=[str(photo)])), tmp_path / "albums")
    link = next(d.glob("*_photo.jpg"))
    with pytest.raises(ValueError):
        remove_album_link(photo)            # a real file: refused
    remove_album_link(link)                 # the album link: removed
    assert photo.exists() and not link.exists()
