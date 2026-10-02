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
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, "destructive calls found:\n" + "\n".join(hits)


def test_apple_writer_is_dry_run_by_default():
    from findpics.albums import write_apple_album
    msg = write_apple_album("x", ["a", "b"])
    assert msg.startswith("[dry-run]")
