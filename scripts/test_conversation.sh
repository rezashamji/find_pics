#!/bin/bash
# End-to-end conversation on the public test library (GPU, vLLM env): `findpics chat`, models loaded once, 3 messages.
# Message 1 streams until every photo is judged. Then: exclusion added; then PART of it undone (burgers stay excluded).
set -e
OUT=data/public/chat_test; rm -rf "$OUT"
printf '%s\n' "all my photos with bread" "drop the sandwiches and burgers" "actually keep the sandwiches" | \
  python -m findpics.cli chat --index data/public/index_testlib --out $OUT --today 2026-10-02
ls $OUT
