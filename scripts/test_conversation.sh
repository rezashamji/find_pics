#!/bin/bash
# End-to-end conversation on the public test library (GPU, vLLM env). 3 messages in one --out folder.
# Message 1 streams until every photo is judged (round lines show how the album and bound grow over time).
# Checks afterwards: message 2 adds an exclusion; message 3 removes part of it; cache reuse makes 2 and 3 cheap.
set -e
OUT=data/public/chat_test; rm -rf "$OUT"
C="python -m findpics.cli ask"
$C "all my photos with bread" --index data/public/index_testlib --out $OUT --today 2026-10-02
$C "drop the sandwiches and burgers" --out $OUT --today 2026-10-02
$C "actually keep the sandwiches" --out $OUT --today 2026-10-02
ls $OUT
