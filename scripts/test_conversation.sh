#!/bin/bash
# End-to-end conversation on the public test library: 3 messages in one --out folder (GPU, vLLM env).
# Checks: follow-up keeps the bread album and adds an exclusion; judge cache reuse; "look harder" -> every photo judged.
set -e
OUT=data/public/chat_test; rm -rf "$OUT"
C="python -m findpics.cli ask"
$C "all my photos with bread" --index data/public/index_testlib --out $OUT --today 2026-10-02
$C "drop the sandwiches and burgers" --out $OUT --today 2026-10-02
$C "look harder" --out $OUT --today 2026-10-02
ls $OUT
