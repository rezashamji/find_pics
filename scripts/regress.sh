#!/bin/bash
# End-to-end regression on public data (GPU, vLLM env): (1) bread conversation, 3 messages, streamed to the end;
# (2) the transformation demo on the Apple-format public library (Kevin Bacon heavier vs fit), then a follow-up.
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
PART=${1:-all}   # all | bread | bacon
if [ "$PART" != bacon ]; then rm -rf data/public/regress_bread
printf '%s\n' "all my photos with bread" "drop the sandwiches and burgers" "actually keep the sandwiches" | \
  python -m findpics.cli chat --index data/public/index_testlib --out data/public/regress_bread --today 2026-10-02
fi
if [ "$PART" != bread ]; then rm -rf data/public/regress_bacon
printf '%s\n' "Find every photo and video of me where I look heavier or out of shape, and the best photos and videos of me from 2010 to 2015 where I look fit. Make two albums: heavier, and fit." \
  "only the ones where I'm outdoors" | \
  python -m findpics.cli chat --index data/public/index_apple_like --out data/public/regress_bacon --me "Kevin Bacon" --today 2026-10-02
fi
echo REGRESS_DONE
