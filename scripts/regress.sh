#!/bin/bash
# End-to-end regression on public data (GPU, vLLM env): (1) bread conversation, 3 messages, streamed to the end;
# (2) the transformation demo on the Apple-format public library (Kevin Bacon heavier vs fit), then a follow-up;
# (3) a person from 3 --ref photos ("Kev": expect ~322 photos + 4 videos); (4) "my dog Max" from 3 --ref photos
# (expect maxdog_0..2 ranked first; "outdoors" keeps maxdog_2 only).
set -e
cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics
PART=${1:-all}   # all | bread | bacon | ref | dog
if [ "$PART" = all ] || [ "$PART" = bread ]; then rm -rf data/public/regress_bread
printf '%s\n' "all my photos with bread" "drop the sandwiches and burgers" "actually keep the sandwiches" | \
  python -m findpics.cli chat --index data/public/index_testlib --out data/public/regress_bread --today 2026-10-02
fi
if [ "$PART" = all ] || [ "$PART" = bacon ]; then rm -rf data/public/regress_bacon
printf '%s\n' "Find every photo and video of me where I look heavier or out of shape, and the best photos and videos of me from 2010 to 2015 where I look fit. Make two albums: heavier, and fit." \
  "only the ones where I'm outdoors" | \
  python -m findpics.cli chat --index data/public/index_apple_like --out data/public/regress_bacon --me "Kevin Bacon" --today 2026-10-02
fi
if [ "$PART" = all ] || [ "$PART" = ref ]; then rm -rf data/public/regress_ref
E=data/public/apple_like_export/B_all_photos
printf '%s\n' "Find every photo and video of Kev." | \
  python -m findpics.cli chat --index data/public/index_apple_like --out data/public/regress_ref --today 2026-10-02 \
  --ref "Kev=$E/1988/B0298-BACON.jpeg,$E/2015/B0037-BACON.heic,$E/2008/B0088-BACON.jpeg"
fi
if [ "$PART" = all ] || [ "$PART" = dog ]; then rm -rf data/public/dogtest/chat   # needs scripts/build_dogtest.py + index
R=data/public/dogtest/refs
printf '%s\n' "photos of Max" "only the ones outdoors" | python -m findpics.cli chat --index data/public/dogtest/index \
  --out data/public/dogtest/chat --today 2026-10-03 --ref "Max=$R/max_ref0.jpg,$R/max_ref1.jpg,$R/max_ref2.jpg"
python -c "import json,glob; [print(f, [i['path'].split('/')[-1] for i in json.load(open(f))['items']]) for f in sorted(glob.glob('data/public/dogtest/chat/turn_*/*/manifest.json'))]"
fi
echo REGRESS_DONE
