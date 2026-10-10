#!/bin/bash
# Mac side of the Apple Photos head-to-head (MAC_INBOX M29 step d). Runs Photos' own search through AppleScript
# (`search for`, which Photos' dictionary describes as the same as typing in its Search field) for the 6 queries in
# one style and writes eval/apple_photos/results_apple_<style>.tsv in the format eval/score_apple_photos.py reads:
#   query <TAB> search_text <TAB> n_returned <TAB> filenames (comma-separated, in the order Photos returns them)
# Usage (repo root, Photos open on the TEST library, never Reza's System Photo Library):
#   bash scripts/apple_photos_search.sh keyword|phrase_of|phrase_with
#   bash scripts/apple_photos_search.sh --one "dog"      # test the AppleScript on one query first
# Read-only: it only searches. It never creates albums, edits or deletes anything.
set -euo pipefail
cd "$(dirname "$0")/.."

search() {   # $1 = search text -> "<n>\t<fn1,fn2,...>"
  osascript - "$1" <<'APPLESCRIPT'
on run argv
  set q to item 1 of argv
  tell application "Photos"
    set r to search for q
    set out to {}
    repeat with m in r
      set end of out to (filename of m)
    end repeat
  end tell
  set AppleScript's text item delimiters to ","
  return ((count of out) as text) & tab & (out as text)
end run
APPLESCRIPT
}

if [ "${1:-}" = "--one" ]; then
  t0=$(date +%s); res=$(search "$2"); echo "n=$(echo "$res" | cut -f1)  ($(( $(date +%s) - t0 )) s)"
  echo "$res" | cut -f2 | tr ',' '\n' | head -10
  exit 0
fi

style=$1
case $style in
  keyword)     texts=("dog" "car" "bicycle" "beach" "sunset" "food") ;;
  phrase_of)   texts=("photos of a dog" "photos of a car" "photos of a bicycle" "photos of a beach" "photos of a sunset" "photos of food") ;;
  phrase_with) texts=("photos with a dog" "photos with a car" "photos with a bicycle" "photos with a beach" "photos with a sunset" "photos with food") ;;
  *) echo "style must be keyword | phrase_of | phrase_with"; exit 1 ;;
esac
keys=(dog car bicycle beach sunset food)
out=eval/apple_photos/results_apple_${style}.tsv
tmp=$out.part
printf "query\tsearch_text\tn_returned\tfilenames\n" > "$tmp"
for i in 0 1 2 3 4 5; do
  res=$(search "${texts[$i]}")
  printf "%s\t%s\t%s\n" "${keys[$i]}" "${texts[$i]}" "$res" >> "$tmp"
  echo "${keys[$i]} '${texts[$i]}': $(echo "$res" | cut -f1) returned"
done
mv "$tmp" "$out"
echo "wrote $out"
