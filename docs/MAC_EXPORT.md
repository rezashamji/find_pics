# Export your Apple Photos library for find_pics (read-only)

`osxphotos export` and `osxphotos query` only READ the Photos library. Nothing is deleted or changed in Photos.
We do not use `--download-missing` (would pull every original from iCloud and fill the disk); items whose
original is only in iCloud are exported as Photos' own preview image (`--preview-if-missing`), which is enough
resolution for search.

Run in Mac Terminal. Keep the laptop plugged in with the lid open (`caffeinate` blocks idle sleep).

```bash
# ---- 0. one-time install ----
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install --python 3.13 osxphotos
osxphotos persons    # lists people Photos knows. Note the exact name it uses for you.
# If you get a permission error: System Settings > Privacy & Security > Full Disk Access > enable Terminal, reopen Terminal.

# ---- 1. one SSH login that stays open all night (upload never re-asks for 2FA) ----
cat >> ~/.ssh/config <<'EOF'

Host fasrc
  HostName login.rc.fas.harvard.edu
  User rshamji
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 16h
  ServerAliveInterval 60
EOF
ssh -fN fasrc        # password + 2FA once
ssh fasrc echo ok    # must print "ok" with no prompt

# ---- 2. export + upload ----
ME="Reza Shamji"     # EXACTLY as printed by `osxphotos persons`
OUT=~/fp_export; mkdir -p "$OUT"
DEST="fasrc:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/data/private/apple_export/"
SHRINK='sips -Z 1600 -s formatOptions 80 {filepath|shell_quote} >/dev/null 2>&1 || true'
COMMON=(--skip-edited --skip-live --skip-bursts --skip-raw --convert-to-jpeg --jpeg-quality 0.85
        --preview-if-missing --filename "{uuid}" --sidecar json --ramdb
        --post-command exported "$SHRINK" --post-command-error continue)

# STAGE 1 (photos). Mac disk is nearly full, so each piece is uploaded and then its LOCAL COPY in ~/fp_export is
# removed (rsync --remove-source-files only touches ~/fp_export, never the Photos library). Peak Mac use = one year.
osxphotos query --json > "$OUT/library_metadata.json"                         # every item: date, place, people, size
rsync -a --partial "$OUT/library_metadata.json" "$DEST"
caffeinate -is osxphotos export "$OUT/A_me" --person "$ME" "${COMMON[@]}"     # A: everything Apple tagged as you
rsync -a --partial --remove-source-files "$OUT/A_me" "$DEST"
for y in $(seq 2000 2026); do                                                  # B: all photos, one year at a time
  [ -f "$OUT/done_$y" ] && continue                                            # safe to re-run after an interruption
  caffeinate -is osxphotos export "$OUT/B_all_photos/$y" --only-photos \
    --from-date "$y-01-01" --to-date "$((y+1))-01-01" "${COMMON[@]}" &&
  rsync -a --partial --remove-source-files "$OUT/B_all_photos/$y" "$DEST/B_all_photos/" &&
  touch "$OUT/done_$y" && echo "year $y uploaded"
done
echo STAGE 1 DONE
```
If you see photos from before 2000, change `seq 2000 2026` to start earlier (the metadata file shows the oldest date).

STAGE 2 (videos, 40k): do NOT run yet. After stage 1, `library_metadata.json` on the cluster tells us how many videos
are only in iCloud, their total size and length; from that Claude computes the exact space/time and a shrink-before-
upload command (the search samples one frame every 2 s at ~1,000 px, so full-quality video is never needed).

What each piece is for:
- `library_metadata.json`: dates, GPS, and Apple's own person tags and scene labels for every item. Lets us measure
  what Apple found vs what we find, and inspect why a search like "bread" fails.
- `A_me`: Apple's own "you" album: reference faces across years + Apple's baseline answer.
- `B_all_photos/<year>`: the full photo library shrunk to 1600 px on the long side (~0.3-0.5 MB each; 147k photos ≈ 45-75 GB on
  the cluster), so we can find what Apple missed. Uploaded year by year; local copies removed after upload.
- `--filename "{uuid}"`: each file is named by Photos' internal id, so it joins exactly to the metadata and can be
  added back to an album later.
