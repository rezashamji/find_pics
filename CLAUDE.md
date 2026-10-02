# find_pics — standing rules for Claude (auto-loaded; survives compaction)

Owner: Reza (rezashamji). Overnight autonomous build started 2026-10-02 ~00:45 ET.

## Resume protocol (do this first after any compaction or wake-up)
1. `cd /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics && source env.sh`
2. Read `PLAN.md` -> the STATUS block at the top says which phase/task is next.
3. `tail -80 JOURNAL.md`, `git log --oneline | head -20`, `squeue -u rshamji`.
4. Continue the next unfinished task. Do not wait for Reza. Do not re-plan from scratch.
5. Before ending a turn: update STATUS in PLAN.md, append to JOURNAL.md, `git commit`, and make sure
   a wake-up is scheduled (ScheduleWakeup, dynamic /loop) unless the whole plan is done.

## Hard rules
- EVERYTHING lives inside this folder. No files in /tmp, ~, or elsewhere. `source env.sh` sets caches.
- Reza's photos are READ-ONLY. `data/private/` is chmod 700. Never delete/modify/move originals.
  Product code must never call any delete API; the only write to Apple Photos is creating albums
  and adding items, dry-run by default.
- Private data and anything derived from it (thumbnails, contact sheets, embeddings, results)
  stays in `data/private/` and is never committed or pushed. Only code/docs/public-data metrics go to git.
- Product uses open-source models only. No paid API calls per image.
- Look at raw outputs. Every reported result set gets a contact sheet that Claude actually views
  (Read tool on the image) before any claim is made. Numbers without raw-trace review do not count.
- Every claim carries its denominator (e.g., "41/50 audited correct", not "82%").
- Heavy compute goes through Slurm (login node has no GPU). Account `kempner_mzitnik_lab`. In practice tonight only
  partition `kempner` (A100-40GB) started promptly; kempner_h100/h200 estimated next-day starts; `mzitnik_lab` account
  cannot submit to FASRC gpu/sapphire/shared. Bad node: holygpu8a19102 (excluded in templates). Cap: 16 GPUs.
  Templates: slurm/templates/index_array.sbatch (main env), vlm_job.sbatch (vLLM env; SCRIPT="..." via env, eval'd).
- If Reza's export appears in data/private/apple_export: find his Apple People name in library_metadata.json
  (persons containing "Reza"), then `bash scripts/run_private.sh "<name>"`; audit EVERY image in both albums by eye
  (crops via engine.person_crop), notes in data/private/audits/ (never committed).
- Journal (`JOURNAL.md`) at every milestone, decision, failure, and job submission. Commit after each.
- Push to GitHub only to `git@github.com:rezashamji/find_pics.git` once Reza has created it (SSH works).
- Writing style for Reza: concise, root-level mechanisms, define jargon inline, no emoji, push back when he is wrong.
- Time limits (Reza, 01:55): every download / install / long step gets an explicit timeout and a progress check
  (e.g. cache size growth over 30 s). If it is stuck or slower than expected, stop and reassess: switch source
  (PyPI vs custom index, HF mirror), shrink it, or run it as a Slurm job (compute nodes have internet). Never wait blindly.
- Git push rule: push ONLY `main` (`git push -u origin main`), never `--all`/`--mirror`. Branch `pre-scrub-backup` and
  refs/original/ still contain public-dataset image renders (CelebA/IMDB, no redistribution); delete them once Reza
  confirms (`git branch -D pre-scrub-backup && rm -rf .git/refs/original`).
- PRODUCT GOAL (Reza, 13:00 10-02): must run on a phone; slow is acceptable. Ship two honest modes: fast (minutes,
  with a stated bound) and exhaustive/"more accurate, takes longer" (near-oracle). Search quality on common AND small
  objects must be good or nobody uses it; measure every mode against the oracle (judge on every item).
  Agreed order: (1) exhaustive mode + fast-vs-oracle gap, (2) reference photos for ANY subject (not just Apple-tagged
  humans) + pet-identity test, (3) per-crop vectors only where (1) shows the cheap stage loses photos.
- "WHO/WHAT" MUST BE ANYTHING (Reza, 13:30): person, animal, thing, place, a picture of a picture. Kinds of sameness:
  person=face model (done), animal=animal re-ID (MegaDescriptor-B-224 is clean vs DogFaceNet), thing=instance vectors on
  detector crops, place=landmark retrieval, picture-of-picture=copy detection. Every kind gets its own measured number
  on a public set (or self-generated known-answer set) before it is claimed. Open-vocab detector: OWLv2 (Apache-2.0).
