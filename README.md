# find_pics

Ask your photo library a question in plain English, and get back albums plus an honest receipt:
how many items were checked, and a statistical lower bound on how many of the true matches it found.

Real output from the public test library (19,218 photos and videos), one sentence in, two albums out:
```
$ findpics ask ~/fp_index "Find every photo of Drew Barrymore from the 1990s, and all my photos that have bread in them."

Album 'Photos with bread': 918 items.
  Library: 19,218 items; in scope after date/media filters: 19,098. Every in-scope item was scored by the fast models.
  Scored all 19,098 in-scope items with the fast models; the judge looked at the top 3,800 plus a random 1,000 of the
  remaining 15,298. The random check found 7 more match(es). Completeness: about 90%; at least 82% with 95% confidence
  (at most ~201 matches could still be hiding). These numbers are relative to the AI judge's yes/no answers.
Album 'Drew Barrymore 1990s': ... Identity from face matching only; weaker face matches are listed separately for you
  to confirm and are NOT in the album.
Review page: ~/fp_albums/index.html
```
How honest is "at least 82%"? On test queries where the true answer is known, the stated lower bound was at or below
the true completeness in 9 of 9 runs, and in 290 of 300 simulated libraries (the 95% target allows about 15 misses in
300). Details: [eval/RESULTS.md](eval/RESULTS.md).

## Why this exists
- Apple Photos and Google Photos already do natural-language search. They never tell you what they missed.
  Apple's People album only merges faces it is very sure about. It holds the rest under "Review More Photos" and
  never says how many there are.
- Their classic search assigns each photo a few words from a fixed list (~1,300 tags), each with its own score
  cutoff. If a photo's "bread" score was just under the cutoff, searching "bread" can never find it.
- find_pics stores a vector per photo instead of words, so any query can be compared later. A local AI judge checks
  the candidates, then checks a random sample of everything else. From that sample it computes a guaranteed-style
  lower bound on completeness, the same "elusion test" lawyers use to certify document review.

Details and sources: [research/00_SYNTHESIS.md](research/00_SYNTHESIS.md).

## What it does
1. **Index once** (like Apple's overnight analysis): faces (detector + identity vectors), an image-text vector per photo
   and per sampled video frame, plus dates.
2. **Ask in plain English.** A local LLM turns the sentence into a plan: who, what it should look like, dates, photo/video,
   and how many albums. Code then executes the plan exactly.
3. **Judge.** A local vision-language model (Qwen3.5) answers a yes/no question for each top candidate. Faces it isn't
   sure about are checked side by side against a reference photo of the person.
4. **Audit.** A random sample of everything not returned is judged too. An exact binomial bound turns that sample into
   "at least X% complete, 95% confidence".
5. **Albums.** You get a folder of links plus a manifest. On a Mac it can also create Apple Photos albums: it only
   creates albums and adds items, and does a dry run unless you pass `--apply`.

**Read-only by design.** No code path deletes, moves or edits your photos, and a test enforces that.

## Install
Linux with an NVIDIA GPU (fast path):
```bash
uv venv --python 3.12 --python-preference only-managed .venv && source .venv/bin/activate   # managed Python ships headers vLLM's compiler needs
uv pip install -e ".[gpu]"
uv pip uninstall onnxruntime && uv pip install --reinstall onnxruntime-gpu   # insightface pulls CPU onnxruntime; keep only the GPU one
```
Mac (Apple Silicon, macOS 14 or later; the current MLX only ships wheels for 14+):
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e ".[mac]"
```
Both dependency sets resolve: checked with `uv pip compile` for x86_64 Linux and for arm64 macOS 15 (vllm 0.30.0 /
mlx-vlm 0.7.4, which has a Qwen3.5 module). The Mac judge (`mlx-community/Qwen3.5-4B-4bit`) is **untested on real
Mac hardware**.

## Use with Apple Photos
1. Export read-only with osxphotos: see [docs/MAC_EXPORT.md](docs/MAC_EXPORT.md).
   Name yourself and your family in Photos' People album first; those tags become the reference faces.
2. `findpics scan ~/fp_export ~/fp_index --metadata ~/fp_export/library_metadata.json`
3. `findpics index ~/fp_index` (or the Slurm array script on a cluster)
4. `findpics ask ~/fp_index "..." --out ~/fp_albums --me "Your Name"`; add `--apple-apply` to create the albums in Photos.

Android / Google Photos: Google's API can no longer read your whole library, so use Google Takeout and point `scan`
at the unzipped folder. Dates are read from the Takeout JSON files.

## Honest limitations
- The completeness bound is relative to the AI judge. If the judge says "no" to a real match, the bound can't see it.
  The report says how many judge answers a human checked.
- Certifying high completeness for rare things in huge libraries needs many judge calls: minutes on a GPU, hours on a laptop.
- Judgments like "looks heavier" are the judge's opinion, not a measurement. Vision-language models have known weight bias.
- Face recognition across large weight change is not well studied. Measure it on your own photos (Apple tags give pairs).
- Model licenses: the InsightFace face models (buffalo_l) and Apple's MobileCLIP are **non-commercial**. Fine for personal
  use; a commercial product needs the swaps listed in [research/03_models_sota.md](research/03_models_sota.md).
