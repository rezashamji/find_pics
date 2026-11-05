# M82 cluster anchor: facts P vs judge P on held-out public photos (the reference for #factsCompare)

**Purpose.** This folder produces the cluster-side reference numbers for the phone's #factsCompare. The pre-registered
wiring rule (MAC_INBOX M82, JOURNAL 11-05 ~14:45) was: per question with >= 5 judge-yes photos, device AUC >= cluster
AUC - .10, same writer on both sides.
AUC = the probability that a random judge-yes photo gets a higher facts P than a random judge-no photo.

**Main finding.** The full-facts-P AUC cannot test the facts chain. Here the facts chain means writer -> line -> towervec
encoder -> facts vector. On these 3 questions the head's PE image vector carries the ranking:
- Shuffling the facts lines across photos moves AUC by at most .020.
- A constant facts text still scores .97-.99.

The facts-only AUC does separate a correct chain from a broken one. It uses the same head with the image vector zeroed.
At the phone's sample size (food: 28 judge-yes in 300) it separates them in every simulated draw. The coordinator adopted
it as the device check (M89: food facts-only AUC >= .70).

## Photos (public; none in any training set of the writers or of towervec)
- **AUC set `dev5k`**: 5,000 complex_bench DEV photos (DISBench Flickr personal libraries, stored at long side 500 px).
  - Rule, fixed before any line or label was looked at: the 24,598 DEV photos sorted by (library, item_id), permuted with
    numpy `default_rng(82)`, first 5,000. Every one of the 13 DEV libraries is in it, with 366-412 photos each.
  - Extension rule (coordinator): a question with < 50 judge-yes (4B judge) gets more photos in the same seeded order.
    Dog had 39. The next 1,693 photos brought it to 50, so `dev5kext` = 6,693 photos.
- **Held out, checked in code**:
  - The trained writer p2b_all trained on TRAIN photos + the extra public set (`eval/facts_writer/train_writer2.rows`).
    The extra set excluded every DISBench photo id and every photo by a DISBench user (`extra_data.manifest`).
  - towervec `tvw_t2b_final_rs2_xfit_full` trained on `pairs_train` + `pairs_trainplan` only (TRAIN libraries; see its
    config.json). DEV question texts are only tokenised.
  - The 17,594 Commons iPhone photos on disk are NOT held out: 17,473 of them are in p2b_all's training data.
- **Parity pool `commons1k`**: the first 1,000 of a NEW Commons iPhone set (`commons_new.py`).
  - Selection: "Taken with iPhone ..." categories; not among the 17,594 on disk; EXIF year 2019-2026 (so not in Open
    Images V4 or YFCC); long side >= 1280; at most 20 per uploader; shuffled with seed 82.
  - Result: 3,000 photos downloaded at 1280 px; 2,380 of the 3,000 have GPS.

## Pipeline (each piece = the phone's)
| piece | cluster | phone |
|---|---|---|
| writer input | `r448/<key>.jpg`: long side 448, Lanczos, JPEG q80 (the exact bytes `write_facts.py` sends) | PhotoKit ~448 px rendition, `processing.resize` 448x448 |
| writer | "Search facts for this photo, one line.", greedy, max 120 new tokens, repetition penalty 1.05 (vLLM: prompt + output; MLX: context 4,096) | `Judge.generate` |
| repeat-stop | `findpics.facts_line.apply_repeat_stop(text, 2)` on the full greedy text (prefix-monotone = streaming stop) | `FactsLine` |
| facts text | `factsPrefix(place, taken) + line`; place = `findpics.ingest.nearest_place(lat, lon)` (= `Geocoder.placeText`), month in UTC | `FactsWriter.write` |
| f, q | towervec encoder fp32: `[CLS]+ids[:254]+[SEP]` / `[CLS]+ids[:64]+[SEP]`, mean pool | `towervec_encoder_fp16` (MLXEmbedders) |
| v | PE-Core-B-16 fp16 image tower on the same r448 file (open_clip squash to 224), unit vector | the index's ~448 px rendition vector |
| t | PE-Core-B-16 text tower of `scorerQuestion(q)` (= q for these 3) | `emb.vector(of:)` |
| head | `tvw_t2b_final_rs2_xfit_full` head, fp32 (= `towervec_head.fphead`) | `head.fold(q:t:).p(f:v:)` |
| judge | `findpics.vlm.VLLMJudge` prob mode: thumbnail <= 896 px, JPEG q88, "<q> Answer with one word: yes or no.", P(yes)/(P(yes)+P(no)) | `visionJudge.pYes`, 896 px |
| yes | judge P >= 0.7; facts P >= 0.7 | `SearchEngine.accept` = 0.7 |

**Judges.**
- `j4` is the phone's judge and the one every number below uses. It is `models/q3vl4b_real`: the
  mlx-community/Qwen3-VL-4B-Instruct-4bit weights (4-bit language model, bf16 vision tower), run in vLLM. App.swift's
  `visionJudge` loads that id; no judge adapter is bundled.
- `j9` is Qwen/Qwen3.5-9B, the judge behind the cluster cascade's cached answers (RESULTS 59d-70). It is optional (the
  coordinator said so) and is still queued on kempner_requeue (`fp_da82_j9_dev5k`, `fp_da82_j9_devext_2000`). score.py
  adds its rows once its files exist.

**Deviation from the phone.** DISBench files are 500 px, so the judge saw 500 px, not 896.

**Writers.**
- (A) BASE = mlx-community/Qwen3-VL-2B-Instruct-4bit. The AUC set uses vLLM on its de-quantized phone form
  `models/qwen3vl_2b_mlx4sim`, which equals the MLX weights (RESULTS 61d (2a)).
- (C) TRAINED = p2b_all. The AUC set uses its RESULTS 61d DEV lines (vLLM phone form, same rendition bytes;
  `short_format.clean` collapsed whitespace, which does not change WordPiece ids).
- Real MLX (mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4, container `tools/mlxpy`) ran only on `dev500` and on the parity set.
  - It ran on gpu_test A100 MIG slices.
  - The base model segfaulted (exit 139) when it ran in the same process right after the trained model (twice). Each
    base run was finished in a fresh process; the writer resumes from its jsonl.

## (A) and (C): AUC per question (4B judge, product prefix, AUC [95% CI], photo bootstrap 2,000 draws, seed 0)
`dev5kext` (6,693 photos; dog reaches 50 judge-yes):

| writer | q | n | judge-yes | facts-yes | missed by facts | extra from facts | AUC [95% CI] |
|---|---|---|---|---|---|---|---|
| (A) base, vLLM | dog | 6,693 | 50 | 42 | 9 | 1 | .995 [.989, .999] |
| (A) base, vLLM | food | 6,693 | 191 | 116 | 86 | 11 | .983 [.975, .989] |
| (A) base, vLLM | beach | 6,693 | 192 | 118 | 84 | 10 | .984 [.975, .991] |
| (C) trained, 61d vLLM | dog | 6,693 | 50 | 46 | 6 | 2 | .996 [.990, .999] |
| (C) trained, 61d vLLM | food | 6,693 | 191 | 110 | 88 | 7 | .979 [.971, .986] |
| (C) trained, 61d vLLM | beach | 6,693 | 192 | 129 | 69 | 6 | .990 [.984, .995] |

`dev5k` (5,000 photos): base .994 / .982 / .979 (judge-yes 39 / 149 / 143); trained .998 / .978 / .987.

Writer output on dev5k:
- base: mean 21.2 tokens; repeat-stop fired 0/5,000 (2/6,693 with the extension).
- trained: mean 74.7 tokens; repeat-stop fired 1/5,000.

Place prefix sensitivity: using the benchmark's own Nominatim address string instead of the product's GeoNames
`placeText` (towervec was trained with the Nominatim form) changes every AUC by <= .001.

## Ablations: what the AUC is made of (dev5k, 4B judge; dog / food / beach, judge-yes 39 / 149 / 143)
| facts P variant | trained writer | base writer |
|---|---|---|
| full (product) | .998 / .978 / .987 | .994 / .982 / .979 |
| lines shuffled across photos (default_rng(0)) | .986 / .962 / .967 | .990 / .966 / .969 |
| constant: prefix + EMPTY line | .989 / .974 / .971 | (same texts) |
| image vector zeroed = facts-only | .980 / .925 / .950 | .891 / .897 / .896 |
| facts-only, lines shuffled | .498 / .509 / .509 | .490 / .532 / .519 |

Paired drops (same bootstrap draws):

| comparison | writer | dog | food | beach |
|---|---|---|---|---|
| full -> shuffled | trained | +.012 [+.001, +.033] | +.016 [+.008, +.025] | +.020 [+.011, +.031] |
| full -> shuffled | base | +.004 [-.002, +.014] | +.015 [+.008, +.023] | +.010 [+.001, +.022] |
| facts-only -> facts-only shuffled | trained | +.48 [+.38, +.58] | +.42 [+.37, +.47] | +.44 [+.39, +.49] |
| facts-only -> facts-only shuffled | base | +.40 [+.28, +.52] | +.37 [+.31, +.42] | +.38 [+.32, +.43] |

**Reading.**
- On these questions no device AUC of the full facts P can test the facts chain. A chain that writes another photo's
  facts loses <= .02; a chain with no facts loses <= .03.
- The ranking comes from v and t (PE image-text). Within one question t is constant, so the facts-only score (v = 0)
  ranks photos by the facts vector alone. That score is high for a correct chain and ~.5 for a broken one.

## The device statistic (`device_stat.py` -> `device_stat_dev5k.txt`)
Simulation: 2,000 draws (default_rng(0)) of 28 judge-yes + 272 judge-no food photos from dev5k. That is the phone's
cached set: 300 photos, 28 food judge-yes.

| statistic | writer | correct chain: median [95% of draws] | broken (shuffled) chain | best cut, error rate |
|---|---|---|---|---|
| full facts P AUC | trained | .979 [.954, .995] | .963 [.932, .987] | .971: correct below in 27%, broken above in 29% |
| facts-only AUC (v = 0) | trained | .926 [.863, .974] | .507 [.411, .610] | .671: 0/2,000 either way |
| full facts P AUC | base | .982 [.962, .996] | .967 [.938, .990] | .976: 26% / 26% |
| facts-only AUC (v = 0) | base | .898 [.827, .958] | .532 [.429, .630] | .696: 0/2,000 either way |

Adopted bar (coordinator, M89): food facts-only AUC >= .70.

A judge-free fallback is weaker: Spearman(facts-only P, cos(v, t)) on 300 random photos.
- food, trained: correct .30 [.19, .40] vs broken .00 [-.11, .12]; the extreme draws overlap (min correct .08, max
  broken .21).
- beach: .45 vs .01.
- dog, base writer: -.09. Not usable for dog.

## MLX vs vLLM (dev500 = the first 500 of dev5k; 4B judge; judge-yes dog 3, food 16, beach 13)
| writer | food AUC MLX vs vLLM, paired diff [95% CI] | beach | exact lines (after repeat-stop) | mean tokens MLX / vLLM |
|---|---|---|---|---|
| base | .974 vs .973, +.001 [-.009, +.014] | .968 vs .963, +.005 [-.001, +.014] | 1/500 | 23.6 / 21.8 |
| trained | .959 vs .972, -.013 [-.042, +.001] | .972 vs .975, -.002 [-.015, +.010] | 154/500 | 75.1 / 75.7 |

Dog has 3 judge-yes in dev500, so it is not compared.

The engine does not move the AUC. The base writer's lines almost never match across engines. Prompt token counts are
identical, and the chat template and generation config are identical. The divergence starts at the FIRST token: MLX
mostly opens with "This photo ..." and vLLM with "The photo ..." / "The photograph ...". So a free-form caption
amplifies a near-tie at token 1. That is a hypothesis: the first-token margins were not measured.

The trained writer's 154/500 exact sits at RESULTS 61d's level (MLX vs vLLM 19/50, vLLM vs itself at batch 1 26/50).
Exact-match parity is therefore only meaningful for the trained writer; the base writer's parity lines can only be
compared loosely.

## (B) Writer parity set: `writer_parity.json`
20 photos from commons1k:
- 2 = the only pool photos whose BASE vLLM line looped. Both ran to the 120-token cap (e.g. "L. M. T. A. R. C. E. D. I.
  N. G. ..."). Repeat-stop v2 fired on 0 of the 1,000 base lines: the base 2B's loops are not comma lists.
- 18 = k-means (k = 18, seed 82) on PE vectors, the photo nearest each centre.

URLs:
- Commons serves only standard thumbnail widths. On 11-05, 448, 336, 400, 450, 480, 640 and 800 px returned HTTP 400
  "Use thumbnail sizes listed on https://w.wiki/GHai"; 250, 330, 500 and 960 returned 200.
- Each URL uses the standard width whose long side is closest to 448: 500x375 landscape, 330x440 portrait (3:4),
  250x444 (9:16).
- The cluster scaled the downloaded file by the phone's best-fit factor min(448/w, 448/h) (Pillow BICUBIC, saved PNG):
  500x375 -> 448x336, 330x440 -> 336x448. It then ran mlx-vlm's processor.
- Recorded per photo: url, sha256 of the downloaded file, resized size fed, licence, author, base and trained lines
  (after repeat-stop) and raw text, tokens with and without the stop, repeat-stop flags, and `line_trained_k48`. That is
  the first 48 tokens of the same greedy run (the phone's K = 48), decoded with the checkpoint's tokenizer, then
  repeat-stop.

Result:
- base MLX lines are 16-40 tokens.
- trained MLX lines are 40-109 tokens; 19/20 reach the 48 cap.
- No repeat-stop fired in MLX for either writer. Neither of the 2 base vLLM loops looped in MLX (27 and 29 tokens):
  near-tie differences between engines.
- Image files are not committed (data/public/device_anchor82/parity/).

## What I looked at
- 10 trained lines next to their photos (500 px, all DISBench has). 9/10 right in scene, objects and OCR (e.g. "VE 2764
  15 02 2014 14 13" read correctly). 1 invented OCR ("DAF" for a DHL banner). Puppies in a market basket were named.
- 12 judge decisions (500 px):
  - dog 4/4 right, including "no" on a fluffy chicken the writer called a dog.
  - food: the 4B is lenient. Sake barrels and a market entrance were both judge-yes at P = 1.000. A woman grinding paste
    on a metate is a defensible yes. A seagull with food is correctly no.
  - beach: one river-bank and one pebble close-up are judge-yes. One rocky shore with a sand patch is no. A distant beach
    is yes.
  - Many "missed by facts" photos are judge errors, not facts errors.
- 3 parity photos at 900 px: concert stage, cafe cup ("850" read), drone-model exhibit. Trained lines right; base caption
  right.

## Commands
```
python eval/device_anchor82/prep.py dev                  # manifest of all 24,598 DEV photos + r448 renditions (CPU)
python eval/device_anchor82/commons_new.py --n=3000      # new Commons iPhone set (API + 1280 px thumbnails)
python eval/device_anchor82/prep.py commons
python eval/device_anchor82/subset.py                    # dev5k, dev500, commons1k, pool6k (= dev5k + commons1k)
bash eval/device_anchor82/launch.sh dev5k j4             # every GPU job: scripts/race_sbatch.sh (kempner_requeue) and/or
bash eval/device_anchor82/launch.sh pool6k pe            #   the same command with SPECS=gpu_test:gpu:1 ACCT=mzitnik_lab
bash eval/device_anchor82/launch.sh pool6k vb            #   (what actually ran: gpu_test A100 MIG slices)
bash eval/device_anchor82/launch.sh devext_2000 j4 ; ... devext_2000 vb ; ... devext_2000 pe
bash eval/device_anchor82/stage2.sh                      # extension size, scoring + ablations, device simulation (GPU job)
python eval/device_anchor82/parity.py pick               # parity set: selection, download, sha256, best-fit input
bash eval/device_anchor82/writer_job.sh 0 1 data/public/device_anchor82/out/writer_parity20 <manifest_parity20.parquet>
python eval/device_anchor82/parity.py assemble           # -> writer_parity.json
bash eval/device_anchor82/writer_job.sh 0 1 data/public/device_anchor82/out/writer_dev500 <manifest_dev500.parquet>
python eval/device_anchor82/score.py --manifest=dev500 --tag=dev500 --pairs=base_mlx:base_vllm,trained_mlx:trained_vllm61d
bash eval/device_anchor82/launch.sh dev5k j9             # optional 9B (queued)
```
Output files:
- `results_dev5k.{json,txt}` (with ablations and pairs), `results_dev5kext.{json,txt}`, `device_stat_dev5k.{json,txt}`,
  `writer_parity.json`, `results_dev500.{json,txt}` (when done).
- Per-photo tables: `data/public/device_anchor82/per_photo_*.parquet`.
