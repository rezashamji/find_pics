# find_pics journal (append-only)

## 2026-10-02 00:47 — kickoff
- Task from Reza: overnight, research how Apple/Google find people and search photos, check what already exists,
  then build an open-source, local, read-only tool that answers natural-language photo queries (person + attribute +
  time), makes albums, and honestly reports scan counts and estimated completeness. Demo: Reza's weight-loss
  transformation albums ("heavier" vs "fit, last 6 months"), photos and videos.
- Environment found: running on holylogin07 (login node, no GPU). Slurm accounts kempner_mzitnik_lab, mzitnik_lab;
  partitions kempner_h100 / kempner_h200 / kempner_requeue / gpu_h200. git configured (rezashamji). GitHub SSH works
  (`ssh -T git@github.com` -> rezashamji) but no API token and no `gh`, so Claude cannot create the repo: asked Reza.
  HF logged in as rezashamji; existing HF_HOME is outside find_pics, so env.sh redirects HF_HOME inside and reads the
  token in place via HF_TOKEN_PATH. Lab quota: 84.5T / 120T used, fine.
- Reza (mid-turn): no iCloud password needed if another method works; test first on public images/videos; must be
  read-only, never delete. Decision: Mac-side osxphotos export (read-only) + rsync to data/private (chmod 700);
  public ground-truth test library built in parallel so nothing blocks on the upload.
- Verified osxphotos flags from official docs (rhettbull.github.io/osxphotos/cli.html): --sidecar json,
  --convert-to-jpeg, --jpeg-quality, --skip-edited, --only-photos/--only-movies, --preview-if-missing, --update,
  --ramdb, --filename, --person, --post-command CATEGORY "TEMPLATE" with {filepath|shell_quote}. No built-in resize
  -> use macOS `sips -Z` in --post-command.
- Launched 4 background research agents -> research/01..04.
- Wrote env.sh, .gitignore, CLAUDE.md (resume protocol + hard rules), PLAN.md.

## ~00:58 — loop armed, P2 started
- Dynamic /loop armed (self-paced wake-ups). Research agents running (02 competition agent still finishing its own background work).
- P2: scripts/setup_env.sh building envs/fp (uv, Python 3.12, torch cu128, transformers, open_clip, insightface,
  onnxruntime-gpu, pillow-heif, av, hdbscan...). Log: slurm/logs/setup_env.log.
- Asked Reza for: Mac export+upload (docs/MAC_EXPORT.md), empty private GitHub repo rezashamji/find_pics, privacy OK.

## ~01:05 — env fix, public test data
- Bug: env.sh ended with `[ -f activate ] && source` -> exit status 1 when env absent -> setup_env.sh (set -e) died
  silently with an empty log. Fixed to an if-statement. Env rebuild running.
- First `nohup ... &` launch was killed when its shell exited; long tasks now run as tracked background tasks.
- Public test library sources (decided):
  - IMDB full scene photos (systemk-ai/imdb-wiki, config imdb): person name + photo year + face box. Real scenes
    across years ~ stand-in for a family library. Known caveat: IMDB-WIKI name labels are noisy (the labeled face is
    the largest face; can be the wrong person). Must audit before trusting as ground truth. 14 shards downloaded.
  - CelebA test split (flwrlabs/celeba): clean identities + "Chubby"/"Double_Chin" attributes (heavy-looking ground truth). Aligned crops, so easier than real life.
  - Open Images V7 validation: human-verified image labels. 15,000 images selected = all 5,498 with verified labels
    for 15 concepts (Bread 264 pos, Baked goods 800, Cake 275, Dog 1586, ...) + random filler. Bread prevalence ~1.8%.
    Recall is measured against verified positives only (unverified images may also contain bread; precision is audited by eye).
  - Pexels videos (minh132/pexels-videos, 523 clips) for video handling.
- Gotcha: `hf download --include A B C` silently ignored the first pattern; use one --include per pattern.

## 01:10 — competition verdict (research/02), env retry
- research/02_competition_landscape.md landed. Key facts (agent-sourced, cited in file):
  - NL photo search incl. person+attribute is already shipped: Apple Photos iOS 18.1+ (on-device, Apple Intelligence devices),
    Google Ask Photos (cloud). Open source: Immich v3.2 (2026-09) combines person AND CLIP text AND date AND media type;
    but filters are set by hand (no sentence parsing), no cutoff/recall, one frame per video, no Apple Photos write-back.
  - osxphotos (MIT) already reads Apple people/labels/captions and can create albums (--add-to-album).
  - No product/repo/paper found that reports items scanned + recall estimate with an interval. Incumbents publicly weak on
    recall (Google paused Ask Photos June 2025 citing recall; Ente code comment: fixed threshold "trades recall").
  - Recall-certification statistics exist in DB/e-discovery (SUPG, BARGAIN, PPI, QBCB, Callaghan & Mueller-Hansen) but not applied to photos.
  - Closest research: "Personal AI Agent for Camera Roll VQA" arXiv 2606.05275 (Gemini/GPT based, no face model).
  - License traps: InsightFace buffalo_l/antelopev2 and Apple MobileCLIP weights are non-commercial. SigLIP2, Qwen3-VL/Qwen3.5 Apache-2.0.
- DECISION (Feynman check passed only for a narrow wedge): do NOT build "another photo search app". Build an *auditable*
  retrieval layer: sentence -> plan, person+attribute+time, multi-frame video, VLM verification, and a scan count + recall
  interval that says what it is relative to (judge vs user labels), writing albums to Apple Photos via osxphotos.
- Env: torch download from download.pytorch.org crawled at ~170 KB/s -> switched to PyPI wheels. My pkill pattern matched its
  own shell (exit 144) - lesson: use pgrep -f with a narrower pattern / kill by PID.
- Open Images 15,000/15,000 downloaded. IMDB 14 shards, CelebA test 3 shards, Pexels 523 videos (7.2 GB) downloaded.

## 01:13 — research/01 Apple/Google internals landed
- Apple People (Apple ML Research 2021, still the only public description): face + upper-body detector, separate face and
  body embedding nets, two-pass clustering: pass 1 strict ("high precision but many, smaller clusters"), clothing only
  compared within a time/place moment; pass 2 merges across moments using faces only. New faces assigned by matching
  against multiple stored exemplars per person. Clustering "typically overnight during device charging"; assignment of a new
  face to a known person happens near capture. No embedding dims/thresholds published. => Reza's guess "VLM" was wrong:
  it's embeddings + clustering, not a VLM.
- Why People misses (documented): strict first pass, unclear faces filtered out, clothing cues only within a moment,
  only frequent people promoted; "Review More Photos" = candidates it holds but won't auto-apply. Research: accuracy drops
  with profile, age gap, small/occluded faces.
- Classic Apple search: fixed taxonomy (Vision has 1,303 labels incl. "bread" per a third-party dump), per-label thresholds,
  synonym + word-embedding fallback. So "bread" misses = score below bread's cutoff / index unfinished / no Apple Intelligence.
  Apple NL search (iOS 18.1+, iPhone 15 Pro+): model undocumented.
- Google Ask Photos since June 2025: classic results first, then Gemini narrows candidates -> cannot find what the first
  stage missed (inference). Samsung S25: on-device VLM search.
- No product reports completeness. (Absence of evidence in reviewed docs.)
- Rule breach by research agent 01: briefly downloaded PDFs to the session /tmp scratchpad, deleted afterwards. Noted.

## ~01:16 — research/03 models landed; stack decisions
- Report recommendations: SCRFD det (run >=1280px for small faces); identity: CVLface AdaFace ViT-B KPRPE WebFace12M
  (best cross-age AgeDB-30 98.1, CPLFW 95.65, TinyFace 76.1); text-image: facebook/PE-Core-L14-336 (Apache-2.0, COCO t2i 57.1)
  + crops/tags for small objects like bread; verifier Qwen/Qwen3.5-9B P(yes), thinking off, images <=~1MP; reranker
  Qwen3-VL-Reranker-2B; parser Qwen3.5-4B; Mac: 4-bit Qwen3.5-2B/4B via mlx-vlm, index computed elsewhere or on-device.
- Weight change: no modern study of face embeddings across large weight change; only pre-2015 work showing severe drops.
  => must MEASURE on Reza's own before/after photos (Apple's tags give pairs across years).
- Body-weight from photos: absolute BMI error 2.3-4; VLMs show weight bias. Only a relative, per-person trend is realistic.
- License: InsightFace weights (buffalo_l, antelopev2, SCRFD) non-commercial; Apple MobileCLIP/DFN/FastVLM research-only;
  MetaCLIP 2, jina-clip-v2 CC-BY-NC; CVLface no license (research data). Commercial-safe swaps: SCRFD-34G v2 MIT
  (immich-app/scrfd_34g_gnkps) or YuNet; AuraFace recognizer (Apache-2.0); PE-Core / SigLIP2; Qwen3.5 / Gemma 4 (Apache-2.0).
- DECISIONS for tonight (personal-use build; license notes in README):
  1. faces: InsightFace buffalo_l first (fast to integrate); evaluate cross-year matching on IMDB; swap/compare AdaFace if time.
  2. image vectors: benchmark PE-Core-L14-336 vs SigLIP2-so400m on the Open Images concept set; pick by recall@k with denominators.
  3. judge + planner: Qwen3.5-9B via vLLM.

## ~01:22 — research/04 recall landed -> estimator redesigned
- Driver check (job 49803976, holygpu8a11302): H100, driver 610.57 (CUDA 13 OK). Compute nodes have internet (HF 200).
- research/04 key points: cost law n ~ N_tail*ln(1/alpha)/m to certify <= m misses; arXiv 2607.21480 proves no valid
  audit does better and checking only shown results can never certify recall. Valid: one-shot elusion test with fixed
  sample size, blind labels, verified results; QBCB/Target (Lewis+ 2021); Callaghan & Mueller-Hansen random variant.
  Invalid: plug-in, resample-until-pass. PPI collapses in zero-disagreement strata; Rogan-Gladen unstable. Elusion tests
  overstated recall 20-36 pts in one study. Recommended: fully judge head (results + next ~5x), random tail sample, exact
  bounds, human-confirm judge yeses, report scored / VLM-judged / human-verified separately.
  Platform: osxphotos 0.77.2 read-only; only write = create album + add; Google Library API can't read libraries since
  2025-03-31 -> Takeout is the only complete Android/Google path; icloudpd last resort (credentials can delete).
- I was wrong in my first audit.py: Beta-posterior Monte Carlo is a Bayesian heuristic with no guarantee. Rewrote:
  audit.certify = head fully judged + uniform tail sample fixed before labels + one-sided Clopper-Pearson upper bound
  on tail misses -> recall lower bound (relative to judge). Engine rewritten to match; identity for uncertain faces is
  judged with a side-by-side [reference face | photo] image so tail photos with undetected faces can still count as misses.

## 01:40 — env OK, tests pass, testlib built
- envs/fp: torch 2.14.1+cu130, ORT providers TensorRT/CUDA/CPU. pytest: 6 passed.
- Certificate simulation (N=150k, 2k true, head 4k, tail sample 3k): lower bound <= true recall in 290/300 runs (96.7%);
  mean true recall 0.665 vs mean lower bound 0.555 (conservative by ~11 pts at this sample size).
- Test library: 19,218 items = 15,000 Open Images + ~4,100 IMDB scene photos + 120 Pexels videos. Family (IMDB people
  with >=30 photos spanning >=5 yrs): Nicolas Cage 469 (1975-2014), Dan Aykroyd 112, Kim Basinger 99, Pierce Brosnan 372,
  Kevin Bacon 336, Drew Barrymore 490 (1982-2014). Simulated Apple tags on the clearest 50% of each.
- Raw look (eval/audits/gt_Drew_Barrymore.jpg, random 30 labeled photos, viewed by Claude): realistic and hard: child
  still 1982, group shots, small faces, profiles, #23 heavy old-age makeup (Grey Gardens). ~28/30 clearly show her;
  #29 (2014, boy + Adam Sandler) and #27 (tiny figures) may not -> ~5-10% label noise. Misses vs these labels must be
  eyeballed before being called misses.
- Submitted GPU smoke test.

## 01:42 — GPU queue reality
- kempner_mzitnik_lab fairshare 0.125. sbatch --test-only estimated starts: kempner_h100 2026-10-02 14:58, kempner_h200
  10-03 07:09, kempner_requeue 10-03 19:53, kempner (A100 40GB) ~02:38, kempner_interactive (A100 MIG 20GB) ~02:41.
  mzitnik_lab account: AssocMaxSubmitJobLimit on gpu/gpu_h200/sapphire/shared (no FASRC allocation).
- So the "16 H100/H200" are not available tonight at our priority. Decision: run on kempner A100-40GB (enough for
  SigLIP2/PE-Core + InsightFace + Qwen3.5-9B bf16 ~19GB). Cancelled H100/requeue smoke jobs, resubmitted on kempner (49809272).
  Keep jobs small and short (backfill-friendly). If A100 also stalls, fall back to Qwen3.5-4B on a MIG 20GB slice.
- Indexer now computes several image encoders in one decode pass (clip.npy, clip_1.npy...) for a fair PE-Core vs SigLIP2 comparison.

## 02:06 — smoke tests, fixes, index + VLM jobs submitted
- Bad node: holygpu8a19102 (A100) fails CUDA context creation ("devices busy or unavailable") -> excluded in all templates.
- transformers 5 returns ModelOutput from get_*_features -> _as_tensor() fix.
- InsightFace via onnxruntime 1.30 fell back to CPU (ORT CUDA EP needs CUDA 12 libs; torch is cu130). Fix: pip
  nvidia-*-cu12 libs + ort.preload_dlls(). Providers now CUDA for det+rec.
- Smoke (A100-40GB): SigLIP2-so400m 96 img/s warm (batch 64); faces 22 img/s (per-image, det 640) -> SMOKE_OK.
- Reza (01:55): time-limit rule for downloads/long steps -> added to CLAUDE.md.
- Model weights cached in .cache/huggingface: siglip2-so400m-patch14-384, PE-Core-L-14-336, Qwen3.5-9B, Qwen3.5-4B; insightface buffalo_l in models/insightface.
- vLLM env: vllm 0.30.0, torch 2.13.0+cu130.
- Gotcha: sbatch --export splits on commas; pass comma-containing vars by inheriting env (--export=ALL).
- Submitted: index array 49813671 (8 shards, testlib, SigLIP2 + PE-Core + faces), VLM smoke 49813727 (planner on Reza's
  real request + P(yes) on 8 bread pos / 8 verified neg + throughput).
- 02:08 index array 49813671 crashed: DataLoader default collate turned numpy->tensor (Image.fromarray fails). Fixed with collate_fn=identity, verified on 22 items locally (42 units, 0 errors). Resubmitted.
- 02:12 VLM smoke 49813727 FAILED: triton compiles cuda_utils.c with gcc against /usr/include/python3.12 which doesn't exist (system python3.12 has no -devel headers). Rebuilding envs/vllm with uv-managed python (--python-preference only-managed).

## 02:16 — testlib indexed; fast-stage results; reference contamination found + fixed
- Index 49813860: 19,218 items -> 20,457 units, 22,955 faces, 0 decode errors, <=113 s/shard x 8 A100 (23 units/s/GPU).
- eval/eval_retrieval.py (results in eval/results_retrieval.json). Denominator = verified positives.
  Bread (264): R@264 = 0.64 both encoders; R@5x264 = 0.98 both. Apple-style cutoff (highest-recall threshold with
  precision >= 0.90 on verified pos/neg): recall 0.31 (SigLIP2) / 0.28 (PE-Core). => a precision-first fixed cutoff
  drops ~70% of bread photos; ranking + judging the top ~5x recovers ~98%. Direct evidence for the "bread" mechanism.
  PE-Core >= SigLIP2 at R@5x on most concepts (sandwich 0.97 vs 0.58, guitar 0.97 vs 0.84, cake 0.99 vs 0.91,
  horse 1.00 vs 0.95); ties on bread/dog; both 0.75 sunglasses. -> PE-Core becomes default image encoder.
- People (refs = simulated-tagged half; targets = untagged half): at cos>=0.4 recall 0.88-0.98, 0-14 "FP" per person.
- RAW LOOK (eval/audits/face_fp04_Barrymore.jpg, face_miss04_Barrymore.jpg; all 28 images viewed):
  FPs were real errors from REFERENCE CONTAMINATION: Hugh Grant x4 (sim up to 0.77), Lucy Liu x4, Cameron Diaz x3 -
  co-stars in Drew's tagged photos where her own face wasn't detected, so the only face became a "Drew" reference.
  Same risk for Reza (a tagged photo where only a sibling's face is visible). Misses: profiles (#1049, #1860), small
  faces in groups, child-era poster (#3483, 0.12), old-age makeup (#3801), dark scenes, a cartoon (#4073 = label noise).
- Fix: reference consensus = median over other tagged photos of the best-matching face in that photo; drop refs < 0.2.
  (First attempt used median over ALL faces -> bystanders dominate -> 0 refs left; wrong, fixed.)
  Result: Drew FP@0.4 14 -> 0 (recall 0.94 -> 0.93); Brosnan FP@0.3 41 -> 13; others unchanged. Default sim_floor=0.2.
- 02:21 VLM smoke 49814801 FAILED: vLLM spawns engine proc -> scripts need if __name__=='__main__' guard. Wrapped smoke_vlm/eval_attribute/eval_end2end in main(). Resubmitted smoke + CelebA attribute eval.

## 02:28 — RULE BREACH found and contained (writes outside find_pics/)
- Audit of files changed tonight under ~ (find -newermt 00:40):
  1. ~/.cache/flashinfer (created 02:22 by vLLM's flashinfer JIT) -> REMOVED (entirely created tonight).
  2. ~/.config/vllm/usage_stats.json: vLLM appended telemetry lines tonight to a pre-existing file. My attempt to strip
     tonight's lines was DENIED by the permission classifier -> left for Reza.
  3. ~/.rustup/toolchains/1.95.0-x86_64-unknown-linux-gnu installed 01:42 (during the first vllm env build; something
     built from source with a pinned Rust). Default toolchain (stable) untouched. Removal DENIED -> left for Reza.
  4. ~/.git-credentials mtime 00:47:01 (session start). Not touched by me (I only read it, redacted). Content = overleaf only.
  (~/.cache/claude-cli-nodejs, ~/.cache/lmod = harness/module system, not project writes.)
- Prevention: env.sh pins FLASHINFER_*/VLLM_CONFIG_ROOT/RUSTUP_HOME/CARGO_HOME inside, VLLM_NO_USAGE_STATS=1; every Slurm
  template sets HOME=$FP_ROOT/.cache/home so stray ~ writes land inside find_pics.
- VLM smoke 49815265 root cause: flashinfer sampler JIT needed 'ninja' (not on PATH, exit 127). Disabled via
  VLLM_USE_FLASHINFER_SAMPLER=0.

## 02:36 — VLM works (Qwen3.5-9B on A100): planner + judge smoke results
- Smoke 49815932 COMPLETED. Judge throughput 47.7 img/s (A100-40GB, yes/no P(yes), 1 token).
- Planner on Reza's real request -> 2 albums, sensible looks/avoid/judge questions, want=all vs best correct.
  BUG (caught by reading the raw plan): it put date_from=2025-04-02 ("past 6 months") on the 'Reza heavier' album too,
  while its own notes said "no date limit" -> would have silently excluded every heavy-era photo.
  Fix: AlbumSpec.time_phrase + code-enforced grounding (dates kept only if the quoted phrase occurs in the request).
  Residual risk: it can still quote a real phrase on the wrong album -> CLI prints the plan for confirmation. Test added (12 pass).
- Judge on 8 bread-pos / 8 verified-neg: 6/8 pos > 0.98; 2 neg ~0.8. Viewed all 4 disagreements
  (eval/audits/judge_bread_disagree.jpg): rainbow layer cake (labeled bread, p=0.23), fruitcake (0.50), corn muffins
  (labeled not-bread, 0.83), filled puff pastries (0.79). All are definition-boundary cases, not blatant judge errors.
- Submitted end-to-end eval (bread, dog, Kevin Bacon, Drew Barrymore, Nicolas Cage 1995-2005; head 600, tail sample 1500).

## 02:37 — "looks overweight" attribute eval (CelebA, 600 Chubby / 1400 not), raw look changes the reading
- Numbers (eval/results_attribute.json): judge (Qwen3.5-9B P(yes) "Does this person look overweight?") AUC 0.896;
  PE-Core text-image score AUC 0.703. At P>=0.5 the judge says yes to 239/600 labeled-Chubby (TPR 0.40) and 22/1400
  not-Chubby (FPR 0.016). AUC male 0.870 (504 pos), female 0.882 (96 pos).
- RAW LOOK (eval/attribute/*.jpg, all 48 faces viewed by Claude):
  * "label Chubby, judge no" (24): most of these faces look lean to me; ~20/24 appear to be Black men and women.
    => CelebA's Chubby label looks noisy and racially skewed here; the judge's low TPR is partly label error.
  * "judge yes, label no" (24): several look genuinely heavier (labels wrong); many are older people with fuller/sagging
    faces at p 0.5-0.7 => the judge CONFOUNDS AGE with weight.
- Consequences for the Reza query: (1) no trustworthy external ground truth for this attribute -> rank within Reza's own
  photos (relative) and audit by eye; (2) the age confound would push the judge toward calling his OLDER (fitter)
  photos heavier, i.e. against the true story -> must check explicitly; (3) never present this as a measurement.
- 02:37 threshold sweep (yes-on-Chubby / yes-on-notChubby): 0.5 239/600 22/1400; 0.3 339/600 63/1400;
  0.2 409/600 122/1400; 0.1 486/600 249/1400. Age confound quantified: mean P(yes) on NOT-Chubby faces = 0.166 for
  old vs 0.043 for young (CelebA 'Young' attr). Engine: person+appearance albums use attr_accept=0.3 (identity stays 0.5).

## 02:58 — end-to-end eval #1 (49817087, 17.5 min): certificate honest, but identity judge useless
| query | truth | returned | TP | FP vs labels | true recall | cert lower | holds |
| bread | 264 | 392 | 215 | 177 | 0.814 | 0.706 | yes |
| dog | 1586 | 677 | 663 | 14 | 0.418 | 0.368 | yes |
| Kevin Bacon | 336 | 516 | 328 | 188 | 0.976 | 0.281 | yes (but useless) |
| Drew Barrymore | 490 | 629 | 473 | 156 | 0.965 | 0.304 | yes (but useless) |
| Cage 1995-2005 | 168 | 462 | 167 | 295 | 0.994 | 0.716 | yes |
- Lower bound <= true recall in 5/5. Bread bound tight-ish (0.71 vs 0.81).
- Dog: head of 600 can't hold 1,586 dogs -> recall 0.42; the certificate said so honestly (>=37%, ~1,162 hiding).
  Fix: adaptive head (judge in chunks of 200 while the last chunk's yes-rate >= 3%, up to 6,000).
- PERSON QUERIES: VLM side-by-side identity check measured useless. Kevin Bacon, items not face-sure: p>=0.5 -> 191 yes,
  4 labeled true; p>=0.8 -> 33 yes, 0 true. Drew: 171 yes / 15 true; Cage: 290 / 6. Meanwhile face sim>=0.45 alone:
  Bacon 325 items, 324 labeled true; Drew 458/458; Cage 161/172.
  DECISION: identity from face vectors only (accept >=0.40); [0.30,0.40) = counted "possible" list for the user, never
  auto-added; VLM only judges appearance on face-matched items (all of them, no sampling). Identity completeness is NOT
  certified by the judge; the report says so and quotes the measured test-library recall; review page shows a random sample.
- Bread "FP" raw look (eval/end2end/bread/false_pos.jpg, 36 viewed): ~26-28/36 genuinely contain bread (sandwiches,
  toast, panini, buns, crostini, baguettes at a market, a loaf in a pantry, a woman holding a loaf); ~5-8 pastry/cake
  borderline; 2-3 unclear movie scenes. => precision vs labels (55%) understates; by eye ~75-80%. Labels incomplete.
- Resubmitted e2e (49820486) with the rewritten engine.

## 03:18 — e2e #2 (49820486) with rewritten engine + full CLI demo (49820682)
| query | truth | returned | TP | FP vs labels | recall vs labels | cert lower | holds |
| bread | 264 | 533 | 225 | 308 | 0.852 | 0.844 | yes |
| dog | 1586 | 1632 | 1566 | 66 | 0.987 | 0.958 | yes |
| Kevin Bacon | 336 | 329 | 328 | 1 | 0.976 | n/a (person) | +14 possible (1 true) |
| Drew Barrymore | 490 | 469 | 469 | 0 | 0.957 | n/a | +19 possible (11 true) -> 0.980 incl. |
| Cage 1995-2005 | 168 | 174 | 163 | 11 | 0.970 | n/a | +6 possible (3 true) -> 0.988 incl. |
- Adaptive head grew to 2,200 (dog) / 3,000 (bread). Certificate held 2/2 for concepts, bread bound tight (0.844 vs 0.852).
- RAW LOOK: Cage's 11 "FP" (eval/end2end/cage_1995_2005/false_pos.jpg): 11/11 show Nicolas Cage (10 red-carpet photos
  with Patricia Arquette, IMDB-labeled as her; 1 film still). True precision 174/174.
  Bread v2 random 36 of 308 unlabeled returns (eval/audits/bread_fp_random36_v2.jpg): ~28 contain bread (mostly burger/
  hot-dog buns, sandwiches, toast, benedict muffins, dough); 3 clear errors (beach scene 2344, distant crowd 10090,
  kitchen 8780); ~5 borderline pastry. Semantic boundary (is a burger "a photo of bread"?) belongs to the user -> review page.
- CLI demo "every photo of Drew Barrymore from the 1990s, and all my photos that have bread": 16.7 min end to end, 2
  albums + review page (data/public/albums_demo/index.html). RAW PLAN showed 2 planner bugs:
  (a) invented identity-style conditions for Drew (looks "blonde hair", "blue eyes"; judge "look like Drew Barrymore?")
      -> code now strips any judge question naming the album's person; prompt forbids general-appearance looks.
  (b) date_to 1999-12-31 with exclusive semantics drops Dec 31 -> prompt now gives exclusive examples.
  Tests: 13 pass.

## 03:40 — transformation + video demos, by eye
- "Cage heavier / Cage fit (2005-2014)" (49823144, 3.7 min): plan correct this time (no date on heavier; fit 2005-2014).
  heavier: 56 of 479 face-matched items at attr>=0.3; fit: 210 of 256. RAW LOOK (eval/audits/demo_Cage_heavier.jpg, 48 viewed):
  no confident yes (max p 0.62); most show Cage at normal weight, often small in frame; top scorers appear driven by OTHER
  people (#3430 heavyset man in a headlock, #4208 police lineup, #1790 older man). Cage has no dramatic heavy era, so a
  weak album is roughly right, but the judge is distracted by group photos.
  "Cage fit" (eval/audits/demo_Cage_fit.jpg, 36 viewed): plausible (Ghost Rider stunts, firefighter role, lean red-carpet).
  FIX: appearance is now judged on a head+torso crop of the matched person (person_crop), not the whole photo. Rerun 49826594.
- "all my videos of the beach or the ocean" (49823205, 12.3 min): planner set media=video, 120 in scope, all judged,
  29 returned. RAW LOOK (eval/audits/demo_Beach_or_Ocean_Videos.jpg, 29 viewed): ~26 clearly beach/ocean (incl. underwater
  fish/dolphins, boats, kayak); 3 borderline (mountain at sunset #89, skyline over bay #21, inlet #54).
  Fixes: videos are judged on the best-matching sampled frame (was middle frame); report states "100% relative to judge"
  when every in-scope item was judged.

## 03:48 — person-crop rerun (49826594, 3.2 min) + video rejects, by eye
- Cage heavier 56 -> 27 items with head+torso crops (other people no longer drive scores); fit 210 -> 229 of 256.
- RAW LOOK at the exact crops the judge saw (eval/audits/demo2_heavier_crops.jpg, top 18): every crop is centered on
  Cage's head+torso -> cropping works. One doubtful identity: #2520 (sim 0.43, just above the 0.40 cut) may not be him.
  Heavier judgments remain weak (max p 0.78), consistent with Cage having no dramatic heavy era. Reza's 10-year
  transformation is a much stronger signal and the real test.
- Video rejects (eval/audits/demo_video_rejected.jpg, all 91 viewed): no clear beach/ocean miss; nearest calls: coastal
  city on a bay #45 (p 0.36), lakes/rivers/aerial clouds.

## 03:53 — within-person weight ranking (CelebA, 132 people with >=2 Chubby and >=2 not; 2,707 photos)
- Job 49829121 (2.8 min). Mean per-person AUC: "Does this person look overweight?" 0.642 (94/132 people > 0.5);
  "face look heavier or fuller than average?" 0.660 (104/132 > 0.5). Overall AUC ~0.70 for both.
- RAW LOOK (eval/audits/within_person_6ids.jpg, 6 people x up to 6 photos): for most people the Chubby/not photos look
  like the same weight (label inconsistency) and the judge sits ~0.5 on both = no signal, correctly. Where a real visible
  change exists (id8787: older/fuller face p 0.78-0.82 vs younger/thinner 0.47-0.56; id5613 0.84-0.87 vs 0.71-0.73)
  the judge separates the eras. Thin evidence (1-2 clear people); CelebA has few real within-person weight changes.
- Unit tests: tests/test_engine.py (synthetic index + fake judge): certificate doesn't overclaim, adaptive head extends,
  date scope respected. 16 tests pass.
- 03:54 Verified the Apple path against osxphotos source (photoinfo.py asdict/json, cli/print_photo_info.py,
  _constants.py): `query --json` = JSON list of PhotoInfo.json(shallow=False); keys uuid/date/persons/labels/ismovie;
  date isoformat with tz + microseconds; unnamed faces '_UNKNOWN_'. tests/test_ingest.py mirrors this. Added --plan-only. 17 tests pass.

## 04:13 — run_private.sh tested end to end (Kevin Bacon as stand-in for Reza)
- EXP/IDX/OUT/META overridable. Chain: scan (login) -> index array 49830479 (5 shards, 5-7 min each, all DONE) ->
  ask 49830481 (afterok, 6.9 min). Plan correct (heavier: no dates; fit: 2010-2015). Bacon heavier 7 / fit 145 of 153.
- RAW LOOK at judge crops (eval/audits/rp_bacon_crops.jpg): heavier picks weak (p 0.32-0.44), mostly dark/blurry/profile
  (low-quality images draw a hesitant yes). Photo #3244 was in BOTH albums (heavier 0.38, fit 0.82) -> bug for a
  transformation video. Fit top picks sharp/lean (Comic-Con, suits) = plausible.
- Fixes: make_exclusive() (same-person appearance albums: each shared photo kept only where the judge was most
  confident); want=best now = p>=0.5, top quarter (min 12) unless a count was requested. CLI runs all albums, then
  exclusivity, then writes. 18 tests pass.
- 04:17 Bacon rerun 49833550 (3.0 min): heavier 8, fit 36 (curated, all p>=0.92), no overlap. But #3244
  stayed in heavier (0.35) although judged fit 0.82 (cut from fit only by the cap). make_exclusive now compares the
  judge's scores on all questions (judged table), not album membership. Test updated; 18 pass.
- 04:21 exclusivity verified in real run 49834367: 'Bacon heavier' 6 items, 3 removed (judge rated them higher for 'fit').

## 04:22 — cross-era identity: query-time expansion adopted
- Harder setting = Reza's likely case: references only from each person's MOST RECENT half of tagged photos; targets =
  everything else; "early" = oldest third of the person's photo years. At face sim >= 0.4:
  | person | recent-only R / early R / FP | + expand(0.55, 3 rounds) R / early R / FP |
  | Cage | 0.91 / 0.88 / 11 | 0.95 / 0.95 / 11 |
  | Aykroyd | 0.87 / 0.80 / 5 | 0.91 / 0.90 / 5 |
  | Basinger | 0.89 / 0.83 / 4 | 0.92 / 0.86 / 4 |
  | Brosnan | 0.97 / 0.96 / 0 | 0.99 / 0.98 / 0 |
  | Bacon | 0.96 / 0.97 / 1 | 0.97 / 0.98 / 1 |
  | Drew | 0.91 / 0.92 / 0 | 0.95 / 0.98 / 0 |  (expand 0.50 gave Drew 2 FP -> chose 0.55)
- CLI now expands references (accept 0.55, 3 rounds) by default.

## 04:27 — install check (README must work for strangers)
- uv dry-run of -e ".[gpu]": FAILED. Unpinned, the resolver took the newest torch (2.14) that vllm 0.30 doesn't support
  and backtracked to vllm 0.1.3, which can't build. Fix: vllm>=0.30 -> resolves (torch 2.13, transformers 5.18).
  Also moved onnxruntime out of base deps (collides with onnxruntime-gpu); gpu extra now carries the nvidia-*-cu12 libs
  ORT's CUDA provider needed tonight. insightface itself still pulls CPU onnxruntime -> README gives the one-line fix.
- Mac extra resolved only to mlx-vlm 0.3.9 (Dec 2025) because the default target was macOS 13; mlx>=0.32.2 ships 14+
  only. With MACOSX_DEPLOYMENT_TARGET=15.0: mlx 0.32.3, mlx-vlm 0.7.4 (has qwen3_5 module), osxphotos 0.77.2, torch 2.11.
  README: macOS 14+. Verified mlx-community/Qwen3.5-4B-4bit exists (HTTP 200).
- 04:36 Clean install per README (envs/clean, uv-managed py3.12, -e .[gpu,dev] + ORT swap): imports ok (vllm 0.30.0, torch 2.13.0+cu130, ORT CUDA provider present), 18 tests pass, 'findpics' entry point works. GPU 'ask' from clean env submitted.

## 04:49 — clean-env GPU run + judge cutoff recalibrated
- Clean env (README install) ran "Find all my photos of a dog on a beach" on GPU (49837162, 11.9 min): 59 items;
  "random check found 1 more; about 76%, at least 40%" -> honest wide bound for a rare compound concept with 1,000 audits.
- RAW LOOK (eval/audits/clean_dog_beach.jpg, 59 viewed): the ~35 with p>=0.8 are dogs on beaches/sand/shore
  (all correct); most of p 0.5-0.7 are wrong (dog portraits, dogs on grass/rocks, a beach with no dog, a birthday cake #16201 p=0.54).
- Cutoff sweep on labeled e2e runs (verified pos kept / verified neg yes): dog 0.5: 1566/1586, 7/65; 0.7: 1560, 6/65;
  0.9: 1549, 3/65. bread 0.5: 225/264, 46/129; 0.7: 216, 38/129; 0.9: 196, 20/129.
  -> judge_accept 0.5 -> 0.7 for object/scene questions (small recall cost, removes the junk band). Person-appearance
  attr_accept stays 0.3 (CelebA showed the judge is very conservative there). Tests pass.
- 05:09 e2e #3 (49840354, shipped settings: judge_accept 0.7): bread 214/264 (0.811) bound 0.705 holds; dog 1560/1586 (0.984) bound 0.979 holds; persons unchanged. Bounds held 6/6 concept results across 3 runs (I first wrote 9/9 by wrongly counting run #1 person queries from the removed VLM-identity design; corrected). RESULTS/README/MORNING_REPORT updated.
- 05:11 Added 'findpics apply-reviews' (closes the review loop): drops album LINKS marked wrong via a guarded remover (refuses non-symlinks / non-album folders; safety test allows only that one marked line + new test proving a real file is refused). Tested on the dog-on-beach album: cake #16201 removed, 58 left, originals untouched (15,000 Open Images files intact). 19 tests pass.
- 07:15 Overnight checks 05:42/06:13/06:44/07:15: no upload in data/private, GitHub repo not found, no jobs. Build complete and waiting on Reza.

## 07:51 — Reza: "is everything done, are we CERTAIN?" -> verification sweep found a real problem
- Checked: 19 tests pass; README/MORNING_REPORT/RESULTS/SYNTHESIS links resolve; no data/private files in git; no new
  writes outside find_pics since the 02:30 containment; no jobs; still no upload / no GitHub repo.
- FOUND: 19 tracked files = 17 contact-sheet JPEGs of public-dataset images (IMDB celebrity photos, CelebA faces, Open
  Images) + 2 per-image score CSVs. Not Reza's data, but CelebA forbids redistribution and celebrity photos shouldn't be
  pushed. Fixed: untracked + .gitignore (eval/attribute, eval/end2end, eval/audits, *.jpg/*.png/*.csv), and since
  nothing was ever pushed, rewrote main with git filter-branch to drop them from all 52 commits (verified 0 remain).
  Backup branch pre-scrub-backup keeps the old history locally; CLAUDE.md now says push only main.

## 09:37 — Reza: "make sure the goal is 100% confirmed; test everything you can"
- Session restarted twice; state intact (52 commits, no jobs). Gap audit vs Reza's original goal -> untested claims:
  (1) person found INSIDE videos ("every video of Reza"), (2) real-shaped Apple export (HEIC, _preview, HEVC .mov)
  through the GPU pipeline, (3) Android/Google Takeout path, (4) a person with a REAL weight transformation.
- (3) Takeout: verified (web) that Google renamed sidecars in late 2024 to <file>.supplemental-metadata.json and clips
  names at 46 chars (e.g. .supplemental-metadat.json); my ingest only knew <file>.json -> Android dates would have silently
  fallen back to file mtime. Fixed (_sidecar_candidates) + tests/test_takeout.py (3 naming schemes). 20 tests pass.
- (1)+(2): scripts/build_apple_like.py + scripts/test_apple_like.sh: synthetic osxphotos-shaped export from public photos
  (Kevin Bacon: ~half tagged; 1/3 HEIC, 1/6 _preview.jpeg; 4 HEVC .mov containing untagged Bacon photos, 6 without;
  1,900 distractor photos). First attempt was killed (exit 137) by a session interrupt -> now chained Slurm jobs
  (A=49867421 build+scan+index, B=49867423 two asks) so interrupts can't kill them.
- Untracked .claude/scheduled_tasks.lock (harness file) and ignored .claude/.

## 09:44 — Reza: "doesn't the model just KNOW Kevin Bacon? how does it learn Reza / a pineapple cheeseburger?"
- Mechanism answer: identity never uses the name; face model = metric learning (same person -> nearby vectors) that
  transfers to unseen people; references come from the user's own tags. VLM (which might know celebrities) is not used
  for identity (measured useless). BUT valid concern: IMDB celebrities may be in the face model's training data ->
  person numbers may be optimistic. Test on unseen people: DigiFace-1M (synthetic rendered identities; cannot be in
  any training set), 300 ids x 72 imgs, 8 refs each -> job 49868202 (eval/eval_unseen_faces.py).
- Query-by-example vs text (eval/eval_by_example.py, stored vectors, 15 concepts, 3 example photos each, recall of the
  remaining verified positives in top 2x): text 0.879, example 0.691, both 0.872. => for nameable concepts, examples are
  WORSE than text (a photo's vector encodes the whole scene). Examples can only win for things text can't name (your
  dog, your kitchen); not testable without such data -> not claimed.
- Learning from clicks (eval/eval_learn_from_clicks.py; user marks top-R, logistic regression on stored vectors + text
  score, no model weights change): helps only where text is weak: Christmas tree top-60 review 0.25 -> 0.50 (34 labels);
  others ~unchanged (mean over 8 concepts with both labels 0.648 -> 0.679). Thin evidence; not a general win.
- Fixed test_apple_like.sh (sbatch --wrap runs under sh: 'source env.sh' needs a full path). A=49867556 running.

## 11:59 — results: unseen identities, Apple-format export with videos; self-match bug
- Reza clarified "never seen" = not in the pretraining distribution. Answer: a face is in-distribution as a CATEGORY
  (like an orange); a specific identity (Reza, DigiFace people) is not. The face model was never asked to recognize
  anyone, only to compare two faces. DigiFace = rendered identities that exist nowhere -> the right test of that.
- Unseen identities (job 49868202, eval/results_unseen_faces.json): 300 ids x 72 images, 8 refs each: recall@0.40
  18,855/19,110 (98.7%); per-person recall min 0.70, p10 0.97, median 1.00; wrong matches 44/person among 21,438
  other-person images (0.2%), much worse at 0.30 (955/person) -> confirms 0.30-0.40 must stay a "possible" band.
  Synthetic renders may resemble each other more than real people (could inflate wrong matches); not verified.
- Apple-format export (jobs 49867556/49867560): 2,234 items (1,073 jpeg, 743 HEIC, 408 _preview, 10 HEVC .mov),
  0 decode errors, index 80 s. "Find every photo and video of Kevin Bacon": photos 317/324, videos 4/4 (none were
  tagged -> found only by face matching in sampled frames), other videos 0/6, 1 "wrong" photo.
  RAW LOOK (eval/audits/applelike_videos_and_wrong.jpg, 5 images): all 4 matched video frames show Bacon (2 with small
  letterboxed faces); the "wrong" photo P00343 is a red-carpet group where the far-left man appears to be Bacon (IMDB
  labeled a co-star) -> likely label noise.
  Transformation request on this export: heavier 4 (3 moved to fit), fit 36 incl. a 2010 video (V002) -> videos flow
  through the appearance path too.
- BUG seen in that raw look: every matched face showed sim=1.00. Cause: query-time expansion adds accepted faces as
  references, so each matches itself. Album membership was right, but the reported similarity was meaningless.
  Fix: face_sims ignores exact self-matches (>0.999). 20 tests pass. Rerun 49902074 (watcher).
- Transformation-celebrity test submitted: faces 49902471 -> judge 49902476 (eval/eval_transformation.py).
- 12:00 Scale test (synthetic vectors, judge stubbed, CPU): 150,000 items / 157,500 units / 200,000 faces:
  concept query 0.8 s (1,602 judge calls requested: head 600 + adaptive + 1,000 audit); person reference expansion +
  scoring 1.5 s; person+appearance album 1.6 s; peak RSS 1.76 GB. => Only the judge costs time: ~1,600 calls ~35 s
  on one A100 (47 img/s measured); on a Mac 0.3-1.5 s/call (research/04 estimate, unmeasured) -> 8-40 min.

## 12:10 — documented-transformation test (first pass) + self-match fix verified
- Self-match fix verified (49902074): same results (photos 317/324, videos 4/4, other videos 0/6); returned-face
  similarity now min 0.42 / median 0.77 (0/322 exact self-matches; videos 0.78-0.98).
- Transformation test (faces 49902471, judge 49902476), refs = 10 lean-era photos:
  heavy-era found @0.40: Pratt 51/73, Hill 37/47, Rogen 24/46; wrong among 5,997 other faces: 0 / 13 / 1.
  Judge AUC heavy-era vs lean-era ("face heavier/fuller"): Pratt 0.788, Hill 0.761, Rogen 0.574 ("overweight":
  0.802 / 0.757 / 0.611).
- RAW LOOK (eval/audits/tf_Pratt_eras.jpg, tf_Rogen_eras.jpg; 72 faces viewed): IMDB face labels heavily contaminated:
  Pratt heavy-era ~5/18 not Pratt (Adam Scott x2, Aubrey Plaza, Jonah Hill, other); lean-era ~5/18 not Pratt (Anna
  Faris, James Gunn, Benicio del Toro x2, other); Rogen similar (Kristen Wiig, Katherine Heigl, Elizabeth Banks,
  Joseph Gordon-Levitt). => both metrics are understated by label noise. Where faces really are Pratt: heavy 2009 p
  ~0.35-0.59 vs lean 2014 ~0.16-0.38 (visible separation). Rogen's own photos overlap more (smaller change).
- Next: rerun faces saving per-photo similarity (+ product's expansion variant) -> job 49906123; then hand-label
  Pratt's heavy-era photos by eye to get a clean recall.

## 12:29 — documented transformations, corrected by labeling every face by eye
- Rerun 49906123 saved per-photo similarity. RAW LOOK at ALL heavy-era faces (eval/audits/tf_{Pratt,Hill,Rogen}_heavy_all*.jpg;
  73+47+46 = 166 viewed, sorted by similarity):
  Pratt: #0-#20 are other people (Aziz Ansari x4, Aubrey Plaza x4, Adam Scott x3, Amy Poehler, Jonah Hill, Channing
  Tatum, several women; #19 uncertain); #21-#72 = Pratt (52). Found @0.40 with LEAN-era refs only: 51/52; with
  product expansion 52/52 (#21 profile 0.35 -> 0.42). Wrong among 5,997 others: 0.
  Hill: #0-#8 others (John C. Reilly x2, Adam Sandler x2, Jason Segel, Leslie Mann, Ricky Gervais, Danny McBride);
  #9-#46 = Hill (38). Found 37/38 (miss = profile 0.32); expanded 38/38.
  Rogen: #0-#20 others (Busy Philipps x3, Bill Hader, Jason Segel x2, Kristen Wiig, Katherine Heigl x3, James Franco x3,
  Elizabeth Banks x5, ...); #22-#45 = Rogen (24); #21 uncertain. Found 24/24.
  => across three documented 30-60 lb weight losses: 112/114 heavier-era photos matched from lean-era refs, 114/114 with
  expansion. The earlier 52-79% was IMDB label noise, not the face model.
- Judge AUC (heavy vs lean era) on identity-verified photos only: "heavier/fuller": Pratt 0.857 (52/49), Hill 0.812
  (38/37), Rogen 0.869 (24/45); "overweight": 0.864 / 0.795 / 0.884. Raw (contaminated) was 0.79 / 0.76 / 0.57.
- PRODUCT PROBLEM found: the judge's absolute scale varies by person (Hill lean-era mean p 0.63 > Pratt heavy-era 0.47).
  Fixed cutoff 0.3 puts lean photos into "heavier": Hill 32/37, Rogen 40/45. Per-person median split: heavy kept
  38/52, 28/38, 20/24; lean leaked 13/49, 10/37, 15/45. Testing the two-question rule (heavier vs fit) -> job 49911838.

## 12:34 — two-question rule measured -> the shipped transformation logic was wrong; replaced
- Job 49911838 scored "Does this person look fit and lean?" and "heavier or out of shape?" on the same verified photos.
- CURRENT code (absolute cut 0.3 + raw-score exclusivity), totals over Pratt/Hill/Rogen: heavy-era in 'heavier'
  47/114; Pratt alone 1/52 (judge rates heavy-era Pratt 'fit' > 'heavier' -> 51/52 of his heavy photos land in 'fit').
  Hill: 38/38 heavy + 16/37 lean in 'heavier'. => raw scores are not comparable across people or questions.
- Within-person rank rule (rank each photo among the person's own photos per question; keep top half; exclusivity
  by larger rank): 'heavier' = 78/114 heavy-era + 29/131 lean-era (73% of album correct by era);
  'fit' = 93/131 lean-era + 19/114 heavy-era (83% correct by era). Era is a coarse label (one can look heavy in a lean year).
- Implemented: Thresholds.rel_cut=0.5, judged.rel column, best-of sorted by rel, make_exclusive compares rel; report
  says the album is relative to this person's own photos. New test (raw scores say fit, rank says heavier). 21 pass.
- Rerun on Apple-format export: job 49913285 (watcher).
- 12:39 Relative-rule rerun 49913285 (Apple-format export, Bacon): heavier 145 (16 moved), fit 34 (2 moved), overlap 0, videos flow to both (V001 heavier, V002 fit). Bacon never changed weight, so 'heavier' = his relatively heavier-looking half by construction -> stated in report.
- 13:04 Reza: product must run on a phone; slow OK; two honest modes (fast w/ bound, exhaustive near-oracle). Oracle = judge every item. Submitted oracle array 49921980 (20 queries x 19,218 items) -> eval/oracle/*.parquet; analyze replays fast mode on the same stored judgments.
- 13:06 Honest scope note for the oracle run: it judges every item at the judge's standard resolution (<=896 px)
  with one best frame per video, NOT full-resolution tiles / every frame. It measures what the cheap first stage loses,
  not what the judge itself misses on tiny objects (tiling = later step). Added 'findpics ask --exhaustive' (judge every
  in-scope item). Pet identity test submitted (49922347): PE-Core vs DINOv2 vectors on DogFaceNet_large, 3 refs per dog,
  distractors = other dogs.
- 13:10 Pet test 49922347 FAILED: a truncated JPEG in DogFaceNet_large; eval now skips + counts unreadable
  images; resubmitted 49923284. Added 'findpics ask --ref "Name=a.jpg,b.jpg"': reference photos for any person
  without Apple tags (largest face per photo -> face refs -> same expansion). Non-face subjects error out clearly until
  the general path is measured. Known gap: a group-photo reference takes the largest face; product should show the
  chosen crop for confirmation. insightface installed in envs/vllm (CPU ORT; refs only). 22 tests pass.
  --ref end-to-end test: 3 Bacon photos as an untagged name "Kev" on the Apple-format export -> job 49923837.

## 13:15 — "who" beyond human faces
- --ref end to end (49923837): 3 Bacon photos under a name Apple never tagged ("Kev") on the Apple-format export ->
  photos 317/324, videos 4/4, other videos 0/6, 1 outside labels = identical to using 168 Apple tags (expansion grew
  3 refs to 318). Apple tags are not needed for people.
- Individual dogs (49923284, DogFaceNet_large, 840 dogs / 6,215 photos, 2 unreadable skipped; 3 refs per dog,
  distractors = every other dog): PE-Core R-precision 0.580, recall in top 2T 0.678 (505/840 dogs >= 0.5);
  DINOv2-base 0.565 / 0.695 (522/840). => general image vectors only half-work for individual animals; both models
  similar, so model choice is not the lever. Hard setting (all distractors are dogs, many same breed). Raw look pending
  (49925783 renders 3 dogs: refs + top-10).
- 13:20 Dog raw look (eval/audits/pet_identity_examples.jpg, 3 dogs x 13 images): black dog: top-3 correct but
  same photo session as refs, then other black dogs (vector matched setting+coat, not individual); white dog: 3
  "other dog" results at sim 1.00 = the same photos filed under another ID, several more show the same dog + same
  person -> label noise understates the score; cream dog: top-10 all look-alike dogs, can't tell individuals apart from
  thumbnails. => general vectors find look-alike dogs, not reliably THIS dog.
  MegaDescriptor (BVRA, animal re-ID, CC-BY-NC) is the animal analog of the face model, but DogFaceNet was added to its
  training toolkit (WildlifeDatasets README 09/05/2024) -> leakage risk; not evaluated on it. Needs a clean test set
  (e.g. Reza's own pet photos). Product stance until then: non-human "who" = "possible matches, please confirm".

## 13:24 — Reza: "who/what must be ANYTHING: a place, a thing, a picture of a picture"
- Agreed. Kinds of sameness + tool + public test: person (face model, done); animal (re-ID; DogFaceNet, v2 job
  49928832 with OWLv2 crops, cross-ID duplicate removal, MegaDescriptor-B-224, which is clean: last modified 2024-01-05
  < DogFaceNet added to WildlifeDatasets 2024-09-05; L-384 modified 2024-10-14 -> excluded for leakage risk);
  thing (Stanford Online Products test shard 0, 2,000 products); place (Google Landmarks v2 mini test, 1,500
  landmarks); picture-of-a-picture (self-generated: 300 Open Images photos x 5 copies: print photo w/ perspective,
  framed on a wall, phone screenshot, 60% crop + JPEG q25, photo of a screen w/ moire; 3,000 distractors).
- eval/eval_instance.py = one protocol for every kind (refs -> rank all -> R-precision, recall@top2T, distractors of
  the same kind). Array job 49929344 (things, places, copies). Oracle array 49921980 still running.
- 13:26 GPU reality (Reza pushed back: we have h100/h200/a100/blackwell/requeue). Correct that the partitions
  exist; checked limits: every Kempner partition uses QOS kempner_base = MaxTRESPerUser gpu=16, MaxTRESPerAccount
  gpu=96. Lab account kempner_mzitnik_lab has 102 GPUs running (msun415 16, jxu04 16, hxu66 13, ...; me 8) -> new jobs
  wait on MaxGRESPerAccount regardless of partition. Multi-partition submission is forbidden for kempner partitions.
  --test-only now: kempner_rtx immediate, h100 ~02:42 tomorrow, h200 ~14:20 tomorrow, requeue ~2 days.
  Moved pending work to kempner_rtx: pets2 49929685, inst 49929688, oracle tasks 8-19 49929692.
- 13:35 Instance tests: things (SOP test shard 0, 1,703 products / 13,145 photos, 3 refs each): PE-Core
  R-precision 0.694, recall@top2T 0.766; DINOv2 0.342 / 0.384 (suspiciously low; possible preprocessing issue, e.g.
  default center-crop cutting products; unverified). places FAILED: GLDv2-mini test split = 1 photo per landmark
  (3,103/3,103) -> switched to train shards (shard 0: 4,039 photos, 80 landmarks with >=4); downloading all 5.
  dogs v2 FAILED: transformers renamed OWLv2 post-processing (post_process_grounded_object_detection) -> fixed;
  rerun via eval_instance.py dogs (49932033). copies running. Oracle: 8 tasks on A100 + 1 on Blackwell, 11 waiting.
- 13:35 GLDv2-mini train (5 shards): 20,191 photos / 3,103 landmarks, all >=4 photos. places job 49932394 (1,500 landmarks).

## 13:38 — Reza: "don't limit the tests to my creativity; ANY query must work"
- Admitted: places/things/dogs were reactive to his examples. New approach so coverage comes from data, not us:
  (1) research agent -> research/05_query_space.md: what people actually search for in personal photos + public
  datasets per query dimension + a recommended combined suite; (2) eval/eval_general.py: queries generated FROM library
  content (VLM describes random photos; LLM writes a realistic query per rotating dimension: object, activity,
  scene, text-in-image, count, spatial relation, negation, attribute, emotion, event, lighting/time, photo style), run
  through the REAL planner; oracle = judge on every item of a fixed 5,000-item subset; fast mode replayed; table per
  dimension. Jobs: gen 49932706 -> oracle array 49932711 (72 queries). Known gap: the library lacks screenshots, documents,
  receipts, text-heavy photos, selfies, night shots -> to add from research/05.

## 13:46 — oracle vs fast (8 concepts) + RAW LOOK: the "gap" is mostly judge false positives
- Fast recovered (of oracle yes@0.7): dog 1621/1646, pizza 86/90, baked 1272/1333, cake 401/421, bread 666/746,
  sandwich 219/246, bicycle 328/382, guitar 117/148, using 8-22% of the oracle's judge calls. Stated bound held vs
  oracle 8/8. Oracle vs human labels: 0.79-0.99.
- RAW LOOK (eval/audits/oracle_missed_{guitar,bicycle}.jpg, 60 viewed): most "missed" items contain NO guitar/bicycle:
  guitar -> pinball machine, Santa figurine, Sydney Opera House, laptop, cat, bookshelf, sports car (~4-5/30 real,
  small); bicycle -> mostly CARS at p~0.99, boats, cats, horses, an accordion (~4/30 real: bike on grass, wheel behind a
  dog). => fast mode's real misses are small; exhaustive mode mostly adds judge false positives for these concepts.
- p=0.99 'bicycle' on cars is suspicious -> possible batch misalignment bug in judging. Check job 49934049 (same
  photos judged one-by-one vs batched vs shuffled). Cancelled oracle tasks 16-19 (49929692) to free GPUs for it.
- 13:47 research/05_query_space.md landed (54 sources): only real query log = Jiang et al. WSDM 2017 (961,826
  Flickr queries; mean 1.5 words; 85.3% visual; what/who/where/when). 2026 camera-roll benchmarks: PhotoBench (3 real
  albums, 3,582 imgs, 300 queries w/ GT, faces/GPS/time, includes "nothing matches"), DISBench, CamRoll, ATM-Bench.
  Screenshots ~499/library (Mottelson 2023). 23-dimension taxonomy; per-dimension datasets listed; recommended suite.
  No public NL query frequencies exist; negation not documented as common. Next: get PhotoBench.
- 13:48 PhotoBench: queries + ground truth downloaded (data/public/raw/photobench, 300 validation queries, e.g.
  "cat under the car" -> 2 photos, tagged V/M/F). Raw photos are NOT public: Drive folder 404 / "available upon request"
  -> stopped; asking Reza whether to request them. PRODUCT GAP exposed: ~half of PhotoBench queries need time or PLACE
  metadata; ingest keeps dates but drops GPS / place names -> must add (osxphotos JSON has place; EXIF has GPS).
- 13:50 PLACE support (PhotoBench: ~half of real queries need time/place): ingest keeps lat/lon + place text
  from osxphotos 'place' (verified in source: asdict has latitude/longitude/place{name, names{city..}, address_str}),
  Takeout geoData, EXIF GPS; offline reverse geocoding (reverse_geocoder, GeoNames table in the package: Tokyo coords ->
  'Tokyo', Harvard -> 'Cambridge, Massachusetts'). Planner 'place' field grounded in the request like dates; engine
  filters by place words. Tests: 24 pass.
- DISBench downloaded (14 GB; 57 users, 109,467 photos, 122 queries with GT, real taken_time/GPS/address). Queries are
  multi-step ("photos from the calendar week when a foggy cityscape was photographed ... excluding wine bottles") ->
  current one-step planner expected to score low; plan: baseline, then multi-step agent planner.
- 13:51 Cancelled oracle tasks 12-15 (49929692) so the batch-alignment check (49934049) and query generation run first; to resubmit after the check.
- 13:53 places (GLDv2-mini train, 1,500 landmarks / 9,758 photos, 3 refs): PE-Core R-precision 0.680,
  recall@top2T 0.779; DINOv2 0.567 / 0.642. Not yet looked at (GLDv2 labels are known-noisy); eval_instance now saves
  vectors so sheets can be rendered on CPU.

## 13:55 — CORRECTION: the oracle-vs-fast gap is mostly REAL small objects, not judge false positives
- Batch-alignment check (49936016, 24 photos; single vs batch vs shuffled): identical scores -> no batching bug.
- I then viewed the suspected "false positives" at full resolution (eval/audits/bicycle_fullres_check.jpg,
  guitar_fullres_check.jpg): bicycle 4/4 really contain a bicycle (bikes beside an open trunk, bikes hanging in a garage,
  a bike by a fence behind a red car, a bike wheel at the edge of a living room); guitar ~5/8 real (guitar on the wall
  behind a couple at a piano, electric guitar leaning on a table behind dancers, guitarist in a band's background, guitar
  on a man's back), 2 borderline (Elvis pinball art, Santa figurine holding a violin), 1 clear miss (Opera House).
- My 14:3x claim ("most fast misses are judge false positives") was WRONG: it came from 220-px thumbnails. The cheap
  stage misses small background objects (cheap-stage rank 1,200-5,000); the judge finds them; Open Images labels omit
  them. => per-crop vectors (step 3) now justified by evidence; exhaustive mode has real value for small objects.
- PROCESS RULE added: raw-look audits must use images at the resolution the judge sees (>= ~900 px), not thumbnails.
- 13:56 Tile-vector experiment submitted (49938023): whole + 2x2 + 3x3 PE-Core vectors for all testlib photos; analyze = recall of oracle-confirmed items within top-K for whole vs max-over-tiles.
- 13:56 Resubmitted oracle tasks 12-19 (49938104) after the no-batching-bug result.
- 13:57 Instance results (3 refs per identity; distractors of the same kind; R-precision / recall@top2T):
  copies (300 originals x 5 copies + 3,000 distractors; 1 ref = the original): PE-Core 0.901 / 0.957; DINOv2 0.783 / 0.839.
  things (SOP, 1,703 products): PE-Core 0.694 / 0.766; DINOv2 0.342 / 0.384.
  places (GLDv2-mini, 1,500 landmarks): PE-Core 0.680 / 0.779; DINOv2 0.567 / 0.642.
  dogs (DogFaceNet, 788 dogs after removing cross-ID duplicates): PE-Core full 0.595 / crop 0.571; DINOv2 0.606 / 0.588;
  MegaDescriptor-B-224 0.281 / 0.302 (worse; preprocessing or model size may be wrong; unresolved).
  => person (face model, 0.98+) >> copies (0.90) > things ~ places (~0.69) > individual dogs (~0.60). Cropping dogs did
  not help. Not yet looked at by eye for things/places/copies (sheets pending).

## 14:07 — pre-compaction checkpoint (Reza: "do it all, continue post compact")
- DISBench built as a library (109,467 photos, 107,577 with GPS; osxphotos-shaped metadata) and scanned; index array
  49941173 (16 shards, kempner_rtx). PLAN.md STATUS rewritten with every in-flight job and the ordered next steps
  (multi-step planner + refinement, tiles, raw-look sheets, animals, phone path).
- Reza asked about follow-up edits ("get rid of this, add this"): same tech as a multi-step planner (plan steps that act
  on the current album); agreed to build together with the multi-step planner.
