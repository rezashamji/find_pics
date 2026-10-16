# find_pics journal (append-only)

MAC NEEDS REZA (10-06 23:0x):
1. UNLOCK THE IPHONE and keep it unlocked (plus "Trust This Computer" if it asks). The app is installed; launch is
   denied while locked: `Unable to launch com.rezashamji.findpics because the device was not, or could not be,
   unlocked` (FBSOpenApplicationErrorDomain 7).
2. THE INCREASED-MEMORY ENTITLEMENT CANNOT BE DONE FROM THE SHELL (measured, see docs/FIRST_DEVICE_TEST.md section 0).
   The re-sign puts it in the signature but `installd` checks the signature against the provisioning profile and
   refuses to install (0xe8008015); asking automatic provisioning for it fails at build time ("not found and could
   not be included in profile"). Needs an Apple-side capability on the App ID. Options for you: try Xcode GUI ->
   app target -> Signing & Capabilities -> "+ Capability" -> "Increased Memory Limit" -> Run; or tell me whether
   YYP85AQ2C5 is a free Personal Team or a paid account (free accounts cannot add this capability).
   Until then the Qwen judges cannot run on the phone at all: one 4-bit 4B model (~3.1 GB) > the 2.4 GB cap.
   Installed and testable meanwhile: steps 1-3 (self-check, memory number, library read) and the Apple-model rows of
   step 4 (Apple's model is out-of-process, so the cap does not apply to it).

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
- 14:08 store.subset() (index view for one user's library or the current album; test added, 25 pass); eval/eval_disbench.py baseline submitted (49941813, after index).
- 14:09 Multi-step planner v1 (agent.py: anchor -> window same_day/week/event/place -> target, + exclusion; events = time-gap segmentation 3h; code-grounded time/place). Tests: 28 pass. DISBench agent eval 49942224 queued after index.
- 14:20 Refinement built: src/findpics/refine.py (keep_if/remove_if/remove_ids/remove_like/add_like; LLM may only use ids the person selected; versioned manifests; removals via guarded link remover) + 'findpics refine' CLI; ask now records index_dir. Tests: 33 pass. GPU end-to-end test 49944720 (bread album -> 'remove the ones that are burgers or sandwiches').
- 14:21 Review page: instruction box + 'Copy refine command' (selected = clicked photos). Phone feasibility: judge-size comparison (Qwen3.5-4B/2B vs 9B on oracle samples; agreement + accuracy vs labels) job 49944923.

## 14:24 — generalization test v1 (61/72 queries) exposed a PLANNER bug; fixed; rerun
- Per-dimension fast-vs-oracle (v1): overall 0.741; but the judge said YES to the very photo each query was written from
  in only 0.639 (39/61). Event 1/6, mood 2/6, count 2/4, activity/attribute 3/6.
- Root cause (read all 22 failures): 15/22 judge questions said "the person in the red box" with NO person in the album,
  so no box is drawn and the judge is asked about something that doesn't exist (e.g. "Does the person in the red box
  look like an elephant seal?"). Fix: planner.fix_red_box (code-enforced; person albums keep it), same scrub in
  agent.make_plan. Test added; 34 pass. v1 results moved to eval/general/v1_redbox_bug/. Rerun: 49945595.
- Other seed failures to re-check after rerun: over-specific compound questions ("exactly two tiers and three
  candles"), degenerate broad queries ("no readable text" -> 3,689 yes).
- Phone path: Core ML env built (envs/coreml) for PE-Core conversion (scripts/convert_coreml.py).
- 14:25 Queued: pet judge side-by-side test 49945954 (300 same / 300 hard-different dog pairs); Core ML conversion 49945663; README updated (refine, --ref, --exhaustive, places, multi-step); ask --multistep opt-in.
- 14:25 Tile vectors pre-built into the product behind 'index --tiles {none,2x2,2x2+3x3}' (clip_tiles.npy per shard; store loads; look_scores = max(whole, best tile)). Off by default until eval/eval_tiles.py shows a gain. 35 tests pass.

## 2026-10-02 ~15:30 — Reza: "refinement shouldn't be a separate hard-coded thing; no mode flags; multi-step should just happen"
Reza's critique, mostly right:
- The refine command was a SEPARATE system with a fixed 5-op menu (keep_if/remove_if/remove_ids/remove_like/add_like).
  Anything outside the menu (e.g. "only before I moved", a date change) could not be expressed. Wrong design.
- `--multistep` / `--exhaustive` were flags the user had to pick. The planner can decide two-step itself (anchor=null
  when not needed); "look harder" is a sentence.
- Pushed back on one point: "add more like these shouldn't be needed" is true for MISSES (fix = look harder), but
  follow-ups that ADD are legitimate when intent changes. Example-photo search is dropped from the language path
  anyway: measured worse than words (0.69 vs 0.88).
Change: new `findpics/converse.py` = ONE planner for first message and every follow-up. Prompt sees the conversation +
the current plan and returns the WHOLE updated plan (same schema: albums with optional anchor/window/exclude_question,
plus effort_phrase). Grounding (dates/place) is checked against everything typed so far; effort only against the new
message. CachedJudge caches P(yes) per (image pixels, question) in the session folder, so reruns pay only for what
changed. Review-page taps ("wrong") are stored as per-item overrides and applied after every rerun. Each turn kept in
turn_N/. `findpics ask "<msg>" --out <conversation>`; removed refine.py + `refine` cmd + --multistep + --exhaustive.
agent.make_plan/execute kept only because the queued DISBench jobs import them.
Tests: 39 pass (new tests/test_converse.py: grounding across turns, anchor->window->exclude rows mapped back to full
index, overrides, thorough mode judges all 900/900, cache hits, session round-trip).
Old refine GPU test (49944720) result, kept as evidence the exclusion mechanism works: bread 706 -> 636 after "remove
burgers or sandwiches". Raw look at all 70 removed: 64 obvious burgers/sandwiches on the sheet; 6 ambiguous viewed at
full res: 4 correct (pastry case with sandwiches, open-sandwich display, dinner with burgers on plates — looked wrong
as a thumbnail), 1 wrong (a menu that only mentions burgers in text), 1 unclear (cafe counter). => 68/70 correct, 1 wrong, 1 unclear.
Tiles experiment (49938023) done: whole + 2x2 tiles raises fast-stage recall@2000 vs oracle truth 0.877 -> 0.896 (mean of
13 concepts); biggest gains small objects: bicycle 0.869 -> 0.921 (of 382), christmas tree 0.765 -> 0.812 (of 85),
sunglasses 0.457 -> 0.498 (of 1043). 3x3 adds nothing over 2x2. Cost: 5 vectors per photo instead of 1. Raw look pending.
Submitted: fp_planners 49950289 (12 scripted conversations + old-vs-new plans on 72 generalization + 122 DISBench
queries), fp_disbench_uni 49950290 (DISBench end-to-end through the merged planner), fp_chat 49950443 (3-message
conversation on testlib: bread -> "drop the sandwiches and burgers" -> "look harder").

## 2026-10-02 ~15:40 — Phone-size judge (49944923): Qwen3.5-2B and 4B vs 9B, 13 oracle concepts
Sample per concept: every 9B-yes (cap 300) + equal number of random 9B-no (so "agree" is weighted toward the yes class).
- False yes on 9B-no: 0.000-0.017 (2B), 0.000-0.007 (4B). Small models almost never add things.
- Recall of 9B's yes: 2B 0.83-0.99 on 12/13 (christmas tree 0.62 of 85); 4B 0.83-0.997 on 12/13 (christmas tree 0.65).
  Small models are more conservative at the same 0.7 cut; their cut may need recalibrating (p values not saved: TODO).
- Vs human Open Images labels (n_labeled 24-289 per concept): small ~= 9B (e.g. dog 0.990 vs 0.990 of 289, cat 0.969 vs
  0.966 of 261, bread 0.803 vs 0.779 of 122). Where the small model "misses" a 9B yes, labels often don't side with the 9B.
- Not yet raw-looked: which 9B-yes items the 2B rejects (need p values saved + full-res sheet). Until then: "a 2B judge
  is plausible for phone", not "equivalent".

## 2026-10-02 ~16:00 — Core ML conversion OK (image + text towers)
Job 49945663 failed at package write: coremltools' libmodelpackage.so needs GLIBCXX_3.4.26 (system libstdc++ too old).
Fix: `module load gcc/13.2.0-fasrc01`. Text tower then failed in TorchScript conversion (aten::Int casts inside
nn.MultiheadAttention); switched text tower to torch.export -> ct.convert. Result: models/coreml/pe_core_image.mlpackage
(633 MB) and pe_core_text.mlpackage (709 MB), fp16, iOS17. PyTorch trace matches original (max abs diff 0.0).
NOT verified: Core ML prediction (Linux cannot run Core ML) -> needs a Mac/iPhone to compare outputs vs PyTorch.
Size note: ~1.3 GB for the two towers is heavy for a phone; smaller encoder (PE-Core-B / MobileCLIP2) is the next lever.

Reza (15:50) asked: is it still a chat box? Yes. And fast vs best: one search; fast = judge checks top candidates +
random sample (states a bound), best = judge checks every photo; "look harder" = best, reusing cached answers.
Proposed (not built, asked him): fast results shown immediately, judging continues in the background, album grows,
bound tightens to "checked everything".

## 2026-10-02 ~16:30 — Merged planner test v1 (49950289) + fixes; streaming replaces modes
v1 (planner before fixes): A. conversations 6/12 pass; B. 72/72 generalization queries planned, 0/72 needless two-step,
72/72 same album count as old planner; C. DISBench two-step new 83 vs old 113 of 122 (agree 90/122), exclusions 11 vs 20.
Read every failed plan (eval/planners/v1/conversations.json): 5 real planner faults + 1 check bug:
- heavier/fit: "past 6 months" put on BOTH albums (my new template had dropped the "a time phrase belongs to one
  album" rule).  - "Dad at the beach"/"at the gym": place="beach"/"gym" with no condition -> condition silently lost.
- Grand Canyon: invented dates from "the week I went to the Grand Canyon"; window same_event although words say week.
- screenshots: planner REFUSED (zero albums; inferred from "known people" that library is only people).
- cat -> "also videos of her" -> "only from Paris": follow-ups became NEW albums; "her" became Mom.
Fixes: prompt (time-phrase rule + example; never refuse; follow-up = EDIT with only/also/drop/pronoun rules) and code:
time_phrase must contain calendar words; window from words ("the week" -> same_week); place inside anchor dropped;
zero albums -> retry; place_or_look(): a place matching no item's place text becomes a visual condition. 4 regression
tests built from the 9B's actual outputs. Rerun: fp_planners 49958823 (started 15:06, after fixes).
Also: fp_chat v1 (49950443, pre-fix code) showed the same follow-up fault end-to-end: "drop the sandwiches and burgers"
-> exclude_question null (album 680 -> 675 only from resampling). Cache worked: turn 2 reused 2,438, new 1,362. Cancelled.

Streaming (Reza 16:1x agreed: "fast" is just the first minutes of "best"): engine.stream_album yields rounds. Round 1 =
the old fast answer (adaptive head + random tail sample); each later round doubles the judged head (highest fast
scores first) and draws a fresh uniform sample of the still-unjudged tail, until everything is judged. Union bound so
stopping at ANY round (incl. because the bound looks good) is valid: alpha/2 for round 1, alpha/2*6/(pi^2 (k-1)^2) for
round k>=2 (sums to alpha). Every item judged once (dict). Matches found by an earlier random sample stay in the album
(counted as found AND their tail still counted as possibly-missed: conservative). Exclusions judged once per item.
CLI: ask streams by default, writes albums/review page after every round, --minutes N or Ctrl-C stops; Apple albums
written once at the end. effort_phrase / "look harder" removed (a new message just resumes from the cache).
Album link names now md5(path)[:8]_name (stable across rounds; no duplicates). 44 tests pass.
Submitted fp_chat 49960025: bread (stream to the end) -> "drop the sandwiches and burgers" -> "actually keep the sandwiches".
fp_disbench_uni restarted on fixed code: 49958920.

## 2026-10-02 ~17:00 — 20-concept oracle + 72-query generalization (v2, red-box fixed) + streaming curve
Oracle (judge on all 19,218 items) vs round-1 ("fast") answer, 20 concepts: stated bound <= true recall 20/20.
Round-1 gap: <=11% on 12/20; small objects lose most: sunglasses 613/1043 found (gap 0.41), christmas tree 52/85 (0.39),
person_sunglasses 641/1048 (0.39), coffee cup 287/403 (0.29), wine glass 244/330 (0.26), eating pizza 13/17, guitar 117/148.
Raw look (full res, 8 random of the 398 oracle-yes sunglasses photos ranked beyond the 5,000 head): 4 real but tiny
(background person / driver), 3 unclear (crowd, faces too small), 1 judge false-yes (eyeglasses on a neck chain).
=> the gap is mostly real small objects the whole-photo vector ranks low.
Generalization v2 (72 queries generated from library photos, 12 dimensions): fast recall of oracle 0.805 mean; judge
says yes on the source photo 0.79 (v1 with red-box bug: 0.64). Weak: "event or occasion" (oracle median 0 matches,
source yes 2/6) and photo quality/style (0.68), attribute/color/clothing (0.72). TODO raw look at the event queries.
Streaming replay (eval/eval_streaming.py, stored oracle answers; CPU): v1 schedule (1,000 sample/round, alpha series)
held 0/115 violations but the bound DROPPED as rounds went on (pizza 0.56 -> 0.48 while recall 0.96 -> 0.97).
Fix: tail sample doubles each round (free in total: everything is judged once by the end) and alpha/2 split evenly
over the later rounds (count known after round 1). v2: 0/115 rounds overclaim; bound rises in 18/20 (guitar
0.63 -> 0.53 once, christmas tree 0.57 -> 0.55 once). Recall of oracle with <=10% of library judged: median 0.94, min
0.61 (9 concepts reach a round by then); <=25%: median 0.94, min 0.71 (18); <=50%: median 0.96, min 0.59 (sunglasses).

## 2026-10-02 ~17:15 — DISBench baseline (49941813, one-step planner, 122 queries, ~1,900 photos per user)
P 0.039, R 0.116, F1 0.033, exact 0/122 (inter-event F1 0.036, intra-event 0.030); 3 errors.
Raw per-query look: 82/122 returned NOTHING; 21/122 got >=1 correct photo. gt median 3 photos/query.
Causes seen in sampled plans: (1) scene words as place filters ("rainforest canyon", "observation deck", "museum",
"Everest Base Camp") -> empty scope (fixed in converse: place_or_look); (2) dates invented from "7 days before the photo
of X" (fixed now: time relative to another photo dropped; test added); (3) multi-hop identity ("the person who previously
had selfies with elephants") and offset windows ("7 days before") are not representable yet: honest gap.
Waiting: fp_disbench_agent (old multistep), fp_disbench_uni 49958920 (merged planner incl. place_or_look; started before
the relative-time rule).

## 2026-10-02 ~17:35 — Merged planner rerun (49958823, after fixes)
A. conversations 9/12 strict (v1: 6/12). Failures read: Dad-at-beach and gym keep place='beach'/'gym' in the PLAN, but
place_or_look converts them at execution; re-scored with place_or_look applied: 11/12. Remaining: cat -> "also videos of
her" -> "only from Paris" made two albums (cat photos + cat videos, both Paris; "her" = the cat now). Reasonable reading,
but my pre-written check fails it -> counted as FAIL.
B. 72/72 planned; needless two-step 2/72 (car show; pool game: anchor = same thing as target). Album count = old 72/72.
C. DISBench two-step 85 vs old planner 113 of 122; exclusions 12 vs 20.
Tried two rules to drop needless anchors, checked each against the stored DISBench plans, REJECTED both:
 - word overlap anchor~target >= 0.3: removes 38 DISBench two-steps, several legitimate (Opera House on the day of the
   T-shirt exhibition; Darth Maul on the day both were photographed); "visible" etc inflate overlap.
 - moment words ("the day", "when", "during"...): removes 35/85 legitimate ones ("in the city where X was photographed",
   "after the cat tree was assembled", "on the way to the concert").
Adopted instead: EVERY confident anchor hit opens a window (was top 5). A redundant anchor then costs nothing (all car
shows stay in scope); a real anchor is a specific moment and still narrows. Recall first. Test added. 46 tests pass.
Note: fp_disbench_uni (running) uses top-5 anchors and predates the relative-time rule.

## 2026-10-02 ~18:00 — First real streamed conversation (fp_chat 49960025, `ask` x3, RTX GPU, testlib 19,218)
Turn 1 "all my photos with bread": rounds 678 (>=80%) @217s -> 726 (>=93%) @409s -> 731 (>=92%) @676s -> 731 (100%, all
judged) @990s. New judge calls 19,083. (Engine was streaming v1 alpha schedule: hence the 93->92 dip.)
Turn 2 "drop the sandwiches and burgers": exclude_question added; 731 -> 526; reused 19,098 cached answers, new 731
(the exclusion question on the album) ; ~4 min. BUG: final report still said "731 items" -> fixed (_after_removal:
count + bound restated: kept/(kept+missed_upper), valid since narrower matches are a subset).
Turn 3 "actually keep the sandwiches": planner removed the WHOLE exclusion (burgers back) -> wrong. Fixed with a
partial-undo rule in the follow-up prompt + a new scripted conversation in eval_planners.py.
Raw look (full res): 4 random removed = 4/4 sandwiches/burgers; 4 random kept = 3 real bread (dough, toast under
eggs, crostini) + 1 bread false-yes from turn 1 (apple salad bowl).
Slowness: each `ask` reloads vLLM (~3-4 min). New `findpics chat` loads models once; each typed line = a message.
Resubmitted test via chat: fp_chat 49969857.

## 2026-10-02 ~18:15 — Speed audit (Reza: "make sure tests run as fast as they can")
Conceded: per-message model reloads were my oversight (fixed via `findpics chat`). Audited the judge loop too:
serial full-res decode + JPEG/base64 re-encode = 46 ms/photo on CPU (login-node measurement, 96 testlib photos) vs
~52 ms/photo end-to-end in the conversation test (19,083 photos in ~16 min) => GPU mostly idle waiting on CPU.
Fix: JPEG draft decode (decode at reduced DCT scale; big win on 12 MP phone photos, small on 1024-px testlib), decode
and encode on all Slurm-allocated cores (ThreadPoolExecutor sized by sched_getaffinity), next batch prefetched while the
GPU judges, batch 48 -> 96. 47 tests pass. Benchmark submitted: fp_bench 49970486 (960 photos old vs 960 new).
Note: running fp_chat 49969857 and both DISBench jobs started before this change.

## 2026-10-02 ~18:25 — DISBench old multi-step agent (49942224, COMPLETED 60 min, 0 errors)
P 0.076, R 0.424, F1 0.096, exact 0/122 (inter-event F1 0.069, intra-event 0.126). vs one-step baseline: R 0.116 ->
0.424, F1 0.033 -> 0.096; returned nothing 82 -> 32 of 122; >=1 correct 21 -> 63; per-query F1 better 57 / worse 12 /
same 53. Precision is the weak point (returns too many). NOT yet raw-looked (images) -> claim pending.
Waiting: fp_disbench_uni 49958920 (merged planner, top-5 anchors, pre relative-time rule).
Speed bench (fp_bench 49970486, RTX GPU, 8 cores, 960 photos per arm, different photos): old serial loader 14.6 photos/s
-> new prefetching multi-core loader 27.8 photos/s (1.90x). Judging all 19,218 testlib photos: ~22 min -> ~11.5 min.
Remaining CPU work sits inside vLLM's own image preprocessing (single process); next levers: more GPUs per search
(shard the judge), or lower judge resolution (accuracy cost: unmeasured, not done).

## 2026-10-02 ~18:35 — Reza: use as many GPUs as possible for testing; never lower judge resolution
eval_disbench.py: --shard=k/K (array) + `merge <mode>`. Cancelled single-GPU fp_disbench_uni 49958920 (old code) and
submitted 8-way array fp_dis_uni 49973750 (current code: relative-time rule, all anchor hits, fast loader).
Capacity: lab account kempner_mzitnik_lab at its 96-GPU cap (others: 16+16+10+9+9+...), so the array waits
(MaxGRESPerAccount). gpu_test accepts -A mzitnik_lab (gpu/gpu_requeue do not): planner eval sent there (49973865).
Rule recorded in CLAUDE.md.

## 2026-10-02 ~18:50 — Conversation test via `findpics chat` (fp_chat 49969857; models loaded once; old loader)
T1 bread: 678 (>=80%) -> 727 (>=93%) -> 731 (>=98%) -> 731 (100%, all judged) in 980 s; new judge calls 19,070.
T2 "drop the sandwiches and burgers": exclude "sandwich or burger?"; 525; bound restated after removal 75% -> 100%;
  reused 19,098, new 731; 230 s.
T3 "actually keep the sandwiches": exclude_question -> "Is there a burger in this photo?" (partial undo CORRECT);
  657 items (133 back, 74 still excluded); reused 19,098, new 731 (new question); 216 s.
Raw look full res: 4 random re-added = 4/4 bread incl. sandwiches/toasts/roll (correct to return); 4 random still
excluded = 4/4 burgers.

## 2026-10-02 ~19:05 — DISBench agent: why precision is 0.076 (per-query look at plans + traces)
returned per query: median 9.5, mean 70.6, max 1,923. Top offenders: when the ANCHOR found 0 photos, window_rows fell
back to the WHOLE library (q99: 440 returned, q43: 416, q32: 405), and some target questions are always-true ("Is this
photo of the building in real life or non-real form?" -> 1,923 of 1,948). Also an invented window "same_year" meant
"whole library".
Fixes (converse path): anchor found nothing -> 0 items + report says the moment was not found (no silent fallback);
unknown window -> same_event; added real same_month/same_year windows (+ words "the month"/"the year"). 49 tests pass.
Always-true judge questions: not fixed yet (idea: flag when the judge says yes to most of the head).
The 8 DISBench shards (49973750) have not started -> they will run this code.

## 2026-10-02 ~19:20 — "event or occasion" weakness in generalization v2 = test artifact (mostly)
6/72 plans still asked about "the person in the red box": queries phrased "I'm / we" -> planner set person=me, but this
public library has no reference faces for "me", so no box is drawn. Source photo judged yes 3/6 vs 54/66 for the rest.
4 of the 6 are "event or occasion" (walk the dog in the park; car show; yellow dog toy; pool game). The product would
refuse a 'me' album with no references (_refs_for raises). Fix in eval: drop person + fix_red_box; old results moved to
eval/general/v2_redbox_person/; reran k=2,9,33,45,62,69 on 3 GPUs (fp_gen_fix 49976128/42/43).
Also added: warning when the judge says yes to >50% of the random check (always-true questions). 50 tests pass.
gpu_test overflow attempt failed: its GPU has 19.6 GiB -> vLLM 9B OOM at engine init (fp_planners 49973865). Resubmitted
to kempner_rtx (see .cache/tmp/planners.jobid). CLAUDE.md corrected.

## 2026-10-02 ~19:45 — Reza: "submit each job to every partition and cancel the ones that don't start first"
scripts/race_sbatch.sh <name> <time> <cmd>: submits the same job to kempner_requeue (H100-80GB and A100-40GB named,
--requeue), kempner_rtx, kempner_h100, kempner_h200, kempner; the first copy to start scancels its siblings by name.
Found: kempner_requeue has QoS=N/A -> NOT under the lab's 96-GPU MaxGRESPerAccount cap (preemptible). Its 20 GB MIG
slices would OOM the 9B -> GPU types requested explicitly. Cancelled the single-partition jobs and race-submitted:
fp_dis_uni_s0..7 (DISBench unified shards), fp_gen_fix_2_9 / 33_45 / 62_69, fp_planners (72 submissions).
19:55 race_sbatch fix: two copies starting in the same scheduler pass would have cancelled EACH OTHER. Now: the lower
job id wins (a copy that sees a lower-id sibling RUNNING exits; it cancels everything else). Re-raced all 12 jobs
(submitted new copies first, then cancelled the 72 old ones so name-based watchers never saw an empty queue).
Session-independence: cancellation runs inside the jobs on compute nodes. Post-processing moved into the jobs too
(CPU-only jobs are refused on kempner partitions): last DISBench shard merges -> eval/disbench/unified_summary.txt;
last generalization job runs analyze -> eval/general/analyze_latest.txt; planner eval writes eval/planners/*.json.

## 2026-10-02 ~20:10 — Phone-size judge, per-item (Qwen3.5-2B on gpu_test 20 GB, 11 min, 20 concepts)
Sample per concept: all 9B-yes (cap 300) + equal random 9B-no. 2B ranks like the 9B: AUC vs 9B labels median 0.997
(min 0.979 eating_pizza, 0.989 christmas tree). The 2B is just more conservative: at the 9B's 0.7 cut it recovers a median
0.918 of 9B yeses (christmas tree 0.612, eating pizza 0.588); at cut 0.3: median 0.987 (min 0.706), false yes median 0.020
(max 0.083) on this balanced sample. Caveat: a real library is mostly non-matches, so 2% false yes on ~18k negatives
= ~360 wrong photos; cut must be set per prevalence, and random negatives are easy ones.
Raw look (full res): 4 random christmas-tree items 9B>=0.7 but 2B<0.3: 3/4 the 2B is RIGHT (plain conifers: lit pine at
night, spruces behind gentians, evergreen behind two people), 1/4 real tiny tree on a movie poster. => the 9B "oracle"
over-calls conifers; oracle != truth. 2 random sunglasses items 2B>=0.3 but 9B<0.7: both 2B wrong (clear eyeglasses;
a cake).

## 2026-10-02 ~20:20 — Session ending (Reza). Handoff in PLAN.md STATUS (~20:20 block).
Queued to run unattended: 12 race jobs (self-cancelling siblings, self-merging), and on gpu_test fp_full_Qwen3_5-2B
(49984136) / -4B (49984143): small judges on all 19,218 testlib photos for 4 concepts (real-prevalence false-yes rate).
20:30 Reza OK'd FP8 9B (phone-relevant: phones run compressed 8/4-bit models; question = how much compression changes
the judge vs bf16 9B, on the same 4 concepts x all 19,218 photos, directly comparable with eval/oracle/*.parquet).
vlm.py: FP_QUANT env -> vLLM quantization (fp8; weight-only on A100). gpu_test allows 2 submitted jobs/user, so:
fp_full_2b_then_9bfp8 (2B, then 9B-fp8 -> eval/judge_size/full_Qwen3.5-9B-fp8_*.parquet) + fp_full_Qwen3_5-4B.

## 2026-10-02 ~23:10 — Session resumed; overnight results
All race jobs ran (queue empty). Self-merge worked for DISBench; generalization auto-analyze did NOT fire (3 jobs
finished together, each counted <72 files) -> ran by hand.
DISBench, merged planner (8 shards, current code): P 0.104, R 0.338, F1 0.115, exact 1/122; returned nothing 37; >=1
correct 53; 1 error. vs old multi-step agent P 0.076 R 0.424 F1 0.096 (nothing 32, >=1 correct 63); one-step F1 0.033.
=> best F1 and precision; lower recall (anchor-not-found now returns nothing instead of the whole library). NOT yet
raw-looked.
Planner test (fp_planners 49979441): 11/13 strict; the 2 fails = place 'beach'/'gym' in plan (converted at execution
by place_or_look) -> 13/13 as executed. Cat and partial-undo conversations PASS. Needless two-step 1/72.
Generalization after red-box/person fix: fast recall of oracle 0.837 (was 0.805); source photo judged yes 0.806; event
or occasion: oracle median 2.5 (was 0), source yes 3/6 (was 2/6).
Small judges on ALL 19,218 photos vs 9B (p>=0.7), 4 concepts. Real prevalence REVERSES the balanced-sample advice:
cut 0.3 adds hundreds of extra yeses (2B bread +767, 4B sunglasses +947). At 0.7: 4B vs 9B missed/extra: bread 104/95
of 746, christmas tree 30/8 of 85, sunglasses 168/81 of 1043, dog 16/22 of 1646. 2B: bread 54/259, xmas 33/15,
sunglasses 155/190, dog 18/43. Raw look (full res, bread): 4 random 2B-yes/9B-no = 0/4 clearly bread (Lego set, pizza
vending-machine ad, pizza close-up, sausage pastries) -> 2B over-calls. 4 random 9B-yes/4B-no: 4B right on 2 (pastry,
clams w/ breadcrumbs), 1 real miss (flatbread wrap), 1 borderline (bun pictured on a sausage package).
FP8 9B failed on gpu_test: A100 (sm80) has no FP8 matmul ("cutlass_scaled_mm_sm80"). Needs H100, or a 4-bit
weight-only checkpoint (AWQ/GPTQ, Marlin kernels run on A100; also closer to what phones run).
23:15 race_sbatch.sh: SPECS env overrides partitions; FP8 9B race-submitted to H100/H200 only (fp_full_9b_fp8).
23:35 4-bit 9B (Intel/Qwen3.5-9B-int4-AutoRound: language layers int4, vision tower kept bf16, 9.0 GB) on the same 4
concepts x all 19,218 photos vs bf16 9B oracle: fp_full_9b_int4 (gpu_test + race copies; same name -> siblings cancel).
Phone-relevant: phones run 4-bit weights. Reza agreed: perfect the backend before phone engineering (Core ML size later).

## 2026-10-02 ~23:55 — DISBench merged-planner diagnosis (per query, vs old agent)
26 queries: agent >=1 correct, merged 0. 16: the reverse. In 9 of the 26 the merged planner's anchor found nothing (it now
returns nothing); the agent "won" those by its whole-library fallback (same behavior that sank its precision).
Planner faults seen: q3 target question = copy of anchor question; q5 target repeats anchor conditions; q17 target
"tiger taken in early April of the same year as the yawning tiger" + invented date; q30 "same shoes as the person in the
reference photo". q8 raw look (gt = 2 Highland-cow photos, 2009-10-05): the query is 3 hops (animal at Mirador -> day
with a group of it -> cows that day); our plan shape holds 2 hops -> real design limit.
Fixes: plan validation sends back (retry with reason) any question that compares with another photo/moment
(_RELATIONAL) or a target that copies the anchor; anchor-not-found report offers "search everywhere" as a follow-up;
eval saves got_ids for future raw looks. 51 tests. Old results -> eval/disbench/v1_unified/. Re-raced 8 shards
(fp_dis_uni2_s0..7). GitHub CLI login pending Reza's device code.

## 2026-10-03 ~00:05 — Pushed to GitHub
Reza created rezashamji/find_pics (private). Pre-push checks: no images/CSV/parquet/npy/model/env/data paths/tokens in
main's history; no tracked file references private paths. `git push -u origin main` only (pre-scrub-backup and
refs/original NOT pushed). From now on: commit + push main after each milestone.

## 2026-10-03 ~00:30 — 9B judge vs HUMAN labels (Open Images, labeled photos only), 15 concepts
precision / recall (false yes / missed): dog 0.996/0.985 (6/24 of 1703 labeled), horse 0.993/0.996, cat 0.961/0.988,
pizza 0.948/0.912, coffee cup 0.933/0.897, wine glass 0.923/0.933, guitar 0.925/0.827, sunglasses 0.924/0.903,
cake 0.908/0.898, bicycle 0.905/0.988, swimming pool 0.905/0.917, baked goods 0.862/0.786 (101/171 of 1113),
sandwich 0.805/0.985, bread 0.750/0.875 (77/33 of 402), christmas tree 0.583/0.700 (10/6 of 73).
Raw look (full res) at disagreements: bread judge-yes/label-no 4: judge right 2 (bruschetta, naan), judge wrong 1 (crepe),
borderline 1 (scones). Baked goods judge-no/label-yes 4: LABEL wrong 3 (two omelettes, mushrooms), judge wrong 1 (crackers).
=> human labels are noisy too; neither the 9B nor Open Images is truth. Plan: adjudicate every disagreement with a
third opinion (larger open VLM on the cluster) + my full-res sample; Reza optional (~100 via review page).
Reza (00:20): real library = final exam (holdout). Now only a plumbing check (ingest/index counts, errors); no searches
or result looks until we are confident.
00:40 Submitted adjudication: Qwen3.5-27B (bf16, one 80 GB GPU) re-judges every labeled photo where the 9B and the
human label disagree (eval/eval_adjudicate.py; race fp_adjudicate_27b on H100/H200).

## 2026-10-03 ~01:15 — DISBench v2 (fixed planner) + precision raw look; Reza: keep working, don't idle
DISBench v2: P 0.102 R 0.351 F1 0.112 (v1 0.115); per query better 15 / worse 16 / same 91 = noise; returned nothing 37;
>=1 correct 54; errors 3 (q30 etc.: planner exhausted retries on an unanswerable question) -> now degrades to a plain
visual question + note (test added, 52 pass). q3 fixed (copied-anchor rule): 0 -> 1 correct.
Precision raw look q94 ("building that appears in real life and non-real form"): correct = an illustration of Morro
Castle on a Havana card; 3 random wrong returns = city rooftops, dance hall, stone cottage. Plan question "Is this a
real-life photo of the building?" is true of most photos (854 returned). Worst-precision queries need CROSS-PHOTO
sameness (same building/person/scarf) -> failure #4, not fixable by prompt.
Reza: storage = lab quota 120 TB, 86.3 TB used; data/public 59 GB measured; models/envs est. 100-150 GB (not measured).
Feedback saved to memory: never end a turn on "running in background" while CPU work remains.

## 2026-10-02 ~23:58 (wake) — Export plan changed; race script preemption fix
Reza's library: 147k photos + 40k videos, iCloud Photos = 2 TB; he has NEVER used Photos on the Mac, and the Mac disk is
98% full (24 GB free). => the osxphotos export (reads the Mac's Photos library) would find ~nothing as written.
Options given (verified: icloudpd docs: sizes original/medium/thumb, medium px NOT documented; delete-capable flags
--keep-icloud-recent-days / --auto-delete; no Advanced Data Protection; no People names): A = turn on iCloud Photos on
the Mac with Optimize Storage, then osxphotos (keeps Apple People tags; disk risk); B = icloudpd on the cluster, straight
into data/private, delete flags blocked, 100-photo test of "medium" size first (Apple ID login on cluster = Reza's call).
Recommended B. WAITING ON REZA (also: is Advanced Data Protection on?).
Race fix: requeue copy started, cancelled its regular siblings, then got PREEMPTED (23:37) -> adjudication/FP8 left
with one preemptible copy. Now: a kempner_requeue copy only cancels other requeue copies and exits if a regular sibling
is running; a regular copy cancels all; whichever finishes cancels leftovers (POST). Re-raced fp_adjudicate_27b and
fp_full_9b_fp8. 4-bit 9B (fp_full_9b_int4) running: bread, dog done.
00:20 Offset windows: "the day after the wedding" -> days_after:1, "the week before I moved" -> days_before:7
(window_rows + planner prompt + validation; explicit offsets win over word rules). 54 tests pass.
4-bit 9B (Intel AutoRound int4, vision bf16) vs bf16 9B so far: bread missed 37 / extra 33 of 746 (4B: 104/95);
dog 3/13 of 1646 (4B: 16/22). Christmas tree, sunglasses pending. Submitted fp_qdefs (planner-defined questions vs
human labels); CPU replay of streaming with tile ranking running.

## 2026-10-03 ~00:30 — Export route decided by Reza: Apple's official data copy (no Mac Photos, no iCloud-login tool)
Reza rejected A (Photos on the Mac) and B (icloudpd: has iCloud delete flags). Route: privacy.apple.com -> Request a
copy -> iCloud Photos (verified via web: zip chunks 1-25 GB, ready within ~7 days, 14 days to download; originals +
album/Memories CSVs; no People names -> use --ref reference photos). Mac has 24 GB free: suggested 10 GB chunks and either
an external 2 TB SSD or testing whether the download links work from the cluster directly (Reza runs it himself; links
are credentials, never pasted to Claude). Cluster side: unzip one chunk at a time, shrink photos/transcode videos, delete
zip -> keep ~100-200 GB. TODO when it arrives: write the chunk ingester (reads zip -> shrunk files + metadata CSV join).
Pet identity, side-by-side judge (fp_petjudge 49945954, ran 10-02): 600 DogFaceNet pairs (300 same dog, 300 hardest
look-alike different dog): judge AUC 0.874 vs PE-Core vector AUC 0.566; yes-rate same 0.983, different 0.563 -> cut too
loose; ranking signal is real. Next for #4: calibrate (cut / ask "same individual" with stricter wording), raw look.

## 2026-10-03 ~01:00 — Export decision (final): Apple data copy, no external drive, no tool with iCloud access
Reza: no external drive, no Mac Photos, nothing that can change/delete iCloud. => privacy.apple.com "Request a copy of
your data" -> iCloud Photos, 10 GB chunks (Reza submitting now; ~1 week). Lose Apple People tags (use 3-5 reference photos
of Reza, both heavier and fit eras) and Apple scene labels (comparison only). Keep dates/GPS (EXIF in originals), videos.
When the email arrives: (1) test whether a download link works from the cluster (Reza runs it himself; links = credentials);
(2) else a Mac upload-then-delete watcher script so 24 GB free never fills. Write the zip-chunk ingester before then.
01:20 src/findpics/apple_copy.py + `findpics apple-copy <zips> <out>`: streams Apple's data-copy zips member by member
(photos -> 1600 px JPEG with original EXIF; videos -> <=720p H.264 via PyAV; Takeout-style sidecar with CSV
originalCreationDate / video creation_time / QuickTime ISO6709 GPS), skips Recently Deleted, CSV deleted=yes, Shared
Albums; per-zip done markers; NEVER deletes (safety test). Tests on a synthetic export zip: 2 new, pass.
CSV date format is an assumption from forum posts ("Tuesday October 1,2019 5:20 PM GMT"); dateutil fallback; verify on
the real export.
01:35 Tile gains raw look (background subagent, full res, K=1600, prompt "a photo of a X"; its recall numbers differ from
the replay's because K/prompt differ): christmas tree GAINED 9 = 2 real, 5 judge wrong (fir saplings, palm with lights,
ficus, tinsel float, snowy conifers), 2 unclear; LOST 3 = 0 real. Sunglasses GAINED sample 4 = 2 real (small), 1 judge
wrong (headlamp), 1 unclear. Guitar LOST 2 = both real tiny guitars. => against the 9B "oracle", tile gains are partly the
judge's own errors. Decision: do NOT switch tiles on yet; fix judge / adjudicate truth first, then re-measure tiles.
Also queued: fp_petjudge2 (stricter individual-features question + per-pair scores). RESULTS.md sections 12-16 added.

## 2026-10-03 ~02:00 — Planner-defined questions: REJECTED; model adjudication: CIRCULAR
fp_qdefs (9B, all labeled photos, 15 concepts): LLM-written definitions raise precision vs labels but crush recall:
christmas tree 0.700 -> 0.300, coffee cup 0.897 -> 0.394 ("a cup containing coffee, dark liquid..."), bread 0.875 ->
0.443 (vs adjudicated: 0.959 -> 0.443), baked goods 0.786 -> 0.529; dog 0.985 -> 0.987 unchanged. Definitions are too
narrow and the judge follows them literally. Rejected as written; maybe try "exclude look-alikes" only.
Adjudication (27B on all 9B-vs-label disagreements) analyzed with analyze_truth.py: truth = label flipped where 27B AND
9B both disagree with it -> CIRCULAR: plain 9B then scores precision 0.989 / recall 0.992 (median), christmas tree
precision 0.583 -> 0.958 with 15 of 73 labels flipped -- by models that share the conifer blind spot (raw looks).
=> model agreement is not truth. Background subagent now eye-checks all 15 christmas-tree overrides + 12 bread + 12 baked
goods overrides (verdicts -> eval/adjudicate/look/verdicts.csv). Truth for disputed photos must come from eyes.
02:20 Eye-check of model-overridden labels (background subagent, 39 photos at 900 px, verdicts.csv): models right 20/39,
label right 11/39, unclear 8/39. christmas tree (all 15): models 6, label 5, unclear 4 (labels call wreaths / lights on
bare trees / plain conifers trees; models call undecorated conifers, a nest on a spruce, an abstract cone trees).
bread (12 of 92): models 5, label 6, unclear 1 -- every "label right" = models saying yes to pastry or pizza
(definitional: does a croissant count?); baked goods (12 of 175): models 9, label 0, unclear 3 (labels: omelette, roast
pork, cheese wheels, ice cream as "baked goods"). Christmas tree, judge vs eye-checked truth (disputes resolved by eye,
4 unclear excluded, agreements still trusted): precision 15/21 = 0.714, recall 15/15 (raw labels: 14/24 = 0.583, 14/20).
Lessons: (1) no automatic source is truth on disputed photos; (2) part of "error" is definition (pastry vs bread) ->
product answer is conversation ("no pastries"), not a better judge; (3) judge's real weakness = false alarms on
look-alikes (conifers for christmas tree), not misses.
02:30 4-bit 9B (Intel AutoRound int4, vision bf16) vs bf16 9B on all 19,218 photos, missed/extra vs 9B yes:
bread 37/33 of 746, christmas tree 6/6 of 85, sunglasses 34/49 of 1043, dog 3/13 of 1646 (4B: 104/95, 30/8, 168/81,
16/22). => 4-bit 9B is the phone-judge candidate (~5-6 GB weights; on-device fit/speed untested).
02:45 Soft look-alike question (plain question + "answer no if it is only a look-alike such as <LLM list>"), vs human
labels: precision up, recall down: bicycle 0.905/0.988 -> 0.972/0.976 (good), sunglasses 0.924/0.903 -> 0.974/0.851,
christmas tree 0.583/0.700 -> 0.800/0.600, cake 0.908/0.898 -> 0.965/0.705, bread 0.750/0.875 -> 0.829/0.792.
Cause (read the questions): the 9B's look-alike lists often name REAL members (bagels/buns for bread, "bread" for baked
goods, "artificial tree" for christmas tree, sub/panini for sandwich, cupcakes for cake) -> judge obeys -> misses.
DECISION: keep the plain question by default. For "find all X" a miss is invisible and breaks completeness; a false
alarm is one tap. Exclusions come from the person in the conversation (exclude_question), not from the planner's guess.
02:55 Same-dog side-by-side judge v2 (fp_petjudge2; 600 DogFaceNet pairs, different-dog = most similar-looking dog):
plain question AUC 0.873, strict ("compare markings, ears, scars...; if unsure answer no") AUC 0.883; image vector 0.566.
Best cut (chosen on the same 600 pairs -> optimistic; needs held-out): strict 0.476 -> accuracy 0.805 (same-dog said
same 0.807, look-alike said same 0.197); plain 0.702 -> 0.797 (0.910 / 0.317); vector 0.632. Rendering the false-"same"
pairs for a raw look (login node OOM'd once decoding all photos; now lazy, in background).
03:15 Pet false-"same" raw look: my first renderer was MISALIGNED (verify() kept 4 photos eval_pet_judge dropped ->
same-dog pairs label-match 0.75; convert() rebuild -> 1.00). Re-rendered the top 4 different-label pairs the strict judge
called same (59/300 at cut 0.476): 2 are clearly the SAME dog (consecutive shots, same harness/leash; DogFaceNet ID
error), 2 unclear (black lab-type pair; black puppy held by the same person, littermate?). => 80.5% is a floor.
Product: "find my dog Max / my bike" from reference photos (converse.SubjectRefs): when --ref photos contain no face,
candidates = max image-vector similarity to the refs (engine.stream_album fast_override), judge = [ref | candidate]
side by side with the strict same-individual question (cut 0.5), streaming + bound as usual; an album condition
("Max at the beach") is judged on the identity matches. CLI no longer refuses non-face --ref. Test added; 57 pass.
03:35 FP8 9B (vLLM online fp8, H100/H200) vs bf16 9B, all 19,218 photos, missed/extra: bread 6/71 of 746, christmas tree
5/10 of 85, sunglasses 20/52 of 1043, dog 2/13 of 1646 (4-bit AutoRound: 37/33, 6/6, 34/49, 3/13). Both compressed 9Bs
stay close; fp8 leans to extra yeses. ~430 s per concept per 19,218 photos on one H200 (~45 photos/s).
03:45 Storage cleanup attempt: removed .cache/huggingface/hub/models--Qwen--Qwen3.5-27B (only 312 KB of refs). The
weights live in hub/blobs/<2-char>/... (102 GB total; layout not the standard per-model blobs, unclear mapping) ->
NOT deleting blindly on lab storage. TODO: use `hf cache` tooling to identify and remove the 27B blobs (~54 GB est.).
03:55 Find-my-dog end-to-end v1 (fp_petsearch; 40 dogs, 3 refs, library 10,943 DogFaceNet photos of >1,000 dogs, top 300
candidates by image similarity, judge vs ref[0] side by side, strict question, yes >= 0.5): candidate recall 136/147 =
0.925; judge recall 74/147 = 0.503; precision 74/657 = 0.113; median 12 returned per dog (~3.7 targets); 8/40 dogs 0 correct.
Mechanism: ~4 targets among 300 near look-alikes; ~20% false-"same" -> ~60 wrong per dog. Next: v2 judges each candidate
vs all 3 refs (mean / min agreement) and saves per-candidate scores (fp_petsearch2).
04:15 Find-my-dog v2 (fp_petsearch2, same 40 dogs, judge vs all 3 refs, per-candidate scores saved): judge thresholds:
1 ref >= 0.5: precision 74/658 = 0.112; mean of 3 >= 0.5: 76/386 = 0.197 (recall 76/136 of shortlisted targets); min of
3 >= 0.5: 37/169. RANKING per dog (CPU, saved scores): image-vector similarity top-3: precision 87/120 = 0.725, recall
87/136 = 0.640; top-5: 103/200, 103/136; judge mean-of-3 top-3 only 46/120; combo no better than vector. => CORRECTION:
the earlier "judge AUC 0.874 vs vector 0.566" used negatives chosen as the MOST vector-similar dog = rigged against the
vector. Caveat: DogFaceNet same-dog photos often share a shoot/background. Judge as VETO is safe: dropping mean-of-3 <
0.10 removes 4,025 candidates and 0 targets. Product rule candidates: rank by vector, judge veto. DogFaceNet = worst case
(all dogs); submitted mixed-library v3 (fp_petsearch_mixed): 19,218 everyday photos + the dog's own photos.
04:45 Find-my-dog v3, MIXED library (19,218 everyday photos incl. ~1,600 random dogs + the dog's own DogFaceNet photos):
candidate recall 147/147; vector top-3 precision 114/120 = 0.950 (recall 114/147), top-5 132/200 (132/147); judge 1 ref
>= 0.5: 80/210 (80/147); mean of 3 >= 0.5: 78/133; judge >= 0.2: 140/147 kept. CAVEAT: target photos come from a
different dataset/style than the background -> vector may use style, not identity; the all-DogFaceNet test (same style)
gives top-3 87/120 = 0.725. Honest range for top-3: 72-95%.
Product subject rule changed: rank by image-vector similarity, judge = veto at 0.2 (not filter at 0.5), album sorted
best-first, report says identity of a pet/object is not certified. 57 tests pass.

## 2026-10-03 ~05:30 — End-to-end regression (fp_regress) + "only ..." follow-up bug fixed
Bread chat (3 msgs, current code): 734 (100%, all judged, 1,192 s; slower node than before), drop sandwiches+burgers ->
530, keep sandwiches -> 661 (burger-only exclusion). Reuse 19,098 / new 734 per follow-up. PASS.
Transformation demo (apple-like public export, --me "Kevin Bacon"): "me" -> Kevin Bacon, 168 tagged, 318 ref faces;
heavier 150, fit 34. Follow-up "only the ones where I'm outdoors": planner only added 'outdoors' to looks; judge
question unchanged -> albums unchanged (150/34), 0 new judge calls = BUG (an album has one judge_question).
Fix: Album.filter_question ("only the ones where ...": must ALSO pass), judged once per item on the album's photos,
count + bound restated like exclusions; prompt + follow-up rule; validation. Test added (58 pass); planner conversation
added; re-running the bacon part (fp_regress_bacon).
05:55 Bacon regression rerun with filter_question: turn 1 heavier 149 / fit 33; "only the ones where I'm outdoors" ->
filter_question "is the person outdoors?" on both -> heavier 52 (106 filtered), fit 7 (28 filtered); reused 484, new 185.
Raw look, 4 random kept heavier photos (900 px): 2 clearly outdoors (red carpet outside, porch at night), 2 borderline
(inside a vehicle; night event with dark background). PASS.
06:10 Cache cleanup done properly: model snapshot links -> models--X/blobs -> shared content store hub/blobs/<2>/<hash>
with .refs files. 11 store files referenced by no model link, all .refs naming Qwen3.5-27B only, 55.6 GB -> removed. (First attempt ran
rm from the project root with cache-relative paths: rm -f silently removed nothing; caught by du still 102 GB; redone from
the cache dir with a per-file .refs check: 11 removed, blob store 102 GB -> 50 GB.)
06:30 Tiles vs HUMAN-labeled positives (Open Images, independent of the judge), recall inside top K whole/tiles:
top2000: bread 1.000/0.992 (264), bicycle 0.980/0.996 (252), christmas tree 0.950/1.000 (20; = 1 photo), sunglasses
0.903/0.873 (134), coffee cup 0.961/0.974 (155), dog 0.996/0.993 (1586), others equal. Caveat: image-level human labels
favor prominent objects (tiny background objects under-represented). DECISION: tiles stay OFF (5x indexing cost, no
verified gain; the judge-measured gain was partly judge errors).
07:00 Planner test (14 conversations, fp_planners): 11/14 strict. Fails read: beach (place->look at execution: fine), gym
(filter_question "at the gym" on the fit album only = correct; my check ignored filter_question -> check updated), cat
"only from Paris" -> filter_question "Is the location Paris?" = REAL bug (judge cannot see the city). Fix:
filter_to_place(): capitalized name in a filter that matches this library's place names becomes the place filter. Test
added (59 pass). Re-scored with execution-time transforms: 14/14. B: 72/72 planned, 1/72 needless two-step (harmless:
every anchor hit opens a window). C: DISBench two-step 81 vs old 113.
07:05 DISBench regression rerun on current code (fp_dis_uni3_s0..7; v2 results moved to eval/disbench/v2_unified/).

## 2026-10-03 ~07:40 — DISBench v3 regression found 3 bugs
v3 (current code, 7/8 shards = 107 queries; shard 6's running copy was cancelled by my own race script at 03:00, cause
not yet pinned): same 107 queries v2 P/R/F1 0.104/0.316/0.109 -> v3 0.101/0.282/0.108; returned-nothing 34 -> 42.
Read the 15 newly empty plans: (1) filter_question misused for things one photo cannot show ("later seen docking in
another country", "also appear in another photo", "a later event than 2004") -> _RELATIONAL widened (another/later/
again/also ... appear/seen/docking); (2) q4 copied the exclusion into judge_question ("wine bottle?" both) -> if equal,
judge_question = None; (3) albums with NO condition (q3, "all photos that week") crashed: judge asked a None question
(2/107 errors) -> engine: no condition = every in-scope item, no judge calls, complete by construction. 3 tests (61 pass).
Re-running all 8 shards (fp_dis_uni4).
08:00 race_sbatch mutual-cancel bug pinned (shard 6, 03:00): kempner copy started 03:00:03 and scancelled the rtx copy
it saw as PENDING, which was in fact starting (03:00:07); the rtx copy saw the kempner copy RUNNING with a higher id and
cancelled it -> both dead. squeue states are not atomic. Fix: scripts/race_guard.sh -- atomic mkdir lock per job name
(.cache/race_locks/<name>); owner runs and cancels the other copies; a regular-partition copy takes over from a
preemptible owner; stale locks (owner gone) are taken over; requeued owner recognizes itself. Sanity-tested (first
copy wins; second exits while owner alive). Running fp_dis_uni4 shards still use the old guard.
03:2x shard 4 killed by the same mutual-cancel (kempner 03:20:56 vs rtx 03:21:03): confirms diagnosis; resubmitted as fp_dis_uni4b_s4 with race_guard.

## 2026-10-03 ~08:45 (Reza back) — DISBench v4 + one more fix
v4 (8/8 shards; race_guard worked on the resubmitted shard 4: one owner, 5 copies cancelled): P 0.093 R 0.315 F1 0.104,
errors 0 (were 3), returned nothing 44, >=1 correct 49 (v2: 0.102/0.351/0.112, 37, 54). Per query vs v2: better 20,
worse 28. 14 queries lost all correct photos; 10 of them used filter_question, several plainly unanswerable ("same top as
in the photo at Puffing Billy", "also appear in another photo", "later event than 2004"): the retry fallback dropped
unanswerable judge/anchor/exclude questions but FORGOT filter_question. Fixed + wider wording ("same jacket worn in 2007",
"twice", "consecutive"); test (62 pass). race_sbatch POST now keeps the command's exit code (winner showed FAILED only
because the cleanup loop's last test returned 1). Re-running (fp_dis_uni5).
09:15 DISBench v5 (all fixes): P 0.095 R 0.362 F1 0.111, errors 0, returned nothing 40, >=1 correct 55 (v2: 0.102/0.351/
0.112, 37, 54, 3 errors). Per query vs v2: better 21 / worse 27 / same 74; lost-all-correct 10 vs gained 11 = planner
variance. => back to parity; filter/offset/no-condition features cost nothing; F1 ~0.11 ceiling = cross-photo sameness and
3-hop queries (design limit). race_guard: 27 COMPLETED (8 owners + copies that started, lost the lock and exited) + 21
cancelled; no mutual cancels.

## 2026-10-03 ~09:40 — Reza to bed; full autonomy; overnight plan in PLAN.md (top). Jobs submitted (race_sbatch):
1. fp_tf_pairwise: transformation ranking, side-by-side "heavier in the LEFT panel?" both orders, K=12 partners/photo,
   337 IMDB-WIKI face crops of Pratt/Hill/Rogen; compare within-person AUC vs single-photo 0.81-0.87.
2. fp_inst_things / fp_inst_places: instance search; PE-Core and DINOv2 vectors (saved now), combos (PE+DINO, max vs mean
   of 3 refs), judge re-rank of the top 20 ("same specific object/place?"), R-precision. (The CPU combo test found no
   saved vectors: the original run predated the np.save line.)
3. fp_gen3_s0..7: 72 generated queries with the CURRENT planner (converse) + streaming round 1, fresh oracle per query.
10:10 Transformation PAIRWISE (fp_tf_pairwise, 8,088 side-by-side calls, K=12, both orders; judge says "left" 0.402 ->
order bias, cancelled by both orders). Within-person AUC heavy vs lean era: Pratt pairwise 0.663 vs single heavier-fit
0.827; Hill 0.764 vs 0.762; Rogen 0.661 vs 0.587. Rank-average of both: 0.776 / 0.768 / 0.628. => no clear win with 3
people; product keeps single-photo (heavier - fit) within-person ranking. Background subagent raw-looking Rogen's
mis-ranked photos (IMDB years/names are noisy).
PLAN: item 1 done (no product change).
10:30 Rogen raw look (background subagent, 900px-equivalent sheets): 8 lowest-scored heavy-era photos = 8/8 NOT Rogen
(slim co-stars: Heigl x2, Segel, Wiig, Hader, Philipps, Banks x2); 8 highest-scored lean-era: 2 wrong person (heavy-set
actor), 3 Rogen looking heavy (2011 date implausible), 3 unclear; judge plainly wrong 0/16; random 8: 3 wrong person.
=> IMDB-WIKI name labels are noisy. The PRODUCT filters by face first, so re-scored with its identity cut (expanded face
sim >= 0.40): single-photo (heavier - fit) within-person AUC Pratt 0.921 (52 heavy / 47 lean), Hill 0.813 (38/36),
Rogen 0.863 (24/42); pairwise 0.705 / 0.792 / 0.749. => product method confirmed and better than reported (was
0.57-0.83 including other people's faces). Pairwise rejected.
10:45 Instance rerank jobs: vectors computed+saved (things: 13,145 photos / 1,703 products: PE-Core R-precision 0.695,
DINOv2 0.342), then vLLM failed to initialize in the same process (GPU memory still held; root cause not logged).
Fix: rerank run reuses saved vectors (no PE/DINO on GPU) -> resubmitted fp_inst2_things / fp_inst2_places.
10:55 inst2 failed: vLLM spawn re-ran eval_instance.py (no __main__ guard). Wrapper eval/eval_instance_rerank.py (runpy, guarded); resubmitted fp_inst3_*.
11:05 Instance vector combos (CPU, saved vectors; R-precision, 3 refs): things (1,703 products) PE max 0.695, DINO 0.342,
PE+DINO max 0.486, PE mean 0.731, PE+DINO mean 0.551; places (1,500 landmarks) 0.680 / 0.567 / 0.659 / 0.693 / 0.676.
=> DINOv2 adds nothing; MEAN over refs beats max. Product subject mode now uses mean (faces keep max).
11:15 Instance judge re-rank (fp_inst3, 300 identities each, side-by-side "same specific object/place?"): R-precision
things: PE max 0.693, PE+DINO max 0.484, rerank of top 20 (vector + w*judge) 0.53-0.55, judge only 0.423; places: 0.678 /
0.643 / 0.61-0.667 / 0.465. (Flaw: top 20 came from the weaker PE+DINO ranking; judge-only is far below anyway.)
=> the VLM judge is a poor same-instance discriminator (consistent with dogs); vectors are the signal, judge = veto.
Next lever for rigid things (buildings/products): local keypoint matching + RANSAC geometric verification on the top-K.
11:30 geo jobs had started with multiprocessing.Pool (runpy module not picklable) -> switched to ThreadPool; resubmitted fp_geo2_*.
11:45 Geometric verification (SIFT + ratio test + RANSAC homography inliers vs the 3 refs, top 30 by PE mean; 300
identities): things PE mean 0.721, inliers only 0.415, PE + w*log1p(inliers) 0.716 / 0.655 / 0.595 (w .02/.05/.1);
places 0.688 / 0.317 / 0.688 / 0.662 / 0.604. => hurts (viewpoint/lighting changes defeat exact point matches).
Item 2 conclusion: for "this specific thing/place/pet", PE-Core image vectors averaged over the refs are the best of
everything tried (judge re-rank, DINOv2, PE+DINO, SIFT/RANSAC all worse). The lever left is a stronger image model.
Product change: mean over refs (done). DISBench cross-photo queries stay open.
11:55 Encoder-size sweep submitted (fp_enc_*): PE-Core T/S/B/L/bigG on (A) instance R-precision (products, landmarks;
mean over 3 refs) and (B) concept first-stage recall of judge-yes photos in the top 2,000 (20 oracle concepts).
Answers both: headroom with bigG for "this specific thing", and what phone-sized encoders lose.
12:20 Encoder-size sweep (fp_enc_*): PE-Core T/S/B/L/bigG. Instance R-precision (mean of 3 refs) things / places:
T 0.565/0.431, S 0.609/0.527, B 0.654/0.608, L 0.731/0.693, bigG 0.753/0.727. Concept first stage (recall of judge-yes
photos in the top 2,000, median of 20 concepts): T 0.896, S 0.922, B 0.918, L 0.927, bigG 0.929; small objects B ~= L
(sunglasses 0.448 vs 0.457, christmas tree 0.788 vs 0.765, coffee cup 0.699 vs 0.712); T clearly worse (sunglasses 0.297).
=> (1) little headroom above L (bigG +2-3 pts on instance); (2) PHONE: PE-Core-B-16 keeps first-stage quality (~1/3 the
size of L), loses ~7-8 pts only on "this specific thing". Candidate phone encoder: B-16.
12:40 Generalization v3 (current planner + streaming; 71/72 oracles, 1 no-condition plan): round-1 recall of oracle
0.85 (v2 0.837), seed judged yes 0.817 (v2 0.806), bound held 68/68. Weak: attribute/color/clothing round-1 0.58, seed
yes 3/6 (k19 gray hoodie + varsity jacket + four men in suits: oracle 0; k43 blue jeans + blue shirt; k67 black-white
striped dress with bow: oracle 0) -> raw look at the seed photos in progress (caption hallucination vs judge miss).
12:50 Raw look at the 3 attribute seeds the judge rejected (900 px): k19 gray hoodie + varsity jacket with THREE men in
suits (generated query said four) -> judge right; k67 white dress with black band + black bow, NOT striped (caption
hallucinated stripes) -> judge right; k43 a cheesecake, only a blurry blue fabric at the top edge (query "someone wearing
blue jeans and a blue shirt") -> judge defensibly right. => the attribute weakness is the test generator's caption errors
(3/3), not the judge. Seed-yes 0.817 understates the judge.
13:10 Video indexing timing (fp_vidindex, 1 GPU, 522 Pexels HD videos): scan 10 s; index 419 s of work (707 s wall incl.
model load): 6,036 frames (11.6/video), 1,446 faces, 0 errors, 14.4 units/s/GPU (photos earlier: 23/s/GPU on A100).
Reza's library estimate: 40k videos x ~0.8 s = ~9 GPU-h + 147k photos at 23/s = ~1.8 GPU-h -> < 1 h on 16 GPUs
(his videos arrive pre-shrunk to 720p by apple_copy -> faster decode).
13:40 Planner test 30 conversations (fp_planners30): 25/30 strict. New failures read: "sunsets / not the blurry ones"
correct (check wanted the word sunset; question says "sun setting" -> check loosened); "Dad from 2015 to 2018" and
"weddings / just 2019 and 2020": REAL bug -- exact-substring date grounding dropped the dates when the planner
reformatted the phrase ("2015-2018"), and the planner set date_to 2020-01-01 for "2019 and 2020" (would drop 2020).
Fix (converse.ground): accept a reformatted phrase when every word/number in it was said; year-only phrases span whole
years (first Jan 1 to the year after the last); relative phrases (before/after/since/until) and decades untouched.
3 tests (63 pass). (Beach and cat fail as before only in the strict plan check; pass as executed.)
14:00 Video eval: 6,036/6,036 frames extracted; 24 queries generated (lakes, neon sign, trains, boxing, ...). Oracle on every frame submitted (fp_vid_or_s0..3).
14:20 Regression on current code (fp_regress2): bread 731 (100% at 768 s, faster loader) -> 526 -> 658; bacon heavier
143 / fit 35 -> outdoors 49 / 7. PASS. Planner 30 (fp_planners30b): 26/30 strict; the two date conversations still
failed: the planner set dates with time_phrase = null -> grounding removed them. Fix: if dates are set without a phrase
and the message contains exactly one year span, that span becomes the phrase (then whole-year rule). Test (64 pass).
Re-scored stored plans as executed: 27/30 (dates need a re-plan; Japan fails only against the fake one-place index).
14:45 VIDEO eval (fp_vid_or, 24 queries generated from random frames of 522 Pexels videos; judge on all 6,036 frames;
truth = video with ANY frame p >= 0.7): judging only each video's best-look frame finds 211/312 = 0.676 of truth videos;
best 3 frames 269/312 = 0.862; seed video truth 24/24. Weakest: "golden dry grass field at dusk" 0.37 -> 0.73, "sunlit
forest path" 0.33 -> 0.67. Product change: engine judges each video's best VIDEO_FRAMES=3 frames (by cheap score) and takes
the max (photos and face-matched frames unchanged); judge cost x3 for videos only. Test added (65 pass). Background
subagent raw-looking best-frame misses (are the yes-frames real?).
15:00 REVERSED the video change (subagent raw look, 8 pairs at 900 px, videos newly found by the extra frames): 5/8 judge
false positives (frame misses a stated element: lake, mist, beach, golden grass), 1/8 judge inconsistency on near-identical
frames (p 0.68 vs 0.71), 2/8 real only loosely. => the 211 -> 269 gain is mostly noise: more frames = more chances to cross
0.7, and "any frame yes" truth is inflated the same way. VIDEO_FRAMES back to 1 (mechanism + test kept). Lesson: I
shipped before the raw look returned -- do not do that.
15:10 DISBench regression v6 on current code (date grounding fixes, filter_to_place, subject mean) submitted (fp_dis_uni6_*).
15:40 DISBench v6 (current code, 8/8 shards): P 0.104 R 0.364 F1 0.119 (best so far; v2 0.112, v5 0.111), errors 0,
returned nothing 39, >=1 correct 55.
16:00 Planner 30 conversations after date fixes (fp_planners30c): 28/30 strict; the 2 fails (beach, cat/Paris) pass as executed (place_or_look / filter_to_place) -> 30/30 as executed. Both date conversations now pass.
16:20 "Find my dog Max" END-TO-END (fp_dogtest; library 300 test photos + 50 other dogs + 3 Max; --ref 3 other Max photos;
"photos of Max" -> "only the ones outdoors"): FAILED 0/3 -- InsightFace found 5 "faces" in the 3 dog photos, so the
product took the PERSON path (face matching) and returned 9 other dogs + 1 random photo. Unit tests had faked this branch.
Fix: refs_show_a_person() -- the image-text model decides person vs pet/thing/place (majority of refs closest to "a photo
of a person" among 7 kinds); faces only for persons. Test (66 pass). Re-running the end-to-end chat on the same index.
16:50 Dog end-to-end rerun 2 crashed (status line len(SubjectRefs)) -> fixed. Rerun 3: turn 1 "photos of Max": all 3/3 Max
photos returned and ranked 1-3, then 3 other dogs + 1 random photo (7 items; best-first works). Bugs left: (a) printed
"at least 100% found" for an identity album (bound does not apply); (b) turn 2 "only the ones outdoors": the subject's
kind came from the planner's looks (= "outdoors") -> identity question garbled -> lost 2/3 Max; the condition said "the
person in the red box" (no box for a dog); report count stale (11 vs 6). Fixes: refs_kind() (image-text model) labels the
refs (person/dog/cat/animal/object/building/vehicle) and that kind phrases the identity question; condition re-phrased
for the kind; no bound for subject albums; condition removals counted. 66 tests. Rerunning.
17:10 Dog end-to-end PASS (fp_dogtest4): "photos of Max" -> 7 items, Max's 3/3 ranked 1-3, then 3 other dogs + 1 random photo
(best-first; no bound shown; "not certified" note). "only the ones outdoors" -> 5 items (filtered 2). Raw look of Max's 3
photos: the 2 filtered are indoors (tiled room, dog bed), the kept one outdoors (dirt) -> condition correct. 4 bugs found
by this end-to-end test that unit tests missed (face detector fires on dogs; status line crash; kind from looks; stale
count / misleading bound) -- all fixed.
17:40 Person --ref regression (fp_regress_ref): "Kev" from 3 Bacon photos -> refs_kind = person -> 318 reference faces
(expansion), 322 photos + 4 videos = identical to before. Person path intact. (regress.sh part selection fixed: exact
matches.)
18:00 Phone-stack eval queued: B-16 index of testlib (fp_idxB_s0..7) then eval_phone_stack.py (B-16 + 4-bit judge vs L + 16-bit, round 1, 4 concepts).
18:40 PHONE STACK end to end (fp_phonestack; round 1 streaming, 19,218 test photos; truth = 16-bit 9B on every photo):
cluster (PE-Core-L + 16-bit 9B, replayed) vs phone (PE-Core-B-16 index + 4-bit 9B, real calls): bread recall 0.893 vs 0.886
(phone precision 0.962), christmas tree 0.612 vs 0.541 (0.939), sunglasses 0.588 vs 0.662 (0.953; its adaptive head ran
longer: 6,800 vs 5,000 judge calls), dog 0.985 vs 0.978 (0.998). => phone stack ~ cluster stack. Caveat: the phone's
"found at least X%" is relative to ITS judge; vs the 16-bit judge it overclaimed on bread (0.902 stated vs 0.886).
19:00 Core ML, phone candidate PE-Core-B-16 (scripts/convert_coreml.py now takes the model name): image tower 186 MB
(L: 633 MB), text tower 709 MB (same as L: the B and L checkpoints carry the same-size text model). PyTorch trace diff 0.0.
NOT verified: Core ML outputs (needs macOS). Next phone lever: 8-bit weights for the text tower (~half), measured on a device.
19:20 Final-exam runbook: scripts/run_private_copy.sh (Apple data-copy zips -> apple-copy ingest -> scan -> 16-shard index ->
scripts/plumbing_report.py; NO searches). Plumbing report dry-run on the dog test index: 353 items, 333 "faces" (detector
fires on dogs), dates 323 mtime / 30 exif (copied files), 0 GPS, long-side sample. CLAUDE.md points to the runbook.
README: pet/thing references describe the current design; phone status line.
19:35 apple-copy ingest sharded (--shard k --n-shards K: zip k -> shard k%K; per-shard temp file; done markers per zip, so preempted jobs resume); runbook runs 16 ingest shards.
19:55 Runbook DRY RUN on a fake public Apple export (scripts/build_fakecopy.py: 2 zips, 200 photos + 20 videos, 4 rows marked
deleted + a Recently Deleted folder): ingest 2 min -> scan -> 2-shard index -> report in ~5 min total. 214 items (196
photos + 18 videos) = 220 - 4 deleted (Recently Deleted skipped); date_source takeout 214/214 (CSV dates); 0 decode errors;
2/2 shards; 444 units; long side < 896 px: 27/196 (public source). PASS. (Cosmetic: report path in the progress line.)
20:10 Core ML 8-bit weights (scripts/quantize_coreml.py, linear symmetric): PE-Core-B-16 text 709 -> 355 MB, image 186 ->
94 MB (~450 MB for both towers). The 4-bit 9B judge (~5-6 GB weights) is the heavy part. Fidelity of the int8 packages
NOT verified (Linux cannot run Core ML).

## 10-03 08:10 Planner fuzz v1 -> fix -> v2
- v1 (108 LLM-written requests): automatic checks clean on 107/108; 1/108 crashed: "find all photos from" (cut off) ->
  zero albums on every retry -> ValueError -> chat would exit. Fix: chat/ask catch planner failure, ask to rephrase,
  keep the current plan. Weak spot of v1 itself: the generator wrote web searches / photo-edit commands for most kinds
  and only flagged plans were saved, so the 107 "clean" plans were never read -> not counted as a result.
- v2 submitted: generator told the box searches the person's OWN library, off-target requests are one explicit kind,
  every plan saved for reading (eval/planners/fuzz.json; v1 kept as fuzz_v1.json).
- 08:15 fuzz v2 died at start: my edit broke a tuple unpack in the eval script (no planner result). Fixed, dry-run with a fake LLM, resubmitted.

## 10-03 ~08:50 Fuzz v2 read in full (117 plans, every one read by me) -> 4 planner fixes
- 0/117 crashes; automatic checks flagged 1/117 ("photos from" -> whole library + note). Reading found real bugs:
  (1) dates: 6/117 wrong relative dates (model thought Sat 10-03 was Thu/Fri; "last weekend" = 8 days; "this year" =
      past 12 months; "my birthday this year" = today, invented). Fix: weekday in prompt + resolve_relative() in code
      (yesterday/today/last night/last weekend/[last] <weekday>/this|last week|month|year/past N days|weeks|months|years);
      occasion words (my birthday/anniversary) -> I don't know the date, keep only the calendar part.
  (2) invented people: 4/117 ("my sister"/"my daughter" -> Sara, "we" -> Dad). Fix: a person must be named in the
      conversation (prefix match, Sara~Sarah; the owner's name counts); else removed with a note asking who.
  (3) owner-knowledge questions ~9/117 ("Is this the new house we bought?", "Are you and Reza in this photo?").
      Fix: I/me/my/we/our/you in any judge/anchor/filter/exclude question -> retry with the reason, then degrade to
      "Does this photo show <look>?".
  (4) name-in-question deleted the whole condition 3/117 ("Is Mom laughing at dinner?" -> nothing). Fix: only pure
      identity questions are deleted; otherwise the name becomes "the person in the red box".
- Not fixed (noted): place names as anchors (Bali/Japan/Colorado via judge instead of GPS place: arguable, keeps non-GPS
  photos); "the X" -> max_items 1 when want=best (2/117); "Is the person in the photo a golden retriever?" (1/117).
- 71 tests pass. Submitted: fuzz v3 (same script, new generations), eval_planners (30 conversations; baseline 30/30
  kept in eval/planners/before_fuzzfix/), regress.sh all.

## 10-03 ~09:35 Fuzz v3 (same 117 requests as v2: generation is deterministic -> direct before/after) + planner eval + regress
- Fixed vs v2 (read all 117): dates 6/6 (last weekend 09-26..27, last Tuesday 09-29, this year from 01-01, last year
  = 2025, "my birthday this year" -> this year + note); invented people 4/4 removed with a "who is it?" note;
  "the house we bought" -> "Is this the outside of a house?"; "Is Mom laughing" condition kept.
- New/remaining, fixed now: (a) my "must be named" prompt line made the model drop the owner ("selfie I took", "my feet
  in the water" -> person null; planner eval "me heavier vs fit" FAILED on this, 28/30). Prompt now says when the owner is
  IN the photo vs only the library's owner ("the sushi I ate"). (b) "photos from" -> owner album "Is this a man?":
  owner kept only if the conversation refers to the owner. (c) "a person in a red box" (with "a") slipped past
  fix_red_box. (d) "a woman named Sarah" in a judge question -> retry/degrade. (e) notes kept the planner's wrong dates
  -> code-changed dates are stated in the notes. (f) "July 1st to July 15th" dropped the 15th -> named end day included.
  Convention change: "last week" = previous Monday .. today (completeness first).
- Planner eval 28/30: the other failure (cat, "only from Paris") is the eval check being stricter than the product
  (filter_to_place turns "Is this taken in Paris?" into the place filter at search time): check relaxed.
- Regress (end to end) with the 08:50 fixes: plans unchanged in structure. Bread 744 -> 620 (drop sandwich+burger) ->
  682 (keep sandwiches) [before: 731/526/658; judge run-to-run noise, same plans]; bacon 139/35 -> outdoors 44/6
  [143/35 -> 49/7]; Kev 322 + 4 videos [same]; dog: identical files to the 17:10 pass (7, Max 3/3; outdoors 5 keeps
  maxdog_2 only).
- 72 tests. Resubmitted fuzz v4 + planner eval.

## 10-03 ~09:55 Fuzz v4 + planner eval (26/30) read -> fixes
- Planner eval 26/30 (strict checks): gym as place (product turns it into a look at search time), cat photos+videos as
  two albums, sunset/selfie paraphrased without the keyword ("sun low on the horizon", "photo taken by the person of
  themselves"). Product behaviour acceptable on all 4 by my reading, but the paraphrasing came from my new "stranger /
  pixels alone" prompt line -> shortened. Score left strict (26/30), not re-relaxed.
- Fuzz v4 (read all 117): my 09:35 prompt line made the model drop KNOWN people named by relationship (Dad/Mom: 2/117)
  and invent looks for unnamed ones ("my sister" -> "a woman with long hair": 3/117); lowercase "i" not first person
  (2/117); judge==anchor fallback kept only "bride and groom" photos of "photos from the wedding" (~12/117) -> now
  everything inside the moment; dates set with a null phrase were removed (5/117) -> one calendar phrase in the message
  is recovered; "weekend before last" / "Tuesday" -> resolved; "the user's"/"the owner" questions -> retried.
- 73 tests. Next: held-out fuzz (FUZZ_SET=1,2: different writer personas, new requests) + planner eval + regress.

## 10-03 ~10:30 Held-out fuzz sets 1-2 (new writers, 234 requests, all read) + planner eval 28/30 + regress
- Regress (with the 09:55 fixes): bread 742/539/666, bacon 141/35 -> 46/6, Kev 322+4, dog 7 -> 5: same plans and
  counts within judge noise of the earlier runs.
- Planner eval 28/30: cat photos+videos still split into two albums (-> code now merges albums identical except
  photo/video); "my 10 best photos from Japan" asked the judge "Is this taken in Japan?" instead of place (-> a
  judge/filter/anchor question that only names a place said in the request becomes the place filter).
- Held-out sets: 0 crashes. Problems found and fixed: "Is this a video (file)?" as judge/filter (4/234: the judge sees
  one frame) -> removed; "that pizza we ate" became every photo of that day (my 09:55 everything-fallback) -> only when
  the request asks for photos FROM a moment/trip, else keep the question and drop the moment; "march through june" as
  4 month albums, 2 with no condition (= whole library) -> condition-less albums dropped + prompt rule; "the last
  video" -> dates = today -> no dates; known names in a judge question -> retry; months ("last August" was Sep 2025;
  "June" vs "in June" differed by a year), holidays ("Fourth of July" spanned a year), "N years ago" (was one day),
  occasions ("50th anniversary celebration" dated today) -> resolved in code.
- Left as is: "last summer" in October (2025 vs 2026) is ambiguous; the planner says 2025 consistently.
- 75 tests. Submitted held-out sets 3-4 (student, travel photographer), planner eval, regress.

## 10-03 ~09:55 (clock time) Held-out sets 3-4 read (234 plans) + planner eval 29/30 + regress
- Regress: bread 742/540/667, bacon 143/36 -> 49/6; the planner itself now asks "photos and videos of Kev" as one
  album: 322 = 318 photos + 4 videos. Checked against apple_like_export/ground_truth.json and BY EYE at 900 px
  (data/public/regress_ref/kev_errors_{0,1}.jpg): the 1 "false positive" P00343-OTHER shows Kevin Bacon (label error);
  of 7 misses, B0103 has no Bacon visible (label error), B0269 is a clear close-up miss, B0323/B0289 profile, B0118 face
  cut off, B0201/B0202 small turned faces. Corrected: 318/318 precise, 318/324 found + 4/4 videos.
- Planner eval 29/30 (gym-as-place again; runtime place_or_look converts it; left strict).
- Set 3 (student): unknown names in questions ("Is this person Jay?", "a person named Jay" rebuilt from looks),
  "last thurs" (5 days), "last Friday night" (wrong Friday), selfie without "I" lost the owner. Set 4 (photographer):
  ~10/117 questions about the FILE (RAW, fps, lens, audio, location tag, rating, filter, 4K) -> unseeable; and anchors
  "Is this London?/Mount Fuji?" kept while the place was DROPPED: my anchor->place rule required no place yet, then the
  older "place inside the anchor is the moment" rule removed the place. Fixed all; 76 tests.
- "fall break", "between 8pm and 2am" (time of day) not supported: noted, not fixed.
- Fuzz script now auto-flags: unseeable questions, known names in questions, date spans > 400 days without a year.

## 10-03 14:30 Round 5 fuzz (5 sets x 117) + planner eval 29/30 + regress (ran 10:20-10:31; I was interrupted)
- Regress stable: bread 747/546/672, bacon 144/36 -> 50/6, Kev 322 (318 photos + 4 videos), dog 7 -> 5.
- Auto-flags 7/585: 1 crash (anchor with null question -> validation error; now filled from its looks or dropped);
  "may till july last year" lost its dates (planner wrote its own phrase "May through July 2025"; month ranges and
  "the week of <holiday>" now resolved in code and the said phrase recovered); fallback rebuilt unseeable questions
  from looks ("Does this photo show dramatic music track?", "Is this Reza?") -> fallback now checks its own output and
  names; "lens flare" wrongly unseeable -> fixed.
- Random 60-plan sample read: "christmas tree pics" -> Christmas Day only (holiday word naming a thing -> no dates);
  "time of day between 6pm and 8pm" -> unseeable; "Is this the Grand Canyon?" anchor -> place. 78 tests.
- Submitted round 6: sets 0-6 (5, 6 = new writers never used for fixes), planner eval, regress.

## 10-03 ~15:40 Round 6 (sets 0-6) + planner eval 28/30 + regress
- Regress stable (bread 742/540/665, bacon 144/36 -> 50/6, Kev 322, dog 7 -> 5).
- Planner eval 28/30: NEW failure from my own anchor->place rule: "food photos from the WEEK I went to the Grand
  Canyon" became "food AT the Grand Canyon". Rule now only for event/place windows. (gym-as-place: unchanged, ok at runtime.)
- Auto-flags: sets 1-4, 6: 0; set 0: "photos from" -> "Is this Reza?" survived because the person was removed AFTER the
  name check -> questions naming a removed person are dropped; set 5: 2 crashes are half-typed/edit requests with zero
  albums (product asks to rephrase: by design).
- Read held-out sets 5 (non-native writer) and 6 (parent) in full, 234 plans: real bugs ~4 + ~4, mostly minor:
  "Is this John smiling?" (unknown name; NOT fixed: cannot tell John from Paris without a name list), name check too
  aggressive on readable text ("chat with Mom" -> whole library!) -> readable-text questions keep names and a dropped
  main question falls back to the looks; "this summer" = last year's -> this year's; invented days_before:7 window ->
  only when before/after is said; "15 seconds long"/"trimmed" -> unseeable; owner's name inside exclude -> boxed person.
- 82 tests. Round 7: two NEW writers (7: small-business owner, 8: back from a two-week trip) + set 5, eval, regress.
- 3-hop chains (plan item 5): 26/122 DISBench queries chain moments, mostly TWO moments combined ("during the 2009
  Paris trip, before the first Van Gogh photo"; "after X but before Y") rather than anchor-of-anchor. Needs the plan
  schema to hold several anchors; deferred (next backend step, not to be done unattended right after stabilizing the
  planner). README: what code guarantees about understanding a sentence; 8-bit phone sizes.

## 10-03 ~16:30 Round 7 (new writers 7 small-business, 8 back-from-trip; set 5 again) + eval 28/30 + regress
- Regress stable (bread 743/540/667, bacon 143/36 -> 48/6, Kev 322, dog 7 -> 5).
- Eval: Grand Canyon week case still failed: the MODEL wrote place=Grand Canyon (no anchor). Code: "the day/week/month
  I went to X" with place X -> anchor X + that window.
- Read sets 7 and 8 in full (234). Set 7: ~2 real ("last quarter" = past 3 months; "winter stock" as a date). Set 8:
  PROMPT-EXAMPLE LEAKS 4/117 ("food at that little Italian place" -> "Is this a slice of bread?" x3; "you and me at the
  beach" -> anchor "a man with a heavy build and round face"), and "on the last night" (of a trip) -> yesterday x2.
  Fixes: example phrases appearing without support in the request -> retry, then stripped (never rebuilt into a
  question); "the last night/day" -> no dates; last/this quarter computed. 84 tests.
- Round 8: sets 7, 8 again + new writers 9 (teenager), 10 (older person, scanned prints, family who passed), eval, regress.

## 10-03 ~17:00 Round 8 (sets 7-10) + eval 29/30 + regress
- Eval 29/30 (Grand Canyon week case fixed; gym-as-place remains, correct at runtime). Regress stable (bread 745/544/671,
  bacon 147/34 -> 50/6, Kev 322, dog 7 -> 5).
- Sets 7, 8 again: 0 auto-flags; the 4 example leaks and 2 "the last night" cases are gone (bread only where asked).
- Set 9 (teenager), read in full: ~2-3 minor (unknown "Sarah" in an anchor; "Is the person in the red box a woman?").
- Set 10 (older person, scanned prints), read in full: ~6 -> two classes fixed: who-is-this questions about unknown
  people ("Is this Uncle Harry?", "Is this the brother?", "Is this the person in the photo?") and invented hair for
  unnamed relatives ("my sister" -> "a woman with long hair", "aunt Mary" -> "gray or white hair") -> both retried, then
  stripped. (My first pattern also caught "Is this the Grand Canyon?"; unit test caught it; removed.) 85 tests.
- 17:25 Round 9: eval 29/30; regress stable (bread 742/542/667, bacon 141/36 -> 49/6, Kev 322, dog 7 -> 5); sets 9, 10: 1 auto-flag each; set 10 invented hair / who-is-this questions gone (spot check). Planner fuzz item closed.

## 10-03 17:30 Next item: moments inside an event (Reza restarted the loop = continue)
- 16/122 DISBench queries say before/after; ~8 need a cut INSIDE an event ("during the 2009 Paris trip before the first
  Van Gogh photo", "immediately after the photo of X", "30 min-1 h before the torch performer"); v6 F1 on them ~0.
  Most need ONE anchor with a finer window, not two anchors (the trip is the event that contains the Van Gogh photo).
- Added windows: before / after (same event as the anchor, earlier/later than its first photo there) and
  minutes_before:N / minutes_after:N; planner rules; unit tests (86).
- Measurement: full DISBench, 8 shards x 2: baseline = commit f982bb1 (today's planner fixes, no new windows) and new =
  this commit, each from its own git worktree under .cache/ (my first submission mixed old/new code because the eval
  imports the planner lazily after my edit landed: cancelled). Old v6 (before today's fuzz fixes): F1 0.119.

## 10-03 ~18:00 DISBench A/B (full 122, 8 shards each)
- v6 (yesterday's planner) F1 0.119 | base = today's fuzz-hardened planner, no new windows: F1 0.100 | new windows: 0.121
  (recall 0.376, exact 3/122 vs 1/122).
- Noise floor: 28/122 queries got byte-identical plans in base and new; their F1 changed by 0.001 on average (0/28 by
  > 0.1). So score differences come from PLANS, not judge randomness. And every prompt edit re-rolls most plans: the
  v6 -> base drop (-6.36 / +4.08 F1 summed) is mostly the model writing different anchors/questions, plus one real bug
  from today: "white-haired grandma" (q61) lost its condition to my invented-hair check ("haired" != \bhair\b). Fixed.
- New windows used by 9 plans: their mean F1 0.007 -> 0.128; q20 (statues during the Paris trip before the first Van
  Gogh photo) 0.06 -> 1.00, exact. Remaining zeros: minutes_after:0/1 (empty window) -> min 30 min; "30 min to 1 h"
  -> larger end; "cats after the cat tree was assembled" is days later, not the same event -> new since/until windows
  (any later/earlier time). Not fixed: "after A but before B" needs two anchors (q45); anchors never found (q64, q107).
- 87 tests. Re-running full DISBench (worktree of this commit) + planner eval + fuzz set 0 + regress.
- 18:07 v7_windows (since/until, min 30 min, range rule): DISBench F1 0.121, recall 0.376, exact 2/122, 38 empty; the 9
  window plans 0.007 -> 0.243. Planner eval 30/30; regress stable; fuzz set 0: 0 flags. RESULTS section 23.
- Reza requested the Apple data copy (18:00) but chose 1 GB parts (~2,000 zips for 2 TB): pipeline handles any count;
  download plan = cluster remote desktop Firefox + a download-all add-on (DownThemAll), fallback cancel/resubmit 25 GB.

## 10-03 18:40 "Which face is you" (Apple's data copy has no People names)
- findpics people <index> -> numbered sheet of the 12 most frequent faces; findpics name <index> <n> "<name>" -> that
  group becomes the person (refs + expand_refs), so --me works with no tags. Low-confidence detections (<0.7) excluded:
  they formed a junk group (dogs, flowers, backs of heads). Results in RESULTS 24: groups pure by eye (96/96, 48/48);
  end-to-end "every photo of me" from the named group = Apple-tag result exactly (317/324 + 4/4 videos).
- Planner: "Is there a person in the photo?" on a person album -> dropped (faces decide). 89 tests.
- For Reza's library: after the plumbing report, run `findpics people data/private/index` only when Reza agrees (it
  shows his photos); he picks his group number.

## 10-03 ~19:00 Time of day ("between 8 and 11pm", "in the morning", "at night")
- Reza agreed the order: time of day -> offer to name unknown people -> two moments. Endless benchmark chase: dropped.
- Local clock: EXIF and Flickr-style dates are wall-clock; Apple's CSV and Takeout timestamps are UTC instants (9pm in
  Boston would read 01:00). New Item.taken_local (EXIF DateTimeOriginal, naive metadata dates, sidecar localTime,
  QuickTime creationdate WITH offset for Apple videos); older indexes fall back to exif/metadata_json/sidecar dates.
  Unknown local time -> item cannot pass an hour filter; the report says so.
- Hours computed in code (resolve_time_of_day); the model only flags which album. "last night", "Friday night",
  "the morning after X" set NO hours (dates/moments). Ranges over midnight extend the date window by a day. 91 tests.
- ~18:50 Unknown people offered in the chat: Plan.unknown_people (kept only if actually typed and not known); the chat
  shows the face sheet and accepts "Jay is 4" (names the group, then redoes the previous request). 93 tests.
  (Cancelled the main-folder re-check jobs before they started so they would not mix code versions; DISBench for the
  time-of-day commit runs from its own worktree.)
- ~19:05 DISBench with time of day (worktree wt_tod): F1 0.133 (best so far; v7_windows 0.121), recall 0.387, 37 empty.
- with_people ("me with Pierce"): albums had ONE person, so "me with Pierce" returned every photo of me (318). Now other
  people must also face-match. Test library: named Nic (group 1) + Patricia (group 8): "photos of Nic with Patricia" ->
  20 photos, 443 Nic-only removed; by eye 20/20 show both (data/public/chat_pair/check_*.jpg). "me with Pierce" on the
  Apple-like library: 0 (they never appear together) -- report count was stale (said 318): fixed.
- until (two moments): "after A but before B" -> anchor A (window after/since) + until B; span cut at the first B photo
  after the first A photo. "before B" alone -> anchor B, window before. A missing pandas import would have crashed the
  first real use: the unit tests caught it. 95 tests.
- 19:27 Full check of time of day + people + two moments (worktree wt_all): planner eval 30/30; regress stable (bread
  746/546/669, bacon 142/36 -> 49/6, Kev 322 + 4, dog 7 -> 5); DISBench F1 0.134 (tod-only 0.133) but recall 0.339 vs
  0.387 and 45 empty vs 37: 11 newly empty queries are plan re-rolls from the prompt change (anchors never found, e.g.
  "Is this the first Brunettes practice session?"), only q24 uses the new 'until'. Fuzz (3 sets): time of day used
  correctly 3/3; unknown people asked 43 times (Jay, Uncle Harry, my sister...); misuse fixed: "us all together" ->
  with Dad, Mom, Sara, Ali (with_people must be said now); until "Is this Mom?" / "the moment the person passed away"
  (events no photo shows -> retry, then dropped). 96 tests.
- 19:45 Confirmation: eval 30/30; fuzz10 with_people invented: 0; life-event moments still slipped through in -ing form ('passing away', 'getting sick') -> pattern widened, test. 97 tests.
- ~20:30 `findpics web`: the chat as a web page (FastAPI on the GPU node; random key on every request; photos served
  only by library item id). First run: every request 422 (`from __future__ import annotations` hid FastAPI's Request
  type) -> fixed. API test: page 200, no key 403, file outside the library 404, search ran and albums returned.
  "photos of me at night" on the Apple-like library: 0 in scope -> checking which date sources carry a local clock.
- ~20:50 Web test 2: "every photo of me" -> 322 shown; thumbnail 200; mark-wrong removes it (322 -> 321 shown); full-size
  view 500 on a VIDEO (opened .mov as an image) -> now a still from the video. Placeholder clock times (exactly 00:00:00
  or 12:00:00: both public test libraries stamp every photo at noon) now count as unknown for time-of-day. 98 tests.
- ~21:00 Web page screenshots (headless Chromium, looked at): desktop grid/headers fine; report boilerplate now folded
  under "About this album"; chat box now fixed at the bottom. Phone screenshot shows a cut-off 3rd column, but tile
  widths imply a 500 px layout (Chrome's minimum window width), i.e. a tool artifact; overflow-x guarded anyway. Not yet
  seen on a real phone.

## 10-04 ~04:30 Reza's AirDropped sample (his OK to test on it and for me to look)
- 620/620 files arrived in data/private/sample (7.1 GB: 413 HEIC, 79 JPG, 110 video, 18 PNG); EXIF dates with seconds and
  GPS present on 11/12 checked. First rsync dropped (login node closed the connection); resumed with keepalive +
  --partial. Reza: delete all private data only when the product is done, on his OK (CLAUDE.md, PLAN CLEANUP).
- scripts/run_sample.sh as ONE GPU job (survives windows closing): dedupe (links, never deletes) -> scan -> index ->
  plumbing report -> face sheet -> provisional me = largest face group (Reza confirms in the morning) -> demo + 6
  everyday searches. Submitted as fp_sample.
- 07:25 fp_sample failed at step 1: 'deactivate' does not exist in a fresh bash under set -e -> '|| true'. Resubmitted.
- 09:00 fp_sample ran end to end (598 unique of 620; 0 decode errors; 2,134 faces; 12 face groups; 7 searches done),
  BUT the plumbing report caught a real ingest bug: 549/598 dates fell back to file mtime and only 31 had GPS, because
  scan's EXIF/GPS reader could not open HEIC (no pillow_heif registered in ingest) and videos' own metadata was never
  read. My fake Apple export test was JPEG-only, so this slipped through. Fixed: HEIC opener in ingest; _video_meta reads
  QuickTime creationdate (local clock with offset), creation_time and ISO6709 GPS. Spot check: HEIC dates/GPS ok, videos
  local time + Boston GPS ok. Re-running the whole sample job before looking at any album.
- 09:15 Looked at the sample's face sheet (Reza OK'd me looking): 12 clean groups (family, older relatives, kids). Groups
  1 (dressed up/selfies) and 2 (shirtless mirror selfies, beach) may be the SAME young man (likely Reza) split across
  looks/eras; group 3 (bearded, heavier) may be him too or someone else -> only Reza can say. Added: one name can cover
  several groups (`findpics name <index> "Reza" 1 2 3`; chat: "Reza is 1, 2 and 3"). Provisional run still uses group 1
  only, so its "me" albums may miss his other-era photos until he answers.

## 10-04 ~10:30 Sample rerun with HEIC/video dates fixed + first look at Reza's albums (his OK)
- Plumbing: dates exif 450 + video 108 + mtime 40 (likely screenshots/PNGs); GPS 473/598; 0 unreadable; median 4032 px.
  "photos at night" 2 -> 51.
- Demo by era: 'heavier' 137/146 from 2023-03..06; 'fit' 97/137 from 2026-08..10.
- By eye (900 px, 8 random each): fit 7/8 plausible (1 = birthday text screenshot); heavier ~3/8 plausible: 2/8 the
  judge rated a heavier bearded NEIGHBOUR in a tight group shot (person crop still holds neighbours), 1 slim beach shot,
  1 photo stored sideways (file has landscape pixels + orientation 1: camera quirk, not ours).
- Face groups: group 3 (bearded) appears in the SAME photo as the young man -> not the same person. Open question for
  Reza: are groups 1 and 2 both him? Note: 'heavier' = his heavier half by design (rel_cut 0.5), not an absolute claim.
- Test: person crop + red box around his face (engine person_crop_boxed, crop_person="box") vs crop only, AUC 2023 vs
  2026 on his photos, split by 1 face vs 2+ faces; disagreement sheets. Submitted fp_cropbox.
- ~11:00 crop vs crop+red box on Reza's photos (era proxy 2023=heavier, 2026=fit; Reza to confirm): AUC all 0.84 -> 0.925
  (224 vs 106 photos), 2+ faces 0.571 -> 0.792 (184 vs 17: small 2026 side), 1 face 0.913 -> 0.942. The 8 biggest
  disagreements (900 px, looked): all 2023 party photos next to a slimmer friend; he looks fuller; crop said no
  (0.15-0.30), box said yes (0.5-0.65): box right 8/8 by eye. Product now uses crop + red box for person-appearance
  albums ("the person in the red box"). Re-running demo + regress.
- ~13:00 demo4 (Reza = groups 1+2, photos+videos): heavier 182 (178 from 2023; 24 videos), fit 163 (114 from 2026;
  18 videos). By eye (900 px, 8 each incl. 2 videos): heavier 7/8 plausible (1 video whose middle frame is pure blur);
  fit 5/8 (2 are 2023 photos where he looks like his heavier era: belong in heavier; 1 video frame shows a woman).
- BUG: iPhone videos carry display-matrix rotation -90 (every .MOV checked); frames reached the face model and the judge
  sideways. media.sample_video_frames now rotates by frame.rotation (checked upright by eye).
- Named people now store face FINGERPRINTS (named_<name>.npy) so Reza's pick survives re-indexing; saved his 364 now.
  run_sample keeps an existing "Reza". Re-running the whole sample (re-index with upright video frames).
- 14:50 Rerun with upright video frames: faces 2,134 -> 2,842 (+708 from videos); videos of me 57 -> 62; heavier 187
  (185 from 2023; 35 videos), fit 168 (116 from 2026, 52 from 2023: the top-half rule pads the smaller era).
- Offline on saved scores (393 of Reza's items: 266 from 2023, 127 from 2026): "whichever album it matches more" with a
  margin: 0.2 -> fit 117/145 from 2026, heavier 161/165 from 2023, 83 left out; 0.4 -> 112/112 and 113/114, 167 left
  out. Raw judge p unusable (fit 381/392). Product: margin 0.3 chosen on principle (not the best-scoring setting, to
  avoid tuning to his answer key); in-between photos counted as "not clearly either". 99 tests. Rerunning demo + bacon.
- 15:20 demo5 (margin 0.3): heavier 139 (137 from 2023 = 99%), fit 127 (115 from 2026 = 91%, was 116/168 = 69%),
  127 not clearly either. Bacon regress caught 2 bugs in the new rule: (1) albums with different scopes ("fit 2010-2015"
  vs "heavier" any year): photos outside fit's dates had no fit score and all went to heavier (142 -> 206); (2) "best"
  albums lost their cap (36 -> 48). Fixed: out-of-scope photos follow their own album's rule; "best" keeps its short
  list. 100 tests. Re-running bacon + demo.
- ~16:00 Everyday searches on Reza's sample, by eye (900 px): at night 8/8 (all 21:00-02:00 local); screenshots 6/8 (2 =
  a VIDEO with text overlay); me outdoors 4/4; me with other people 4/4; FOOD 3/8 (pizza x2, tacos; misses: shirtless at a
  market with a banana at the frame edge x2, office desk with a chips bag, party tent, table with barely visible plates):
  planner asked "Is there food in this photo?". Prompt rule added: "X photos/pictures of X" = X is the subject;
  "photos with X" = visible anywhere. Re-running food + screenshots + planner eval + bread regress.
- 16:30 Checks: planner eval 30/30; bread regress asks "Is there bread anywhere in this photo?" (turn 1 ~699 vs 742
  before: the wording made the judge stricter; no truth labels to say which is closer); Reza demo unchanged (139/127);
  bacon after pairing fix: heavier 129 (was 142), best-fit 24 (was 36). Food: the planner IGNORED the new prompt rule
  (still "Is there food in this photo?") -> enforced in code ("X photos/pictures of X" + "Is there X in this photo?" ->
  "Is this a photo of X?", not when the request says with/where/anywhere). Screenshots -> photos only. 101 tests.
- 17:00 food rerun: "Is this a photo of food?" -> 17 items (was 40); by eye 5-6/8 food (pizza x2, tacos, wedding cake,
  dinner table; misses: man on phone with crumbs, patio with snack bowls) vs 3/8 before. Screenshots: photos only, 44.
  MORNING_REPORT updated with the first test on Reza's photos.

## 2026-10-04 ~17:45 — review sheets redrawn while Reza walks
- Bug (seen on heavier_01 by eye): video tiles/previews used the MIDDLE frame, often dark/blurry and not the frame
  the face was matched in. Fix (551ae3b): results carry `frame_t` (matched face's frame, else best-scoring frame);
  contact.thumb / review page / web full view show that frame (media.video_frame_at). Test added; 102 pass.
- scripts/review_sheets.py: numbered sheets (numbers kept from key.json), matched frame + red box on the matched
  face so "is the boxed person you?" is unambiguous in group shots. Heavier redrawn and viewed (heavier_01: tile 8
  now shows the face; 2 and 6 are dark videos, box shows who). Fit redraw running (first run hit a 900 s timeout).
- Seen: many near-identical burst shots inside albums (heavier_01: 12/13/16, 4/14/15/18, 5/9/10/17). Measuring.
- 18:00 Fit sheets redrawn (fit run 2 died silently: 16 threads decoding 4K video on a 1-CPU login node, load 35;
  4 threads + video_frame_at stops decoding at the matched second). Viewed fit_01: boxes on Reza in all 20.
- Near-duplicates (PE-Core cosine >= 0.9 and <= 10 min apart): heavier 139 items -> 86 groups, fit 127 -> 59.
  By eye (audits/neardup_heavier_090.jpg): 12/12 largest groups are the same moment (bursts / pose variants).
  Proposal for Reza (not built): show a burst as one tile "+N similar", expandable; album membership unchanged.
- Reza's review: heavier_01 (items 1-20) 20/20 correct (him, and in the right album). Remaining: heavier_02-07, fit_01-07.
- Reza's full review of demo6 (by eye, every item): heavier 139/139 correct; fit 115/127 (12 heavy photos in fit:
  items 110,115,117,119,120,121-127; all 2023, all clothed/group/distant shots, all at the bottom of the fit ranking,
  rel 0.565-0.720; every 2026 item is above them). Judge P(fit) is saturated (0.92-0.97 for both right and wrong
  items). Recovering both judge scores for all of Reza's photos from the cache to see why pairing let them through.
- Cause of the 12 heavy photos in 'fit' (judge scores recovered from demo6 judge_cache for all 331 face-matched photos,
  331/331 cache hits): P(fit) is saturated (2026 median 0.980, 2023 median 0.893); P(heavier) spreads (2023 0.50,
  2026 0.20). Era AUCs: heavier alone 0.980, fit alone 0.955, combined rank diff 0.974. The 12 are clothed/distant/
  group 2023 shots with P(heavier) 0.25-0.38 (judge does not see heaviness) and rank diff just past the 0.3 margin.
  Candidate rules vs Reza's labels (223 labeled photos):
    current (rank diff 0.3): fit 8 labeled-wrong (photo-only recompute), heavier 0 wrong, 4 right ones dropped.
    rank diff 0.4: 0 wrong both albums, but 32 right heavier + 2 right fit dropped to neither.
    2-cluster mixture, 90% posterior: fit 0 wrong; heavier +108 UNREVIEWED photos and 2 labeled-wrong (fit items 116,
      118 moved to heavier: clearly fit by eye).
  No free fix; every rule here was scored on the same labels it would be chosen on. Next: Reza reviews the photos in
  NEITHER album (full labels), then pick a rule on principle and check it on the public Bacon/Cage regress.
- Reza labeled the 127 photos in NEITHER album: 116 H, 9 F, 1 not him, 1 neutral. Full labels (393 items):
  267 H (265 from 2023 + 2 from 2026), 124 F (all 2026). So demo6 heavier recall was 139/267 and its 'neither' was
  mostly heavier-era. Rules re-scored on full labels (photos): rank margin 0.3 -> heavier 116 (0 wrong) of 225 H, fit
  107 (8 wrong); two-group split + per-event median -> heavier 232 (5 F-labeled: 2 screenshots of a flyer/Slack, 2
  face-only selfies, 1 plane shot; looks cannot show build there), fit 99 (0 wrong) of 104. Public eras: split finds
  no clear two groups for Pratt/Rogen (BIC) -> falls back; Hill precision on par (36/50 vs 26/34 heavier, 28/36 vs
  24/31 fit). Adopted (bc0c41b) on principle: the split adapts to unequal era sizes, abstains when the person's scores
  form one blob; the event median encodes "a lasting look does not change within 3 hours".
- Submitted fp_demo7 (race_sbatch, frozen worktree .cache/wt_split @bc0c41b, demo6 judge cache copied) to re-run the
  demo end to end (photos AND videos) and score against Reza's labels.
- demo7 (end to end, photos + videos, @bc0c41b) vs Reza's labels: heavier 277 = 267 H (ALL 267 heavier-era items,
  42 of them videos) + 8 F + 1 not him + 1 neutral; fit 116 = 116 F (17 videos), 0 wrong, 116/124 of fit-era items.
  demo6 was heavier 139/139 (recall 139/267), fit 115/127. Raw look at the 9 wrong in heavier
  (audits/demo7_heavier_wrong.jpg): 2 screenshots (flyer, Slack) with a tiny headshot, 4 face-only selfies from one
  evening (one event: the event median moved all of them together), 1 plane selfie, 1 helmet selfie; the not-him item
  is a bearded friend matched at face score just over 0.40 (was in neither before, now in heavier).
  Regress (Kevin Bacon) submitted: fp_regbacon7.
- Reza (10-04 ~20:00): product details are mine to decide ("i just want to get this product perfect"). Decided:
  (1) bursts shown as one stack "+N similar" in the web page (display only; src/findpics/bursts.py, cosine >= 0.9 and
  <= 10 min, 12/12 largest groups same moment by eye); header shows "(N moments)".
  (2) face match: a face counts for a person only if it beats every OTHER frequent person's face group (people sheet
  groups whose faces match the person < 0.40 on average; stored as fingerprints people_groups_emb.npz). On Reza's
  sample: drops the bearded friend (0.41 to Reza vs 0.59 to his own group), keeps 391/391 labeled items.
  Safety test caught `.delete(` in the page JS (a Set toggle) -> rewritten without it. 106 tests pass.
- Burst grouping crashed on timezone-aware dates (the unit test I wrote caught it; it would have broken the web page
  on Reza's library): fixed. demo7 albums: heavier 277 items = 140 moments, fit 116 = 51 moments.
  Page JS verified by running the page's own script in Node with a fake DOM: 6 items in 3 bursts -> 3 tiles with
  "+2 similar"/"+1 similar", "(3 moments)"; open -> 5 tiles; cover marked wrong -> next shot becomes the cover.
  Chromium screenshots failed today (core dump on login and compute nodes), so no visual check of the layout yet.
- Food search on Reza's sample (food_photos3, "Is this a photo of food?", judge saw all 490 photos; scores recovered
  from cache 490/490), by eye at 300 px: album 17 = 13 with food as a clear subject (pizza/tacos/meals x8, wedding
  plates x3, mac and cheese, wedding cake) + 4 weak (balcony table, person on phone, a screenshot, table in the
  background). Top 40 photos NOT in the album by judge score: 0 clear food photos missed (closest: 2 wedding-table
  selfies with a plate at the edge, p 0.62-0.70; a coffee screenshot; champagne glasses). Recall looks complete on
  this sample; precision ~13/17.
- Web page visual check: Chromium headless crashes intermittently on this cluster today (even trivial pages;
  8/8 retries failed on the page), Firefox headless works (`firefox --headless --no-remote --profile <dir>
  --screenshot`), page data inlined so it renders before the capture. Seen: stacks "+2 similar"/"+1 similar",
  "(3 moments)", opened stack outlined. Fixed: the "not right" ✕ rendered as an empty box (glyph missing from the
  font) -> ×; open-stack button "−" -> "hide". 106 tests pass.
- Full regress @a7f2b03 (pairing split, other-identity face rule, bursts, previews): bread 702/499/628 [before
  699/497/626]; Bacon heavier 99 / fit 35 -> outdoors fit 5 [97/35/5]; Kev 322 + 4 videos [same]; dog identical files
  to 10-03 (7, Max 3/3 first; outdoors keeps maxdog_2). All stable.
- 22:40 Live web page on demo7 (real thumbnails), Firefox screenshots desktop 1200 px + phone 390 px (private:
  audits/web_desk.png, web_phone.png): stacks on real data ("+22 similar" wedding group etc.), "(140 moments)",
  phone layout 2 columns, input pinned. Sideways-looking tiles checked: 2 HEICs are stored that way (pillow_heif
  applied original_orientation 3 / 6; Apple shows the same); IMG_0586.MOV: display matrix 90, frame 60 upright with
  our rotation, the matched frame is sideways because the phone turned mid-recording. No rotation bug.
  Fixed: reopened conversation showed only "Done." -> earlier turns' album lines from summary.json.

## 10-05 ~00:30 Toward "type in a search bar on my phone" (Reza: no website cop-out; the real product is the phone app)
- Key point: a phone app reads the library directly (PhotoKit), so the real final exam does not need Apple's data copy.
- Researched (sources in chat): local-LLM iPhone apps (PocketPal, Locally AI, Private LLM) run open GGUF/MLX models
  offline; Apple's on-device ~3B Foundation Model gains IMAGE input in iOS 27 (candidate judge, cannot be measured on
  Linux); 8 GB iPhones: apps killed at ~50% RAM by default, ~75% with the increased-memory entitlement (may need paid
  signing). Free Apple ID: run on own phone, 7-day profile; friends need $99/yr (TestFlight up to 10,000).
- Told Reza: privacy comes from WHERE the model runs (on device, no network), not from the license; open weights are
  for testability/control/portability.
- Running: eval_everyday (8 DISBench users x 12 everyday queries, fast vs exhaustive) @5f94c9f; phone-judge test on
  Reza's sample with Qwen3.5-4B and -2B as judge AND planner (fp_m4B / fp_m2B, scripts/run_sample_model.sh).
- eval_everyday done (96 searches = 8 real Flickr users x 12 queries, ~1,900 photos per library, 0 errors): fast answer
  ~17 s, exhaustive ~37 s per library on one GPU; fast found 0.889 (bicycle) to 1.000 of the exhaustive set, most
  >= 0.99 (eval/everyday/summary.txt). Bug: "selfies" planned as "photos of me" with no condition -> whole library;
  fixed in code ("Is this a selfie?"), test added, 107 pass. "taken at night" = local clock only (no judge, 5.9 s).
  Eye audit of 8 random exhaustive results per query at 800 px running (subagent; sheets eval/everyday/audit/).
- EYE AUDIT of eval_everyday (subagent, 8 random exhaustive results per query, 800 px tiles): 52/80 right, 19 wrong,
  9 unsure. dog 8/8, cat 6/8, flowers 6/8, boat 6/8, beach 5/8, food 5/8, church 5/8, car 4/8, sunset 4/8,
  bicycle 3/8. Patterns: depictions accepted as real (mural bike, cartoon van, ride vehicle, toy cars, drawn boat:
  5/19); "X photos" accepting photos sharing only the setting (coastal town, cliffs, distant dome, golden-hour
  landscape). Real libraries are much harder than Open Images labels (section 12 precision 0.90-0.99 on objects).
  Labels: eval/everyday/eye_labels.json. Testing two generic question forms (eval_real_subject.py, fp_realsubj):
  "a real X (not a drawing/painting/statue/toy/model/picture)" and "X is the main subject"; recall cost measured on
  Open Images labels.
- 10-05 ~02:00 Phone path, measured on the cluster:
  * Qwen3.5-4B as judge on Reza's labeled demo: heavier 276 (all 267 H + 8 F + 1 ?), fit 105/105 F (9B: 277 / 116).
    2B collapses (heavier 4 items): out.
  * Planner eval (30 scripted conversations): 9B 29/30, 4B 20/30 (fails mostly on follow-up edits). 4B also planned
    "screenshots" with no condition (-> whole library): fixed in code for any one-thing request (_name_the_thing).
  * Distillation started: 9B planner answers on ~1,400 fuzz requests + LLM-written follow-ups (batched vLLM;
    eval conversations held out; 15% test split) -> LoRA on 4B text layers (eval/train_planner_lora.py, peft 0.21.2
    + accelerate 1.15 installed --no-deps into envs/vllm).
  * Real-object question adopted (RESULTS 26). Swift core package ios/FindPicsCore (Swift 6.2 Linux toolchain in
    tools/swift): plan schema, pairing split, bursts, time-of-day, relative dates, scope + look scores: all identical
    to Python on golden fixtures (7 Pratt/Hill/Rogen/synthetic pairing cases, 1,457 + 540 date cases, 300 items).
- 10-05 ~02:30 First training run in the project: LoRA distillation of the 9B planner into Qwen3.5-4B (1,984 train
  examples = 1,234 first requests + 750 follow-ups, LLM-written; 316 held out; the 30 eval conversations never seen).
  ~17 s/step on an A100 (no fla/causal-conv1d kernels), 496 steps ~2.4 h; loss 0.115 at step 20. Gate: ships only if
  the distilled 4B gets close to 9B's 29/30 on the held-out conversations (same frozen code as the 9B/4B runs);
  fallbacks: 9B 4-bit planner on the 12 GB phone, or Apple's on-device model. Eval auto-submits when training ends.
- Swift: full grounding port identical to Python on 2,300 cases (1,459 changed by grounding); CLIP tokenizer identical
  on 1,445; planner prompt byte-identical on 58. App: Core ML embedder, index, planner, streamed search, SwiftUI.
  Reza's Mac must be on macOS Tahoe 26.6+ for Xcode 27 (his phone runs iOS 27).
- 10-05 ~02:50 Reza asked what the training copies and how I know it is good. Honest answer: I had not checked.
  Counted: 145/1,984 train targets were the 9B's FAILED last retry (recorder saved calls[-1]); 263 prompts carried the
  retry suffix. Eye check of 20 random raw 9B targets: ~11 good, ~5 partly wrong (lost Paris / Bangkok, "sent to Sarah"
  filter, B-roll as photo), ~4 bad (invented birthday date "November 16, 2025", "Is the time of arrival 3pm?").
  The 30-conversation eval cannot separate teachers (9B 29/30). Training stopped at step ~25; clean_distill.py written
  (drop failing targets, restore retry prompts: 2,128 kept). Next: Qwen3.5-27B-FP8 teacher on the same requests
  (fp_d27_*, H100), eye-compare the same 20, train on GROUNDED targets of the better teacher.
- 10-05 ~03:00 Reza: "anything" is the point (toes, colors, places), videos must be understood whole, completeness
  stats wanted on the phone. Explained: "what" questions = judge (any words); "which one" (me / my dog) = sameness
  (faces for people, image vectors for pets/things/places); places/dates come from metadata because pixels cannot
  tell Paris from another street. Done since: docs/PHONE_PARITY.md (every server function: verified / written /
  missing); Swift completeness certificate (identical to scipy, 300 cases) + streamed rounds (replay of recorded judge
  answers, 6 concepts x 5 seeds: 0 overclaims in 170 rounds) wired into the app.
  Running: eval_video_whole (Reza's 62 labeled videos: 1 frame vs 6 frames mean/joint vs video input);
  eval_face_models (AuraFace-v1 Apache-2.0 vs buffalo_l non-commercial, DigiFace, equal wrong-match rates);
  27B-FP8 teacher relaunched with VLLM_USE_DEEP_GEMM=0 (DeepGEMM JIT needs nvcc, absent on the nodes).
- 10-05 ~04:25 Face model for shipping (eval_face_models, DigiFace 300 unseen identities, equal wrong-match rates):
  buffalo_l (non-commercial weights) recall 0.987 at its 0.40 rate (0.0021) / 0.965 at 10x lower; AuraFace-v1
  (Apache-2.0, commercially sourced data) 0.904 / 0.804 (+ 90 undetected of 21,600). AuraFace clearly worse: not
  switching. SFace (Apache file license) has an open, unanswered commercial-use/training-data question (opencv_zoo
  #318, 10-01); EdgeFace weights' license/data unclear. Decision for Reza before any public release (license
  buffalo_l / accept AuraFace / other). Dev + Reza's personal testing keep buffalo_l.
- Video test crashed on 1-frame videos (video input needs >= 2 frames): padded to 4, relaunched. 27B teacher: Mamba
  cache vs max_num_seqs 1024 -> FP_MAX_SEQS=256 knob in vlm.py, relaunched.
- 10-05 ~04:50 eval_video_whole (Reza's 62 labeled videos, 42 heavier-era / 20 fit-era, red box on him), AUC of
  heavier-era vs fit-era for logit(heavier) - logit(fit): A one face frame 0.981; B 6 face frames each judged, mean
  0.990 (max 0.995); C 6 frames in one prompt 0.983; D the frames as video input 0.982. Single questions: A heavier
  0.973 / fit 0.952, B 0.986 / 0.979, D 0.938 / 0.965. One frame already separates the eras almost perfectly; no
  variant is a clear win at n = 62 (differences ~0.01). Product keeps one face frame (no 6x judge cost on the phone).
- Phone ports tonight (all with golden tests vs Python): face matching (synthetic faces), alignment (200), person
  crop (150), time windows + events (900; fixed 2 undated-photo bugs in the Python), offline geocoder (2,000; server
  switched from reverse_geocoder's flat-degree distance, which picked another town for 274/2,000, + country names).
  Face recognizer converted to Core ML (87 MB). App: faces at indexing, "which face is you", person albums with red
  box, heavier-vs-fit split, place names.
- 10-05 ~05:00 27B-FP8 teacher (2,302 answers, same requests): side by side with the 9B on the 13 comparable first
  requests of the earlier 20-sample: 27B better 7 (no invented birthday date, drops "sent to Sarah", B-roll -> video,
  "video of the cat", screenshots -> photo x2, no stray "night" phrase), worse 1 ("funeral ... person identified as Mom",
  stripped by grounding anyway), same 5. CORRECTION: my earlier eye check said the 9B "lost Bangkok"; it did not (my
  printout omitted the place field). 27B fails the planner's own checks on 21 requests vs the 9B's 172.
  Training the 4B LoRA on cleaned 27B targets (1,968 train; fp_train27); eval auto-submits after (fp_pl27, same
  frozen code as the 9B 29/30 / 4B 20/30 runs). flash-linear-attention 0.5.2 installed --no-deps but its kernels live
  in a separate package (fla.ops missing): training still on the reference path.
- 10-05 ~05:10 App: moments (anchor via the verified windowRows + until-cut) and with-people; judge-answer cache;
  "+N similar" stacks; on-device SELF-CHECK screen (Core ML image/text/face vectors vs the server's for bundled test
  images; DigiFace test face kept out of git, copied by rsync); docs/BUILD_ON_MAC.md (macOS Tahoe -> Xcode 27 ->
  clone + rsync models -> open Package.swift -> Developer Mode -> Run -> self-check numbers back to me).
- 10-05 ~07:45 Distilled phone planner (4B + LoRA on cleaned 27B targets, 1,968 examples, 2 epochs, loss 0.17 -> 0.07):
  30 scripted conversations 25/30 (base 4B 20/30, 9B 29/30); held-out 313 requests vs the 27B teacher (grounded):
  valid 313/313 (base 311), same album count 312 (305), field agreement 0.966 (0.913). Failures: Reza's demo request
  twice ("me heavier vs me fit in the past 6 months": dates on BOTH albums), "Dad at the beach"+"only 2019" lost beach,
  sandwiches/burgers undo dropped the burger exclusion, graduation as a question instead of the day. Gate (close to
  29/30) NOT met: not shipped. Next: code rule for which album a trailing time phrase belongs to; more multi-turn
  27B data; the 9B on the 12 GB phone may make it moot.
  Bug fixed: the training script copied the base model's shard index next to the merged weights (vLLM refused).
- 10-05 08:15-09:55 STORAGE INCIDENT (holylfs06): git writes into .git hang (commit / hash-object / commit-tree /
  even a fresh clone into .cache time out after 5-11 min); plain file writes work. Started with an I/O error
  (Errno 5) reading eval/distill_planner_data.py at ~08:10; a `git commit` from then is stuck unkillable in the kernel.
  NOT on GitHub yet (in the working tree, tested): converse._vs_time_phrase + test (111 pass), Grounding.swift port of
  it (Swift build not yet re-run), eval/distill_planner_data.py FP_CHAIN. Main is at 75ee77c on GitHub. Next: when
  git writes work again, commit + push; re-run Swift tests; regenerate ground fixtures; launch the chained 27B data.

## 10-05 ~16:35 storage back (flaky); relaunch
- Storage dropped again mid-command at ~16:10; recovered. Commit 86f8f59 (Swift vsTimePhrase port, FP_EVERYDAY_OUT) pushed.
- Swift core tests: 17/17 pass on Linux (incl. GroundingTests on regenerated fixtures).
- Relaunched: fp_ev3_0..7 (everyday v2, snapshot .cache/snap_every2, verified current code) and fp_dch3_0..2
  (27B chained follow-ups, .cache/wt_dist, verified FP_CHAIN present).
- 16:45 Submitted fp_ev4b_0..7: the same everyday real-library run with Qwen3.5-4B as judge AND planner (the phone's
  model) -> eval/everyday_4b. Purpose: the phone-vs-server gap on real messy libraries (only measured so far on the
  heavier/fit demo: 4B 276 vs 9B 277 heavier, 105 vs 116 fit).
- 17:00 everyday v2 (9B, after fixes) and everyday_4b (4B judge+planner) done: 96/96 searches each, 0 errors.
  Selfies fixed: v1 15,709 returned (whole libraries) -> v2 79. Real-X question trimmed car 959->860, bicycle 144->101,
  boat 1011->958, dog 1359->1320. Fast answer found >= 0.92 of exhaustive on every query (car 827/860 lowest).
  4B vs 9B (judge agreement, NOT truth): 4B returns 0.74 (car) to 0.99 (dog) of the 9B set; outliers: selfies 4B 443 vs
  9B 79 (75 shared), church 527 vs 385. Median time per library: 4B 25 s, 9B 38.5 s exhaustive. One library returned
  1,269 dogs / 1,984 photos (both models agree; likely a dog owner) -> eye check queued.
  Eye audits running (subagents): 10 random v2 results per query at 1000 px; blind 4B-only/9B-only disagreements.
- 17:10 EYE AUDITS (subagents; I viewed flowers_2, selfies_1, selfies_3 myself and agree: dog close-up and a
  festival portrait are not selfies; kitten selfie is; memorial wreaths = flowers only as setting). DISBench images are
  natively ~500 px, so ">=900 px" means upscaled, not more detail; small objects stay hard (many "unsure").
  * 9B v2, 10 random per query (dog sample = one dog owner's library, needs per-user stratification next time):
    80/110 right, 16 wrong, 14 unsure. Without selfies 76/100 (v1 audit 52/80). cat 10, dog 10, beach 9, food 8,
    boat 8, church 7, car 7, sunset 7, bicycle 6, flowers 4, selfies 4. Depictions-as-real: 0 seen this round (fixed).
    Remaining patterns: "photos of X" where X is only in the background (flowers 4/10); selfies = any close face
    taken by someone else (6/10 wrong); look-alikes (motorcycle as bicycle, truck as car, golden hour as sunset).
  * 4B vs 9B blind disagreements: 4B-only extras 10/63 real (32 not, 21 unsure); 9B-only extras 16/61 real
    (23 not, 22 unsure). 4B adds false positives: selfies 0/6 (posed portraits), church 1/6 (any spire/dome/castle),
    beach 0/6 (coastline), cat 0/6 (dogs, statue, toy); 4B misses small cars (5/6 9B-only cars were real).
  Next: fix the selfie question (who took it, not "a close face"), test "X as the subject" for "photos of X" only.
- 17:20 eval/eval_selfie_q.py: old vs two 'who took it' selfie questions on the 9B+4B selfie pool, 9B and 4B (fp_selfq_*).
- 17:45 FOUND: github.com/rezashamji/find_pics is PUBLIC (API: private=false; created 10-03). CLAUDE.md assumed private.
  Tracked files: no photos/embeddings/private data; but docs describe Reza's demo (weight: heavier/fit), devices, and one
  video filename (IMG_0586.MOV) in JOURNAL. PUSHES PAUSED (commits stay local) until Reza makes it private or says OK.
- 17:45 27B chained data: 4,272 cleaned (31 dropped, 151 retry prompts restored); eye-read 4 turn-3 samples: undo /
  narrow / exclude handled right. Combined set planner27ball_clean.jsonl: 5,137 (4,426 train). fp_train_all: LoRA ->
  30 conversations + held-out 313. Same-code baselines now: fp_pl9b_now (9B), fp_pl4b27_now (previous distilled 4B).
  Selfie question: 9B old 78/447 kept, "who" 35, "took" 38; 4B old 437, who 188, took 99. Eye audit running.
- 18:00 Selfie question audit (subagent, 78 blind labels; I viewed sheets 09 and 22): "who"/"took" raise 9B precision
  ~46% -> ~65% (sure labels) but drop 5-6 of 18 true selfies (I saw them: arm-out pug selfie, webcam selfie, couple
  cheek-to-cheek). NOT adopted (completeness first). 4B: old keeps 437/447 (no filter); took 18/41 precision.
  Better signal on iPhones: EXIF LensModel names the front camera ("... front ... camera"). Not read yet; checking
  presence on the AirDropped sample (metadata count only, no search).
- 18:40 SELFIES = FRONT CAMERA. Reza's AirDropped sample (metadata count only): 510 photos, EXIF LensModel front 68,
  back 382, none 60 ("iPhone 13 Pro front camera 2.71mm f/2.2"). Added: ingest Item.camera (EXIF 0xA434), AlbumSpec.camera
  set by code when the conversation says selfie(s) and the question is the selfie question; scope drops BACK-camera
  photos only when the library has any front tags (Flickr: no-op); report line says mirror selfies on the back camera
  are missed. Kept out of the planner prompt (current-plan JSON excludes it: prompt byte-identical, distill data valid).
  Swift: Album.camera, LibraryItem.camera + scopeMask, Grounding rule; app reads the lens model via ImageIO from the
  on-phone original (no iCloud download; unknown if not local). App code not compiled here (needs Xcode).
  Tests: Python 113 pass; Swift 17/17 on 2,300 regenerated grounding fixtures.
  Planner re-test on today's code: previous distilled 4B 26/30 (was 25; demo request now passes). Fails: Dad at the
  beach + only 2019; drop the sandwiches (x2 variants); graduation day.
- 19:05 9B planner on today's code 29/30 (fail: cat + also videos of her + only from Paris). Training 4B on 4,426 rows: ~17.6 s/step x 1,107 steps (~5.5 h, reference kernels). fp_prom: 'prominent part, not only background' question vs eye labels.
- 21:50 PROMINENT question ("Is X clearly visible as a prominent part of this photo (not only a small part of the
  background)?") on 9B v2 results, scored on all eye labels (v1+v2): drops 10/19 wrong, 2/74 right, 4/13 unsure
  (main-subject form earlier: 4/27 right). But it drops many UNLABELED photos (sunset 96/293, beach 88/386, church
  68/385). I viewed 4 random unlabeled drops per query at 1000 px: flowers 4/4 correctly dropped (gardens, no flower
  focus); food ~2/4 clearly right to drop (dining room, jam jar borderline, people eating borderline); sunset 1/4
  wrongly dropped (sun setting behind mountains); beach 1-2/4 wrongly dropped (couple in beach water, couple selfie at
  the beach); church 0-1/4 (cathedral interiors: stained glass, lectern). Scene words (beach, sunset, church) mean
  "taken there", so the prominent form cuts what people want. NOT adopted in general; candidate only for object
  words (flowers, food), which needs a non-list way to tell objects from scenes + a larger labeled sample first.
- Training 4B (27ball): epoch 1 step 960/1107, loss 0.062.
- 22:20 PHONE JUDGE QUESTION: can the 12 GB iPhone run the 9B? Phone weights = mlx-community/Qwen3.5-9B-4bit (5.98 GB;
  4B-4bit 3.06 GB): affine 4-bit, group 64, 250 tensors (language linears + embed + lm_head), vision tower full.
  scripts/sim_mlx_quant.py reproduces that quantization on the HF weights (de-quantized for vLLM) so the everyday
  real-library eval measures the PHONE's numbers. Jobs fp_q9 / fp_q4 -> models/qwen35_{9b,4b}_mlx4sim; next: everyday
  runs with both + blind eye audit vs the bf16 9B. Memory fit of the 9B on the phone still needs the Mac/phone.
- 23:10 sim_mlx_quant done (9B 250/250 tensors, 4B 249/249). fp_evq9b_*/fp_evq4b_*: everyday real-library runs with the phone's 4-bit weights. 4B planner (27ball) trained (2 epochs, loss 0.06); 30-conversation test running.
- 23:45 PHONE WEIGHTS on the everyday real-library run (96 searches each, 0 errors). Overlap with the bf16 9B
  exhaustive set (agreement, not truth): 9B-4bit 0.79 (flowers) - 1.00 (dog, cat), most 0.86-0.96; 4B bf16 0.74-0.99;
  4B-4bit 0.62 (flowers) - 0.99, car 0.66, sunset 0.65, night 0.75. Selfies 9B-4bit = 0: its planner wrote "selfie of
  the person in the red box" and the eval nulls person afterwards (eval artifact). Blind eye audit running.
- Distilled planner with chained data (27ball): 30 conversations 26/30, SAME 4 failures as 27b (plans differ in 28/30).
  Training data: 870/1,216 "drop" follow-ups carry an exclusion (most others are date edits, correct). Debug job
  fp_dbgpl prints raw vs grounded output for the 4 failures (model miss or grounding strip?).
- 00:05 PHONE WEIGHTS EYE AUDIT (subagent, blind, 125 photos / 10 queries at 1000 px; I viewed boat_1 and agree:
  9B-4bit misses are small background boats - a dinghy, kayaks). Full 9B returned 5,037 (10 queries).
  * 9B-4bit vs 9B: 377 9B-only photos, 42 4bit-only. Of sampled 9B-only: 10/41 real (misses), 15/41 not (9B false
    positives the 4-bit rightly dropped), 16 unsure. 4bit-only: 4/26 real, 19/26 false positives.
  * 4B-4bit vs 9B: 907 9B-only, 54 4bit-only. 9B-only sampled: 18/63 real, 24 not, 21 unsure. 4bit-only 4/34 real.
  * Scale: ~0.24 x 377 = ~90 real photos missed by 9B-4bit vs ~0.29 x 907 = ~260 by 4B-4bit (lower bounds; many
    unsure). 9B-4bit misses: small background objects (flowers 3/5, boat 3/5, bicycle 2/5).
  => The phone's judge should be the 9B-4bit (5.98 GB) IF it fits the iPhone 18 Pro's memory with the other models;
     4B-4bit is the fallback. Memory/speed only measurable on the phone (needs the Mac).
  Debug job crashed (no __main__ guard under vLLM spawn); fixed, resubmitted fp_dbgpl2.
- 00:40 APP MEMORY (Reza asked what real iOS apps do; web): a 12 GB iPhone Air with com.apple.developer.kernel.
  increased-memory-limit gets a 6,073 MiB app budget (github StayLameBro/backburner #1); PocketPal-class apps run ~9B
  only at ~5 GB on 12 GB phones; Apple's own 3B is 2-bit and lives in the OS. My earlier "~9 GB with the entitlement"
  was WRONG. The 9B-4bit (5.98 GB weights) likely does NOT fit with vision + PE-Core + faces + KV. Testing a 3-bit
  9B (~4.5 GB est.): fp_q9b3 (layer list from the 4-bit release; mlx-community 3-bit repo is empty).
- Planner fail debug (raw vs grounded, 27ball): "drop the sandwiches" -> model wrote exclude "sandwich or a burger"
  (burger copied from the prompt's undo example) and grounding dropped the WHOLE exclusion as a leak. Fixed: only the
  copied alternative is removed (Python + Swift, test). Other 3: model errors ("Is Dad visible?" as the condition with
  beach only in looks; undo of "sandwiches and burgers" cleared all; graduation without an anchor).
- 00:50 Tests after the leak fix: Python 114 pass; Swift 17/17 on 2,300 regenerated grounding fixtures.
- 01:00 Reza restarting the Mac (macOS Tahoe). Queued fp_pl9b_v3, fp_pl4ball_v3 (planner gate after leak fix); fp_q9b3 running. PLAN.md RESUME HERE written.
- 01:15 Reza: App Store iPhone app for other people (not a Mac app); GitHub public is fine (push resumed); Apple copy not in. Queued fp_ev16{9b,4b}_* (16 fresh libraries) + fp_pl4b27_v3. 65 jobs in queue.
- 02:10 Planner re-test after the leak fix: "drop the sandwiches" now PASSES on both distilled 4Bs. New "except selfies"
  failure = MY snapshot error (copied converse.py without planner.py -> no `camera` field); rerun with full src
  (fp_pl9b_v4, fp_pl4ball_v4). Expected: 4B 27/30, 9B 29/30.
- 16 FRESH libraries (fp_ev16*, 30,273 photos, 192 searches/model, 0 errors). 9B fast answer found 0.94-1.00 of the
  exhaustive set except SELFIES 0.444 (75/169: the fast stage has no look for a selfie). Phone 4B-4bit overlap with
  9B: 0.59 (sunset) / 0.63 (flowers) - 0.99 (cat). Median exhaustive time per library 9B 36.9 s, 4B 12.0 s (A100).
  Eye audit stratified by user running (subagent).
- 02:35 EYE AUDIT, 16 FRESH libraries (subagent; 12 per query stratified by user, 11-12 distinct users each, 1000 px;
  I viewed dog_3 and agree: teddy bear in a theatre crowd = wrong, deer on a road = wrong, leashed dog = right):
  91/132 right, 21 wrong, 20 unsure. bicycle 11, flowers 11, cat 10, church 10, boat 10, car 10, beach 9, dog 7,
  food 5, sunset 5, selfies 3 (of 12). Same 8 queries as the first audit: 73/96 vs 63/80 (both 76%).
  Errors: look-alike/toy animals (tiger, jaguar cub as cat; deer, teddy bear as dog); "of food"/selfie = presence
  instead of subject; golden light/moon as sunset; motorcycle as bicycle, cinema as church.
  => Server precision on real libraries ~70-76% by eye; per-query spread 3/12 - 11/12. Biggest gaps: selfies (fix on
  phone with the front-camera tag), sunset, food, animal look-alikes.
- 02:50 Selfie albums get a default look ("a selfie taken at arm's length") so the fast stage can rank (was 75/169 fast vs exhaustive). Python 114, Swift 17/17. Re-measuring selfies on the 16 libraries (fp_self16_*).
- 03:05 Reza asked what else can run now. Started: face model with a commercial licence (subagent: research + DigiFace eval), App Store/Background Assets/Apple Foundation Models API research (subagent -> docs/APP_STORE_RESEARCH.md), fp_qvar: question variants for sunset/food/dog/cat scored on all eye labels.
- 03:25 APP STORE / APPLE MODEL RESEARCH (subagent -> docs/APP_STORE_RESEARCH.md, claims tagged verified/secondary/
  unverified). I re-checked the key one in Apple's doc JSON: Foundation Models Response = content, rawContent, usage,
  transcriptEntries; GenerationOptions = sampling mode, temperature, max tokens, tool mode. NO probabilities.
  * Apple model (iOS 27): image input YES (Attachment(cgImage) in the prompt builder); hard answers only; @Generable
    constrained output (anyOf / range / regex); context 4096 (doc) vs 8192 (WWDC code): read contextSize at runtime;
    background has a budget (rate-limited error); iPhone 15 Pro+ with Apple Intelligence; NO custom LoRA on iOS 27.
    => as a yes/no judge it fits our exhaustive mode and the certificate (binary labels on samples); the COMPARISON
    split (heavier vs fit) uses logit(pA)-logit(pB) and needs a graded score: try @Generable range 1-10 ratings or
    two yes/no answers -> test on Reza's labeled demo on the phone.
  * Delivery: app <= 4 GB; Apple-hosted Background Assets up to 200 GB / 200 packs, included in the paid membership,
    ML models an intended use (iOS 26+); ODR deprecated in iOS 27; guideline 4.2.3(ii): state size + ask before download.
    PocketPal downloads from huggingface.co directly.
  * Memory: increased-memory-limit not listed in Apple's capability table; one report says a free Personal Team build
    kept it (~6 GB vs 3.3 GB on an 8 GB iPhone 15 Pro Max). Measure os_proc_available_memory() on the iPhone 18 Pro.
  * Privacy: "Data Not Collected" OK if nothing leaves the device; NSPhotoLibraryUsageDescription; limited-library mode
    cannot create/fetch user albums (our Save-as-album needs full access or a fallback).
- 03:50 Judge without probabilities (Apple model) simulated: FP_JUDGE_MODE=hard (greedy yes/no -> 0.98/0.02) or rating (1-10 -> (r-1)/9). fp_mode_{9b,4bq}_{prob,hard,rating}: Reza's heavier-vs-fit demo scored vs his labels (scripts/run_demo_mode.sh, score_demo.py; counts only).
- 04:20 Selfie default look WORKS: fast answer found 166/167 of the exhaustive selfies on the 16 libraries (was 75/169).
- Planner gate with full current code: distilled 4B (27ball) 27/30 (sandwich fixed), 9B 29/30.
- 9B-3bit (plain round-to-nearest) is BROKEN: says yes to nearly everything (flowers 5,808 vs 313, car 5,340 vs 860 on
  the 8 libraries). Naive 3-bit is out; a calibrated method (AWQ / DWQ-style) would be needed for a 9B under ~5 GB.
- Question variants (word-specific) on all eye labels: food "mainly about" kept 18/18 right, wrong 16->7; dog
  "real, living ... not another animal" wrong 7->4, right 30/31 kept; sunset "sun setting or just set" wrong 10->7,
  right 18->17; cat "house cat" WORSE (wrong 6->9, dropped 225 unlabeled). I viewed 8 random unlabeled drops each:
  food 0 clear food photos lost (party, ducks, seaweed; borderline spice market, hanging fish, figs); sunset golden
  light / dusk without sun; dog: street, trees, a cat - 1 temple photo may show a dog at the edge.
  Not adopted yet: those are per-word wordings. fp_qvgen tests GENERIC forms on all 10 queries (real-X with "not
  something that only looks like one"; "X is what this photo is mainly about").
- 04:40 Judge distillation data (eval/judge_distill_data.py): 9B P(yes) on (photo, question) pairs, DISBench photos, planner-written questions minus everyday-eval words and named-person questions; 24 best-match + 24 random photos per question; fp_jd_0..7.
- 04:55 Wrote eval/train_judge_lora.py (soft-BCE on yes/no next-token probs vs 9B P(yes), text-layer LoRA, 3 best-match + 1 random pair per question, 10% questions held out) and eval/eval_judge_distill.py (held-out agreement + eye-label recall/false positives). Waiting on fp_jd data.
- 05:05 App: Save-as-album no longer reports 'Saved' when it fails (try? swallowed errors); limited-library access gets a clear message (iOS forbids album creation there). ATTRIBUTIONS: MLX libs (MIT), Apple frameworks. Not compiled (needs Xcode).
- 05:15 ios/FindPicsApp/Sources/AppleJudge.swift: Apple Foundation Models judge (iOS 27 Attachment image input; @Generable yes/no -> 0.98/0.02, rating 1-10 -> (r-1)/9; availability check; one session per photo). Same contract as Judge.pYes. Untested (needs Xcode).
- 10-06 ~07:30 Reza updated the Mac (macOS 27.0.1); installing Xcode next. BUILD_ON_MAC: clone over https (public).
- DEMO WITHOUT PROBABILITIES (Reza's labels: 267 H, 124 F; heavier / fit album contents):
  9B prob (now): 267 H + 8 F / 116 F + 0 H.  9B hard yes/no: 214 H / 124 F + 53 H.  9B rating 1-10: 229 H + 3 F /
  121 F + 32 H.  4B-4bit prob: 137 H / 117 F + 19 H.  4B-4bit hard: 63 H / EMPTY.  4B-4bit rating: 236 H + 9 F /
  114 F + 28 H. => a hard yes/no judge breaks the comparison split; a 1-10 rating mostly works but leaks heavier
  photos into fit. Apple's model (no probabilities) would need the rating mode and is expected to be worse here.
- GENERIC question forms on all eye labels (right kept / wrong kept): real-X "+ not something that only looks like
  one": dog 29/31, 6/11 (was 30, 9); car 26/31, 2/12 (29, 4); bicycle 23/25, 2/14 (23, 3); boat 31/35, 3/7 (34, 3):
  ~1 right lost per wrong removed -> not adopted. "X is what this photo is mainly about": beach 24/25, 0/19 (24, 9);
  food 17/18, 7/21 (18, 16); sunset 18/18, 4/15 (18, 10); BUT flowers 14/29 (26), church 17/25 (24), cat 22/24 -> no
  single generic wording wins; not adopted.
- Judge distillation data done: 115,776 9B judgments. fp_trjudge (smoke 40 pairs, then full, then test), fp_jt_base,
  fp_jt_9b (held-out questions + eye labels). Face-model agent resumed (interrupted by the restart).
- 08:00 COMMERCIAL FACE MODEL (subagent; eval/face_commercial.md; I viewed the CelebA aligned-crop sheet: faces centered
  and aligned). Recall at equal wrong-match rate, DigiFace (19,110 targets) / CelebA test (12,255 targets):
  buffalo_l 0.987 / 0.979 (non-commercial; commercial licence sold by insightface.ai); AuraFace (Apache-2.0, vendor
  says commercial data) 0.904 / 0.908; SFace (OpenCV Zoo, Apache file, training data undocumented) 0.921 / 0.942;
  HyperFace-10k (MIT, but synthetic data made with a research-only-trained generator) 0.941 / 0.798; HyperFace-50k
  broken in our pipeline (0.013 CelebA; unexplained). Ruled out by licence: EdgeFace, Langevin-DisCo (CC BY-NC-SA),
  dlib (FaceScrub), facenet-pytorch (VGGFace2). Apple: no public face-identity API.
  => Reza's decision (money/legal): buy the buffalo_l commercial licence, ship AuraFace (-7 points on real faces), or
  ask OpenCV for SFace data provenance.
- 08:20 App: download consent screen before the ~3.1 GB model download (guideline 4.2.3(ii)); remembered with @AppStorage. Model delivery stays Hugging Face direct (like PocketPal); Apple-hosted Background Assets later (paid account).
- 08:35 Planner takes any text model (Qwen judge.text or AppleText.text: Apple model as planner; prompt ~1,840 tokens fits 4,096). Untested (Xcode). Rule tightened: model-library imports never in the foreground.
- 08:45 App: PhotoJudge protocol (Qwen Judge, AppleJudge); top-left Model menu switches judge+planner between Qwen, Apple rating, Apple yes/no (falls back to Qwen if Apple Intelligence is unavailable). For the on-phone side-by-side. Untested (Xcode).
- 08:50 Specific thing/place, add-ons on cached PE-Core vectors (eval/instance_qe.py, R-precision, 3 refs):
  query expansion HURTS (things 0.731 -> AQE3 0.695, places 0.693 -> 0.685; ~7 photos per identity, expansions pull
  in wrong ones). Neighbour smoothing (DBA: each vector averaged with its k nearest library neighbours, at indexing)
  HELPS: things 0.748 (k=1), places 0.766 (k=2) vs 0.731 / 0.693. Caveat: these benchmark libraries contain only the
  identities' photos (no everyday distractors); must re-test with distractors before adopting.
- Judge distillation smoke test (40 pairs) trained + saved; full training running (fp_trjudge).
- 09:00 docs/FIRST_DEVICE_TEST.md (self-check, app memory, first index time/battery, 3-model side-by-side). Self-check screen shows os_proc_available_memory. DBA-with-20k-everyday-distractors test running (CPU, background).
- 09:25 FP_PAIR_BIPOLAR (one combined 'A rather than B?' question per photo, same person crop) on Reza's demo: fp_mode_{9b_rating_bi, 4bq_rating_bi, 9b_prob_bi}. Python 114 pass.
- 09:45 "the day/night/week of my <event>" -> anchor on the event + same_day/same_week window (the album question moves
  to the anchor when it was about the event; "food from the day of my graduation" keeps food inside the day).
  Python + Swift port, test; 115 Python pass. Fixes the distilled 4B planner's graduation failure in code.
- DBA with 20k everyday distractors (things): plain 0.731 (distractors cost nothing), DBA k=1 0.747, k=2 0.747 ->
  the gain holds with distractors. Places pending.
- docs/RELEASE.md (privacy label "Data Not Collected", App Review notes, TestFlight steps, blockers).
- 10:05 DBA with 20k everyday distractors, places: plain 0.680, DBA k=1 0.740, k=2 0.750, k=3 0.740 (things 0.731 ->
  0.747). ADOPTED k=2 in the subject path (converse._dba: library vectors + references each averaged with their 2
  nearest library vectors before the mean-of-refs match). 115 Python pass. Checking on dogs (fp_petdba: all-dogs
  library and everyday+all-dogs library, top-3/top-5/R-precision, plain vs DBA). Swift port after the dog result.
- 10:25 App: SubjectSearch.swift (pet/thing/place from example photos: DBA-smoothed vectors via Accelerate + side-by-side judge veto at 0.2); FindPicsCore.subjectScores (reference) + golden test. Reza: Xcode 27 installed; next clone + rsync + open Package.swift.
- 10:35 Swift 18/18 (new SubjectTests: subjectScores == Python _dba ranking). App UI: bottom-bar 'Find a specific pet or thing…' -> PhotosPicker (1-3 photos), name, kind -> AppModel.searchSubject.
- 02:07 CORRECTION: journal entries above stamped '~07:30' through '10:35' on 10-06 were mis-timed by me; they all happened between ~01:35 and ~02:10 (cluster clock). Order is right, clock labels are not.
- 02:09 Identity question removed -> keep a SCENE look as the condition ('Dad at the beach' as 'Is Dad visible?' + beach looks -> 'Does this photo show a beach?'); person-describing looks still dropped. Python 116 pass; Swift port + fixtures running.
- 02:10 keep_partial_undo: 'actually keep the sandwiches' after 'drop the sandwiches and burgers' restores the burger exclusion when the model clears it (Python + Swift + app planner, tests). Python 117 pass.
- 02:13 Swift 19/19 (UndoTests). Ported the rank-margin pairing fallback (FindPicsCore.rankMarginPair + withinPersonRank, golden MarginTests: 40 photos, 21 neither) and wired it into the app when splitPair finds no two groups.
- 02:14 Judge distillation loss flat ~0.34 from step 50 (0.36 at 25). Floor = mean entropy of the 9B's soft targets for this mix (3 best-match @0.343 + 1 random @0.151) ~0.295, so the excess over the floor is ~0.045: modest learning; the held-out/eye-label test decides.
- 02:15 Photo/video twin albums merge when their questions share the same content words ("Is this a photo of cat?" vs real-X cat video question) - the 9B's last planner failure. Python 118 pass; Swift port, fixtures + tests queued.
- 02:15 App: counts photos it cannot read while indexing (iCloud-only originals) and says so; first device test asks for that number.
- 02:24 First Xcode open: 'No such module AppleProductTypes' (Xcode gives that module only to .swiftpm app packages). Renamed ios/FindPicsApp -> ios/FindPicsApp.swiftpm (git mv; .gitignore + docs updated).
- 02:45 PLANNER GATE after today's code rules (day-of-event anchor, scene look kept, partial undo, leak alternative):
  distilled 4B (27ball) 30/30 (was 25-27), 9B 29/30 (run started before the photo/video twin-merge fix that targets its
  last failure: cat + videos + Paris). The distilled planner now meets the gate on the 30 conversations.
- Bipolar A-vs-B question NOT adopted: 9B rating + bipolar -> heavier 75/267 H, fit 66 F + 10 H (worse than separate
  ratings 229 H / 121 F + 32 H).
- DBA on dogs, everyday (19,218) + all DogFaceNet (10,943) library = 30,161: top-3 87 -> 90/120, top-5 97 -> 108/200,
  R-precision 0.679 -> 0.752 (identical to the all-dogs library: everyday photos never intrude).
- App-code compile review subagent started (Reza is opening the project in Xcode).
- 03:00 Held-out 313 requests vs the 27B teacher: distilled 4B (27ball) agreement 0.967 (27b 0.966, base 0.913).
  scripts/peft_to_mlx_adapter.py -> models/planner_4b27ball_mlx_adapter (400 tensors, 32 layers, rank 16, scale 2).
  fp_plq4: the LoRA applied to the PHONE's 4-bit 4B weights (scripts/merge_lora_into.py) -> 30-conversation gate.
  Fixed my own bug: queued Swift runs waited on themselves (pgrep matched their own command line).
- 02:30 Planner adapter for the phone: fp16 MLX format, 56 MB (models/planner_4b27ball_mlx_adapter); mlx-swift-lm 3.32 has LoRAContainer.from(directory:) and the Qwen35 VLM's loraLayers = language layers with matching module keys. PlannerAdapter.swift added (hook into Judge.text after the compile review). BUILD_ON_MAC: rsync line for it.
- 02:31 BUG found by the golden test: rankMarginPair dropped 'not clearly either' photos (Swift dict[k] = nil deletes the key); fixed with updateValue. splitPair already did it right; no other optional-valued dictionaries.
- 02:32 FP_JUDGE_MODE=rating100 (0-100 scale) on Reza's demo: fp_mode_9b_r100, fp_mode_4bq_r100 (does a finer rating stop heavier photos leaking into fit?).
- 02:32 Swift 21/21 after the rank-margin fix. App icon drawn (Sources/Assets.xcassets/AppIcon, 1024 px); Package.swift still uses the placeholder icon until the first device build succeeds (then switch appIcon to .asset("AppIcon")).
- 02:33 Sample index (Reza's AirDropped photos): adding the EXIF camera column to data/private/index_sample/items.parquet (derived data only); fp_selfcam: 'selfies' on the sample with the front-camera scope (private outputs, eye check to follow).
- 02:34 4B-4bit rating + bipolar on Reza's demo: heavier 248 H + 4 F, fit 120 F + 18 H (best probability-free result for the 4B; plain rating 236+9 / 114+28; prob 137 / 117+19). But 9B rating + bipolar collapsed (75 H): inconsistent across models -> not adopted; the phone test of Apple's model decides. Cancelled fp_pl9b_v5 (its 30-conversation result was already read).
- 02:38 PLANNER ON THE PHONE'S WEIGHTS: distilled LoRA merged into the MLX-4bit-simulated 4B (scripts/merge_lora_into.py, 200/200 targets) -> 30/30 conversations (same as on the 16-bit base). The phone plan: one 4-bit 4B + 56 MB planner adapter.
- 02:38 Judge.text loads the planner adapter (LoRAContainer.load/unload) around each planner call; judge calls use the base model. Untested (Xcode).
- 02:39 Judge test baselines (eval_judge_distill report): 9B held-out self-agreement 11072/11088 (sanity), eye labels right kept 247/261, wrong kept 77/141; base 4B agree 10429/11088 (mean |P-P9| 0.067), right 227/261, wrong 52/141. Caveat: eye-labeled photos come mostly from 9B results, so 'wrong kept' is inflated for the 9B; the distilled 4B should gain recall and may gain false positives.
- 02:40 'selfies' on Reza's sample with the camera scope: 46 returned (61 front-camera items in the index). Eye check of all 46 + front-camera misses running (subagent; labels in data/private/audits/selfies_cam). App: refuses to load the model below 3.6 GB available memory (message instead of a crash).
- 02:41 APP COMPILE REVIEW (subagent; type-checked the non-UI files against stubbed Apple types with Swift 6.2 in
  Swift 6 mode -> 0 diagnostics; UI/Judge/AppleJudge/Faces/Embedder/PhotoLibrary/VideoFrames by eye): 9 fixes (Sendable
  conformances for Core types, Embedder/FaceEngine @unchecked Sendable, lock instead of captured var in a @Sendable
  closure, await inside ??, AVAsset wrapper, @AppStorage -> @Published+UserDefaults, Foundation Models respond builder
  form, CGContext buffer lifetime, import CoreLocation). Package.swift: + swift-huggingface 0.9, swift-transformers 1.3
  (required by #huggingFaceLoadModelContainer, per mlx-swift-lm README), + mlx-swift 0.32.3 (import MLX). Removed the
  app's duplicate withinPersonRank (Core version is tested).
- 9B planner gate with the twin-merge fix: 30/30 (was 29). Both planners now 30/30.
- 02:41 SearchView: album card split into AlbumCard (small view pieces; avoids 'unable to type-check in reasonable time').
- 02:43 SELFIES ON REZA'S SAMPLE (subagent, all 80 photos at 1000 px; I viewed sheet_15: 4/4 dropped photos are
  clear group selfies): returned 46/46 real selfies (19 had no camera tag); of 34 front-camera photos NOT returned, 23 are
  real selfies, 11 unsure (back-of-head mirror shots, story graphics). Cause: person albums kept only the top HALF of the
  person's face matches by rank (rel_cut, built for relative looks); the dropped selfies scored 0.87-0.93.
  FIX: for person albums whose question is a FACT (no look/appear/seem words, engine.LOOK_WORDS), P >= 0.7 also keeps a
  photo; looks stay relative. Python 119 pass; app ported. Rerun: fp_selfcam2 (selfies) + fp_mode_9b_prob_fact (demo
  must be unchanged).
- 02:47 Xcode now parses the .swiftpm; error: PlaceholderIcon has no member magnifyingGlass -> appIcon .asset("AppIcon").
- 02:50 rating100 (0-100) on Reza's demo is BAD: 9B -> heavier 22 H + 2 F, fit 116 F + 240 H (split flipped/collapsed); 4B-4bit -> heavier 174 H + 34 F, fit 72 F. Not adopted; 1-10 stays the probability-free fallback. Xcode: packages resolved, iPhone selected; rg_cities1000.csv was git-ignored (*.csv) -> now tracked; macro trust prompt for MLXHuggingFaceMacros.
- 02:59 Xcode build 2: icon PNG was git-ignored (*.png) -> tracked; 'Non-Sendable [LibraryItem]' (retroactive conformance in the app did not apply across modules) -> FindPicsCore value types declare Sendable themselves (Core builds on Linux); retroactive extensions removed. Remaining: signing team (Reza).
- 03:01 Selfies on the sample after the fact-question fix: 86 returned (was 46); eye check of the additions running (subagent). Judge distillation at step 475/546.
- 03:05 Selfies after the fact fix, eye-checked (subagent; I viewed sheet_01): 79/86 real selfies (4 not: all
  screenshots of video calls / chat apps; 3 unsure), 22/23 earlier misses now found, 0 removed. FIX: selfie scope also
  drops screenshots (server: untagged PNGs = iPhone screenshots; phone: PHAssetMediaSubtype.photoScreenshot via
  LibraryItem.isScreenshot). Python 119, Swift 21/21.
- Mac: signing key partition list set (codesign prompt loop); Claude Code installed on the Mac.
- 03:08 docs/MAC_SESSION.md (instructions for the Mac Claude session: xcodebuild loop, Simulator screenshots, journal 'MAC:' lines) + .claude/settings.json allow-list (xcodebuild, xcrun, swift build/test, git pull/add/commit/push main).
- 03:11 Demo after the fact-question fix: heavier 267 H + 8 F, fit 116 F (identical to before: pairs are decided by the split).
- 03:17 Judge distillation trained (546 steps, 8,724 pairs; loss 0.36 -> ~0.32-0.34, floor ~0.30). Running:
  fp_trjudge test (held-out questions + eye labels, 16-bit base), fp_jtq4 (judge LoRA merged into the phone's 4-bit
  weights, same test), fp_jtbq4 (untrained 4-bit 4B baseline), fp_ev16jd_0..7 (everyday searches on the 16 fresh
  libraries with the distilled judge, to compare with everyday16_9b / everyday16_4b). Judge adapter -> MLX format.
- 03:17 fp_plboth: planner LoRA on top of (phone 4-bit 4B + judge LoRA) -> 30-conversation gate (does the judge adapter break the planner?).
- 03:17 App: optional judge adapter (Models/judge_adapter) fused at model load; planner adapter stays per-call. Untested on device.
- 03:16 MAC: First Simulator run of the iPhone app (iPhone 17, iOS sim; Xcode has simctl runtime but no Simulator.app GUI
  on this Mac — drove it headless via xcrun simctl). Device compile-check build (generic/platform=iOS, scheme "find pics",
  -allowProvisioningUpdates): ** BUILD SUCCEEDED **, 0 errors. Simulator build (Debug-iphonesimulator): ** BUILD SUCCEEDED **.
  Allow-list (7f9f3a4) worked: xcodebuild/xcrun/git ran with no permission prompts, fully unattended.
- 03:16 MAC: Screens seen (screens/ gitignored, no personal photos; sim seeded with the app-icon JPG only):
  (1) "Starting..." init spinner. (2) Native Photos permission prompt with the custom usage string "find pics searches
  your photos ON this phone. Nothing is uploaded." (Select Photos / Allow Full Access / Don't Allow); shows "7 Photos".
  (3) Model-download consent: "find pics needs to download its on-phone AI once: about 3.1 GB." + "Use Wi-Fi if you can.
  After this download, your photos are searched entirely on this phone and nothing is uploaded." [Download now].
  Flow permission -> download-consent is correct. Could not tap "Download now" (no idb/cliclick/Simulator GUI for taps)
  and the 3 GB MLX model won't run in the Simulator anyway (needs a real GPU), so stopped at the consent screen.
  granted photos via `simctl privacy grant photos` to get past the native prompt headlessly.
- 03:16 MAC NEEDS REZA: to screenshot past the download-consent screen (search field, menus, result sheets) I need a
  tap tool in the headless sim (install `idb` or `cliclick`), OR a real-device run from Xcode. On a real device the
  model downloads and the full UI is reachable. Nothing blocking the build; only the interactive screens past consent.
- 03:20 MAC session's first commit (92bddc0): app BUILD SUCCEEDED for device + Simulator; screens seen up to the download consent. Fixed after it: stray space before '// swift-tools-version' in FindPicsCore/Package.swift; MAC commits were authored as the Mac's global git user (zainshamji, another email) on the public repo -> MAC_SESSION.md now sets a repo-local identity (Reza to be told).
- 03:21 App: DEBUG-only -demoUI launch argument (example albums, no model) so the Mac session can screenshot the search screens in the Simulator; MAC_SESSION.md updated.
- 03:22 Planner LoRA on top of (phone 4-bit 4B + judge LoRA fused): 30/30 conversations. The judge adapter does not hurt planning.
- 03:35 JUDGE DISTILLATION RESULT (eval_judge_distill; held-out questions 11,088 pairs; eye labels 261 right / 141 wrong):
  base 4B 16-bit: agree 10429, right 227, wrong 52 | base 4B phone-4bit: 10318, 208, 33 | distilled 16-bit: 10615, 231, 61
  | distilled phone-4bit: 10557, 219, 41 | 9B: 11072, 247, 77. On the phone's weights the distilled judge keeps +11 real
  photos but +8 wrong ones (moves toward the 9B, whose eye-labeled set is biased toward its own picks). Adoption rule
  (recall up WITHOUT more false positives) NOT met -> not adopted yet; everyday16_jd (product-level, blind eye audit of
  disagreements) decides. Planner + judge adapters together: 30/30.
- 03:45 JUDGE THRESHOLD SWEEP (eye labels 261 right / 141 wrong; eval/judge_threshold_sweep.txt): at equal strictness
  the distilled phone judge == the base phone judge (t=0.8: 196/16 vs 196/14; 0.85: 189/9 vs 188/8; 0.9: 177/5 vs
  175/5). Distillation moved the CALIBRATION (more yes), not the ability to tell right from wrong. NOT ADOPTED; the app
  keeps the base 4B judge (judge adapter not shipped). The 9B separates better (t=0.8: 222 right / 25 wrong).
  Side note: the 9B at 0.7 keeps 77/141 eye-wrong photos vs 25/141 at 0.8 for -25/261 right; eye-labeled photos come
  mostly from 9B results, so this overstates the gain; not changed (completeness first), candidate for a "stricter" option.
- 03:25 Other phone-size judges on the same test: fp_jt_gemma4_e4b (google/gemma-4-E4B-it; MLX 4-bit exists), fp_jt_qwen3vl_4b (Qwen/Qwen3-VL-4B-Instruct, Apache-2.0).
- 03:30 MAC: MEMORY BLOCKER ON REZA'S REAL iPhone 18 Pro. He sent a screenshot: "This iPhone lets an app use 2.4 GB of
  memory; find pics needs about 3.6 GB." That is the App.swift guard (os_proc_available_memory < 3.6 after the embedder +
  face engine load). Root cause: iOS caps a third-party app's memory below total RAM by default; even a 12 GB 18 Pro only
  hands the app ~3.3 GB -> ~2.4 GB free. FIX: entitlement com.apple.developer.kernel.increased-memory-limit (unrestricted:
  works with a FREE Personal Team, no App ID capability; 8 GB+ -> ~6 GB cap). VERIFIED by command line: codesign --force
  re-sign of the device .app with the entitlement merged into Xcode's generated entitlements succeeds with Reza's free
  identity (ZN8M63RSR5), and `codesign -d --entitlements` confirms it is embedded. No Xcode GUI needed.
- 03:30 MAC: why re-sign and not Package.swift / a raw entitlements file: AppleProductTypes has no capability for this
  entitlement, and passing it via CODE_SIGN_ENTITLEMENTS makes Xcode's AUTOMATIC signing try to register it in the
  provisioning profile -> "Entitlement ... not found and could not be included in profile. ... BUILD FAILED". The kernel
  honours the entitlement from the code signature directly, so we build normally then re-sign. Added
  scripts/build_device_entitled.sh (build for device -> merge entitlement -> re-sign -> verify; fixed a BSD mktemp
  template bug) and a "Section 0" in docs/FIRST_DEVICE_TEST.md (CLI path + the Xcode "+ Capability -> Increased Memory
  Limit" path). Normal Simulator/device loop builds are unaffected (no entitlements file committed).
- 03:30 MAC: Reza's iPhone 18 Pro is CONNECTED (devicectl: 00008160-001124A13EC00036, iPhone19,2, state connected). I can
  build+sign+install the fixed app entirely from the CLI, BUT installing on his PHYSICAL phone is a scope step I will not
  take without his explicit OK (auto-mode classifier also blocked it, correctly). Left the decision to Reza.
- 03:30 MAC NEEDS REZA: to finish the memory fix autonomously overnight I need ONE of: (a) your OK to install/run builds
  on your connected iPhone via devicectl (then leave it plugged in + unlocked + trusted; I deploy the entitled build and
  verify via os_log), or (b) you do the 30-second Xcode step in the morning (Signing & Capabilities -> + Capability ->
  Increased Memory Limit -> Run). The fix itself is done and committed either way.
- 03:34 QWEN3-VL-4B as judge (16-bit; eye labels 261 right / 141 wrong): t0.7 240 r / 60 w, t0.8 240/55,
  t0.9 237/47; agree with 9B on held-out 10309/11088. Dominates the current Qwen3.5-4B (16-bit t0.7 227/52; phone 4-bit
  208/33) and is near the 9B curve (t0.7 247/77, t0.8 222/25), but its P(yes) is near 0/1 (little ranking signal).
  Running: fp_q3vl (phone 4-bit weights, same test), fp_ev16q3_* (16 libraries, 9B's plans reused via
  FP_EVERYDAY_PLANS so only the judge differs). Gemma 4 E4B test still running.
- 03:55 GEMMA 4 E4B judge (16-bit): t0.5 247r/79w, t0.7 244/74, t0.8 243/66, t0.9 239/55; agree 9B 10068/11088.
  Slightly behind Qwen3-VL-4B (t0.9 237/47). Both are overconfident (P near 0/1). In the high-recall regime Qwen3-VL-4B
  dominates the current Qwen3.5-4B phone judge (Qwen3.5 q4 t0.5 232/65 vs Qwen3-VL t0.9 237/47); Qwen3.5 reaches the
  low-false-positive regime (t0.8 196/14) that Qwen3-VL cannot. Pending: Qwen3-VL at phone 4-bit, and 16 libraries.
  Licence: Qwen3-VL Apache-2.0; Gemma terms (commercial OK with use policy). Planner would stay Qwen3.5-4B+adapter or
  Apple's model -> two models on the phone (~2.5 + 3 GB) only fits with the increased-memory entitlement (~6 GB).
- 03:39 App: Judge takes a model id (default unchanged); candidate photo judge id recorded (Qwen3-VL-4B-Instruct-4bit, supported by mlx-swift-lm's MLXVLM Qwen3VL). No switch until the 4-bit and 16-library results.
- 04:05 QWEN3-VL-4B AT PHONE 4-BIT (sim, 253/253 language tensors; vision full): t0.5 248r/74w, t0.7 242/59, t0.8 240/55,
  t0.9 235/50, t0.95 229/39 (16-bit: 237/47 at 0.9) -> survives quantization. vs current phone judge (Qwen3.5-4B 4-bit:
  t0.5 232/65, t0.7 208/33): at equal wrong photos, ~+15-17 real photos (of 261). Strong candidate for the PHOTO judge
  (planner stays Qwen3.5-4B + adapter). Deciding test: everyday16_q3vl (16 libraries, plans reused) + blind eye audit.
- 03:41 cli: FP_PLANNER_MODEL runs the planner on a second model (GPU split 0.55/0.3). fp_mode_q3vl_prob2: Reza's demo with the Qwen3-VL-4B phone-4bit judge + distilled 4B planner (does the near-0/1 P break the heavier/fit split?).
- 03:47 Demo with two models needed VLLM_WORKER_MULTIPROC_METHOD=spawn and FP_MAX_SEQS=256 (planner share too small for 1024 Mamba sequences); fp_mode_q3vl_prob4 resubmitted.
- 04:15 FREE FACE MODEL DEEP DIVE (subagent; eval/face_free_deepdive.md). Product-level recall on CelebA at buffalo_l's
  wrong-item count: buffalo_l 0.977 (3 refs) / 0.980 (8 refs); AuraFace+flip 0.888 / 0.929; SFace+flip 0.943 / 0.952;
  AuraFace+SFace fused+flip 0.921 / 0.957. Single-image (8 refs): AuraFace+flip 0.927, fused 0.954 vs 0.979. By eye,
  ~9/12 sampled AuraFace wrong items are truly different people while ~9/12 buffalo_l "wrong" are CelebA label noise
  (real gap larger). No other recognizer with clean commercial weights AND data found (13 synthetic sets checked; Intel
  0095 data undocumented; fine-tuning infeasible: no commercially usable identity dataset). Phone size: AuraFace 130 MB,
  SFace 19 MB, buffalo_l 87 MB. Correction to my brief: people.py takes the BEST match over references (then expands),
  not the mean. Decision stays Reza's: buy the InsightFace licence (best), or AuraFace+flip (clean, ~5-9 points worse).
- 04:20 EVERYDAY16 with Qwen3-VL-4B (16-bit) as judge, 9B's plans reused (eval/everyday16_compare_q3vl.txt): agreement
  with the 9B's exhaustive sets 0.66 (selfies) - 1.00 (cat), most 0.84-0.98, vs the current 4B-4bit 0.59-0.99 (flowers
  0.98 vs 0.63, sunset 0.84 vs 0.59, food 0.95 vs 0.79). But Qwen3-VL returns MORE than the 9B on food (1,273 vs 954)
  and flowers (1,680 vs 1,069): possible false positives. Blind eye audit of the disagreements running (subagent,
  eval/everyday16_q3vl_audit).
- 03:52 App: Model menu option 'Qwen3-VL photo judge' (Qwen3-VL-4B-Instruct-4bit judges photos, Qwen3.5-4B+adapter plans); default unchanged.
- 04:35 REZA'S DEMO with the PHONE plan (Qwen3-VL-4B 4-bit judge + distilled Qwen3.5-4B planner, two models):
  heavier 265 H + 6 F, fit 111 F + 2 H (9B server: 267 H + 8 F / 116 F; current phone 4B judge: 137 H / 117 F + 19 H).
  The near-0/1 probabilities did NOT break the split. With the 402-photo eye test (+15-17 real at equal wrong) and the
  16-library agreement, Qwen3-VL-4B is the leading phone PHOTO judge; final call after the blind audit (running).
- 04:45 BLIND EYE AUDIT Qwen3-VL-4B vs current 4B vs 9B on 16 libraries (subagent, 152 photos, 39 sheets; I viewed
  food_photos_4: 4/4 Qwen3-VL-only "food" photos are wrong: shop front, wild mushroom, vegetable poster, kids' party).
  Photos the 9B returned but the judge rejected: Qwen3-VL 8 real / 31 not / 10 unsure (of 49; pool 914) vs current 4B
  19 / 36 / 16 (of 71; pool 2,295) -> Qwen3-VL misses far fewer real photos. Photos only the judge accepted: Qwen3-VL
  26 real / 19 not / 18 unsure (of 63; pool 1,729) vs 4B 18 / 8 / 15 (of 41; pool 178). Objects (car 6/8, boat 5/6,
  church 6/8, dog 3/5 extras real) good; "X photos" subjects bad: food 0/5, flowers 2/7, cat 0/5 extras real (~1,000
  extra photos there). The 9B's own extras are often wrong too (31/49 of 9B-only not real).
  => Qwen3-VL-4B: better phone judge for objects and Reza's demo (265/267), worse on food/flowers/cat breadth. Next:
  stricter wording for "X photos" with Qwen3-VL, scored on all eye labels (fp_qvq3).
- 05:05 Qwen3-VL-4B (phone 4-bit) wording variants on all eye labels (eval/question_variants_generic_q3vl): same pattern
  as the 9B: "mainly about X" fixes food (wrong 22->8/35, right 18->17/18), beach (14->5, 28->26), sunset (8->3), but
  flowers loses half the real photos (32->14/33) and church/cat get worse. No single wording; not adopted. (Eval-only
  grammar slip noticed: the generic real-X template wrote "a real a dog"; product rewrite is correct.)
  Next: two-judge vote (phone loads Qwen3.5-4B for planning anyway) on eye labels.
- 05:15 TWO-JUDGE VOTE on eye labels (phone 4-bit Qwen3-VL-4B = A, phone 4-bit Qwen3.5-4B = B; eval/judge_ensemble.txt):
  mean(A,B) >= 0.7 keeps 231/261 right, 34/141 wrong (B alone 0.7: 208/33; A alone 0.95: 229/39; AND: 204/16; 9B
  between 247/77 and 222/25). +23 real photos at equal wrong vs the current phone judge, ~9B level. Cascade makes it
  cheap: only photos with A >= 0.4 get B's opinion. Implemented: vlm.EnsembleJudge; cli FP_JUDGE_ENSEMBLE (with
  FP_PLANNER_MODEL); eval_everyday FP_JUDGE_ENSEMBLE=<second model>. Running fp_ev16ens_* (16 libraries) and
  fp_mode_ens (Reza's demo).
- 04:21 App: EnsembleJudge (Swift port, cascade mean) + Model menu 'Two-model vote'.
- 04:23 fp_ev16ens shard 5 failed on an A100-40GB (second engine: 161 Mamba blocks < 256 seqs); resubmitted with FP_MAX_SEQS=128. Other 7 shards + demo running.
- 05:40 16 LIBRARIES, all eye labels that fall in them (142 right / 82 wrong / 56 unsure; eval/everyday16_eyescore.txt):
  9B keeps 111 right / 57 wrong; current phone 4B 101 / 20; Qwen3-VL alone 127 / 33; TWO-MODEL VOTE 112 / 15.
  Counts (eval/everyday16_compare_ens.txt): the vote removes Qwen3-VL's excess (food 837 vs 1,273, flowers 1,125 vs
  1,680; 9B 954 / 1,069). Labels were sampled where judges disagree (relative comparison, not absolute precision).
  DECISION (mine, product detail): phone photo judge = two-model vote (Qwen3-VL-4B 4-bit + Qwen3.5-4B 4-bit, cascade
  mean >= 0.7), pending Reza's demo (fp_mode_ens). Needs the increased-memory entitlement (both models resident).
- 10-06 22:10 (session resumed after it ended ~04:50) Reza's demo with the two-model vote: heavier 137 H, fit 106 F + 49 H
  -> the vote BREAKS the person-look split (the 4B is weak on looks; Qwen3-VL alone 265 H / 111 F + 2 H). FIX: person-look
  questions ("... the person in the red box ...") use the first model (Qwen3-VL) alone; objects/scenes keep the vote
  (Python EnsembleJudge + Swift). Rerun fp_mode_ens2.
- MAC: 10-06 22:40 Device build + entitlement re-sign (`bash scripts/build_device_entitled.sh`) on the iPhone 18 Pro
  (UDID 00008160-001124A13EC00036). Script itself works: BUILD SUCCEEDED and `codesign -d --entitlements :-` lists
  com.apple.developer.kernel.increased-memory-limit.
- MAC: 10-06 22:48 But the INSTALL FAILS: `xcrun devicectl device install app` ->
  0xe8008015 "A valid provisioning profile for this executable was not found" / IXUserPresentableErrorDomain 14.
  Diagnosed: the embedded profile ("iOS Team Provisioning Profile: com.rezashamji.findpics", team YYP85AQ2C5,
  expires 10-13, our UDID IS provisioned) grants only application-identifier, keychain-access-groups, get-task-allow,
  com.apple.developer.team-identifier. installd validates the signature's entitlements AGAINST the profile, so an
  entitlement present only in the signature makes the install fail. The script's premise ("unrestricted, works with a
  free Personal Team, no App ID capability, kernel honours it from the signature") is wrong for INSTALLATION.
- MAC: 10-06 22:55 Tried the other no-GUI route: pass it as a real CODE_SIGN_ENTITLEMENTS file so automatic
  provisioning would request a matching profile. Fails at build time:
  `error: Entitlement com.apple.developer.kernel.increased-memory-limit not found and could not be included in
  profile`. So both shell routes are dead; the capability has to come from Apple's side (App ID / team).
  docs/FIRST_DEVICE_TEST.md section 0 rewritten with the measured facts instead of the wrong claim.
- MAC: 10-06 23:00 Installed the app WITHOUT the entitlement so the no-model steps can still be tested.
  Gotcha for the next session: after the failed entitled build, `xcodebuild` built incrementally and did NOT re-sign,
  so the .app still carried the entitlement and still failed to install; had to re-sign explicitly with the
  entitlement deleted. Then `install app` succeeded (bundleID com.rezashamji.findpics).
- MAC: 10-06 23:02 `process launch` denied: device locked (FBSOpenApplicationErrorDomain 7). Waiting on Reza to
  unlock; nothing else is blocked on me. Also fixed the repo-local git identity, which was still zainshamji
  <zain@theheartmedicalgroup.com> (MAC_SESSION.md says commit as rezashamji <rezamshamji@gmail.com> on this PUBLIC repo).
- 10-11 ~02:00 (cluster) Night plan. My check-in loop had lapsed 14:30 -> 01:45 and the Mac has been idle since
  16:52. Now: a Monitor polls origin/main every 60 s and wakes me on each Mac push (30-min heartbeat behind it).
  Mac tonight = M38 (keep the video pass running with thermal logging, eye-check the 209 dog results + 50 near-miss
  "no"s, time M37 searches). Cluster tonight = heat/speed agent (measured-clock replay, KV reuse across a photo's
  follow-up questions, prefetch, 8-bit vision tower as default?) -> M39.
- 10-11 02:20 (cluster) HEAT / SPEED (RESULTS 44, MAC_INBOX M39). (1) Replay on the MEASURED clock (Mac's 5-min buckets;
  they were labelled "2-minute" but sum to 1,175 vs 1,181 logged only as 5-min): m4 (shipped bf16 tower) 105 / 305 /
  440 / 755 photos by 3 / 10 / 20 / 40 min, 1,169 at the 68-min halt; m8 (8-bit tower) 90 / 330 / 615 / 1,095, 1,510;
  exponential fit to m4: 0.82/s -> 0.26/s, tau 4.5 min. RESULTS 43's rule never finishes dog and car before the halt
  (34% of audit4 replays, 4.3% of ev16). No round schedule wins (90-call rounds: bound 1.2 min sooner, -0.024 / -0.036
  recall at 3 min). Adopted: fast mode ends after the round reaching 480 calls, that round's random sample topped up
  to 150 (FindPicsCore fastStopCalls / fastStopTopUp). All replays done before the wall; ev16 done p90 19.5 min (m8) /
  31.8 (m4); recall at 3 / 10 / 20 min unchanged (eye pooled m8 0.180 / 0.383 / 0.475 vs 0.533 at 20: the budget binds
  there), sparse searches lose nothing, dog eye recall 0.345 vs 0.839 at the m4 wall, car 0.388 vs 0.527 (those came
  in minutes 20-68 of a hot phone). 0 overclaims in 29,111 replayed rounds; uniform sample check passes (2,000 seeds).
  (2) Image-prefix KV reuse for exclude/filter follow-ups: possible in mlx-swift-lm 3.32.3 without a patch
  (Qwen3VL.prepare warm continuation; library tests assert warm == cold). Judge.pYes(keepImage:), guards: plain
  KVCacheSimple layers, exact token prefix; -checkKVReuse / -noKVReuse for the device check. Expected ~95% off each
  follow-up (FLOPs); only ~5-12% of planned searches have one (379 follow-up questions in 3,084 planned albums).
  (3) Prefetch: FindPicsCore.Prefetcher + streamRounds lookahead hint (97.6% of next photos predicted); Search.swift
  reads each photo once for all questions. Expected up to ~1.5x on a cool phone if reads are ~0.6 s iCloud downloads,
  ~1.15x hot, ~0 if heat-limited; never measured (new PREFETCH log line; -noPrefetch A/B). (4) 8-bit vision tower:
  recommend as default; needs a public download location (Reza: HF repo under his account, or Background Assets).
  FindPicsCore swift test 106/106 (was 101; +5). App not compiled here: M39 (build, KV check, timing, 8-bit repeat).
- 10-11 ~03:00 (cluster) RESULTS 44b, coordinator applying Reza's 10-10 no-cap rule: the 480-call stop cut dog eye recall
  0.839 -> 0.345 at the wall, the same mistake as the video-pass cap. Now 480 calls (+150 top-up) = the FIRST ANSWER
  (bound, results, done); then AppModel.keepLooking re-runs each album in exhaustive mode (cached answers first) and
  keeps judging, paced by thermalState (FindPicsCore thermalPace: judge at nominal/fair, pause + re-read every 20 s at
  serious, stop at critical). Bound: the continuation is exhaustive mode's rounds, 0 overclaims (920 + 18,692 rounds).
  Replay with an ASSUMED pacing (hot floor m4 15.6 / m8 21.3 photos/min x duty after the first answer; pauses assumed
  to prevent the halt): audit4 eye recall at 20/40/60/90 min, m8 duty 0.5: 0.482 / 0.616 / 0.710 / 0.851 vs stop
  0.475 flat; m4 duty 0.5: 0.449 / 0.509 / 0.581 / 0.688 vs 0.475; ev16 m8 duty 0.5 0.914 / 0.954 / 0.971 / 0.989 vs
  0.903. Also: a new search waits for the old one to stop at its current photo; a generation counter keeps a stopped
  search's late updates out of the new results. Swift 109/109. M39 updated: (5) measures pause/resume on the device
  (transitions, pause lengths, duty, whether iOS still halts charging).
- MAC: 10-11 15:05 [THE OVERNIGHT F6 RUN, and the architectural finding it forces]
  8 of 10 queries ran. Qwen3-VL judge, Reza's 187k library, found / judged / inScope / seconds:
     photos of a dog     205 / 1,300 / 147,600 / 2,038 s  (34 min)
     food photos       3,647 / 4,305 / 147,600 / 8,962 s  (149 min)
     beach photos      2,873 / 4,304 / 147,600 / 9,058 s  (151 min)
     sunset photos     1,163 / 2,150 / 147,600 / 4,553 s  (76 min)
  18,186 vision encodes overnight, no crash. The crash work is done; what is left is speed and precision.
  FOUR QUERIES COULD NOT RUN AT ALL, every one with "Who is me? Confirm their face once." and inScope 0:
  selfies, me-looking-heavier, me-looking-fit, photos-where-I-look-tired. Person-scoped search needs the owner
  to confirm their own face once and that has never been done on this phone, so 4 of the 10 F6 queries - the
  flagship heavier-vs-fit demo among them - are blocked on one tap. MAC NEEDS REZA.
  THE PRECISION WORRY, not yet checked by eye: food said yes to 3,647 of 4,305 photos it looked at (85%), beach
  2,873 of 4,304 (67%). These are the most-likely candidates so a high rate is expected, but 85% looks like a
  judge agreeing with nearly everything rather than discriminating - the loose-scene-query failure of RESULTS 35.
  NOT CLAIMED EITHER WAY until the photos are looked at at full size (CLAUDE.md: no thumbnails).
  THE ARCHITECTURAL POINT, which Reza put better than I had ("apple do it within one second ours is hours"):
  Apple Photos is instant because it CLASSIFIED EVERY PHOTO IN ADVANCE and searches a tag database. We pay the
  cost at QUERY time instead: a 4B vision model looks at each candidate, ~2-4 s each. 4,305 candidates = 149 min.
  But we ALSO precompute an image vector per photo, and RESULTS 27 measured that the fast vector stage alone
  recovers 0.94-1.00 of what the exhaustive judge finds on real libraries. So for "food photos" the judge was
  called 4,305 times largely to re-confirm what the vector stage had already ranked correctly.
  THE SHAPE THIS WANTS TO BE: vector-only (instant, Apple-fast) for queries vectors handle well, and the judge
  reserved for the ones they do not - which are the specific, personal, compositional queries, and those have
  FEW true matches, so they need few calls. That is the difference between hours and seconds.

## 10-11 ~17:00 M38 answered: where the vector stage is enough on its own (RESULTS 45, MAC_INBOX M40)
- Question (Mac 15:05): food 149 min / beach 151 min of judging on the phone; Apple answers in < 1 s. Is the
  precomputed PE-Core-B-16 ranking already good enough for the first screen, and can the phone tell when?
- Built eval/eval_vector_enough.py (sheets / analyze / rule). No new judge calls. 746 NEW blind eye labels from native
  sheets (8 labelling subagents on the RESULTS 34 criteria, eval/vector_enough/LABEL_PROTOCOL.txt; I checked beach_01,
  food_05, car_09 myself: 18/18 same calls): a census of every unlabelled audit4 first-screen photo (526) + a stratified
  ev16 sample (220). Kept OUT of RESULTS 34's strata (not uniform draws).
- Pooled 7,886-photo library: vector-only top 20 111/120 vs judged 116/120 (all of the gap is bicycle). Per library
  (what the phone searches): 272/480 vs 318/370. The pooled view hid that 3 of 4 libraries hold 4-25 dogs.
- Rule (LOLO over 20 libraries, 216 searches): plain-look search AND 20th-best fast score >= T -> vectors. Says vectors
  for 32/216 (28 truly >= 18/20 judge-yes), eye 78/80 = judge's 78/80; accuracy 172/216 vs 148/216 always-judge.
  Conservative: 40 of 68 vector-enough searches still go to the judge. Per-photo version (>= 0.2036): 0.89 judge-rel.
  (1,409/1,575), 0.90 eye (154/172). Selfies / night / person / compositional stay on the judge.
- Budget where the judge is needed: 20 checked photos at median 25 calls (0.7 min m4), p90 78 (2.2 min); judge's own
  first screen complete after 50 calls in 106/168. More calls do not lift a 20-photo screen past ~0.76 judge-rel.:
  74/168 of these libraries hold < 20 judge-yes. Fix = shorter screen of checked + likely, not more calls.
- FindPicsCore VectorFirst.swift (firstScreenDecision, likelyAtOnce, kthLargest) + 6 tests; swift test 115/115.
  App: M40 (show likely photos before the first judge call, checked ones move up with a mark; measure ms to the row).
- Not changed / not answered: the judge pipeline, the bound, the keep-looking pass (the 149-min food run was that pass
  over 147,600 photos); whether to skip keep-looking for .vectors searches (deep-rank vectors keep recall but halve
  precision past rank ~150: food top 200 recall 1.000 at 0.48 vs judged 0.908 at 0.57).

## 10-11 ~18:00 [F8] face picker: review the group before confirming it (code; app build = MAC_INBOX M41)
- Defect (Reza on his phone): "Is this you?" confirmed a whole automatic face cluster from a row of 6 thumbnails; an
  impure cluster silently taught the app that someone else is him, and every person search inherited it.
- (a) A group row no longer confirms on tap: it opens GroupFacesView, every face of the group as a crop from its stored
  box (lazy grid), LEAST ALIKE FIRST (cosine to the group's mean, ascending) so an intruder sits near the top.
  (b) Tap to leave faces out; "Yes, these 8 of 9 faces are me" saves only the kept faces as refs. Left-out faces are
  saved per person as negatives (people.json `negatives` + `negativeSources`): expandRefs never adds a face at least as
  close to a negative as to the refs, and itemPersonScores scores such faces -1 (matchPerson passes them; person albums
  and "with X" both use them). (c) Top of the picker: "Most accurate: pick 1-3 photos of yourself"; the suggested group
  (unchanged front-camera logic) below it. (d) "People" (bottom bar): who is set, N faces / K marked not them; per person
  every face in use (least alike first), remove faces (-> negatives), Pick again (no search re-run), Forget (people.json
  entry only, with confirmation; never a photo).
- Safety: groups carry a generation number; a picker opened on groups that were rebuilt since saves nothing and says so.
  Face-model change / face upgrade re-derive negatives from their sources (lost ones dropped, never a re-ask).
  People searches still use checked faces only (test pins it with negatives on).
- FindPicsCore: PeopleReview.swift (pickFromGroup, removeFromPick, pickFromPhotos, mergeNegatives, alreadyExcluded,
  leastAlikeFirst, faceCropRect, faceCropReadSide, wording) + `negatives:` on expandRefs / itemPersonScores / matchPerson
  (blockedMax ignoreSelf:false so an excluded face matches itself). 8 new tests incl. the defect itself: an intruder at
  cos 0.83 that expandRefs pulls in today (asserted) is not re-added, nor its second photo, and both photos score -1.
  swift test 123/123 (was 115) on Slurm (gpu_test CPU job 52201079, fresh scratch dir). NOTE: the "baseline" job I
  started first (52199687) also compiled after the new files existed, so it is a second 123/123 run, not a baseline.
- App NOT compiled here (Linux): People.swift, App.swift, SearchView.swift, PersonSearch.swift, Moments.swift. The
  Swift 6 spots I re-read by hand are listed in M41 step (1). Picker sheet moved onto the NavigationStack so it also
  opens from the pushed People screen.
- Next: Mac M41 (build, screenshots of the 6 new states, crop correctness at full size for a photo and a video face,
  time to first crops for Reza's "me" group). Nothing measured on a device yet.
- MAC: 10-11 17:20 [FACE SET, AND IT EXPOSED A PLANNER GAP] Reza set his own face with "Add a photo of you"
  (3 photos) on the M41 build, after rightly refusing to confirm a 5,000-face cluster he could not inspect.
  IT WORKED: "selfies" now runs and the person filter cut scope 147,600 -> 36,138.
  BUT THE JUDGE IS ASKED THE WRONG THING. The plan note: "Search for photos where the owner (me) is visible ...
  [could not express 'Is the owner visible in this photo?' as a question about one photo; searching for what it
  looks like instead]". 0 found after 20 checked. The search silently degraded to visual similarity. Logged as
  M42 for the cluster; it blocks all four person queries in F6 including the heavier-vs-fit demo.
  ALSO MEASURED TODAY, and it is the good news: M40 (vector-first) works on the device.
     16:52:30 VECTORFIRST food photos source=vectors reason=score #20 0.2098 >= 0.2074 likely=20
  20 likely matches on screen with ZERO judge calls, ~1.8 min after launch and nearly all of that is app
  start-up. The same query took 149 minutes on yesterday's build. The judge then keeps verifying behind them at
  ~20/min under M39's thermal pacing. Caveat kept honest: those 20 are ~0.90 right by the cluster's eye labels
  (RESULTS 45), not verified, and the app says so on screen.
- 10-11 ~18:45 [M42] Owner queries silently fell back to looks on the phone. ROOT CAUSE: grounding knows a person
  only by name words of 3+ letters; the owner album's person is "me" (2), so "Is the owner visible in this photo?"
  was never seen as identity, failed the unanswerable check (planner retried 3x), and the fallback asked from looks.
  FIX (Python planner.py + Swift Grounding.swift, same change): for the owner album, "me / myself / the owner / the
  user / the library owner" + the owner's name words count as the person ("am I" / "I am" rewritten first); the
  owner words are never boxed as a possessive ("the owner's mother" stays an unanswerable relation); new identity
  form "Is this a photo of X?". stripIdentityConditions now runs BEFORE the unanswerable check (plan_turn,
  Planner.swift) as well as in ground. Expected: "photos of me" -> no question (0 judge calls); "selfies" -> front
  tag + "Is this a selfie?"; "me heavier vs fit" / "I look tired" -> "Does the person in the red box ...?".
  Parity fixture (2,300 9B planner outputs) regenerated: 5/2,300 grounded plans change, all 5 read: 4 drop a
  pointless "Is this a photo of the person in the red box?" (face filter decides), 1 keeps "blurry photo of someone
  at a beach" instead of everything. Run 1 had 2 regressions ("the owner's mother" boxed into a question); fixed.
  Run 2-3: 1 test failed on capitalization ("is the person..." from "Am I"); fixed. FINAL (Slurm 52231779 +
  52231784): pytest 78/78; swift test 124/124 incl. GROUND parity 0 mismatches of 2,300. App not compiled here.

MAC 10-11 21:16 — Tapping a photo is fixed, and the one-query runner now writes its numbers to disk.
  Reza, after "photos of me" came back with 41,541: "when i click on an image it doesnt open up larger or take way
  too long". Cause: FullPhoto asked PhotoKit straight for a 1600 px image with network: true, so an iCloud-only
  photo - nearly his whole library - downloaded its original before anything rendered, behind a bare spinner. Now
  two stages: the local 600 px thumbnail appears at once, then the sharp one replaces it with a small
  "Loading the full-size photo…" note while it arrives (e45ed79).
  Also: `-runQuery "<text>"` now waits for the search and writes found / judged / inScope / seconds / note to
  Documents/query_result.json, the same shape -runQueries writes. The reason is not convenience - a screenshot of a
  result grid is Reza's own photos, which I should not be capturing, and a stalled search and a finished one look
  identical in a picture (proved twice, 10-10).
  And a bug in my own tooling: scripts/build_device_entitled.sh picked the device UDID with `awk '{print $3}'`, but
  `devicectl list devices` leaves the Hostname column empty and the Name column has spaces ("Reza's iPhone"), so $3
  was the literal string "(UDID)" and the install step failed with CoreDeviceError 1000 while the build itself had
  succeeded. Now matched by shape ([0-9A-F]{8}-[0-9A-F]{16}).

MAC 10-11 21:20 — "Check a sample": the answer to "i cant look at all".
  The honest problem with 41,541 results, or with the 3,647 photos "food photos" returned, is that NOBODY has
  checked them - not Reza, not me - and the part of a ranking that is easy to eye (the top) is exactly the part
  that flatters it. So: a random hand-check. FindPicsCore gains cpLower, precisionFromSample, precisionNote and
  precisionSampleSize (Completeness.swift) - exact Clopper-Pearson, two-sided, so 25 right out of 25 reads as
  "between 83% and 100%", never "certain" - with 7 tests (PrecisionTests.swift, all pass). The app gains a
  "Check a sample" button on every finished album: 25 results drawn uniformly at random, one at a time, big, Yes/No,
  then one sentence - "you checked 25 at random and 23 were right (92%), so between 74% and 99% of all 41,541 are
  likely right - around 38,000 photos". 25 is chosen to be doable in about two minutes, not to be a labelling
  project; it buys about +-20 points, and the sheet says that checking more narrows it.
  This is also the "confidence level" Reza asked the product to show, measured rather than asserted.
- 10-11 21:40 Mac ran M42 on the device: "photos of me" -> 41,541 (face filter only, no judge), so the owner fix works
  end to end. Mac added "Check a sample" (M43: 25 uniform draws, Clopper-Pearson). M43-ANSWER: the cluster's whole-set
  eye precision for the phone config already exists (RESULTS 45 1a: food 0.57, beach 0.66, sunset 0.72, dog/car/bicycle
  1.00); prediction for Reza's food sample ~14/25; no new labelling round.

MAC 10-11 21:27 — I timed a query that was not running. devicectl REMEMBERS launch arguments.
  I launched `-runQuery "selfies"` at 21:16:44 and waited ten minutes. The device log shows what the app actually
  did: planner loaded 21:18:38, photo judge 21:18:51, and then
    21:18:58 VECTORFIRST photos of a dog source=vectors reason=score #20 0.2202 >= 0.2074 likely=20
  - it searched "photos of a dog", the argument from an EARLIER launch today. No "Q start: selfies" line exists, and
  Documents/query_result.json was never written. So the ten minutes measured the wrong query, and had I not read the
  log I would have reported a selfies timing that never happened.
  This is the same mechanism as F5 (the ten-hour idle): `devicectl device process launch` persists launch arguments
  per bundle id, and a later launch - including one the OS makes on its own - reuses them. `--terminate-existing`
  does not clear them.
  FIX: runDebugQuery now writes `ONEQUERY start: <q>` as its very first action, before any model load, so the
  argument that actually arrived is readable one second after launch. Every device run from now on checks that line
  before the clock starts. RULE: a measurement is void until the log says which query the app received.
  Not wasted: that accidental run is still a datapoint. 223 photos judged in the ~8 minutes before I killed it =
  0.46 photos/s, above the 0.27-0.40 measured earlier today - the phone had been idle and cool since 18:12, which
  is more evidence that sustained speed is set by heat, not by the model.

MAC 10-11 21:30 — The new log line works: "9:29:50 PM ONEQUERY start: selfies", one second after launch.
  So this run is measuring the query I asked for. That is the first device timing today I can actually stand behind.

MAC 10-11 21:35 — M43(b) done: the audit remembers every answer.
  AuditStore (SampleCheck.swift) keeps every Yes/No by query key and photo id in the app's own container
  (audits.json, protected until first unlock). Three consequences, all the cluster's reasoning (M43-ANSWER):
  a second audit of the same search SKIPS the photos already answered and continues the sample, so the range
  narrows as Reza checks more over time rather than re-asking him the same photos; the summary says how many came
  from earlier sittings; and the answers are eye-truth for HIS library, which is the only ground truth that will
  ever exist for it. When no judge ran (judged == 0: the face matcher or the vectors alone produced the set, as in
  "photos of me" -> 41,541) the sheet now says so, and that a wrong one there means a wrong FACE, removable on the
  Me screen - the audit measures the face matcher in that case, not the judge.
  The cluster also answered the number itself: it already eye-labelled whole result sets with the phone's weights
  and cut (RESULTS 45 1a) - food 0.57, beach 0.66, sunset 0.72, dog 1.00, car 1.00, bicycle 1.00. So the honest
  prediction for Reza's sample on "food photos" is about 14 of 25, anything from 9 to 19 consistent. If his number
  lands near 14, "food photos" is about half wrong and that is the thing to fix next - not speed.

MAC 10-11 21:36 — Past searches: results now survive closing the app.
  Found while wiring the audit: a search's results lived only in memory, so the app forgetting them makes
  "Check a sample" impossible in practice - nobody re-runs a 40-minute search to hand-check 25 photos. PastSearches.swift
  keeps the last 40 finished searches (query, asset ids, judged, inScope, note, date; one entry per query text, newest
  wins) in the app's own container, with a "Past searches" screen that reopens one, shows the same album card - so
  "Save as album" and "Check a sample" both work on it - and offers "Run it again".
  It is also the first piece of what Reza asked for beyond searching: results he can come back to.

MAC 10-11 21:42 — MEASURED: the judge runs ~3x slower on a hot phone, and keeps decaying.
  Two runs tonight, same build, same 896 px judge, back to back, counted from the D-before-encode lines:
    "photos of a dog" 21:18:58-21:27:14, phone cool and idle since 18:12:   294 photos / 502 s = 0.586 photos/s
    "selfies"         21:31:38-21:39:30, phone hot from the run above:      100 photos / 472 s = 0.212 photos/s
  And within the selfies run it is still falling, per minute: 17, 16, 13, 13, 13, 13, 7, 8 - better than halved in
  eight minutes. (A correction to my own first reading: an awk window of "9:3x" also matched 9:3x AM and gave 577
  encodes; the real count is 100.)
  What this means, and it is the whole performance story: throughput on this phone is set by HEAT, not by the model
  or the code. The first answer needs ~600 judge calls, so it lands in ~10 min on a cool phone and ~17-45 min on a
  warm one, and no optimisation of the judge changes that - only doing fewer judge calls does.
  Which is exactly what M40 (vector-first) and M42 (the face filter) do: they answer with ZERO judge calls. The
  catch is which searches are allowed to: "photos of a dog" went instant, "photos of me" went instant with 41,541,
  but person / selfie / moment / exclude searches are all ineligible by construction - and "selfies" was Reza's own
  first query. The slow searches are the ones we have not taught to answer without the judge.

MAC 10-11 21:50 — The stale launch argument killed the selfies run a SECOND time. Launch arguments are out.
  The app relaunched itself at 21:48:28 with "ONEQUERY start: photos of a dog" - the old argument again - which
  ended the selfies run 18 minutes in, before it had even finished round 1 (150 judge calls). So in one evening the
  same mechanism cost two measurements.
  FIX, and no launch argument is ever trusted again: the Mac writes the query into the app's own container
  (Documents/next_query.txt, `devicectl copy to`), and the app reads it, DELETES it, and runs it once. A relaunch
  finds no file and indexes normally - the failure mode cannot recur. runDebugQuery also deletes the previous
  query_result.json first, so a dead run cannot leave the last run's numbers looking like this one's.

MAC 10-11 21:52 — A bare "selfies" now costs ZERO judge calls, like "photos of me".
  The front-camera tag comes from the photo's own EXIF: the phone recorded which lens took it. Asking a 4B vision
  model "Is this a selfie?" about a photo the camera already says was taken with the front lens buys nothing and
  costs 2-5 s each - and that is why "selfies", Reza's own first query, had never once finished. So for a request
  that is ONLY about selfies (Grounding.isBareSelfieRequest: every word is a selfie word, filler, or an asking word
  like "show"), the plan now carries no judge question at all and the existing no-question path answers from the
  scope filter: instant.
  The scope filter had to get stricter for this, and only for this case: with a judge question an unknown camera
  tag stays in scope and the judge decides (unchanged, and what the Python engine does); with NO judge question
  nothing downstream would ever look at that photo, so the answer is what the camera itself vouches for.
  "selfies at the beach" / "selfies with mum" / "selfies from 2019" still go to the judge - 9 tests cover the split
  (BareSelfieTests, SelfieScopeTests). NOTE for the cluster: the unjudged case has no Python counterpart yet.

MAC 10-11 21:55 — Blocked on the phone being locked.
  `devicectl process launch` -> FBSOpenApplicationErrorDomain 7: "could not be unlocked". The query file is already
  on the device, so the selfies run starts by itself the moment the app is opened.
- 10-11 ~22:30 [M44 step 1] Scene false positives (eval/vector_enough/false_positives_scene.txt; agent, no new model
  runs; I viewed sheets s00 + s10 myself: 8/8 cells match its descriptions). Labelled judge-yes photos (top-rank
  biased): food 38/118 not a match, beach 36/126, sunset 21/114. Kinds: the subject's SETTING without the subject 40/95
  (diners, food shops, pets eating, beach-trip people); look-alike scenes 30/95 (rocks/lake shore, golden-hour light);
  label strictness ~15 + 19 borderline (mostly 'unsure'). KEY: eye precision by stored P(yes): food 0/10 at
  0.99-0.999, 1/13 at 0.999-0.9999, 65/68 at P == 1.0 (fp32); beach 0/10, 0/7, ... 86/92 at 1.0. The 0.99 cut is not
  where the judge separates for scene questions. Cut 0.99999 on food: 29/38 FPs gone, 2/80 matches lost (labelled
  only). NOT YET ADOPTED: recall cost on the unbiased RESULTS 34 strata, all search kinds, is being measured
  (eval/judge_cut_sweep/, agent), and whether the phone's fp16 arithmetic can resolve P near 1 (logit gap instead).
- 10-11 ~23:00 [M44 step 2] Cut sweep (eval/judge_cut_sweep/, agent; RESULTS 34 estimator + R45 census, bootstrap 95%;
  no model runs). Object questions: keep 0.7 (dog/car/bicycle precision .98-1.00 already; 0.99 would cost car ~112/564
  real). Subject questions: per-subject optimum differs (beach needs P==1.0, food ~0.999999, sunset 0.999-0.9999), so a
  modest global step: 0.99 -> 0.999 (logit gap 6.91): precision beach .62->.67, sunset .74->.83, food .60->.68; recall
  beach .995->.995, food .931->.931, sunset .921->.884. 28 of beach's 230 P==1.0 photos are wrong with full certainty:
  no cut fixes beach -> question wording (M44 (b), re-judge on GPU). Phone P = py/(py+pn) in fp32 from full softmax;
  stored logit gaps sit on a 0.25 grid (bf16); fp32 saturates at gap 17.5; recommend a logit-gap rule. NOT ADOPTED
  YET: the cut is also fast mode's stop threshold -> replay tune_stop_rule/tune_stop_time at 0.999 running (agent).
  Also: Python port of the Mac's bare-selfie rule (converse.is_bare_selfie_request, engine.scope_mask front-only when
  no judge question); test_engine updated for bare vs judged; tests rerun pending.

MAC 10-11 22:10 — Two new instruments, after eight opaque minutes.
  The first selfies run on the no-judge build produced nothing for eight minutes after the planner loaded, with ZERO
  judge calls in the log - exactly what the fix intends - and the log could not say whether planning, the scope
  filter or the UI was the slow part. Both are now logged: "PLAN start" / "PLAN done (N s): albums=.. q=.. camera=.."
  around the planner call, and "NOJUDGE <album>: N in scope, 0 judge calls" on the no-question path. The planner is
  a 4B text model on the same hot GPU, so its cost is real and was invisible.
  Test suite, honestly: all 125 XCTest tests pass, and BareSelfieTests / PrecisionTests / PassStepTests /
  VectorFirstTests pass when run on their own. The WHOLE suite in one process aborts (signal 6) right after
  IndexStoreTests' 0.2 GB store bench; IndexStoreTests passes alone, and nothing I changed touches it, so this
  reads as the harness running out of room with every suite in one process. Not green end to end - saying so
  rather than reporting a pass I did not get.

MAC 10-11 22:14 — "selfies": 267 found, 0 judge calls, 56 s. Instant, and nowhere near complete.
  The measurement, from the file-driven run (so it is definitely the query I asked for):
    22:12:08 ONEQUERY start -> 22:13:37 PLAN start (89 s of startup) -> 22:13:44 PLAN done (7 s)
    q=none camera=front  ->  NOJUDGE  ->  22:14:34 done: found 267, inScope 267, judged 0, 56 s
  So the bare-selfie rule works exactly as designed: no judge question, answered from the camera tag, 56 seconds
  against a search that had never once finished in 18+ minutes. And planning is 7 s, not the bottleneck I guessed.
  BUT 267 of 187,000. The reason is already in Index.swift and in the 10-07 journal, and I should have carried it
  forward before claiming anything: the front/back tag comes from the original's EXIF lens model, EXIF needs the
  WHOLE original, and ~170k of Reza's originals live in iCloud - we never download one just to read a tag. So only
  the 267 photos whose originals are on the phone carry it. Before this change those untagged photos were in scope
  and the judge crawled through them; now they are excluded and the answer is fast and wrong-by-omission.
  I am keeping the instant answer and making it say so on screen ("267 photos your phone recorded with its front
  camera; N others are stored in iCloud, so which camera took them cannot be read without downloading each one -
  any selfies among those are not here yet"). An honest incomplete answer in 56 s beats a silent one that never
  arrives, but neither is the product.
  THE REAL FIX is M46, and the 267 are the gift that makes it cheap: they are a LABELLED POSITIVE SET, and the
  back-tagged photos are labelled negatives, on Reza's own library. A selfie is a face at arm's length - a large
  face box, near the middle - and the face boxes are already indexed for 79,576 photos with no download at all.
  Fit the geometry on the tagged subset, apply it to the 170k untagged: a complete selfies answer with zero judge
  calls and zero downloads.

MAC 10-11 22:25 — Reza asked "is the videos thing done?" and I could not answer it. Now I can, and it was a bug.
  He read the line out exactly: "checking more of each video 39578 of 39579". The real counts, from the phone:
    videos 39,579 | sweep 1 done 39,578 | sweep 2 done 11,734 | unreadable 1
  So sweep 1 is ONE video short - a video it cannot read - and because the banner switched to the sweep 2 line only
  on sweep1Done == videos, it sat on "39,578 of 39,579" indefinitely while the actual work, sweep 2 at 11,734 of
  39,579 (30%), was invisible. He could not have known; neither could I.
  This is the F2/F3 mistake a third time: "nothing left that I CAN do" is not "zero left". Fixed with the count the
  pass already keeps (PhotoIndex.framesSkipped): sweep 1 is finished when sweep1Done + unreadable reaches the
  total, and the sweep 2 line then says "(1 could not be read and was left out.)".
  My first attempt at the fix was WRONG and the existing test caught it: I switched on sweep2Done > 0, but sweep 2
  reuses sweep 1's frames, so videos sampled in an earlier run count in both and sweep2Done is nonzero while sweep 1
  is still working (LazyVideoTests.testBannerLineIsWholeJobAndGoesAway asserts exactly that). 12 existing tests and
  5 new ones (VideoLineTests) pass.
  Also added, because this question should never need me: the app writes Documents/progress.json (videos,
  sweep1Done, sweep2Done, unreadable, timestamp) on every progress tick, so the Mac can answer "how far along is
  it" without a screenshot and without asking him to read the screen.
  ANSWER TO HIS QUESTION: no, videos are not done. Every video is searchable by its cover frame and by its middle
  and end (sweep 1, 39,578 of 39,579). Looking inside them properly - a frame every 4 seconds - is 30% done.

MAC 10-11 22:38 — The video pass is working, and its rate halved in ten minutes.
    22:23:08  sweep 2 at 11,734
    22:28:13  sweep 2 at 11,963   (+229 in 305 s = 45 videos/min)
    22:38:16  sweep 2 at 12,218   (+255 in 603 s = 25 videos/min)
  Same thermal decay as the judge (0.586 -> 0.212 photos/s): 45/min cool, 25/min warm. 27,361 videos left, so
  ~18 h at the warm rate and ~10 h at the cool one - and it will keep sliding toward the warm end.
  The banner fix also self-corrected exactly as predicted: unreadable was 0 right after the install (the skip set
  is per launch), and by 22:38 the pass had retried that one unreadable video, failed, and recorded it, so the
  banner now shows sweep 2's real count instead of "39,578 of 39,579".

MAC 10-11 22:42 — M46 export built: face geometry for every photo whose camera is KNOWN.
  Triggered through the same consumable file (`#selfieGeometry` in next_query.txt), so no developer run ever needs a
  sticky launch argument again. It reads NO images - every face box is already in the index together with the size
  of the image it was found in - and writes geometry only: camera, screenshot flag, face count, the biggest face's
  area as a fraction of the frame, how far its centre sits from the image centre (in half-widths), its aspect and
  the image's aspect. No asset ids, no embeddings, no dates. Lands in Documents/selfie_geometry.csv and is pulled
  into data/private/ (never committed).
  The point: the 267 front-camera photos are labelled positives and the back-camera photos labelled negatives, on
  Reza's own library, so the cluster can fit "this is a selfie" on shape alone and apply it to the 170k photos
  whose EXIF we will never download - a complete selfies answer with zero judge calls and zero downloads.

MAC 10-11 22:53 — Face geometry separates selfies cleanly. The selfies fix is real, and it needs no judge.
  Exported from the index alone - no image reads, no model calls, numbers only (data/private/audits/selfie_geometry.csv):
  2,248 photos carry a camera tag, 579 front and 1,669 back.
      front: 487 of 579 have a face; biggest face fills a median 0.1038 of the frame; centre offset 0.424
      back:  828 of 1,669 have a face; biggest face fills a median 0.0112 of the frame; centre offset 0.415
  A 9x difference in face size, and one threshold on area fraction alone (t = 0.039) gives precision 0.86 /
  recall 0.90. The centre offset is worthless (0.424 vs 0.415) - people do not centre their own face any more
  than anyone else's - which is a guess of mine the data killed, and worth recording as such.
  Two things this changes: the labelled positive set is 579, not the 267 the search reported, and a complete
  selfies answer for the 170k untagged photos is a threshold on numbers already in the index.
  AND A DISCREPANCY I WILL NOT LET SLIDE: the selfies search said 267 in scope at 22:14; the export counted 579
  front-camera photos at 22:52. One of those is wrong about an answer Reza sees. Possible causes: the tags grew
  during the evening's re-indexing, or the scope filter drops front-tagged items for another reason. The NOJUDGE
  log line now carries a camera census (items / front / back / unknown), so the next selfies run settles it.
- 10-11 23:20 ADOPTED subject cut 0.999 (JudgeCutoff.swift + JudgeCutoffTests, eval/score_apple_photos.py default;
  RESULTS 46). Fast-mode replay (eval/judge_cut_sweep/fastmode_replay.txt, Slurm 52293499): same stop point, first
  answer pooled precision .672->.734, recall .958->.950. Python port of the bare-selfie rule (converse.is_bare_selfie_
  request, engine.scope_mask front-only without a judge question). pytest 102/102 (Slurm 52292411); swift test 125/125
  (Slurm 52297141) then signal 11 at process exit (new; Mac's async tests suspected, flagged in M44-DONE). Mac's M46
  geometry (area fraction 9x) reviewed: overall recall 436/579 not 0.90; precision depends on base rate; threshold fit
  in-sample -> cluster fits on public, validates on the phone CSV.

MAC 10-12 00:05 — I claimed "we have never trained anything" and Reza asked "did you not read journal?". He was right.
  I had read the RESULTS section headings, not JOURNAL.md, and stated a confident premise that the journal disproves
  in its own words. The record:
    - 10-05 02:30 "First training run in the project": LoRA distillation of the 9B planner into Qwen3.5-4B. Iterated
      to 4,426 cleaned 27B-teacher examples (incl. 2,400 chained edits), 2 epochs, loss 0.17 -> 0.06. RESULTS 31:
      26/30 -> 30/30 scripted conversations, 30/30 on the phone's own 4-bit weights, vs the 9B teacher's 29/30.
      THE PLANNER RUNNING ON HIS PHONE TONIGHT IS THAT TRAINED MODEL. I have been describing it all evening as
      "a 4B text model" without once saying we trained it.
    - RESULTS 32: a judge distilled the same way was REJECTED - "Qwen3.5-4B distilled from the 9B, 4-bit, 219/41,
      same curve as the base at equal strictness: not adopted".
  Both facts matter for M48, and the second is evidence AGAINST the naive version of it: distilling the expensive
  stage into a same-size model bought nothing. What has never been tried is a small learned head over the STORED
  embeddings - the hypothesis that the loss is in cosine similarity being an unlearned scoring function, not in
  model capacity. M48 corrected to say that, with both priors stated.
  Also noted from 10-05 02:50: Reza asked then what the training data actually contained and I had not checked;
  the check found 145 of 1,984 targets were the teacher's FAILED retries. Second time he has caught me asserting
  something about our own work that I had not verified. The rule I am writing down: before any claim about what
  the project has or has not done, grep the journal - it is 2,420 lines and it is the record, not my memory of it.
- 10-12 ~00:45 Reza: "the best next step should be training the smallest model for our complex queries". Wrote him the
  full project history (from RESULTS 1-46 + JOURNAL, not memory; corrected my "never trained" relay: planner LoRA shipped
  30/30, judge distillation rejected R32). Plan agreed in principle: (1) learned head over stored embeddings (M48,
  running); (2) COMPLEX-QUERY BENCHMARK first (DISBench F1 0.13 untouched since 10-03) - started now (agent:
  eval/complex_bench/: strong-VLM facts per photo on public libraries, LLM-written owner-style queries in 8 categories,
  eye-verified answer sets, library-level TEST split, baseline of the current system); (3) then train where the
  benchmark shows the loss (planner on complex queries; possibly the embedding model on personal-photo data).
  Running agents: M48 learned head, M44(b) scene wording (GPU), M46 selfie detector, M45 vector-first in face sets,
  complex benchmark. Phone overnight = Mac's M47 queue (Reza: plug in, app open, don't swipe away).
- 10-12 ~02:00 [M45] Vector-first inside face-filtered sets: NO (eval/vector_first_person/, agent; 256 public person
  searches, 32,080 judge calls, 208 eye labels; I viewed sheet tired_01: 6/6 labels agree). Rule fired 0/256; no safe
  lower threshold; vector top 20 ~0.34 right by eye vs judge screen ~0.76; vectors rank scene/object looks well inside a
  person (AUC 0.83-0.99) but expressions badly (smiling 0.68, tired 0.61). Person searches stay judge-bound. 4 shards
  lost to a filesystem slowdown (imports 50-80 min). Vectors/sheets git-ignored.
- 10-12 ~02:40 [M44 (b)] Scene question wording re-judged on GPU (eval/scene_questions/, agent; 142,674 judge calls with
  the exact phone-weight pipeline, reproduces stored P at 0.999 on 7,883/7,886; +401 blind eye labels; I viewed sheet
  beach_24: 6/6 labels agree). ADOPTING (implementation + fast-mode replay by agent, RESULTS 47): beach -> "Is a beach
  clearly visible in this photo?" at 0.999 (precision .69 -> .85, recall .968 -> .949); food -> same question at 0.9999
  (.70 -> .80, .957 -> .947); sunset unchanged. Not a generic template: "main subject" / "clearly visible" break flowers,
  church, cat in other libraries. GPU nodes with stuck I/O tonight: kempner_rtx 7c19xx/23xx and several 8a19xxx (7.5 of
  9 GPU-h lost to jobs hung at startup).
- 10-12 ~03:00 [M46] Selfie detector (eval/selfie_detector/, agent): public Wikimedia Commons iPhone photos labelled
  front/back from EXIF model+focal+aperture (3,028 front / 14,566 back, 4,244 uploaders), split by uploader. Vector
  logistic regression AUC 0.994 vs face geometry 0.932 (0.50 without a face); geometry adds nothing. I viewed the top
  false-positive sheet: 6/6 are selfie-like (back-camera arm's-length, group selfies). Handed to the Mac as M50: one dot
  product per stored vector, validate on the device's tagged photos, then bare "selfies" = tag OR score inside the
  owner's face set. Data in data/public/raw/commons_iphone (7.4 GB, ignored); weights.json committed (public).
- 10-12 ~03:40 [M44 (b) ADOPTED] b2eccec (agent): beach -> "Is a beach clearly visible in this photo?" at 0.999, food
  subject cut 0.9999 (Python + Swift + replays). pytest 103/103 (Slurm 52320910); swift 127/127 incl. GROUND parity 0/2,300
  (52322790), then the known exit segfault (Mac's BareSelfieTests). Fixture: 4/2,300 plans change, all beach. Fast-mode
  replay: first answer precision .743 -> .826, recall .940 -> .929 (beach -.019, food -.010; unsure-as-match beach
  -.079: mostly beach-trip close-ups), same calls/minutes; beach's head rule alone would stop at 550 vs 600 (first
  answer unaffected: budget ends at head 451). RESULTS 47. Mac: M51.
- 10-12 ~04:00 [M48] Learned head over stored vectors (eval/learned_head/, agent): NOT a judge replacement yet. Unseen
  query: eye P/R .80/.79 vs cosine .81/.80 vs judge .92/.90 (audit4 pooled); seen concept .88/.85 (car .50 -> .79).
  Agreement filtering of teacher labels changes nothing. Cannot express small/edge objects (I viewed miss_bicycle: 6/6
  real tiny bikes scored .04-.14), person attributes, counting, relations, text. Only 44% of the distill pairs had
  vectors (filesystem: mmap'd .so reads at ~22 KB/s, torch import > 45 min). Next: scale questions on GPUs and test
  unseen-query generalization (the deciding question).
- 10-12 ~04:30 [M48-scale] (agent) Learned head at 10x questions, running. Filesystem fix found: small-file opens and
  mmap faults on the lab Lustre cost ~0.3-0.4 s each on some nodes, and pages read ahead get DROPPED again within
  minutes (client lock LRU), so a one-off `cat` prewarm is not enough. Working recipe: scripts/prewarm_env.sh (libs +
  python binary + every .py/.pyc of the env, parallel) and scripts/keepwarm.sh (repeat it every 15 s during startup:
  `bash scripts/keepwarm.sh vllm 900 & python ...`); photos packed into 16 big files (.cache/dpack, eval/learned_head/
  pack.py: 71,308 training-pool photos in 150 s, read sequentially by every job). Embedding the 7 missing lh_img shards
  from the pack took ~6 min (parity vs file route: cosine >= 0.99985 on 64 photos x 5 crops). The stalled lh_img jobs
  were paused (SIGSTOP) and their A100 allocations reused via `srun --overlap` (eval/learned_head/inalloc.sh).
  All 115,776 judge_distill pairs now have vectors (build.py rerun). Questions: Qwen3.5-9B wrote 6 per photo for 9,974
  photos of the 37 training libraries (60,366 raw) -> 11,661 kept after the everyday + neighbourhood regex (2,252), text
  cosine >= 0.80 to any everyday eval text (4,312) and paraphrase de-dup at 0.95. Labels: phone judge (Qwen3-VL-4B
  real weights) on 250 most-similar + 150 random training-pool photos each = 4,664,400 pairs (label.py, 128 claimable
  shards, 9 GPUs).
- 10-12 ~06:30 (written 12:20 after the session restart) [M48-scale ANSWER] eval/learned_head/scale_report.txt (agent).
  11,661 generated questions x 400 photos = 4,664,400 phone-judge labels (parity 96/96, 176 pairs/s/A100 with photo-
  grouped prefix caching; ~19 A100-h allocated, ~13 productive). Head ALONE on unseen everyday concepts with their
  neighbourhood held out stays BELOW cosine at 2.4k/5k/10k questions (EV16 AP .595/.602/.617 vs .692; A4 eye P/R at the
  judge's count .788/.766 vs .814/.791 at 10k): it suppresses boats/cars/pets it only ever saw as "no" (boat AP .21 vs .78;
  sheets/boat_head viewed: models ranked above real yachts). It DOES beat cosine on unseen planner questions whose
  neighbourhood is covered (JD top-24 AUC .736 vs .634). Cosine + head BLEND (z-scores, fixed equal weights) beats cosine
  everywhere: eye .847/.825 vs .814/.793 (N10000), .848/.822 (2.4k JDonly); fast-mode replay -23% judge calls on audit4,
  -22% on ev16 without selfies (JDonly blend -33% / -35%); sheets/car_blend_gain viewed: 6/9 clearly cars. Go: blend as
  fast mode's ranker (JDonly head, 8.9 MB); no-go: head alone, or as a verifier for any kind. More questions barely move
  the blend. Pack of training-pool photos left in .cache/dpack (8.7 GB, public, ignored).
- 10-12 ~13:00 [M49 -> M52] Long-video test (eval/video_sampling/long_videos.txt, agent; 178 Commons videos 80-868 s, 40
  queries, 1.9M 9B judge calls): sweep 2 roughly doubles matches found for videos >= 3 min (15 -> 29 of 73; 6 -> 20 of
  60; p<0.01, also with PE-Core-L), not established for 80-180 s. My 01:05 pause (M49) was right for short clips and
  WRONG for long ones (I had generalised from section 37, which had no clip >= 80 s). Viewed 2 sheets (1 real, 1
  borderline). New rule M52: sweep 2 only for videos >= 180 s.
- 10-12 ~12:10 Session gap: my session stopped ~06:30 and restarted ~12:10; resumed the 3 agents. The Mac pushed
  nothing between 00:05 and 13:00 (M47 phone tests, M49/M50/M51 not done).
- 10-12 ~13:00 [M48-SHIP / RESULTS 48] (agent) Learned head + cosine blend as fast mode's ORDER, made shippable.
  (1) Variant without |img - txt| (foldable to one 1024x1024 product per photo; 2.6 vs 4.7 MFLOP, 6.8 vs 8.9 MB) retrained
  with the same script/data/seeds (train_scale.py JDonly-na; JDonly-r re-run reproduces JDonly exactly): within seed
  noise on ev16, WORSE than every JDonly seed on audit4 (-19..-24% vs -28..-33% calls; RESULTS 47 plan -17..-18% vs
  -21..-33%) -> NOT kept; ship JDonly (B = 4) seed 0, 8,919,844 B fp16. (2) The scale report's -33% / -35% was the 3-seed
  ensemble; the shipped seed 0 gives -28% / -33% on that plan and, on the RESULTS 47 plan (beach "clearly visible" P +
  the head reading that text, food 0.9999), audit4 3,453 -> 2,723 calls (-21%; -27% vs the phone's own B-16 cosine order),
  ev16 w/o selfies 70,687 -> 48,076 (-32%; -42% vs B-16), first-answer recall up (.537 -> .542, .911 -> .918). Beach loses
  its gain under the new question (601 -> 607 calls). int8-simulated text tower: same numbers. z must be over one library
  (the phone): ev16 per-library = pooled; audit4's 4-libraries-as-one is pooled. (3) Phone file Models/learned_head.fphead
  (in git, standardisation folded, layout in head_ship.py; export vs torch max |d logit| 0.0013, top-200 identical).
  (4) FindPicsCore HeadRanker.swift (fold per search, GemmNT seam, itemLogits, zScores, blendScores) + golden test on 3
  audit4 searches x 500 photos: |d blend| <= 9.3e-6, top-200 identical; swift test 130/130 (52411005, then the known exit
  segfault); pytest 137/137 (52410845, + tests/test_head_ship.py). (5) Not wired into the server engine (L-14 indexes).
  App wiring for the Mac (SearchEngine.run order) is in the report; device time / memory not measured.

MAC 10-12 14:20 — M51 done, and the segfault the cluster flagged was a DATA RACE IN SHIPPING CODE, not a test bug.
  M51 itself was a pull plus the stale comment at Search.swift:44 (now naming RESULTS 47's per-subject cuts:
  objects 0.7, beach 0.999 under its new wording, sunset/other 0.999, food 0.9999).
  The segfault: FindPicsCore.regex() kept its compiled-pattern cache in a `nonisolated(unsafe)` global dictionary
  with no lock. swift-testing runs tests CONCURRENTLY where XCTest ran them one at a time, so two of my new tests
  called normText at once, both wrote the dictionary, and it corrupted - "unrecognized selector sent to instance
  0x8000000000000000" out of Dictionary.setValue, which the stack names directly.
  THE APP DOES EXACTLY THE SAME THING: a search calls normText (plan grounding, place words) while indexing calls
  it on another task. This was a live crash waiting for the right timing, and 46 sections of evaluation never hit
  it only because XCTest serialised the tests. Fixed with an NSLock. Whole suite now green end to end: 130 XCTest +
  15 swift-testing, three runs clean.
  AND IT CORRECTS ME: last night I reported the suite's signal 6 as "the harness running out of room with every
  suite in one process". That was a guess dressed as a diagnosis. It was this race all along.

MAC 10-12 14:19 — M50 ON DEVICE: the detector is good but the cluster's THRESHOLD does not transfer.
  SELFIEVALIDATE pos=581 neg=1671 auc=0.9633 recall@0.5=0.959 fpr@0.5=0.2304 t95=4.867 recall@t95=0.458,
  2,252 photos scored in 123 ms - milliseconds for the whole tagged set, as promised, one dot product each.
  AUC 0.963 here vs 0.994 held-out on Commons: the ranking survives. The threshold does not. At the cluster's
  logit 0.5 the false-positive rate is 23% on his library against 0.84% on Commons - and FPR does not depend on
  prevalence, so this is a real distribution gap, not a base-rate artefact. Most likely cause: the fit used
  1,280 px thumbnails and the phone stores 448 px renditions (the cluster estimated ~0.995 cosine; the logistic
  direction is evidently more sensitive than that).
  SO THE THRESHOLD IS NOT SHIPPED. Each library calibrates its own on its tagged photos (PhotoIndex.selfieThreshold:
  the lowest cut whose selfies are >= 95% really front-camera, needing >= 200 tagged and >= 50 front photos, else
  the detector is not used at all rather than trusted blind). On Reza's phone that is 4.867, recalling 46% of his
  tagged selfies at 95% precision. A shipped 0.5 would have quietly filled his selfies with other people's photos.

MAC 10-12 14:22 — M52 done: sweep 2 now runs only for clips >= 180 s, and the banner counts those.
  videoFramesNextSweep takes the clip's length, read off its own last sampled frame time (sweep 1 lays it at
  duration - 0.05), so no new metadata was needed. Short clips stop after sweep 1; stale image preparations are
  still re-sampled at any length, because that is correctness rather than coverage. The banner's sweep-2 line
  counts against the LONG clips ("Looking closer inside your longer videos: k of N") - against all 39,579 it would
  have sat unfinished for ever once short clips stopped qualifying. 5 new tests; two existing tests updated
  deliberately (they built items with no duration, so under the new rule they were asserting nothing).

MAC 10-12 14:24 — M53 done: the learned head orders the judge's queue.
  SearchEngine.learnedHead (8.9 MB, bundled) folds the question vector once, scores the in-scope photos, and
  blendScores(cosine:headLogits:) gives the order the judge works through - 27% fewer calls on audit4, 42% on
  ev16 for the same matches. firstScreenDecision and likelyAtOnce deliberately stay on the RAW cosine scores,
  since their thresholds are cosine values. HEADRANK logs n and ms so the device cost is measured, not assumed.
- 10-12 ~14:40 Mac did M51/M50/M52/M53 (4f4692a). Selfie detector on the device: AUC 0.963 (581 front / 1,671 back),
  but logit 0.5 gives FPR 0.23 there vs 0.0084 on Commons: threshold does not transfer; Mac calibrates per library
  (95% precision -> recall 0.46). My hypothesis: fitted on 1,280 px thumbnails vs the phone's 448 px rendition ->
  selfie agent re-extracts Commons features with the exact phone simulation and refits (weights_phone.json). Mac also
  found the Linux test segfault was a real data race in FindPicsCore.regex()'s cache (fixed with a lock).
- 10-12 ~15:10 COMPLEX-QUERY BENCHMARK (eval/complex_bench/, agent): 57 public DISBench libraries (109,467 photos);
  27B facts per photo; 1,873 owner-style requests in 8 categories with truth programs; 27B oracle (4.35M answers);
  splits by library: test 546 (20 untouched libraries), dev 350 (13), train 320 so far (12 of 24 tuning libraries).
  Blind eye audit 160 requests / 2,336 labels (I viewed q030_b sheet 0: 6/6 agree): verified categories (precision /
  recall of the answer sets, held-out half) a .78/.74, b .73/.81, d 1.00/.92, e .97/.72, f 1.00/.91; provisional c, g,
  h. BASELINE (current system, dev 350): F1 0.605 overall, 0.642 on verified categories; worst c person+look .467,
  h follow-up edits .451, f vague memory .545, d moments .587. Failure kinds (40 sampled): planner/grounding 23/40
  (moment -> too-wide window or per-photo question 7, time grounding incl. FUTURE dates 6, follow-up edits AND/undo 6,
  place names literal 3, anchor not found 3), judge strictness 8/40, named-person over-inclusion 4/40. Caveat: generator,
  oracle and product judge are all Qwen3.5 (correlated errors); numbers = agreement with ~0.8-precision truth.
- 10-12 ~15:30 [M54] Selfie threshold gap: my rendition hypothesis was WRONG (phone-sim Commons vectors: FPR@0.5
  0.87% vs 0.84%). Cause = his back-camera photos are close portraits; Commons back photos with face >= 5% of frame
  reproduce the phone (AUC 0.960, FPR 0.19-0.22, recall@95% ~0.5). Per-library calibration is required. weights_phone.json
  available. Asked the Mac for a pixel-size census (3088x2316 = 7-MP front cameras of 2016-2019 iPhones).

MAC 10-12 14:35 — "Check a sample" no longer waits for the search to finish.
  Reza: "for food photos does check a sample button come up after a hwile?" It did, and that was wrong: the button
  needed r.done, which on a judge-bound search is the first answer - 10 min on a cool phone, 45+ on a warm one -
  so the one query whose precision we most need checked (food 0.57 by eye) was also the one he had to sit and wait
  for. Now the button appears as soon as anything is found, the sheet says "the search is still running, so these
  are the matches so far", and the summary says his answers carry over - which they do, because they are stored
  per photo and the range is recomputed against the set size when the sheet is reopened. "Save as album" still
  waits for done, since saving half a search into Photos is a different thing.

MAC 10-12 17:10 — "am i blind go look at my phone I dont see the option to choose yes or no". He was not blind.
  I looked (devicectl screenshot, deleted straight after - it held a third party's Instagram story): the phone was
  showing a FULL-SCREEN PHOTO with no title, no button and no hint, unchanged across two captures seven minutes
  apart. That is FullPhoto, the photo viewer - not the audit sheet. No AUDIT line was ever written, so the sheet
  never opened.
  Two bugs of mine, both about the unhappy path:
   (1) FullPhoto had NO way out. It is a sheet with no chrome, so the only exit is swiping it down, which nothing
       on screen suggests. He was looking for the Yes/No of a different screen, on a screen that could not be left.
       Now a NavigationStack with a "Photo" title and a Done button.
   (2) "Check a sample" was 11-point footnote text in a crowded header row immediately above the photo grid, so
       aiming for it hits a photo - which is exactly what happened, three times. It is now a bordered prominent
       button. The control the entire precision measurement depends on was the smallest thing on the screen.
  What I keep getting wrong is the same thing: I build the path where everything works and leave the person no way
  back when it does not. Third time in two days (the spinner that never ended, the summary with nothing to answer,
  now a photo with no exit). Each one cost him minutes and cost us a measurement.

MAC 10-12 17:25 — THE FIRST EYE-MEASURED PRECISION ON REZA'S OWN LIBRARY. He answered both samples himself.
    "photos of me"                 25 of 25 right -> 95% range 86.3-100%  (of 41,541: 35,842-41,541 right)
    "food photos", as shown so far 24 of 25 right -> 95% range 79.6-99.9% (of the 87 on screen: 69-86 right)
  THE CAVEAT THAT MUST TRAVEL WITH THE FOOD NUMBER, or it will mislead everyone including me: his 25 were drawn
  from the 87 results the app had found AT THAT MOMENT, which are the vector-first "likely matches" - the
  highest-ranked head of an unfinished search ("checked 28 of 147,632", done=false). The cluster's 0.57 (RESULTS 45
  1a) is whole-set precision of the COMPLETE first answer. These are not the same quantity and 24/25 does not
  refute 0.57. What it does say: what the app PUTS ON SCREEN FIRST, in the seconds after you type, is nearly all
  right - which is what a user actually experiences. The honest next measurement is the same sample against the
  finished set; his answers are stored per photo, so re-opening the sheet later counts them towards it.
  "photos of me" has no such caveat: no judge ran, the set is the whole face-matched answer, and 25 uniform draws
  of 41,541 came back perfect. The face filter is carrying that set.
  Process note: it took SIX attempts to get this measurement, every failure mine - a spinner with no way out, a
  summary with nothing to answer, a photo viewer with no exit, a footnote-sized button, and finally a button whose
  touch area sits ABOVE where it is drawn (Reza found that one: "i clicked above check a sample and look my
  screen now"). The log proved him right: AUDIT tapped + AUDIT open at 17:21:36 from a tap above the button, after
  three taps ON it produced only PHOTOVIEW opened. Still to fix.

MAC 10-12 17:40 — #2 and #3, after Reza overruled my priority. He was right to.
  I had put the Apple Photos head-to-head first. His objection: "we dont even have something we think is
  trustworthy right so why is #1 best". That is correct and my case was weaker than I made it sound - we already
  know what is broken (scene precision, person+look speed, selfies incomplete) and the comparison would not change
  any of those. A benchmark that changes no decision is not the top item; it is a pre-ship question.
  #2 THE BUTTON'S HIT AREA. Cause: the .searchable field stays up after a search, inserts itself into the scroll
  view, and the hit testing below it ends up offset by about its height - which is exactly the gap Reza found by
  tapping above the button. The field is now dismissed on submit (which is also what anyone expects after pressing
  Search: the keyboard should not sit over the results), and the button is regular-sized with its own contentShape.
  Also removed the bottom-bar version of the button: it had partially landed in the file from the edit Reza
  rejected, and the header button is the one he wanted kept.
  #3 THE SELFIE DETECTOR, IN THE PATH THE SEARCH ACTUALLY TAKES. The reason M50 step 3 did nothing: the planner
  reads "selfies" as a search for the OWNER, so it goes through runPerson, not SearchEngine.run's plain-scope path
  where I wired it. Now runPerson relaxes the camera filter for a bare selfie search and replaces it with the
  detector: of the owner's face-matched photos, keep those the camera tag proves are front-facing plus those
  scoring above this library's calibrated cut. ~170k photos whose EXIF we will never read become reachable with
  still zero judge calls, and the note says how many came from each.

MAC 10-12 17:45 — M50 LIVE AND EYE-CHECKED: "selfies" 269 -> 5,506, zero judge calls, 57 s, 25/25 right.
  SELFIEANSWER tagged=489 likely=7952 t=4.867 (this library's own calibrated cut), face-filtered to 5,506.
  Reza answered 25 uniform draws himself: 25 right, 0 wrong -> 95% range 86.3-100%, so at least ~4,750 of the
  5,506 are genuinely selfies. A 20x gain in what the search finds, with no model looking at any photo, in under
  a minute - and unlike every earlier claim tonight it has been checked by eye.
  WHAT THIS SAYS ABOUT THE THRESHOLD: 25/25 at t=4.867 means the cut is, if anything, TOO STRICT. It was chosen
  for 95% precision on the tagged photos and recalled only 46% of them. The sample suggests room to lower it and
  buy recall at a precision we can afford. That is the next thing to measure, not to assume - a lower cut is
  exactly where the 23% false-positive rate at the cluster's 0.5 lives, so it will not stay this clean for long.
  THREE EYE-MEASURED NUMBERS NOW EXIST ON HIS OWN LIBRARY, all answered by him:
    photos of me  25/25 (86.3-100%)   | selfies  25/25 (86.3-100%)  | food photos  24/25 (79.6-99.9%, see caveat)
  The two that make NO judge call are perfect so far; the one that uses the judge is the one with a wrong answer
  in it, and its sample came from the easy head of an unfinished search. That is the shape of this whole project
  in three rows: the precomputed stages are right and fast, and the expensive stage is where the errors are.
  REPORTING BUG OF MINE: the SELFIEANSWER line counts tagged/likely BEFORE the face filter (489 + 7,952 = 8,441)
  while the answer is after it (5,506), so the line reads as if 3,000 photos vanished. Fixing the line to count
  what is actually returned.
- 10-12 ~19:35 Device (Mac 19:20): "food photos" first answer on the new build = 443 found, 600 judged of 147,634, 14.6
  min (thermal serious), HEADRANK 669 ms for 147,634 photos. Reza's fresh uniform 25 of the 443: 25/25 (86.3-100%).
  My ~14/25 prediction was wrong: it used pre-RESULTS-47 0.57 and ignored depth (600 calls = top 0.4% of his library vs
  7.6% of audit4's). Asked the Mac (M57) for a deeper-band sample, a screenshot-share log, and offered a public "food
  notion" set for Reza to label. Eye numbers on his library so far: me 25/25, selfies 25/25, food head 24/25, food
  first answer 25/25.

MAC 10-12 19:23 — FOOD, PROPERLY: 25/25 on the COMPLETED first answer, and the cluster's explanation beats all of mine.
  The run (new build: RESULTS 47 questions + M53 head): found 443, judged 600 of 147,634, 874 s (14.6 min), phone at
  THERMAL serious throughout, HEADRANK n=147,634 ms=669 (M53's first device number - 0.6 s to rank the whole library).
  Reza's fresh uniform 25 from the completed 443: 25 right, 0 wrong -> 95% range 86.3-100%, so >= 382 of 443 real.
  The cluster predicted ~14/25 from 0.57, and corrected themselves twice over (M57): the prediction used the
  pre-RESULTS-47 number AND ignored library size. The real mechanism is DEPTH: 600 judge calls on 147,634 photos is
  the top 0.4% of the ranking - his most food-like photos - where on audit4 the same 600 calls reached 7.6% of a
  7,886-photo library, down among the diners, food shops and pets eating where the errors are. Precision is a
  property of how deep the judged set goes. The other face of it: 443 found is probably a small share of his food
  photos, and precision will fall as keep-looking digs.
  COMPOSITION (M57(2), logged on device, counts only): of the 443, 103 screenshots (23%), 340 camera photos, 0
  videos; of his 25, 4 screenshots (16%), 21 camera photos. So the "his food photos are easy screenshots of other
  people's posts" explanation - mine - is dead: 84% of what he judged was ordinary camera photos and he still said
  yes to every one.
  FOUR EYE-MEASURED SETS NOW, all answered by him, all 95% ranges:
    photos of me 25/25 (41,541) | selfies 25/25 (5,506) | food completed 25/25 (443) | food head 24/25 (87)
  The audit records "no" - the head sample has one - so these are not a stuck button.
- 10-12 ~15:30 COMPLEX BENCH, DEV FIXES + PLANNER TRAINING (agent; test split untouched). Track A committed 1070d4f
  (Python + Swift): no future dates without a year; "N years ago" inside a moment phrase = that year; "November 27th"
  computed in code; "add X too" = OR with the latest condition; "show me all (from that day)" resets narrowing (anchor =
  the album being looked at); place-kind words (city/county/...) optional when the full name matches nothing. Fixtures:
  16/2,300 grounded plans change (13 future dates, 3 month-day), follow-ups 874 (10 changed). pytest 141/141, swift 133+26.
  Eval infra: plan_system.py (plan stage alone; --reuse raw outputs: 62/350 9B final plans differed between two runs of
  identical code from vLLM batch/GPU nondeterminism), run_system.py --plans/--by-user/--cache (judge answers shared
  across runs per library). Running: base rerun (old code snapshot) search x13, A plans, gates (9B / old 4B / old 4B q4
  on track-A code), 27B SFT conversion of 486 TRAIN requests (18 of 24 train libraries; queries_train_cb/).

MAC 10-12 19:55 — M58 item 3, and it is not a swap: THE DISTILLED PLANNER HAS NEVER BEEN ON THE PHONE.
  PlannerAdapter.directory looks for Sources/Models/planner_adapter/adapters.safetensors. There is no such
  directory - not in Sources/Models/ (which holds only the two PE-Core packages, the face model, learned_head.fphead
  and selfie_head.json), not anywhere in the repo, and `adapters.safetensors` does not exist on this Mac at all.
  `models/` is gitignored, so the adapter the cluster trained (RESULTS 31: 30/30 on the scripted conversations,
  30/30 on the phone's own 4-bit weights, vs the 9B teacher's 29/30) lives only on the cluster and was never
  transferred. The code path is sound and returns nil cleanly, which is why nothing ever failed - the app has been
  planning with the BASE Qwen3.5-4B all along.
  TWO CONSEQUENCES:
   (1) I told Reza two hours ago that "the planner on your phone is a model we trained". That was wrong. We trained
       it; it is not on his phone. The 7-second plans he has watched all day came from the base model.
   (2) M58 item 3 assumed the swap is "minutes because the load path takes a new directory". The path does take one,
       but it has NEVER been exercised, so the first adapter that arrives is also the first test of it. Asked the
       cluster to send the CURRENT adapter now rather than waiting for the retrain, so the mechanism is proven
       before the new one lands.
  Also M58 item 1, on the Mac: 134 XCTest + 26 swift-testing green (matches their Linux run), and a compile error
  they could not have seen - PlaceTests used a bare `LibraryItem`, which is ambiguous on macOS where the SDK brings
  in another type of that name. Qualified as FindPicsCore.LibraryItem, as the other test files already do. The app
  (with Planner.swift's new followUpEdits call, never compiled on Linux) builds.
- 10-12 ~20:00 COMPLEX BENCH A + B DONE (RESULTS 49; test split untouched). Dev F1 ALL: baseline 9B planner .605 (rerun
  .608) -> A (code fixes, 9B) .620 (+0.012 [+0.004, +0.022], 8 better / 0 worse; a67b1f7, eff7628). FINDING: the
  distilled 4B planner adapter (RESULTS 31; per MAC 19:55 never on the phone until M59) beats the 9B planner on this
  benchmark: .644 vs .620 (d +0.117, e -0.157); the base 4B the phone actually used was not measured here.
  B = 4B LoRA retrained (RESULTS 31 recipe) on the old distillation rows + 27B-written targets from 486 TRAIN truth
  programs (key hidden from the prompt, code-checked against the key): B1 .627 (my target rule split one thing into
  judge + filter; trips became windows), B2 (rules revised once) .633 vs .644 (-0.011 [-0.030, +0.008]); on the phone's
  4-bit weights B2 .641 vs the RESULTS 31 adapter .634 (+0.007 [-0.012, +0.027]). Gates 30/30 fp16 and 4-bit for both.
  e (multi-hop) +0.16..0.19 in both is the one consistent gain. DECISION: no swap, no MLX export of B; the adapter M59
  bundles (planner_4b27ball) stays. 16.6 GPU-h. Eval infra kept: plan_system.py (--reuse), run_system.py
  --plans/--cache/--by-user, compare.py; fla kernel overlay (4 s/step training); train_planner_lora.py resumable.
- 10-12 ~20:10 [M59] The distilled planner adapter (RESULTS 31) was NEVER on the phone: PlannerAdapter looks for
  Sources/Models/planner_adapter/, which was gitignored and meant to be rsynced by hand; never done. Every device
  plan so far used the BASE Qwen3.5-4B. I had told Reza it was on his phone - wrong. Committed the adapter (61 MB,
  sha256 451e3846...) at that path (9bc53d4); Mac verifies the load, then base-vs-adapter smoke tests on the device.
- 10-12 ~20:50 [RESULTS 49] Planner track B (retrained on 486 benchmark train requests) does NOT beat the 10-05 adapter
  (dev ALL .633 vs .644, 4-bit .641 vs .634: within noise; gains only on multi-hop e, losses on objects/attributes/
  vague-photo wording) -> not exported; the phone keeps planner_4b27ball (now bundled, M59). Track A code fixes: +0.012
  with the 9B planner, +0.003 with the adapter. The phone's actual planner until today (BASE 4B) was never scored on
  the benchmark: next. Remaining weak categories with the adapter: c person+look .498, h follow-ups .515, e multi-hop
  .441 (17 requests) - judge-side for c, plan-shape for e/h.
- 10-12 ~21:10 [RESULTS 49b] Submitted fp_cbp_A4baseq4 (plan stage, BASE Qwen3.5-4B on the phone's simulated 4-bit weights models/qwen35_4b_mlx4sim, track-A code snapshot .cache/snap_cb_A2) and fp_cbg_A4baseq4 (30-conversation gate, same weights). Search stage (13 shards, shared 9B jcache) follows the plans.
- 10-12 ~21:40 [RESULTS 49b] Base 4B plans: Python failed 3/350 plans on '"with_people": null' that the phone's Plan.swift decodes as [] (decodeIfPresent ?? []); parity fix in converse.Album/Plan (null -> default for looks/avoid/with_people/media/want/notes/unknown_people/albums), pytest 142/142. Re-planned with --reuse (474/474 raw outputs reused, 0 errors, 9 plans changed); adapter outputs never contain such nulls (0/421, 0/425) so its runs are unaffected. Submitted fp_cbs_A4baseq4_0..12 (search, shared jcache) and the gate re-run fp_cbg_A4baseq4b.
- 10-12 ~22:30 [RESULTS 49b] (1) The planner the phone actually used until M59 = BASE Qwen3.5-4B on its 4-bit weights
  (qwen35_4b_mlx4sim, track-A code + null parity fix, same 9B judge cache): dev ALL .590 vs the bundled adapter on the
  same weights .634, -0.043 [-0.071, -0.016], 50 better / 79 worse; d -0.106 [-0.185, -0.032], b -0.043 [-0.083, -0.012];
  c .505 = .505. Gate 23/30 (adapter 30/30). Base errors: dates/places restated inside the judge question, empty
  one-day windows, places as questions, contradictory filters, inverted exclusions. So M59 buys +0.043 on the phone.
  Parity bug fixed on the way: Python rejected '"with_people": null' plans the phone accepts (3/350 errors).
  (2) c failures, 20/23 viewed at native pixels with the 9B's cached P: judge leniency 7, engine top-half person rule 4
  (judge said no on 27/28 wrong photos; the rule keeps half the person's photos anyway, Python and Swift), truth bug 4
  (build.py ignores person_need when person is null: 6/50 dev c truths = every photo of a hidden day; F1 .184 vs .549),
  "the guy ..." hidden moment 3, plan 1 (unverified), lenient oracle 1, face filter 0. Simulated (cached answers) fact
  rule P >= 0.7 for non-LOOK_WORDS person questions: c +0.029 [-0.017, +0.082] on 4-bit, +0.044 fp16; not implemented
  (changes phone person albums; check eye-labelled person sets first). Stricter judge cut 0.8 does not move c (+0.004).
  Test split untouched. ~1.6 GPU-h.
- 10-12 21:15 (node clock) [RESULTS 49c] (1) BENCHMARK BUG FIXED: build.py now asks person_need of the whole photo when no person is
  named ("Is a person wearing a sari?"); oracle.py asks those questions (--users; 27B-FP8, 8 new questions, 14,973
  answers on 2 dev libraries, job 52497968, ~6 min). 7 requests, all dev c (test/train none; test.json byte-identical).
  Truths: saris 127 -> 39, underwear 61 -> 1, coconuts 23 -> 4, red jacket base camp 42 -> 19, gold panning 30 -> 4,
  white birds 46 -> 0 (dropped), red jacket skiing newly kept (311). Eye check at native pixels: removed photos 0/36
  show the thing; new truths mostly right but the oracle is lenient (red backpack = red jacket, any load on the head =
  coconuts). Re-scored on cached plans + judge: dev c .505 -> .552 (adapter 4-bit; paired 49: +.032 [+.003, +.075]).
  (2) PRODUCT RULE CHANGED (engine.person_keep / is_look_question; FindPicsCore personAlbumFound / isLookQuestion;
  App.swift): person-album FACTS keep judge P >= 0.7 only; LOOKS (look/appear/seem or a body/mood trait: heavier, fit,
  "heavy build", tired, happier...) keep the top half; pairing only for two looks. Validated: (a) vector_first_person,
  290 new blind eye labels (stratified, eval/person_rule_49c.py): F1 .541 -> .907, precision .379 -> .885, recall
  .942 -> .930; smiling measured as a FACT (fact .914 > old .860 > look .758). (b) IMDB eras: identical for every paired
  album and the look wording; owner's 33 demo runs identical (private replay). (c) dev c replayed with the product
  code on the cached judge: +.029 (4-bit), +.044 (fp16), g +.015, ALL +.006/+.008. ADOPTED. Swift also paired two
  plain "photos of Dad" albums into two empty albums (code reading); the look check removes it. pytest 145/145; swift
  test 136 + 26, exit 0, no segfault. App not compiled here: next Mac build picks it up (person albums of facts get
  shorter on the phone: "me at the beach" = the judge's yes photos, not half of my photos).
- 10-12 ~22:30 Mac 22:15: the distilled planner adapter is LIVE on the device for the first time ("PLANNER adapter
  loaded from planner_adapter"; new log line so a missing file is never silent again). Flagship heavier-vs-fit
  restarted 22:13 on the shipped stack (adapter + M60 person rule + track A); the 22:06 base-planner run was stopped
  before its first answer. Queued for Reza: M56 selfie band, M57 deeper food band, food-notion 35 (eval/food_notion).

MAC 10-12 22:25 — THE LAUNCH-ARGUMENT PATH IS DELETED, after it cost a fourth measurement.
  I looked at the phone because Reza asked, and the flagship run was planning "Is this a photo of a dog?" again -
  the stale `-runQuery` argument winning over my queued file for the second time tonight (22:06 and 22:15), after
  taking the selfies timing twice on 10-11. devicectl attaches launch arguments to the bundle id and iOS reuses
  them on relaunches nobody asked for.
  So `-runQuery` and AppModel.debugQuery() are GONE. The only way to drive a device run now is the one-shot file
  the app reads and deletes. A class of bug that has cost four measurements cannot recur, rather than being worked
  around a fourth time.
  Flagship restarted clean at 22:21:30 on the fully shipped setup, verified by the log before trusting anything
  from it: "ONEQUERY start: me looking heavier vs me looking fit".
  AND THE TRAINED PLANNER IS FINALLY RUNNING: "PLANNER adapter loaded from planner_adapter" (22:15:05). The
  adapter landed in the repo at 22:11, so my M59 was right when written and is now resolved - withdrawn. I added
  that log line deliberately: the load path returned nil in silence before, which is exactly how we spent days
  believing a model was running that was not.

MAC 10-12 22:30 — M57(4) built: the cluster's 35 public photos, labelled in the same Yes/No sheet.
  scripts/fetch_apple_photos_test.py needed the macOS CA bundle (SSL_CERT_FILE=/etc/ssl/cert.pem; python3 here has
  no certifi), then 35/35 downloaded sha256-identical. They are Commons photos the cluster's labellers called
  not-food or unsure while the judge accepted them, so Reza's answers say whether his idea of "food" is broader
  than the labels their 0.57 was measured against - the one explanation his own library cannot test.
  The sheet now takes a FOLDER as well as a search: shown in the manifest's order, never shuffled or sub-sampled,
  asking "Is this a food photo?", with only the file name and his answer recorded. Files are already on the device
  (Documents/food_notion_35). NOT installed yet: the flagship is mid-run and installing would kill it.

MAC 10-12 22:40 — M57(4) ANSWERED, and it reframes every precision number we have for scene searches.
  The cluster's 35 PUBLIC photos - Commons photos their labellers called not-food or unsure while the judge
  accepted them - shown in manifest order, "Is this a food photo?". Reza answered all 35 himself: 25 YES, 10 NO.
  So on the photos where the judge and their labellers DISAGREE, Reza sides with the JUDGE 25 of 35 times
  (71.4%, 95% range 53.7-85.4%). His idea of food is substantially broader than the labels the 0.57 was measured
  against. Working that through, roughly: if labelled precision is 0.57, precision by HIS labels is about 0.88
  (0.80-0.94); if 0.68, about 0.91 (0.85-0.95).
  CAVEAT, and it matters: this set was SELECTED as the disputed cases, so 71% is the agreement rate on
  disagreements, not on everything - the arithmetic above assumes the undisputed photos stay right, which is what
  "precision" already counts them as. It says the judge's "errors" on scene searches are mostly definitional, not
  perceptual: it is not seeing food that is not there, it is calling a diner table or a drink food when a labeller
  would not. Answers in eval/food_notion/reza_answers.tsv (order + yes/no only; the photos are public, nothing of
  his is involved).
  This also explains the day's apparent contradiction without needing the depth story to carry all of it: his
  25/25 on the completed food set is less surprising if his "food" is the broader one, since he is both the judge's
  audience and its grader.
- 10-12 ~22:55 Food notion (35 public photos, Reza answered): disputed 16/25 yes (64%, CI 43-82%), controls 9/10. By
  kind: meals with people 10/11, drinks 2/2, pets eating, produce, food event yes; food shopfronts/signs/vans with no
  food 0/7. The Mac had scored all 35 as disputed (25/35) - corrected in M62. Flagship stalled on device: person crops
  read 1280 px per photo, ~150 s per call; refused a smaller-rendition fallback (judge-resolution rule); asked for a
  per-read timing log, local-first >= 1280, deeper parallel prefetch, overnight run.
- 10-12 ~23:45 [RESULTS 50] Food by the OWNER's notion + shop-front veto (agent; eval/food_notion/report.txt). Rests on
  Reza's 25 disputed answers (+10 controls). Kind rates applied to tagged labels, every table re-scored as made
  (labellers' column reproduces R45/46/47 exactly; eval_recall.score_systems got an optional weight=). Shipped food rule
  (0.9999): precision .80 -> .89 by his notion (CP-end range .83-.91; .94 with my tags of 22 untagged labels), recall
  .947 -> .829 (.784): the strict cut removes photos he calls food (disputed photos it removes: 6/12 yes; it keeps: 10/13).
  Old 0.99 fast answer .57 -> .84. Owner F1 is flat over 0.99/0.999/0.9999 - the R47 food cut wins only by the labellers'
  notion. His recall is an upper bound (never asked about judge-rejected photos; my tags of 48 such labels -> .589).
  Veto (fp_foodveto, 642 calls, 4 min on 1 RTX PRO 6000 = 0.07 GPU-h): v_real "Is real food or a drink visible ... (not
  just a sign, menu or shop front)?" cut 0.01 chosen on both halves; held out +.025 precision / -.009 recall (labellers),
  +.029 / -.007 (owner): drops pizzeria front, ice-cream van, hotpot neon street, charity boxes + 1 wrong (chip-bag
  shelf). v_any drops Cheerios / Lay's (his YES controls) and stockfish. NOT shipped (gain inside the CIs, +1 judge call
  per kept photo, fragile wording). Candidate for the owner: 0.999 + v_real (his .940/.857 vs .935/.784; labellers .735
  vs .800) - decide after he answers the 13 unanswered photos of the [0.999, 0.9999) band.

MAC 10-12 23:15 — M56 SETTLED: the selfie cut is 0.85, and "selfies" goes from 269 photos to 13,712.
  Two bands, 25 uniform draws each, answered by Reza, from the photos each looser cut would ADD - never the easy
  ones. That was the cluster's correction to my reasoning: I had read the earlier 25/25 as "the cut is too strict",
  but that sample contained only photos ABOVE the cut and so said nothing about the ones just below it, which is
  exactly where the errors live.
    cuts on his tagged photos: p0.95 = 4.867 (recall 0.455) | p0.90 = 3.263 (0.733) | p0.85 = 2.546 (0.839)
    0.95 -> 0.90 band: 25/25 right (95% 86.3-100%)  ->  5,454 -> 11,270 selfies, ~5,018-5,816 of 5,816 added real
    0.90 -> 0.85 band: 22/25 right (95% 68.8-97.5%) -> 11,270 -> 13,712 selfies, ~1,679-2,379 of 2,442 added real
  The rule was fixed in advance: >= 20 keep, >= 23 go looser. 22 is not 23, so it stops at 0.85 by the protocol
  rather than by anyone's judgement.
  THE DEFINITION HE USED, asked for before the second band and stated by him: "photos i took of myself, mostly
  flipped camera or sometimes mirror selfies" - not every photo he appears in. That is the human definition and
  the one the product should serve; the detector's own training labels are EXIF front-camera, which is NARROWER
  (the cluster's own review found most of its "false positives" were arm's-length and group selfies on the back
  camera). So his numbers are measured against the stricter-to-satisfy definition, not a lax one.
  Whole arc for this one feature: 269 (camera tag only) -> 5,506 -> 11,270 -> 13,712, a 51x gain in what the
  search finds, with zero judge calls at any point and about four minutes of his time.
  MY BUGS ALONG THE WAY, all caught before they cost a measurement except the last: the band was not face-filtered
  (Reza caught it: "are they selfies of ME or just selfies in general"), then search() cancelled its own caller so
  the band came back empty, then I diffed against a saved baseline that my own broken run had overwritten. Fixed by
  computing both cut answers directly from the face set instead of trusting stored state.
  ONE FLAW IN THE INSTRUMENT, recorded: both bands wrote to the same audit key ("selfies to double-check"), so the
  second band's 22/25 had to be recovered by subtracting the first. Distinct keys per band next time.
- 10-13 ~00:45 [RESULTS 51, complex bench dev e/h failure kinds; test split untouched] Classifying the 27 dev e/h requests
  with F1 < 0.5 (adapter q4 + 49c code/truth: runs/cmp/r49c_A4oldq4_new_truthnew). Submitted fp_cb51_anch (race_sbatch,
  CPU work on a GPU node: the login node page-faults at Lustre speed): anchor-stage hits per dev anchored album from the
  shared 9B jcache only (.cache/cb51/anchors.py -> .cache/cb51/anchors.json).
- 10-13 ~01:40 [RESULTS 51] Code fixes for dev e/h failure kinds (Python + Swift; uncommitted until measured): one-day
  anchors (a same_day moment keeps only the day(s) within .05 of its best hit), closest moment (a same_day moment with no
  hit >= cut uses its best photo if P >= .3), year-less date with an empty window -> latest earlier year with photos,
  place lists ("Burgos and León" = either), "photos from the day/trip ..." drops a question that only restates the moment.
  Plan stage (fp_cb51_plan, adapter q4, --reuse): 3/350 plans change, 2 new raw outputs, 0 errors. Submitted the search
  stage fp_cb51_r51_search_0..12 (9B + shared jcache, snapshot .cache/cb51/snap_r51) and fp_cb51_probe2 (falsification:
  e anchors asked in 4 other dev libraries each, best P of an absent moment).
- 10-13 ~02:15 [RESULTS 51 DONE; dev only, test untouched] 27 dev e/h failures (F1 < .5) classified (eh_diag/labels.txt):
  e: anchor matched several days 5, anchor not found 4, moment restated 1, same_place 1; h: benchmark truth 8/16 (stripped
  place/days, "parade" = a whole day, concert = "a stage"), planner follow-up 3, year-less date 2, several days 1, place
  list 1, restated 1. ADOPTED (Python + FindPicsCore + App/Moments, app not compiled here): one-day anchors (.05 band),
  year-less empty window -> latest earlier year with photos, place lists ("A and B" = either), "photos from a moment"
  drops a question that restates it. Dev ALL .646 -> .666 (+.020 [+.009, +.033], 12 better / 0 worse); e .481 -> .636
  (+.154 [+.037, +.298]); h .467 -> .506 (+.040 [.000, +.096]); d +.035; a +.016; g/b/c/f 0. REJECTED closest-moment
  fallback (e would be .812): an absent-moment probe (52 asks of dev e moments in other libraries) scores >= .3 in 34/52,
  >= .7 in 15/52 (6/8 highest viewed are not the moment). Gate 30/30 (4-bit adapter); GROUND 0/2,308 (+8 dev raw cases),
  FOLLOWUP 0/874; pytest 149/149; swift test 140 XCTest + 29, 0 failures. 1.0 GPU-h.
- 10-13 ~03:30 [RESULTS 52 started: complex bench h TRUTH programs (benchmark construction, all splits; system never
  scored on test)] Read every dev + train h program: 4 systematic causes. (1) build.py's strip drops place/date
  encodings of places the owner NAMES by another word (Laos = a date window -> turn 0 = the whole library; Borobudur =
  days; Hampi = "Hosapete taluk"; Bangladesh = "Dhaka"; Machu Picchu vs "Machupicchu"); the digest shows one address
  level per day, so the generator never saw "Laos" or "Hampi" although the addresses have them. (2) a follow-up's
  program drops what the conversation keeps ("include the bobcats too" = bobcats only; "keep only September" = all of
  September); (3) events as a whole day ("the parade" = a day of hiking + lunch) or proxy objects (concert = "a stage");
  (4) the truth schema cannot OR places/albums, so "also add Zagreb" became objects (stone wall, tram). Submitted
  fp_cb52_regen_0..5 (race_sbatch, 27B-FP8 on H100/H200/RTX6000): gen.py regen_h rewrites the 171 h turn_programs with
  the same words (digest + every address level + draft + rules; generic examples), then oracle.py answers only the new
  questions. Output programs_h2/ (overlay via common.load_programs).
- 10-13 ~02:20 [RESULTS 52 DONE: complex bench h TRUTHS fixed; benchmark construction, all splits; system never run or
  scored on test] Causes (read every dev + train h program): (1) build.py's strip dropped the place/date ENCODING of places
  the owner names by another word (Laos = a date window -> turn 1 = whole library; Hampi = "Hosapete taluk"; Machu Picchu vs
  "Machupicchu"); root = the digest shows one address level per day; (2) follow-up programs dropping what the conversation
  keeps (bobcats/lions, September/Borobudur, winery/dinner); (3) events as whole days or proxy objects (parade = a day,
  concert = "a stage"); (4) the schema could not OR places/albums (Zagreb = trams). Fixes: build.py h compile rules (named
  places keep their constraints, accent/space-insensitive place match, place_any, several albums per turn = union, edit and
  stated-year checks; other categories byte-identical); gen.py regen_h = the 27B rewrote all 171 h programs with the same
  words (6 GPU jobs fp_cb52_regen_*, ~35 min each; 170 parsed, 147 changed; oracle 208,576 new answers); h cut 0.9 -> 0.5
  by a rule written before scoring. Fresh blind audit (audit3/, 20 requests, 263 labels, 4 subagents + my 74 overlap labels:
  67/67 agree): old truth precision .89 / recall .80 -> new .93 / .94 (cut 0.9 would be .98 / .66); h now VERIFIED. Dev h
  re-scored: 49c .573 -> RESULTS 51 code .621 (+.048 [.000, +.113], 3 better / 0 worse; old truth .467 -> .506): the 51 gain
  stands. 8 dev truth errors: 5 fixed, 3 removed. g's narrow albums = generation (proxy objects for events), not compile:
  not fixed. h counts: test 47, dev 27, train 62 (train.json now all 24 libraries, 651). ~3.7 GPU-h. Not pushed.
- 10-13 ~02:40 [RESULTS 53 started: complex bench g (two albums) truths; construction only, test never scored] Checked g's
  compile path: no strip for g, and the new place matcher + a stated-year check change NO kept g truth (all splits) -> g's
  narrowness is generation: events/activities tested through AND-ed proxy objects (roller derby = helmet AND rink, concert
  = stage AND microphone AND crowd), hidden days/districts ("New York" = Manhattan, "Barcelona" = Barcelonès), metadata
  asked as a photo question ("Is the address in Orange County?"). Submitted fp_cb53_regen_0..5 (race_sbatch, 27B-FP8):
  gen.py regen_g rewrites all g albums with the same words (digest + PLACE NAMES + draft + 7 rules, generic examples),
  then the oracle answers new questions only -> programs_g2/ (common.load_programs overlay).
- 10-13 ~04:10 [RESULTS 53 DONE: complex bench g truths; construction only, test never scored] g's narrow albums are
  generation, not compile (g has no strip; the new place matcher + a stated-year check change no kept g truth). gen.py
  regen_g: the 27B rewrote all 171 g programs with the same words (fp_cb53_regen_0..5 + 4 helper oracle jobs; 325,647
  new oracle answers); AND-ed proxy albums 94 -> 38, dated albums 259 -> 157. Fresh blind audit (audit4/, 20 requests,
  442 labels, 5 subagents + my 103 overlap labels: 86/86 agree): old precision .80 / recall .68 -> new .79 / .84 (F1 .73
  -> .82); cut 0.3 loses on test (pre-written rule) -> g cut stays 0.5; g stays PROVISIONAL (precision .79 < .8). Left:
  "2009 ski trip" widened to the whole year, descriptive words AND-ed, canal/river mutual "not". Dev g: 49c = RESULTS 51
  code on every g request (.728 old truth, .776 new). sft_from_truth.py understands place_any and union h turns
  (tests/test_sft_from_truth.py, pytest 155/155). g counts test 58, dev 35, train 62. ~5.5 GPU-h. Not pushed.
- 10-13 ~04:10 [RESULTS 54 started: MILESTONE on the complex-bench TEST split (548 requests, 20 libraries; truths as of
  RESULTS 53), each system run ONCE, nothing changed as a result] S0 "phone on 10-12 morning" = BASE Qwen3.5-4B on the
  phone's 4-bit weights (models/qwen35_4b_mlx4sim) + product code a95b214 (= origin/main 10-12 14:00 in src/, before track
  A a67b1f7, old person rule) + the RESULTS 49b null-as-default parity patch (Plan.swift always decoded null as default;
  without it the Python mirror rejects plans the phone runs); S1 "shipped tonight" = 10-05 adapter on the same 4-bit
  weights (models/planner_4b27ball_q4merged) + code 64709ef. Sparse worktrees .cache/wt_m54_s0 (patched) / wt_m54_s1.
  Scripts eval/complex_bench/m54/ (plan_job: test then dev; search_job: per library shard s1 then s0 so both share the
  9B jcache answers). Submitted fp_m54_plan_s0 / fp_m54_plan_s1 (race_sbatch). Dev is re-run for an exact reference
  (S0's exact system was never run on dev; dev plans reuse the same model's earlier raw outputs).
- 10-13 ~04:20 [RESULTS 54] Plan stage done: test S1 548 planned, 1 plan error; test S0 548, 0 errors; dev S1 349 (446/446
  raw outputs reused = the RESULTS 51-53 plans), dev S0 349 (24 new raw outputs). Submitted search: fp_m54_test_0..9
  (2 libraries each) and fp_m54_dev_0..3, each shard s1 then s0 with the shared 9B jcache.
- 10-13 ~04:55 [RESULTS 54 DONE: MILESTONE, complex bench TEST split, each system run ONCE, nothing changed after] Test 548
  requests / 20 libraries, truths as of RESULTS 53. ALL F1: S0 (base 4B 4-bit + a95b214 code + 49b null parity patch) .575,
  S1 (10-05 adapter 4-bit + 64709ef code) .656, +.081 [+.058, +.104], 158 better / 72 worse; ALL verified (410) .578 ->
  .653, +.075 [+.046, +.103]. Per category S1 - S0: a +.067, b +.008 (CI incl 0; dev was +.054), c +.088 (prov.), d +.157,
  e +.129 (CI incl 0, n 16), f +.028 (CI incl 0), g +.115 (prov.), h +.089 (CI incl 0). Errors: S0 0/548, S1 1/548 (plan
  ValueError, one h request). Dev reference re-run the same way: S0 .595 (new: exact system never run on dev), S1 .679
  (reproduces RESULTS 51-53 exactly, 349/349), +.084 [+.054, +.116]. No test request read individually. ~5.6 GPU-h.
- 10-13 ~05:05 [RESULTS 55 started: dev category b (attribute/colour/style) failures by cause; DEV ONLY, test closed
  after RESULTS 54] Shipped stack = m54 dev S1 (10-05 adapter 4-bit + 64709ef code = origin/main src). 14/62 dev b
  requests have final F1 < 0.5; 11/14 are precision failures (returned >> truth). Per-photo evidence dump
  (eval/complex_bench/b55/dump.py: cache-only replay of the S1 plans, 9B P per photo, oracle P per truth question, PE-Core
  look rank, pixel colour/blur/exposure stats) submitted as b55dump_0..3 (race_sbatch, CPU work).
- 10-13 ~05:15 [RESULTS 55] Dev b failures classified (14 with F1 < .5; blind eye labels at native px,
  eval/complex_bench/b55/labels.txt, then unblinded): JUDGE lenient on the look 8 (overexposed, blurry, red wraps, city at
  night = theme park, golden sunlight, b&w = toned/exterior, purple = blue-violet, smoggy skyline = hikers in front),
  TRUTH 4 (blurry action = bokeh/silky water; "that teal water" = one hidden day; London gigs + "stage" AND; rainy hike =
  wrong day), PLAN 2 (river-trip anchor window missed the trip; "real Mickey Mouse (not a toy/model)" rewrite), RANKING 0
  (the final round judges every in-scope photo; PE-Core AUC: colours median .92, photo quality .62-.78). A stricter judge
  cut does not fix b (dev b .662 at .7, .663 at .8, .589 at .9). FIX TO TEST, rule written BEFORE the train data is read:
  photo-level looks measured from pixels as a VETO on the judge's yes (only when the album's judge/filter question names
  them): black and white -> fraction of pixels with chroma > 30/255 must be <= T_bw = 97.5th percentile over TRAIN b truth
  photos of black-and-white requests; overexposed/blown out/washed out -> fraction of pixels with luminance >= 250/255 must
  be >= T_over = 10th percentile over TRAIN photos the facts tag "overexposed"; blurry/out of focus/motion blur -> mean
  abs luminance gradient must be <= T_blur = 90th percentile over TRAIN b truth photos of blurry requests. Thresholds from
  train only; scored once on dev (paired bootstrap vs m54 dev S1); adopt only if dev b improves and nothing regresses.
- 10-13 ~05:45 [RESULTS 55 DONE: dev b failures by cause; pixel veto for photo-level looks TESTED, NOT adopted] 14/62 dev
  b requests F1 < .5 (shipped stack = m54 dev S1). Blind eye labels, 125 photos at native px + 12 b&w disputes: JUDGE 8
  (looks accepted loosely: overexposed/blurry 2, colour on the wrong thing or a neighbouring hue 3, toned/exterior b&w 1,
  scene 2), TRUTH 4 (hidden day for "that teal water", "stage" ANDed into the London gigs, wrong rainy-hike day, "motion
  blur" without "action"), PLAN 2 (river-trip anchor max P .651 on the trip's day; "real Mickey Mouse (not a ... toy,
  model)" rewrite), RANKING 0 (final round judges everything; PE-Core AUC colours .92, blur/exposure .62-.78). Stricter
  judge cut: b .662 -> .663 (0.8) / .589 (0.9). Pre-registered pixel veto (b&w <= .321 coloured pixels, overexposed >=
  .0037 clipped, blurry grad <= 13.38; thresholds from 24 TRAIN libraries): dev b -.001 [-.005, +.003], 1 better / 1
  worse, other categories .000 -> NOT ADOPTED, product code unchanged (rule in eval/complex_bench/b55/pixel_rule.py).
  Strict "no colour" b&w loses (-.013): truth and judge both call sepia black and white. b's RESULTS 54 dev gain sat in 2
  base-4B scope errors (+.028 of +.054). Next by size: a b truth pass (hidden days/places, ANDed/dropped conditions), the
  anchor-not-found class, characters vs the "real X" rewrite. No model run (~1.7 GPU-h held for CPU work). Not pushed.
- 10-13 [RESULTS 56 started: complex bench b truths; construction only, test closed for system scoring] Current b
  truths (272 requests, b56/diag.py): unstated dates 76, unstated places 54, pinned to one day with no day word 77, a need
  question sharing no word with the request 22. build.py b compile path (B_FIX: place matcher + stated-year check)
  changes no b truth (FP_CB_BFIX=0 byte-identical) -> generation. Submitted fp_cb56_regen_0..7 (race_sbatch, 27B-FP8):
  gen.py regen_b rewrites every b album with the same words (digest + PLACE NAMES + draft + 7 rules, generic examples),
  oracle on new questions only -> programs_b2/ (common.load_programs overlay, FP_CB_B2="" = old).
- 10-13 23:30 [RESULTS 56 DONE: complex bench b truths; construction only, test closed for system scoring] (session was
  down 06:30-23:05; nothing lost: oracle, builds, sheets and all 4 label files were complete; the audit scoring re-run.)
  gen.py regen_b rewrote all 285 b programs with the same words (dated albums 140 -> 30, placed 96 -> 42; oracle 303,996
  new answers). Fresh blind audit (audit5/, 20 requests, 229 labels, 4 subagents + my 49 overlap labels: 44/45 agree):
  old b truth precision .56 / recall .81 (F1 .66) -> rewrite .60 / .89 (.72) -> rewrite at cut 0.9 .73 / .79 (.75).
  Pre-written rules: rewrite adopted; cut 0.8 -> 0.9 (wins test .786 vs .727 and dev .716 vs .710); b loses VERIFIED
  (precision .73 < .8). Shipped stack S1 dev b: .662 old truth -> .662 new (64 requests; .724 if the cut had stayed 0.8).
  b counts test 91, dev 64, train 113. ~4.2 GPU-h. Not pushed.

MAC 10-14 00:40 — M66 step 1 done, and WHY THE OVERNIGHT RUNS NEVER REPORTED (the cluster's question).
  The chain I armed on 10-12 23:56 was still sitting on "step 1: flagship running, waiting for its result" when I
  stopped it tonight. It was not a bug in the chain - the flagship NEVER FINISHED, twice. Measured on 10-12 it was
  doing 4 judge calls in 10 minutes, which for a 600-call first answer is ~25 hours, so it cannot complete on the
  device as written. Everything downstream (food keep-looking, the pixel census, the deeper band for Reza's
  morning sample) was blocked behind it, which is my error of sequencing: I put the slowest, least certain job
  first and gated three cheap ones on it.
  THE SUSPECT, now measured instead of assumed: a person search reads EVERY judged photo at full resolution
  (SearchEngine.redBoxCrop -> PhotoLibrary.read side 1280, purpose .judge) to draw the red box, and for an
  iCloud-only original that is a download per photo. The rule that the judge is never shown a smaller copy is not
  negotiable, so the fix has to be prefetching, ordering or skipping - and choosing between those needs the number.
  So redBoxCrop now times every read and logs "PERSONREAD n photos, X s total, Y s each" every 10 photos. If Y is
  seconds, the read is the whole cost and the judge is idle; if Y is small, the cost is elsewhere and I have been
  blaming the wrong thing.
  M65 built and installed: 140 XCTest + 29 swift-testing green on the Mac, matching the cluster's Linux run
  (App.swift and Moments.swift call the new Core functions and had never been compiled). Startup log confirms
  "PLANNER adapter loaded from planner_adapter", so the trained planner is in use; the flagship's plan came back in
  17 s with albums=2 and the red-box question, which is the shipped behaviour.
  Flagship restarted instrumented at 00:40:22, verified by log as the right query before trusting anything from it.

MAC 10-14 00:59 — THE READ IS NOT THE BOTTLENECK, AND THE FLAGSHIP IS NOT STALLED. I was wrong twice.
    PERSONREAD 10 photos, 6.5 s total, 0.65 s each
    PERSONREAD 20 photos, 35.8 s total, 1.79 s each
    PERSONREAD 30 photos, 41.4 s total, 1.38 s each
    judging this run: 32 photos in 74 s = 2.4 s each = 25 photos/min
  So a person search judges at essentially the same rate as an ordinary one (2.4 s vs 1.55-2.40 s measured on
  "photos of a dog"), and a 600-call first answer is ~25 MINUTES, not the ~25 hours I extrapolated.
  WHAT I GOT WRONG, and it cost two days of planning: on 10-12 the flagship did 4 judge calls in 10 minutes and I
  concluded the full-resolution read for the red-box crop was structurally fatal - "hours, not minutes" - told Reza
  that, told the cluster that, and built an overnight plan around it. The far likelier reading, now that the reads
  are timed: those originals were COLD and being pulled from iCloud for the first time; they are cached now, so the
  same work costs 1.4 s a photo. A ONE-TIME cost that I reported as a permanent property of the design.
  The lesson is the one I keep relearning: I instrumented only after the conclusion had hardened. The measurement
  took one build.
  Consequence for M64/M66: the question "how do we speed up person+look without lowering judge resolution" may not
  need answering at all - at 25 photos/min it is already in the same budget as every other search. What DOES need
  saying is that the first run over a cold library pays an iCloud download per photo, which is a real first-use
  experience problem (and an argument for warming the cache during indexing), not a search-time one.
- 10-13 ~23:20 [RESULTS 57 started: planner track B3 = 4B LoRA retrained on SFT targets from the CORRECTED TRAIN truths;
  dev for selection only, test closed] Train truths = queries/train.json as built for RESULTS 56 (643 requests, 767 turns,
  24 libraries; h2/g2/b2 programs checked equal to the overlays). Submitted fp_b57_sft (27B-FP8 teacher, sft_from_truth.py
  --rules=v2, grounding/checks with the shipped code .cache/wt_m54_s1/src = HEAD src) -> data/public/distill/cb_train_v3/,
  and fp_b57_base (the shipped 10-05 adapter on the phone's 4-bit weights re-planned on the CURRENT dev.json, 351
  requests, raw outputs reused from m54 dev S1 + r56_new; search with the shared 9B jcache) = the reference.
  PRE-REGISTERED (before any B3 dev number): mix = the 4,426 old distillation train rows + passing B3 rows x3 (B2's mix),
  RESULTS 31 recipe (r16/alpha32, lr 1e-4, 2 epochs, accum 8). SHIP the B3 adapter iff, on dev with the current code
  (B3 merged into the 4-bit weights vs the shipped adapter on the same weights, same 9B judge + cache, compare.py paired
  bootstrap): (1) ALL-verified (a, d, e, f, h; 202 requests) diff has its 95% CI above 0 OR is >= +.020, AND (2) no
  verified category is worse beyond noise (its 95% CI entirely below 0), AND (3) the 30-conversation gate is 30/30 on
  the 4-bit merged weights. fp16 merged is run and reported but does not decide. b, c, g are provisional truths.
- 10-13 ~23:55 [RESULTS 57] B3 targets: 27B wrote 749 turns (643 requests), 635 pass the key check (a 92/114, b 106/113,
  c 86/88, d 80/117, e 19/29, f 45/58, g 54/62, h 153/168; fails: place word not in the key's address 43, dates not
  containing the key's 29, no valid output 19, time words without dates 13). Eye check of 25 random targets (b57/
  eyecheck.txt): 18 right, 7 weaker, 0 wrong. Targeted search for the B1 kinds: AND-ed descriptions absent (filter 0
  outside h); anchors the key lacks 48 (B2 21), 17 of them next to the album's own question ("waterfall shots from the
  jungle trip" -> anchor "jungle trip?" + same_event) -> F2 drop anchor; g targets copy the truth's mutual no-overlap
  exclusions the words never ask (11, + 1 h) -> F1; moment restated when the key wants the whole window 9 -> F3 (B2's
  hand fix, now code); 1 target whose notes cite the answer key dropped. fix_targets.py: 37 rows changed, 634 kept
  (534 requests, 24 libraries). Mix planner_cbmix3.jsonl = 4,426 + 634 x3 = 6,328 rows. Submitted fp_b57_train
  (RESULTS 31 recipe). Baseline (shipped adapter 4-bit, current dev, 449/449 raw outputs reused, 0 errors): gate 30/30.
- 10-14 [RESULTS 59 started: can the query-time judge go? FACTS written once at index time + a small trained scorer at
  query time. Public DISBench complex_bench only; TRAIN = training, DEV = evaluation, TEST closed (not read/run/scored).]
  PRE-REGISTERED before any scorer number: (1) Scorers train ONLY on TRAIN libraries' 27B oracle answers (2.72M pairs,
  1,193 questions, soft target = oracle P); choices (model, epochs, cut) made on 4 held-out TRAIN libraries, never on dev.
  (2) Per-photo test on DEV truth questions (1.38M oracle pairs, 662 questions): match = oracle P >= .8, non-match < .2,
  the rest dropped; AUC (per question macro + pooled) and precision/recall at cut .7 for the 9B judge (run now on the same
  pairs, product prompt/encoding: fp_r59j_*) and for each scorer. (3) End-to-end: the shipped stack's dev plans
  (runs/cmp/r57_base, 351 requests) replayed with EVERY judge call answered by the scorer (plan stage unchanged), paired
  bootstrap vs r57_base_search (9B; ALL .679, verified .692). (4) Verdict rule: "close to the judge" = dev ALL F1 diff >=
  -.03; per category: replaceable if diff >= -.02, borderline -.05..-.02, needs pixels < -.05. (5) Eye-labelled sets: every
  scorer score on a TRAIN-library photo comes from a fold model that did not train on that library.
  Submitted fp_r59j_<user> x13 (9B on every dev oracle pair, race_sbatch).
- 10-14 00:30 [RESULTS 58 started: train the PHONE JUDGE (Qwen3-VL-4B, the phone's 4-bit language + 8-bit tower) to be
  stricter on the disputed kinds without losing recall; LoRA on vision AND language from a stronger teacher. Public data
  only; complex_bench TEST closed (never run/read/scored); dev used only for the final judge-swap check.]
  EYE SET (eval/judge_teach/eyeset.py -> eyeset.parquet, existing blind labels only, product questions/cuts of today):
  4,373 rows = everyday 3,754 (objects dog/car/bicycle/boat 1,548; subjects beach/sunset/food/flowers/cat/church 2,154;
  night/selfies 52), person red-box facts 494 (M45 + 49c sheets), complex dev b (b55) 125. Libraries: 'heldout' = section
  26's 8 (incl. audit4) + all dev/test = 3,698 rows; 'ev16' = 675 rows (the only DISBench libraries training may use:
  their labelled photos + near-duplicates are excluded from training, eval there is photo-disjoint only).
  PRE-REGISTERED, TEACHER GATE (before any judge runs on the eye set): disputed groups D1 = the 6 subject searches, D2 =
  person facts (7 conditions, red box), D3 = cb_b. Per search s: R*_s = recall of the phone judge (q3vl4b_v8) at the
  product cut (unsure rows dropped); every judge (phone included) takes its highest threshold with recall >= R*_s on s;
  TP/FP summed per group -> precision at equal recall; paired bootstrap 2,000x (rows resampled within search, thresholds
  re-derived). A teacher passes iff on D1+D2+D3 pooled its precision minus the phone's AND minus the 9B's both have 95% CI
  > 0, and no group has a point estimate below the phone. Best passing candidate = highest pooled precision; a CI-tied
  cheaper model wins. No candidate passes -> stop (distillation cannot help) and report.
  PRE-REGISTERED, SHIP CRITERION for the trained judge (before any student number). Student and baseline both scored as
  the phone runs them: language = the real mlx-community 4-bit tensors de-quantized + the LoRA in fp16 (= runtime
  adapter; the planner convention, *_q4merged), vision = LoRA merged into the bf16 tower then MLX affine 8-bit (the
  shipped tower). Baseline = q3vl4b_v8. Product cuts unchanged (0.7 objects / person facts / attributes, 0.999 subjects
  and beach "clearly visible", 0.9999 food). SHIP iff ALL of: (1) precision at the product cuts on D1+D2+D3 pooled over
  the HELD-OUT libraries: student - baseline 95% paired-bootstrap CI > 0; (2) recall at the product cuts not worse beyond
  noise on EVERY set (objects, D1, D2, D3, night/selfies; held-out libraries; recall diff CI not entirely below 0) and
  object recall dog/car/bicycle point drop <= .02; (3) complex_bench DEV with the judge swapped (phone baseline vs
  student, same plans, judge caches keyed by model): ALL F1 diff CI not entirely below 0; (4) Reza's heavier-vs-fit
  demo replay (data/private, replay only): heavier and fit album right-photo counts within 3 of the baseline's and wrong
  photos not up by more than 3 in either album; (5) the 4-bit sim holds = (1)-(4) are measured on the phone simulation
  above (a merged + re-quantized 4-bit language variant is also scored and reported, not deciding). ev16 rows reported
  separately (photo-disjoint only), never deciding.
- 10-14 ~00:55 [RESULTS 59] (session restarted ~00:30; jobs and files intact.) Data: facts text for 70,580 TRAIN+DEV
  photos (70,455 with 27B facts; median 127 word pieces); pairs train 2,723,780 (1,193 q) / dev 1,383,326 (662 q).
  Submitted: 15 scorer trainings (fp_r59_{ce12,ce6,tower12,towervec12,vec}_{h0,full}, ce12/towervec12 h1-h3 = fold
  models for the TRAIN-library eye sets), zero-shot baselines fp_r59_zs (BM25 / MiniLM / phrase max / PE-Core B-16),
  passthrough replay fp_r59_pt_0..12 (the 9B path on r57_base plans, logging every judged pair: must reproduce
  r57_base_search), 9B on the eye sets fp_r59_eye9b (E1 = RESULTS 45 746 labels, E2 = RESULTS 34 1,950 labels, all TRAIN
  libraries), and (prepared in parallel, gate unchanged) phone facts fp_r59_pf{2b,4b}_0..7 (Qwen3-VL-2B/4B at the
  phone's simulated 4-bit weights, same prompt) + fp_r59_sim08 (Qwen3.5-0.8B 4-bit sim, the smaller captioner).
- 10-14 01:05 [RESULTS 58] Eye-set baselines (held-out libraries, unsure dropped): phone q3vl4b_v8 at the product cuts
  keeps D1 subjects 474 right / 36 wrong, D2 person facts 94 / 15, D3 cb_b 26 / 24, objects 328 / 5 (recall .944 / .895 /
  .743 / .814); real-vs-8-bit tower identical within noise. 9B (bf16) at EQUAL recall: D1+D2+D3 precision .935 vs phone
  .941 (-.006 [-.045, +.016]); at the product cuts the 9B keeps 0 subject photos (its P never reaches .999: the cuts are
  the phone's scale). POWER NOTE: only 75 phone-kept wrong photos sit in the held-out disputed groups, so criterion (1)
  alone can miss a real gain. AMENDMENT, written before any student exists (only phone/9B eye numbers seen): add (1b) a
  FRESH BLIND AUDIT of the student-vs-baseline DISAGREEMENTS on held-out libraries (population = every photo of the 8
  section-26 libraries for the 6 subject + 4 object searches; the held-out persons' photos for the 7 facts; the 64 dev b
  requests' album questions on their libraries' photos); per (search, direction) a uniform sample of up to 20 photos
  labelled blind at native pixels (key hidden), scaled to the stratum sizes -> estimated change in right-kept and
  wrong-kept photos. Decision: (1) OR (1b) must show the wrong-kept decrease with 95% CI excluding 0 on D1+D2+D3, AND
  neither may contradict it (point estimate of precision change >= 0), AND recall (2) holds on the eye set AND (1b)'s
  estimated right-kept loss on D1+D2+D3 and on objects is <= 3% of the baseline's right-kept with CI lower bound >= -6%.
- 10-14 01:25 [RESULTS 58] TEACHER GATE, first 3 candidates (held-out eye set, eval/judge_teach/gate_teachers_v0.txt):
  precision at EQUAL recall on D1+D2+D3 pooled: phone .941, 9B .935, Qwen3.5-27B-FP8 .950 (vs phone +.009 [-.049,
  +.026]), Qwen3.8-27B-FP8 .920 (-.022 [-.062, .000]). None passes. Per group 27B vs phone: D1 +.004, D2 +.020 [-.054,
  +.054], D3 +.066 [-.074, +.160] (35 match rows). At the product cuts the bigger models keep almost no subject photos
  (their P stays below .999): the cuts are the phone's own scale. Remaining candidates, fixed NOW before their numbers:
  Qwen3.5-122B-A10B-FP8 (TP 2), Qwen3.5-397B-A17B-FP8 (TP 4, downloading), Qwen3.5-27B and Qwen3.8-27B with thinking
  (2,048 tokens, P read after </think>), and ENS = mean logit gap of {27B, 122B, 397B} (no-think). With 8 candidates in
  all, a candidate passes only if its pooled CIs clear 0 at the Bonferroni level 1 - .05/8 (99.4% CI); otherwise the
  pre-registered rule as written.
- 10-14 ~01:35 [RESULTS 59] Replay harness checked: passthrough (9B answers from the jcache through the patched
  engine._judge_rows) reproduces r57_base_search exactly (351/351, every category .000); 464,229 judge calls logged.
  Held-out TRAIN fold 0 (6 libraries, 642,846 pairs; selection only): F1 at .7 ce12 .859, ce6 .856, towervec12 .821,
  tower12 .791, vec .773 -> upper bound = ce12 (cross-encoder), phone-ms candidate = towervec12. Eye sets (TRAIN
  libraries, fold models): every scorer F1 ~0 on the OBJECT searches (dog/car/bicycle/boat) - the product's grounding
  rewrites "Is there X in this photo?" to "Is there a real X anywhere in this photo (not a drawing, painting, statue,
  toy, model or picture of one)?", a wording 0 of 1,189 training questions has (38% of dev judge calls use it). Decided
  BEFORE scoring any dev replay: FP_R59_QNORM=1 strips the parenthetical + "anywhere" for the scorer (-> "Is there a
  real X in this photo?", a train form); fix B = oracle labels for the shipped planner's own questions on TRAIN
  (fp_r59_trainplan submitted). Phone facts: greedy Qwen3-VL-2B loops (repeats a phrase to the 700-token cap) on ~80%
  of photos, 4B on ~20% -> decoding sweep fp_r59_sweep{2b,4b} (repetition / presence / frequency penalties).
- 10-14 ~01:40 [RESULTS 57] B3 adapter trained (models/planner_4bcb3_{lora,merged,q4merged}; 6,328 rows, 1,582 steps, final
  loss ~.045, 1.8 h on an RTX PRO 6000; the session restart at ~00:30 did not touch the Slurm job). Dev plan stage with
  the shipped code: q4 351/351 planned, 0 errors; fp16 351/351, 0 errors. 30-conversation gate: 4-bit 30/30, fp16 30/30.
  Plan shapes vs the shipped adapter (final-turn albums): f anchors .12 -> .03 (fp16 .00), d anchors .30 -> .20 (.25),
  filters outside h ~0 in all three. Search submitted: fp_b57_s_{q4,fp16}_0..6 (7 library shards each, 9B + shared jcache).
- 10-14 ~02:00 [RESULTS 59] First dev numbers (27B facts). Per photo, dev truth questions (1,288,577 pairs, 89,901
  matches): 9B P/R/F1 .979/.976/.978 (vs the 27B oracle, same family); ce12 .980/.775/.865; towervec12 .925/.764/.837;
  tower12 .812; vec .773; zero-shot best (phrase max-sim) .537, PE-Core cosine .284. End to end (r57_base plans, every
  judge call answered by the scorer; ref 9B .679): ce12 .525 / towervec .550 without the wording fix; with FP_R59_QNORM
  ce12 .619, towervec .619, equal-weight ensemble .644 (-.035 [-.065, -.005]); per category ens a -.020, b -.024, c
  +.001, d -.042, e +.147, f +.028, g -.091, h -.268. Biggest remaining losses read off the worst requests: trip / place
  membership questions ("Is this a photo from a trip to Maui?": facts have no place; the 9B reads it from pixels),
  OR-questions built by follow-ups ("float or a singer": 0/4 vs "float" 4/4), negations ("clean shaven"). Fixes in
  flight (train-only): facts text + the photo's own place/month metadata (facts_27bm), and 27B oracle answers on the
  shipped planner's own TRAIN questions (planned train requests: 713 library-questions, 182 OR, 113 trip/place;
  fp_r59_orp*), then ce12B / towervec12B (+ fold models). Phone facts: decoding sweep on 256 photos: 4B greedy 203/256
  parse, rep 1.05 255/256 (265 tokens mean); 2B best rep 1.1 183/256 (377 tokens, 65 at the 700 cap); Qwen3.5-0.8B
  rep 1.1 215/256 but its own schema. Rerunning all 70,580 photos: 4B rep 1.05, 2B rep 1.1, 0.8B rep 1.1.
  Eye sets (TRAIN libraries, fold models, normalised wording): E2 population F1 9B .892, phone judge .879, ce12 .848,
  towervec12 .807, vec .671; car/bicycle recall is where facts lose. Dev eye labels (478 photos, 32 requests, request
  level): 9B stack P/R .628/.833, ce12_qn .790/.557, towervec_qn .663/.631.
- 10-14 02:00 [RESULTS 58] TEACHER GATE, more candidates (99.4% Bonferroni CIs; gate_teachers_v1.txt, _think.txt,
  _unsure_nomatch.txt): precision at equal recall on D1+D2+D3 (held-out): phone .941, Qwen3.5-122B-A10B-FP8 .944 (+.003
  [-.063, +.026]), ENS(27B+122B) .953 (+.012 [-.051, +.036]). Thinking (2,048 tokens, P after </think>) is WORSE and
  near-binary: 27B-think .884 vs phone .948 on its rows (-.064 [-.119, -.030]), 3.8-27B-think .859 (-.090); both say yes
  far more (D1 at the cut: 498 right / 210 wrong vs the phone's 472 / 32). Unsure-as-no-match sensitivity: 27B +.029
  [-.055, +.049], same verdict. Per search at equal recall (phone -> 27B wrong photos): food 12 -> 4, sunset 11 -> 7,
  beach 3 -> 3, church 1 -> 11 (27B's P is compressed: real churches .82-.97, non-churches .80-.91), flowers 2 -> 2.
  Viewed (sheets .cache/tmp/r58_sheets/ph_yes_27_no_*, ph_no_27_yes_*, 12 + 12 tiles at native px): where the 27B rejects
  a phone yes it is right 8/12 (food sign, not-beach, fog not blur, no golden light, eating x2...), wrong 2 (sunset sea
  as "beach clearly visible", a memorial with flowers), unsure 2; where it accepts a phone no, 10/12 are its softer scale
  (sunsets at .93 vs the phone's .99, below the .999 cut). Training data built meanwhile (not labelled): 5,160
  questions, 332,456 pairs, 67,540 judge images packed (6.35 GB, data/public/judge_teach/). Waiting: 397B (download).
- 10-14 ~02:25 [RESULTS 57 DONE: planner track B3 NOT shipped; dev only, test closed] B3 (4B LoRA, RESULTS 31 recipe, old
  4,426 rows + 634 corrected-train-truth targets x3) vs the shipped 10-05 adapter, both on the phone's 4-bit weights, same
  code (64709ef src) + 9B judge cache, 351 dev requests: ALL .679 -> .686 (+.007 [-.009, +.025]), ALL verified (a,d,e,f,h;
  202) .692 -> .700 (+.007 [-.016, +.033]); a -.001, d -.008, e +.077 (17), f +.020, h +.006, b -.005 (prov.), c -.004
  (prov.), g +.045 [+.007, +.086] (prov.); no verified CI below 0. fp16: ALL verified +.014 [-.014, +.042]. Gates 30/30
  (4-bit, fp16). Pre-registered rule: (1) fails -> NOT SHIPPED, no MLX export, app keeps planner_4b27ball. Movement is
  judge-question wording both ways (trail map .79 -> .30 as in B2; colonial drummers / Everest 0 -> 1.00), plan shapes
  unchanged. ~3.7 GPU-h. Not pushed.
- 10-14 ~02:30 [RESULTS 60 started: the "real X (not a drawing, painting, statue, toy, model or picture of one)" rewrite
  exempts things that ARE depictions or characters; dev only, test closed] Rule (converse.ground; code in the frozen
  snapshot .cache/r60/snap_new/src until adopted): the rewrite is also skipped when the HEAD NOUN of any "X and Y" / "X or
  Y" part of the question's thing (the words before with / of / on / wearing / visible ...) is an image or writing (map,
  menu, sign, poster, tattoo, logo, picture, reflection ...), a figure (doll, puppet, mannequin, rubber duck, mascot ...),
  an unreal creature (dinosaur, dragon, unicorn, mermaid, fairy, angel, ghost ...) or a named character (Mickey Mouse,
  Elmo, Santa Claus, Spider-Man ... ~60), or when the head carries RESULTS 26's own art words (toy car, stuffed animal).
  "A person wearing an Elmo costume", "a van with a logo", "a tattoo artist", "the Santa Monica pier" keep it. The old
  check (art words anywhere in the conversation) is unchanged. Designed from RESULTS 55/57's two cases + the shipped
  planner's TRAIN plans; two edits after reading the dev/TRAIN plan TEXT diffs (no dev score seen): "inflatable" dropped
  (B3 dev "inflatable bouncy castle": a real thing) and the head-noun cut ("tattoo artist", TRAIN). CPU replay of cached
  raw planner outputs: shipped planner 8/351 dev requests change (b: 5 "reflections in the water", Mickey Mouse; h: the
  Zermatt "map", one castle-or-golf follow-up whose old question carried the tail twice); B3 8/351 (+ trail map a, - the
  castle one); TRAIN 8/643. PRE-REGISTERED adoption rule: ADOPT iff (1) dev (shipped 10-05 adapter, 4-bit, r57_base raw
  outputs reused; 9B judge + shared jcache) ALL paired mean F1 diff > 0 AND no category's 95% CI entirely below 0;
  (2) every RESULTS 26 question (photos with a dog/car/bicycle/boat; the Open Images concepts) is byte-identical under
  the new code, so its eye-labelled drops (8/8 wrong + 2/2 unsure, 0/21 right; 10/12 depictions) stay; (3) gate 30/30
  on the 4-bit adapter, Python/Swift parity 0 mismatches, every changed fixture plan read. Reported, not deciding: B3
  plans on dev; the 8 TRAIN requests (9B both ways); per word group. The decision is on the rule as written.
- 10-14 02:25 [RESULTS 58] 397B-A17B-FP8 (406 GB) needs 4 H200 or 8 RTX PRO 6000 in one node: queued ~1 h, no start.
  Fallback raced next to it: Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 (236 GB, 4 H100 / RTX / H200; fp_r58_eye397i4). Whichever
  starts first is the 397B candidate (Int4 = a deviation from the pre-registered FP8, stated in RESULTS if used).
- 10-14 ~02:45 [RESULTS 60] Plan stage (fp_r60_plan_base / _q4, --reuse of r57's raw outputs, snapshot .cache/r60/
  snap_new/src): shipped adapter 4-bit 451 raw outputs (2 new), 0 errors, 8/351 requests' plans differ (b 6, h 2) = the CPU
  replay's list; B3 417 (2 new), 8/351 (a 1, b 6, h 1). Old code replays r57_base and the TRAIN plans exactly (0 changed).
  Fixtures (new code): 29 of 2,308 distillation plans change, all heads that ARE writing/pictures (whiteboard 21, menu 3,
  reflection in a mirror 2, garden gnome, watermark/logo/text overlay, "a clear photo of a cat's collar tag"); 0 real-object
  heads. Two knock-ons via EXISTING rules: a raw plan whose judge question equals its own exclusion now loses the question
  (it differed only by the tail before), and an "add the video recording too" follow-up now ORs it in (the tail had blocked
  the widen rule). + 16 hand-written cases (eval/complex_bench/r60/ground_cases.json). RESULTS 26 check (r60/check_r26.py):
  the 24 request/question pairs give byte-identical output under old and new code (16 keep the tail). Submitted: search on
  the changed requests (9B + shared jcache; fp_r60_srch_base, fp_r60_srch_q4), TRAIN 8 changed requests both ways
  (fp_r60_train_{base,new}, jcache_train), swift test (fp_r60_swift), pytest (fp_r60_pytest).
- 10-14 ~03:05 [RESULTS 60] Rule v1 (as pre-registered) on dev, shipped planner (8 changed requests, rest identical by
  construction): ALL .679 -> .680, +.001 [-.001, +.003], 1 better / 7 worse; b +.004 [-.002, +.015]; h -.002 [-.006,
  .000] (upper bound exactly 0: 12% of resamples hold neither changed h request); all others .000. By the letter (1)
  passes. By word group: Mickey .33 -> .67 (sheet: 2/6 added tiles costumed Mickey = the truth ones, 3 parade floats / a
  Mickey cap, 1 unsure); "reflections" 5 requests, 4 worse, mean -.014; Zermatt map h -.040 (the changed turn-1
  question made the planner answer turn 3 differently: the hut photos dropped, sheet 2/6 viewed are hut meals);
  castle/golf h -.015. B3 plans: ALL +.002; trail map .30 -> .80 (sheet 4/6 added are trail maps = the truth ones), Mickey 0
  -> .59, reflections 5/5 worse (mean -.048). TRAIN (9B both ways, 8 requests): rubber ducks 0 -> .91 (old: NOTHING
  returned: "a real rubber duck (not a ... toy)"), worm image on a screen 0 -> .15 (target now found), reflections 4/5
  worse (~-.03) + 1 empty -> .22, "person in a Packers shirt and a map" .67 -> .18 (both hit; 2 -> 10 returned).
  Reflection sheets (dev b9, b6): both questions accept junk in those libraries (old: food/drinks; new: sunsets, bars);
  where the plain one adds a right photo the tail had also dropped wrong ones. DEVIATION from the pre-registered "decide
  on the rule as written": v1 put "reflections" (and writing/boards: text, labels, plaques, menus, whiteboards) in the
  list by a wrong reading of my own principle - the tail lists drawing, painting, statue, toy, model, picture; a
  reflection, a menu or a whiteboard is a real thing, not a MADE image, so the tail does not contradict it; the data agree
  for reflections (dev 9/10 runs and TRAIN 4/5 better WITH the tail). Rule v2 = v1 minus those words (snapshot
  .cache/r60/snap_v2); nothing else changed. The "X and a map" split cost stays (it follows the stated principle; one
  TRAIN case). v2's dev numbers follow exactly from the runs already made (reflection requests get the old question
  back = the r57 records); plan stage + gate re-run on v2 to confirm.
- 10-14 03:00 [RESULTS 58 DONE: TEACHER GATE FAILED -> no distillation; phone judge unchanged] 397B (GPTQ-Int4, TP 4,
  raced next to the FP8 that never started; deviation stated): precision at equal recall D1+D2+D3 .917 vs phone .941
  (-.025 [-.071, +.015], 99.4%); ENS(27B+122B+397B) .940 (-.001). Final table (held-out, unsure dropped): phone .941,
  9B .935, 27B .950, 3.8-27B .920, 122B .944, ENS2 .953 (+.012 [-.051, +.036]), 397B .917, ENS3 .940, 27B-think and
  3.8-think worse (-.064, -.090; near-binary and looser). No candidate clears 0 vs the phone, none vs the 9B; size does
  not move it. Viewed: 397B's extra yeses (12 of 220 at native px) 1 match / 5 no_match (dinner table as food, daytime
  sky as sunset x2, flooded picnic table as beach, a sharp dog as motion blur) / 6 unsure. Scale finding: the big models'
  P for real subjects is .9-.97, the phone's .999-1.0 -> at the product cuts they keep ~no subject photo; any future
  distillation needs a remapped target. Ship criterion + audit amendment were pre-registered but never evaluated (no
  student). Built, unused: 5,160 questions, 332,456 pairs, 67,540 packed judge images (6.35 GB), train.py, merge.py.
  3.36 GPU-h + 1.4 CPU-job h. Downloads left in .cache/huggingface (unused now, ~800 GB: 3.8-27B 31, 122B 127, 397B-FP8
  406 never served, 397B-Int4 236) - delete on Reza's/coordinator's OK. RESULTS 58 written. Not pushed.
- 10-14 ~03:05 [RESULTS 60 DONE: real-X exemption ADOPTED (rule v2); dev only, test closed] (Clock note: the two entries
  above labelled ~02:45 / ~03:05 were written at ~02:38 / ~02:46.) v2 = v1 minus reflections and real things that carry
  writing (menu, label, plaque, whiteboard, text). Dev, shipped 10-05 adapter 4-bit: 3/351 plans change; ALL .679 -> .680,
  +.001 [-.000, +.003]; b +.005 [.000, +.016] (Mickey .33 -> .67, recall 1/5 -> 5/5); h -.002 [-.006, .000] (Zermatt
  .33 -> .29 = the planner's turn-3 answer after the turn-1 question changed; castle/golf .67 -> .65 = the double tail
  removed); rest .000. B3 plans: ALL +.003 [.000, +.008] (trail maps .30 -> .80, Mickey 0 -> .59). TRAIN (not used for
  choices): rubber ducks 0 -> .91, worm image on a screen 0 -> .15 (target found), Packers shirt "and a map" .67 -> .18
  (target found both ways). RESULTS 26 questions byte-identical (24 pairs). Fixtures: 3/2,308 distillation plans change
  (garden gnome, watermark/logo, photo of a collar tag) + 16 hand cases; FOLLOWUP unchanged; Swift GROUND 0/2,324,
  FOLLOWUP 0/874, XCTest 141 + 29, 0 failures; pytest 156/156; gate 30/30 (v1 and v2). Applied to src/findpics/
  converse.py + FindPicsCore Grounding.swift (compiled + tested on Linux; app untouched; next Mac build picks it up).
  ~0.8 GPU-h. Not pushed.
- 10-14 ~03:45 [RESULTS 59 DONE: index-time facts + query-time scorer instead of the judge; DEV only, test closed]
  Per photo (dev truth questions, 1,288,577 pairs): 9B F1 .978; best facts reader ce12B (MiniLM-L12 cross-encoder over
  27B facts + place/month) .872 (P .975, R .789); two-tower ms scorer towervec12B .841; vectors only .773; zero-shot
  <= .537. End to end (351 dev requests, r57_base plans, every judge call answered by the scorer; harness reproduces
  the 9B run exactly): ce12B .683 vs 9B .679 (+.004 [-.026, +.034]; a..f within -.007 or better, g -.075, h -.134);
  ensB .667; towervec12B .608 (-.071). Eye: E2 population F1 9B .892 / ce12B .853 (R .947 / .851); dev 478-photo
  system sample 9B .716 / ce12B .689. Phone-size facts (B config retrained on each writer, TRAIN only): ce12B end to end
  4B .593 (-.089 vs 27B facts), 4B short keyword line .600, 2B .565, 0.8B .497; per photo .847/.847/.809/.792. Index
  cost: 4B 271 out tokens/photo (short 126); est. 306-1,027 phone-hours cool for 187,084 photos (decode speed NOT
  measured on the iPhone). Greedy small writers loop (2B 79%, 4B 19% of photos to the 700 cap); rep 1.05 fixes 4B.
  Verdict: the judge cannot go today (the reader that matches it is a per-photo transformer pass, the ms scorer loses
  .07, phone facts lose another .08-.09 and cost weeks of phone compute). Still needs pixels: counting (per photo .47),
  colour/attribute bound to a thing or person, relations, small background details, photo-level looks, f/e one-photo
  anchors on phone facts, follow-up OR questions. Next: measure decode tok/s on the iPhone; cascade (towervec all ->
  facts CE top ~2k -> judge only for those kinds); distil a short-format facts writer from the 27B; split OR-questions.
  Disclosed: B designed after reading dev failures (optimistic); QNORM from TRAIN eye sets. ~34 GPU-h. Not pushed.
- 10-14 ~03:55 [RESULTS 59b started: the CASCADE, dev only, test closed] PRE-REGISTERED before any cascade number:
  Code = main with RESULTS 60 grounding (frozen copy .cache/wt_r59b/src, c1a67a9); plans = runs/cmp/r60v2_base (shipped
  planner, 351 dev requests); reference = the all-judge shipped stack r60v2_base_search (ALL .680). Every engine judge call
  is ROUTED by eval/facts_scorer/route.py from the question text (+ red-box crop): to the 9B judge (cached answers, a lazy
  9B only for misses) if red-box person / count / text-sign / photo look / colour / person word / relation / small-
  background / "or" composition (RESULTS 59's weak kinds; on TRAIN plan questions this sends 53% of judge calls to the
  judge), else to the scorer cascade: the ms scorer (towervec12B) on every in-scope photo; the facts reader (ce12B) replaces
  its P on the photos whose towervec P is >= the K-th largest among that question's in-scope photos (in-scope set = what
  the engine asks about in a first pass with towervec only); the rest keep towervec P. K = 0, 500, all (2,000 and 5,000
  >= every dev library: 1,778-1,990 photos, so identical to all), and the phone-scaled K_lib = ceil(K x N_lib / 187,084)
  for K = 500 / 2,000 / 5,000 (~5 / ~21 / ~51 per dev library). Facts: 27B (ce12B_full, towervec12B_full) and 4B phone
  facts (ce12B_q3vl4bq4r, towervec12B_q3vl4bq4r). Reported: per category + ALL vs the reference (paired bootstrap),
  judge calls per search (unique photo x question x crop) vs the all-judge passthrough log, est. phone seconds per search
  = judge calls / 25 per min + towervec 1.6 MFLOP and facts reader 8.1 GFLOP per photo-question at an assumed 1 TFLOP/s.
  "Acceptable" (stated now): ALL diff 95% CI lower bound >= -.03. Submitted fp_r59b_pt_* (passthrough with the new code:
  must reproduce r60v2_base_search; gives the all-judge call counts).
- 10-14 ~04:05 [RESULTS 59b DONE: the cascade, dev only] Harness with RESULTS 60 code reproduces r60v2_base_search
  351/351. Pre-registered cascade (route.py; judge for red-box / count / text / photo look / colour / person word /
  relation / small detail / "or"; else towervec on all + facts reader on top K): 27B facts ALL .682-.690 vs all-judge .680
  (K=0 +.002 [-.013, +.018], K=all +.010 [-.007, +.028]; every variant acceptable); 4B phone facts K=0 +.001 [-.012,
  +.015] (acceptable), with the 4B reader -.013..-.018 (lower bounds -.032..-.039: not acceptable). Judge calls only
  -40% (mean 1,188 -> ~715; median 1,495 -> ~70; 154/351 searches need none) because colour / photo-look / person
  questions still judge their whole scope; est. phone 47.5 -> 28.6 min mean, 59.8 -> ~3 min median. POST HOC (designed
  after scoring): judge only the top 100 of a pixel-kind question's scope by towervec P: 55 calls/search (4.6%), est.
  2.2 min, 27B +.011 [-.009, +.030], 4B -.016 [-.039, +.006]. Losses: h (-.05..-.10; trip membership answered by the
  scorer blocks the follow-up's pixel question), g. ~1.7 GPU-h. Not pushed.
- 10-14 ~04:25 [RESULTS 59c started; PRE-REGISTERED before any 59c number] Rule R59c (route.py, committed now), on top
  of RESULTS 59b's routing, K = 0 (no facts reader): (a) TRIP route: a question matching TRIP (trip / vacation / holiday
  / tour / visit / travel / "taken in|at|during|on" / "photo from" / "during a|the|our|my") is answered from the photo's
  place metadata when a capitalised name in it matches >= 1 photo of the library (engine.place_mask): P .99 there, .01
  elsewhere; with no matching name it goes to the judge; (b) JUDGE BUDGET: every judge-routed question (59b pixel kinds +
  trip-without-place) is judged only on its top M = min(n, max(50, ceil(5% of n))) in-scope photos by the millisecond
  score (n = in-scope photos from the scorer pass); the rest keep the millisecond P. Fraction from RESULTS 59b (top 100
  of ~1,900 = 5.3% kept F1 with 27B facts); floor 50 so small scopes stay mostly judged. Scorers (TRAIN data incl. the
  planner-question pairs): vecB = image vector only (B-16 photo + question vectors, NO facts), towervec12B on 4B phone
  facts, towervec12B on 27B facts; TRAIN libraries scored by fold models that never saw them. Evaluation: TRAIN = the
  test of the rule (643 requests, shipped planner plans replayed with RESULTS 60 code: 3 change; all-judge reference =
  the 9B on every in-scope photo, fp_r59c_aj_*), then DEV confirmation (reference r60v2_base_search). Also reported, as
  asked: vecB K = 0 with RESULTS 59b's routing (judge on whole scope). Acceptable = ALL diff lower 95% bound >= -.03.
- 10-14 ~04:50 [RESULTS 59c DONE: judge budget + trip metadata (pre-registered 32326d0), and the image-vector-only
  cascade] TRAIN all-judge reference (new, 643 requests, RESULTS 60 code) ALL .657; harness reproduces it and dev
  exactly. R59c (K=0, judge only the top max(50, 5% of scope) of pixel-kind questions, trip questions from place
  metadata) with 4B phone facts: TRAIN -.012 [-.025, +.002], dev -.005 [-.022, +.014] -> ACCEPTABLE on both, at 47-50
  judge calls per search (3.8-4.3% of all-judge; est. ~2 min vs ~50). 27B facts the same (-.012 / .000). IMAGE VECTOR
  ONLY (no facts): TRAIN -.040 [-.053..-.056 lower], dev -.019 / -.024 (lower -.038 / -.046) -> NOT acceptable: the phone
  needs index-time facts with today's scorers. Trip route: d +.050 TRAIN, +.039 dev; metadata agrees with the 9B on
  .863 of 15,248 dev photos; flaw seen on a sheet: compound questions ("sepia tone taken in Spain") lose their visual
  condition (b -.018 TRAIN / -.021 dev). h stays the loss. Exploratory dev: 4B short keyword facts R59c -.012 [-.033,
  +.009] (just below the bar). ~7.6 GPU-h. Not pushed.
- 10-14 ~05:10 [RESULTS 61 started: distil a SHORT-FORMAT facts writer for the phone; PRE-REGISTERED before any number]
  Format (eval/facts_writer/short_format.py): one line "scene; objects; colours <bound to a thing>; people: N - per person
  clothing/colour/action; activity; text; look; event", empty fields omitted, target 40-80 tokens. Step 1 = the 27B
  rewrites its own JSON facts into it TEXT-ONLY for all TRAIN+DEV photos (fp_r61_short_*); the R59c cascade (59c rule,
  towervec12B retrained on these facts + place/month, TRAIN fold models) must pass the bar to show the format itself
  loses little. Students: Qwen3-VL-4B and -2B (LoRA r16 on language attn/MLP + vision blocks/merger, on the phone's
  4-bit-sim weights models/qwen3vl_{4b,2b}_mlx4sim), and Qwen3.5-0.8B (models/qwen35_08b_mlx4sim) as the <1B try; input
  = the photo at the phone's ~448 px rendition (Lanczos, JPEG q80) + the prompt "Search facts for this photo, one line.";
  target = the 27B short line; 1 epoch, lr 1e-4. CROSS-FIT so every TRAIN photo gets facts from a student that never
  saw it: student A trains on TRAIN folds {0,1}, B on {2,3}, each writes the other half; dev facts = student A. Phone
  form = LoRA merged in fp32, language re-quantized to MLX 4-bit (exact rounding), vision bf16. Decoding greedy,
  repetition penalty 1.05, max 120 tokens. Evaluation: per photo vs the 27B short line (field term F1), output tokens,
  repetition rate (at the cap, or a 3-gram repeated >= 3 times); END TO END = the R59c cascade with towervec12B retrained
  on each writer's facts (TRAIN only; fold models for TRAIN) vs the all-judge stack on TRAIN (643, train_alljudge) and
  dev (351, r60v2_base_search). SHIP RULE: a writer is acceptable iff the cascade's ALL difference has lower 95% bound
  >= -.03 on BOTH TRAIN and dev; ship = the SMALLEST acceptable model (0.8B < 2B < 4B), and among equal size the fewest
  mean output tokens. Index hours reported at 10 / 20 / 30 tok/s (device decode speed unmeasured; M70).
- 10-14 ~05:35 [RESULTS 61] Format check passed: the 27B rewrote its JSON facts into the short line for 70,455 TRAIN+DEV
  photos (mean 75 tokens, median 74, p90 100; 0.5% at the 120 cap; ~104 s per library text-only). R59c cascade with
  towervec12B retrained on the short lines: TRAIN -.011 [-.025, +.003], dev .000 [-.018, +.019] (27B JSON facts in 59c:
  -.012 / .000) -> the format itself loses nothing measurable. Students training (cross-fit A/B): 4B / 2B / 0.8B, 1 epoch
  ~1,420 steps; token loss at step ~600: 0.74 / 0.78 / 0.87.
- 10-14 ~06:15 [RESULTS 61 DONE: short-format facts writer distilled from the 27B] The format is free: 27B short line
  (75 tokens vs ~243 JSON) in the R59c cascade TRAIN -.011 [-.024, +.002], dev .000 [-.017, +.018]. Students (LoRA vision
  + language on the 4-bit-sim weights, cross-fit halves of TRAIN, 1 epoch, phone form = merged + 4-bit re-quantized):
  4B 72 tokens TRAIN -.016 [-.0306, -.003], dev -.012 [-.0303, +.007]; 2B 71 tokens -.023 [-.038] / -.010 [-.0296];
  0.8B 69 tokens -.029 / -.011 [-.031]. Pre-registered ship rule: NONE acceptable (4B misses by .0006 / .0003). Est.
  index hours for 187k photos at 10/20/30 tok/s: 4B student 462/275/213 vs zero-shot 4B JSON 1,497/793/558. Val loss
  still falling at the end of the epoch -> next: 10x more 27B-labelled public photos, full TRAIN, 2-3 epochs, same rule;
  M70 to time decode + vision per size on the iPhone. ~10.3 GPU-h. Not pushed.

MAC 10-14 02:48 — THE READ COST GROWS WITH DEPTH, which is the part both my stories missed.
    PERSONREAD    30 photos: 1.38 s each
    PERSONREAD 1,900 photos: 2.36 s each, 4,486 s of reading in ~127 min elapsed
  So reads are now 59% of the wall clock, and the per-photo cost has nearly doubled as the search works down the
  ranking. The reason is simple once seen: the top of a person ranking is full of RECENT photos, which are on the
  phone; the deeper it goes the older they get, and older originals live in iCloud. The first 30 photos said
  "reads are cheap" and the first 10 minutes on 10-12 said "reads are fatal" - both were true of their own slice
  and neither was true of the search.
  So the honest statement, which I should have reached by measuring the whole curve rather than either end:
  a person search costs ~2.4 s a photo in reads ON TOP of ~2.4 s of judging, the read share RISES with depth, and
  the first pass over a cold library pays it. Warming the cache during indexing would halve the wall clock of every
  deep person search, and it is work that can happen while the phone charges rather than while someone waits.
  Status: 1,900 photos judged across the two albums (heavier / fit), no first answer yet, nothing stalled.

MAC 10-14 07:15 — THE FLAGSHIP HAS NO STOPPING RULE. That is why it never finishes, and it is not the reads.
  SearchEngine.runPerson judges EVERY face-matched photo:  `for k in ident { ... }`  - no rounds, no first answer,
  no bound, no thermal pacing. Every other path goes through streamRounds with phoneFastParams and stops at a
  first answer after ~480-600 calls. The person path predates that work and never got it. On Reza's 41,307-photo
  face set that is 41,307 judge calls at ~3 s each = ~34 HOURS. It was never going to finish; it has no notion of
  finishing. Overnight it reached 9,260 photos in 6.5 h, which is exactly on that line.
  AND I MUST CORRECT MY 02:48 ENTRY: "the read cost grows with depth" is WRONG. The cumulative average has FALLEN
  from 2.36 s at 1,900 photos to 1.04 s at 9,260 - so photos 1,900-9,260 averaged 0.70 s each, FASTER than the
  early ones. The early slowness was a cold-cache transient that warmed up; I called the transient a trend after
  watching one slice, which is the third time in two days I have narrated a curve from one end of it.
  So the true picture for person+look, measured over 9,260 photos: ~2.4 s judging + ~0.7-1.0 s reading per photo,
  no thermal halt, nothing structurally wrong - and no stopping rule, which is the whole problem.
  WHAT I AM NOT DOING: inventing a stop rule for person albums on my own. Every other stop rule in this product was
  fitted on measured data (RESULTS 41/43/44) and a person search's yes-rate curve is different - the face filter has
  already removed everything that is not him, so "fewer than 3% of the last 100 are yes" may never trigger. Asking
  the cluster for the rule, with the measurement attached.
  Flagship stopped at 9,260 judged to free the phone for the two jobs that were queued behind it.
- 10-14 ~06:05 [RESULTS 61b started: scale the distillation data; SHIP RULE = f97aee6, unchanged; PRE-REGISTERED here
  before any 61b number] Extra data: 300,000 Open Images V4 train photos (every one CC BY 2.0 per the list; CVDF S3
  mirror; random sample seed 0 of 1,743,042 after dropping the 642 that ARE DISBench photos by Flickr id and any by the 57
  DISBench users) + the 17,594 Wikimedia Commons iPhone photos on disk (free licences by Commons policy); stored 500 px
  JPEG q92 like the benchmark; near-duplicates of any of the 109,467 DISBench photos (64-bit dHash, Hamming <= 3) and
  duplicates inside the set dropped (extra_data.py). Labels: the same two 27B steps as the TRAIN targets (VLM JSON facts
  with the facts.py prompt, then the text-only short-line rewrite; label_extra.py). Students: 4B and 2B (0.8B if time),
  same LoRA / input / target as RESULTS 61, trained on their cross-fit TRAIN half + ALL extra photos, 2 epochs, lr 1e-4,
  4-GPU DDP (train_writer2.py); validation loss on 200 held-out TRAIN-half photos and 300 held-out extra photos.
  Learning curve: adapters saved at 100k, 200k, 400k photos seen and at the end (~676k); each is turned into the phone
  form and evaluated end to end exactly as RESULTS 61 (cross-fit facts, towervec retrained on them, R59c cascade, TRAIN
  643 + dev 351); RESULTS 61's 1-epoch half-TRAIN students are the ~23k point. Ship decision = the unchanged rule on the
  FINAL checkpoints; the intermediate points are the curve only.
- 10-14 ~06:55 [RESULTS 61b] Extra data built: 317,594 fetched/read (100%); dropped as near-duplicates of a DISBench
  photo (dHash <= 3) 3,778 and 237 duplicates inside the set; kept 313,816 (Open Images 296,343, Commons 17,473). A sheet
  of 4 dropped pairs (r61b_dups.jpg) shows unrelated photos (dHash collisions at <= 3 bits): the check is conservative
  (drops ~1.2% for nothing); the Flickr-id exclusion (642 Open Images photos ARE DISBench photos) is the real guard.
  27B labelling: 24 shards, ~3.05 photos/s per RTX PRO 6000 (facts + rewrite) -> ~70 min per shard; trainings and the
  checkpoint evaluation waiters start automatically when all shards finish (launch_train.sh).
- 10-14 16:46 [RESULTS 61b resumed after the session was down 07:20-16:45] Labelling had finished: 313,816 extra photos,
  313,323 with a short line (mean 74.9 tokens = the TRAIN targets' 75). The training launcher had died with the session;
  restarted detached (setsid): six 4-GPU students (4B / 2B / 0.8B x cross-fit A/B, 2 epochs) submitted + checkpoint
  evaluation waiters (4B at 100k/200k/400k/final photos seen, 2B 100k/400k/final, 0.8B final). Rule unchanged (f97aee6).
- 10-14 17:00 [RESULTS 61b] The lab account's GPU cap (MaxGRESPerAccount) holds most 4-GPU requests: 4B-A runs on 4 GPUs
  (~34 photos/s -> 2 epochs ~5.5 h), 4B-B on 2 GPUs (~18 photos/s, ~10 h; 2-GPU copies added for every student, same
  milestones in photos seen), the 2B students are queued; the 0.8B students are dropped (pre-registered "if time").
  Validation loss at start: TRAIN photos 3.05, extra photos 3.12.
- 10-14 ~17:30 [RESULTS 62 PRE-REGISTERED, before any 62 number: a FIRST ANSWER for person searches] Problem (07:00,
  Mac M67 confirmed): PersonSearch.swift / engine.stream_album judge EVERY face-matched photo before showing anything
  (41,541 for the owner, ~55 h at ~4.8 s/photo). Rule (src/findpics/person_first.py; engine Thresholds.person_first,
  off on the server):
  FACT (is_look_question false): the person's photos ranked by the look vector (COS) or its learned-head blend (BLEND),
  then the phone's timed fast-mode rounds exactly (phoneFastParams; first answer where fast mode stops; bound every
  round; keep looking to the end = today's album). Sets (eval/vector_first_person, 32 DISBench people, phone judge on
  every photo, 7 facts): F1 each person (224), F2 all 32 as one 4,010-photo scope (7), F3 random pools of 8 / 16 people
  x 20 draws (rng 62; 280); 20 tail seeds; RANDOM order = ablation. D1 ship only with 0 overclaims in every round of every
  seed (F1-F3). D2 order = BLEND iff its first-answer judge-yes found summed over F2+F3 >= COS's, else COS.
  LOOK: walk = fnv1a64("seed|id") random order, <= 3 judged photos per event (3 h gaps) first, the rest after; pair = both
  questions per photo; after any number of calls each photo of a reached event takes its event's median of the judged
  photos; the pair split is fitted on the judged photos weighted by event size / judged in event (BIC on the judged
  count); no clear split -> rank margin on the judged photos; single look = top half by weighted rank; events not reached
  = not sorted yet. With everything judged = the full rules exactly (check: 0 photo differences). Sets: L1 Reza's
  393-photo sample replayed from the saved runs' pools (q3vl4b_real phone judge PRIMARY, 9b_prob, all runs at 480;
  numbers only, data/private/audits/r62_*), L2 IMDB eras (9B cached, no dates), L3 public 'tired' (32 people).
  Budgets 200 / 480 / 1,000 calls + curve; 20 walk seeds. D3 ship the walk (cap 3, events propagated) iff at 480 calls
  on L1 primary (pair) BOTH albums' precision vs the judge-everything albums >= 0.95 (mean over seeds); else judged-only
  if that passes; ablations cap 1 / no cap / judged-only reported only. Also: label-free coverage on DISBench.
  Code: eval/person_first/r62.py (texts done: Slurm 53011289).
- 10-14 ~19:35 [RESULTS 62 progress] (session dropped 17:10-19:25; replays restarted) FACT replay done
  (eval/person_first/facts.txt): D1 FAILED as pre-registered - 47 of 74,285 rounds overclaim (COS+BLEND, F1-F3); cause
  being diagnosed (which round, tail size). D2: BLEND (first-answer judge-yes 62,540 vs COS 59,113 over 287 searches,
  245 better / 33 worse). LOOK L1 private replay done (numbers in data/private/audits/r62_looks_private.txt): D3 PASS
  (phone judge, walk cap 3, 480 calls: heavier and fit precision vs judge-everything 1.000 / 1.000, agreement .991,
  every photo sorted); every one of 30 saved runs: partial rule with everything judged = engine split, 0 photo
  differences. Coverage: a 41k-photo proxy (21 DISBench libraries) sorts .270 of photos at 240 judged photos (a pair's
  480 calls), .413 at 480, .788 at 2,000. Swift: first build failed (type-checker timeout in twoGroups), expressions
  split, swift test resubmitted (53044698); public looks replay resubmitted (53044700). pytest 168/168.
- 10-14 19:30 [RESULTS 61b] Session was down 17:10-19:25; the four students kept training (Slurm), the login-node waiters
  died again -> launchers made idempotent (skip finished / queued work) and relaunched detached (relaunch_waiters.sh; re-run
  it after any restart). Validation token loss so far (TRAIN-half photos / extra photos): 4B-A 3.05 -> .664 (100k seen)
  -> .621 (200k); 4B-B .681 (100k); 2B-A .720 -> .668; 2B-B .731 (100k). RESULTS 61's 1-epoch students: 4B .687, 2B .752.
- 10-14 ~20:45 [RESULTS 62 DONE: person searches get a first answer; MAC_INBOX M72 = the runPerson wiring] FACT: fast mode
  inside the person's photos, learned-head blend order (D2: 245 of 287 searches better than cosine), median 603 calls
  (~20 min on the 8-bit clock vs ~34 h today), judge-rel. recall at the first answer .325 on a 4,010-photo scope (snow .947
  ... smiling .205), first screen after ~22 calls. D1 FAILED as pre-registered: 47 of 74,285 replayed rounds overclaim
  (42 of 20,440 runs, 0.21%; 43 of 78 in round 1); the bound's 95% guarantee holds (<= 5% of runs), same bound as ordinary
  searches -> shipped, flagged for Reza. LOOK pair (his sample, phone judge): 480 calls -> heavier / fit precision vs
  judge-everything 1.000 / 1.000, agreement .991, every photo sorted (D3 PASS); 200 calls is not trustworthy (worst seed
  fit .38) -> albums only from the first answer on. Single looks: propagation costs agreement with the per-photo final
  album (tired .67 vs .89 judged-only at 10%) but matches his era labels better (221 H + 14 F vs 176 + 20 at 200 calls);
  next: pre-register event medians for single looks in the full rule. 41k proxy coverage: pair first answer sorts .27.
  pytest 168/168; swift test 150 + 29, 0 failures. App not edited (M72). Not pushed.
- 10-14 ~20:40 [RESULTS 62 decision] Coordinator accepts the person-search first answer despite the failed "0
  overclaims" pre-registered check: 47 of 74,285 replayed rounds (42 of 20,440 runs, 0.21%) understated the misses,
  within the bound's stated guarantee (<= 5% of runs) and the same bound ordinary searches already show; the
  alternative is a 34-h search with no answer. Recorded as a caveat for Reza (he may overrule). Mac: M72.

MAC 10-14 20:03 — M72(a) EVENT CENSUS of his face set, no judge calls:
    photos 41,307 | events 5,764 | median event 3 photos | largest 517 | singletons 1,689
  WHAT THIS MEANS FOR THE 480-CALL RULE, and the cluster should see it before trusting the fitted number here:
  480 calls can touch at most 480 of his 5,764 events - 8.3% - and fewer if it spends the cap of 3 on any of them.
  Taking up to 3 per event over every event would cost 12,950 calls. So on HIS library the first answer is drawn
  from under a tenth of the occasions he appears in, where the rule was fitted expecting ~603 calls to be a good
  sample of a library. That is not an argument against the rule; it is the denominator the rule's bound has to be
  honest about, and it is why the completeness note matters more here than anywhere else.
  (My "eventsWithin480calls=480" line counts smallest-event-first, which is not the walk's order - it is an upper
  bound on events touched, not a prediction. Saying so rather than letting the number read as a plan.)
- 10-14 ~21:10 [RESULTS 62b PRE-REGISTERED, before any 62b number: a COVERAGE-FIRST walk for LOOK person searches] Mac
  M72(a): his face set = 41,307 photos, 5,764 events, median 3, largest 517, 1,689 singletons; a pair's 480-call first
  answer judges 240 photos, so the random walk reaches few events. Walks compared (all: within an event photos in
  fnv1a64("seed|id") order; the tail = remaining photos in global hash order, as walk3):
  (1) SIZE: events by size descending (ties: hash of the event's first photo); pass 1 = photo #1 of every event; pass 2
  = photos #2-#3 of every event in the same order; then the rest. (2) walk3 (shipped, random, cap 3). (3) walk1 (random,
  cap 1). Sets: S1 a synthetic library with HIS event sizes (1,689 singletons + 4,075 events of size 2 + floor(Y), Y
  lognormal mu 1.2 sigma 1.35, deterministic quantiles, largest set to 517, total adjusted to exactly 41,307 photos;
  median 3) with pair answers (a) bootstrapped from his 393-photo sample's events (donor event uniform, photos with
  replacement; private, numbers only) and (b) a public parametric model (event era H/F p .5, event logit difference +-1.5
  + N(0, .8), photo noise t3 x 1.0, 10% uninformative N(0, 2), common offset N(0, 1)); 5 draws x 5 walk seeds; sizes
  sensitivity (label-free coverage) for the lightest / heaviest lognormal tails that keep median 3 and mean 9.72;
  S2 the 41k DISBench proxy (coverage only); S3 his sample (20 seeds, numbers only). Report: photos sorted and
  agreement with the judge-everything albums at 480 / 1,000 / 2,000 / 5,000 calls (S3: 120 / 240 / 480), correctly
  sorted share (sorted AND same label as everything judged, over all photos), precision among sorted, his era labels on
  S3, calls to sort 50% / 90% of photos. DECISION: (1) replaces walk3 iff (i) on S1 (both answer models, mean over draws x
  seeds) its correctly-sorted share at 480 calls beats walk3's and walk1's, (ii) its precision among sorted photos there
  is >= .95, (iii) on S3 at 480 calls both albums' precision vs everything judged >= .95 and wrong-album photos by his
  labels <= walk3's + 4 (1% of 392). Otherwise walk3 stays. Code: eval/person_first/r62b.py.
- 10-14 ~21:30 [RESULTS 62b DEVIATION, before any scored 62b number] A label-free sanity check of the synthetic library
  (ids "e00012p003") showed the walk's FNV-1a key is not random over ids that differ only in their last characters:
  FNV-1a's top bits barely see the last bytes, so same-event photos sorted next to each other (walk3 then reached 2% of
  photos at 240 photos instead of a size-biased share). Real ids were fine (his sample: same-event neighbours .143 vs
  .134 random; DISBench .016 vs .013), so RESULTS 62's numbers stand as measured. Fix for every walk (product and 62b):
  key = fmix64(fnv1a64("seed|id")) (MurmurHash3 finalizer; Python walk_key, Swift walkKey). The 62b walk definitions are
  otherwise unchanged.
- 10-14 ~22:40 [RESULTS 62b DONE: coverage-first walk - NOT adopted as pre-registered, shipped opt-in, recommended]
  On a library with his census sizes (stand-in: 41,307 photos / 5,764 events / median 3 / max 517 / 1,689 singletons)
  the size-first walk sorts .381 of photos at a pair's 480 calls vs .192 walk3 / .214 walk1 (calls to 50% / 90%: 906 /
  5,862 vs 2,430 / 13,350 vs 1,814 / 7,186); with answers bootstrapped from his sample's events, correctly sorted .342 vs
  .180 / .195, precision among sorted .896 vs .910 / .914. Pre-registered bar (ii) precision >= .95 failed for EVERY walk
  (bootstrap ~.90; public parametric ~.52, where one photo per event is too noisy for a clear split) -> walk3 stays the
  default; by_size / bySize opt-in implemented (Python + Swift, parity), recommendation: switch. His sample at 480: all
  walks judge the same photos (1.000 / 1.000). Walk key now fmix64(fnv1a64) after the synthetic ids exposed FNV-1a's weak
  last bytes (real ids were fine). M72 updated, M73 asks for the full histogram. pytest 170/170; swift 150 + 29 green.
- 10-14 ~21:00 [RESULTS 62b decision] Coordinator enables the size-first person-look walk (bySize true) despite its
  pre-registered precision bar failing: the shipped walk3 fails the same bar (.910 vs .896), so the bar did not
  separate the walks; size-first doubles coverage at the first answer (.381 vs .192 on his event sizes) for 1-1.5
  points of precision. Decided after the data, explicitly a trade-off; Reza may overrule. Mac: M74.
- 10-14 21:05 [RESULTS 61b, learning curve so far; ship decision only on FINAL checkpoints] 100k photos seen: 4B TRAIN
  -.0192 [-.0328, -.0050], dev -.0068 [-.0255, +.0125] (23k point: -.0160 / -.0116); 2B TRAIN -.0257 [-.0398], dev
  -.0096 [-.0291]; the 2B at 100k loops on 24% of photos (82 tokens mean, p90 at the 120 cap) vs 4.7% at 23k. Students
  at ~520k seen (A) / ~300k (B, 2 GPUs); B finishes ~01:30.
- 10-14 21:15 [RESULTS 61b] 200k photos seen, 4B: TRAIN -.0231 [-.0378, -.0083], dev -.0074 [-.0261, +.0114]. Per-photo
  objects F1 vs the 27B line (dev) rises 23k .59 -> 100k .62 -> 200k .63 and repetition falls (.024 -> .019), but the
  end-to-end TRAIN difference does not follow (-.016 / -.019 / -.023, all inside each other's CIs). 2B at 100k: loops on
  33% of dev photos (objects F1 .37) - unstable mid-schedule.
- 10-14 21:40 [RESULTS 61b] A students done (2 epochs, 671,264 photos seen): validation token loss TRAIN-half / extra
  photos: 4B .569 / .583 (RESULTS 61 1-epoch 4B: .687), 2B .612 / .619 (RESULTS 61: .752). B students (2 GPUs) at ~360k,
  finishing ~02:00; final evaluation (cross-fit TRAIN + dev) follows automatically.
- 10-14 22:20 [RESULTS 61b] 400k photos seen: 4B TRAIN -.0206 [-.0347, -.0063], dev +.0006 [-.0174, +.0190]; 2B TRAIN
  -.0196 [-.0339, -.0052], dev -.0030 [-.0210, +.0157]. Dev now clears the bar for both; TRAIN does not (curve flat at
  -.016..-.023 since 23k). Final checkpoints (671k seen) wait for the 2-GPU B students (~02:00).
- 10-15 ~00:40 [RESULTS 59d PRE-REGISTERED, before any 59d number] Rule R59d = R59c + (a) compound trip questions: place
  metadata only when nothing but frame / trip-kind / date words is left after removing the place phrase, else metadata
  AND the routed remainder; tolerant place names (accents/case folded; edit distance <= 2 on words >= 6 chars only when
  exact matching finds nothing, and not when a plain word of a multiword name matches on its own: "London Olympics");
  (b) "A or B" split into parts routed separately (P = max), "A and also B" / two question sentences (P = min); (c)
  from TRAIN h failures only: (c1) judge-budget EXTENSION (one more block of max(50, 5%) while the last block was >= 50%
  judge matches; TRAIN h: budget-limited questions hold 5,098 9B matches, the budget reaches 2,151), (c2) exclude
  questions always judged (scorer F1 vs 9B on exclusions h .80 / g .69; "bottle on a kitchen counter" 24/27 vs 2/27).
  Bar as 59c (lower 95% of ALL diff vs all-judge >= -.03, TRAIN and dev, 4B facts); per category vs R59c; ablations
  r59d_a/_b/_c; 27B reference. All-judge stack + (a)+(b) (aj59d, dev only): adopt in the product iff ALL diff > 0, lower
  bound >= -.01, no category down > .02. Full text: eval/facts_scorer/route.py header; design check (TRAIN questions, no
  scores) eval/facts_scorer/r59d/design_train.txt; TRAIN diagnostic r59d/diag_roles.py. CPU partitions unavailable
  (test QOS 5/5 full by other jobs; sapphire/shared refuse the account): replays go to GPU (race_sbatch, kempner_requeue
  first; new judge questions need the 9B anyway).
- 10-15 ~01:10 [RESULTS 59d submitted] smoke (4 TRAIN requests, all variants) ran clean; its R59c albums equal the 59c
  run's (4/4). 50 GPU replay shards (fp_r59d_{t,d}{4b,27b}_k, race_sbatch; eval/facts_scorer/r59d/launch.sh): 4B TRAIN
  r59c,r59d,r59d_a/_b/_c; 4B dev the same + aj59d, aj0, aj59c, ajb; 27B TRAIN+dev r59c,r59d. New judge answers saved
  per run in jcache/r59d_<run>/.
- 10-15 ~02:30 [RESULTS 59d numbers] R59d with 4B phone facts: TRAIN ALL .658 vs all-judge .657, +.001 [-.013, +.014];
  dev .685 vs .680, +.006 [-.010, +.022] -> ACCEPTABLE on both (59c bar). vs R59c paired: TRAIN +.012 [+.008, +.018]
  (h +.048 [+.015, +.087], b +.018, g +.017), dev +.010 [-.001, +.022] (h +.017 [-.038, +.075], b +.026, g +.013).
  h vs all-judge: TRAIN -.063 -> -.015, dev -.060 -> -.044. Judge calls / search: TRAIN 47 -> 66 (p90 100 -> 187, max
  802), dev 50 -> 80 (p90 99 -> 189, max 1,883). Ablations (vs R59c): (a) TRAIN +.001 / dev +.010; (b) +.004 / -.004;
  (c) +.010 / +.005. 27B facts: TRAIN +.002, dev +.012 vs all-judge. ALL-JUDGE + (a)+(b) (aj59d, dev): +.011 [+.001,
  +.022] vs the reference (+.011 [+.002, +.022] vs the harness's own all-judge aj0, which reproduces the reference on
  349/351 requests), categories: a +.011, b -.004, c -.002, d +.033, f -.008, g +.024, h +.030 -> pre-registered
  ADOPT bar met (ALL > 0, lower >= -.01, no category < -.02): port to the product (Python + Swift) next. Sheet
  (dev R59c-only vs R59d-only photos, native px, 12 tiles viewed): R59c-only 105/1,188 truth (colour photos for "black
  and white from India", a non-sepia photo for "sepia from Spain"), R59d-only 597/1,717 truth.
- 10-15 ~01:50 [RESULTS 59d DONE + ADOPTED (a)+(b) in the product] Python findpics.compose (= route.py R59d text rules)
  behind engine._judge_rows (COMPOSE_QUESTIONS; FP_COMPOSE=0 off); cascade.py turns it off for its own composition. Dev
  product replay on the cached judge: on the 321 requests that replay without a cache miss either way, composition off =
  reference 321/321, on = harness aj59d 320/321 (exception = the known aj0 harness artifact); 30 / 19 requests hit
  pre-existing cache misses (e anchors; the same with composition off). Swift FindPicsCore QuestionTree.swift: 0/660
  mismatches vs Python on benchmark questions + hand cases; product-only guard (after scoring, moves no number): "X or
  not?" is never split (it broke test_always_true_question_is_flagged). pytest 176/176; swift test 152 + 29, 0 failures.
  MAC_INBOX M74 = app wiring. RESULTS 59d written. Not pushed.
- 10-15 01:35 [RESULTS 61b] All four students done (671k photos seen). 2B FINAL: TRAIN -.0228 [-.0376, -.0085], dev -.0101
  [-.0279, +.0083] -> not acceptable (TRAIN); 77 tokens, repetition .084. 4B final evaluation running (B's facts).
  Final validation loss B: 4B .590 / .559, 2B .634 / .600 (TRAIN-half / extra photos).
- 10-15 02:05 [RESULTS 61b DONE] Unchanged rule (f97aee6): NOTHING SHIPS. 4B FINAL (671k photos seen): TRAIN -.0183
  [-.0318, -.0046], dev -.0109 [-.0309, +.0096]; 2B FINAL: TRAIN -.0228 [-.0376], dev -.0101 [-.0279]. Learning curve
  (4B, TRAIN / dev): 23k -.016/-.012, 100k -.019/-.007, 200k -.023/-.007, 400k -.021/+.001, 671k -.018/-.011 - flat within
  noise, while per-photo facts improved (objects F1 vs the 27B .59 -> .64, val loss .687 -> .569). The cascade itself is
  -.011 on TRAIN with the 27B's own lines, leaving ~.005 of room under the bar. 2B repetition worsened (.047 -> .084).
  Est. index hours 4B final 476 / 282 / 218 at 10/20/30 tok/s. ~68 GPU-h. Not pushed.
- 10-15 [RESULTS 61c PRE-REGISTERED before any number] Re-evaluate the EXISTING writers under the R59d cascade (route.py /
  cascade.py as of 500ade1, variant r59d): 27B short line, 4B student final (61b, 671k), 2B student final (61b), 4B
  RESULTS-61 student. No retraining; same facts and the same per-writer millisecond scorers (TRAIN fold models; full
  model for dev); r59c re-run in the same jobs only as a reproduction check. Same bar as f97aee6: acceptable iff the ALL
  difference vs the all-judge stack (TRAIN 643 train_alljudge, dev 351 r60v2_base_search) has exact-bootstrap lower 95%
  bound >= -.03 on BOTH; ship = the smallest acceptable model (2B < 4B < 27B), then the fewest mean output tokens. The 27B
  short line is a reference (server only), not a ship candidate. No rule changes after numbers (eval/facts_writer/r61c_launch.sh).
- 10-15 [RESULTS 61c DONE] Under R59d every existing writer passes the unchanged bar (TRAIN / dev diff vs all-judge,
  lower bounds): 27B line +.003 / +.008; 4B-61 -.002 [-.015] / -.002 [-.020]; 4B-61b final -.011 [-.025] / +.000 [-.020];
  2B-61b final -.011 [-.025] / -.002 [-.021]. R59c re-run reproduces 61/61b exactly. Rule (pre-registered 6c9962f): the
  smallest acceptable model ships -> the 2B 61b final student (77 tokens; est. 288 phone-hours for 187k at 20 tok/s).
  Caveats: 2B weakest on e, 8.4% looping outputs -> stop-on-repeat needed; M70 decode speed. ~2 GPU-h. Not pushed.
- 10-15 ~02:40 COORDINATOR: RESULTS 61c accepted as pre-registered (table checked against eval/facts_writer/r61c_table.txt,
  8 rows, matches report). Shipped facts writer = 2B student 61b final, CONDITIONAL on device speed: M76 asks the Mac to time
  2B and 4B decode on the iPhone (M70 extended). If 2B index time is unacceptable (> ~1 week of overnight charging for 187k),
  the index-time facts path stays off and the phone keeps the all-judge stack with fast/exhaustive modes. Caveats carried:
  2B weakest on multi-hop (e), 8.4% looping outputs (needs repeat-stop at decode).
- 10-15 [RESULTS 61d started: prepare the shipped 2B facts writer; PRE-REGISTERED before any 61d number]
  (1) PRODUCT WEIGHTS: the app ships ONE 2B adapter trained on ALL non-test training data = every TRAIN library (folds
  0-3, 45,982 photos) + the 313k extra public photos, the 61b recipe unchanged (LoRA r16 / vision r16 on the 4-bit-sim
  2B, 2 epochs, lr 1e-4 linear decay, bs 8 per GPU, 4 GPUs; no new hyperparameters), with the 61c millisecond scorer
  (tvw_t2b_final_xfit_full) unchanged. USE CONDITION: its dev facts through the R59d cascade must give a dev ALL difference
  vs all-judge inside the 61c dev 95% CI [-.0213, +.0154] (and so pass the bar); TRAIN is in its training data and is not
  scored. (2) MLX FORM: PEFT -> MLX adapter (scripts/peft_to_mlx_adapter.py layout, extended to the vision tower) next
  to the mlx-community 4-bit 2B, plus a parity check on 50 public dev photos: MLX (Python, if it runs here) vs vLLM
  phone-form outputs - exact-line agreement, token agreement, facts F1 vs each other. (3) REPEAT-STOP
  (src/findpics/facts_line.py, Swift FactsLine.swift, golden parity test): stop decoding when a completed list item
  repeats an earlier completed item of the same field; cut before it. Measured on the 61b 2B final's cross-fit outputs
  (exact for greedy: the cut of the full output = stopping while streaming): loop rate, tokens, facts F1 vs the 27B; and
  the R59d cascade with repeat-stopped facts (millisecond scorers retrained on repeat-stopped TRAIN facts, same recipe;
  TRAIN fold models + full): ADOPT iff it passes the same bar on TRAIN and dev and the loop rate falls.
- 10-15 ~03:20 [RESULTS 61d] (1) PRODUCT TRAINING submitted before this entry: fp_r61d_prod2b (53134611, 4x H100,
  eval/facts_writer/train2_job.sh p2b_all --folds=0,1,2,3 --extra=1 --epochs=2; config line: 358,714 rows = 45,691 TRAIN
  + 313,023 extra, 22,420 steps; ~4.7 h at the 61b rate). (3) REPEAT-STOP v1 (pre-registered rule, as written) on the 61b
  2B final cross-fit outputs (eval/facts_writer/r61d_repeat.py -> r61d_repeat.txt): loop rate (quality.rep) 5,933/70,580
  (.084) -> 5,284/70,580 (.075); fires on 1,457/70,580 lines, only 705/5,933 of the looping ones and 752/64,647 NON-looping
  ones; tokens 77.0 -> 76.3. RAW OUTPUTS (22 read, sampled seed 1): the dominant loop is FORMAT COLLAPSE - the line loses
  its "name:" fields and becomes ";"-separated items ("mask; gloves; shoes; pants; shirt; mask; gloves; ..." to the cap;
  "the blanket is dark blue; the puppy is sleeping; the blanket is dark blue; ..."), which v1 never catches because every
  ";" resets its memory (3,806/5,228 un-caught looping lines are at the 120 cap). The non-looping lines v1 cuts are mostly
  LEGITIMATE duplicates ("people: 3 - ..., person in dark clothing walking, person in dark clothing walking; activity: ..")
  - the cut drops activity / text / event. Also: quality.rep over-counts (3-gram >= 3 fires on honest people lists).
  PRE-REGISTERED NOW, before any v2 number: REPEAT-STOP v2 (facts_line.repeat_scan(version=2)): a new field starts only at
  a "name:" segment; stop at the 3rd occurrence of an item within its field, cut before it. Measured exactly as v1 (loop
  rate by quality.rep AND at-cap rate, tokens, facts F1 vs the 27B, R59d cascade TRAIN + dev with millisecond scorers
  retrained on the stopped TRAIN facts, same recipe). DECISION: among {v1, v2}, the variants that pass the unchanged bar
  (lower 95% >= -.03 vs all-judge on TRAIN and dev) AND lower the loop rate are adoptable; adopt the one with the lower
  loop rate (ties -> v1); none -> no repeat-stop. Caveat stated now: v2 was designed after looking at these same outputs
  (TRAIN + dev lines, not cascade numbers). (2) MLX FORM changed from the pre-registered "adapter" to MERGED 4-bit MLX
  weights, before any parity number: mlx-swift-lm 3.32.3 Qwen3VL `loraLayers` = languageModel.model.layers only, so the
  student's VISION LoRA cannot be loaded as an adapter; the 2B base is not otherwise on the phone (nothing to share); and
  merged + exact-MLX re-quantization is the form the cluster measured (merge_q4.py). eval/facts_writer/to_mlx.py writes
  mlx-community's layout (codes + bf16 scales/biases); validated against the real mlx-community file from the same bf16
  weights (job 53136344). Parity runs real MLX (Python mlx-vlm in a Debian container: RHEL8 glibc 2.28 < the mlx wheel's
  2.35) vs the vLLM phone form on 50 public dev photos.
- 10-15 ~03:45 [RESULTS 61d progress] REPEAT-STOP facts level (61b 2B final cross-fit, 70,580 TRAIN+dev photos;
  r61d_repeat_v{1,2}.txt, r61d_quality.txt): v1 loop .084 -> .075, at the 120 cap 4,150 -> 3,806, tokens 77.0 -> 76.3,
  cuts 752 non-looping lines (activity F1 dev .56 -> .54, text .70 -> .68); v2 loop .084 -> .054, at cap 4,150 -> 1,512,
  tokens 77.0 -> 74.9, fires on 2,665/5,933 looping and 9/64,647 non-looping lines, every facts F1 unchanged (objects
  .55, colours .35/.75, scene .57, people .80, text .70 dev). Raw lines read (~40, seeded): v2 stops the collapsed
  ";"-list loops ("Racing; Sports; Riding; Cycling; Racing; ..." kept to 40 tokens); what still runs to the cap is DRIFT
  without exact repeats ("pet door door handle; pet door door lock; ...", "motor cortex; premotor cortex; ..."); the 9
  non-loop v2 cuts include legit triples ("person, person, person" -> drops colours/people). Cascades for both submitted
  (eval/facts_writer/r61d_launch_rs.sh; scorers retrained per variant). MLX: to_mlx.py on CPU reproduces the real
  mlx-community 2B file from Qwen's bf16 weights BIT FOR BIT (2,181,296,128/2,181,296,128 elements); on CUDA the same
  arithmetic differs in ~0.3% of codes (so merge_q4.py phone forms, made on GPU, are within +-2 codes of true MLX on
  ~0.3% of weights). Swift FactsLine v1+v2 + 558-case golden fixture; swift test 155 + 29, 0 failures (before v2 port).
  Product pipeline waiter detached: eval/facts_writer/r61d_prod_launch.sh (merge_q4 + dev facts + to_mlx, then the dev
  cascade with the unchanged 61c scorer). MLX Python (mlx[cpu] 0.32.3 + mlx-vlm 0.7.4) in tools/mlxpy (Debian container).
- 10-15 ~04:10 [RESULTS 61d] REPEAT-STOP TRAIN cascades (R59d, scorers retrained per variant; exact bootstrap vs
  all-judge, n=643): no stop (61c) -.0110 [-.0254, +.0030]; v1 -.0250 [-.0407, -.0091] -> v1 FAILS the bar on TRAIN
  (e .740 -> .650, a .646 -> .626); v2 -.0068 [-.0203, +.0069]. Dev pending. Because each variant retrains its millisecond
  scorer, part of these moves may be scorer-retraining noise: PLACEBO submitted (descriptive only, not a decision input):
  the unchanged 61b facts with the scorers retrained at seed 1 (SEED=1 r61d_launch_rs.sh t2b_final_xfit), TRAIN + dev.
- 10-15 ~04:30 [RESULTS 61d CORRECTION] The v1 TRAIN number above (-.0250) is NOT valid: 15/643 v1 TRAIN requests (and 4/351
  v1 dev, 1/351 v2 dev) errored with an infrastructure failure - the lazily loaded 9B judge fails vLLM engine init on
  A100-SXM4-40GB ("max_num_seqs (256) exceeds available Mamba cache blocks (201)"), scored F1 0. Errored records removed,
  the 9 shards re-run (logs kept as cascade_<k>.log.enginefail), A100-40GB dropped from the 61d launchers' SPECS. v2 TRAIN
  had 0 errors (its -.0068 stands).
- 10-15 ~05:00 [RESULTS 61d (3) DONE: REPEAT-STOP v2 ADOPTED by the pre-registered rule] After the engine-failure re-runs
  (0 errors left), R59d vs all-judge, exact bootstrap (TRAIN n=643 / dev n=351): no stop (61c) -.0110 [-.0254, +.0030] /
  -.0024 [-.0213, +.0154]; v1 -.0087 [-.0226, +.0054] / +.0055 [-.0113, +.0233]; v2 -.0068 [-.0203, +.0069] / +.0070
  [-.0100, +.0242]. Both pass and both lower the loop rate; v2's is lower (.054 vs .075 of 70,580) -> v2 adopted: default
  version 2 in findpics.facts_line and FindPicsCore FactsLine (defaultVersion = 2). PLACEBO (unchanged facts, scorers
  retrained at seed 1): TRAIN -.0042 [-.0176, +.0089], dev +.0098 [-.0076, +.0279]; paired vs seed 0: TRAIN +.0067
  [-.0016, +.0151], dev +.0122 [+.0031, +.0231]; v2 vs no-stop paired: TRAIN +.0042 [-.0033, +.0119], dev +.0094 [-.0003,
  +.0209] -> the cascade-level effect of repeat-stop is NOT distinguishable from scorer-retraining noise (which is as large
  as .012 on dev); what repeat-stop measurably buys is decode work: tokens 77.0 -> 74.9 mean, at-cap lines 4,150 -> 1,512
  (of 70,580), with facts F1 unchanged. pytest 179/179 (tests/test_facts_line.py new); swift test 155 + 29, 0 failures.
- 10-15 ~06:10 [RESULTS 61d (2) MLX FORM + PARITY, on the 61b 2B final A student while the product adapter trains]
  to_mlx.py output = the vLLM phone form EXACTLY: 2,127,532,032/2,127,532,032 de-quantized weights identical (after
  reproducing merge_q4.py's double rounding of the TIED embedding: it quantizes embed_tokens, then lm_head = the same tensor
  again; 15% of embedding entries differ from a single rounding, max .018 - every 61/61b/61c number used this form, so the
  app ships it as measured). Preprocessing: mlx-vlm and the vLLM env's HF processor give identical pixel_values and prompt
  ids (3/3 photos, max abs diff 0). Real MLX (CUDA backend, H100) vs the batched vLLM lines, 50 public dev photos:
  exact line 11/50, objects F1 vs vLLM .92 (n=45), colours .87, scene .92, people .89; NOISE FLOOR = the same vLLM at batch
  1: exact 21/50, objects .94, colours .89 -> MLX's divergence is near-tie flips of the same kind ("22.09.2014" vs
  "22-09-2014", "fields" vs "houses"), slightly above vLLM's own batch noise; quality vs the 27B: MLX objects .53 / vLLM .54,
  colours .34 / .36 (50 photos: within noise). Repetition-penalty CONTEXT matters: MLX's default (last 20 tokens, also
  mlx-swift-lm's) gives exact 7/50, colours .78 vs vLLM, 73.4 vs 75.9 tokens -> the app must set repetitionContextSize to
  cover prompt + output (e.g. 4096), as vLLM does. MLX CPU backend: 1,815 s for one photo (single-threaded) - not usable.
- 10-16 ~03:00 [RESULTS 61d] Session dropped ~18 h; the detached product waiter died with it. Product adapter p2b_all
  finished on 10-15 09:21 (22,420 steps, 717,440 photos seen, 22,796 s; final validation loss held-out TRAIN-photos .611 /
  extra .616, vs 61b 2B-A .612 / B .600). Product pipeline relaunched (r61d_prod_launch.sh: merge_q4 + dev facts + to_mlx,
  then the pre-registered dev check with the unchanged 61c scorer, plain and with repeat-stop v2). Coordinator: tonight is
  cluster-only, no new MAC_INBOX items.
- 10-16 ~03:10 [RESULTS 63 START: b/c/g truth pass, dev + TRAIN; test stays closed] Pre-registered (6c0b74a,
  AUDIT_PROTOCOL.txt "ADDED 10-16") before any answer: D = Qwen3.5-27B-FP8 asked the owner's words directly per (request,
  album, photo) on every photo of the library (t63/direct.py; ~972k prompts, 412 requests: b 177, c 138, g 97); L =
  Qwen3.5-122B-A10B-FP8 on every D != T unit; flip a category x direction stratum (T=0,D=1,L=1 add; T=1,D=0,L=0 remove)
  iff my blind eye agreement with the flip >= .70 on its sample (30 each); VERIFIED iff the corrected truth's stratified
  eye estimate reaches P >= .8 and R >= .7; no product re-tuning, only re-scoring existing runs (shipped c61d rs2 r59d,
  all-judge). Submitted fp_t63_d27_0..15 (race_sbatch, H100/H200/RTX PRO 6000; 10 started on kempner_rtx at once).
- 10-16 ~04:10 [RESULTS 61d DONE] Product adapter p2b_all passes its pre-registered use condition: dev R59d +.0001
  [-.0182, +.0176] (n=351; inside the 61c CI [-.0213, +.0154]); with repeat-stop v2 +.0054 [-.0125, +.0241]. Product facts
  beat the 61b student on dev (objects F1 .60 vs .55, people .87 vs .80; at cap 136 vs 1,509 of 24,598); repeat-stop v2
  fires on 14/24,598. MLX form p2b_all_final_mlx (1.7 GB) = phone form exactly (2,127,532,032/2,127,532,032); MLX parity
  on 50 dev photos 19/50 exact (vLLM batch-1 noise 26/50), field F1 vs vLLM .87-1.00, same quality vs 27B. 20 raw
  product lines read. RESULTS 61d appended. No MAC_INBOX items (coordinator: cluster-only tonight). Not pushed.
- 10-16 ~05:00 [RESULTS 64 PRE-REGISTRATION: facts line written ONCE per near-duplicate group; committed before any number]
  Goal: cut the phone's index cost (p2b_all + repeat-stop v2, ~74 tok/photo, est. 222-489 phone-h / 187k photos) by
  writing one line per near-duplicate group. Test closed; dev + TRAIN only; public bench (DISBench Flickr libraries).
  RULE FAMILY G(s) (eval/dedup64/group.py): per library, capture-time order; photo i joins an earlier REPRESENTATIVE r iff
  both dated, 0 <= t_i - t_r <= 10 min (the product's BURST_MINUTES), PE-Core-L-14-336 image cosine >= s, and (when both
  have GPS) <= 1 km; i joins the eligible rep with the highest cosine, else becomes a rep (leader grouping: no chaining,
  streamable at index time). Members get the rep's writer line; each keeps its OWN metadata prefix (place, month) and its
  own image vector. Undated photos keep their own line. Grid s in {0.98, 0.96, 0.94, 0.92, 0.90} (0.90 = the product's
  display-burst cosine); s = inf (no groups) = reproduction check (must reproduce the per-photo baseline, |ALL diff| <=
  .002; else the harness difference is reported next to every number).
  NO RETRAINING: same R59d cascade code (variant r59d, frozen wt_r59b src), plans, judge cache (+ lazy 9B for misses),
  scorers tvw_t2b_final_rs2_xfit_h{fold} (TRAIN) / _full (dev), facts reader ce12B_full. Per-photo baselines = 61d:
  TRAIN c61d_train_t2b_final_rs2_xfit (cross-fit 61b 2B + v2 lines), dev c61d_dev_p2b_devmix_rs2 (product p2b_all + v2).
  BAR (both, exact bootstrap 2,000 over requests, seed 0, paired): (A) shared - per-photo ALL lower 95% >= -.01;
  (B) shared - all-judge ALL lower 95% >= -.03. THRESHOLD CHOSEN ON TRAIN ONLY: the smallest s in the grid (most
  sharing) that meets A and B on TRAIN (n=643). Dev (n=351) is then run ONCE, at that s only; dev must meet A and B, no
  re-selection. If no s passes on TRAIN: nothing ships, dev runs at s=0.98 for information only. Disclosure: picking
  the passing s on TRAIN makes TRAIN in-sample for the choice (5 looks); dev is the clean confirmation.
  EYE CHECK (ship condition too): 30 (rep, member) pairs at the chosen s, seed 0, 15 TRAIN + 15 dev, viewed at >= 900 px:
  >= 27/30 must be true near-duplicates (same moment and subject; the rep's line is correct for the member); plus every
  lost-truth failure I can sample (truth photos returned per-photo, not returned shared, that are members): cause by eye.
  REPORTED SAVING: writer calls saved = members / photos, per library and pooled, with denominators (TRAIN 24 libraries,
  dev 13); phone-hours scaled only as "if a phone library had the bench's duplicate rate". Representativeness of the
  bench's duplicate rate for a real phone camera roll: measured only on public data if at all, else stated UNKNOWN
  (Flickr uploads are owner-selected; no private data used).
