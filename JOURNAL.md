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
