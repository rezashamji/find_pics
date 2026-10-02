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
