# Free face recognizer we may ship: deep dive (2026-10-06)

Follow-up to eval/face_commercial.md. Code: eval/eval_face_free.py (embed / analyze / sweep). Results:
eval/results_face_free.json, eval/results_face_free_sweep_{celeba,digiface}.json. Audit sheets (public CelebA images,
git-ignored): eval/audits/face_free_{aura_misses,buffalo_wrong,aura_wrong}_p*.jpg, made by eval/audits/face_free_audit.py.

## 1. Wider licence search (web, primary sources)

Result: no recognizer beyond AuraFace has both permissive weights and training data documented as commercial.

| Candidate | Weights licence | Training data | Verdict | Evidence |
|---|---|---|---|---|
| AuraFace-v1 (fal) | Apache-2.0 | "trained on a commercial dataset comprising face images from various sources" (no more detail; no v2 found) | clean per vendor statement, not auditable | https://huggingface.co/fal/AuraFace-v1 |
| OpenCV Zoo SFace | Apache file | undocumented; upstream repo lists CASIA-WebFace / VGGFace2 / MS-Celeb-1M; issues #313, #318 unanswered | grey | https://github.com/opencv/opencv_zoo/issues/318 , https://github.com/zhongyy/SFace |
| Intel OMZ face-reidentification-retail-0095 | Apache-2.0 (model.yml) | not documented anywhere found | grey (same as SFace); 1.1M params, 2.2 MB fp16; not measured | https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/face-reidentification-retail-0095 |
| LVFace (ByteDance 2025) | code MIT; models "for non-commercial research purposes only" | Glint360K, WebFace42M | no | https://github.com/bytedance/LVFace |
| AdaFace, GhostFaceNets, TransFace, TopoFR, UniFace pack | MIT or none | MS1M / WebFace / Glint360K / CASIA (research-only) | no | repos on GitHub |
| Glint360K-trained models | n/a | "available for non-commercial research purposes only" | no | https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc |
| Synthetic-data models/sets: HyperFace, Vec2Face/HSFace, DCFace, IDiff-Face, UIFace, CemiFace, ID3, GANDiffFace, Digi2Real, DigiFace, FaceSynthetics, SFHQ, FLUXSynID, EigenFace | various | every chain contains a research-only generator, encoder or source set (WebFace, Glint360K, CASIA, FFHQ, Flux-dev) | no / grey | see face_commercial.md and repos |
| VoxCeleb1/2 | metadata CC BY-SA 4.0 (Oxford page) | frames are third-party YouTube copyright | no for training | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html |
| Casual Conversations v1 (Meta) | n/a | "You cannot train any model with the provided labels" | no | https://ai.meta.com/datasets/casual-conversations-dataset/ |
| Apple Vision | n/a | no public face-identity API (Apple DTS: "You would need to train your own model") | n/a | https://developer.apple.com/forums/thread/751504 |
| buffalo_l commercial licence | sold by InsightFace, price "Custom" | | paid option | https://www.insightface.ai/ |

## 2. Free accuracy boosts (single-image protocol of face_commercial.md)

Same detector + alignment for all (SCRFD + norm_crop, 112 px ArcFace template). Alignment is already what each model
expects: OpenCV's FaceRecognizerSF.alignCrop uses the identical 5-point template and 112 px, and feeds BGR (swapRB in
feature()); AuraFace is an insightface ArcFaceONNX (RGB, mean/std 127.5). Channel-order checks confirm the defaults
(AuraFace fed BGR: CelebA 0.871 vs 0.908; SFace fed RGB: 0.909 vs 0.942). So the only free boosts are flip and fusion.
Recall at buffalo_l's wrong-match rate at 0.40 (DigiFace 0.0021, CelebA 1.15e-5), 8 references, max over references.
Baselines reproduce face_commercial.md exactly.

| Variant | DigiFace (of 19,110) | CelebA (of 12,255) | CelebA @10x lower rate |
|---|---|---|---|
| buffalo_l (non-commercial) | 0.987 (18,855) | 0.979 (11,994) | 0.822 |
| buffalo_l + flip | 0.988 (18,885) | 0.978 (11,990) | 0.808 |
| AuraFace | 0.904 (17,272) | 0.908 (11,127) | 0.709 |
| **AuraFace + flip** | **0.927 (17,721)** | **0.927 (11,357)** | 0.699 |
| SFace (grey) | 0.921 (17,601) | 0.942 (11,541) | 0.820 |
| SFace + flip (grey) | 0.927 (17,715) | 0.948 (11,616) | 0.824 |
| AuraFace + SFace, z-fused (grey) | 0.937 (17,902) | 0.947 (11,605) | 0.780 |
| AuraFace + SFace, z-fused + flip (grey) | 0.949 (18,138) | 0.954 (11,696) | 0.795 |
| AuraFace + HyperFace-10k, z-fused + flip (grey) | 0.960 (18,337) | 0.933 (11,434) | 0.741 |
| all three, z-fused + flip (grey) | 0.961 (18,361) | 0.949 (11,635) | 0.800 |

Flip = average of the L2-normalized embeddings of the crop and its mirror (2x compute). z-fusion = concatenation of
L2-normalized embeddings, each scaled by 1/sqrt(std of its random-pair scores), so the fused score is the sum of
std-normalized scores (no labels used). The @10x-lower column on CelebA sits on ~14 wrong pairs and is noisy.

## 3. Product-level recall (people.py exactly)

Pipeline per identity: n references -> expand_refs (3 rounds) -> face_groups "other identities" rule ->
item_person_scores -> accept. Every image of every identity forms one library (CelebA 18,295 faces / 755 people;
DigiFace 21,510 / 300). Note: people.py matches by the MAX over references (nearest reference), not the mean of
reference embeddings; that is what is measured here. The shortcut used for speed was asserted equal to
item_person_scores on the first 3 identities of every run.

Two operating points:
- (a) "mapped": buffalo_l at the shipped cuts (expand 0.55, accept 0.40, possible 0.30); other models at the cuts with
  the same pairwise wrong-match count on that dataset.
- (b) "equal wrong items" (headline): each model, buffalo_l included, gets its best expansion cut from a grid (none,
  mapped 0.55, same-person-score quantiles 0.2/0.4/0.6/0.8) and the accept cut that gives the SAME number of wrong items
  as buffalo_l at its shipped cuts (CelebA: 62 for 3 refs, 180 for 8 refs). Tuned on the labels: an upper bound,
  applied to every model alike.

CelebA (real faces), recall of the person's other photos:

| Variant | 3 refs (a) mapped | 3 refs (b) equal wrong (of 16,030) | 8 refs (a) mapped | 8 refs (b) equal wrong (of 12,255) |
|---|---|---|---|---|
| buffalo_l | 0.976, 62 wrong | **0.977** (15,658) | 0.979, 180 wrong | **0.980** (12,005) |
| AuraFace | 0.921, 930 wrong | 0.845 (13,539) | 0.935, 1,124 wrong | 0.912 (11,172) |
| **AuraFace + flip** | 0.935, 773 wrong | **0.888** (14,228) | 0.946, 878 wrong | **0.929** (11,389) |
| SFace + flip (grey) | 0.956, 418 wrong | 0.943 (15,114) | 0.963, 552 wrong | 0.952 (11,668) |
| AuraFace + SFace z-fused + flip (grey) | 0.957, 247 wrong | 0.921 (14,769) | 0.963, 346 wrong | 0.957 (11,726) |

DigiFace (rendered faces): at buffalo_l's shipped cuts every model returns ~60,000 wrong items over 300 queries
(precision 0.25 for buffalo_l): rendered faces look far more alike than real ones, so the 0.40 cut is far too loose
there and these numbers say little about users. At equal wrong items (60,038 / 60,712): buffalo_l 0.994 / 0.996,
AuraFace+flip 0.949 / 0.969, SFace+flip 0.951 / 0.972, AuraFace+SFace+flip 0.971 / 0.980 (3 / 8 refs; denominators
20,610 / 19,110).

Raw-trace check (CelebA, 3 refs, operating point b; full-resolution sheets viewed):
- buffalo_l's 62 "wrong" items are mostly CelebA label noise: 8 of 12 sampled are one identity pair (9715/6709) whose
  photos show the same woman under two ids; 1 more (9310/9128) is clearly the same woman. So buffalo_l's real error
  count is lower than 62.
- AuraFace+flip's 62 wrong items are mostly real errors: about 9 of 12 sampled are clearly different people (e.g. two
  different men, a brunette matched to a blonde); 2 are the same label-noise pairs.
- AuraFace+flip misses that buffalo_l finds (1,423 such items): blurry or video frames, upward gaze, sunglasses, heavy
  makeup, profile. In 12/12 sampled the item looked like the same person.
- So the equal-wrong-items gap (0.977 vs 0.888 with 3 refs) understates buffalo_l's real lead.

## 4. Fine-tuning

Not feasible on clean data: no free face dataset with identity labels that is clearly commercially usable was found
(every candidate is research-only, evaluation-only, or third-party copyright). Paid routes only: a licensed or
consented dataset from a vendor (e.g. Generated Photos sells commercial licences; price and generator provenance not
published), or the InsightFace commercial licence.

## Summary table

| Option | Licence evidence | DigiFace single-image (of 19,110) | CelebA single-image (of 12,255) | Product CelebA, equal wrong items, 3 / 8 refs | Phone size (fp16) |
|---|---|---|---|---|---|
| buffalo_l | non-commercial; paid licence "Custom" | 0.987 | 0.979 | 0.977 / 0.980 | 87 MB (shipping now, dev only) |
| AuraFace + flip | Apache-2.0; vendor says commercial data | 0.927 | 0.927 | 0.888 / 0.929 | 130 MB (65M params), 2 passes per face |
| SFace + flip | Apache file; data undocumented (grey) | 0.927 | 0.948 | 0.943 / 0.952 | 19 MB (9.7M params) |
| AuraFace + SFace + flip | inherits SFace's grey status | 0.949 | 0.954 | 0.921 / 0.957 | 149 MB, 4 passes per face |
| HyperFace-10k (+ fusions) | MIT weights; research-only chain | 0.949 alone | 0.808 alone | not headline | 87 MB |

Caveats: CelebA celebrities may be in buffalo_l's (and possibly AuraFace's/SFace's) training data. Tuning used the
labels, for every model alike. The phone detects faces with Apple Vision landmarks, not SCRFD; alignment is the same
template, but the numbers need a device check. Speed on the phone was not measured (ResNet-100 is ~1.5x buffalo_l's
ResNet-50 parameters). int8 quantization was not tested.

## Recommendation

Cleanest free option: AuraFace with flip test-time augmentation. On real faces it gives up about 9 points with 3 reference
photos (0.888 vs 0.977) and 5 points with 8 (0.929 vs 0.980), at the same number of wrong matches, and its errors are
real wrong people where buffalo_l's are mostly label noise. More reference photos close much of the gap, so the app
should ask for 8 or more. If the quality gap is unacceptable, the only clean way to near-buffalo_l quality is the
paid InsightFace licence. SFace (or AuraFace+SFace) is better but stays grey until OpenCV documents the training data.
Legal caveats for Reza: fal's "commercial dataset" claim is unaudited; the Apache licence covers copyright, not
biometric-privacy law (BIPA/GDPR), which applies to on-device face matching regardless of model.
