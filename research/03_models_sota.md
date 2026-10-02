# 03 — Model selection, state of the art as of 2026-10-02

Project: find_pics, natural-language search over a personal library of about 150k photos and videos.
Constraints:
- Open-source weights only.
- H100/H200 for development.
- An Apple Silicon MacBook for end users, ideally.

## How this was produced, and its limits

Five parallel research passes (web search, Hugging Face API, GitHub, arXiv), followed by my own spot checks of the highest-stakes license claims against the live HF API on 2026-10-02.

Every pass hit a shared web-search budget cap partway through. Later checks were done by fetching known URLs directly. As a result, **some mid/late-2026 releases may be missing**. Items that were not confirmed are marked UNVERIFIED.

Tags:
- `[DOC: url]` means the claim was read at that source.
- `[INF]` means my inference or estimate.
- `UNVERIFIED` means I could not confirm it.

### Spot-checked directly against the HF API on 2026-10-02 [DOC: https://huggingface.co/api/models/<id>]

| HF id | License tag | Gated | Created |
|---|---|---|---|
| google/gemma-4-E4B-it | apache-2.0 | no | 2026-03-02 |
| Qwen/Qwen3.5-4B | apache-2.0 | no | 2026-02-27 |
| Qwen/Qwen3.8-27B | apache-2.0 | no | 2026-08-05 |
| Qwen/Qwen3-VL-Reranker-2B | apache-2.0 | no | 2026-01-07 |
| Qwen/Qwen3-VL-Embedding-2B | apache-2.0 | no | 2026-01-07 |
| facebook/PE-Core-L14-336 | apache-2.0 | no | 2025-04-11 |
| google/siglip2-so400m-patch16-naflex | apache-2.0 | no | 2025-02-18 |
| qihoo360/fg-clip2-large | apache-2.0 | no | 2025-10-13 |
| apple/MobileCLIP2-S2 | **apple-amlr** (research only) | no | 2025-08-25 |
| allenai/Molmo2-8B | apache-2.0 (but the card restricts its training data, see §6) | no | 2025-12-14 |
| openbmb/MiniCPM-V-4.6 | apache-2.0 | no | 2026-04-13 |
| minchul/cvlface_adaface_vit_base_kprpe_webface12m | **no license tag** | no | 2024-06-06 |
| bytedance-research/LVFace | "mit" tag, but the README says the models are non-commercial (see §2) | no | 2025-08-10 |
| immich-app/scrfd_34g_gnkps | mit | no | 2024-08-12 |
| fal/AuraFace-v1 | apache-2.0 | no | 2024-08-16 |

Also checked: the ModelScope `iic/cv_resnet_facedetection_scrfd10gkps` License field reads "MIT License" [DOC: https://modelscope.cn/api/v1/models/iic/cv_resnet_facedetection_scrfd10gkps].

### Metric glossary (one line each)

- **WIDER FACE Easy/Medium/Hard AP.** Average precision of face boxes on the standard face-detection benchmark. "Hard" is dominated by tiny, occluded and group-shot faces, so it is the number that matters for "small faces in group photos".
- **TAR@FAR=1e-4 (IJB-B/IJB-C).** The fraction of true same-person pairs accepted when the threshold is set so that only 1 in 10,000 different-person pairs is wrongly accepted. This is the realistic verification metric.
- **LFW / CFP-FP / CPLFW / AgeDB-30 / CALFW.** Pair-verification accuracy:
  - LFW is easy and saturated.
  - CFP-FP and CPLFW test frontal-vs-profile pose.
  - AgeDB-30 and CALFW test cross-age pairs; AgeDB-30 pairs are about 30 years apart.
- **TinyFace rank-1.** Identification accuracy on low-resolution faces, a proxy for "far away in the background".
- **Market-1501 / MSMT17 mAP, Rank-1.** Whole-body person re-identification across cameras. The person wears the same clothes; these benchmarks are short-term surveillance data.
- **COCO / Flickr30k R@1.** Recall at 1: the fraction of queries whose correct item is ranked first. T→I means a text query retrieving images.
- **ImageNet zero-shot top-1.** Classification accuracy when the class names are used as text prompts, with no training on ImageNet.
- **MMEB-V2.** A multimodal embedding benchmark suite covering image, video and document retrieval tasks.
- **POPE.** Yes/no object-presence hallucination benchmark.
- **HallusionBench.** A harder visual-hallucination and illusion benchmark.

---

## 1. Face detection (small faces, profiles, group shots)

### Candidates

WIDER FACE numbers are Easy / Medium / Hard.

| Model | Where | Size | WIDER FACE | Weight license | Mac path |
|---|---|---|---|---|---|
| SCRFD-10G-KPS (`det_10g.onnx` inside buffalo_l / antelopev2) | github.com/deepinsight/insightface/tree/master/detection/scrfd | 4.23M params, 10 GFLOPs, 5.0 ms | 95.40 / 94.01 / 82.80 (VGA input) [DOC] | **Non-commercial research only** [DOC: insightface README] | ONNX via onnxruntime + CoreML EP |
| SCRFD-34G (InsightFace) | same | 9.80M params, 11.7 ms | 96.06 / 94.92 / 85.29 (VGA) [DOC] | Non-commercial (InsightFace copy) | ONNX |
| SCRFD-34G-GNKPS v2 (Alibaba ModelScope release) | `iic/cv_resnet_facedetection_scrfd10gkps`; ONNX rehost `immich-app/scrfd_34g_gnkps` | 9.84M params, 11.8 ms on a 2080Ti | 96.17 / 95.19 / 84.88 [DOC: ModelScope card] | **MIT** [DOC: ModelScope API, HF API] | ONNX |
| MogFace-large | ModelScope `iic/cv_resnet101_face-detection_cvpr22papermogface` | ResNet-101 | 97.0 / 96.3 / 93.0 [DOC] | **MIT** [DOC: ModelScope API] | PyTorch; heavy |
| RetinaFace-R50 | github.com/biubug6/Pytorch_Retinaface | about 27M (UNVERIFIED) | 94.82 / 93.84 / 89.60 single-scale [DOC] | MIT repo; weights distributed under it [DOC] | PyTorch → ONNX / Core ML [INF] |
| RetinaFace-MobileNet0.25 | same | 1.7 MB | 88.67 / 87.09 / 80.99 [DOC] | MIT | same |
| YOLOv5-Face s / l | github.com/deepcam-cn/yolov5-face | 7.1M / 46.6M | s: 94.67 / 92.75 / 83.03; l: 95.78 / 94.30 / 86.13 [DOC] | **GPL-3.0** | ONNX / Core ML |
| YOLOv8-face n / m | github.com/derronqi/yolov8-face | — | n: 94.6 / 92.3 / 79.6; m: 96.6 / 95.0 / 84.7 [DOC] | **GPL-3.0**; Ultralytics base is AGPL-3.0 | Core ML export |
| akanametov/yolo-face (v8 to v12, "v26") | github.com/akanametov/yolo-face | — | no E/M/H table | GPL/AGPL | Core ML export |
| YuNet | opencv_zoo `face_detection_yunet` | about 75k params (UNVERIFIED) | about 0.88 / 0.87 / 0.75 [DOC: opencv_zoo eval table] | **MIT** [DOC] | ONNX / OpenCV; very fast |
| TinaFace R50 | github.com/Media-Smart/vedadet | R50 + DCN | 96.3 / 95.7 / 93.0 at 1100×1650 [DOC] | Apache-2.0 repo | DCN ops are hard to export [INF] |
| DSFD | TencentYoutuResearch / hukkelas | VGG/ResNet | about 96.6 / 95.7 / 90.4 (UNVERIFIED) | Apache-2.0 | about 100 ms per image on a V100 [DOC: hukkelas README] |
| MediaPipe BlazeFace | Google | 128×128 input | no WIDER FACE numbers | Apache-2.0 code; CC-BY-4.0 model cards [DOC] | TFLite |
| Apple Vision `VNDetectFaceRectanglesRequest` rev 3 | macOS / iOS API | — | not published | Proprietary OS API, free to call | Native, runs on the Apple Neural Engine (ANE) |

Notes on specific rows:

- **BlazeFace.** Google documents the short-range model as best within about 2 m and the full-range model within about 5 m [DOC: github.com/google-ai-edge/mediapipe docs]. With a 128 px input it will miss small background faces in group shots [INF]. Not recommended.
- **Apple Vision.**
  - Rev 3 adds detection of masked faces and head pitch.
  - `VNDetectFaceCaptureQualityRequest` is documented as "a comparative measure of the same subject … not designed to compare faces of different people". Use it to pick the best face of a person, not as a global quality gate.
  - `VNDetectHumanRectanglesRequest` rev 2 has `upperBodyOnly` (default true) and full-body modes.
  - Source: [DOC: https://developer.apple.com/videos/play/wwdc2021/10040/].
  - Vision exposes **no public face-identity embedding**, so you need your own Core ML model for recognition (UNVERIFIED as an absolute claim; no such API was found).
- **YuNet** is designed for faces of roughly 10 to 300 px [DOC: opencv_zoo]. PhotoPrism and digiKam 8.6 both use it [DOC: PhotoPrism docs; digiKam 8.6 release notes].
- **Data-provenance caveat on every detector.** The WIDER FACE dataset is distributed under **CC BY-NC-ND** [DOC: http://shuoyang1213.me/WIDERFACE/]. Nearly every open face detector was trained on it, so even "MIT" detector weights carry a training-data question. Whether trained weights inherit dataset terms is legally unsettled [INF].
- **fal/AuraFace-v1 trap.** It is tagged Apache-2.0, but it ships `scrfd_10g_bnkps.onnx`, which is **byte-identical (same SHA256) to InsightFace's non-commercial SCRFD** [DOC: HF x-linked-etag comparison by the research pass]. Only its recognizer (`glintr100.onnx`, which differs from InsightFace's) is AuraFace's own work.

### Recommendation

- **Dev / personal use:** SCRFD-34G (or the 10G in buffalo_l) run at higher input resolution, e.g. 1280 px long side or tiled. The published Hard AP is measured at VGA (640 px); small-face recall rises with input resolution [INF].
- **Commercial-safe:**
  - ModelScope SCRFD-34G-v2 (MIT, ONNX at `immich-app/scrfd_34g_gnkps`) as the main detector.
  - YuNet (MIT) as a tiny fallback.
  - Both still carry the WIDER FACE caveat.
  - Avoid the YOLO-face family (GPL/AGPL) in closed-source distribution.
- **Mac:**
  - Run the ONNX detector through onnxruntime's CoreML execution provider (Immich lists `CoreMLExecutionProvider` [DOC: immich `machine-learning/immich_ml/models/constants.py`]).
  - Or use Apple Vision rev 3 as a zero-license-cost native detector.
  - Mixing detectors between server and Mac changes face crops and alignment, which shifts embeddings slightly. Pick one per index [INF].

---

## 2. Face recognition / identity embeddings

### Candidates

All accuracies are in %. IJB-C is TAR@FAR=1e-4. "AgeDB" is AgeDB-30.

| Model (HF id / repo) | Backbone, training data | LFW | CFP-FP | CPLFW | AgeDB | CALFW | IJB-B | IJB-C | TinyFace R1 | Weight license |
|---|---|---|---|---|---|---|---|---|---|---|
| **CVLface AdaFace ViT-B KP-RPE** `minchul/cvlface_adaface_vit_base_kprpe_webface12m` | ViT-B (~0.1B params), WebFace12M | 99.82 | 99.30 | 95.65 | 98.10 | 95.93 | 96.55 | **97.82** | **76.10** | No license on card; card says follow the training-dataset license, so **research-only in practice** [DOC card; legal reading INF] |
| CVLface AdaFace IR101 `minchul/cvlface_adaface_ir101_webface12m` | IR-101, WebFace12M | 99.82 | 99.24 | 94.57 | 98.00 | 96.12 | 96.46 | 97.72 | 72.42 | same |
| AdaFace R100 (github.com/mk-minchul/AdaFace) | R100, WebFace12M | 99.82 | 99.26 | 94.57 | 98.00 | 96.12 | 96.41 | 97.66 | 72.29 | MIT code; weights inherit WebFace terms (UNVERIFIED) |
| LVFace-B / L (bytedance-research/LVFace, ICCV 2025) | ViT, Glint360K | — | — | — | — | — | 96.51 | 97.70 / 97.66 | — | HF tag "mit", but the README says the models are **non-commercial research only** [DOC] |
| TopoFR R200 (github.com/DanJun6737/TopoFR, NeurIPS 2024) | R50–R200, MS1MV2 / Glint360K | — | — | — | — | — | — | up to 97.84 (Glint360K) | — | No license file, so all rights reserved; research data |
| InsightFace **buffalo_l** (`w600k_r50`) | R50, WebFace600K | — | — | — | — | — | — | 97.25 | — | **Non-commercial research only**; commercial license via recognition-oss-pack@insightface.ai [DOC] |
| InsightFace **antelopev2** (`glintr100`) | R100, Glint360K | — | — | — | — | — | — | 97.13 | — | same |
| TransFace (ICCV 2023) | ViT-S/B/L, MS1MV2 / Glint360K | — | — | — | — | — | — | IJB-C at 1e-6: 88.6–89.9 | — | No license file |
| ModelScope CurricularFace IR101 `iic/cv_ir101_facerecognition_cfglint` | IR101, Glint360K | — | — | — | — | — | — | 96.17 | — | **Apache-2.0** label [DOC: ModelScope API]; Glint360K data terms apply [INF] |
| ModelScope TransFace `iic/cv_vit_face-recognition` | ViT | — | — | — | — | — | — | 97.45 | — | **MIT** label [DOC: ModelScope API]; data caveat |
| **AuraFace-v1** recognizer `fal/AuraFace-v1` | ArcFace-style R100, "commercial dataset" | 99.65 | 95.19 | 90.93 | 96.10 | 94.70 | — | not reported | — | **Apache-2.0**; card: "trained on commercially and publicly available data sources to enable its usage in commercial setting" [DOC: https://huggingface.co/fal/AuraFace-v1] |
| EdgeFace-Base (Idiap) | 18.2M params, WebFace4M/12M subsets | 99.83 | 97.01 | 93.75 | 97.60 | 96.07 | — | — | — | HF cards **CC-BY-NC-SA-4.0** (GitHub code is BSD-3) [DOC] |
| EdgeFace-S γ=0.5 | 3.65M params | — | — | — | — | — | 93.58 | 95.63 | — | same |
| OpenCV SFace `face_recognition_sface_2021dec` | MobileFaceNet, 128-d | 99.40 | — | — | — | — | — | — | — | **Apache-2.0** label; training data undocumented (opencv_zoo issue #313, open) [DOC] |
| dlib `dlib_face_recognition_resnet_model_v1` | ResNet-34-ish, 128-d, ~3M faces | 99.38 | — | — | — | — | — | — | — | **Public domain / CC0** [DOC: github.com/davisking/dlib-models]; FaceScrub/VGG data caveat [INF] |
| facenet-pytorch (VGGFace2) | InceptionResnetV1 | 99.65 | — | — | — | — | — | — | — | MIT code; weight terms unstated (grey zone) [DOC] |
| GhostFaceNets | MobileNet-class, MS1MV3 | 99.73 | 96.83 | — | 98.0 | — | 93.12 | 94.94 | — | MIT repo; MS1M is research-only (grey) |

Sources for the table:
- CVLface rows: [DOC: https://raw.githubusercontent.com/mk-minchul/CVLface/main/cvlface/pretrained_models/README.md]. The IJB columns there are labelled "@0.01", which means 0.01% = 1e-4 [INF].
- AdaFace row: [DOC: https://github.com/mk-minchul/AdaFace].
- InsightFace rows: [DOC: https://github.com/deepinsight/insightface/tree/master/model_zoo].
- LVFace row: [DOC: https://github.com/bytedance/LVFace].
- TopoFR row: [DOC: https://github.com/DanJun6737/TopoFR].
- EdgeFace rows: [DOC: https://huggingface.co/Idiap/EdgeFace-Base].
- "TopoFX" in the brief is almost certainly **TopoFR**. No model named TopoFX exists [INF].

### Other models in this space

- **FRoundation** (arXiv 2410.23831) shows that general foundation models make poor face matchers:
  - Zero-shot, DINOv2 averages 64.70% and CLIP 82.64% across face benchmarks.
  - Even after fine-tuning on face data, they reach only about 96%, below purpose-trained face models [DOC].
  - **Do not use the CLIP/SigLIP search embedding as an identity signal.**
- **SapiensID** (CVPR 2025, `mk-minchul/sapiensid`) is a unified face + body model. Its face average across five benchmarks is 97.31, vs 97.63 for AdaFace. License: **CC BY-NC 4.0** [DOC: https://arxiv.org/html/2504.04708].
- **Apple Photos** keeps person clusters and names inside Photos.app. `osxphotos` can read them for libraries stored in Photos.app [DOC: https://github.com/RhetTbull/osxphotos].
  - These are labels, not embeddings.
  - They are useful as free ground truth for evaluating your own clustering [INF].

### Age robustness

- **AgeDB-30 (about 30-year gaps)** is the closest public proxy for the 10-year problem.
  - WebFace12M-trained AdaFace and CVLface reach **98.0–98.1**.
  - The specialised age-invariant model MTLFace (CVPR 2021) reached 96.45 vs a 95.15 ArcFace baseline on smaller data [DOC: arXiv 2103.01520 via search summary].
  - Training-data scale beats age-specific architectures here [INF; not a controlled comparison].
- **Best-Rowden & Jain, TPAMI 2018** (longitudinal mugshot study; circa-2015 commercial matchers):
  - Genuine-pair similarity scores decline with elapsed time.
  - About 99% of subjects remained recognisable at 0.01% false-accept rate (FAR) up to about 6 years.
  - [DOC via abstract summary: https://pubmed.ncbi.nlm.nih.gov/28092523/; exact wording UNVERIFIED]
  - Beyond about 6 years, degradation is expected [INF].
- **NIST FRTE** runs an aging category for gaps of 10 years or more ("Border ΔT ≥ 10 yrs") [DOC: NEC press release 2025-04]. The false-non-match values were not extracted (UNVERIFIED).

### Body-weight change robustness

This is the weakest-evidence area in the whole report.

- **Wen, Guo & Li, IJCB 2014, "A study on the influence of body weight changes on face recognition".**
  - Method: partial least squares on LBP/SIFT features.
  - 69.6% verification on 242 real subjects (2 images each); 88.7% on a synthetic set.
  - Concluded that large weight change "significantly" reduces accuracy.
  - [DOC: https://ieeexplore.ieee.org/document/6996243/]
- **Singh, Nagpal, Singh & Vatsa, IEEE Access 2014, "On recognizing face images with weight and age variations".**
  - Introduced the **WIT (WhoIsIt)** database: 110 subjects, 1,109 images.
  - Rank-1 identification of about 20% with Gabor features + neural networks.
  - [DOC: cited in https://iab-rubric.org/images/pdf/papers/2015_BTAS_RegularizingDeep.pdf]
- **Nagpal, Singh, Vatsa & Singh, BTAS 2015 / IEEE Access 2015, "Regularized deep learning for face recognition with weight variations".**
  - Introduced the **eWIT** database: 200 public figures, 2,036 images, labelled thin / moderate / heavy, but with an average within-subject age gap of 28.8 years. Weight and age are confounded.
  - Rank-1 results:
    - COTS VeriLook: 14.3%
    - Their weight-regularised deep Boltzmann machine (DBM): 23.4%
  - [DOC: same PDF, Table 3]
- **Modern margin-loss embeddings (ArcFace, AdaFace, ViT) have no peer-reviewed evaluation on weight-change pairs.** None was found in these passes (UNVERIFIED absence).
  - Indirect evidence: AgeDB and CALFW celebrity pairs contain weight swings mixed with age, and current models score about 96–98% on them [INF].
  - Mechanism [INF]:
    - Weight change mostly alters soft tissue (cheeks, jawline, neck).
    - Margin-loss models lean on periocular, nasal and brow structure, which changes less. So expect partial robustness, not immunity.
- **What to do instead of trusting the literature** [INF]:
  1. Build a small labelled before/after set from the user's own library and measure genuine-pair cosine similarity against the elapsed-time/weight gap.
  2. Use **multiple references spread across years** per person. Score a face against the max (or top-k mean) over those references, not against one centroid.
  3. Exploit **transitive chaining** through intermediate years in the clustering graph. Year 0 links to year 3, year 3 links to year 6, and so on, which bridges gaps a single direct comparison misses.

### Recommendation

- **Dev / personal (non-commercial):**
  - Primary: `minchul/cvlface_adaface_vit_base_kprpe_webface12m`. It has the best TinyFace (low-res), CPLFW (pose) and AgeDB (age) combination found.
    - Needs keypoints from `minchul/cvlface_DFA_mobilenet`.
    - PyTorch with `trust_remote_code`; MPS-capable [INF].
  - Simpler fallback: `cvlface_adaface_ir101_webface12m`.
  - Baseline / easiest ONNX path: **buffalo_l** (what Immich ships).
- **Mac:**
  - IR101 via PyTorch MPS, or ONNX models via onnxruntime CoreML EP [INF].
  - EdgeFace-S is the low-power option (non-commercial).
- **Commercial-safe:**
  - **AuraFace-v1 recognizer** (Apache-2.0; commercial training data claimed; AgeDB 96.10, CPLFW 90.93), clearly weaker than AdaFace on pose.
  - Second options: ModelScope CurricularFace-IR101 (Apache-2.0 label) or TransFace (MIT label), both with Glint360K data caveats.
  - Then SFace (Apache-2.0, provenance unclear) or dlib (CC0, weakest).
  - Or license InsightFace commercially.

---

## 3. Person re-identification / whole-body identity

### Candidates

Numbers are mAP / Rank-1.

| Model | Repo / weights | MSMT17 | Market-1501 | License |
|---|---|---|---|---|
| CLIP-ReID ViT-B/16 (AAAI 2023) | github.com/Syliz517/CLIP-ReID (Google Drive) | 73.4 / 88.7; +SIE+OLP 75.8 / 89.7 | 89.6 / 95.5 | MIT code; CLIP base MIT [DOC: arXiv 2211.13977] |
| SOLIDER Swin-B (CVPR 2023) | github.com/tinyvision/SOLIDER-REID | 77.1 / 90.7 | 93.9 / 96.9 | MIT code; pretrained on LUPerson (data terms UNVERIFIED) [DOC] |
| **PersonViT ViT-B** (2024) | github.com/hustvl/PersonViT; HF `lakeAGI/PersonViTReID` (not gated) | **80.8 / 92.0** | **95.0 / 97.6** | Apache-2.0 code; HF weights tagged plain "cc" (variant unspecified) [DOC: arXiv 2408.05398] |
| TransReID ViT-B | github.com/damo-cv/TransReID | 61.8 / 81.8 (baseline) | 87.1 / 94.6 | MIT |
| CAL (cloth-changing, CVPR 2022) | github.com/guxinqian/Simple-CCReID | LTCC clothes-changing 40.1 top-1 / 18.0 mAP (vs 74.2 / 40.8 general) | — | Apache-2.0 code; no weights linked [DOC: arXiv 2204.06890] |
| **SapiensID** (CVPR 2025) | github.com/mk-minchul/sapiensid | LTCC clothes-changing 66.30 top-1 / 42.35 mAP | — | **CC BY-NC 4.0** [DOC] |
| KPR (ECCV 2024) | VlSomers/keypoint_promptable_reidentification | — | — | Hippocratic License 3 (use-restricted, not OSI) |
| PLIP (NeurIPS 2024), person text-image model | github.com/Zplusdragon/PLIP | — | — | MIT code; SYNTH-PEDES dataset "commercial usage is forbidden" |
| ReID5o (NeurIPS 2025) | Zplusdragon/ReID5o_ORBench | — | — | Apache-2.0 code; weights on Baidu Cloud |

### Apple's design

Source: "Recognizing People in Photos Through Private On-Device Machine Learning" (2021) [DOC: https://machinelearning.apple.com/research/recognizing-people-photos].

- Apple computes face and upper-body embeddings separately.
- Upper-body embeddings "are less robust than face embeddings because they rely on a person's temporary appearance", so torso comparisons are allowed **only within a "moment"** (a time + location cluster).
- **Pass 1** is conservative agglomerative clustering on a fused distance, D_ij = min(F_ij, α·F_ij + β·T_ij), where F is face distance and T is torso distance.
- **Pass 2** is face-only HAC (hierarchical agglomerative clustering, merging the closest groups step by step) across moments, using median linkage.
- New faces are assigned to cluster exemplars by sparse coding.

### Open equivalent

No turnkey open equivalent exists [INF]. Immich, PhotoPrism, digiKam and LibrePhotos are all face-only.

You can assemble one from:
- an open face embedder,
- an open ReID embedder,
- EXIF time/GPS "moments".

SapiensID is the nearest single-model equivalent, but it is non-commercial.

### Key limitation

Market-1501 and MSMT17 assume the same clothes. On cloth-changing benchmarks accuracy roughly halves: CAL drops from 74.2 to 40.1 top-1 on LTCC.

### Recommendation [INF]

- Use ReID **only to link faceless bodies (back of head, far away) to a face-identified person within the same event**, following Apple's design. Never use it as a cross-day identity signal.
- Dev: PersonViT-B or CLIP-ReID ViT-B. Both are plain ViTs, easy to export to ONNX / Core ML.
- Research option for long-term identity: SapiensID.
- Commercial: CLIP-ReID (MIT code; you would retrain on licensable data, since Market/MSMT terms are research-oriented, UNVERIFIED).

---

## 4. Clustering faces into people at library scale (~300k–500k faces)

### What existing open photo apps do

**Immich** (verified from main-branch source)
- Defaults [DOC: https://github.com/immich-app/immich/blob/main/server/src/dtos/config.dto.ts]: `modelName: 'buffalo_l'`, `minScore: 0.7` (detection), `maxDistance: 0.5` (cosine distance), `minFaces: 3`.
- The docs suggest a max distance between 0.3 and 0.7 [DOC: https://docs.immich.app/features/facial-recognition].
- The algorithm (`person.service.ts`, `handleRecognizeFaces`) is an **incremental DBSCAN-like** procedure:
  1. kNN search within distance 0.5.
  2. A face is "core" if it has at least `minFaces` neighbours.
  3. Join the person of a neighbour, or the nearest already-assigned face; otherwise, if core, create a new person.
  4. Non-core faces are deferred, and a nightly job retries them.
- It is order-dependent [DOC: source].
- **Licensing:** Immich has a **private emailed permission** from InsightFace (Jia Guo, 2023-03-18) to use buffalo_l / antelopev2 [DOC: immich machine-learning README]. That permission does not transfer to you [INF].

**PhotoPrism** [DOC: https://docs.photoprism.app/developer-guide/vision/face-recognition/]
- Detector YuNet; embedding SFace (128-d) by default. FaceNet and AuraFace are optional.
- Clustering is DBSCAN.
- Defaults: minimum face size 25 px; clustering uses only faces ≥ 112 px with score ≥ 85; CLUSTER_CORE 5; CLUSTER_DIST 0.72; MATCH_DIST 0.25; MATCH_MARGIN 0.01.
- Pattern: cluster only on large, high-quality faces, then attach the rest.

**digiKam 8.6**
- YuNet + SFace, a KNN + SVM classifier ensemble, and face image quality assessment [DOC: https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/].

**LibrePhotos**
- PCA then HDBSCAN, then an MLP classifier trained on user labels. Default model is InsightFace `buffalo_sc` [DOC: librephotos `face_classify.py`, migrations 0125/0126].

### Benchmarks

Source: learn-to-cluster, MS1M part1_test, 584k faces / 8,573 identities, pairwise F-score [DOC: https://github.com/yl-1993/learn-to-cluster].

| Method | F-score | Notes |
|---|---|---|
| Chinese Whispers | 53.93 | — |
| KNN-DBSCAN | 67.93 | precision 95.25, recall 52.79 |
| FastHAC | 70.63 | — |
| L-GCN | 78.68 | — |
| GCN-V+E | 87.93 | — |
| STAR-FC | 91.97–93.21 | [DOC: github.com/sstzal/STAR-FC] |
| Ada-NETS | about 92.7 | [DOC: github.com/damo-cv/Ada-NETS] |

The GCN methods (graph convolutional networks: neural networks that run over the kNN graph) are **supervised on MS1M features from one specific embedder**:
- Change the embedder and you must retrain them.
- Their transfer to a personal library (few identities, heavy-tailed, children, aging) is untested [INF].
- For one user's library, density clustering plus human-in-the-loop corrections is the pragmatic choice [INF].

### Thresholds (always calibrate per model)

- **Immich** uses buffalo_l at cosine distance 0.5, i.e. similarity ≥ 0.5 [DOC].
- **deepface** ships cosine-distance thresholds of 0.55 for buffalo_l and 0.593 for SFace [DOC: deepface `threshold.py`].
- The cosine similarity corresponding to FAR=1e-4 for ArcFace-R100-class models is commonly cited as about 0.3–0.4 (UNVERIFIED).
- Calibrate on 1–2k labelled pairs from the user's own library. Use `osxphotos` / Apple Photos person labels as free ground truth if available [INF].

### Recipe [INF]

1. **Build the kNN graph.** FAISS, k = 50–100, over all face embeddings. 500k × 512-d is minutes on an H100 and feasible on an M-series Mac.
2. **Pass 1, high precision.**
   - Use only faces ≥ ~80–112 px with high detector and quality scores.
   - Link only pairs above a strict similarity (start around 0.55–0.6 for buffalo-class models and tune).
   - Run HDBSCAN (`sklearn.cluster.HDBSCAN`) or connected components with min size 3–5.
3. **Pass 2, attach.** Assign small, profile and blurry faces to the nearest cluster exemplars (max over exemplars, not the centroid, to handle age/weight drift), with a margin rule as in PhotoPrism's MATCH_MARGIN. Otherwise leave them unassigned.
4. **Pass 3, bodies.** Within an event, link faceless bodies via ReID (§3).
5. **Human in the loop.**
   - Rank candidate merges by inter-cluster similarity and ask the user yes/no.
   - Bias toward over-splitting: merges are cheap for users, splits are expensive.

---

## 5. Text-image retrieval embeddings (the "bread" problem)

### Comparison

Zero-shot numbers. "Params" are image tower + text tower.

| Model (HF id) | Params | Res | ImageNet-1k ZS | COCO T→I / I→T R@1 | Flickr T→I / I→T R@1 | License (commercial?) | Mac |
|---|---|---|---|---|---|---|---|
| openai/clip-vit-large-patch14-336 (baseline) | 304M + 124M | 336 | 76.2 (UNVERIFIED) | 37.1 / 57.9 | 66.9 / 87.7 | MIT, yes | MPS, MLX example |
| **facebook/PE-Core-L14-336** | 0.32B + 0.31B | 336 | 83.5 | **57.1 / 75.9** | 85.5 / 96.6 | **Apache-2.0**, yes; not gated | PyTorch / timm / open_clip; MPS (UNVERIFIED) |
| facebook/PE-Core-G14-448 | 1.88B + 0.47B | 448 | 85.4 | 58.1 / 75.4 | 85.7 / 96.2 | Apache-2.0 | too heavy for Mac |
| facebook/PE-Core-B16-224 | 0.09B + 0.31B | 224 | 78.4 | 50.9 / – | – | Apache-2.0 | yes |
| facebook/PE-Core-S16-384 / T16-384 | 0.02B / 0.01B image tower | 384 | 72.7 / 62.1 | 42.6 / 33.0 | – | Apache-2.0 | tiny |
| **google/siglip2-so400m-patch14-384** | ~0.4B + ~0.45B | 384 | 84.1 | 55.8 / 71.7 | 85.7 / 94.9 | **Apache-2.0**; not gated | MPS / ONNX |
| google/siglip2-so400m-patch16-512 | same | 512 | 84.3 | 56.0 / 71.3 | 85.5 / 95.4 | Apache-2.0 | MPS |
| google/siglip2-so400m-patch16-naflex | same | native aspect ratio | — | — | — | Apache-2.0 | MPS |
| google/siglip2-giant-opt-patch16-384 | ~1.1B + text | 384 | 85.0 | 56.1 / 72.8 | 86.0 / 95.4 | Apache-2.0 | heavy |
| google/siglip2-base-patch16-384 | ~86M + text | 384 | 80.6 | 54.6 / 71.4 | 83.8 / 94.9 | Apache-2.0 | yes |
| **qihoo360/fg-clip2-large** (FG-CLIP 2, arXiv 2510.10921) | ~0.3B + text | 336 | 83.0 | **58.6 / 75.1** | 84.8 / 96.6 | **Apache-2.0** | MPS (UNVERIFIED) |
| facebook/metaclip-2-worldwide-* (H/14 etc.) | ~0.63B + text | 224–378 | 81.3 (H/14) | multilingual focus (XM3600 T→I 51.5) | — | **CC-BY-NC-4.0, NO** | — |
| apple/DFN5B-CLIP-ViT-H-14-378 | 633M + 354M | 378 | 84.2 | 55.6 / 71.9 | 82.0 / 94.0 | **apple-amlr, NO** (research only) | — |
| apple/MobileCLIP2-S4 / S2 | 322M + 124M / 36M + 63M | 256 | 81.9 / 77.2 | 51.5 / 69.3 ; 48.8 / 66.7 | 78.0 / 92.4 ; 74.8 / 90.4 | **apple-amlr, NO** | best on-device (Core ML, iOS demo) |
| laion/CLIP-ViT-bigG-14-laion2B-39B-b160k | 1.8B + 0.7B | 224 | 80.1 | 51.4 / 67.4 | 79.6 / 92.9 | MIT | heavy |
| BAAI/EVA-CLIP-18B | 18B | 224 | ~83.8 (UNVERIFIED) | — | — | Apache-2.0 tag | impractical |
| jinaai/jina-clip-v2 | 0.3B + 0.56B | 512 | UNVERIFIED | UNVERIFIED | UNVERIFIED | **CC-BY-NC-4.0, NO** | ONNX |
| jinaai/jina-embeddings-v4 | ~4B (Qwen2.5-VL-3B) | dynamic | — | — | — | **Qwen Research License, NO** | — |
| nomic-ai/nomic-embed-vision-v1.5 | 93M | 224 | 71.0 | — | — | Apache-2.0 (relicensed) | ONNX; weak |
| **Qwen/Qwen3-VL-Embedding-2B / 8B** (Jan 2026, arXiv 2601.04720) | 2B / 8B VLM | dynamic, ≤1280 visual tokens | — | MMEB-V2 image retrieval 74.8 / 80.2; overall 73.2 / 77.9 | — | **Apache-2.0** | mlx-embeddings / community mlx-swift port |
| Alibaba-NLP/gme-Qwen2-VL-2B-Instruct | 2.2B | — | — | MMEB-V2 overall 59.1 (GME-7B) | — | Apache-2.0 | superseded |
| VLM2Vec/VLM2Vec-V2.0 | 2B/7B | — | — | MMEB-V2 overall 59.2 | — | Apache-2.0 | superseded |
| nvidia/MM-Embed | 8B | — | — | — | — | CC-BY-NC-4.0, NO | — |

Sources for the table:
- SigLIP 2 rows: Table 1 of https://arxiv.org/html/2502.14786 [DOC].
- PE-Core rows: https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md and Table 5 of arXiv 2504.13181 [DOC].
- OpenAI, bigG and DFN rows: open_clip `docs/openclip_retrieval_results.csv` [DOC].
- MobileCLIP2 rows: arXiv 2508.20691 and https://huggingface.co/apple/MobileCLIP2-S4 [DOC].
- MetaCLIP 2 row: arXiv 2507.22062 [DOC].
- FG-CLIP 2 row: arXiv 2510.10921 [DOC].
- Qwen3-VL-Embedding row: its HF cards [DOC].

**Comparability caveat.** These numbers come from different evaluation harnesses. FG-CLIP 2's paper re-ran SigLIP 2-L and got Flickr T→I 82.6 against SigLIP 2's own 85.0. Treat gaps of 1–2 points as noise [DOC/INF].

**Apple license detail.**
- The MobileCLIP/MobileCLIP2/DFN weights use the "Apple Machine Learning Research Model License". It grants use "exclusively for Research Purposes", and research purposes explicitly exclude "any commercial exploitation, product development or use in any commercial product or service" [DOC: https://huggingface.co/apple/MobileCLIP2-S4/blob/main/LICENSE].
- `apple/coreml-mobileclip` (v1 Core ML packages) is tagged `apple-ascl`. That conflicts with the AMLR terms, and its LICENSE file was not readable, so get a legal read before relying on it (UNVERIFIED).
- Personal use is fine.

**Other notable models**
- **TIPS / TIPSv2 (Google DeepMind):** weights CC-BY-4.0, commercial use with attribution. Patch–text alignment, useful for region scoring. TIPSv2 g/14 Flickr T→I 85.9 [DOC: https://github.com/google-deepmind/tips].
- **DINOv3 dino.txt:** custom DINOv3 License (commercial allowed with restrictions), gated by a request form. No retrieval numbers published [DOC: github.com/facebookresearch/dinov3 LICENSE.md].

### Compositional queries: which models fail and how

| Benchmark (paper) | What it tests | Result |
|---|---|---|
| ARO (Yuksekgonul et al., ICLR 2023, arXiv 2210.01936) | relations, attributes, word order | CLIP COCO-Order 0.46, below chance: it ignores word order. NegCLIP (trained with word-swapped hard negatives) reaches 0.86. VG-Relation 0.59 → 0.81 [DOC]. |
| SugarCrepe (Hsieh et al., NeurIPS 2023, arXiv 2306.14610) | replace/swap/add with fluent negatives | Earlier benchmarks were "hackable" by text-only models. NegCLIP-style gains were "hugely overestimated". "Swap" is hardest: OpenAI L/14 Swap-obj 60.2 [DOC]. |
| Winoground (Thrush et al., CVPR 2022, arXiv 2204.03162) | two images, two captions, same words in different order | CLIP B/32 group score 8.0 vs chance 16.7 and humans 85.5 [DOC]. |
| NegBench (Alhamoud et al., CVPR 2025, arXiv 2501.09425) | negation ("not", "without") | CLIP-family models are often at chance. Synthetic-negation fine-tuning gives +10% recall and +28% MCQ accuracy [DOC]. |
| CountBench (Paiss et al., ICCV 2023, arXiv 2302.12066) | counts 2–10 | CLIP-B/32 31.7% → 75.9% with count-aware fine-tuning [DOC]. |
| "CLIP Under the Microscope" (Abbasi et al., CVPR 2025, arXiv 2502.19842) | small objects | The image encoder is biased to **large objects**; the text encoder is biased to the **first-mentioned object**. Both trace to LAION data [DOC]. |
| arXiv 2604.11496 (2026) | SugarCrepe with newer encoders | PE scores 84.4 average vs CLIP 73.0 under plain global cosine. The paper attributes much of the failure to global pooling and recovers the gap with region-to-segment alignment over frozen features [DOC]. |
| CORE (arXiv 2609.04083, 2026) | VLM embedders and rerankers | Qwen3-VL-Embedding-8B: COLA 0.486, SugarCrepe++ 0.699, NegBench 0.500. **Qwen3-VL-Reranker-8B: COLA 0.767, SugarCrepe++ 0.856, but NegBench 0.261**. Rerankers fix attribute/relation binding and get **worse at negation** [DOC]. |

- SigLIP 2 Winoground/SugarCrepe numbers were not found (UNVERIFIED).

What the table means for this project:
- **Bag-of-words behaviour** (a global embedding ignores word order and binding: "dog chasing man" ≈ "man chasing dog") is reduced but not gone in PE-Core.
- **Negation and counting are not solved by any global embedding or by a VLM reranker.**
- **Small objects are lost in global pooling.** This is the mechanism behind the bread problem: one pooled vector is dominated by large regions.

### Mitigations [INF unless tagged]

1. **Multi-crop / tiled embeddings.**
   - Embed the full image plus, for example, a 2×2 or 3×3 grid of crops.
   - Score each image as the max over its vectors.
   - Costs N× index size, which is cheap on an H100.
2. **Higher resolution / NaFlex.**
   - Going from 256 to 512 px gains under 1 COCO point [DOC], but COCO captions describe salient content and understate small-object gains.
   - Helps, but not sufficient on its own.
3. **Tags and captions into a text index, fused with the dense score.** Options:
   - RAM++ `xinyu1205/recognize-anything-plus-model` (Apache-2.0)
   - Florence-2-large (MIT, 0.77B; captions, object detection, OCR)
   - OWLv2 (Apache-2.0)
   - Grounding DINO (Apache-2.0, UNVERIFIED)
   - Avoid **YOLO-World**, which is **GPL-3.0** [DOC: github.com/AILab-CVC/YOLO-World].
4. **Structured filters for counts and negation.** Use detector counts and scene labels instead of embeddings (see the NegBench and CORE results above).
5. **VLM reranking of the top-K** for attribute and relation binding (§6).

### Recommendation

- **Server index:** `facebook/PE-Core-L14-336`.
  - Apache-2.0.
  - Best documented COCO T→I at ~0.3B image params.
  - Best documented dual-encoder SugarCrepe.
  - Video-capable through frame averaging.
- **Alternatives:**
  - Add or swap in `google/siglip2-so400m-patch16-512` (or NaFlex) if non-English queries matter. PE-Core is English-centric (UNVERIFIED).
  - Evaluate `qihoo360/fg-clip2-large` for fine-grained "bread" queries. Its claims come only from its own paper.
- **Mac:**
  - Index on the H100 and ship vectors, so the Mac only runs the **text tower** at query time. That is cheap for every model listed [INF].
  - For on-device indexing of new photos: PE-Core-B16/S16 or SigLIP 2 B/16 (Apache-2.0).
  - MobileCLIP2 only if the project stays non-commercial.
- **Optional heavy tier:** Qwen3-VL-Embedding-2B (Apache-2.0) on the H100. Estimated 50–100 images/s, so under an hour for 150k [INF]. Impractical on a Mac (days) [INF].

---

## 6. VLMs for verification / reranking

### Releases newer than the candidate list in the brief

All Apache-2.0 and not gated unless noted [DOC: HF API]:
- **Qwen3.5** (Feb 2026): 0.8B, 2B, 4B, 9B, 27B, 35B-A3B (mixture-of-experts: 35B total, 3B active per token), and larger. Natively multimodal, image + video.
- **Qwen3.6-35B-A3B** (Apr 2026).
- **Qwen3.8-27B** (Aug 2026).
- **Gemma 4** (E2B, E4B, 12B, 26B-A4B, 31B). **Apache-2.0 and not gated**, unlike Gemma 3's Gemma Terms of Use.
- **MiniCPM-V 4.6** (1.3B, Apr 2026).
- **Molmo2** (4B, 8B; Dec 2025).
- **Qwen3-VL-Reranker / Qwen3-VL-Embedding** (2B, 8B; Jan 2026).
- **LFM2.5-VL** (LFM license, which has a revenue cap).

### License, gating and runtime support

| Model (HF id) | Params | License | Gated | Video | vLLM | mlx-vlm (Mac) |
|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-{0.8B,2B,4B,9B} | 0.9 / 2.3 / 4.7 / 9.7B | Apache-2.0 | no | yes | yes | yes (`qwen3_5`; mlx-community 2B/9B quantised repos) |
| Qwen/Qwen3.6-35B-A3B, Qwen3.8-27B | 36B MoE (3B active) / 27.8B | Apache-2.0 | no | yes | yes | yes (quantised repos) |
| Qwen/Qwen3-VL-{2B,4B,8B,32B}-Instruct, 30B-A3B | 2.1–33B | Apache-2.0 | no | yes | yes | yes (`qwen3_vl`) |
| Qwen/Qwen2.5-VL-3B-Instruct | 3.75B | **qwen-research, non-commercial** | no | yes | yes | yes |
| Qwen/Qwen2.5-VL-7B / 32B | 8.3 / 33.5B | Apache-2.0 | no | yes | yes | yes |
| Qwen/Qwen2.5-VL-72B | 72B | Qwen license (separate license needed above 100M MAU) | no | yes | yes | n/a on a laptop |
| OpenGVLab/InternVL3_5-{1B…241B} | 1–241B | Apache-2.0 | no | yes | yes | partial (`internvl_chat`; no 8B MLX repo found) |
| google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it | 5.1B (2.3B effective) … 31B | **Apache-2.0** | **no** | yes (+audio) | yes | yes (`gemma4`) |
| google/gemma-3-{4b,12b,27b}-it, gemma-3n | 4–27B | **Gemma Terms of Use** | **yes** | 3n: audio | yes | yes |
| openbmb/MiniCPM-V-4_5 / MiniCPM-V-4.6 | 8.7B / 1.3B | Apache-2.0 (registration optional) | no | yes | yes | 4.6 yes; 4.5 UNVERIFIED |
| allenai/Molmo2-{4B,8B} | 4.9 / 8.7B | Apache-2.0 tag, **but the card says it was trained on third-party data "for academic and non-commercial research use only"** | no | yes | yes | yes (`molmo2`) |
| HuggingFaceTB/SmolVLM2-{256M,500M,2.2B} | 0.26–2.25B | Apache-2.0 | no | yes | yes | yes |
| lmms-lab/LLaVA-OneVision-1.5-{4B,8B}-Instruct | 8.5B | Apache-2.0 | no | yes | UNVERIFIED | UNVERIFIED |
| microsoft/Phi-4-multimodal-instruct | 5.6B | MIT | no | no (image + audio) | yes | yes (`phi4mm`) |
| apple/FastVLM-{0.5B,1.5B,7B} | up to 7.8B | **apple-amlr, research only** | no | no | not listed | yes (`fastvlm`) |
| moonshotai/Kimi-VL-A3B-Instruct | 16.4B MoE | MIT | no | yes (UNVERIFIED) | yes | yes |
| zai-org/GLM-4.1V-9B-Thinking, GLM-4.6V-Flash | 10.3B | MIT | no | yes | yes | yes |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 109B MoE | Llama 4 Community License: **no rights for EU-domiciled entities to the multimodal models**; separate license above 700M MAU | yes | no | yes | impractical |
| LiquidAI/LFM2.5-VL-{450M,1.6B,3B} | 0.45–3.1B | **lfm1.0: not licensed for commercial use at ≥ $10M annual revenue** | no | yes | yes | official MLX repos |

Sources for the table:
- vLLM support: https://docs.vllm.ai/en/latest/models/supported_models.html [DOC].
- MLX support: https://github.com/Blaizzy/mlx-vlm model directories [DOC].
- Licenses: HF API and LICENSE files [DOC]:
  - https://huggingface.co/apple/FastVLM-7B/raw/main/LICENSE
  - https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE
  - https://huggingface.co/LiquidAI/LFM2.5-VL-3B/raw/main/LICENSE

### Benchmarks most relevant to yes/no verification

| Model | HallusionBench | POPE | MMMU | RealWorldQA | Video-MME (w/ sub) |
|---|---|---|---|---|---|
| Qwen3.5-9B | **69.3** | — | 78.4 | 80.3 | 84.5 |
| Qwen3.5-4B | 65.0 | 86.0 | 77.6 | 79.5 | 83.5 |
| Qwen3-VL-4B | 64.1 | — | 70.8 | 73.2 | 76.0 |
| Qwen3.5-2B (thinking / non-thinking) | 58.0 / 51.3 | 88.7 | 64.2 | 74.5 / 71.2 | 75.6 |
| InternVL3.5-8B | 54.5 | 88.7 | 73.4 | — | — |
| Gemma 4 26B-A4B / 31B | 66.1 / 59.9 | — | 78.4 / 79.6 | 72.2 / 70.3 | — |
| Gemma 4 E4B | — | 86.9 | — | 61.8 | — |

Sources for the table [DOC]:
- Qwen3.5 rows: https://huggingface.co/Qwen/Qwen3.5-4B and https://huggingface.co/Qwen/Qwen3.5-2B cards.
- POPE for Qwen3.5 and Gemma 4 E4B: Liquid's evaluation on https://huggingface.co/LiquidAI/LFM2.5-VL-3B.
- InternVL3.5 row: arXiv 2508.18265.
- Gemma 4 26B/31B rows: https://huggingface.co/Qwen/Qwen3.6-35B-A3B comparison table.

How to read the benchmarks:
- These are vendor-reported numbers.
- **POPE is saturated (86–89) and does not separate the candidates. HallusionBench does.**
- **No public benchmark measures graded attribute judgments** ("smiling", "looks heavier"). You need a small labelled set of your own [INF].

### Yes-probability scoring (the core verification mechanism)

- **VQAScore** (Lin et al., ECCV 2024, arXiv 2404.01291) defines alignment as P("Yes" | image, "Does this figure show '{text}'? Please answer yes or no."). It **beats CLIPScore on compositional prompts** [DOC]. The `t2v_metrics` library implements it for Qwen2.5-VL, Qwen3-VL, Qwen3.5 and Gemma 3 [DOC: https://github.com/linzhiqiu/t2v_metrics].
- **Qwen3-VL-Reranker** scores relevance as sigmoid(logit_yes − logit_no) [DOC: arXiv 2601.04720].
  - It is trained for query relevance, not for arbitrary predicates [INF].
  - It improves attribute and relation binding but fails at negation (CORE, §5).
- **Practical gotchas:**
  - **Disable thinking.** Qwen3.5-4B/9B default to thinking mode; set `enable_thinking: False`. Otherwise the first token is not Yes/No [DOC: Qwen3.5 cards].
  - **Cap image resolution.** Qwen3-VL/3.5 use one LLM token per 32×32 px, and the default max is about 16.7 MP. A 12 MP iPhone photo therefore becomes **~11.9k visual tokens**. Cap at roughly 0.5–1 MP [DOC: Qwen3-VL-8B `preprocessor_config.json`; arithmetic INF].
  - **Calibrate.** Raw P(Yes) is biased by the prompt template and by yes-bias. Fit temperature or Platt scaling per template on 100–300 labelled pairs. Measure refusal and hedging rates on sensitive prompts such as body weight [INF].
- **Runtime:**
  - vLLM: `logprobs` / `top_logprobs` with `max_tokens=1`. Read the raw top-k rather than constrained `choice` decoding, because whether vLLM reports logprobs before or after constraint masking is UNVERIFIED.
  - mlx-vlm server: `top_logprobs` up to 20 [DOC: mlx-vlm README].

### Throughput estimates [INF]

No published H100 "images/s for one-token yes/no" benchmark was found. Estimates are FLOP-based: prefill-bound, about 400 effective TFLOPS bf16.

| Model size | ~800 visual tokens (~1024×768) | ~256 tokens (512 px cap) |
|---|---|---|
| 8–9B (Qwen3.5-9B, Qwen3-VL-8B) | ~20–25 img/s | ~60–70 img/s |
| 4B | ~40 img/s | 100+ img/s |
| 2B | ~60–80 img/s (likely JPEG-decode-bound) | — |

- FP8 adds roughly 1.5×.
- Reranking the top 100 per query takes about 1–4 s on one H100.

**Mac:**
- llama.cpp community numbers for a 7B Q4 model, prompt processing over 512 tokens [DOC: https://github.com/ggml-org/llama.cpp/discussions/4167]: M4 Pro ~440 tok/s, M4 Max ~886 tok/s, M5 Max ~3220 tok/s.
- So an 8–9B VLM is about 1.8 s per image on an M4 Pro and about 0.25 s on an M5 Max. A 2B model is about 4× faster [INF].
- On a laptop, verify only the top 10–30 with a 2–4B model.

### Recommendation

| Use | Model | Notes |
|---|---|---|
| H100 verifier | `Qwen/Qwen3.5-9B` (non-thinking, P(Yes)) | Best HallusionBench in class; Apache-2.0 |
| H100 cheaper verifier | `Qwen/Qwen3.5-4B` | Apache-2.0 |
| Relevance reranker | `Qwen/Qwen3-VL-Reranker-2B` | Apache-2.0; vLLM pooling runner, vLLM ≥ 0.14 per a vLLM project post |
| Mac | `Qwen3.5-2B` or `4B` via mlx-vlm, 4-bit | Top 10–30 candidates only |
| Alternative family | `google/gemma-4-E4B-it` | Apache-2.0, not gated, MLX available |

- Use `Qwen3.5-2B` or `4B` as both verifier and query parser on the Mac.
- Avoid in a product:
  - FastVLM (apple-amlr)
  - Qwen2.5-VL-3B (non-commercial)
  - Gemma 3 (custom terms, gated)
  - Molmo2 (data restriction in card)
  - LFM (revenue cap)
  - Llama 4 (EU exclusion for multimodal)

---

## 7. Small local LLM for query parsing → JSON plan

| Model | Params | License | Gated | Reported tool-use / instruction-following | Mac |
|---|---|---|---|---|---|
| Qwen/Qwen3-4B-Instruct-2507 | 4.0B | Apache-2.0 | no | BFCL-v3 61.9; IFEval 83.4 [DOC: card] | mlx-community, GGUF |
| **Qwen/Qwen3.5-4B** (also the VLM) | 4.7B | Apache-2.0 | no | IFEval 89.8; BFCL-v4 50.3 [DOC: card] | mlx-vlm |
| Qwen/Qwen3.5-2B | 2.3B | Apache-2.0 | no | IFEval 61.2 (non-thinking) / 78.6 (thinking); BFCL-v4 43.6 [DOC] | mlx-vlm |
| ibm-granite/granite-4.0-micro | 3.4B | Apache-2.0 | no | BFCL-v3 59.98 [DOC: card] | mlx-community |
| ibm-granite/granite-4.2-3b (Aug 2026) | ~3B | Apache-2.0 | no | BFCL-v4 52.41 [DOC] | UNVERIFIED |
| HuggingFaceTB/SmolLM3-3B | 3.1B | Apache-2.0 | no | IFEval 76.7 [DOC] | mlx-community |
| microsoft/Phi-4-mini-instruct | 3.8B | MIT | no | function calling supported; score UNVERIFIED | mlx-community |
| meta-llama/Llama-3.2-3B-Instruct | 3.2B | Llama 3.2 Community License (the EU restriction applies only to the multimodal 3.2 models [DOC: Llama 3.2 USE_POLICY]) | **yes** | weak on BFCL at 1B (21.4, per Liquid's eval) | mlx-community |
| google/gemma-3-1b-it / 4b | 1–4B | Gemma Terms of Use | **yes** | Gemma-3-1B BFCL-v3 16.6 (per Liquid's eval) | yes |
| LiquidAI/LFM2.5-1.2B-Instruct | 1.2B | lfm1.0 ($10M revenue cap) | no | BFCL-v3 49.1; IFEval 86.2 [DOC] | official MLX |

### Structured output

Constrained decoding means the runtime masks tokens so the output must match a JSON schema or grammar.

- **vLLM:** `response_format: json_schema` or `structured_outputs` (json / regex / choice / grammar), using the xgrammar or guidance backends [DOC: vLLM `structured_outputs.md`].
- **llama.cpp:** GBNF grammars and JSON-schema → grammar conversion [DOC: llama.cpp `grammars/README.md`].
- **Ollama:** `format` accepts a JSON schema [DOC: ollama `docs/api.md`].
- **mlx-vlm server:** `json_schema` [DOC].
- **mlx-lm:** exposes `logits_processors` hooks for Outlines-style constraints [DOC]. Native Outlines-MLX support is UNVERIFIED.

### Mac latency [INF]

- A 4B 4-bit model generates about 80–90 tok/s on an M4 Pro and 140–200 tok/s on M4/M5 Max (scaled from llama.cpp #4167).
- A ~100-token JSON plan therefore takes about 0.5–1.5 s.
- Thinking mode multiplies this. Keep the parser in non-thinking mode.

### Recommendation

- **Use the same Qwen3.5-4B (or 2B) checkpoint as both query parser and verifier**, so only one model sits in memory on the Mac. Run it non-thinking with JSON-schema-constrained decoding.
- `Qwen3-4B-Instruct-2507` is a text-only alternative that is non-thinking by design.
- Relative dates ("summer 2019", "when I was in college") need user-profile facts. Resolve them in code after parsing, not inside the LLM [INF].

---

## 8. Video indexing

### What existing tools do

- **Immich** runs ML only on the generated thumbnail: "If a face is visible in the video's thumbnail it will be picked up by facial recognition" [DOC: https://docs.immich.app/FAQ/].
  - The FAQ says this only about faces. That CLIP search is likewise thumbnail-only is a secondary-source claim, from immich-moments.
- **immich-moments** [DOC: https://github.com/Booyaka101/immich-moments] is a useful reference design:
  - PySceneDetect ContentDetector (threshold 27, scenes 1.5–20 s).
  - One CLIP embedding per scene.
  - Face matching against Immich people.
  - faster-whisper transcripts in SQLite FTS5.

### Frame sampling: evidence

- **CLIP4Clip** (arXiv 2104.08860): parameter-free **mean pooling** of per-frame CLIP embeddings is the best option without lots of training data [DOC].
- **Lei, Berg & Bansal, "Revealing Single Frame Bias"** (arXiv 2206.03428): a single-frame model plus a frame ensemble matches multi-frame models on standard benchmarks, because of "static appearance bias" [DOC].
- **PE-Core** builds video embeddings by averaging 8 uniformly sampled frames [DOC: arXiv 2504.13181].

### Recommended sampling [INF]

Phone clips are mostly single continuous takes with **no cuts**, so shot detectors return one shot per clip. Instead:

1. Decode at **1 frame per 1–2 s**.
2. Embed each frame with the same image model used for photos.
3. Keep a frame only if its cosine distance from the last kept frame exceeds a threshold. Always keep the first and last frames, with a maximum gap of about 10–20 s.
4. Store **per-segment vectors with timestamps** plus a **mean-pooled clip vector**.
5. Run face detection on the kept frames.

At 10–20% videos this adds a few hundred thousand vectors, which is trivial for FAISS.

### Shot detection

Only needed for edited content (Memories exports, forwarded clips).

| Detector | License | Results |
|---|---|---|
| PySceneDetect AdaptiveDetector | BSD-3 | BBC F1 91.59; AutoShot (short-video set) F1 73.86 [DOC: PySceneDetect `benchmark/README.md`] |
| TransNetV2 | MIT | ClipShots 77.9 / BBC 96.2 / RAI 93.9 F1; GPU, 48×27 input [DOC: github.com/soCzech/TransNetV2] |
| AutoShot (CVPRW 2023) | MIT | Weights only on Baidu Drive [DOC] |
| OmniShotCut (2026, arXiv 2604.24762) | UNVERIFIED | F1 0.881 vs TransNetV2 0.814 on its own bench [DOC] |

### Video embedding models

MSR-VTT zero-shot text-to-video R@1:

| Model | MSR-VTT T2V R@1 | License / notes |
|---|---|---|
| InternVideo2-Stage2-1B | 51.9 | HF says Apache-2.0, **gated** with a use agreement [DOC: arXiv 2403.15377; HF card] |
| InternVideo2-6B | 55.9 | MIT on card [DOC] |
| VideoPrism-LvT-B / L | 50.1 | CC-BY-4.0 weights; **JAX only** [DOC: github.com/google-deepmind/videoprism] |
| **PE-Core-L (8-frame average)** | **50.3** | Apache-2.0 |
| PE-Core-G (8-frame average) | 51.2 | Apache-2.0 |
| LanguageBind-Video | 42.6 | — |
| Qwen3-VL-Embedding-8B | MSR-VTT not reported; MMEB-V2 video retrieval 58.7; up to 64 frames | Apache-2.0 [DOC] |
| ViCLIP | — | trained on InternVid (**CC BY-NC-SA 4.0**) |

**Conclusion [INF]:** PE-Core frame averaging is within about 1–5 points of dedicated video encoders, on a benchmark known to have static-frame bias. So use **the same image model on keyframes**:
- one embedding space for photos and videos,
- free temporal localisation,
- nothing extra to ship to the Mac.

A dedicated video model is an optional second index for motion queries ("jumping into the pool").

### Decoding on Linux GPU nodes

- **decord:** last release 0.6.0 in June 2021, alpha status [DOC: PyPI]. **Do not use** for iPhone HEVC 10-bit.
- **torchcodec** (Meta, BSD-3) is the recommended default [DOC: github.com/meta-pytorch/torchcodec/releases].

  | Version | Change |
  |---|---|
  | 0.8 | Beta CUDA (NVDEC) backend and robust 10-bit |
  | 0.14 | HDR decode, with float32 output that preserves range |
  | 0.16 | HEIC/AVIF image decode and batched nvJPEG |
  | 0.17 (Sep 30, 2026) | Native NVDEC for HEVC 4:4:4 at 8/10/12-bit; needs torch ≥ 2.11 |

  Codec coverage follows the system FFmpeg build.
- Alternatives:
  - **PyNvVideoCodec 2.x** (NVIDIA, MIT; direct NVDEC, decoder caching for short clips) [DOC: NVIDIA developer blog].
  - `ffmpeg -hwaccel cuda` / `hevc_cuvid` [DOC: NVIDIA Video Codec SDK FFmpeg guide].
  - DALI (its HEVC 10-bit support is UNVERIFIED).
- **Hardware:** H100 and H200 each have **7 NVDEC engines** with HEVC Main/Main10 decode. **Hopper cannot hardware-decode AV1** (Blackwell can) [DOC: https://docs.nvidia.com/dynamo/multimodal/video-decode-gpu-requirements].

### iPhone HDR gotcha

- iPhone 12 and later record **Dolby Vision Profile 8.4**: HEVC 10-bit with an HLG base layer [DOC: Apple "Incorporating HDR video with Dolby Vision" PDF].
- Converting to 8-bit RGB without tone-mapping gives **washed-out frames**, which will shift CLIP and face embeddings [DOC for the washing out; INF for the embedding effect].
- Detect HDR with ffprobe: `color_transfer` = `arib-std-b67` (HLG) or `smpte2084` (PQ).
- Tone-map on CPU with the ffmpeg `zscale` + `tonemap=hable` chain, or on GPU with `libplacebo` (`apply_dolbyvision=true`) [DOC: libplacebo filter docs].
  - libplacebo needs Vulkan, which may be absent on headless HPC nodes [INF].

### Stills and Live Photos

- **HEIC on Linux:** pillow-heif (BSD-3; bundles LGPL libheif/libde265) [DOC: github.com/bigcat88/pillow_heif], or torchcodec ≥ 0.16.
- **HEVC patents:** decoding for a personal, non-distributed tool is the practical norm. Bundling decoders in a distributed app needs review [INF; not legal advice].
- **Live Photos** are a HEIC still plus a ~3 s MOV. Index the still, and attach the MOV as a child asset so it does not produce duplicate hits [INF].

### Mac and audio

- **Mac:** use AVFoundation / VideoToolbox hardware HEVC decode, which handles HDR and Dolby Vision natively (UNVERIFIED in these passes, but standard platform behaviour).
- **Audio:** `openai/whisper-large-v3-turbo` (MIT, 809M params, ~8× faster than large-v3) [DOC: HF card]. Put transcripts in a BM25/FTS index alongside the visual index.

---

## 9. Body-weight / appearance estimation

### What the literature supports

| Work | Data | Result |
|---|---|---|
| Wen & Guo 2013, Image and Vision Computing | ~14.5k MORPH-II mugshots | Geometric face ratios (cheek-to-jaw width, width/height, perimeter/area); BMI MAE 2.65–4.29 depending on group [DOC: as cited in Dantcheva et al.] |
| Kocabey et al. 2017, ICWSM, "Face-to-BMI" | VisualBMI, 16.5k Reddit r/progresspics images, self-reported labels | VGG-Face features + SVR; Pearson **r = 0.65** (men 0.71, women 0.57) [DOC: arXiv 1703.03156] |
| Dantcheva et al. 2018, ICPR | 1,026 celebrities, web-sourced BMI | BMI MAE ≈ 2.3; weight MAE ≈ 8 kg [DOC: https://www-sop.inria.fr/members/Antitza.Dantcheva/show_me_your_face.pdf] |
| Siddiqui et al. 2020 | 5 CNNs | MAE 1.04–6.48; the 1.04 is on a small Bollywood set, likely optimistic [DOC: arXiv 2010.07442; INF] |
| "Digital Scale" 2025, arXiv 2508.20534 | 85k smartphone images from a weight-loss app | Full body MAPE 7.9% (MAE 2.56 BMI); face-only 11.1%; 13.4% on an unseen dataset. Weights not public. Authors warn about misclassification across clinical thresholds [DOC] |

How to read these numbers [INF]:
- r = 0.65 means only ~42% of BMI variance is explained.
- An MAE of 2.3–4 BMI is roughly 7–12 kg for a 1.75 m adult, comparable to many people's entire weight swing.
- Training sets are mugshots, celebrities and weight-loss posters with self-reported labels: strong selection bias.
- **No model was validated for within-person change across casual phone photos.**

### Facial adiposity

- Coetzee, Perrett & Stephen 2009, "Facial adiposity: a cue to health?", Perception 38(11) [DOC: PMID 20120267] argues facial adiposity is a perceivable cue to health. Its exact correlations are UNVERIFIED in these passes.
- A 2025 PLOS One scoping review ("Is the human face a biomarker of health?") calls the evidence "inconclusive" [DOC: DOI 10.1371/journal.pone.0318138].

### Confounders [INF]

These can swamp a few-kg signal:
- Selfie-distance perspective distortion: close-range photos enlarge the nose and narrow the face width ratio (Ward et al. 2018, JAMA Facial Plastic Surgery, PMID 29494735; magnitudes UNVERIFIED).
- Which lens was used (ultra-wide, 1×, 3×).
- Head yaw and pitch; smiling, which widens the lower face.
- Hair and beard; lighting; beauty filters; clothing on body crops.
- Aging.

### Body-shape models

| Model | License | Notes |
|---|---|---|
| SMPL / SMPL-X | **Non-commercial**; also prohibits surveillance use; commercial license only via Meshcapade [DOC: https://smpl-x.is.tue.mpg.de/modellicense.html] | Base body model for most methods below |
| SHAPY (CVPR 2022) | Non-commercial research [DOC: github.com/muelea/shapy] | Regresses SMPL-X shape plus measurements |
| HMR2.0 / 4D-Humans | MIT code, but needs SMPL | — |
| CameraHMR | UNVERIFIED | — |
| Multi-HMR | UNVERIFIED (likely NC) | — |
| **SAM 3D Body** (Meta, Nov 2025) + **MHR** body model | SAM License (commercial allowed, with trade-control terms); MHR is Apache-2.0; HF-gated [DOC: github.com/facebookresearch/sam-3d-body; github.com/facebookresearch/MHR] | The only license-clean path to body shape found |

- Image-to-mesh regressors **regress toward the mean body shape**, because they are trained largely on near-average pseudo-ground-truth [INF; this is the stated motivation of SHAPY and CameraHMR].
- Do not trust per-photo shape coefficients for a few-kg difference.

### Aesthetic / quality predictors

| Model | License | Notes |
|---|---|---|
| **improved-aesthetic-predictor** (CLIP ViT-L/14 + MLP) | Apache-2.0 [DOC: github.com/christophschuhmann/improved-aesthetic-predictor] | Nearly free on top of CLIP embeddings; the best fit for a Mac app [INF] |
| Q-Align / OneAlign | **S-Lab License, non-commercial** [DOC] | — |
| VisualQuality-R1 (Qwen2.5-VL-7B) | Apache-2.0 | Technical quality only, not aesthetics [DOC] |
| pyiqa (IQA-PyTorch) | **PolyForm Noncommercial 1.0.0** [DOC: github.com/chaofengc/IQA-PyTorch] | Wrapper for NIMA / MUSIQ / TOPIQ etc.; wrapped models' own licenses vary |

- Apple's on-device aesthetics model is not open.

### VLM bias

- **SocialCounterfactuals** (Howard et al., CVPR 2024, arXiv 2312.00825) found that VLMs associate physical characteristics (including "obese" and "skinny") with occupation stereotypes [DOC].
- An LVLM follow-up (arXiv 2405.20152) shows physical characteristics shift competence and toxicity outputs [DOC].
- **No study was found validating VLM "looks overweight" judgments against ground truth** (UNVERIFIED absence).
- Inference: a CLIP query like "overweight person" retrieves stereotype-correlated content (food, setting, clothing), not adiposity.

### Realistic vs unreliable

| Realistic | Unreliable or ethically sensitive |
|---|---|
| A within-person **relative** index for one consenting user: landmark face-width ratios on near-frontal, non-selfie-distance photos, aggregated per month and shown as a trend with error bands | Absolute BMI or weight from a single photo (MAE 2.3–4 BMI) |
| Validation against the user's own dated weight log (Spearman rank correlation) before enabling | VLM labels such as "overweight" or "obese", which are bias-prone and unvalidated |
| Aesthetic and quality ranking (CLIP-MLP) | Any body metric computed on other people in the library |

### Ethics and legal [INF; not legal advice]

- Inferred body weight is health-related data. Under GDPR it is **special-category data (Art. 9)**.
- Computing it for non-consenting third parties (friends, children) is the main risk.
- Apple Photos and Google Photos offer no such search, which I infer is deliberate.

### Recommendation

- Make it opt-in and limited to the user's own confirmed identity cluster.
- Show a relative trend only. Never show a BMI number or labels like "overweight".
- Keep everything on-device and deletable.
- Ship it only if the index correlates with the user's own weight log.

---

## 10. Recommended stacks

### A. Recommended stack (dev on H100, personal / non-commercial use; best accuracy)

| Component | Model | Weight license | Why |
|---|---|---|---|
| Face detection | InsightFace SCRFD-10G (buffalo_l `det_10g`) or SCRFD-34G, at ≥1280 px or tiled | Non-commercial research | WIDER FACE Hard 82.8–85.3 at VGA; ONNX; what Immich uses |
| Face embedding | `minchul/cvlface_adaface_vit_base_kprpe_webface12m` (fallback `cvlface_adaface_ir101_webface12m`; baseline buffalo_l) | Research-only in practice (dataset terms) | Best combination of AgeDB 98.1 (age), CPLFW 95.65 (pose), TinyFace 76.1 (low-res), IJB-C 97.82 |
| Clustering | FAISS kNN, then two-pass HDBSCAN / Immich-style density clustering, multi-exemplar attach, human merges | — (code) | GCN methods need retraining per embedder; personal libraries are small and heavy-tailed |
| Body / ReID linking | PersonViT-B (`lakeAGI/PersonViTReID`) or CLIP-ReID ViT-B, **within-event only**; research: SapiensID | Apache/MIT code; weights "cc" / research data | Best Market/MSMT numbers; clothing changes across days break ReID |
| Text-image retrieval | `facebook/PE-Core-L14-336` + tiled crops; evaluate `qihoo360/fg-clip2-large`; SigLIP 2 So400m if multilingual | Apache-2.0 | Top COCO T→I at 0.3B; best dual-encoder SugarCrepe; video-capable |
| Tags / captions (small objects, counts) | RAM++ + Florence-2-large; OWLv2 for counts | Apache-2.0 / MIT | Fixes pooling dilution; makes counts and negation structured |
| VLM verifier | `Qwen/Qwen3.5-9B` (H100), P(Yes) scoring, non-thinking, ≤1 MP | Apache-2.0 | Best small-model HallusionBench; vLLM + MLX support; video-capable |
| Reranker | `Qwen/Qwen3-VL-Reranker-2B` | Apache-2.0 | Purpose-trained; improves attribute/relation binding (not negation) |
| Query parser | `Qwen/Qwen3.5-4B` (same model family), JSON-schema constrained | Apache-2.0 | IFEval 89.8; one model for parsing and verification on Mac |
| Video | torchcodec (NVDEC) + HDR tone-map → 0.5–1 fps + embedding-change dedup → PE-Core frames; TransNetV2 for edited clips; whisper-large-v3-turbo | BSD-3 / MIT | Same embedding space as photos; per-moment hits |
| Aesthetics | improved-aesthetic-predictor (CLIP-L MLP) | Apache-2.0 | Near-free |
| Weight trend | Landmark facial-width index on the user's own cluster, monthly aggregate, validated against a weight log | — | Absolute BMI estimators are too inaccurate |
| Mac runtime | Index on H100; ship vectors; on Mac run PE-Core text tower, Qwen3.5-2B/4B 4-bit via mlx-vlm, ONNX face models via onnxruntime CoreML EP | — | Heavy image-side work stays on GPU |

### B. License-safe alternative for a commercial product

| Component | Model | Weight license | Caveat |
|---|---|---|---|
| Face detection | ModelScope SCRFD-34G-GNKPS v2 (`immich-app/scrfd_34g_gnkps` ONNX); fallback YuNet; or Apple Vision on Mac | MIT / MIT / OS API | WIDER FACE is CC BY-NC-ND (training-data provenance) |
| Face embedding | `fal/AuraFace-v1` **recognizer only** (`glintr100.onnx`); second options: ModelScope CurricularFace-IR101 (Apache-2.0 label), SFace (Apache-2.0), dlib (CC0) | Apache-2.0 | Weaker on pose (CPLFW 90.9) and age (AgeDB 96.1) than AdaFace. **AuraFace bundles InsightFace's non-commercial SCRFD; do not ship that file.** Or buy an InsightFace license |
| Clustering | Same as A | — | — |
| Body linking | CLIP-ReID architecture retrained on licensable data, or skip | MIT code | Public ReID datasets are research-oriented (UNVERIFIED) |
| Retrieval | PE-Core-L14-336 / SigLIP 2 / FG-CLIP 2 | Apache-2.0 | Avoid MobileCLIP/DFN (apple-amlr), MetaCLIP 2 and jina-clip-v2 (CC-BY-NC), jina-embeddings-v4 (Qwen research license) |
| Tags | RAM++, Florence-2, OWLv2 | Apache-2.0 / MIT | Avoid YOLO-World (GPL-3.0) and Ultralytics YOLO (AGPL-3.0) |
| VLM verifier / parser | Qwen3.5-{2B,4B,9B}, or Gemma 4 E4B / 12B | Apache-2.0 | Avoid FastVLM (apple-amlr), Qwen2.5-VL-3B (research), Gemma 3 (custom terms), Llama 4 (EU exclusion), LFM (revenue cap), Molmo2 (card's data restriction) |
| Reranker | Qwen3-VL-Reranker-2B | Apache-2.0 | Avoid jina-reranker-m0 (CC-BY-NC) |
| Video | torchcodec / PyNvVideoCodec, PE-Core frames, TransNetV2, whisper-turbo | BSD-3 / MIT | HEVC patent licensing if you redistribute decoders; avoid ViCLIP (InternVid NC data) |
| Aesthetics | improved-aesthetic-predictor | Apache-2.0 | Avoid Q-Align (S-Lab NC) and pyiqa (PolyForm NC) |
| Body shape (if ever) | SAM 3D Body + MHR | SAM License / Apache-2.0 | Avoid SMPL/SMPL-X/SHAPY (non-commercial) |

### Main license traps

1. InsightFace code is MIT, but **every pretrained InsightFace model is non-commercial**: buffalo_l, antelopev2 and SCRFD [DOC]. Immich's permission is private to Immich.
2. **AuraFace's "Apache-2.0" repo ships InsightFace's SCRFD** byte-for-byte.
3. **Apple's open models** (MobileCLIP 1/2, DFN-CLIP, FastVLM) are research-only under apple-amlr.
4. HF license tags can be wrong:
   - LVFace is tagged "mit" but its README says the models are non-commercial.
   - EdgeFace is BSD-3 on GitHub but CC-BY-NC-SA on HF.
   - Molmo2 is tagged Apache-2.0, but its card restricts the training data to non-commercial use.
5. Training-data terms (WIDER FACE CC BY-NC-ND; WebFace, MS1M, Glint360K research terms) hang over nearly all face weights. No face model is fully clean, except possibly AuraFace (commercial training data claimed) and dlib's CC0 release (with its own dataset caveat).

---

## Sources

### Face detection and recognition
- InsightFace README and model zoo: https://github.com/deepinsight/insightface ; https://github.com/deepinsight/insightface/tree/master/model_zoo ; https://github.com/deepinsight/insightface/tree/master/detection/scrfd
- ModelScope SCRFD (MIT): https://modelscope.cn/api/v1/models/iic/cv_resnet_facedetection_scrfd10gkps ; HF rehost https://huggingface.co/immich-app/scrfd_34g_gnkps
- RetinaFace: https://github.com/biubug6/Pytorch_Retinaface
- YOLO-face: https://github.com/deepcam-cn/yolov5-face ; https://github.com/derronqi/yolov8-face ; https://github.com/akanametov/yolo-face
- YuNet: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- TinaFace: https://github.com/Media-Smart/vedadet
- WIDER FACE license: http://shuoyang1213.me/WIDERFACE/
- MediaPipe: https://github.com/google-ai-edge/mediapipe ; https://developers.google.com/edge/mediapipe/solutions/vision/face_detector
- Apple Vision (WWDC21): https://developer.apple.com/videos/play/wwdc2021/10040/
- CVLface: https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m ; https://raw.githubusercontent.com/mk-minchul/CVLface/main/cvlface/pretrained_models/README.md
- AdaFace: https://github.com/mk-minchul/AdaFace
- LVFace: https://github.com/bytedance/LVFace ; https://huggingface.co/bytedance-research/LVFace
- TopoFR / TransFace: https://github.com/DanJun6737/TopoFR ; https://github.com/DanJun6737/TransFace
- EdgeFace: https://huggingface.co/Idiap/EdgeFace-Base ; https://github.com/otroshi/edgeface
- AuraFace: https://huggingface.co/fal/AuraFace-v1
- SFace: https://huggingface.co/opencv/face_recognition_sface ; https://github.com/opencv/opencv_zoo/issues/313
- dlib models: https://github.com/davisking/dlib-models ; http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html
- facenet: https://github.com/timesler/facenet-pytorch ; https://github.com/davidsandberg/facenet
- deepface: https://github.com/serengil/deepface
- FRoundation: https://arxiv.org/abs/2410.23831
- osxphotos: https://github.com/RhetTbull/osxphotos

### Age and weight change
- Best-Rowden & Jain, TPAMI 2018: https://pubmed.ncbi.nlm.nih.gov/28092523/
- MTLFace: https://arxiv.org/abs/2103.01520
- Weight variation: https://iab-rubric.org/images/pdf/papers/2015_BTAS_RegularizingDeep.pdf ; https://ieeexplore.ieee.org/document/6996243/

### Person re-identification
- CLIP-ReID: https://github.com/Syliz517/CLIP-ReID ; https://arxiv.org/abs/2211.13977
- SOLIDER: https://github.com/tinyvision/SOLIDER-REID
- PersonViT: https://arxiv.org/html/2408.05398
- TransReID: https://github.com/damo-cv/TransReID
- CAL: https://arxiv.org/abs/2204.06890
- SapiensID: https://arxiv.org/html/2504.04708
- PLIP: https://github.com/Zplusdragon/PLIP
- Apple people recognition: https://machinelearning.apple.com/research/recognizing-people-photos

### Clustering and photo apps
- Immich: https://github.com/immich-app/immich/blob/main/server/src/dtos/config.dto.ts ; https://docs.immich.app/features/facial-recognition ; https://github.com/immich-app/immich/blob/main/machine-learning/README.md ; https://docs.immich.app/FAQ/
- PhotoPrism: https://docs.photoprism.app/developer-guide/vision/face-recognition/
- digiKam 8.6: https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/
- LibrePhotos: https://github.com/LibrePhotos/librephotos
- Clustering benchmarks: https://github.com/yl-1993/learn-to-cluster ; https://github.com/sstzal/STAR-FC ; https://github.com/damo-cv/Ada-NETS

### Text-image retrieval and compositionality
- SigLIP 2: https://arxiv.org/abs/2502.14786
- Perception Encoder: https://arxiv.org/abs/2504.13181 ; https://github.com/facebookresearch/perception_models
- MetaCLIP 2: https://arxiv.org/abs/2507.22062 ; https://github.com/facebookresearch/MetaCLIP
- MobileCLIP2: https://arxiv.org/abs/2508.20691 ; https://huggingface.co/apple/MobileCLIP2-S4/blob/main/LICENSE ; https://github.com/apple/ml-mobileclip
- FG-CLIP 2: https://arxiv.org/abs/2510.10921
- open_clip results: https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_retrieval_results.csv
- jina models: https://huggingface.co/jinaai/jina-clip-v2 ; https://huggingface.co/jinaai/jina-embeddings-v4
- Qwen3-VL-Embedding / Reranker: https://arxiv.org/abs/2601.04720 ; https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B ; https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B
- TIPS: https://github.com/google-deepmind/tips
- DINOv3: https://github.com/facebookresearch/dinov3
- ARO: https://arxiv.org/abs/2210.01936
- SugarCrepe: https://arxiv.org/abs/2306.14610
- Winoground: https://arxiv.org/abs/2204.03162
- NegBench: https://arxiv.org/abs/2501.09425
- CountBench: https://arxiv.org/abs/2302.12066
- CLIP Under the Microscope: https://arxiv.org/abs/2502.19842
- Local alignment over frozen features: https://arxiv.org/html/2604.11496v1
- CORE: https://arxiv.org/html/2609.04083
- Tagging and detection: https://huggingface.co/xinyu1205/recognize-anything-plus-model ; https://github.com/AILab-CVC/YOLO-World

### VLMs and query-parsing LLMs
- VQAScore: https://arxiv.org/abs/2404.01291 ; https://github.com/linzhiqiu/t2v_metrics
- Qwen3.5: https://huggingface.co/Qwen/Qwen3.5-4B ; https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.6: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Qwen3.8: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3-VL preprocessor config: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/preprocessor_config.json
- Gemma 4: https://ai.google.dev/gemma/docs/core/model_card_4
- InternVL3.5: https://arxiv.org/abs/2508.18265
- MiniCPM-V: https://huggingface.co/openbmb/MiniCPM-V-4_5
- Molmo2: https://huggingface.co/allenai/Molmo2-8B
- FastVLM license: https://huggingface.co/apple/FastVLM-7B/raw/main/LICENSE
- Qwen2.5-VL-3B license: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE
- LFM license: https://huggingface.co/LiquidAI/LFM2.5-VL-3B/raw/main/LICENSE
- Llama 3.2 use policy: https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/USE_POLICY.md
- vLLM: https://docs.vllm.ai/en/latest/models/supported_models.html ; https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/structured_outputs.md
- mlx-vlm: https://github.com/Blaizzy/mlx-vlm
- llama.cpp Apple Silicon benchmarks: https://github.com/ggml-org/llama.cpp/discussions/4167
- Qwen3-4B-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

### Video
- immich-moments: https://github.com/Booyaka101/immich-moments
- CLIP4Clip: https://arxiv.org/abs/2104.08860
- Single-frame bias: https://arxiv.org/abs/2206.03428
- PySceneDetect benchmark: https://github.com/Breakthrough/PySceneDetect/blob/main/benchmark/README.md
- TransNetV2: https://github.com/soCzech/TransNetV2
- AutoShot: https://github.com/wentaozhu/AutoShot
- OmniShotCut: https://arxiv.org/html/2604.24762v2
- InternVideo2: https://arxiv.org/abs/2403.15377
- VideoPrism: https://github.com/google-deepmind/videoprism
- torchcodec: https://github.com/meta-pytorch/torchcodec/releases
- decord: https://pypi.org/project/decord/
- NVIDIA decode requirements: https://docs.nvidia.com/dynamo/multimodal/video-decode-gpu-requirements
- pillow-heif: https://github.com/bigcat88/pillow_heif
- whisper-large-v3-turbo: https://huggingface.co/openai/whisper-large-v3-turbo

### Body weight, body shape, aesthetics and bias
- Face-to-BMI: https://arxiv.org/abs/1703.03156
- Dantcheva et al.: https://www-sop.inria.fr/members/Antitza.Dantcheva/show_me_your_face.pdf
- Siddiqui et al.: https://arxiv.org/abs/2010.07442
- Digital Scale: https://arxiv.org/html/2508.20534v1
- Facial adiposity: https://pubmed.ncbi.nlm.nih.gov/20120267/
- SMPL-X license: https://smpl-x.is.tue.mpg.de/modellicense.html
- SHAPY: https://github.com/muelea/shapy
- SAM 3D Body: https://github.com/facebookresearch/sam-3d-body
- MHR: https://github.com/facebookresearch/MHR
- improved-aesthetic-predictor: https://github.com/christophschuhmann/improved-aesthetic-predictor
- Q-Align: https://github.com/Q-Future/Q-Align
- pyiqa: https://github.com/chaofengc/IQA-PyTorch
- SocialCounterfactuals: https://arxiv.org/abs/2312.00825 ; LVLM follow-up https://arxiv.org/abs/2405.20152
