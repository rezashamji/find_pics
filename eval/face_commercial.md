# Face recognizer we may ship commercially (2026-10-06)

Goal: replace InsightFace buffalo_l (weights non-commercial) with a recognizer whose weights AND training data allow
commercial use in an iPhone App Store app.

## Candidates and licence evidence

| Model | Weights licence | Training data (licence) | Size (fp32) | Commercially clean? | Source |
|---|---|---|---|---|---|
| buffalo_l w600k_r50 (current) | non-commercial; commercial licence sold by InsightFace | WebFace600K (non-commercial) | 174 MB | No (unless licence bought) | https://www.insightface.ai/ |
| AuraFace-v1 glintr100 | Apache-2.0 | "commercial dataset ... from various sources" (undisclosed) | 261 MB | Yes per vendor statement; data not auditable | https://huggingface.co/fal/AuraFace-v1 |
| OpenCV Zoo SFace 2021dec | Apache-2.0 file in model dir | undocumented; SFace paper used CASIA-WebFace / VGGFace2 / MS1MV2 (all research-only) | 37 MB | Unresolved: issues #313 and #318 (open, 10-05) ask exactly this, no maintainer answer on provenance | https://github.com/opencv/opencv_zoo/issues/318 , https://github.com/opencv/opencv_zoo/issues/313 |
| HyperFace-10k-LDM / -50k-StyleGAN (Idiap, IResNet50) | MIT | synthetic HyperFace images; BUT generated with Arc2Face (trained on WebFace42M, research-only), an ArcFace model and StyleGAN/LDM galleries | 174 MB (44M params; ~87 MB fp16) | Grey: weights MIT, upstream generator/data chain is non-commercial | https://huggingface.co/Idiap/HyperFace-10k-LDM , https://arxiv.org/html/2411.08470v2 , https://github.com/foivospar/Arc2Face |
| EdgeFace (Idiap) | CC BY-NC-SA 4.0 | WebFace260M (non-commercial) | 2-18M params | No | https://huggingface.co/Idiap/EdgeFace-XS-GAMMA |
| Langevin-DisCo (Idiap, synthetic) | CC BY-NC-SA 4.0 | synthetic (StyleGAN2) | 44M params | No | https://huggingface.co/Idiap/Langevin-DisCo-30k |
| dlib resnet v1 | public domain | VGG + FaceScrub (non-commercial) + web scrape | 22 MB | No (data) | https://github.com/davisking/dlib-models |
| facenet-pytorch (VGGFace2) | MIT code | VGGFace2 (withdrawn by Oxford; image copyright with owners) | 107 MB | Risky | https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ |
| DigiFace-1M-trained models | n/a | DigiFace-1M is non-commercial research only | n/a | No | https://github.com/microsoft/DigiFace1M |
| Apple API | n/a | n/a | n/a | No face-identity API: Vision has detection/landmarks/quality only; PhotoKit does not expose People | https://developer.apple.com/forums/thread/688345 |

## Measurement (eval/eval_face_commercial.py, results in eval/results_face_commercial*.json)

Protocol = eval_face_models.py: 8 reference faces per identity, every other image of that identity a target, every other
identity's image a distractor; recall compared at EQUAL wrong-match rate (buffalo_l's rate at its 0.40 threshold, and 10x
lower). One shared detector + alignment (buffalo_l SCRFD + norm_crop 112 px) for all recognizers, so only the recognizer
differs. Aligned crops checked by eye: eval/audits/face_commercial_crops_{digiface,celeba}.jpg, 48/48 aligned in each.

- DigiFace: 300 identities, 21,600 images, 90 not detected; 19,110 targets, 6.43M distractor scores; buffalo rate 0.0021.
  None of the candidates was trained on DigiFace (no contamination).
- CelebA test split (real faces): 755 identities, 18,298 images, 3 not detected; 12,255 targets, 13.79M distractor
  scores; buffalo rate 1.15e-5 (159 wrong matches). Bias: real-data models (buffalo_l, maybe SFace/AuraFace) may have
  seen these celebrities in training; HyperFace cannot have.

| Model | DigiFace recall @0.0021 | @10x lower | CelebA recall @1.15e-5 | @10x lower |
|---|---|---|---|---|
| buffalo_l | 0.987 (18,855/19,110) | 0.965 | 0.979 (11,994/12,255) | 0.822 |
| AuraFace | 0.904 (17,272/19,110) | 0.804 | 0.908 (11,127/12,255) | 0.709 |
| SFace | 0.921 (17,601/19,110) | 0.839 | 0.942 (11,541/12,255) | 0.820 |
| HyperFace-10k-LDM (BGR, as documented) | 0.941 (17,982/19,110) | 0.876 | 0.798 (9,774/12,255) | 0.686 |
| HyperFace-50k-StyleGAN (BGR) | 0.788 | 0.250 | 0.013 (161/12,255) | 0.006 |

buffalo_l and AuraFace reproduce the 10-05 run exactly (0.987 / 0.904). HyperFace-50k is broken in this pipeline (CelebA
distractor scores so high that the matching threshold is 0.925; feeding RGB instead gives DigiFace 0.948 but CelebA
0.390), which contradicts its published LFW 98.3%; cause not found, so no claim about that model's real quality.

## Recommendation

No candidate is both clean and close. On real faces only SFace (0.942 vs 0.979) is near buffalo_l, and its training
data is undocumented (likely research-only sets). AuraFace is the only one with an explicit commercial-data statement,
and it loses 7 points (0.908). HyperFace-10k does well on synthetic faces but drops to 0.798 on real ones and its data
chain is non-commercial upstream. Options for Reza: (a) ship AuraFace and accept lower recall, (b) buy an InsightFace
commercial licence for buffalo_l, (c) get written provenance for SFace from OpenCV. Getting legal sign-off is his call;
the numbers support (b) on quality, then (a).
