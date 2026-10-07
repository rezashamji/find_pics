# Server pipeline vs iPhone app: every piece (inventory 10-05)

Built from the function list of every module in `src/findpics/` (4,228 lines). Status:
**verified** = Swift port gives identical answers to the Python on recorded cases (runs on Linux, `ios/FindPicsCore`);
**written** = Swift code exists but uses Apple-only frameworks, so it has never been compiled or run;
**missing** = not on the phone yet; **not needed** = server/testing tooling.

## Understanding the request
| server piece | what it does | phone |
|---|---|---|
| converse.build_prompt / TEMPLATE | the planner prompt | verified (58 prompts byte-identical) |
| converse.plan_turn | model -> JSON -> retry -> grounding | written (app Planner.swift) |
| converse.ground + planner.ground_dates / ground_place / strip_identity_conditions / fix_red_box | code rules on the plan | verified (2,300 plans, 1,459 changed by the rules) |
| resolve_relative / resolve_time_of_day | dates and clock times from words | verified (540 + 1,457 cases) |
| _unanswerable / _drop_unanswerable | questions one photo cannot answer | verified (inside the 2,300) |
| the planner MODEL | 9B on the server | open: Apple's on-device model vs our 4B (distilled 4B: 27/30 conversations vs 9B 29/30) — measure on the phone |

## Knowing what is in the picture
| server piece | what it does | phone |
|---|---|---|
| models.ImageTextEncoder (PE-Core-L) | photo/text vectors for "anything" | written: PE-Core-B-16 Core ML (converted; never run on a device). B-16 vs L measured close on the server (RESULTS 20, 22) |
| CLIP tokenizer | words -> ids for the text vectors | verified (1,445 requests) |
| index.py tiles (2x2 crops) | small objects | not used on the server either (decided off, RESULTS 13) |
| media.sample_video_frames / _frames_for | several frames per video, the matched frame | written: every 2 s, max 40, vectors + faces per frame; judge sees the matched frame (eval 10-05: one matched face frame already separates Reza's eras, AUC 0.981; more frames / video input no clear win) |
| models.FaceEncoder + people.* (face_sims, item_person_scores, expand_refs, other_identities, face_groups) + face_profiles | who is in the photo; "which face is you"; other-person rule | SHIPPED MODEL (10-07, Reza): AuraFace-v1 + flip (Apache-2.0), Core ML 130.8 MB fp16 with the mirror averaging inside the graph (torch = onnxruntime cosine 1.000000 on 8 aligned public faces); cuts recalibrated (FaceProfile == face_profiles.py, golden test); matching logic verified on synthetic faces at the new cuts; alignment verified (200 cases); detection = Apple Vision. Index records the face model per entry; a model change re-embeds the stored faces and re-derives saved people (FaceMigration). buffalo_l: server dev option only (non-commercial) |
| SubjectRefs / reference_crop / side_by_side + _dba | "this specific dog / thing / place" from example photos | ranking VERIFIED (FindPicsCore.subjectScores == Python _dba ranking, golden test); app path WRITTEN (SubjectSearch.swift: Accelerate ranking, side-by-side judge veto at 0.2, photo picker UI) |
| selfie rule (camera tag) | "selfies" = front camera + selfie question + default look | VERIFIED (grounding fixtures, scopeMask); app reads EXIF lens model (written) |
| judge / planner choice | Qwen (downloaded) vs Apple Foundation Models (rating or yes/no) | WRITTEN (AppleJudge.swift, AppleText, Model menu); Apple model gives no probabilities |
| ingest.reverse_geocode | GPS -> place names, offline | verified (2,000 points, identical to the server's new sphere-distance version + country names) |
| ingest dates / EXIF / video metadata | when and where | replaced by PhotoKit (creation date, location); local clock uses the phone's CURRENT time zone (server: the capture time zone) |

## Narrowing and checking
| server piece | what it does | phone |
|---|---|---|
| engine.scope_mask / time_of_day_mask | media, dates, clock, place filters | verified |
| engine.look_scores | ranking by meaning | verified (without tiles) |
| vlm.VLLMJudge.p_yes | yes/no per photo | written (Judge.swift, MLX Qwen3.5) |
| vlm.draw_box + engine.person_crop_boxed | red box on the person being judged | geometry verified (150 cases); drawing written |
| engine.stream_album / _finish + audit.certify / tail_sample_size | rounds + a stated completeness bound | verified (certificate = scipy on 300 cases; rounds: 0 overclaims in 170 replayed rounds); used by the app |
| converse._album_stream: anchors + agent.window_rows/events, until, with_people, exclude/filter | "the week I went to X", "after A before B", "me with Jay" | windows + events verified (900 cases; fixed an undated-photo bug in the Python on the way); exclude/filter written; anchor search, until, with_people not wired in the app yet |
| filter_to_place / place_or_look | place name -> GPS filter or a look | VERIFIED (FindPicsCore.filterToPlace / placeOrLook, 8 golden cases); wired into the app search |
| engine pairing (_two_groups / _split_pair) | heavier vs fit split | verified (7 cases) |
| engine.make_exclusive rank-margin fallback + _report | pairing fallback, album report text | rank margin VERIFIED (rankMarginPair, 40-photo golden case) and wired; report text: short notes only |
| bursts.burst_ids | "+N similar" stacks | verified; app grid uses it |
| converse.CachedJudge | reuse judge answers on follow-ups | written (in memory) |

## Not needed on the phone
apple_copy.py, cli.py, web.py, report.py, contact.py, albums.write_folder_album (PhotoKit save is written), store.py
(replaced by the app's index), agent.make_plan/execute (older planner), Session (chat state lives in the app).

## Biggest gaps, in order of what a person would notice
1. Faces (any "me" / "Dad" search): AuraFace + flip finds fewer of a person's photos than buffalo_l did (CelebA, equal
   wrong items, shipped cuts: 0.882 vs 0.976 with 3 refs, 0.916 vs 0.979 with 8; RESULTS 36); not yet measured on the phone.
2. The app has never been compiled or run (needs Reza's Mac on macOS Tahoe 26.6+ and Xcode 27).
3. Which model reads requests and judges photos on the phone (Apple's vs ours): accuracy and speed unmeasured.
4. Anchors / time windows ("the week I went to X"), place names, several frames per video.
5. Completeness bound in fast mode; judge-answer cache for follow-ups.

## RESULTS: indexing from the ~480 px local renditions (8ded039), measured 10-07 on public data
Method: simulate PhotoKit's rendition (long side 480, Lanczos, JPEG round trip), then the phone's 10-07 pipeline
(Pillow bilinear squash to 224 + fp16 weights); compare with the server vector of the FULL photo (<= 1600 px).
Sets: Open Images (1024 px) and Pexels video frames (1600x900). Scripts + JSON: eval/rendition480_parity.py,
eval/rendition480_decompose.py, eval/rendition480_chroma.py (-> same-name .json).
Image cosine vs server (mean / p5 / min):
- 480 + JPEG q80 (4:2:0):  Open Images 200: 0.955 / 0.932 / 0.921 (2000: 0.954 / 0.933 / 0.895); Pexels 200: 0.949 / 0.932 / 0.910
- 480, no JPEG (Lanczos):  0.998 / 0.996 / 0.981 and 0.996 / 0.992 / 0.981 -> the SIZE is harmless; the JPEG is the damage
- JPEG q80 at full size only: 0.981 and 0.990. 480 + q80 4:4:4: 0.970 / 0.969; 480 + q95 4:4:4: 0.992 / 0.992;
  480 + q95 4:2:0: 0.973 / 0.978. 448 instead of 480: 0.952 / 0.946. 960 + q80: 0.978 / 0.983. Extra blur: worse.
  So both JPEG quantization and chroma subsampling matter, and the real number depends on how PhotoKit encodes its
  derivatives (unknown here): anywhere from ~0.95 (q80 4:2:0) to ~0.99 (q95 4:4:4). Must be measured on the phone.
Top-k agreement with the server ranking, 20 everyday queries, 2000-photo pools, all via 480 + q80:
- Open Images: overlap@600 0.884 (min 0.843), @100 0.866, @20 0.858 (min 0.70). Pexels: see rendition480_parity.json.
  This is about the same disagreement as the pre-10-07 Core Image resize bug (0.880 on the same pool).
Quality proxy (Open Images verified labels, noisy; same labels for both paths), 72 classes with >= 15 positives in the
2000 pool, query "a photo of a <class>": mean AP server 0.438 vs 480 path 0.440; 27 classes better, 18 worse (|d| >
0.005). So for whole-photo classes the 480 vectors DIFFER from the server's but do not rank worse. Not tested: small
objects (the case the product most worries about).
Faces (InsightFace SCRFD as a stand-in for Apple Vision; AuraFace+flip embeddings):
- Open Images "Human face" photos (400): detected 777 full vs 761 at 480 (747 matched); faces >= 40 px (the
  min_face_px of face groups / reference picking, counted in the INDEXED image's pixels) 600 full vs 376 at 480.
  Same-face cosine, full vs 480: mean 0.763, p5 0.327, min 0.105 (n=747); faces >= 40 px at 480: mean 0.916,
  p5 0.777 (n=368). AuraFace thresholds: accept 0.53, group 0.62.
- Pexels frames (400): 118 vs 119 detected (111 matched); >= 40 px 104 vs 37; cosine mean 0.607, p5 0.192 (n=111);
  >= 40 px at 480: mean 0.894, p5 0.734 (n=36).
  Detection survives 480; identity vectors of small faces do not, and the 40 px gate removes ~40-65% of faces.
