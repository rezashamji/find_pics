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
| the planner MODEL | 9B on the server | open: Apple's on-device model vs our 4B (+ distilled reader) — measure on the phone |

## Knowing what is in the picture
| server piece | what it does | phone |
|---|---|---|
| models.ImageTextEncoder (PE-Core-L) | photo/text vectors for "anything" | written: PE-Core-B-16 Core ML (converted; never run on a device). B-16 vs L measured close on the server (RESULTS 20, 22) |
| CLIP tokenizer | words -> ids for the text vectors | verified (1,445 requests) |
| index.py tiles (2x2 crops) | small objects | not used on the server either (decided off, RESULTS 13) |
| media.sample_video_frames / _frames_for | several frames per video, the matched frame | written: every 2 s, max 40, vectors + faces per frame; judge sees the matched frame (eval 10-05: one matched face frame already separates Reza's eras, AUC 0.981; more frames / video input no clear win) |
| models.FaceEncoder (InsightFace buffalo_l) + people.* (face_sims, item_person_scores, expand_refs, other_identities, face_groups) | who is in the photo; "which face is you"; other-person rule | matching logic verified (synthetic faces); alignment verified (200 cases); recognizer converted to Core ML (87 MB, torch=onnx cosine 1.0); detection = Apple Vision (written). License: buffalo_l non-commercial; AuraFace (Apache) measured clearly worse (0.904 vs 0.987 recall at equal wrong-match rate) |
| SubjectRefs / reference_crop / side_by_side | "this specific dog / thing / place" from example photos | missing (vector part is cheap to add; the side-by-side judge step is not) |
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
| filter_to_place / place_or_look | place name -> GPS filter or a look | missing |
| engine pairing (_two_groups / _split_pair) | heavier vs fit split | verified (7 cases) |
| engine.make_exclusive rank-margin fallback + _report | pairing fallback, album report text | missing |
| bursts.burst_ids | "+N similar" stacks | verified; app grid uses it |
| converse.CachedJudge | reuse judge answers on follow-ups | written (in memory) |

## Not needed on the phone
apple_copy.py, cli.py, web.py, report.py, contact.py, albums.write_folder_album (PhotoKit save is written), store.py
(replaced by the app's index), agent.make_plan/execute (older planner), Session (chat state lives in the app).

## Biggest gaps, in order of what a person would notice
1. Faces (any "me" / "Dad" search) + a commercially usable face model.
2. The app has never been compiled or run (needs Reza's Mac on macOS Tahoe 26.6+ and Xcode 27).
3. Which model reads requests and judges photos on the phone (Apple's vs ours): accuracy and speed unmeasured.
4. Anchors / time windows ("the week I went to X"), place names, several frames per video.
5. Completeness bound in fast mode; judge-answer cache for follow-ups.
