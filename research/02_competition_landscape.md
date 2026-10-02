# find_pics: competition landscape (as of 2026-10-02)

Scope: everything that already does some or all of the following:
- natural-language (NL) search over a personal photo and video library;
- combining a specific person, a visual attribute or judgment, a date range and the media type;
- building albums from the results, ideally written back into Apple Photos;
- honestly reporting how many items were scanned plus an estimate of **recall** (the fraction of all true matches that were found), with a confidence interval (CI).

**Evidence tags**
- `[DOC: url]`: I or a research subagent read that page or source file during this session.
- `[INF]`: inference from documented facts.
- `[UNVERIFIED]`: not confirmed from a primary source.

**How the GitHub numbers were collected**
- Stars and last-push dates were pulled from the GitHub REST API on 2026-10-02.
- For small repos the API rate-limited us, so their star counts were scraped from the repo HTML page. These are marked "~".

**Research limits.** The shared web-search budget ran out late in the session. As a result:
- Amazon Photos, Lomorage internals, Mylio internals and a handful of App Store apps are UNVERIFIED.
- The "nobody does recall estimation for photo search" finding comes from a search that is thorough but incomplete. It is not proof of absence.

---

## Bottom line (12 lines)

1. **Person + attribute in one NL query, including inside videos, already ships at scale.**
   - Apple Photos does it on-device on iOS 18.1+ with Apple Intelligence hardware; Apple's own example is "Maya skateboarding in a tie-dye shirt".
   - Google Ask Photos does it in the cloud.
   - Do not treat NL search itself as novel.
2. **Immich v3.2 (released 2026-09-10; 115k stars, AGPL-3.0, actively maintained) is the open-source equivalent.**
   - One query can combine people (ANDed), a CLIP text term, a date range and the media type.
   - It is not NL-parsed: the user sets the filters by hand.
   - For video, CLIP and face search each see **one** ffmpeg-chosen frame.
3. **Apple Photos read and write is solved.**
   - osxphotos (MIT, 3.9k stars) already queries by named person, date, Apple ML label, AI caption and movie type, and does `--add-to-album`.
   - At least six MCP servers (Model Context Protocol: a standard way to expose tools to an LLM agent) wrap it, e.g. sweetrb/apple-photos-mcp with opt-in `create-album`.
4. **"Query, then album" is arriving in the big products.**
   - Siri in iOS 27 (WWDC 2026) searches photos by people and subject and adds them to a shared album.
   - Google Gemini Spark (Sept 2026; paid tiers, US only) creates albums from searches by subject and date.
5. **Identity + date timelapses already exist as open source:** immich-automated-selfie-timelapse (~807 stars) and grow-up. Neither selects by a visual or body judgment.
6. **No product or repo found reports items scanned plus an estimated recall with a CI.** Checked: the 5 big platforms, 12 self-hosted managers, 15+ CLIP/MCP/VLM tools and the research systems.
7. **Recall is exactly where the incumbents are publicly weak.**
   - Google paused Ask Photos in June 2025 to restore "the speed and recall of the original search".
   - Apple users report searches returning "6 of hundreds".
8. **The statistics for certified recall already exist, but not for photos.**
   - Database and e-discovery work already does this: SUPG, LOTUS `sem_filter`, BARGAIN, ABAE, prediction-powered inference, and the QBCB / Callaghan stopping rules from technology-assisted review (TAR).
   - No evidence of it being applied to personal-photo or CLIP/VLM image search.
9. **Big caveat.** Those methods certify recall *relative to an oracle* (the labeller treated as ground truth, here e.g. a VLM judge). An honest "recall vs. what you meant" needs user labels on a random sample.
10. **Subjective body judgments ("where I look overweight") are uncalibrated everywhere.** Photo-to-BMI regressors lose accuracy on unseen data (MAPE 7.9% in-distribution vs 13% on an unseen dataset) and are built for posed photos.
11. **License traps.**
    - InsightFace `buffalo_l` / `antelopev2` face weights (Immich's defaults) are non-commercial research only.
    - Apple MobileCLIP / MobileCLIP2 weights (Queryable, Ente) are research-only.
    - SigLIP / SigLIP2 and Qwen2.5-VL / Qwen3-VL are Apache-2.0.
12. **Verdict.** Retrieval is mostly done. The narrow defensible wedge is **auditable retrieval**: a stated scan count, a certified recall CI, calibrated subjective attributes, multi-frame video coverage, and write-back to Apple Photos, built as a layer on top of osxphotos (and optionally Immich), not as another photo manager.

---

## Comparison table

Legend:
- Y = documented yes; N = documented no, or absent from every doc read; P = partial; ? = UNVERIFIED.
- "Writes Apple albums" means it creates or fills albums *inside Apple Photos*.
- "Recall est." means it reports scanned count plus estimated recall or completeness with uncertainty.

| Product | Person search | NL attribute search | Person+attr+date in one query | Video | Writes Apple albums | Recall est. | Local-only | License (code / key weights) |
|---|---|---|---|---|---|---|---|---|
| **Apple Photos** (iOS 18.1+/26/27) | Y | Y (AI devices only) | P: person+attr Y; relative date ranges ? | Y (moments in clips) | Y (native; Memory movies; Siri iOS 27 adds to shared album) | N | Y, except Enhanced Visual Search | Proprietary |
| **Google Photos Ask Photos / Gemini Spark** | Y (face groups, "me") | Y (incl. subjective) | Y | Y | N (Google albums only; Spark paid, US) | N | N (cloud) | Proprietary |
| **Samsung Gallery** (One UI 8.5 + Gemini) | ? | Y | P (subject + date) | Y (jumps to section of a video) | N | N | ? | Proprietary |
| **Microsoft Photos** (Copilot+ PCs) | N (no biometrics in semantic search) | Y | N | ? | N | N | Y (NPU) | Proprietary |
| **Amazon Photos** | ? | ? | ? | ? | N | ? | N | Proprietary (UNVERIFIED) |
| **Immich** v3.2 | Y (InsightFace) | Y (CLIP/SigLIP2) | Y via filters/API (not NL-parsed) | P (1 frame/video) | N | N (ranked list, no cutoff) | Y (self-host) | AGPL-3.0 / buffalo_l non-commercial |
| **Ente Photos** | Y (YOLO5Face+MobileFaceNet) | Y (MobileCLIP) | P (filters intersect; free text + person ?) | P (thumbnail only) | N | N (fixed cosine ≥ 0.175) | On-device ML; E2EE cloud storage | AGPL-3.0 / MobileCLIP research-only |
| **PhotoPrism** | Y (YuNet+SFace) | P (labels, keywords, optional VLM captions; no embedding search) | Y in query syntax, but attr only via labels/captions | P (`type:video` filter) | N | N | Y | AGPL per project (GitHub: NOASSERTION) |
| **LibrePhotos** | Y (InsightFace + HDBSCAN) | Y (CLIP ViT-B/32 + FAISS) | ? | ? | N | N | Y | MIT |
| **digiKam** 9.x | Y (YuNet+SFace) | P (local LLM translates NL into filters; no visual embeddings) | P (via advanced search) | ? | N | N | Y | GPL-2.0 |
| **Damselfly** | Y (FaceONNX) | N | N | N (planned) | N | N | Y | GPL-3.0 |
| **Nextcloud Memories + Recognize** | Y | N (fixed tags) | N | P (MoViNet tags) | N | N | Y | AGPL-3.0 |
| **Photoview / Photonix / HomeGallery** | Y | N / N / similar-image only | N | ? | N | N | Y | AGPL / AGPL / MIT |
| **Queryable** (iOS) | N | Y (MobileCLIP-S2) | P (attr + date presets, no person) | P (still frame) | N/? | N | Y | MIT / MobileCLIP research-only |
| **PicQuery, SmartScan, Space Gallery** (Android) | N / N / Y (YuNet+MobileFaceNet) | Y | N | N / Y / N | N | N | Y | MIT / GPL-3.0 / GPL-3.0 |
| **rclip** (CLI) | N | Y (OpenCLIP B/32 DataComp) | N | N | N | N | Y | MIT |
| **clip-retrieval** (infra) | N | Y | N | N | N | N | Y | MIT |
| **osxphotos** | Y (Apple's named people) | P (Apple's ~1.5k labels, AI caption, OCR) | Y (CLI flags) | Y (`--movies`) | **Y** (`--add-to-album`) | N | Y | MIT |
| **sweetrb/apple-photos-mcp** | Y | P (labels/keywords/OCR) | Y | Y (media types) | **Y** (opt-in) | N | Y (data local; LLM host may be cloud) | MIT |
| **thenavidm/apple-photos-mcp-cli** | Y (named only) | P (closed vocabulary + host LLM "look_at_photos") | Y | Y | **Y** | P (index coverage counts, `unmatched_terms`; no recall) | P (previews go to host LLM) | MIT |
| **photomind-mcp** | Y | Y (OpenCLIP B/32) | Y | ? | N (read-only) | N | Y | MIT |
| **VLM taggers** (pyimgtag, icloud-image-labeler, Photos-Caption-Assistant) | P (pyimgtag DBSCAN faces) | Y (VLM keywords written to Photos) | via osxphotos afterwards | Y (3–5 frames/video) | keywords (albums via osxphotos) | N | Y | MIT |
| **Peakto** (Cyme) | Y | Y | ? | Y (NLE integration) | ? | N | Y | Commercial |
| **Excire Foto / Search** | Y (age, expression) | Y | ? | Y (clip analysis in Search) | N | N | Y | Commercial |
| **Mylio Photos** | Y (claimed) | Y (claimed) | ? | ? | ? | N | Y | Commercial |
| **Jumper** (video editors) | Y (faces in footage) | Y | Y (person filter + visual search) | Y (video only) | N | N | Y | Commercial |
| **Camera-Roll VQA agent** (research, Jun 2026) | P (profile-photo-conditioned captions, no face ID) | Y | Y (search + time/location `list` tool) | ? | N | N (evidence recall vs gold labels only) | N in main runs (Gemini/GPT-4o-mini); Qwen3-VL ablation | ? |
| **OmniQuery** (CHI 2025) | N (no face recognition) | Y | P | Y (Whisper) | N | N | N (GPT-4o) | ? |
| **immich-automated-selfie-timelapse / grow-up** | Y (via Immich) | N | identity + date only | N | N | N | Y | MIT / MIT |
| **SUPG / LOTUS / BARGAIN / ABAE** | — | — | — | — | — | **Y (recall/precision targets w.p. 1−δ, relative to oracle)** | Y | Apache-2.0 (SUPG, LOTUS) |

Note: "w.p. 1−δ" means "with probability at least 1−δ", where δ (delta) is a small failure probability the user chooses.

---

## Per-product notes

### A. Big platforms

**Apple Photos.**
- NL search: you describe a photo, e.g. "Maya skateboarding in a tie-dye shirt", and it finds photos or "a key moment in a video" [DOC: https://support.apple.com/guide/mac-help/use-apple-intelligence-in-photos-mchl35c53342/15.0/mac/15.0].
- MacRumors examples include "Dad smiling" and "Mum in green hat with wine glass" (person + attribute) and in-video moments [DOC: https://www.macrumors.com/how-to/ios-use-natural-language-search-photos/].
- Requires iPhone 15 Pro / 16 or later, or an M1+ Mac [DOC: same MacRumors page].
- **People recognition:**
  - On-device face and upper-body embeddings (vectors summarising a face or body), grouped by clustering [DOC: https://machinelearning.apple.com/research/recognizing-people-photos].
  - Older search maps queries onto a fixed tag list [DOC: https://machinelearning.apple.com/research/on-device-scene-analysis].
- **Dates:** multiple criteria such as place + month work on Mac [DOC: https://support.apple.com/guide/photos/search-for-photos-and-videos-pht64de33e5a/mac]. Parsing a relative range like "past 6 months" together with a person and an attribute is UNVERIFIED.
- **Albums:**
  - "Create a Memory" turns a text prompt into a *movie*, not an album [DOC: Mac guide above].
  - Smart Albums use condition menus with no semantic text condition [DOC: https://support.apple.com/guide/photos/create-smart-albums-pht6d60ca71/mac].
  - At WWDC 2026, Siri (iOS 27) searched photos "from the previous weekend, with subject matter and specific people called out" [DOC: https://www.apple.com/newsroom/2026/06/apple-unveils-next-generation-of-apple-intelligence-siri-ai-and-more/] and added photos of named people to a shared album [DOC: https://www.idropnews.com/news/wwdc-2026-apple-siri-ai-ios-27/264792/].
- **Developer access:**
  - iOS 26's `IntentValueQuery` / `SemanticContentDescriptor` hands an image to an app so the app can search *its own* content. It does not expose the Photos semantic index [DOC: https://developer.apple.com/videos/play/wwdc2025/275/].
  - The Apple newsroom post mentions no Photos search API [DOC: newsroom above].
  - No public PhotoKit semantic or people search API: strong prior, UNVERIFIED.
  - [INF] Third parties can only read Apple's stored ML outputs (labels, captions, named people) via the SQLite database, which is what osxphotos does. They cannot call Apple's NL search.
- **Recall:**
  - No completeness reporting in any Apple doc read.
  - Complaints:
    - indexing still unfinished three months after iOS 18 [DOC: https://www.bgr.com/tech/3-months-later-ios-18-still-hasnt-finished-indexing-photos-on-some-iphones/];
    - "insect" returned 6 of hundreds of photos [DOC (search snippet): https://forums.macrumors.com/threads/ios-18-master-the-new-search-features-in-the-photos-app.2438664/page-2].
  - iOS 27 "rebuilt" search for stability and efficiency [DOC: newsroom above].
- **Implication** [INF]: on recent Apple hardware, the free built-in tool already does "me + attribute", but it is a black box with no coverage guarantee and no API.

**Google Photos (Ask Photos, Gemini app, Gemini Spark).**
- **Ask Photos:**
  - Pipeline: a Gemini agent picks a tool, vector retrieval (matching on embeddings rather than exact keywords) finds candidates, and an answer model checks them with dates and locations. Covers photos and videos [DOC: https://9to5google.com/2024/05/25/google-photos-ask-photos-works/].
  - Setup requires confirming your "me" face group. It accepts subjective queries such as "Photos that'd make great phone backgrounds" [DOC: https://support.google.com/photos/answer/15318661?hl=en].
- **Status:**
  - Rollout paused June 2025 over "latency, quality and ux", promising to bring back "the speed and recall of the original search" [DOC: https://www.androidauthority.com/google-photos-ask-photos-pause-3563859/].
  - A classic/Ask toggle was added in 2026 [DOC: https://9to5google.com/2026/03/10/google-photos-ask-search-toggle/].
- **Gemini app + Photos:** searches but explicitly cannot create or edit albums [DOC: https://support.google.com/gemini/answer/15734842?hl=en].
- **Gemini Spark** (Sept 2026): "Find specific photos and videos by subject, location, date, or event", creates private and shared albums, and runs recurring jobs. Rolling out to AI Pro/Ultra subscribers in the US [DOC: https://9to5google.com/2026/09/03/gemini-spark-google-photos/; https://techcrunch.com/2026/09/04/googles-gemini-spark-can-now-manage-your-google-photos-library/].
- **Highlight videos:** pick person + date range; "Help me select" auto-chooses clips [DOC: https://support.google.com/photos/answer/6128862?hl=en&co=GENIE.Platform%3DAndroid].
- **Privacy and recall:** cloud processing; queries "may be reviewed by humans" with an opt-out [DOC: support 15318661]. No recall reporting.

**Samsung Gallery.**
- One UI 8.5 (S26) search jumps to "the exact section of the video" [DOC: https://www.sammobile.com/news/one-ui-8-5-gallery-instant-search-results-more/].
- The Gemini–Gallery integration finds items by description, date, location and subject ("my cat from last month"), and cannot move, delete or share items [DOC: https://sammyguru.com/galaxy-s26-gemini-samsung-gallery/].
- On-device vs cloud: UNVERIFIED.

**Microsoft Photos.**
- Semantic search on Copilot+ PCs runs locally on the NPU (neural processing unit, a chip for running ML models).
- "No biometric data is collected, processed, or stored for … semantic search", so there is no person identity in it [DOC: https://support.microsoft.com/en-us/windows/apps/photos/search-photos].

**Amazon Photos.** UNVERIFIED: every Amazon page returned 404 or 503 and the search budget was exhausted. [INF] It is not a plausible threat on the "local, open, auditable" axis either way.

### B. Self-hosted / open-source photo managers

**Immich** (https://github.com/immich-app/immich; AGPL-3.0; 115,441 stars; pushed 2026-10-02; v3.2.4 2026-09-28).
- **Models:**
  - Defaults: CLIP `ViT-B-32__openai`, faces `buffalo_l` (minScore 0.7, maxDistance 0.5, minFaces 3), OCR `PP-OCRv5_mobile` [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/dtos/config.dto.ts].
  - Selectable: OpenCLIP (OpenAI, LAION, DFN), SigLIP, SigLIP2 up to `ViT-gopt-16-SigLIP2-384`, multilingual NLLB / XLM-R / M-CLIP, and InsightFace `antelopev2` / `buffalo_s|m|l` [DOC: https://raw.githubusercontent.com/immich-app/immich/main/machine-learning/immich_ml/models/constants.py].
- **Face clustering:** a modified incremental DBSCAN, a density clustering method that forms a group once "a certain number of similar faces (by default 3)" are close enough [DOC: https://docs.immich.app/features/facial-recognition].
- **Combined query:** `SmartSearchDto` accepts all of these at once [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/dtos/search.dto.ts]:
  - `query` (CLIP text) and `personIds`;
  - `takenAfter` / `takenBefore` and `type` (image/video);
  - location, camera, `ocr`, `rating`, `size` (max 1000);
  - a v3.2 structured `filter` with AND/OR branches.
- **Multiple people are ANDed:** the query requires `count(distinct person) = personIds.length` [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/utils/database.ts].
- **v3.2.0 release (2026-09-10):** "combining multiple search filters with both AND and OR operations". Its screenshot caption reads "images with Jason in it, taken in the last 30 days in BC, Canada … with the context search term 'Swimming'" [DOC: https://github.com/immich-app/immich/releases/tag/v3.2.0, verified by me].
  - Combining was "not yet" possible in Jan 2024 [DOC: https://github.com/immich-app/immich/discussions/6180].
- **Ranking:**
  - Results are ordered by cosine distance with `LIMIT size+1 OFFSET`. There is no similarity threshold, so you get a ranked list, never "the set of matches" [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/repositories/search.repository.ts].
  - No count of matches and no recall estimate.
- **Video:**
  - CLIP and face detection both run on the single `Preview` image [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/services/smart-info.service.ts; .../person.service.ts].
  - For video, that preview is one frame chosen by ffmpeg's `thumbnail` filter [DOC: https://raw.githubusercontent.com/immich-app/immich/main/server/src/utils/media.ts].
  - [INF] So anything visible only mid-clip is invisible to Immich search.
- **Apple Photos:** no native sync; the feature request was closed as a duplicate [DOC: https://github.com/immich-app/immich/discussions/14842]. The usual path is one-way, osxphotos export then immich-go upload [DOC: https://jacobian.org/til/immich-setup/].
- **Weights license:** InsightFace says its "pretrained models … are for non-commercial research only, whether downloaded automatically or manually", and for `buffalo_l` you must contact them for licensing [DOC: https://github.com/deepinsight/insightface (README, read via raw)].
- **Bottom line:** Immich already implements the person ∧ text ∧ date ∧ type *query algebra* (filters combined with AND).
  - What it lacks: NL parsing into those filters, a relevance cutoff, multi-frame video indexing, calibrated judgments, Apple write-back and any completeness statement.

**Ente Photos** (https://github.com/ente-io/ente → ente/ente; AGPL-3.0; 29,188 stars; pushed 2026-10-01).
- **Models:** on-device MobileCLIP (v1) for "magic search"; YOLO5Face + MobileFaceNet for faces; custom "linear incremental clustering"; ONNX Runtime [DOC: https://ente.com/ml/].
- **Threshold:** results kept at cosine ≥ 0.175. The code comment says this "heuristic threshold trades recall against query-dependent false positives" [DOC: https://raw.githubusercontent.com/ente/ente/main/web/packages/new/photos/services/ml/clip.ts]. In other words, the developers know recall is uncontrolled.
- **Video:** indexes only the thumbnail [DOC: https://raw.githubusercontent.com/ente/ente/main/web/packages/new/photos/services/ml/blob.ts].
- **Filter combination:**
  - Mobile "hierarchical search" intersects face (up to 4), file type, magic, location and album filters [DOC: https://raw.githubusercontent.com/ente/ente/main/mobile/apps/photos/lib/utils/hierarchical_search_util.dart].
  - Help docs: "Search for a person's name and then filter by date" [DOC: https://ente.com/help/photos/features/search-and-discovery/].
  - Arbitrary free text ANDed with a person: UNVERIFIED.
- **Weights:** MobileCLIP weights are under Apple's research-only model license [DOC: https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS, read by me].

**PhotoPrism** (https://github.com/photoprism/photoprism; 40,266 stars; pushed 2026-10-01; GitHub license field NOASSERTION, AGPL per project UNVERIFIED).
- **Labels and captions:** offline TensorFlow labels at 224 px, plus optional Ollama or OpenAI-compatible vision models for captions and labels [DOC: https://docs.photoprism.app/user-guide/ai/].
- **Faces:** "YuNet locates them, and SFace turns each one into a vector" [DOC: same].
- **Query syntax** combines `person:"A & B"`, `label:`, `after:`/`before:`, `year:` and `type:video` [DOC: https://docs.photoprism.app/user-guide/search/filters/].
- **No embedding search:** attribute matching only works if a label or caption happens to contain the word [DOC: ai page; absence of CLIP].
- Read-only MCP support added May 2026 [DOC: https://docs.photoprism.app/release-notes/].

**LibrePhotos** (https://github.com/LibrePhotos/librephotos; MIT; 8,089 stars; pushed 2026-09-30).
- Models: CLIP ViT-B/32 + FAISS (an approximate nearest-neighbour search library) for semantic search; InsightFace + HDBSCAN for faces; LFM2.5-VL captions conditioned on people and location; MobileCLIP-S2 / SigLIP2 scene tags [DOC: https://github.com/LibrePhotos/librephotos].
- Combined person + semantic + date queries and video indexing: UNVERIFIED.

**digiKam** (real repo invent.kde.org/graphics/digikam; the GitHub mirror is archived; GPL-2.0).
- Releases: 9.0 (2026-03-08) and 9.1 (2026-06-07) [DOC: https://www.digikam.org/news/].
- 8.6 added YuNet + SFace faces and YOLOv11 auto-tags [DOC: https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/].
- August 2026: a local LLM "translates your natural language query into digiKam's advanced search criteria" [DOC: https://www.digikam.org/news/2026-08-20-advanced_search_improvements_with_llm/].
- [INF] This is the only NL-to-filter parser found in open source, but it has no visual embeddings, so "overweight" only works if such a tag already exists.

**Damselfly** (https://github.com/Webreaper/Damselfly; GPL-3.0; 1,790 stars).
- YOLO objects; FaceONNX faces, which replaced Azure in v4.5.3 [DOC: https://github.com/Webreaper/Damselfly/releases].
- No CLIP search; video "possibly" in future [DOC: https://github.com/Webreaper/Damselfly].

**Photonix** (AGPL-3.0; 1,956 stars). "Not feature complete for a version 1.0" [DOC: https://github.com/photonixapp/photonix]. Effectively stagnant [INF].

**Lomorage.** Homepage claims "Describe a scene with AI search, or find photos by person, place and the words inside them", with on-phone AI [DOC: https://lomorage.com/]. Models, license and repo are UNVERIFIED.

**Nextcloud Memories + Recognize.**
- Memories (AGPL-3.0, 3,848 stars) delegates AI to Recognize [DOC: https://memories.gallery/].
- Recognize (AGPL-3.0, 699 stars) uses EfficientNet tags, face-api.js faces and MoViNet video classification. It produces fixed tags, with no free-text semantic search [DOC: https://github.com/nextcloud/recognize].

**Others.**
- Photoview (AGPL-3.0, 6,538 stars): dlib faces, no semantic search [DOC: https://github.com/photoview/photoview].
- HomeGallery (MIT, 1,186 stars): "similar image search" + faces [DOC: https://home-gallery.org/].
- Pigallery2, Lychee, Piwigo, Spacedrive: not checked (UNVERIFIED).

### C. CLIP search apps and CLIs

CLIP is a model that maps images and text into one vector space, so text-to-image search is a nearest-neighbour lookup.

**Queryable** (https://github.com/mazzzystar/Queryable; **MIT**, the LICENSE file has never changed; 2,989 stars; last push 2026-03-29).
- Model: MobileCLIP-S2 since 2024-09-01, OpenAI ViT-B/32 before that [DOC: README, read by me].
- Video is matched "by its still frame, never by motion or audio". Date presets narrow results, but ranking is similarity only. $4.99 one-time purchase [DOC: https://queryable.app/].
- No person filter or album creation documented.
- **License tension:** the shipped MobileCLIP weights are research-only per Apple's LICENSE_MODELS [DOC, read by me], yet the app is sold on the App Store [INF].

**PicQuery** (https://github.com/greyovo/PicQuery; MIT; 532 stars). Android port of Queryable; V2 rewritten in Flutter with desktop builds. No person, video or date filters [DOC: repo].

**SmartScan** (https://github.com/smartscanapp/smartscan-android; GPL-3.0; ~499 stars). On-device image **and video** search with a quantized CLIP. No faces [DOC: repo; https://github.com/smartscanapp/smartscan-android-lib].

**Space Gallery** (https://github.com/sharap/space-gallery; GPL-3.0; ~0 stars). Android CLIP ViT-B/32 + YuNet/MobileFaceNet faces + OCR via ONNX. No video search or albums [DOC: repo].

**rclip** (https://github.com/yurijmikhalevich/rclip; MIT; 1,012 stars; v4.0.1 2026-09-19).
- Model: OpenCLIP `ViT-B-32-256-datacomp_s34b_b86k` [DOC: https://github.com/yurijmikhalevich/rclip/blob/main/rclip/model_download.py].
- Images only; supports text+image arithmetic queries [DOC: README].

**clip-retrieval** (https://github.com/rom1504/clip-retrieval; MIT; 2,799 stars). Batch embedding, FAISS indices and a web UI; infrastructure, not an app [DOC: repo].

**llm-clip** (https://github.com/simonw/llm-clip; Apache-2.0; ~78 stars). CLIP embeddings via the `llm` CLI [DOC: repo].

**aiPhotos** (https://github.com/giulianoberteo/aiPhotos; MIT; ~0 stars).
- A local VLM fills a JSON schema per photo, then search fuses BM25 (keyword ranking) with embedding similarity using reciprocal-rank fusion.
- Faces are clustered; video is "next up"; no Apple Photos integration [DOC: repo].

**Searchable** (closed-source, Mac/iOS) [DOC: https://www.engineerdraft.com/en/searchable/]. Details UNVERIFIED.

**Not checked:** PicSeek, Hashtag Photos, Lens Go, Pixel Search, PhotoSearch AI (search budget exhausted). UNVERIFIED, not "absent".

### D. Apple Photos integration layer (most directly reusable)

**osxphotos** (https://github.com/RhetTbull/osxphotos; MIT; 3,882 stars; v0.77.2 2026-09-27).
- **Query flags:** `--person`, `--label`, `--keyword`, `--from-date` / `--to-date`, `--movies`, `--album` [DOC: README].
- **Album write-back:** `--add-to-album` creates the album if needed [DOC: https://pypi.org/project/osxphotos/0.67.10].
- **Python API:** exposes `persons`, `face_info`, `labels` (Apple ML labels), `search_info`, `score` (aesthetic), `ai_caption`, `media_analysis` [DOC: https://rhettbull.github.io/osxphotos/API_README.html].
- **Recent changes** [DOC: https://github.com/RhetTbull/osxphotos/blob/main/CHANGELOG.md]:
  - v0.74.1 (2025-11) added `ai_caption`.
  - v0.77 (Sept 2026) reads macOS 27's `leo.sqlite` search info and adds `human_actions`, `pets`, `events`, `age_groups`.
- **Not found:** any access to Apple's semantic embeddings (UNVERIFIED absence).
- **Writes** go through photoscript (AppleScript).

**MCP servers wrapping Photos.**
- **sweetrb/apple-photos-mcp** (MIT, ~26 stars) [DOC: https://github.com/sweetrb/apple-photos-mcp, README read by me]:
  - Query by date range, album, keyword, person, ML label, place, GPS radius, media type, aesthetic score and OCR text.
  - Opt-in `create-album` / `add-to-album` / `set-keywords` (enabled by `APPLE_PHOTOS_MCP_ENABLE_WRITES=1`); never deletes.
  - No CLIP or VLM search.
- **thenavidm/apple-photos-mcp-cli** (MIT, ~0 stars) [DOC: https://github.com/thenavidm/apple-photos-mcp-cli, README read by me]:
  - Explicitly describes Apple's labels as "a closed vocabulary of roughly 1,500 words", and "Faces only work if you named them".
  - `look_at_photos` sends previews to the host LLM.
  - `doctor` reports e.g. "37129 assets indexed (35983 with ML labels…)", and `unmatched_terms` flags query words with no Apple label.
  - **Closest thing to "honest coverage" found in the wild:** it reports index coverage, but not recall.
- **marcomc/mcp-osxphotos** (MIT, ~2 stars): wraps the osxphotos CLI (person, label, date, video, album creation) [DOC: repo].
- **maximbilan/photos-macos-mcp** (MIT, ~3 stars): PhotoKit, read-only, Vision keyword matching that "analyzes up to 1000 photos" [DOC: repo].
- **davidcjw/photomind-mcp** (MIT, ~0 stars): OpenCLIP B/32 on top of osxphotos, with person and date filters; read-only [DOC: repo].
- **jamjamCH/iCloud-Photo-Curator** (MIT, ~5 stars): host-LLM vision; album creation is experimental, dry-run by default [DOC: repo].
- **baney75/stillport** (MIT): calls Photos' AppleScript `search` [DOC: repo]. Whether that reaches Apple's NL search is UNVERIFIED.

**Local VLM taggers that write back into Photos.**
- **pyimgtag** (https://github.com/kurok/pyimgtag; MIT; ~5 stars): Gemma 4 via Ollama; video frames at 10/50/90%; DBSCAN faces; `--write-back` keywords [DOC: repo].
- **icloud-image-labeler** (https://github.com/ziadalzarka/icloud-image-labeler; MIT): any OpenAI-compatible endpoint, default `qwen/qwen3.5-9b`; 5 frames per video; writes keywords, title and description via osxphotos + PhotoScript [DOC: repo].
- **Photos-Caption-Assistant** (https://github.com/JohnKFisher/Photos-Caption-Assistant; MIT): Swift, `qwen2.5vl:7b` via Ollama, photos + videos, writes captions [DOC: repo].
- [INF] These already give "VLM judgment → keywords inside Apple Photos → query and album via osxphotos". None estimates how many matches it missed.

### E. Commercial local-first managers

- **Peakto** (Cyme): NL AI search ("Ferris wheel at sunset"), face tagging, local AI, Apple Photos listed as compatible, Premiere/FCP integration [DOC: https://cyme.io/en/peakto/]. Write-back of albums into Photos: UNVERIFIED.
- **Excire Foto 2027:** local face recognition including age groups and expressions, AI keywording, free-text search, photo + video [DOC: https://excire.com/en/excire-foto/]. **Excire Search 2026** (Lightroom plugin) adds prompt search and video clip analysis [DOC: https://excire.com/en/excire-search/].
- **Mylio Photos:** claims offline faces, SmartTags and NL search [DOC: https://mylio.com/]. Details UNVERIFIED.
- **Jumper** (video editors; standalone app + Premiere/Resolve/FCP/Avid plugins):
  - Offline visual, speech and face search.
  - "combine the person filter with a visual search" [DOC: https://getjumper.io/features/face-recognition; https://getjumper.io/].
  - Exposes agent/MCP tools [DOC: https://getjumper.io/ai-agents].
  - Footage only, Pro tier; models undisclosed.
  - [INF] The best existing local person ∧ visual search *for video*, but aimed at editors' footage bins, not phone libraries.
- **Memories.ai:** "Large Visual Memory Model" for video; cloud platform; on-device via Qualcomm "starting in 2026" [DOC: https://memories.ai/; https://techcrunch.com/2026/03/16/memories-ai-is-building-the-visual-memory-layer-for-wearables-and-robotics]. Not a personal-library tool today [INF].

### F. Research prototypes

**Lifelog Search Challenge (LSC) 2022–24.**
- The dataset is one lifelogger's wearable camera, with faces blurred or omitted, so identity retrieval is out of scope.
- Systems combine CLIP / OpenCLIP / BLIP-2 similarity with time, place and event filters (MyScéal, LifeSeeker, Memento, MyEachtra, MemoriEase, LifeInsight, Voxento) [DOC: https://arxiv.org/abs/2506.06743].
- Recall is measured by organizers against ground truth. No system estimates its own recall [DOC: same; INF].

**Personalized VLMs and retrievers.**
- **PerVL / PALAVRA** ("This is my unicorn, Fluffy", ECCV 2022):
  - Learns a word embedding for a personal concept inside frozen CLIP. Compositional queries "[CONCEPT] sitting at the back end of the sailboat"; Recall@K / MRR.
  - Concepts are objects (DeepFashion2, YouTube-VOS) [DOC: https://arxiv.org/abs/2204.01694; https://github.com/NVlabs/PALAVRA].
- **This-Is-My** (CVPR 2023) [DOC: https://arxiv.org/abs/2306.10169] and **Return of Fluffy** (CBMI 2025) [DOC: https://arxiv.org/abs/2510.05411]: instance + text compositional retrieval.
- **ConCon-Chi** (CVPR 2024): concept × context compositional retrieval with chimeric objects [DOC: https://openaccess.thecvf.com/content/CVPR2024/html/Rosasco_ConCon-Chi_Concept-Context_Chimera_Benchmark_for_Personalized_Vision-Language_Tasks_CVPR_2024_paper.html].
- **MyVLM** (ECCV 2024; uses a face-recognition concept head for people) [DOC: https://arxiv.org/abs/2403.14599], **Yo'LLaVA** (NeurIPS 2024) [DOC: https://arxiv.org/abs/2406.09400], **RAP** (CVPR 2025) [DOC: https://arxiv.org/abs/2410.13360] and **PeKit** [DOC: https://arxiv.org/abs/2502.02452]:
  - These handle *people* as concepts, but are evaluated on captioning, VQA and recognition, not gallery recall.
- [INF] Nobody benchmarks "specific person × subjective attribute" retrieval over a personal gallery.

**Composed image retrieval (CIR)** (query = reference image + modifying text).
- Pic2Word (CVPR 2023) [DOC: https://openaccess.thecvf.com/content/CVPR2023/html/Saito_Pic2Word_Mapping_Pictures_to_Words_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2023_paper.html] and SEARLE/CIRCO (ICCV 2023; CIRCO has multiple ground truths per query) [DOC: https://github.com/miccunifi/SEARLE] preserve *semantics*, not identity.
- Instance-level CIR is the closest formal analogue to "photo of me + 'looking heavier'":
  - i-CIR / BASIC (NeurIPS 2025) [DOC: https://arxiv.org/abs/2510.25387]
  - OACIR (CVPR 2026) [DOC: https://arxiv.org/abs/2604.05393]
  - Whether either covers people is UNVERIFIED.

**Personal photo QA agents.**
- **OmniQuery** (CHI 2025) [DOC: https://arxiv.org/abs/2409.08250]:
  - Uses GPT-4o; temporal constraints act as strict filters.
  - The authors say social queries are hard because there is no facial recognition.
  - Accuracy measured by user ratings; no recall.
- **Personal AI Agent for Camera Roll VQA** (Nguyen, Singh, Kim, Lee, Li; arXiv 2606.05275, June 2026) [DOC: https://arxiv.org/abs/2606.05275, verified by me]. **Closest research system.**
  - Data: 50 users, 31,476 images, 2,500 QA pairs.
  - Agent tools: `search` (embeddings), `grep` (BM25), `list` (time/location filter), `get`, `view` (a VLM looks at the pixels).
  - Identity comes from conditioning the captioner on the user's profile photo; there is no face-ID model.
  - Main runs use Gemini-2.5-Flash / GPT-4o-mini, with a Qwen3-VL-8B ablation.
  - Reports *evidence recall* measured against gold labels, not an estimated CI.
  - Code: https://github.com/thaoshibe/camroll (license UNVERIFIED).
- **ReaLMem** (Sept 2026): a benchmark of real multi-year personal visual archives [DOC: https://arxiv.org/abs/2609.19167].

**BMI and body judgments from images.**
- **Face-to-BMI** (2017): r = 0.65 [DOC (search snippet): https://arxiv.org/abs/1703.03156]. Here r is the correlation with true BMI.
- **Digital Scale** (2025): full-body images, MAPE (mean absolute percentage error) 7.9% in-distribution vs 13% on an unseen dataset [DOC: https://arxiv.org/abs/2508.20534].
- **Consumer body-fat apps** (MeThreeSixty, Prism, bodyfatestimator.ai) all require posed photos [DOC: https://apps.apple.com/us/app/methreesixty-3d-body-scanner/id1472541261; https://www.prismlabs.tech/white-papers/body-composition-dxa-alternative-2026; https://www.bodyfatestimator.ai/].
- [INF] "Where I *look* overweight" is a subjective, pose- and outfit-dependent judgment, not BMI. Running it on third parties ("my dad") carries body-image and ethics risk that no source addresses.

### G. Recall-estimation prior art (the claimed differentiator)

**SUPG** (Kang et al., PVLDB 2020) [DOC: https://ddkang.github.io/papers/2020/supg-paper.pdf; code https://github.com/stanford-futuredata/supg, Apache-2.0, 7 stars, last push 2021].
- Setup: a cheap proxy score plus an expensive oracle.
- It picks a threshold so that a recall target holds with probability ≥ 1−δ.
- Uses importance sampling (sampling items in proportion to roughly √(proxy score), then reweighting to stay unbiased).
- Guarantees rest on the central limit theorem, i.e. they hold only asymptotically [DOC: critique in https://arxiv.org/abs/2509.02896].

**LOTUS `sem_filter` cascades** (PVLDB 2025) [DOC: https://www.vldb.org/pvldb/vol18/p4171-patel.pdf; https://lotus-ai.readthedocs.io/en/latest/approximation_cascades.html; https://github.com/lotus-data/lotus, Apache-2.0, 1,708 stars].
- SUPG-style thresholds; parameters `recall_target`, `precision_target`, `failure_probability`.
- Guarantees are relative to the oracle model.
- Whether image columns are supported is UNVERIFIED.

**BARGAIN** (Zeighami, Shankar, Parameswaran; SIGMOD '26) [DOC: https://arxiv.org/abs/2509.02896].
- Gives guarantees that hold at any sample size (finite-sample), using betting-based confidence sequences.
- Beats SUPG on oracle cost.
- Code: UNVERIFIED.

**ABAE** (PVLDB 2021) [DOC: https://www.vldb.org/pvldb/vol14/p2341-kang.pdf].
- Estimates COUNT/SUM with stratified sampling (proxy-score strata) and bootstrap CIs.
- [INF] A COUNT of true matches with a CI is exactly the denominator of recall.

**Prediction-powered inference** (Angelopoulos et al., Science 2023) [DOC (search snippet): https://ui.adsabs.harvard.edu/abs/arXiv:2301.09633; code https://github.com/aangelopoulos/ppi_py].
- Combines model predictions with a small gold-labelled set to produce valid CIs.

**TAR / e-discovery and systematic-review screening.**
- **QBCB** (Lewis, Yang, Frieder, CIKM 2021) gives a recall CI from a random sample of positives [DOC: https://arxiv.org/abs/2108.12746].
- **Callaghan & Müller-Hansen 2020** run a hypergeometric test on random samples of the *unscreened remainder*: stop when "recall < target" is rejected [DOC: https://doi.org/10.1186/s13643-020-01521-4].
- **Quant / QuantCI** heuristics [DOC: https://dl.acm.org/doi/abs/10.1145/3469096.3469873].
- **Cormack & Grossman SIGIR 2016** [DOC (citation only): https://dblp.org/db/conf/sigir/sigir2016.html].

**Applied to personal-photo or CLIP/VLM image search: no evidence found.**
- Searched: arXiv full-text queries, the LSC review, OmniQuery and Camera-Roll VQA.
- Semantic Scholar and arXiv APIs then rate-limited, so the search is incomplete [DOC: subagent search log; INF].

### H. Transformation / progress-photo tools

**Identity + date → aligned timelapse (open source).**
- **immich-automated-selfie-timelapse** (https://github.com/ArnaudCrl/immich-automated-selfie-timelapse; MIT; ~807 stars):
  - Pulls every photo of an Immich person.
  - Filters on head pose, blink, blur, brightness and face size; caps per day, week or month; eye-aligns; renders with ffmpeg.
  - The author says filtering "is not 100% accurate" [DOC: repo].
- **grow-up** (https://github.com/chr1shaefn3r/grow-up; MIT; ~23 stars): Immich person + MediaPipe (Google's open-source face/pose landmark library) quality filters + best frame per period [DOC: repo].
- **Folder-input aligners with no identity step:**
  - facemation: https://github.com/FWDekker/facemation
  - face-movie: https://github.com/leachiM2k/face-movie
  - Ivolution: https://github.com/jlengrand/Ivolution (~10 stars)
  - [DOC: repos]

**Consumer apps (user imports or captures the photos).**
- **AgeLapse** (iOS, free, on-device): "finds your face in every photo and aligns them" in a "fitness transformation" mode, from photos you import [DOC: https://apps.apple.com/us/app/agelapse/id6503668205].
- **Dayloop** (iOS): face or full-body alignment of photos the user picks [DOC: https://apps.apple.com/app/id6740197860].
- **Facelapse** (Android) [DOC: https://www.facelapse.io/].
- **Body Tracker / PhotoJourney** (iOS) [DOC: https://apps.apple.com/us/app/body-tracker-cam-photojourney/id6499454966].
- **Fitness Camera** (Android; fires the shutter when your pose matches) [DOC: https://fitnesscamera.app/].
- **Progress – AI Timelapse** (iOS) [DOC: https://apps.apple.com/ms/app/progress-ai-timelapse/id6477857716].
- **MacroFactor** before/after [DOC: https://help.macrofactorapp.com/en/articles/351-how-to-create-and-share-before-and-after-photos].
- **1 Second Everyday** (date-ordered clips, no person selection) [DOC: https://help.1se.co/en/articles/10290563-quick-start-guide-for-1-second-everyday-on-ios].

**Video editors.** CapCut glow-up templates [DOC: https://www.capcut.com/explore/glow-up-transformation] and Instagram Edits [DOC: https://www.engadget.com/2244219/instagram-adds-feature-that-automatically-trims-clips-for-reels/] use photos you select by hand.

**Platform auto-curation.**
- Google Highlight videos select by person + date [DOC: support 6128862].
- Apple "Create a Memory" works from a free-text prompt [DOC: Apple Mac guide]. No documented support for physique judgments.

[INF] **Nothing found auto-splits a library into a "heavier era" vs a "fit now" era by a visual body judgment, and nothing reports what it missed.**

---

## Verdict

### What genuinely does not exist yet (as far as this search found)

1. **A coverage / recall certificate on personal photo search.**
   - A statement like "scanned 48,211 items (incl. 3,104 videos at k frames each); found 212; estimated recall 0.91 [0.84, 0.96] at 95%, relative to the judge; user-validated on n=60 samples".
   - Zero products, zero repos and zero papers found. Incumbents publicly struggle with recall (Ask Photos pause; Apple "6 of hundreds"). Ente's own code comment admits its threshold "trades recall against … false positives".
2. **Calibrated subjective attribute judgments for a specific person over candid library photos.**
   - Example: "heavier vs. leaner", with an explicit per-item confidence and a user-correctable decision boundary.
   - Products use raw CLIP rank (Immich, Ente, Queryable), closed tag vocabularies (Apple labels via osxphotos, PhotoPrism, digiKam) or opaque cloud LLMs (Google).
   - BMI regressors exist but are trained and validated on posed images.
3. **An open, local system that does all of the following end-to-end:**
   - parses NL into person ∧ attribute ∧ date ∧ media-type;
   - indexes *multiple frames per video* (Immich and Ente use one frame);
   - writes albums back into Apple Photos.
   - Each piece exists separately: Immich has the query algebra, digiKam the NL-to-filter step, osxphotos the write-back, pyimgtag the multi-frame VLM tagging. Nobody has joined them.
4. **A benchmark for "specific person × subjective attribute × time" retrieval with recall ground truth** over real personal libraries.
   - Closest: Camera-Roll VQA (2026; QA, not exhaustive retrieval) and PerVL / ConCon-Chi (objects, not people).

### What exists and should be reused, not rebuilt

| Need | Reuse | Link | License caveat |
|---|---|---|---|
| Read Apple Photos (named people, dates, labels, AI captions, movies) and write albums/keywords | osxphotos (+ photoscript) | https://github.com/RhetTbull/osxphotos | MIT |
| Agent / LLM tool surface over Photos with opt-in writes | sweetrb/apple-photos-mcp (reference design) | https://github.com/sweetrb/apple-photos-mcp | MIT |
| Person ∧ CLIP-text ∧ date ∧ type query engine (if self-hosting) | Immich v3.2 Search API v2 (`/search/smart`) | https://github.com/immich-app/immich | AGPL-3.0; buffalo_l non-commercial |
| CLIP/SigLIP embeddings | OpenCLIP; SigLIP2 | https://github.com/mlfoundations/open_clip ; https://huggingface.co/google/siglip2-so400m-patch14-384 | open_clip code permissive; SigLIP2 Apache-2.0 [DOC: HF license tag] |
| Face detection/recognition | InsightFace (research only) or YuNet + SFace (OpenCV; as used by PhotoPrism/digiKam) | https://github.com/deepinsight/insightface | InsightFace weights non-commercial [DOC]; YuNet/SFace license UNVERIFIED |
| VLM judge over frames, written back as keywords | pyimgtag / icloud-image-labeler patterns; Qwen2.5-VL / Qwen3-VL | https://github.com/kurok/pyimgtag ; https://github.com/ziadalzarka/icloud-image-labeler | Qwen VL Apache-2.0 [DOC: HF license tag] |
| Recall/precision-target selection with proxy + oracle | SUPG; LOTUS `sem_filter` cascades; BARGAIN (paper) | https://github.com/stanford-futuredata/supg ; https://github.com/lotus-data/lotus ; https://arxiv.org/abs/2509.02896 | Apache-2.0 |
| CI on total match count (recall denominator) | ABAE; ppi_py | https://github.com/stanford-futuredata/abae ; https://github.com/aangelopoulos/ppi_py | UNVERIFIED |
| Stopping rule / recall CI from random sample of unretrieved items | Callaghan & Müller-Hansen; QBCB | https://doi.org/10.1186/s13643-020-01521-4 ; https://arxiv.org/abs/2108.12746 | papers |
| Identity → aligned timelapse rendering | immich-automated-selfie-timelapse; grow-up | https://github.com/ArnaudCrl/immich-automated-selfie-timelapse ; https://github.com/chr1shaefn3r/grow-up | MIT |

### Blunt assessment

- **For one person on a recent iPhone or Mac, most of the retrieval is already free** [INF].
  - Apple Photos NL search for "Reza" plus an attribute, combined with osxphotos for date ranges and album creation, gets most of the "heavier era vs fit now" result today.
  - Immich v3.2 gives the same query algebra in open source.
- **Rebuilding "NL photo search with faces" is not a contribution.**
- **The defensible wedge is narrow: honest, auditable retrieval.**
  - (a) A certified recall CI.
  - (b) An explicit statement of what "match" means: oracle-relative vs user-validated recall. This is the key subtlety, and every SUPG/LOTUS/BARGAIN guarantee is oracle-relative.
  - (c) Multi-frame video coverage with a declared frame budget.
  - (d) Calibrated subjective attributes.
  - Delivered as a layer over osxphotos (and optionally Immich), outputting Apple Photos albums.
- **Weak moat if productized** [INF]. Apple or Google could add a count/coverage line, and Immich could add thresholds. The stronger framing may be a research artifact: a "certified-recall personal retrieval" method plus a benchmark. That is where the gap is cleanest.
- **Licensing** [INF]. If anything is ever distributed commercially, the default Immich face weights (InsightFace) and the MobileCLIP weights are off the table. SigLIP2 and Qwen VL are Apache-2.0.

---

## Sources

**Big platforms**
- https://support.apple.com/guide/mac-help/use-apple-intelligence-in-photos-mchl35c53342/15.0/mac/15.0
- https://www.macrumors.com/how-to/ios-use-natural-language-search-photos/
- https://machinelearning.apple.com/research/recognizing-people-photos
- https://machinelearning.apple.com/research/on-device-scene-analysis
- https://support.apple.com/guide/photos/search-for-photos-and-videos-pht64de33e5a/mac
- https://support.apple.com/guide/photos/create-smart-albums-pht6d60ca71/mac
- https://support.apple.com/en-us/122033
- https://www.apple.com/newsroom/2026/06/apple-unveils-next-generation-of-apple-intelligence-siri-ai-and-more/
- https://www.idropnews.com/news/wwdc-2026-apple-siri-ai-ios-27/264792/
- https://developer.apple.com/videos/play/wwdc2025/275/
- https://developer.apple.com/videos/play/wwdc2026/240/
- https://www.macrumors.com/guide/ios-26-photos-app/
- https://5minutephotos.substack.com/p/5-minute-photos-os-27-in-action
- https://www.bgr.com/tech/3-months-later-ios-18-still-hasnt-finished-indexing-photos-on-some-iphones/
- https://forums.macrumors.com/threads/ios-18-master-the-new-search-features-in-the-photos-app.2438664/page-2
- https://9to5google.com/2024/05/25/google-photos-ask-photos-works/
- https://support.google.com/photos/answer/15318661?hl=en
- https://blog.google/products-and-platforms/products/photos/ask-photos-google-io-2024/
- https://www.androidauthority.com/google-photos-ask-photos-pause-3563859/
- https://9to5google.com/2026/03/10/google-photos-ask-search-toggle/
- https://support.google.com/gemini/answer/15734842?hl=en
- https://9to5google.com/2026/09/03/gemini-spark-google-photos/
- https://techcrunch.com/2026/09/04/googles-gemini-spark-can-now-manage-your-google-photos-library/
- https://support.google.com/photos/answer/6128862?hl=en&co=GENIE.Platform%3DAndroid
- https://www.sammobile.com/news/one-ui-8-5-gallery-instant-search-results-more/
- https://sammyguru.com/galaxy-s26-gemini-samsung-gallery/
- https://support.microsoft.com/en-us/windows/apps/photos/search-photos

**Self-hosted managers**
- Immich:
  - https://github.com/immich-app/immich
  - https://github.com/immich-app/immich/releases/tag/v3.2.0
  - https://docs.immich.app/features/searching/
  - https://docs.immich.app/features/facial-recognition
  - https://github.com/immich-app/immich/discussions/6180
  - https://github.com/immich-app/immich/discussions/14842
  - https://jacobian.org/til/immich-setup/
- Immich source files (raw.githubusercontent.com/immich-app/immich/main/…):
  - server/src/dtos/config.dto.ts
  - server/src/dtos/search.dto.ts
  - server/src/utils/database.ts
  - server/src/repositories/search.repository.ts
  - server/src/services/smart-info.service.ts
  - server/src/services/person.service.ts
  - server/src/utils/media.ts
  - machine-learning/immich_ml/models/constants.py
- Ente:
  - https://ente.com/ml/
  - https://ente.com/help/photos/features/search-and-discovery/
  - raw.githubusercontent.com/ente/ente/main/web/packages/new/photos/services/ml/clip.ts
  - raw.githubusercontent.com/ente/ente/main/web/packages/new/photos/services/ml/blob.ts
  - raw.githubusercontent.com/ente/ente/main/mobile/apps/photos/lib/utils/hierarchical_search_util.dart
- PhotoPrism:
  - https://docs.photoprism.app/user-guide/ai/
  - https://docs.photoprism.app/user-guide/search/filters/
  - https://docs.photoprism.app/release-notes/
- LibrePhotos:
  - https://github.com/LibrePhotos/librephotos
  - https://docs.librephotos.com/docs/user-guide/features
- digiKam:
  - https://www.digikam.org/news/
  - https://www.digikam.org/news/2025-03-15-8.6.0_release_announcement/
  - https://www.digikam.org/news/2026-08-20-advanced_search_improvements_with_llm/
- Damselfly:
  - https://github.com/Webreaper/Damselfly
  - https://github.com/Webreaper/Damselfly/releases
- Others:
  - https://github.com/photonixapp/photonix
  - https://lomorage.com/
  - https://memories.gallery/
  - https://github.com/nextcloud/recognize
  - https://github.com/photoview/photoview
  - https://home-gallery.org/

**CLIP tools, Apple Photos integration, commercial**
- CLIP apps and CLIs:
  - https://github.com/mazzzystar/Queryable
  - https://queryable.app/
  - https://mazzzystar.com/2024/07/21/Two-Years-of-an-AI-Photo-Album-Search-App/
  - https://github.com/greyovo/PicQuery
  - https://github.com/smartscanapp/smartscan-android
  - https://github.com/sharap/space-gallery
  - https://github.com/yurijmikhalevich/rclip
  - https://github.com/rom1504/clip-retrieval
  - https://github.com/simonw/llm-clip
  - https://github.com/giulianoberteo/aiPhotos
  - https://www.engineerdraft.com/en/searchable/
- osxphotos:
  - https://github.com/RhetTbull/osxphotos
  - https://rhettbull.github.io/osxphotos/API_README.html
  - https://github.com/RhetTbull/osxphotos/blob/main/CHANGELOG.md
  - https://pypi.org/project/osxphotos/0.67.10
- MCP servers:
  - https://github.com/sweetrb/apple-photos-mcp
  - https://github.com/thenavidm/apple-photos-mcp-cli
  - https://github.com/marcomc/mcp-osxphotos
  - https://github.com/maximbilan/photos-macos-mcp
  - https://github.com/davidcjw/photomind-mcp
  - https://github.com/jamjamCH/iCloud-Photo-Curator
  - https://github.com/baney75/stillport
- VLM taggers:
  - https://github.com/kurok/pyimgtag
  - https://github.com/ziadalzarka/icloud-image-labeler
  - https://github.com/JohnKFisher/Photos-Caption-Assistant
- Commercial:
  - https://cyme.io/en/peakto/
  - https://excire.com/en/excire-foto/
  - https://excire.com/en/excire-search/
  - https://mylio.com/
  - https://getjumper.io/
  - https://getjumper.io/features/face-recognition
  - https://getjumper.io/ai-agents
  - https://memories.ai/
  - https://techcrunch.com/2026/03/16/memories-ai-is-building-the-visual-memory-layer-for-wearables-and-robotics

**Model and weight licenses**
- https://github.com/deepinsight/insightface (README "License" section; python-package README)
- https://github.com/apple-aiml-research/ml-mobileclip/blob/main/LICENSE_MODELS
- https://huggingface.co/google/siglip2-so400m-patch14-384
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

**Research**
- Lifelog and personalized retrieval:
  - https://arxiv.org/abs/2506.06743
  - https://arxiv.org/abs/2204.01694
  - https://github.com/NVlabs/PALAVRA
  - https://arxiv.org/abs/2306.10169
  - https://arxiv.org/abs/2510.05411
  - https://openaccess.thecvf.com/content/CVPR2024/html/Rosasco_ConCon-Chi_Concept-Context_Chimera_Benchmark_for_Personalized_Vision-Language_Tasks_CVPR_2024_paper.html
  - https://arxiv.org/abs/2403.14599
  - https://arxiv.org/abs/2406.09400
  - https://arxiv.org/abs/2410.13360
  - https://arxiv.org/abs/2502.02452
- Composed image retrieval:
  - https://openaccess.thecvf.com/content/CVPR2023/html/Saito_Pic2Word_Mapping_Pictures_to_Words_for_Zero-Shot_Composed_Image_Retrieval_CVPR_2023_paper.html
  - https://github.com/miccunifi/SEARLE
  - https://arxiv.org/abs/2510.25387
  - https://arxiv.org/abs/2604.05393
- Personal photo QA agents:
  - https://arxiv.org/abs/2409.08250
  - https://arxiv.org/abs/2606.05275
  - https://arxiv.org/abs/2609.19167
- BMI from images:
  - https://arxiv.org/abs/1703.03156
  - https://arxiv.org/abs/2508.20534

**Recall estimation**
- Database / ML systems:
  - https://ddkang.github.io/papers/2020/supg-paper.pdf
  - https://github.com/stanford-futuredata/supg
  - https://www.vldb.org/pvldb/vol18/p4171-patel.pdf
  - https://lotus-ai.readthedocs.io/en/latest/approximation_cascades.html
  - https://github.com/lotus-data/lotus
  - https://arxiv.org/abs/2509.02896
  - https://www.vldb.org/pvldb/vol14/p2341-kang.pdf
  - https://ui.adsabs.harvard.edu/abs/arXiv:2301.09633
  - https://github.com/aangelopoulos/ppi_py
- TAR / systematic-review screening:
  - https://arxiv.org/abs/2108.12746
  - https://doi.org/10.1186/s13643-020-01521-4
  - https://dl.acm.org/doi/abs/10.1145/3469096.3469873
  - https://dblp.org/db/conf/sigir/sigir2016.html

**Transformation tools**
- Open-source timelapse pipelines:
  - https://github.com/ArnaudCrl/immich-automated-selfie-timelapse
  - https://github.com/chr1shaefn3r/grow-up
  - https://github.com/FWDekker/facemation
  - https://github.com/leachiM2k/face-movie
  - https://github.com/jlengrand/Ivolution
- Consumer apps:
  - https://apps.apple.com/us/app/agelapse/id6503668205
  - https://apps.apple.com/app/id6740197860
  - https://www.facelapse.io/
  - https://apps.apple.com/us/app/body-tracker-cam-photojourney/id6499454966
  - https://fitnesscamera.app/
  - https://apps.apple.com/ms/app/progress-ai-timelapse/id6477857716
  - https://help.macrofactorapp.com/en/articles/351-how-to-create-and-share-before-and-after-photos
  - https://help.1se.co/en/articles/10290563-quick-start-guide-for-1-second-everyday-on-ios
- Video editors:
  - https://www.capcut.com/explore/glow-up-transformation
  - https://www.engadget.com/2244219/instagram-adds-feature-that-automatically-trims-clips-for-reels/
- Body-fat estimation:
  - https://apps.apple.com/us/app/methreesixty-3d-body-scanner/id1472541261
  - https://www.prismlabs.tech/white-papers/body-composition-dxa-alternative-2026
  - https://www.bodyfatestimator.ai/
