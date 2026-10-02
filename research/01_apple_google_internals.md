# Apple Photos, Google Photos, Samsung Gallery: how they find people and things, and why they miss

Research date: 2026-10-02. Tags: **[DOCUMENTED: src]** = stated in the linked source (vendor docs, papers, or a named report). **[INFERENCE]** = my reasoning from documented facts; not stated by the vendor. Community/press reports are tagged DOCUMENTED but labeled "report" because they are anecdotes, not vendor statements.

Jargon used throughout (defined once):
- **Embedding**: a fixed-length vector of numbers that a neural net outputs for an input (a face crop, an image, a text query). Inputs that are "the same thing" are trained to land close together, measured by cosine distance.
- **Precision**: of the photos returned, the fraction that are correct. **Recall**: of all the correct photos in the library, the fraction that were returned.
- **Operating point / threshold**: the score cutoff above which a label is said to be present. A higher cutoff means higher precision and lower recall.
- **Agglomerative clustering**: start with every item as its own group, then repeatedly merge the closest pair of groups. **HAC** (hierarchical agglomerative clustering) is the same idea with an explicit rule (**linkage**) for measuring the distance between two groups.
- **ANE**: Apple Neural Engine, the on-chip accelerator for neural networks.

---

## Bottom line

1. Apple People pipeline: one detector finds faces and upper bodies, two networks embed them, a greedy conservative pass clusters them (body cues only within the same "moment"), a face-only HAC pass merges across moments, and new faces are assigned by sparse coding against stored example faces. [DOCUMENTED: [A-PPL21]]
2. Clustering runs "periodically, typically overnight during device charging." Per-photo assignment runs near capture time. Everything runs on-device. [DOCUMENTED: [A-PPL21], [S-PETS]]
3. Missed photos are partly by design: the first pass is tuned for "high precision but many, smaller clusters," unclear faces are filtered out, and only "frequently occurring" people are promoted into the gallery. [DOCUMENTED: [A-PPL21]]
4. Profile views, age gaps (especially children), small faces, and occlusion all measurably reduce face-matching accuracy, including in mobile-class networks like the one Apple's is based on. [DOCUMENTED: [CFP16], [AIRF], [DEB17], [WIDER]]
5. Classic Photos search is a fixed-taxonomy multi-label tagger (>1,000 classes, with a per-class threshold), plus synonyms and a word-embedding fallback for queries that match no tag. [DOCUMENTED: [WWDC19], [A-ANSA22]]
6. "bread" is a class in the Vision revision-1 taxonomy. So a miss on "bread" is more likely a score below that class's threshold, an unfinished index, or a hierarchy/routing effect than a missing label. [DOCUMENTED: [GIST]; the rest is INFERENCE]
7. Apple Intelligence natural-language search (iOS 18.1 and later; iPhone 15 Pro, iPhone 16 and later) has **no published model description**. I checked the 2024 and 2025 AFM reports and the June 2026 AFM 3 article. [DOCUMENTED: [G-IPH-SRCH]; absence verified in [A-AFMTR24], [A-AFMTR25], [A-AFM3]]
8. Apple keeps the search index on the device. Even search on iCloud.com queries the device's local index. Only landmark matching goes to a server, under homomorphic encryption. [DOCUMENTED: [APS26], [A-HE24]]
9. Google: face groups and search run on Google's servers. Ask Photos returns fast classic results first, then Gemini works through a set of candidate photos (with AI captions) to narrow them down. Samsung S25 runs a compressed vision-language search model on-device. [DOCUMENTED: [GB-JUN25], [GH-PRIV], [SAM25]; "server" is INFERENCE]
10. None of the products I reviewed tells the user how complete a person album or a search result is. They show indexing-progress messages and disclaimers, never a recall estimate. [DOCUMENTED absence across the docs below; this is not proof that no such feature exists anywhere]

---

## 1. How Apple Photos recognizes people

**Pipeline (Apple's 2021 article, which describes iOS 15):**
- **Detection.** "A deep neural network that takes a full image as input, and outputs bounding boxes for the detected faces and upper bodies." Faces are paired with bodies "using a matching routine that takes into account bounding box area and position, as well as the intersection of the face and upper body regions." [DOCUMENTED: [A-PPL21]]
- **Embeddings.** Face crops and upper-body crops go into "a pair of separate deep neural networks." [DOCUMENTED: [A-PPL21]]
  - The face network is "inspired by" AirFace with "significantly increased network depth." It uses MobileNetV3-style inverted-residual bottlenecks with squeeze-and-excitation attention, and ends in MobileFaceNet's "linear global depth-wise convolution." The output is L2-normalized, meaning scaled to unit length so it lies on a sphere. [DOCUMENTED: [A-PPL21]]
  - Training uses an ArcFace-style additive angular margin, which forces same-person embeddings into a tighter angle, plus "margin-mining" that up-weights hard negatives (as in SV-Softmax and CurricularFace). It trains from random initialization with AdamW and a One-Cycle learning-rate schedule. [DOCUMENTED: [A-PPL21]]
  - Augmentation includes "synthetic mask augmentation" for COVID-era masks. [DOCUMENTED: [A-PPL21]]
  - An "embedding confidence branch" flags out-of-distribution inputs, meaning inputs unlike the training data. [DOCUMENTED: [A-PPL21]]
  - **Embedding dimension, input resolution, and thresholds are not disclosed.** Speed: "less than 4ms" per face embedding on the ANE, "an 8x improvement over an equivalent model running on GPU." [DOCUMENTED: [A-PPL21]]
- **Clustering, pass 1 (greedy, conservative).** "When it joins two instances, they are permanently associated," so Apple tunes it so "each first-pass cluster only groups together very close matches, providing high precision but many, smaller clusters." [DOCUMENTED: [A-PPL21]]
  - Each cluster is represented by the running average of its embeddings. [DOCUMENTED: [A-PPL21]]
  - Distance: D = min(F, α·F + β·T), where F is face-embedding distance and T is upper-body distance. Upper bodies are compared "only from the same moment," where a moment is assets grouped by time and location metadata. [DOCUMENTED: [A-PPL21]]
- **Clustering, pass 2 (HAC, face only).** This pass forms "groups across moment boundaries" and is "increasing recall significantly." Linkage is "the median distance between the members of two HAC clusters," switching to random sampling when there are too many pairs to compare. [DOCUMENTED: [A-PPL21]]
  - Reported runtime: 2.4 s vs 69.2 s (iteration 35) and 3.5 s vs 205.8 s (iteration 65), compared with standard average-linkage HAC. [DOCUMENTED: [A-PPL21]]
- **Gallery selection.** "The set of the K largest clusters is likely to correspond to K different individuals." Photos picks gallery clusters "using a number of heuristics based on the distribution of cluster sizes, inter- and intra-cluster distances, and explicit user input." [DOCUMENTED: [A-PPL21]]
- **Assigning new photos.** Each person is represented by several "canonical exemplars," meaning stored example embeddings, rather than one average. A new face embedding y is approximated as a sparse weighted combination of all exemplars by solving min‖y − D·x‖² + λ‖x‖₁; the L1 term forces most weights to zero. The face goes to the person whose exemplars carry the largest total weight ("energy"). [DOCUMENTED: [A-PPL21]]
  - Apple says this beats nearest-neighbor matching "when the size of each cluster is relatively small, and when more than one cluster in the gallery could belong to the same identity." [DOCUMENTED: [A-PPL21]]
- **Fairness numbers.** Apple published only Figure 7: accuracy on the worst and best subsets of an internal dataset. [DOCUMENTED: [A-PPL21]]
  - No augmentation + large-margin softmax: 0.23 / 0.66.
  - No augmentation + margin-mining: 0.68 / 0.84.
  - Augmentation + margin-mining: 0.86 / 0.96.
  - There is no breakdown by demographic group and no false-match rate. Training data includes "a paid crowd-sourced model … spanning various age groups, genders, and ethnicities."

**How naming propagates:**
- Naming a person in one photo "automatically" adds them to People & Pets and names them "in other photos and videos in your library." On Mac, "All the photos in the collection are assigned the name." [DOCUMENTED: [G-IPH-PPL], [G-MAC-PPL]]
- Name suggestions come from Contacts. [DOCUMENTED: [S-PPL]]
- Corrections: "This is Not [name]" removes a photo; Merge combines duplicate groups; on Mac, "Review More Photos" offers suggested photos to accept or reject one by one. [DOCUMENTED: [S-PPL], [G-MAC-PPL]]
- Names and favorites sync across devices through iCloud Photos. [DOCUMENTED: [S-PPL], [A-LEGAL]]
- Mechanism behind these UI actions [INFERENCE]: a name attaches to a whole cluster, Merge is a union of clusters, and "Not X" acts as a constraint that the two items cannot be linked. Whether confirmed photos become exemplars in the sparse-coding dictionary is **not documented**.

**Later updates (2022–2026):**
- iOS 17 added dog and cat recognition in "People & Pets." [DOCUMENTED: [S-PETS]]
- I found no later Apple ML Research article on people recognition. I checked all 54 entries on the highlights index. The 2021 article remains the only public description of the mechanism. [DOCUMENTED: absence on the [A-HL] index]

## 2. When it runs, and where

- **Overnight batch.** The clustering algorithm "runs periodically, typically overnight during device charging." [DOCUMENTED: [A-PPL21]]
- **At capture.** Sparse-coding assignment is used "to quickly identify people as someone captures photographs." Face detection was designed for both "processing their photo libraries for face recognition" and "analyzing a picture immediately after a shot." [DOCUMENTED: [A-PPL21], [A-FD17]]
- **Conditions.** For People & Pets detection, "Your iPhone or iPad must be locked and connected to power." On Mac it runs "as long as you aren't using the app." It "could happen quickly or take up to a few days," and the album shows "Finding People..." while it works. [DOCUMENTED: [S-PETS]]
- **On-device, not server.**
  - Apple: "This on-device analysis includes scene classification, people and pets identification." [DOCUMENTED: [A-LEGAL]]
  - In 2017 Apple said "we couldn't use iCloud servers for computer vision computations" because iCloud photos are encrypted. [DOCUMENTED: [A-FD17]]
  - Precision note: the 2026 Platform Security Guide says that under *standard* protection, Photos service keys sit in Apple's hardware security modules and "can be accessed by some Apple services." Only with Advanced Data Protection are Photos end-to-end encrypted. So "Apple cannot decrypt" is true only with ADP on. "Apple does not" do server-side vision on photos is a policy statement. [DOCUMENTED: [APS26], [A-LEGAL]]
- **The search index is local.** "Search on iCloud.com … uses the local index on a user's iPhone, iPad, or Mac, enabling search from iCloud.com without needing to make an index available on iCloud servers." [DOCUMENTED: [APS26]]
- **The one server lookup: Enhanced Visual Search (landmarks).**
  - On the device, a model finds a "region of interest" that may contain a landmark and embeds it, quantized to 8 bits. [DOCUMENTED: [A-HE24]]
  - The query goes to the server using homomorphic encryption (the server computes similarity on encrypted vectors without seeing them) and private nearest-neighbor search. [DOCUMENTED: [A-HE24]]
  - Fake queries add differential privacy (ε = 0.8, δ = 10⁻⁶), and a third-party OHTTP relay hides the device's IP address. [DOCUMENTED: [A-HE24]]
  - An on-device reranker picks the final candidate. [DOCUMENTED: [A-HE24]]
- **New device / sync.** Apple documents that names sync. It does not document whether embeddings, clusters, or scene tags sync.
  - Macworld (2017) reported that face data passes through iCloud only in encrypted form. [DOCUMENTED: [MW17], report]
  - Community threads describe each device re-scanning on its own. [DOCUMENTED: [AC-HUNDREDS], report]
  - [INFERENCE] Expect at least the search index to be rebuilt on a new device.
- **How long for a ~150k library?** Apple gives no figure beyond "up to a few days." Reports:
  - 160,000 photos plus 15,000 videos stuck at "Photos is analyzing your library to provide accurate search results" on iOS 18. [DOCUMENTED: [AC-160K], report]
  - 40,000 images plus 3,000 movies took "over a week" on a Mac mini set never to sleep. [DOCUMENTED: [AC-160K], report]
  - An iPhone 16 Pro Max still indexing roughly 3 months after iOS 18 shipped. [DOCUMENTED: [BGR24], report]
- **Back-of-envelope compute** [INFERENCE]:
  - Scene model: Apple says all ANSA heads run in "under 9.7 milliseconds" on an iPhone 13 [A-ANSA22]. 150,000 × 9.7 ms ≈ 24 min.
  - Faces: 150k–300k faces (my assumption of 1–2 per photo) × <4 ms ≈ 10–20 min [A-PPL21].
  - Pure model inference is therefore under about an hour. Days to weeks of wall-clock time must come from other factors:
    - scheduling, since work only runs while the phone is locked and charging;
    - thermal and power budgets;
    - video analysis;
    - downloading iCloud originals when storage is optimized (unverified);
    - the heavier Apple Intelligence indexing models, whose cost is undisclosed.

## 3. Why the People album misses photos

**Documented design causes:**
1. **Precision-first first pass.** Joins are permanent, so thresholds are tight, giving "many, smaller clusters." A person's photos can sit in fragments that are not attached to the named cluster. [DOCUMENTED: [A-PPL21]]
2. **Body cues only work within a moment.** A back-turned photo can be linked only through clothing in the same time/place moment. The cross-moment pass "uses only face embedding matching." [DOCUMENTED: [A-PPL21]] → [INFERENCE] A faceless shot from a different day cannot be linked at all.
3. **Gallery promotion.** Only clusters passing size and distance heuristics become "known individuals." Clusters that are not promoted show up only via "Add People … to see more faces." [DOCUMENTED: [A-PPL21], [G-IPH-PPL]] A user reports needing several photos of a person before iOS shows a face icon. [DOCUMENTED: [AC-SINGLE], report]
4. **Unclear faces are filtered.** Detections that are "false positives or out-of-distribution" are removed via the confidence branch. [DOCUMENTED: [A-PPL21]] Which real faces get dropped (tiny, blurry, extreme pose) is [INFERENCE].
5. **Split identities.** Apple: "Sometimes the same person is identified in more than one group." [DOCUMENTED: [S-PPL]] [INFERENCE] Photos assigned to an unnamed duplicate cluster look "missing" from the named album until the clusters are merged.
6. **Unconfirmed suggestions.** "When Photos suggests additional photos … you can review them and confirm" (Review More Photos). Photos therefore holds lower-confidence candidates that it does not apply automatically. [DOCUMENTED: [G-MAC-PPL]; iPhone usage reported in [AC-HUNDREDS]]
7. **Indexing not finished.** Covered in Section 2. [DOCUMENTED: [S-PETS], [AC-160K]]

**Documented by research (accuracy drops under hard conditions):**
- **Profile faces.**
  - On the CFP benchmark, humans scored 96.24% on frontal–frontal pairs vs 94.57% on frontal–profile. A deep model dropped from 96.4% to 84.91%. [DOCUMENTED: [CFP16]]
  - A 2018+ server-scale model (ArcFace ResNet100) scores CFP-FP 98.87% vs LFW 99.83%. CFP-FP is frontal-vs-profile pairs; LFW is "Labeled Faces in the Wild," mostly near-frontal pairs. [DOCUMENTED: [ARC]]
  - The mobile-class AirFace, which Apple's network is based on, scores about 94% on CFP-FP vs about 99.2% on LFW. [DOCUMENTED: [AIRF]]
- **Age change.**
  - AirFace scores about 93% on AgeDB (age-gap pairs). MobileFaceNet scores 96.07% on AgeDB-30 vs 99.55% on LFW. [DOCUMENTED: [AIRF], [MFN]]
  - Children: true-accept rate at a 0.1% false-accept rate was 90.18% after 1 year between photos, falling to 73.33% after 3 years. For ages 0–4, a prior study cited there found 47.93%. [DOCUMENTED: [DEB17]]
  - [INFERENCE] A child photographed over many years is the textbook case for being split into several clusters.
- **Small or occluded faces.**
  - WIDER FACE defines small faces as 10–50 px. For that scale, generic region proposals stayed "below 30%" detection rate even at 10,000 proposals, and detection rate "decreases as occlusion level increases." [DOCUMENTED: [WIDER]]
  - A 2019 state-of-the-art detector, RetinaFace, reaches 91.4% AP on the WIDER Hard set. AP (average precision) is the area under the precision-recall curve. [DOCUMENTED: [RETINA]]
  - Mobile face embedders typically work on 112×112 crops. [DOCUMENTED: [MFN]] [INFERENCE] A 20-px face scaled up to that size carries little identity information, so it is likely to be filtered or left unassigned.
- **Masks.** These were bad enough that Apple added synthetic-mask training. [DOCUMENTED: [A-PPL21]]

## 4. How Apple Photos search works, and why "bread" can fail

**Classic path (all devices; the only path without Apple Intelligence):**
- **Model.** ANSA (Apple Neural Scene Analyzer) has been on-device since 2016.
  - The iOS 16 backbone is a MobileNetV3 variant trained contrastively on "a few hundred million image-text pairs." Contrastive training pulls matching image and caption embeddings together and pushes mismatches apart, as in CLIP. [DOCUMENTED: [A-ANSA22]]
  - Size and speed: 16M parameters; all heads in under 9.7 ms; about 24.6 MB of ANE memory and 16.4 MB on disk (iPhone 13). [DOCUMENTED: [A-ANSA22]]
  - There is also an on-device text encoder: 44M parameters, pruned about 40% to 26M, 8-bit, 3–4 ms per query. [DOCUMENTED: [A-ANSA22]]
  - Image-language pretraining gave +10.5% mAP over the old classification backbone. Benchmarks for the compact model vs Apple's ViT-B/16: ImageNet top-5 80.44 vs 87.54; COCO text-to-image top-5 57.32 vs 64.94. [DOCUMENTED: [A-ANSA22]]
- **Search mechanism.**
  - "Visual content search … is serviced through a fixed taxonomy image tagger. Synonyms of tags in the taxonomy are searchable, while a word-embedding model is used to mitigate null queries suggesting the closest searchable concept." [DOCUMENTED: [A-ANSA22]]
  - Tagging became "a small head" on the image-language backbone. [DOCUMENTED: [A-ANSA22]]
- **Taxonomy and thresholds.** In 2019, Apple said the Vision classifier "is in fact the same network we ourselves use to power the photo search experience." [DOCUMENTED: [WWDC19]] Its properties:
  - more than 1,000 categories;
  - multi-label: each class gets its own independent confidence, so scores do not sum to 1;
  - hierarchical: "dog might have children like Beagle, Poodle";
  - only visually identifiable classes: no "holiday," no occupations, no proper nouns or adjectives;
  - a class-specific threshold ("operating point") set from Apple's internal precision/recall tests. Confidence is lower for objects that are obscured (the transcript reads "office gated," evidently "obfuscated") or that "appear in odd lighting or at odd angles."
  - The 2019 taxonomy may have changed since.
- **Label count.** The revision-1 label list has 1,303 classes. It includes `bread`, `baked_goods`, `white_bread`, `bagel`, `croissant`, `naan`, `pita`, `sandwich`, and `food`. [DOCUMENTED: [GIST], a third-party dump of `knownClassifications`]
- **Other query types** (iOS 27 guide): date, place, business names, category, events, named people and pets, text inside images, and captions. [DOCUMENTED: [G-IPH-SRCH]]

**Apple Intelligence path (iOS 18.1 → iOS 27):**
- **Features.**
  - Natural-language descriptions such as "Maya skateboarding in a tie-dye shirt." [DOCUMENTED: [NR-OCT24]]
  - Finding "specific moments in clips," meaning segments inside videos. [DOCUMENTED: [NR-JUN24]]
  - "Smart completion suggestions." [DOCUMENTED: [NR-OCT24]]
- **Devices.** iPhone 15 Pro, iPhone 15 Pro Max, iPhone 16 or later; supported languages only. [DOCUMENTED: [G-IPH-SRCH]]
- **Privacy.** "User photos and videos are kept private on device." [DOCUMENTED: [NR-JUN24]]
- **Model: undisclosed.**
  - Not described in the AFM 2024 report, the AFM 2025 report, or the AFM 3 article (June 2026). [DOCUMENTED absence: [A-AFMTR24], [A-AFMTR25], [A-AFM3]]
  - Related published Apple components: the ANSA image-text backbone with its on-device text encoder [A-ANSA22], MobileCLIP [MCLIP], and MobileCLIP2/FastVLM [A-FVLM25].
  - [INFERENCE] The most plausible design: parse the query into structured parts (person names, dates, places) and run dense image-text embedding retrieval for the descriptive part, with per-segment embeddings for video. This is not confirmed. Third-party blog claims that Photos uses OpenAI's CLIP are unsourced; I ignored them.

**Why "bread" can return too few photos:**

| # | Cause | Status |
|---|---|---|
| 1 | The photo's `bread` score is below the class threshold (bread is small, partly hidden, or poorly lit) | Mechanism DOCUMENTED [WWDC19]; Photos' actual operating points are NOT documented |
| 2 | The index is not finished ("Photos is analyzing your library…") | DOCUMENTED [AC-160K], [S-PETS] |
| 3 | No Apple Intelligence (device or language), so only the taxonomy path runs | DOCUMENTED [G-IPH-SRCH] |
| 4 | The query is not a tag, so the word-embedding fallback suggests a neighbor concept instead of returning photos | DOCUMENTED [A-ANSA22]; *unlikely for "bread"*, which is a class [GIST] |
| 5 | Thresholds are set for precision over recall | INFERENCE. Google explicitly cut to "the most precise 1100 classes" for launch [GR13]; Apple tuned People for precision [A-PPL21]; Apple's search thresholds are unpublished |
| 6 | Hierarchy: the photo is tagged with a child class (`bagel`, `croissant`, `naan`) or only the parent (`baked_goods`, `food`), and the query for "bread" does not expand to those | INFERENCE. The hierarchy is DOCUMENTED [WWDC19]; whether search expands across it is not |
| 7 | The whole-image score is dominated by the salient subject (people, the table), so a peripheral loaf scores low | INFERENCE |
| 8 | In the natural-language path, results are cut off at a ranking or similarity cutoff, and contrastive models handle attributes and relations poorly | INFERENCE; bag-of-words weakness DOCUMENTED [ARO] |

## 5. Google Photos, Ask Photos, Samsung Gallery

**Google face groups:**
- Three steps:
  - detect faces;
  - "create face models that numerically represent the images of faces, predict the similarity … and estimate whether different images represent the same face";
  - group the faces.
  [DOCUMENTED: [GH-FACE]]
- Like Apple, it can use "photos being taken close together in time and detecting that a person is wearing the same clothing … when a face isn't visible." [DOCUMENTED: [GH-FACE]]
- Merges are suggested with Same / Different / Not sure buttons, and "Face group suggestions aren't perfect." [DOCUMENTED: [GH-FACE]]
- Face models are stored until you delete them or the account is inactive for more than 2 years. The feature is not available in all regions. [DOCUMENTED: [GH-FACE]]
- [INFERENCE] Processing is server-side: models are retained per account, and Ask Photos works only on backed-up images [GH-ASK].
- Google's FaceNet produces 128-byte embeddings trained with a triplet loss and targets clustering [FACENET]. Google has not stated that production Photos uses it. [INFERENCE that it is related]

**Google classic search:**
- Launch in 2013: a CNN based on the 2012 ImageNet-winning design. About 2,000 candidate classes were "refined … to the most precise 1100 classes for our launch," keyed to Freebase entities and combined with text tags and EXIF metadata. [DOCUMENTED: [GR13]]
- September 2024: descriptive natural-language queries for all users (English first) and sorting by date or relevance. "Nearly half a billion people use search in Photos every month." The mechanism is not disclosed. [DOCUMENTED: [GB-SEP24]]

**Ask Photos (Gemini):**
- **How it works.**
  - It "understands your query, and then forms a plan … issues a sophisticated search on your behalf, identifying … places, people and dates, but also natural language concepts." [DOCUMENTED: [GB-IO24]]
  - Then "Gemini's multimodal capabilities … understand exactly what's happening in each photo and can even read text," "crafts a helpful response and picks which photos and videos to return." [DOCUMENTED: [GB-IO24]]
  - Corrections go into a "Remember List" that it uses later. [DOCUMENTED: [GH-ASK]]
- **Two-stage since June 2025.** Ask Photos now shows classic-search results "right away while Gemini models continue to work in the background to find the most relevant photos." The help page says Gemini works "to narrow down and highlight the most relevant results." [DOCUMENTED: [GB-JUN25], [GH-ASK]]
- **Evidence of a candidate set.** Feedback reports include "a list of photos and videos that were studied by Ask Photos," with "sometimes AI-generated text descriptions of what's happening in the photo." [DOCUMENTED: [GH-PRIV]]
- [INFERENCE] This is retrieve-then-LLM-filter. Gemini can only promote photos the first stage retrieved, so end-to-end recall is capped by first-stage recall, and the Gemini stage is a precision step ("narrow down").
- **Availability.**
  - May 2024: announced.
  - September 2024: early access in the US.
  - June 2025: broader US rollout.
  - November 2025: "more than 100 new countries and regions, and 17 new languages."
  - February 2026: the "Ask" button in the US.
  - September 2026: "Photos in Gemini Spark" for US AI Pro and Ultra subscribers.
  [DOCUMENTED: [GB-IO24], [GB-SEP24], [GB-JUN25], [GB-NOV25], [GB-FEB26], [GB-SEP26]]
- **Eligibility.** Age 18 or older; Face Groups on, with your own face chosen; location estimates on. [DOCUMENTED: [GH-PRIV], updated February 18, 2026]
- **Completeness.** Not reported. The help page says it "may give unexpected or inaccurate results." [DOCUMENTED: [GH-ASK]]

**Samsung Gallery (Galaxy S25, April 2025):**
- Tag types "tripled" compared with the previous series; a "zero-shot" image analysis engine for objects never seen in training. [DOCUMENTED: [SAM25]]
- People grouping goes "beyond facial features to include clothing, time and location." [DOCUMENTED: [SAM25]]
- Natural-language search uses "a vision-language model that learns by associating images with text," with generative AI synthesizing training queries, "optimized and compressed … so it runs quickly on-device." [DOCUMENTED: [SAM25]]
- No recall reporting was found. [DOCUMENTED absence: [SAM25]]

## 6. Does any product tell the user how complete a result is?

**No evidence found.** What the products do show:
- **Apple.**
  - Indexing state, not completeness: "Finding People..." and "Photos is analyzing your library to provide accurate search results." [DOCUMENTED: [S-PETS], [AC-160K]]
  - "Review More Photos" exposes pending candidates, without a count of how many true matches might remain. [DOCUMENTED: [G-MAC-PPL]]
  - Apple does compute per-class precision/recall curves internally; the Vision API's operating points come from "internal tests." [DOCUMENTED: [WWDC19]] [INFERENCE] None of this is shown in the Photos UI; no Apple doc I reviewed mentions it.
- **Google.** Disclaimers only: "Face group suggestions aren't perfect"; Ask Photos "may give unexpected or inaccurate results." [DOCUMENTED: [GH-FACE], [GH-ASK]]
- **Samsung.** Nothing found. [DOCUMENTED absence: [SAM25]]
- **Scope.** This conclusion rests on the vendor documents listed below. It is absence of evidence, not proof. [INFERENCE] All three vendors' documented design choices lean toward precision. People clustering is "high precision", Google launched with "the most precise" classes, and Ask Photos "narrow[s] down." None estimates or shows the fraction of true matches it missed.

---

## Sources

- **A-PPL21** — Apple ML Research, "Recognizing People in Photos Through Private On-Device Machine Learning," 2021-07-28. https://machinelearning.apple.com/research/recognizing-people-photos
- **A-ANSA22** — Apple ML Research, "A Multi-Task Neural Architecture for On-Device Scene Analysis," 2022-06-07. https://machinelearning.apple.com/research/on-device-scene-analysis
- **A-FD17** — Apple ML Research, "An On-device Deep Neural Network for Face Detection," 2017-11-16. https://machinelearning.apple.com/research/face-detection
- **A-HE24** — Apple ML Research, "Combining Machine Learning and Homomorphic Encryption in the Apple Ecosystem," 2024-10-24. https://machinelearning.apple.com/research/homomorphic-encryption
- **A-HL** — Apple ML Research highlights index (checked 2026-10-02). https://machinelearning.apple.com/highlights
- **A-AFMTR24** — Apple Intelligence Foundation Language Models (2024). https://arxiv.org/abs/2407.21075
- **A-AFMTR25** — Apple Intelligence Foundation Language Models Tech Report 2025. https://arxiv.org/abs/2507.13575
- **A-AFM3** — Apple ML Research, "Introducing the Third Generation of Apple's Foundation Models," 2026-06-08. https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models
- **A-FVLM25** — Apple ML Research, "FastVLM," 2025-07-23. https://machinelearning.apple.com/research/fast-vision-language-models
- **MCLIP** — Vasu et al., "MobileCLIP," arXiv 2311.17049. https://arxiv.org/abs/2311.17049
- **WWDC19** — WWDC19 Session 222, "Understanding Images in Vision Framework" (transcript). https://developer.apple.com/videos/play/wwdc2019/222/
- **GIST** — VNClassifyImageRequest revision-1 identifiers (1,303), third-party dump. https://gist.github.com/ktustanowski/56c0d7541813868fed4aceb60ab5d149
- **S-PETS** — Apple Support 105081, "If you can't find pets in the People & Pets album," 2025-03-18. https://support.apple.com/en-us/105081
- **S-PPL** — Apple Support 108795, "Find People and Pets in Photos," 2026-01-16. https://support.apple.com/en-us/108795
- **G-IPH-PPL** — iPhone User Guide, "Find and name people and pets." https://support.apple.com/guide/iphone/find-and-name-people-and-pets-iph9c7ee918c/ios
- **G-MAC-PPL** — Photos for Mac Guide, "Find and name photos of people and pets." https://support.apple.com/guide/photos/find-and-name-people-and-pets-phtad9d981ab/mac
- **G-IPH-SRCH** — iPhone User Guide (iOS 27), "Search for photos and videos." https://support.apple.com/guide/iphone/search-for-photos-and-videos-iph392d77d5f/ios
- **A-LEGAL** — Apple, "Photos & Privacy," 2025-12-12. https://www.apple.com/uk/legal/privacy/data/en/photos/
- **APS26** — Apple Platform Security Guide, August 2026. https://help.apple.com/pdf/security/en_US/apple-platform-security-guide.pdf
- **NR-JUN24** — Apple Newsroom, "Introducing Apple Intelligence," 2024-06. https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/
- **NR-OCT24** — Apple Newsroom, "Apple Intelligence is available today," 2024-10. https://www.apple.com/newsroom/2024/10/apple-intelligence-is-available-today-on-iphone-ipad-and-mac/
- **AC-160K** — Apple Community thread 255823627 (Oct 2024–Feb 2025), report. https://discussions.apple.com/thread/255823627
- **AC-HUNDREDS** — Apple Community thread 255402671 (Jan 2024), report. https://discussions.apple.com/thread/255402671
- **AC-SINGLE** — Apple Community thread 255197932 (Oct 2023), report. https://discussions.apple.com/thread/255197932
- **BGR24** — BGR, "3 Months Later, iOS 18 Still Hasn't Finished Indexing Photos," 2024-12-27, report. https://www.bgr.com/tech/3-months-later-ios-18-still-hasnt-finished-indexing-photos-on-some-iphones/
- **MW17** — Macworld (G. Fleishman), People sync, 2017-12-13, report. https://www.macworld.com/article/230735/apple-photos-what-do-you-need-to-do-to-get-people-to-sync.html
- **CFP16** — Sengupta et al., "Frontal to Profile Face Verification in the Wild," WACV 2016. http://www.cfpw.io/paper.pdf
- **ARC** — Deng et al., "ArcFace," arXiv 1801.07698. https://arxiv.org/abs/1801.07698
- **AIRF** — Li et al., "AirFace," arXiv 1907.12256. https://arxiv.org/abs/1907.12256
- **MFN** — Chen et al., "MobileFaceNets," arXiv 1804.07573. https://arxiv.org/abs/1804.07573
- **DEB17** — Deb, Nain, Jain, "Longitudinal Study of Child Face Recognition," arXiv 1711.03990. https://arxiv.org/abs/1711.03990
- **WIDER** — Yang et al., "WIDER FACE," arXiv 1511.06523. https://arxiv.org/abs/1511.06523
- **RETINA** — Deng et al., "RetinaFace," arXiv 1905.00641. https://arxiv.org/abs/1905.00641
- **ARO** — Yuksekgonul et al., "When and why vision-language models behave like bags-of-words," arXiv 2210.01936. https://arxiv.org/abs/2210.01936
- **FACENET** — Schroff et al., "FaceNet," arXiv 1503.03832. https://arxiv.org/abs/1503.03832
- **GH-FACE** — Google Photos Help, "Set up & manage your face groups." https://support.google.com/photos/answer/6128838
- **GH-SRCH** — Google Photos Help, "Search by people, things & places." https://support.google.com/photos/answer/15235862
- **GH-ASK** — Google Photos Help, "Use Ask Photos to search, edit & get assistance." https://support.google.com/photos/answer/15318661
- **GH-PRIV** — Google Photos Help, "Gemini features in Photos privacy hub," updated 2026-02-18. https://support.google.com/photos/answer/15344015
- **GB-IO24** — Google Blog, "Ask Photos" (I/O), 2024-05-14. https://blog.google/products/photos/ask-photos-google-io-2024/
- **GB-SEP24** — Google Blog, "Improved search in Google Photos — plus early access to Ask Photos," 2024-09-05. https://blog.google/products/photos/google-ask-photos-early-access/
- **GB-JUN25** — Google Blog, "We're improving Ask Photos and bringing it to more Google Photos users," 2025-06-26. https://blog.google/products/photos/updates-ask-photos-search/
- **GB-NOV25** — Google Blog, "6 new things you can do with AI in Google Photos," 2025-11-11. https://blog.google/products-and-platforms/products/photos/nano-banana-ai-templates-ask-photos/
- **GB-FEB26** — Google Blog, "9 fun questions to try asking Google Photos," 2026-02-10. https://blog.google/products-and-platforms/products/photos/ask-button-ask-photos-tips/
- **GB-SEP26** — Google Blog, "5 Google Photos updates…," 2026-09-24. https://blog.google/products-and-platforms/products/photos/google-photos-updates/
- **GR13** — Google Research Blog, "Improving Photo Search: A Step Across the Semantic Gap," 2013-06-12. https://research.google/blog/improving-photo-search-a-step-across-the-semantic-gap/
- **SAM25** — Samsung Newsroom, "[Interview] A New and Enhanced Gallery Experience… Galaxy S25," 2025-04-30. https://news.samsung.com/global/interview-a-new-and-enhanced-gallery-experience-how-samsung-transformed-photo-searching-and-video-editing-with-the-galaxy-s25-series

[A-PPL21]: https://machinelearning.apple.com/research/recognizing-people-photos
[A-ANSA22]: https://machinelearning.apple.com/research/on-device-scene-analysis
[A-FD17]: https://machinelearning.apple.com/research/face-detection
[A-HE24]: https://machinelearning.apple.com/research/homomorphic-encryption
[A-HL]: https://machinelearning.apple.com/highlights
[A-AFMTR24]: https://arxiv.org/abs/2407.21075
[A-AFMTR25]: https://arxiv.org/abs/2507.13575
[A-AFM3]: https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models
[A-FVLM25]: https://machinelearning.apple.com/research/fast-vision-language-models
[MCLIP]: https://arxiv.org/abs/2311.17049
[WWDC19]: https://developer.apple.com/videos/play/wwdc2019/222/
[GIST]: https://gist.github.com/ktustanowski/56c0d7541813868fed4aceb60ab5d149
[S-PETS]: https://support.apple.com/en-us/105081
[S-PPL]: https://support.apple.com/en-us/108795
[G-IPH-PPL]: https://support.apple.com/guide/iphone/find-and-name-people-and-pets-iph9c7ee918c/ios
[G-MAC-PPL]: https://support.apple.com/guide/photos/find-and-name-people-and-pets-phtad9d981ab/mac
[G-IPH-SRCH]: https://support.apple.com/guide/iphone/search-for-photos-and-videos-iph392d77d5f/ios
[A-LEGAL]: https://www.apple.com/uk/legal/privacy/data/en/photos/
[APS26]: https://help.apple.com/pdf/security/en_US/apple-platform-security-guide.pdf
[NR-JUN24]: https://www.apple.com/newsroom/2024/06/introducing-apple-intelligence-for-iphone-ipad-and-mac/
[NR-OCT24]: https://www.apple.com/newsroom/2024/10/apple-intelligence-is-available-today-on-iphone-ipad-and-mac/
[AC-160K]: https://discussions.apple.com/thread/255823627
[AC-HUNDREDS]: https://discussions.apple.com/thread/255402671
[AC-SINGLE]: https://discussions.apple.com/thread/255197932
[BGR24]: https://www.bgr.com/tech/3-months-later-ios-18-still-hasnt-finished-indexing-photos-on-some-iphones/
[MW17]: https://www.macworld.com/article/230735/apple-photos-what-do-you-need-to-do-to-get-people-to-sync.html
[CFP16]: http://www.cfpw.io/paper.pdf
[ARC]: https://arxiv.org/abs/1801.07698
[AIRF]: https://arxiv.org/abs/1907.12256
[MFN]: https://arxiv.org/abs/1804.07573
[DEB17]: https://arxiv.org/abs/1711.03990
[WIDER]: https://arxiv.org/abs/1511.06523
[RETINA]: https://arxiv.org/abs/1905.00641
[ARO]: https://arxiv.org/abs/2210.01936
[FACENET]: https://arxiv.org/abs/1503.03832
[GH-FACE]: https://support.google.com/photos/answer/6128838
[GH-SRCH]: https://support.google.com/photos/answer/15235862
[GH-ASK]: https://support.google.com/photos/answer/15318661
[GH-PRIV]: https://support.google.com/photos/answer/15344015
[GB-IO24]: https://blog.google/products/photos/ask-photos-google-io-2024/
[GB-SEP24]: https://blog.google/products/photos/google-ask-photos-early-access/
[GB-JUN25]: https://blog.google/products/photos/updates-ask-photos-search/
[GB-NOV25]: https://blog.google/products-and-platforms/products/photos/nano-banana-ai-templates-ask-photos/
[GB-FEB26]: https://blog.google/products-and-platforms/products/photos/ask-button-ask-photos-tips/
[GB-SEP26]: https://blog.google/products-and-platforms/products/photos/google-photos-updates/
[GR13]: https://research.google/blog/improving-photo-search-a-step-across-the-semantic-gap/
[SAM25]: https://news.samsung.com/global/interview-a-new-and-enhanced-gallery-experience-how-samsung-transformed-photo-searching-and-video-editing-with-the-galaxy-s25-series
