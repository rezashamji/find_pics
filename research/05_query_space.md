# 05 — The query space: what people actually search for, and public data to test each kind

Date: 2026-10-02. Scope: find_pics must answer any natural-language request over a personal photo/video library.
Goal of this note: derive the query dimensions from evidence (logs, user studies, benchmarks built from real
camera rolls, vendor documentation), then map each dimension to public datasets that have ground truth.

Tags:
- `[DOC: S#]` = read at that source (list at the end).
- `[INF]` = my inference.
- `UNVERIFIED` = I could not confirm it.

HF license/gating fields were read live from `https://huggingface.co/api/datasets/<id>` on 2026-10-02 [DOC: S40].
An HF tag is whatever the uploader claimed. It is not proof of the original license.

---

## 0. Bottom line

1. **There is exactly one large real query-log study of personal photo search.** It is Jiang et al., WSDM 2017: 961,826 personal-search queries from Flickr, 2014–15 [DOC: S1].
   - Queries are very short: 1.5 words on average after stopwords are removed.
   - 85.3% of them are "visual", meaning they ask about what is in the picture.
   - They fall into four kinds: **what / who / where / when**. Users search "what" and "who" relatively more than they tag them.
   - That study comes from the keyword era. No public log of *natural-language* personal photo search exists (Apple/Google have not published one). [DOC: S1; INF for the "none exists" claim, based on my searches]
2. **Newer evidence comes from benchmarks built from real camera rolls.** These are PhotoBench, CamRoll, DISBench, ATM-Bench and MemexQA [DOC: S2–S6].
   - Their queries are written by annotators or LLMs, not logged from real users. So they show what people *plausibly* ask, not measured frequencies.
   - They agree on two things.
     - (a) Most real queries combine sources: visual content + time/place metadata + identity. PhotoBench puts roughly 55% of queries in combined categories (read off a figure, so approximate).
     - (b) Event context matters, e.g. "the beach two days after the fireworks".
3. **Vendor docs (Apple, Google) show which dimensions shipping products chose to support, not how often people use them** [DOC: S7–S12]:
   - people/pets by name, "me"
   - actions and expressions ("Dad smiling", "kids laughing")
   - objects, scenes, food
   - places, dates, trips
   - text in images (Google: exact match with quotes; Apple: Receipts / Handwriting / QR Codes / Documents collections)
   - duplicates (Apple's Duplicates album)
   - video moments ("baby crawling")
   - "best photo" curation (Google)
4. **No single public benchmark covers most dimensions.**
   - PhotoBench is the closest for retrieval. It has 3 real albums with GPS, time, faces, and *zero-result* queries.
   - It has little or nothing for OCR on screenshots/receipts, duplicates, aesthetics, counting, negation, or video moments [DOC: S2; INF on the gaps].
   - So the test suite has to be assembled from parts (section 4).

---

## 1. Evidence: what people search for in their own libraries

| Source | Method / N | What it says about the query space |
|---|---|---|
| Jiang, Kalantidis, Cao et al., "Delving Deep into Personal Photo and Video Search", WSDM 2017 [S1] | Flickr search logs Oct 2014–Oct 2015: 4.3M queries, >133k users, of which **961,826 personal-search queries** (339,349 unique) | <ul><li>Personal queries average **1.5 non-stopword words**. More than 90% of queries (all three search types) are under 4 words.</li><li>**85.3% of personal queries are visual**, vs 70.4% for web and 60.9% for social search.</li><li>Queries map to **4W categories**: what (object, thing, action, plant), who (person, animal), where (scene, country, city, GPS), when (year, month, holiday, date). Users search *what* and *who* more often than they tag them; tags lean toward *when* and *where*.</li><li>**Median clicked position is 2** (32 for web), so the top 2–3 results matter most.</li><li>About **77% of photos and 85% of videos have no user tags**, so content recognition is the only route.</li><li>In a 20-user "toy survey" of ideal queries, nearly all were task-driven ("show me photos with my kid at the playground a couple weeks ago") or question-driven ("what was the hotel name in our trip to Greece?", "who did I have dinner with in WSDM 2016?").</li><li>The exact 4W percentages are shown only in a figure (Fig. 2), not in the text: UNVERIFIED numbers.</li></ul> |
| Whittaker, Bergman, Clough 2010, "Easy on that trigger dad" [S13] | Family photo retrieval tasks (field study) | <ul><li>People **failed to find ~39%** of target photos of significant family events.</li><li>A 2022 follow-up (smartphone era): failures rose to 71% for photos in computer folders, and fell to 29% with phone apps that have timeline, search and face recognition [S13b].</li><li>The target was **events and people**, years later.</li></ul> |
| Rodden & Wood, CHI 2003 [S14] | 13 participants, 6 months, instrumented photo tool | <ul><li>People organize by **time, event, location**.</li><li>They create almost no manual metadata.</li><li>The advanced content-based search of that era was rarely used.</li></ul> |
| Mottelson 2023, "Why do people take screenshots on their smartphones?" (DIS) [S15] | 52 iPhone users, 1,679 screenshots annotated, plus metadata on all screenshots | <ul><li>Libraries hold on average **498.8 screenshots** (SD 705.3, median 195).</li><li>**76%** were for personal use, not sharing.</li><li>Instagram, Safari and Facebook account for >30% of the sample.</li><li>So **screenshots and text-heavy images are a large part of real libraries.**</li></ul> |
| PhotoBench (Xu, Shan et al., arXiv 2603.01493, 2026) [S2] | 3 real personal albums, **3,582 images**, **1,188 bilingual queries** synthesized from the owners' "life trajectory", expert-annotated ground truth | <ul><li>Source-aware taxonomy: Vision (V), Metadata (M: time/GPS), Face (F: identity/relationship), and the combinations VM, VF, MF, VMF.</li><li>Distribution read off Fig. 2, so approximate and UNVERIFIED: V ~32%, VM ~25%, VF ~12%, MF ~10%, M ~8%, VMF ~8%, F ~5%. Combined categories total about 55%. The text itself says only "a significant proportion".</li><li>83.4% of images have GPS + timestamp. 20 recurring people. 25.1% of images are portraits.</li><li>Includes **"unfulfillable" zero-ground-truth queries**, e.g. "Sunset photos at the beach last summer" when none exist, and scores rejection.</li><li>A second taxonomy (Table 8) has dimensions Location / Time / Person / Object / Concept, each either "fact" ("in May 2024", "a red car") or "cognitive" ("during Spring Festival", "with my girlfriend", "cozy moments with family"). Most queries carry 1–2 cognitive dimensions.</li><li>Examples: "cat under the car" (V), "passing Tianjin West Station" (M), "Dabao's photo" (F), "boyfriend on the beach at Shenzhen Bay" (VMF), "Photos of New Year's Eve dinner with my parents".</li><li>Finding: embedding-only retrieval collapses on M and F constraints.</li></ul> |
| CamRoll (arXiv 2606.05275, 2026) [S3] | 50 users, 31,476 images, 2,500 QA. Public part: 20 YFCC users, ~15.6k images | <ul><li>Questions are **written by annotators imagining the owner's life, not by owners** (the authors call this a limitation).</li><li>Answer types: What 611, Visual (attributes/counts) 518, Where 173, When 123, Who 75. These sum to 1,500 of 2,500, so the denominator of that table is unclear: UNVERIFIED.</li><li>Scope: 46.2% need a single image, 32.2% multiple images, 20.0% the whole roll.</li><li>Examples: "What did I eat for breakfast?", "Where did I eat dinner the day before going to the museum?", "When did I last see my grandfather?", "How many balloons were in the photo?"</li></ul> |
| DISBench / DeepImageSearch (ICML 2026, arXiv 2602.10809) [S4] | 57 YFCC users, **109,467 photos**, 122 queries (6.1% kept from 2,000 candidates) | <ul><li>Queries are **intra-event 46.7% / inter-event 53.3%**. All need context reasoning, e.g. "Find all photos with the sea taken at the beach two days after watching the fireworks show".</li><li>Target themes: portraits/people 41.8%, nature 18.9%, daily items 14.8%, scenic spots/architecture 11.5%.</li><li>Average 3.84 target images per query.</li><li>Best model reaches EM 28.7.</li></ul> |
| ATM-Bench ("According to Me", arXiv 2603.01990) [S5] | One person's ~4-year archive: 3,759 images, 533 videos, 6,741 emails; 1,038 QA | <ul><li>Answer types: open-ended 514, number 360, list-recall 139.</li><li>Example: "I remember seeing and taking a photo of a deer during a recent hiking trip — when and where did that happen?"</li><li>Some questions need receipts/emails, e.g. hotel expense totals.</li></ul> |
| MemexQA (Jiang et al. 2017) [S6] | 101 Flickr users, 630 albums, 13,591 photos, ~20.6k crowdsourced QA | <ul><li>Five question types: **what, who, where, when, how many**.</li><li>Example: "When did we last go hiking/camping?", "Who had a birthday party in January 2017?"</li><li>Albums come from VIST/SIND.</li></ul> |
| Ego4D NLQ (2022) [S16] | Egocentric video, ~74k queries / 800 h | <ul><li>13 memory-question templates in three groups: objects ("where did I put X", "how many X", "what is the state of X"), place, and people ("who did I talk to in location X").</li><li>The template list I found online had 12 items: UNVERIFIED exact list.</li></ul> |
| EgoLifeQA (CVPR 2025) [S17] | 6 people × 1 week, Aria glasses, 500 MCQ | Categories: EntityLog (object details, last use, location, price), EventRecall, HabitInsight, RelationMap (people), TaskMaster. |
| Lifelog Search Challenge 2022–24 review [S18] | ~725k wearable-camera images, 18 months; 24 tasks/yr | <ul><li>Task types: **known-item** (one image, clues revealed over 5 min), **ad-hoc** (find all), **QA** ("How many times did I visit an outdoor/farmers market in February 2020?").</li><li>Metadata includes OCR, GPS, biometrics.</li><li>Metaphorical or cultural language ("Starship Enterprise") made one task hard.</li></ul> |
| Apple Photos docs / coverage [S7–S9] | Vendor documentation | <ul><li>Natural-language examples: "Maya skateboarding in a tie-dye shirt", "Dad smiling", "Snowy sleeping" (pet), "kids laughing at swings", "couple dancing at wedding", "car next to lake", "yellow flowers in vase by window", "pizza with mushrooms", "Mum in green hat with wine glass".</li><li>Video moments: "baby crawling", "fireworks over lake".</li><li>Utilities collections: Duplicates, Receipts, Handwriting, Illustrations, QR Codes, Documents, Screenshots/Recently Saved, Maps.</li></ul> |
| Google Photos / Ask Photos [S10–S12] | Vendor documentation | <ul><li>"Alice and me laughing", "Kayaking on a lake surrounded by mountains", "Emma painting in the backyard", "colorful sunsets in Mexico".</li><li>**Exact text in quotes** searches text inside photos.</li><li>Ask Photos: "What did I eat on my trip to Barcelona?", "best photo from each national park I've visited", "What themes have we had for Lena's birthday parties?", "When your vouchers expire", "What's my car's license plate number?" (secondary source), "Suggest photos that'd make great phone backgrounds", "Where did we camp last time we went to Yosemite?"</li></ul> |

**What is *not* known [INF]:**
- No source gives a measured frequency for fine-grained dimensions such as negation, counts, emotion or aesthetics in real personal search.
- The only measured distribution is 4W on keyword-era Flickr logs. Flickr users also skew toward hobbyist photographers, not phone camera rolls.
- Treat everything below the 4W level as **"documented as asked" (exists), not "documented as frequent"**.

---

## 2. Taxonomy of query dimensions

Some columns abbreviate:
- **Freq.** = the frequency figure reported in the evidence, if any.
- In the Freq. column, "WSDM" means Jiang et al. 2017 [S1].

| # | Dimension | Evidence it is asked | Freq. | Example |
|---|---|---|---|---|
| D1 | **Specific person** (named, "me", relationship) | S1 (who), S2 (F, VF, MF, VMF ≈ 35% combined, approx.), S3 (Who 75), S7, S10 | WSDM: who is one of the top-2 searched categories (figure only). PhotoBench: about 35% of queries involve a face | "Alice and me laughing", "my sister" |
| D2 | **Several people / co-occurrence / relationship** | S2 ("dinner with my parents"), S6 ("who had a birthday party") | — | "me and Mom at the beach" |
| D3 | **Pets, category and specific animal** | S1 (who includes animal), S7 ("Snowy sleeping"), S10 (pets search) | — | "Snowy sleeping" |
| D4 | **Object / food / thing (category)** | S1 (what), S3 (What 611), S7 (food), S4 (daily items 14.8%) | WSDM: what is one of the top-2. CamRoll: largest answer type | "pizza with mushrooms" |
| D5 | **Specific instance** (my car, my mug, this painting) | S10 (license plate), S18 (KIS), Ego4D "where did I put my X" | — | "my blue bike" |
| D6 | **Scene / place type / landmark** | S1 (where = scene, city), S4 (scenic spots 11.5%) | — | "sunset over city skyline", "Eiffel Tower" |
| D7 | **Geographic place (GPS / place name)** | S1, S2 (M, VM ≈ 33% approx.), S3 (Where 173), S10 | PhotoBench M+VM+MF+VMF ≈ 51% need metadata (approx.) | "photos from Tokyo in 2025" |
| D8 | **Time** (absolute, relative, holiday, season) | S1 (when), S2 (Time "cognitive": "Spring Festival"), S3 (When 123), S6 | — | "last summer", "a couple weeks ago" |
| D9 | **Event / occasion / trip** | S1, S4 (intra/inter-event 100%), S6 (weddings, birthdays), S13, S14, S11 | DISBench: every query is event-anchored | "Lena's birthday parties", "our trip to Greece" |
| D10 | **Activity / action** | S1 (what includes action), S7 ("skateboarding", "dancing"), S10 ("kayaking") | — | "kids laughing at swings" |
| D11 | **Text in image / OCR** (receipts, signs, documents, screenshots, QR) | S8 Utilities, S10 (quoted exact text), S11 ("vouchers expire"), S15 (screenshots ~499/library), S5 (receipts), S18 (OCR) | Mottelson: mean 498.8 screenshots per library (median 195) | "\"Blue Bottle\"", "my passport", "Wi-Fi password screenshot" |
| D12 | **Counting / quantity** | S3 ("how many balloons"), S5 (number 360/1,013), S6 (how many), S18 (QA) | ATM-Bench: 360 of 1,013 main-set questions have numeric answers (these include aggregate counts over the archive, not only in-image counts) | "group photo with 5 people" |
| D13 | **Attributes** (color, clothing, appearance) | S7 ("tie-dye shirt", "green hat"), S2 ("a red car"), S3 (Visual 518) | CamRoll Visual 518 | "Mum in green hat" |
| D14 | **Spatial relation / composition** | S2 ("cat under the car"), S7 ("car next to lake", "flowers in vase by window") | — | "dog on the couch" |
| D15 | **Negation / exclusion** | No direct log evidence found. UNVERIFIED as a frequent need. PhotoBench zero-GT queries test the related "nothing matches" case | — | "beach photos without people" |
| D16 | **Expression / emotion / mood** | S7 ("Dad smiling", "kids laughing"), S10 ("laughing"), S2 ("cozy moments") | — | "Dad smiling" |
| D17 | **Aesthetics / quality / "best"** | S11 ("best photo from each national park", "great phone backgrounds"), S2 (aesthetics in V) | — | "best photo of the trip" |
| D18 | **Duplicates / near-duplicates / copies** | S8 (Apple Duplicates), California-ND motivation [S30] | — | "remove burst duplicates" (find_pics may only *list* them, never delete) |
| D19 | **Media type** (video, selfie, screenshot, live photo) | S9 ("by media type"), S10 | — | "videos of the kids" |
| D20 | **Video moment** | S7 ("baby crawling", "fireworks over lake") | — | "the moment she blew out the candles" |
| D21 | **Multi-hop / compositional across photos** | S2 (≈55% combined), S3 (32.2% multi-image + 20.0% whole roll), S4 (100%), S5 | DISBench inter-event 53.3% | "dinner the day before the museum" |
| D22 | **Question answering, not retrieval** | S1 (toy survey), S3, S5, S6, S11 | — | "what was the hotel name?" |
| D23 | **Empty-answer / false-memory queries** | S2 (zero-GT queries with rejection scoring) | — | "beach sunset last summer" (none exist) |

**Scope note [INF]:**
- D22 (answer a question) and D21 (multi-hop) sit on top of retrieval. For find_pics they decompose into D1–D20 plus a planner step.
- D23 matters directly for the recall/precision honesty goal in PLAN.md: the tool must be able to say "none found".

---

## 3. Public datasets per dimension

Columns:
- **HF** = available on Hugging Face (id), "no" if only the original site.
- **Personal-like** = how close the images are to camera-roll photos (H/M/L) [INF].

Licenses are as documented. "NC" = non-commercial. Every image stays out of git per CLAUDE.md, whatever its license.

### 3a. Benchmarks built from real personal libraries (multi-dimension)

| Dataset | Size | Labels | License | HF | Notes |
|---|---|---|---|---|---|
| **PhotoBench** [S2, S19] | 3 albums, 3,582 imgs, 1,188 queries (300 val with GT, 887 test with GT hidden) | Query → exact GT set; V/M/F taxonomy; GPS + time; face identities; zero-GT queries | HF tag cc-by-nc-4.0; paper metadata says CC BY 4.0; GitHub viewer says MIT, so the sources conflict: UNVERIFIED, treat as NC | `SorrowTea/PhotoBench` (~11 GB; "Protected" variant ships features only) | The only one with face + GPS + time + empty-result queries. Chinese-centric places. |
| **DISBench** [S4, S20] | 57 users, 109,467 photos, 122 queries | GT photo IDs; timestamps, GPS, device, reverse-geocoded address | HF tag MIT; images are YFCC CC (per-image license) | `RUC-NLPIR/DISBench` (14.5 GB) | Best "real library backbone" with real EXIF. Few queries. |
| **CamRoll-yfcc20** [S3, S21] | 20 users, ~15k imgs, MCQ QA | Semantic + episodic QA; capture timestamps; owner profile photo | cc-by-nc-sa-4.0 | `thaoshibe/camroll-yfcc20` (2.29 GB) | QA, not retrieval sets. No GPS mentioned. |
| **ATM-Bench** [S5, S22] | 1 archive: 3,759 imgs + 533 videos + 6,741 emails; 1,038 QA | Answer + evidence IDs; GPS geocode cache | HF tag cc-by-nc-4.0 (repo code MIT) | `Jingbiao/ATM-Bench` (~3.3 GB) | Includes receipts/email-grounded questions. |
| **MemexQA** [S6] | 101 users, 630 albums, 13,591 photos, ~20.6k QA | what/who/where/when/how-many QA; time, GPS, tags, captions | UNVERIFIED | no (memexqa.cs.cmu.edu) | Older; VIST albums. |
| **LSC / NTCIR Lifelog** [S18, S23] | ~725k wearable images, 18 months | KIS/ad-hoc/QA topics; OCR, GPS, biometrics | Request from organisers, ethics approval | no | Wearable first-person view, unlike phone photos; identity retrieval out of scope (faces redacted). |
| **EgoLife / Ego4D NLQ** [S16, S17] | Video | Memory queries | Ego4D: signed license | no / partial | Video-memory taxonomy source; not camera-roll-like. |

### 3b. Single-dimension datasets with ground truth

| Dim | Dataset | Size | Labels | License | HF | Personal-like |
|---|---|---|---|---|---|---|
| D1/D2 person | **PIPA** [S24] | 37,107 Flickr album photos, 63,188 head boxes, 2,356 identities | Identity per head box; album structure | Flickr CC images (per-image); annotation license UNVERIFIED | no (GitHub metadata `coallaoh/PIPA_dataset`) | **H**: real albums, back views, non-frontal |
| D1 person | CelebA / LFW (already in repo) | — | Identity, 40 attributes (CelebA) | Non-commercial research (CelebA) | — | L–M |
| D2 relations | **PISC** [S25] | 22,670 imgs | Person boxes + pairwise relation (friends/family/couple/professional/commercial) | CC BY 4.0 | no | M |
| D1 | Gallagher Collection Person [S25] | 589 imgs, 931 faces, 32 identities | Identity in family photos | UNVERIFIED | no | H, but tiny |
| D3 pets (category) | **Oxford-IIIT Pet** [S26] | ~7.4k imgs, 37 breeds | Breed, head box, trimap | CC BY-SA 4.0 | `HuggingFaceM4/Oxford-IIIT-Pets` | M |
| D3 pets (individual) | **PetFace** [S27] | 1,012,934 imgs, 257,484 individuals, 13 families | Individual ID, breed, sex, color | UNVERIFIED (request page) | no | M (face crops) |
| D3 | DogFaceNet (already in repo) | 840 dogs / 6,215 photos used | Individual ID | — | `dimidagd/DogFaceNet_large` (unofficial) | M |
| D4 objects | **Open Images V7** [S28] | ~9M imgs; 16M boxes / 600 classes; 3.3M relation annotations; point labels; localized narratives | Labels, boxes, relations, narratives | Annotations CC BY 4.0, images CC BY 2.0 | partial mirrors | **H**: Flickr user photos |
| D4/D14 | COCO (already in repo) | 123k | 80 classes, captions | CC BY 4.0 annotations; Flickr images | yes | M–H |
| D5 instance | **ILIAS** [S29] | 1,000 object instances; 1,232 image queries + 4,715 positives + **1,000 text queries**; 100M YFCC distractors | Instance identity | MIT (repo); HF tag "cc" | `vrg-prague/ilias` | M; built to emerge after 2014 so YFCC holds no false negatives |
| D5 personal concept | ConCon-Chi; PerVL "This-Is-My" [S31] | Small | Personal concept × context | UNVERIFIED | UNVERIFIED | M |
| D6 landmarks | **GLDv2** [S32] (mini already in repo) | 5M imgs, 200k landmarks; retrieval index 762k, 118k queries | Landmark ID | CC BY-NC-SA 4.0 (per paper/summary; image licenses vary) | `zguo0525/google-landmarks-v2-mini` (unofficial) | M |
| D6 | ROxford / RParis [S32] | 4,993 / 6,322 imgs, 70 queries each | Easy/hard positives | Research | mirrors | M |
| D7/D8 place+time | **YFCC100M metadata** [S33] | 99.2M photos + 0.8M videos | EXIF time, GPS, user, per-image CC license | Per-image CC | `dalle-mini/YFCC100M_OpenAI_subset` (subset) | **H** |
| D9 events | **PEC** [S34] | 61,364 imgs, 807 albums, 14 classes (birthday, Christmas, hiking, …) | Album event label | UNVERIFIED | no | **H** |
| D9 events | **ML-CUFED** [S34] | 94,798 imgs, 1,883 albums, 23 event types (multi-label) + importance scores | Album events, per-image importance | UNVERIFIED (YFCC-derived) | `Shawn-Huang/CUFED-AlbumBench` (unofficial, tag cc-by-nc-2.0) | **H**; importance also tests D17 "best" |
| D9/D21 | **VIST / SIND** [S35] | 10,049 albums, 209,652 photos; SIND v1 81,743 photos / 20,211 sequences | Per-image captions + stories | UNVERIFIED | no | **H** |
| D10 actions | Stanford40, HICO-DET | — | Action labels | — | — | M. Not researched here: UNVERIFIED |
| D11 scene text | **TextOCR** [S36] | 28k imgs (Open Images), 903k words | Word boxes + transcripts | CC BY 4.0 | `yunusserhat/TextOCR-Dataset` (mirror) | **H** |
| D11 | **TextCaps** [S36] | 28,408 imgs, 145,329 captions that mention the text | Captions quoting the text, which can serve as retrieval queries | Open Images CC BY 2.0 images | `facebook/textvqa` (TextVQA, cc-by-4.0) | **H** |
| D11 receipts | **CORD v2** [S37] | 1,000 receipts | Fields (menu, price, total) | CC BY 4.0 | `naver-clova-ix/cord-v2` | **H**: phone photos of receipts |
| D11 receipts | **SROIE** [S37] | 973–987 receipts | Company, date, address, total | HF mirror tag cc-by-4.0; original ICDAR terms UNVERIFIED | `jsdnrs/ICDAR2019-SROIE` | M (scans) |
| D11 screenshots | **ScreenQA (RICO)** [S38] | 86,025 QA over RICO app screens | QA on screen text/UI | CC BY 4.0 | `bevaya/RICO-ScreenQA` | M (Android UI, older) |
| D12 counts | **CountBench** [S39] | 540 (491 still downloadable), 60 per count 2–10 | Count + caption | LAION-derived; license tag absent: UNVERIFIED | `vikhyatk/CountBenchQA`, `nielsr/countbench` | M |
| D13 attributes | CelebA 40 attributes (repo) + Open Images relations ("is" attributes) [S28] | — | Hat, glasses, color… | — | — | M |
| D13 clothing | CUHK-PEDES / RSTPReid [S41] | 40,206 imgs / 20,505 imgs | Free-text clothing descriptions | UNVERIFIED | — | **L**: surveillance pedestrians. Not recommended |
| D14 spatial | **VSR** [S42] | 10,972 COCO image-text pairs, 66 relations | True/false per relation | CC BY 4.0 | `cambridgeltl/vsr_random` | M–H |
| D14 | What'sUp [S42] | 820 imgs | left/right/on/under | UNVERIFIED | — | L (staged) |
| D14/D21 composition | **SugarCrepe** [S43] | ~7.5k pairs (replace/swap/add × obj/att/rel) on COCO | Positive vs hard-negative caption | MIT (per mirror) | `HuggingFaceM4/SugarCrepe_*` | M–H |
| D14 | **Winoground** [S44] | 400 examples (800 imgs, 800 captions) | Same words, different order | Gated license | `facebook/winoground` (gated: auto) | M |
| D15 negation | **NegBench** [S45] | 79k examples, 18 variants; COCO Retrieval-Neg 5,000; MCQ-Neg 5,914; plus VOC, MSR-VTT | Negated queries with GT | Code MIT; data via repo | no official HF | M–H (COCO) |
| D16 expression | **CelebA "Smiling"** (repo); **ExpW** [S46] | ExpW 91,793 faces | 7 expressions | ExpW "Other": UNVERIFIED | unofficial | M |
| D16 | RAF-DB / AffectNet [S46] | ~30k / ~400k+ | Expressions, valence/arousal | **Restricted**: RAF-DB NC + no redistribution; AffectNet signed by PI only | HF copies exist but are **unauthorized re-uploads: do not use** | M |
| D16 mood | **EmoSet-118K** [S47] | 118,102 labeled (3.3M total) | 8 emotions + brightness, colorfulness, scene, object, facial expression, action | UNVERIFIED | `Woleek/EmoSet-118K` (unofficial) | M |
| D17 aesthetics | **AVA** [S48] | 255,530 (DPChallenge) | Score distributions, 66 semantic, 14 style tags | Research: UNVERIFIED | `trojblue/AVA-aesthetics-10pct…` (subset) | L (contest photos) |
| D17 quality | **SPAQ** [S48] | 11,125 smartphone photos, 66 phones | Quality MOS + attributes + scene | UNVERIFIED | no | **H** |
| D17 | KonIQ-10k [S48] | 10k YFCC imgs | Quality MOS | UNVERIFIED | — | H |
| D18 near-dup | **California-ND** [S30] | 701 photos from one real vacation; 10 raters; 4,609 ND pairs | Pairwise ND votes (82% of pairs had some disagreement) | UNVERIFIED | no | **H** |
| D18 copies | **DISC21** [S49] | 1M reference + 50k dev + 50k test queries | Edited copy → source | Meta research license (manipulated-media research) | no | M |
| D20 video moments | **DiDeMo** [S50] | 10,464 Flickr personal videos, 26,892 moments, 40,543 descriptions | Moment timestamps | "Other": UNVERIFIED | `VLM2Vec/DiDeMo`, `friedrichor/DiDeMo` | **H**: unedited personal videos |
| D20 | **QVHighlights** [S51] | 10,148 YouTube videos, 10,310 queries, 18,367 moments, saliency scores | Moments + highlight scores | Annotations CC BY-NC-SA 4.0; videos under fair use | `ayushsdev/qvhighlights-videos` | M (vlogs, news) |
| D20 | Charades-STA [S52] | 16,128 query–moment pairs (indoor Charades) | Moments | Charades: non-commercial, no redistribution | `lmms-lab/charades_sta` (unofficial) | M (staged indoor) |
| D23 empty | PhotoBench zero-GT queries [S2]; "Moment of Untruth" negative queries for video MR [S53] | — | — | — | — | — |

Also relevant: **MIEB** (ICCV 2025) [S54] is a 130-task image-embedding benchmark. It includes document/OCR, compositionality and visual STS, so it is a ready harness for the embedding layer. It does not cover personal libraries.

---

## 4. Is there one benchmark that spans most dimensions?

**No** [INF, from coverage below].

| Benchmark | Covers well | Missing |
|---|---|---|
| PhotoBench | D1, D2, D4, D6–D9, D13, D14 (some), D21, D23 | D11 screenshots/receipts at scale, D12, D15, D17 GT, D18, D20 (photos only) |
| DISBench | D7–D9, D21, real EXIF at 109k scale | Faces as names, OCR, video; only 122 queries |
| CamRoll | D4, D8, D12 (visual), D22 | Retrieval GT sets, GPS |
| ATM-Bench | D11 receipts/email, D12 aggregates, D20 videos (533), D22 | One person; QA not retrieval |
| LSC | D8–D11 (OCR), D21 | Wearable view, faces redacted, access by request |

The most useful combination is **PhotoBench + DISBench + CamRoll + ATM-Bench**. Together they cover the "real library, real metadata, multi-source" core.

Every other dimension needs a probe set injected into a library.

---

## 5. Recommended test suite: one synthetic "personal library"

### Design principles [INF]

1. **Backbone = real libraries with real EXIF.** Time and place ground truth must come from real capture metadata. Fake timestamps on CelebA-style images would test the planner, not the system.
2. **Inject small probe sets** of 200–1,000 items each, so the library stays at a realistic size (20–40k items). It should not be dominated by any one source.
3. **Source-leakage control.** If every positive for a dimension comes from one injected dataset, a model can "succeed" by recognizing the dataset's style (scanned receipt, 48×48 face) instead of the content. So:
   - For every probe, include **same-source hard negatives**: receipts from a different shop, TextOCR images without the queried word, COCO images that differ only in the swapped relation.
   - Report a per-source "always say yes" baseline next to each metric.
4. **Every query gets a denominator** (GT-set size), plus zero-GT queries (D23) so precision on "nothing exists" is measured.
5. **Licenses**: nearly all of this is NC or research-only. That is fine for internal evaluation. Images stay in `data/` and are never committed (existing rule).

### Proposed components (13)

| # | Component | Use for | Size to inject |
|---|---|---|---|
| 1 | **DISBench** (5–10 users' photos + their queries) | Backbone with real EXIF; D7, D8, D9, D21 | ~15–20k photos, 122 queries (subset relevant to chosen users) |
| 2 | **PhotoBench** (validation split; kept as its own library because its ground truth is per-album) | D1, D2, D7, D8, D13, D21, D23 with faces | 3,582 imgs, 300 val queries |
| 3 | **CamRoll-yfcc20** | D22 QA; second backbone family | 2–5 users (~2.5–5k imgs) |
| 4 | **PIPA** subset + existing CelebA/LFW | D1 non-frontal identity in album context | ~150 identities, ~3k photos |
| 5 | **PetFace** (or existing DogFaceNet) + **Oxford-IIIT Pet** | D3 individual and breed | ~100 individuals × 5 + 37 breeds × 20 ≈ 1.2k |
| 6 | **ILIAS** (positives + text queries) + existing GLDv2-mini | D5 specific objects, D6 landmarks | ~500 instances (~2.5k imgs) + ~300 landmarks |
| 7 | **TextOCR/TextCaps** + **CORD v2** + **SROIE** + **ScreenQA/RICO** | D11 signs, receipts, documents, screenshots | ~800 + 500 + 300 + 800 ≈ 2.4k |
| 8 | **SugarCrepe** + **VSR** + **NegBench (COCO Retrieval-Neg)** + **Winoground** | D14 spatial/composition, D15 negation (on COCO images inside the library) | ~1.5k COCO images; Winoground's 800 images kept as a side test (gated, tiny) |
| 9 | **CountBench** | D12 | 491 |
| 10 | **ML-CUFED** or **PEC** (whole albums) | D9 events/occasions; D17 per-image importance | ~60 albums (~3–5k imgs) |
| 11 | **EmoSet-118K** subset + CelebA "Smiling" / ExpW | D16 expression and mood | ~1k |
| 12 | **SPAQ** + **California-ND** | D17 quality, D18 near-duplicates, both on smartphone-like photos | ~1k SPAQ + 701 California-ND |
| 13 | **DiDeMo** (+ QVHighlights val as a second check) | D19/D20 video moments in personal-style videos | ~300 DiDeMo videos (~1.2k descriptions) |

Approximate total: **~35–45k items**. That is the right order for a phone library and runs in one Slurm index job [INF].

Run component 2 (PhotoBench) and the ATM-Bench archive as **separate libraries** with their own query sets. Their ground truth is defined relative to their own album. Mixing their photos into the shared library would make some of their "unique" answers ambiguous [INF].

### Gaps this suite still leaves [INF]

- **Real query frequencies**: none are public for natural-language personal search. Weight dimensions using the 4W ordering from WSDM 2017 [S1] and the PhotoBench mix [S2], and flag the weighting as an assumption.
- **The owner's own identity ("me")**: only PhotoBench and CamRoll (profile photo) model an "owner". PIPA does not.
- **Body shape / appearance-change queries** (the PLAN.md example "where he looks heavy"): no public dataset was found with labels for this in personal photos (UNVERIFIED that none exists). Within-person attribute ground truth will have to be self-labelled.
- **Handwriting, QR codes, ID documents** (Apple Utilities categories): not covered above. Candidates (IAM handwriting, synthetic QR) were not researched here: UNVERIFIED.

---

## Sources

- S1. Jiang, Kalantidis, Cao, Farfade, Tang, Hauptmann. "Delving Deep into Personal Photo and Video Search." WSDM 2017. https://www.skamalas.com/docs/WSDM_2017.pdf (full text read)
- S2. Xu, Shan et al. "PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval." arXiv 2603.01493. https://arxiv.org/abs/2603.01493 (HTML + PDF text read)
- S3. "Personal AI Agent for Camera Roll VQA" (CamRoll). arXiv 2606.05275. https://arxiv.org/html/2606.05275v1 ; https://github.com/thaoshibe/camroll
- S4. "DeepImageSearch: Benchmarking Multimodal Agents for Context-Aware Image Retrieval in Visual Histories" (DISBench). arXiv 2602.10809. https://arxiv.org/html/2602.10809v2 ; https://github.com/RUC-NLPIR/DeepImageSearch
- S5. Mei et al. "According to Me: Long-Term Personalized Referential Memory QA" (ATM-Bench). arXiv 2603.01990. https://arxiv.org/html/2603.01990v1 ; https://github.com/JingbiaoMei/ATM-Bench
- S6. Jiang et al. "MemexQA: Visual Memex Question Answering." arXiv 1708.01336. https://arxiv.org/abs/1708.01336
- S7. MacRumors, "iOS 18.1: Use Natural Language Search in Photos." https://www.macrumors.com/how-to/ios-use-natural-language-search-photos/ (secondary; examples match Apple's guide)
- S8. Apple Support, "Find receipts, QR codes, recently edited photos, and more on iPhone." https://support.apple.com/guide/iphone/find-receipts-qr-codes-edited-photos-iph995007f21/ios ; Duplicates: https://support.apple.com/guide/iphone/merge-duplicate-photos-and-videos-iph1978d9c23/ios
- S9. Apple Support, "Use Apple Intelligence in Photos on iPhone." https://support.apple.com/en-tj/guide/iphone/iphf7de217f0/ios
- S10. Google Photos Help, "Search by people, things & places in your photos." https://support.google.com/photos/answer/15235862
- S11. Google blog, "Ask Photos" (I/O 2024). https://blog.google/products-and-platforms/products/photos/ask-photos-google-io-2024/ ; Sept 2024 update: https://blog.google/products-and-platforms/products/photos/google-ask-photos-early-access/
- S12. Google blog, "We're improving Ask Photos…" (2025-06-26). https://blog.google/products-and-platforms/products/photos/updates-ask-photos-search/ ; license-plate example: https://www.analyticsvidhya.com/blog/2024/05/how-to-use-ask-photos-feature-in-google-photos/ (secondary)
- S13. Whittaker, Bergman, Clough. "Easy on that trigger dad." Pers. Ubiquit. Comput. 2010. https://link.springer.com/article/10.1007/s00779-009-0218-7 ; S13b follow-up: https://link.springer.com/article/10.1007/s00779-022-01677-x (both via search summary; numbers not read in full text)
- S14. Rodden & Wood. "How do people manage their digital photographs?" CHI 2003. http://www.rodden.org/kerry/chi2003.pdf (via search summary)
- S15. Mottelson. "Why do people take Screenshots on their Smartphones?" DIS 2023. http://aske.mottelson.dk/wp-content/uploads/2023/05/Mottelson2023b.pdf (full text read)
- S16. Grauman et al. "Ego4D." https://ego4d-data.org/docs/benchmarks/episodic-memory/ ; template list via search summary
- S17. "EgoLife: Towards Egocentric Life Assistant." arXiv 2503.03803. https://arxiv.org/html/2503.03803v1
- S18. "The State-of-the-Art in Lifelog Retrieval: A Review of Progress at the ACM LSC 2022–24." arXiv 2506.06743. https://arxiv.org/html/2506.06743
- S19. PhotoBench release: https://github.com/LaVieEnRose365/PhotoBench ; https://huggingface.co/datasets/SorrowTea/PhotoBench
- S20. https://huggingface.co/datasets/RUC-NLPIR/DISBench
- S21. https://huggingface.co/datasets/thaoshibe/camroll-yfcc20
- S22. https://huggingface.co/datasets/Jingbiao/ATM-Bench
- S23. NTCIR-14 Lifelog-3 overview. https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings14/pdf/ntcir/01-NTCIR14-OV-LIFELOG-GurrinC.pdf (via search summary)
- S24. Zhang et al. "Beyond Frontal Faces" (PIPA), CVPR 2015. https://arxiv.org/abs/1501.05703 ; https://github.com/coallaoh/PIPA_dataset ; https://exposing.ai/pipa/
- S25. PISC: https://sites.google.com/site/yongkangwong/datasets ; Gallagher: http://chenlab.ece.cornell.edu/people/Andy/ImagesOfGroups.html (via search summary)
- S26. Oxford-IIIT Pet. https://www.robots.ox.ac.uk/~vgg/data/pets/
- S27. PetFace (ECCV 2024). https://arxiv.org/abs/2407.13555 ; https://github.com/mapooon/PetFace
- S28. Open Images V7. https://storage.googleapis.com/openimages/web/factsfigures_v7.html ; https://research.google/blog/open-images-v7-now-featuring-point-labels/
- S29. ILIAS (CVPR 2025). https://github.com/ilias-vrg/ilias ; https://huggingface.co/datasets/vrg-prague/ilias
- S30. Jinda-Apiraksa et al. "California-ND." https://qualinet.github.io/databases/image/california_nd_an_annotated_dataset_for_near_duplicate_detection_in_personal_photo_collections/
- S31. ConCon-Chi (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024/papers/Rosasco_ConCon-Chi_Concept-Context_Chimera_Benchmark_for_Personalized_Vision-Language_Tasks_CVPR_2024_paper.pdf ; PerVL: https://arxiv.org/abs/2204.01694 ; This-Is-My: https://arxiv.org/html/2306.10169 (not read in full)
- S32. GLDv2: https://arxiv.org/abs/2004.01804 ; ROxford/RParis: https://arxiv.org/abs/1803.11285
- S33. YFCC100M: https://cacm.acm.org/research/yfcc100m/ ; https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset
- S34. CUFED / ML-CUFED / PEC: https://arxiv.org/abs/1707.05911 ; https://arxiv.org/html/2109.12499
- S35. VIST: Huang et al. 2016. https://www.researchgate.net/publication/305334393_Visual_Storytelling (via search summary)
- S36. TextOCR: https://arxiv.org/html/2105.05486v1 ; TextCaps: https://arxiv.org/abs/2003.12462
- S37. CORD: https://github.com/clovaai/cord ; SROIE mirror: https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE
- S38. ScreenQA: https://arxiv.org/abs/2209.08199 ; https://huggingface.co/datasets/bevaya/RICO-ScreenQA
- S39. CountBench: https://teaching-clip-to-count.github.io/ ; https://huggingface.co/datasets/vikhyatk/CountBenchQA
- S40. Hugging Face Hub API, queried 2026-10-02: `https://huggingface.co/api/datasets/<id>`
- S41. CUHK-PEDES / ICFG-PEDES / RSTPReid statistics: https://arxiv.org/pdf/2505.06566 (via search summary)
- S42. VSR: https://github.com/cambridgeltl/visual-spatial-reasoning ; What'sUp: https://aclanthology.org/2023.emnlp-main.568/
- S43. SugarCrepe (NeurIPS 2023 D&B): https://papers.neurips.cc/paper_files/paper/2023/file/63461de0b4cb760fc498e85b18a7fe81-Paper-Datasets_and_Benchmarks.pdf
- S44. Winoground: https://huggingface.co/datasets/facebook/winoground
- S45. NegBench (CVPR 2025): https://arxiv.org/abs/2501.09425 ; https://github.com/m1k2zoo/negbench
- S46. AffectNet license: http://mohammadmahoor.com/wp-content/uploads/2023/03/AffectNet-Agreement-v2-30Mar2023.pdf ; RAF-DB: http://www.whdeng.cn/RAF/model1.html ; ExpW: https://hyper.ai/en/datasets/17382
- S47. EmoSet (ICCV 2023): https://arxiv.org/abs/2307.07961
- S48. AVA / KonIQ-10k / SPAQ (via search summaries): https://www.researchgate.net/publication/261336804_AVA_A_large-scale_database_for_aesthetic_visual_analysis ; SPAQ & KonIQ per https://arxiv.org/pdf/2108.05997
- S49. DISC21: https://ai.meta.com/datasets/disc21-dataset/ ; https://arxiv.org/abs/2106.09672
- S50. DiDeMo: https://arxiv.org/abs/1708.01641
- S51. QVHighlights: https://arxiv.org/abs/2107.09609 ; https://github.com/jayleicn/moment_detr
- S52. Charades-STA / Charades terms (via search summary): https://paperswithcode.com/dataset/charades
- S53. "Moment of Untruth: Dealing with Negative Queries in Video Moment Retrieval." https://arxiv.org/html/2502.08544v2 (title only; not read)
- S54. MIEB (ICCV 2025): https://arxiv.org/abs/2504.10471
