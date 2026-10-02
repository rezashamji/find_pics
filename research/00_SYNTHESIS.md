# find_pics research synthesis (2026-10-02)

The four reports this summarizes have the citations: [01 Apple/Google internals](01_apple_google_internals.md),
[02 competition](02_competition_landscape.md), [03 models](03_models_sota.md),
[04 recall + platform access](04_recall_estimation_and_platform_access.md). Each claim below points to the report that sources it.

## 1. What Apple actually does (01)
- **People album = face vectors + clustering, not a VLM.** A detector finds faces and upper bodies. Two networks turn each
  crop into an embedding: a list of numbers where the same person gives nearby lists. A strict greedy pass groups only
  very close matches ("high precision but many, smaller clusters"). Clothing is used only within one time-and-place
  moment. A face-only pass then merges groups across moments. New photos are matched against several stored example faces per person.
- **When it runs:** clustering runs overnight while charging; a new face is matched to a known person near capture time. All on-device.
- **Why it misses your dad:** by design. Apple avoids wrong merges, drops unclear faces, and only promotes frequent people.
  Accuracy also drops with profile views, age gaps and small faces. Apple holds back candidates it isn't sure of and shows
  them under "Review More Photos". It never tells you how many it held back.
- **Why "bread" fails:** classic search is a fixed list of about 1,300 tags, each with its own score cutoff. "bread" is on the
  list, so a miss means that photo's bread score fell under the cutoff, or the index wasn't finished. The tag is
  decided once at indexing time and the lost information never comes back. That is why the stored thing should be a
  vector compared against any query later, not a word.
- Apple's newer natural-language search (iOS 18.1+, Apple Intelligence devices only) has **no published model description**.

## 2. What already exists (02): be blunt
- Plain natural-language photo search already ships at scale: Apple (on-device), Google Ask Photos (cloud), and
  Samsung S25 (an on-device vision-language model, VLM).
- Open source: Immich v3.2 (Sept 2026) already combines person + CLIP text + date + media type in one query. It does
  not parse sentences, has no cutoff, uses one frame per video, and does not write to Apple Photos.
- osxphotos already reads Apple's people tags and labels and can create albums. Identity timelapse tools exist.
- **Nobody** (5 big platforms, 12 self-hosted managers, 15+ tools, research systems) reports how many items were
  checked plus a statistically valid completeness bound. Google paused Ask Photos in June 2025 to improve "recall".
- **Verdict:** another search app is pointless. The gap is *auditable* retrieval:
  1. a sentence goes in;
  2. a person + appearance + time query is executed exactly;
  3. every sampled frame of a video is considered;
  4. every result is checked by a local VLM;
  5. you get a scan count and a recall lower bound that states what it is relative to;
  6. albums are written into Apple Photos, additive only.

## 3. Method (03 + 04): what find_pics does and why
| Step | Model / method | Why |
|---|---|---|
| Faces | InsightFace SCRFD + ArcFace (buffalo_l) | fast, standard. Non-commercial weights; commercial swap: SCRFD-34G MIT + AuraFace recognizer |
| Person = set of reference faces | max similarity to any reference, not the average | a 10-year weight change can make the average match neither era |
| Image vectors | PE-Core-L14-336 vs SigLIP2-so400m: chosen by measured bread/concept recall | both Apache-2.0 |
| Planner | Qwen3.5 (Apache-2.0), fixed template + injected request + today's date | the "stagnant prompt" idea; exact dates/media handled by code, not the model |
| Judge | Qwen3.5-9B P(yes) on a red-boxed photo; identity via side-by-side reference crop | runs only on the head + an audit sample, not 150k |
| Completeness | elusion certificate: judge the whole head, uniform random tail sample fixed before any label, one-sided Clopper-Pearson bound on tail misses | the only valid family (04); cost about N_tail * ln(1/alpha) / m judge calls to certify at most m misses |

Cost reality (04):
- For a rare concept (bread, 300 in 150k), certifying recall of at least 0.92 takes about 22-32k judge calls. That is
  minutes on an H100 but hours on a Mac. So on a Mac the tight bound is an opt-in "audit mode"; the default shows a looser bound.
- The bound is relative to the VLM judge. A human check of a judge sample is reported separately. Certifying recall
  against human judgment over the deep tail is infeasible for a person to do.

Weight change (03): no modern study measures face recognition across large weight change, so this is measured on our
own data: IMDB people across years, and Reza's Apple-tagged photos across eras.
Body judgments (03): absolute BMI from photos is off by 2.3-4 BMI on average, and VLMs show weight bias. So "looks heavier" is
reported as a judge's opinion on a relative, per-person scale, never as a measurement.

## 4. Platform path (04)
- Apple: osxphotos (read-only DB access) for metadata and pixels. The only write is create album + add, dry-run by default.
  An iOS app can't read Apple's People data at all.
- Android/Google: the Library API has not been able to read a user's library since 2025-03-31. Google Takeout is the only complete export; treat it as a folder.
- Mac speed: SigLIP-class embeddings run at about 5 img/s on an M4, so 150k items take roughly 2-8 h once. Judging takes 0.3-1.5 s per image.
