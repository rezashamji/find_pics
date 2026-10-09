# find_pics: measured results on a public test library (2026-10-02)

All numbers come from runs logged in [JOURNAL.md](../JOURNAL.md). Each states its denominator. Each "looked at" note
means the contact sheet was viewed image by image, not summarized by a script.

## Test library
19,218 items: 15,000 Open Images photos (human-verified labels for 15 concepts), about 4,100 IMDB scene photos
(6 "family members" photographed across 32-39 years each), and 120 Pexels videos.
- To mimic Apple's People album, only the clearest 50% of each person's photos carry a "tag". The job is to find the rest.
- Labels are imperfect. IMDB names one person per photo, and Open Images verified bread only on some images.
  Several "errors" below turned out to be label errors when I looked at them.

Indexing on one A100-40GB GPU per shard: about 23 image units per second (faces + image vector). The library took
113 s on 8 GPUs. 20,457 units (videos contribute several frames each), 22,955 faces, 0 decode errors.

## 1. Why "bread" search fails in Apple-style systems (fast stage only)
Apple-style search gives each photo a tag only when its score clears a fixed cutoff tuned for precision. We simulated
that: the cutoff with the highest recall that still has ≥90% precision on verified bread / not-bread images.

| encoder | bread recall at that cutoff | bread in top 264 | bread in top 1,320 (5×) |
|---|---|---|---|
| SigLIP2-so400m | 0.31 | 0.64 | 0.98 |
| PE-Core-L14-336 | 0.28 | 0.64 | 0.98 |

Denominator: 264 verified bread photos. A precision-tuned cutoff throws away about 70% of the bread. Ranking
everything and letting a judge check the top recovers it.

PE-Core matched or beat SigLIP2 at top-5× on most of the 15 concepts (sandwich 0.97 vs 0.58, guitar 0.97 vs 0.84, cake
0.99 vs 0.91) and became the default.

## 2. Person matching (face vectors)
Reference faces come from the tagged half; the targets are the untagged half.

| person (photo span) | untagged | found at sim ≥ 0.40 | wrong items added |
|---|---|---|---|
| Nicolas Cage (1975-2014) | 235 | 0.94 | 11 |
| Dan Aykroyd | 56 | 0.88 | 5 |
| Kim Basinger | 50 | 0.90 | 4 |
| Pierce Brosnan | 186 | 0.98 | 0 |
| Kevin Bacon | 168 | 0.96 | 1 |
| Drew Barrymore (1982-2014) | 245 | 0.93 | 0 |

These numbers are after the reference-contamination fix. Looked at: before the fix, Drew's 14 "wrong" matches were
Hugh Grant, Lucy Liu and Cameron Diaz. They are co-stars in tagged photos where her own face wasn't detected, so their
faces became "Drew" references. Misses were profile views, tiny faces, a childhood poster, old-age makeup, and a cartoon.

## 3. Full pipeline vs known answers (judge = Qwen3.5-9B on GPU; shipped settings, run 49840354)
| query | true matches | returned | recall vs labels | stated lower bound (95%) | bound held? |
|---|---|---|---|---|---|
| bread | 264 | 459 | 0.811 | 0.705 | yes |
| dog | 1,586 | 1,614 | 0.984 | 0.979 | yes |
| Kevin Bacon (all) | 336 | 329 (1 wrong) | 0.976 | not certified* | (+14 "possible", 1 real) |
| Drew Barrymore (all) | 490 | 469 (0 wrong) | 0.957 | not certified* | (+19 "possible", 11 real) |
| Nicolas Cage 1995-2005 | 168 | 174 | 0.970 | not certified* | 11 "wrong" = all actually Cage (looked at) |

(The earlier run at judge cutoff 0.5 gave bread 0.852 / bound 0.844 and dog 0.987 / bound 0.958; both held.
Across all three full-pipeline runs tonight (bread and dog each time), the stated bound was at or below the truth in
6 of 6 results. That's a small denominator; the 300-run simulation in section 4 is the larger check.)

*The VLM cannot recognize faces. On "is the person on the left the same as in this photo?" it said yes 191 times for
Kevin Bacon, and 4 of those were correct. So identity comes only from face vectors, and person completeness is reported
as the measured test-library rate plus a human review sample, never as a machine-certified bound.

Looked at, bread: in a random 36 of the 308 bread returns without a bread label, about 28 contain bread (burger buns,
sandwiches, toast), 3 are clear errors, and about 5 are pastry borderline cases.

## 4. Simulated certificate validity
On a simulated 150k library (2,000 true matches), the lower bound was at or below the true recall in 290 of 300 runs (96.7%).

## 5. "Looks overweight" judge vs CelebA "Chubby" labels (600 positive / 1,400 negative)
AUC (chance a random labeled-chubby face scores above a random other face): judge 0.896; fast text-image score 0.703.
Rate of judge "yes" on labeled-chubby / other faces at each P(yes) cutoff:

| cutoff | labeled chubby | other |
|---|---|---|
| 0.5 | 239/600 | 22/1,400 |
| 0.3 | 339/600 | 63/1,400 |
| 0.2 | 409/600 | 122/1,400 |
| 0.1 | 486/600 | 249/1,400 |

Looked at: about 20 of the 24 most confident "labeled Chubby, judge says no" faces look lean to me, and most of them
appear to be Black men and women. That suggests the CelebA label itself is noisy and skewed. The judge also confuses
age with weight: its average P(yes) on not-chubby older faces is 0.166, vs 0.043 for young faces.

## 6. Within-person ranking (the transformation-video question)
132 CelebA people with ≥2 "Chubby" and ≥2 not-"Chubby" photos (2,707 photos). The question: within one person, does
the judge rank the chubby-labeled photos higher?
- "Does this person look overweight?": mean per-person AUC 0.642; above 0.5 for 94/132 people.
- "Does this person's face look heavier or fuller than average?": mean 0.660; above 0.5 for 104/132 people.

Looked at: for most people the two sets of photos look the same weight (the label is inconsistent), and the judge
sits near 0.5 on both. Where a visible change exists, the judge separates the eras (one person: 0.78-0.82 vs
0.47-0.56). That is thin evidence. Your own before/after photos are the real test.

## 7. Matching across eras (references from the recent era only)
This simulates the case where Apple tags only recent photos of you. References come from the newest half of each
person's tagged photos. "Early" is the oldest third of their photo years. Face similarity cutoff 0.40.

| person | recent-only: recall / early recall / wrong | + query-time expansion (0.55, 3 rounds) |
|---|---|---|
| Nicolas Cage | 0.91 / 0.88 / 11 | 0.95 / 0.95 / 11 |
| Dan Aykroyd | 0.87 / 0.80 / 5 | 0.91 / 0.90 / 5 |
| Kim Basinger | 0.89 / 0.83 / 4 | 0.92 / 0.86 / 4 |
| Pierce Brosnan | 0.97 / 0.96 / 0 | 0.99 / 0.98 / 0 |
| Kevin Bacon | 0.96 / 0.97 / 1 | 0.97 / 0.98 / 1 |
| Drew Barrymore | 0.91 / 0.92 / 0 | 0.95 / 0.98 / 0 |

Aging is tested here, not large weight change: none of these people had a dramatic one.

## 8. People no model has seen (DigiFace-1M: rendered identities that exist nowhere)
A face is in the training distribution as a category; a specific identity, like you, is not. 300 synthetic people,
72 images each, 8 used as references:
- at similarity ≥ 0.40: recall 18,855/19,110 (98.7%); per-person minimum 0.70, 10th percentile 0.97, median 1.00;
  44 wrong matches per person among 21,438 other-person images;
- at ≥ 0.30: recall 99.8%, but 955 wrong per person. That's why 0.30-0.40 is shown only as "possible".

## 9. Apple-format export, including videos (synthetic export built from public photos)
2,234 items in the exact osxphotos layout: 1,073 JPEG, 743 HEIC, 408 iCloud-preview JPEGs, 10 HEVC `.mov` videos.
0 decode errors. "Find every photo and video of Kevin Bacon":
- photos 317/324;
- **videos 4/4**, none of them tagged, so they were found only by face matching inside sampled frames;
- 0 of 6 other videos returned;
- 1 photo outside the labels, which looks like Bacon in a group shot (label noise).

## 10. Documented weight transformations (the closest public stand-in for your query)
People with published weight losses: Chris Pratt (lost 60 lb, 2013), Jonah Hill (about 40 lb, 2011), Seth Rogen (30 lb,
2009-10). IMDB-WIKI face crops, labeled by era. **The labels were badly contaminated:** of the "heavy-era" faces, 21/73
for Pratt, 9/47 for Hill and 21-22/46 for Rogen were other people (co-stars). I found this by looking at all 166.

**Face matching across the weight change, with references from the lean era only, counting only photos that are
really the person:** Pratt 51/52, Hill 37/38, Rogen 24/24 at similarity ≥ 0.40 (112/114). With the product's
cross-era expansion: 114/114. 0, 13 and 1 wrong matches among 5,997 other faces.

**The judge, heavier era vs lean era, verified photos only (AUC, where 0.5 = chance):** "face heavier/fuller" Pratt 0.857,
Hill 0.812, Rogen 0.869.

**Building the two albums (heavy-era photo in "heavier", lean-era in "fit"), totals over the 3 people:**

| rule | heavy-era → "heavier" | lean-era → "heavier" (leak) | lean-era → "fit" | heavy-era → "fit" (leak) |
|---|---|---|---|---|
| fixed cutoff + raw scores (old code) | 47/114 (Pratt 1/52) | 18/131 | 110/131 | 64/114 |
| rank within the person (shipped now) | 78/114 | 29/131 | 93/131 | 19/114 |

The judge's absolute scale differs by person, so only within-person ranking works. These numbers are by era; a photo
can look heavy in a "lean" year.

## 11. "Find this specific ___" for any kind of thing (3 reference photos; distractors of the same kind)
R-precision (of the top-T results, T = that identity's other photos, the share that are correct) / share found within the top 2T:

| kind | public set | best vector | result |
|---|---|---|---|
| person | DigiFace-1M, IMDB transformations | face model | 98.7% found; 114/114 across weight change |
| picture of a picture | 300 photos × 5 copies (print, framed on wall, screenshot, crop+JPEG, screen photo) | PE-Core | 0.90 / 0.96 |
| specific thing | Stanford Online Products, 1,703 products | PE-Core | 0.69 / 0.77 |
| specific place | Google Landmarks v2 mini, 1,500 landmarks | PE-Core | 0.68 / 0.78 |
| specific dog | DogFaceNet, 788 dogs | DINOv2 ≈ PE-Core | 0.61 / 0.72 |

Cropping the dog with a detector did not help, and a dedicated animal re-ID model (MegaDescriptor-B-224) scored 0.30.
People are solved; individual animals are not. Things and places sit in between. These are fast-stage numbers only;
the judge has not yet been tested on "is this the same object or place?"

## 12. The judge vs human labels (Open Images, labeled photos only; judge = Qwen3.5-9B, yes at P >= 0.7)
| concept | labeled | precision | recall | | concept | labeled | precision | recall |
|---|---|---|---|---|---|---|---|---|
| dog | 1,703 | 0.996 | 0.985 | | cake | 389 | 0.908 | 0.898 |
| horse | 368 | 0.993 | 0.996 | | bicycle | 381 | 0.905 | 0.988 |
| cat | 413 | 0.961 | 0.988 | | swimming pool | 280 | 0.905 | 0.917 |
| pizza | 109 | 0.948 | 0.912 | | baked goods | 1,113 | 0.862 | 0.786 |
| coffee cup | 209 | 0.933 | 0.897 | | sandwich | 100 | 0.805 | 0.985 |
| guitar | 243 | 0.925 | 0.827 | | bread | 402 | 0.750 | 0.875 |
| sunglasses | 236 | 0.924 | 0.903 | | christmas tree | 73 | 0.583 | 0.700 |
| wine glass | 149 | 0.923 | 0.933 | | | | | |

Neither side is ground truth. Full-resolution look at disagreements: bread judge-yes/label-no 4 photos: judge right 2
(bruschetta, naan), wrong 1 (crepe), borderline 1 (scones); baked goods judge-no/label-yes 4: the LABEL was wrong on 3
(two omelettes, mushrooms). The judge also calls plain conifers "christmas trees" (3 of 4 in a sample). A larger model
(Qwen3.5-27B) was asked about every disagreement, but scoring the 9B against "27B and 9B agree" is circular (they share
blind spots), so disputed photos were checked BY EYE (39 at 900 px): models right 20/39, labels right 11/39, unclear 8/39
(baked goods: models 9/12, labels 0/12; christmas tree: 6 vs 5 of 15; bread: 5 vs 6 of 12, where every label win was the
model counting pastry or pizza as bread). Christmas tree against eye-checked truth: precision 15/21 = 0.714, recall 15/15
(vs raw labels 0.583 / 0.700). The judge's real failure is false alarms on look-alikes (plain conifers), not misses.
Rewriting the judge's question with an LLM-written definition was tested and rejected: precision up, recall collapsed
(christmas tree 0.70 -> 0.30, coffee cup 0.90 -> 0.39, bread 0.88 -> 0.44). A softer version (plain question + "answer no if
it is only a look-alike such as <LLM list>") also trades recall for precision (cake recall 0.90 -> 0.71; bicycle the only
clear win, 0.905/0.988 -> 0.972/0.976) because the LLM's look-alike lists name real members (bagels as "not bread").
Default stays the plain question; exclusions come from the person ("no pastries"), not from the planner.

## 13. Streaming: how fast the album approaches "the judge looked at every photo" (CPU replay of stored answers, 20 concepts)
Round 1 = the fast answer; each later round doubles the judged head and draws a larger random check of the rest.
The 5% error budget is split across rounds in advance (union bound), so the stated bound holds whenever you stop.
- Stated lower bound above the true recall: **0 of 115 rounds**.
- Recall of the oracle set once 10% of the library is judged: median 0.94 (9 concepts reach a round by then);
  25%: median 0.94, min 0.71 (18); 50%: median 0.96, min 0.59 (sunglasses).
- Small objects are slow: sunglasses 0.59 after 26% judged, christmas tree 0.61 in round 1.
- Ranking by whole photo + 2x2 tile vectors (same judge answers, same 2,000 judge calls in round 1): round-1 recall
  median 0.905 -> 0.913; christmas tree 0.612 -> 0.741, sunglasses 0.595 -> 0.664, person wearing sunglasses
  0.488 -> 0.603, coffee cup 0.707 -> 0.754; worse: guitar 0.804 -> 0.784, wine glass 0.800 -> 0.773.
  **Caveat from looking at the photos** (top 1,600, prompt "a photo of a ..."): of the 9 christmas-tree photos tiles
  newly brought in, 2 are real trees, 5 are judge errors (plain firs, a palm with lights, a ficus, a tinsel float), 2
  unclear; sunglasses 4 sampled gains: 2 real, 1 judge error (headlamp), 1 unclear; guitar: tiles pushed out 2 real
  tiny guitars. Measured against the judge, part of the tile gain is the judge's own errors. Against HUMAN-labeled
  positives (independent of the judge), top-2,000 recall whole vs tiles: bicycle 0.980/0.996, christmas tree
  0.950/1.000 (1 photo), sunglasses 0.903/0.873, bread 1.000/0.992, rest equal. Decision: tiles stay off.
- Real run (chat, one GPU, before the 1.9x loader speed-up): bread 678 (>=80%) at 3.6 min -> 731 (100%, all 19,218
  judged) at 16.5 min. Judge throughput now 27.8 photos/s (was 14.6).

## 14. Phone-size judges vs the 9B, on ALL 19,218 photos (4 concepts; disagreements are not all errors of the small model)
| judge | bread (746 9B-yes): missed / extra | dog (1,646): missed / extra |
|---|---|---|
| Qwen3.5-4B | 104 / 95 | 16 / 22 |
| Qwen3.5-2B | 54 / 259 | 18 / 43 |
| Qwen3.5-9B, 4-bit weights (Intel AutoRound, vision kept 16-bit) | 37 / 33 | 3 / 13 |

Christmas tree (85 9B-yes): 4B 30 / 8, 4-bit 9B 6 / 6. Sunglasses (1,043): 4B 168 / 81, 4-bit 9B 34 / 49.
9B with 8-bit (FP8) weights, missed / extra: bread 6 / 71, christmas tree 5 / 10, sunglasses 20 / 52, dog 2 / 13.

Lowering the small judges' cut to 0.3 looked fine on a balanced sample but adds hundreds of extra yeses at real
prevalence (2B bread +767). Full-resolution look: the 2B's extra "bread" yeses were 0/4 bread (incl. a Lego set);
the 4B's misses vs the 9B were right twice (pastry; clams with breadcrumbs), wrong once (flatbread wrap).

## 15. Hard real-user queries (DISBench, 122 queries, each user's ~1,900 photos)
| planner | precision | recall | F1 | returned nothing | >=1 correct |
|---|---|---|---|---|---|
| one-step | 0.039 | 0.116 | 0.033 | 82 | 21 |
| old multi-step (falls back to the whole library when the moment is not found) | 0.076 | 0.424 | 0.096 | 32 | 63 |
| merged conversational planner (moment not found -> says so, returns nothing) | 0.102 | 0.351 | 0.112 | 37 | 54 |
| same, after 10-03 fixes (no-condition albums, unanswerable filters dropped, offsets; 0 errors) | 0.095 | 0.362 | 0.111 | 40 | 55 |
| same, after the 10-03 night fixes (date grounding, place filters; 0 errors) | 0.104 | 0.364 | 0.119 | 39 | 55 |

Looking at returned photos: the worst precision comes from queries that need cross-photo sameness ("the building that
appears both in real life and as a drawing" -> the plan's question is true of most photos, 854 returned); 3-hop queries
(animal at place X -> day with a group of it -> cows that day) do not fit the 2-step plan shape.

## 16. Same individual dog, judge shown both photos side by side (600 DogFaceNet pairs; different-dog pairs are the most similar-looking dog)
Judge AUC 0.874 vs image-vector AUC 0.566; but it says "same" on 0.563 of different-dog pairs (cut too loose).

## 17. "Find my dog" from 3 reference photos (40 dogs, DogFaceNet)
| library | ranking | top-3 precision | top-3 recall | top-5 precision | top-5 recall |
|---|---|---|---|---|---|
| 10,943 shelter-dog photos of >1,000 dogs (worst case: every photo a dog) | image vector | 87/120 = 0.725 | 87/136 | 103/200 | 103/136 |
| same | side-by-side judge, mean of 3 refs | 46/120 = 0.383 | 46/136 | 67/200 | 67/136 |
| 19,218 everyday photos (~1,600 random dogs) + the dog's own photos | image vector | 114/120 = 0.950 | 114/147 | 132/200 | 132/147 |

The mixed-library number may be inflated: the dog's photos come from a different dataset (style) than the background.
The judge is a safe veto (keeping P >= 0.2 kept 140/147 true photos) but a poor filter (at 0.5 it found 80/147).
Correction: an earlier pair test (judge AUC 0.874 vs vector 0.566) picked each "different dog" as the most
vector-similar one, which is rigged against the vector; the full search above is the fair comparison.

## 18. Ranking one person's photos from heaviest to leanest (Reza's demo), after the product's face filter
IMDB-WIKI face crops of three actors with documented weight changes; eras from published reports. The dataset's name
labels are noisy (of Seth Rogen's 8 lowest-ranked "heavy-era" photos, 8/8 were slim co-stars), so photos are first
filtered the way the product does (face similarity >= 0.40 to the person's references). Within-person AUC, heavy era
vs lean era:
| person | heavy / lean photos kept | single photo, P(heavier) - P(fit) | side-by-side pairwise |
|---|---|---|---|
| Chris Pratt | 52 / 47 | 0.921 | 0.705 |
| Jonah Hill | 38 / 36 | 0.813 | 0.792 |
| Seth Rogen | 24 / 42 | 0.863 | 0.749 |
The product keeps single-photo scoring ranked within the person; side-by-side comparison was worse.

## 19. "This specific thing / place" (3 reference photos; R-precision; 1,703 products, 1,500 landmarks)
| method | products | landmarks |
|---|---|---|
| PE-Core image vector, best match to any reference | 0.695 | 0.680 |
| PE-Core, MEAN over the 3 references (product default now) | **0.731** | **0.693** |
| DINOv2 | 0.342 | 0.567 |
| PE-Core + DINOv2 | 0.486-0.551 | 0.659-0.676 |
| judge re-rank of the top 20 (side by side "same specific object/place?"), 300 identities | 0.53-0.55 | 0.61-0.67 |
| judge only, top 20 | 0.423 | 0.465 |
| SIFT keypoints + RANSAC inliers, top 30, 300 identities | 0.415 (combined: <= 0.716) | 0.317 (combined: <= 0.688) |
Image vectors averaged over the references beat every add-on tried; the next lever is a stronger image model.

## 20. Image-text encoder size (PE-Core family): headroom and phone cost
| model | products R-prec | landmarks R-prec | concept first stage, median of 20 (top-2,000 recall of judge-yes) | sunglasses | christmas tree |
|---|---|---|---|---|---|
| T-16-384 | 0.565 | 0.431 | 0.896 | 0.297 | 0.647 |
| S-16-384 | 0.609 | 0.527 | 0.922 | 0.386 | 0.753 |
| B-16 | 0.654 | 0.608 | 0.918 | 0.448 | 0.788 |
| L-14-336 (current) | 0.731 | 0.693 | 0.927 | 0.457 | 0.765 |
| bigG-14-448 | 0.753 | 0.727 | 0.929 | 0.403 | 0.753 |
The first stage barely depends on size (the judge does the rest); "this specific thing" does. Phone candidate: B-16.

## 21. Videos: one frame or several? (522 Pexels HD videos, 6,036 sampled frames, 24 queries from random frames)
Truth = videos where the judge says yes on ANY frame (312 across the 24 queries). Judging only each video's frame with
the best cheap score finds 211/312 = 0.676; judging its best 3 frames finds 269/312 = 0.862. BUT a full-resolution look at
8 videos newly found by the extra frames: 5 judge false positives, 1 inconsistency on near-identical frames, 2 real only
under a loose reading -> the gain is mostly the judge getting more chances to cross 0.7 (and "any frame yes" inflates the
truth the same way). The product stays at 1 frame per video; the gap needs human-labelled truth to measure. Indexing speed: 522 videos in 419 s on one GPU (14.4 frames/s).

## 22. The whole phone stack on a real first round (19,218 test photos; truth = the 16-bit 9B judge on every photo)
| concept (truth) | cluster stack (PE-Core-L + 16-bit 9B): recall | phone stack (PE-Core-B-16 + 4-bit 9B): recall / precision / judge calls |
|---|---|---|
| bread (746) | 0.893 | 0.886 / 0.962 / 4,400 |
| christmas tree (85) | 0.612 | 0.541 / 0.939 / 1,600 |
| sunglasses (1,043) | 0.588 | 0.662 / 0.953 / 6,800 |
| dog (1,646) | 0.985 | 0.978 / 0.998 / 3,200 |
The phone's stated bound is relative to its own (4-bit) judge; against the 16-bit judge it overclaimed once (bread
0.902 stated vs 0.886). Not yet measured: speed and memory on an actual phone. Core ML sizes (fp16): PE-Core-B-16 image
tower 186 MB (L: 633 MB); text tower 709 MB for both (shared text model); with 8-bit weights: 94 MB + 355 MB.
Outputs of the converted models are not yet verified on a device (Linux cannot run Core ML).

## 23. Moments inside an event (10-03): before/after windows, DISBench full 122
| planner | F1 | recall | exact | nothing returned |
|---|---|---|---|---|
| v6 (10-02) | 0.119 | 0.364 | 1/122 | 39 |
| 10-03 fuzz-hardened, no new windows (v7_base) | 0.100 | 0.340 | 1/122 | 40 |
| + windows before/after/since/until/minutes_* (v7_windows) | 0.121 | 0.376 | 2/122 | 38 |
- Judge noise is ~0: 28/122 queries with identical plans changed F1 by 0.001 on average. Differences are plan changes;
  any prompt edit re-rolls most plans (+-0.02 F1 overall).
- The 9 queries whose plan uses a new window: mean F1 0.007 -> 0.243 (q20 "statues during the Paris trip before the
  first Van Gogh photo" 0.06 -> 1.00 exact; q96 "30 min to 1 h before the torch performer" 0 -> 0.57).
- Still 0: "after A but before B" (needs two anchors, q45); anchors never found (q64, q107).

## 24. "Which face is you" without Apple People names (10-03)
- people.face_groups: greedy groups on a 20k face sample (cosine >= 0.55, faces >= 40 px and detector score >= 0.7).
- Test library (19,218 items): 12 groups in 1 s; by eye 96/96 sampled crops match their group's person (Cage, Brosnan,
  Barrymore, Bacon, Aniston, ...). Without the detector-score filter one "group" was dogs/flowers/backs of heads (det
  median 0.55 vs >= 0.84 for real groups). File-name identity labels gave purity 0.5-1.0, but the eye check shows the
  labels, not the groups, are wrong (IMDB-WIKI noise).
- Apple-like library (2,234 items): top group = Kevin Bacon 98% of 242 items; sheet 48/48 crops correct by eye.
  "Find every photo and video of me" with --me = the named group: 322 items = 317/324 Bacon photos + 4/4 videos + 1
  (P00343, which shows Bacon: label error). Identical to the Apple-tag path.

## 25. "Me heavier vs me fit" on the owner's own photos, every item labeled by the owner (10-04)
Sample: 598 photos/videos the owner chose from 4 months; 393 face-matched to him. He labeled all 393 by eye
(267 heavier-era, 124 fit-era, 1 not him, 1 neutral; his labels match the year in 389/391).
| rule for two opposite looks of one person | 'heavier' album | 'fit' album |
|---|---|---|
| rank margin 0.3 (demo6) | 139, all right; 139/267 found | 127, 12 heavier-era (all suited / distant / group shots) |
| two-group split on log-odds difference, per-event median, >= 90% sure (demo7, bc0c41b) | 277: all 267 found + 8 fit-era + 1 not him + 1 neutral | 116, 0 wrong; 116/124 found |
The 8 fit-era photos in 'heavier' (seen at full size): 2 screenshots with a tiny headshot, 4 face-only selfies from
one evening, a plane selfie, a helmet selfie: build not visible, a look-based rule cannot place them. Cause of the old
errors: P(fit) saturates (2026 median 0.98, 2023 median 0.89) and the rank margin ignores unequal era sizes.
Public eras (Pratt / Hill / Rogen face crops, scores from section 13): the split finds no clear two groups for Pratt and
Rogen and falls back to the rank margin; Hill on par. Kevin Bacon regress: runs, albums plausible by eye (no truth).
Caveat: the owner's eras are 3 years apart with nothing between; a gradual change is harder and untested on his photos.

## 26. Everyday searches on REAL people's libraries (DISBench: 8 Flickr users x 12 queries, ~1,900 photos each) (10-05)
- Speed (one A100): fast answer ~17 s, exhaustive (judge on every in-scope photo) ~37 s per library. Fast found
  0.889 (bicycle) to 1.000 of the exhaustive set; most >= 0.99. 0 errors in 96 searches.
- Precision BY EYE (8 random exhaustive results per query, 800 px): **52/80 right**, 19 wrong, 9 unsure (dog 8/8,
  cat 6, flowers 6, boat 6, beach 5, food 5, church 5, car 4, sunset 4, bicycle 3). Much worse than Open Images
  labels (section 12): real libraries hold murals, toys, models, paintings and scenery that only shares a setting.
- Fix 1 (adopted): "photos with X" asks "Is there a real X anywhere in this photo (not a drawing, painting, statue,
  toy, model or picture of one)?". Re-judging the audited results: 8/8 wrong + 2/2 unsure dropped, 0/21 right dropped.
  Open Images recall vs labels falls 1-6 points (dog 0.985 -> 0.969, horse 0.996 -> 0.939), but looked at: 10 of 12
  newly rejected "positives" are depictions (nativity horse, rocking horse, spring riders, painting, stuffed dogs,
  statues, a bronze trophy, a coyote); 1 real dog in a graphic frame lost, 1 arguable.
- Not adopted: "Is X the main subject of this photo?" for "X photos": dropped the wrong beach/sunset/church photos
  (7/8) but also 4/27 right ones (half the flower photos). "Selfies" bug found and fixed (was: every photo).

## 27. Everyday searches on 16 MORE real libraries (DISBench users not in section 26; 30,273 photos) (10-06)
- 192 searches per model, 0 errors. Fast answer vs exhaustive (9B): 0.94-1.00 of the exhaustive set for every query
  except selfies 75/169 -> fixed by a default look for selfie albums: 166/167.
- Precision BY EYE, 12 photos per query stratified by user (11-12 users each), 1000 px: **91/132 right**, 21 wrong,
  20 unsure (bicycle 11, flowers 11, cat 10, church 10, boat 10, car 10, beach 9, dog 7, food 5, sunset 5, selfies 3).
  The 8 queries shared with section 26: 73/96 here vs 63/80 there (both 76%).
- Errors: look-alike or toy animals (tiger as cat; teddy bear, deer as dog); "food photos" / selfies returning photos
  where food or a person is merely present; golden light as sunset; motorcycle as bicycle.

## 28. The phone's model sizes on real libraries (8 libraries of section 26; truth = blind eye labels, 1000 px)
Phone weights reproduced exactly (MLX affine 4-bit, group 64, scripts/sim_mlx_quant.py). Overlap with the 16-bit 9B is
agreement, not truth; the eye audit of disagreements (125 photos) decides:
| phone model | 9B-only photos (sampled real) | phone-only photos (sampled real) | est. real photos missed |
|---|---|---|---|
| 9B 4-bit (6.0 GB) | 377 (10/41 real) | 42 (4/26 real) | ~90 |
| 4B 4-bit (3.1 GB) | 907 (18/63 real) | 54 (4/34 real) | ~260 |
| 9B 3-bit (plain rounding) | broken: says yes to nearly everything (flowers 5,808 vs 313) | | |
iPhone app memory budget is ~6.1 GB even with the increased-memory entitlement (12 GB iPhone Air report), so the 9B
4-bit likely does not fit beside the other models; measure on the phone.

## 29. A judge without probabilities (Apple's Foundation Models give hard answers only) on Reza's heavier-vs-fit demo
(his labels: 267 heavier, 124 fit; heavier album / fit album contents)
| judge | heavier | fit |
|---|---|---|
| 9B, P(yes) (product) | 267 H + 8 F | 116 F + 0 H |
| 9B, yes/no only | 214 H | 124 F + 53 H |
| 9B, 1-10 rating | 229 H + 3 F | 121 F + 32 H |
| 9B, 1-10 rating, one combined "A rather than B?" question | 75 H | 66 F + 10 H |
| 4B 4-bit, P(yes) | 137 H | 117 F + 19 H |
| 4B 4-bit, yes/no only | 63 H | empty |
| 4B 4-bit, 1-10 rating | 236 H + 9 F | 114 F + 28 H |
A hard yes/no judge breaks the comparison; a rating mostly works but leaks heavier photos into "fit".

## 30. "This specific thing / place / dog": neighbour smoothing (10-06)
Each library vector and each reference averaged with its 2 nearest library vectors before the mean-of-references match
(adopted in converse._dba and the phone's SubjectSearch). Query expansion was tried first and hurt (products 0.731 ->
0.695).
| set (3 references) | library | before | after |
|---|---|---|---|
| products (R-precision) | 13,145 + 20,000 everyday | 0.731 | 0.747 |
| landmarks (R-precision) | 9,758 + 20,000 everyday | 0.680 | 0.750 |
| dogs, 40 (top-3 / top-5 / R-precision) | 10,943 dogs + 19,218 everyday | 87/120, 97/200, 0.679 | 90/120, 108/200, 0.752 |

## 31. Request -> plan, 30 scripted conversations (10-06)
Distilled 4B planner (LoRA on 4,426 27B-teacher examples incl. 2,400 chained edits) 26/30 -> **30/30** after code
rules for its failures (copied-example alternative dropped, partial undo kept, "the day of my X" as a moment, scene look
kept when an identity question is removed). Same adapter on the PHONE's 4-bit weights: 30/30. 9B: 29/30 (photo/video
twin-merge fix pending re-test).
Selfies: question variants asking WHO took it raised precision but lost 5-6/18 real selfies (not adopted); instead
the front-camera EXIF tag (Reza's sample: 68 front / 382 back / 60 none of 510) scopes selfie searches.

## 32. Which photo judge for the phone? (10-06; truth = eye labels, 261 right / 141 wrong photos, plus a blind audit)
| judge (phone-size where marked) | right / wrong kept at P>=0.7 | best high-recall point | Reza's demo (H / F albums) |
|---|---|---|---|
| Qwen3.5-9B (server) | 247 / 77 | 0.8: 222 / 25 | 267 H + 8 F / 116 F |
| Qwen3.5-4B, 4-bit (current phone judge) | 208 / 33 | 0.5: 232 / 65 | 137 H / 117 F + 19 H |
| Qwen3.5-4B distilled from the 9B, 4-bit | 219 / 41 | same curve as the base at equal strictness: not adopted | |
| Qwen3-VL-4B, 4-bit | 242 / 59 | 0.95: 229 / 39 | 265 H + 6 F / 111 F + 2 H |
| Gemma 4 E4B (16-bit) | 244 / 74 | 0.9: 239 / 55 | |
Blind audit on 16 real libraries (Qwen3-VL vs the 9B's sets): it rejects far fewer real photos than the current 4B
(8 real of 49 sampled 9B-only photos vs 19 of 71) and its extra photos on objects are mostly real, but on "X photos"
subjects it is too broad (food 0/5, flowers 2/7, cat 0/5 extras real; ~1,000 extra photos). Wording fix under test.
Two-model vote (Qwen3-VL-4B + Qwen3.5-4B, both phone 4-bit; mean P(yes) >= 0.7, second model asked only when the first
gives P >= 0.4): eye labels 231 / 261 right with 34 / 141 wrong. On the 16 libraries, against every eye label that
falls in them (142 right / 82 wrong): 9B 111 / 57, current phone 4B 101 / 20, Qwen3-VL alone 127 / 33, vote 112 / 15.
Chosen as the phone's photo judge (needs the increased-memory entitlement: both models resident, ~5.5 GB).

## 33. A judge that fits WITHOUT the increased-memory entitlement: Qwen3-VL-2B, 4-bit (10-06)
Without the entitlement the iPhone 18 Pro leaves ~2.4 GB after the embedder + face models; Qwen3-VL-4B 4-bit (3.09 GB
of weights) does not fit, Qwen3-VL-2B 4-bit (1.78 GB, mlx-community) does. Same eye labels and code as section 32
(phone 4-bit simulated on the cluster; eval/judge_sweep_q3vl2b.txt).
| judge (phone 4-bit) | P>=0.7 right / wrong | P>=0.95 | Reza's demo (H / F albums), same phone planner |
|---|---|---|---|
| Qwen3-VL-4B | 242 / 59 | 229 / 39 | 265 H + 6 F / 111 F + 2 H |
| Qwen3-VL-2B | 229 / 76 | 206 / 61 | 256 H + 6 F / 111 F + 11 H |
| Qwen3.5-4B (pre-vote phone judge) | 208 / 33 | 150 / 1 | 137 H / 117 F + 19 H (base planner) |
The 2B never gets below 44/141 wrong (t0.995: 171 right). Its extra wrong photos are scene questions (sunset 13/15
wrong kept vs the 4B's 3, beach 17/19 vs 7): by eye, sepia beach posts as "sunset", drinks as "food", leafy
branches as "flowers". It also misses small background objects (cars, a dinghy). Usable as a degraded fallback for
person looks; not good enough for scene searches.

## 34. RECALL: what share of the real matches does the phone's photo judge keep? (10-07; truth = blind eye labels)
Setup: Qwen3-VL-4B at the phone's 4-bit weights (models/qwen3vl_4b_mlx4sim), the planner's questions (plans of
everyday16_q3vl), keep at P(yes) >= 0.7 (the app's cut), judged on EVERY photo (= exhaustive mode) of 4 DISBench
libraries (section 26's, the 4 with the most earlier eye labels; 7,886 photos) x 6 queries = 47,316 judge calls
(eval/eval_recall.py; 8 Slurm shards). Rejected photos were split into strata by where misses should be:
A "doubt" = P 0.05-0.7 or another judge (9B / 4-bit 9B / 4-bit 4B exhaustive runs) kept it; B = top 5% image-vector
similarity; C = the rest (~7,000 per query). Uniform random draws inside each stratum + kept photos mixed into the same
sheets, labeled blind at native pixels (DISBench ~500 px = the judge's input; small regions cropped and enlarged):
1,950 eye labels in 2 rounds (eval/recall_audit/labels.txt; round 2 = more of C after round 1 showed C sets the width).
Kept photos with an earlier eye label count exactly; the rest of the kept set is estimated from a 40-photo sample.
Recall = found / (found + estimated missed); "unsure" counted as no match (sensitivity below).
| query | kept | kept, real (est.) | rejected labeled (A/B/C) | real misses seen (A/B/C) | recall | 95% bootstrap | 95% Bayesian |
|---|---|---|---|---|---|---|---|
| photos with a dog | 1,305 | 1,305 | 21 / 40 / 200 (+8 earlier) | 5 / 0 / 1 (+3) | **0.97** | 0.93-0.99 | 0.89-0.99 |
| photos with a car | 461 | 446 | 69 / 40 / 200 (+10) | 27 / 1 / 1 (+2) | **0.79** | 0.69-0.88 | 0.64-0.85 |
| photos with a bicycle | 52 | 49 | 23 / 40 / 200 (+7) | 1 / 0 / 0 (+2) | **0.94** | 0.94-0.94 | 0.34-0.93 |
| beach photos | 420 | 208 | 48 / 40 / 200 (+8) | 0 / 0 / 0 | **1.00** | 1.00-1.00 | 0.70-1.00 |
| sunset photos | 254 | 117 | 44 / 40 / 200 (+10) | 1 / 0 / 0 (+1) | **0.98** | 0.95-0.99 | 0.54-0.98 |
| food photos | 205 | 108 | 48 / 40 / 200 (+4) | 0 / 0 / 0 | **1.00** | 1.00-1.00 | 0.54-0.99 |
| all 24 searches | 2,697 | 2,233 | | est. 162 missed | **0.93** | 0.89-0.96 | 0.82-0.93 |
- Intervals. Bootstrap resamples inside each stratum; it treats "0 real in 200" as exactly 0, so it is optimistic for
  the big C stratum. Bayesian (Jeffreys prior per stratum) still allows misses there: it is the honest one for rare
  subjects. Conservative Clopper-Pearson with Bonferroni over strata: dog >= 0.82, car >= 0.50, bicycle >= 0.15,
  beach >= 0.39, sunset >= 0.25, food >= 0.24, pooled >= 0.47 (uninformative except for dog/car).
- Why bicycle/food/sunset intervals stay wide: 0 misses in 200 of ~7,400 photos still allows ~0.5% = ~35 misses,
  against only 49-117 real matches. Bounding bicycle recall >= 0.8 would need ~2,000 more zero-miss bulk labels;
  a cheaper next step is a second strong judge over the bulk to carve out a smaller stratum to label.
- "Unsure" counted as a match (both sides): pooled 0.86 (Bayesian 0.76-0.88); car 0.62 (vehicle specks that cannot
  be identified), food 0.79 (people eating with plates in view), beach 0.96, sunset 0.95, dog 0.97, bicycle 0.90.
- The pooled number is 58% dogs (1,305 of 2,233 found matches come from one dog owner's library).
- What gets missed (seen): SMALL OR PARTIAL OBJECTS in busy scenes. Cars: parked cars behind people or monuments,
  tiny cars in harbour villages and aerial city views, a car roof at the frame edge, a pickup towing a statue (car
  doubt stratum: 27 of 69 sampled were real, P 0.02-0.68). Dogs: a small dog on a lead among legs at a flea market,
  a dog's body cut off at the frame edge, a black dog behind two women, a tiny dog on a beach (P=0.000). Bicycles: a
  bicycle behind a bush, behind a railing, lying in grass. Sunsets: afterglow behind a terrace / balcony view.
  No real beach or food photo was found among 288 + 292 labeled rejected photos.
- By-product, precision of the kept set (same labels): dog 1,305/1,305 est., car 446/461, bicycle 49/52, but beach
  208/420, sunset 117/254, food 108/205 (unsure as no match; as match 293, 208, 154). Scene/"X photos" queries are
  where this judge is loose: rocky or lake shores and sand close-ups as beach, daytime backlight / dusk / night as
  sunset, storefronts, drinks and people at dinner as food (as in sections 32-33).
- App line (honest for these 6 kinds, 4 Flickr libraries, ~500 px photos; a real phone library is the final exam):
  "In tests on real photo libraries, this search found about 9 in 10 of the matching photos. Small things in the
  background, like a parked car, are missed more often (about 8 in 10)."

## 35. Fixing loose scene searches ("X photos") without giving back recall (10-07)
Problem (section 34): kept photos that are real, beach 208/420, sunset 117/254, food 108/205. Same judge (Qwen3-VL-4B,
phone 4-bit). Scored two ways: (1) the 4 recall libraries of section 34 (all 7,886 photos re-judged with each new
question; precision and recall from the same 1,950 blind labels, stratified, unsure = no match); (2) every earlier
eye label from OTHER libraries for all six "X photos" queries, so a rule that fixes food but breaks flowers shows up
(these are biased samples, mostly judge disagreements: compare rules within a row, not with section 34).
New judge calls: 71,819 (eval/eval_scene_fix.py, 8 Slurm shards). Re-judged planner question = section 32 scores
(153/153 same keep decision).
| rule | beach (1): kept, precision, recall | sunset (1) | food (1) | (2) right kept / wrong kept, all six queries* |
|---|---|---|---|---|
| planner question, P >= 0.7 (now) | 420, 0.49, 1.000 | 254, 0.46, 0.979 | 205, 0.53, 1.000 | 91/126 right, 48/117 wrong |
| (a) P >= 0.9 | 398, 0.52, 0.995 | 242, 0.48, 0.979 | 189, 0.57, 1.000 | |
| (a) P >= 0.95 | 381, 0.54, 0.995 | 230, 0.51, 0.979 | 184, 0.59, 1.000 | 88/126, 33/117 |
| **(a) P >= 0.99** | **348, 0.59, 0.995** | **208, 0.56, 0.971** | **165, 0.66, 1.000** | **82/126, 19/117** |
| (b) strict wording >= 0.7 | 279, 0.74, 0.990 | 552, 0.21, 0.992 | 251, 0.43, 1.000 | (per-kind only) |
| (c) + "Would most people describe this as a photo of X?" >= 0.5 | 322, 0.64, 0.995 | 197, 0.59, 0.971 | 149, 0.73, 1.000 | 64/126, 11/117 |
| (c) + "Is X what this photo is mainly about?" >= 0.5 | 315, 0.66, 0.995 | 240, 0.49, 0.979 | 147, 0.73, 0.991 | 71/126, 22/117 |
*(2) per query at P >= 0.99 vs 0.7: food right 11/11 kept, wrong 19 -> 9 of 35; beach right 16 -> 15 of 28, wrong 5 -> 1
of 25; sunset right 4 -> 4 of 18, wrong 7 -> 2 of 22; flowers right 22 -> 19 of 23, wrong 5 -> 2 of 11; cat right
11 -> 10 of 11, wrong 8 -> 4 of 16; church right 27 -> 23 of 28, wrong 4 -> 1 of 8.
With the "describe" question: flowers right 22 -> 10 of 23, church 27 -> 16 of 28, cat 11 -> 8 of 11.
- (b) Stricter per-kind wording does not generalise. "Food (a dish, meal or snack), not drinks or a storefront" keeps
  MORE wrong photos (251 vs 205). "The sun setting or just set, not daytime backlight" says yes to 552 photos (precision 0.21).
  "A sandy beach by the sea" is the best beach rule (0.74), but by eye the real beaches it drops are black-sand and
  pebble beaches (Santorini loungers on black sand, Red Beach, waves on a rocky shore): it encodes "sandy", which is
  wrong for real users.
- (c) A second question helps food, beach and sunset but drops real flower, church and cat photos in other libraries.
  By eye: flower beds, a hillside of yellow broom, fireweed at temple columns, tulips on a buffet, a cherry tree street,
  a band playing in a church, a church fair on the lawn, a cat on a cottage path. The planner cannot tell in general
  which subjects need the subject to be "the main thing". The same pattern held for the 9B and earlier wordings.
- (a) P >= 0.99 is the only generic rule that improves all six queries in (2) and loses at most one real photo per
  query in (1). Real photos it drops, all viewed: a sun behind the Statue of Liberty (sunset, P 0.97), people eating on
  sand under palms (beach, 0.98), a pebble shore (0.88), a wedding bouquet (flowers, 0.85), wreaths at a memorial (0.96),
  a cherry tree street (0.94), stained glass in a church (0.97), a church fair (0.88), a singer in a church (0.98),
  a field with a far spire (0.99), and a small cat on a cottage path (0.99). All are cases where the subject is
  present but small or not the point.
- RECOMMENDATION (not applied; coordinator decides): for subject questions of the form "Is this a photo of X?", which
  the planner writes for "X photos", keep at P >= 0.99 instead of 0.7. Object questions ("Is there a real X anywhere
  ...?") and person looks stay as they are. This keys on the question template, not on a list of words. Precision
  rises from 0.46-0.53 to 0.56-0.66, and recall changes by 0.000 to -0.008 on the 4 libraries. Optional per-search
  "strict" toggle: add the "describe as" veto (food 0.73, beach 0.64, sunset 0.59). Not a default: it costs flowers
  and church.
- Still loose even at 0.99: about 4 in 10 kept beach and sunset photos are not real matches. The judge's own idea of
  "beach" (shores) and "sunset" (low or warm light) is wider than people's. A bigger fix needs a stronger judge or a
  user-facing "more like this / not this" step.

## 36. Phone face model AuraFace-v1 + flip: recalibrated face cuts (10-07; public data, people.py exactly)
Reza's decision (10-07): ship AuraFace-v1 (fal, Apache-2.0) with flip averaging. buffalo_l (non-commercial) stays a
server dev option. Every face cut was set for buffalo_l, so each one was re-set with the criterion that set it
(eval/face_calibrate.py; results in eval/results_face_calibrate.json). Embeddings come from eval/face_free_deepdive.md:
same SCRFD detector and 5-point alignment, CelebA test split (18,295 faces, 755 people) and DigiFace (21,510 faces,
300 people). The whole dataset is one library; per person, 3 or 8 reference photos; the targets are their other photos.

Shipped cuts (FaceProfile.swift == face_profiles.py):

| Cut | buffalo_l | AuraFace+flip | Criterion |
|---|---|---|---|
| grouping + "group already named" | 0.55 | **0.62** | groups as pure as buffalo_l's: 0 faces of another person in the 12 groups, on BOTH sets (buffalo_l 0.55: 0/361 CelebA, 0/805 DigiFace; AuraFace 0.62: 0/355, 0/701; at 0.60: 0/360, 12/729; at 0.58: 0/360, 2/747; at 0.55: 1/361, 47/799) |
| expansion of the references | 0.55 | **0.62** | the lowest cut whose expansion does not pull in other people (below) |
| person accept | 0.40 | **0.53** | no more wrong items than buffalo_l at its shipped cuts (CelebA: 62 with 3 refs, 180 with 8), one cut for both |
| "closer to someone else" | 0.40 | **0.53** | tied to person accept, as before; any value from 0.46 to 0.54 gives identical results |
| possible list (server) / pickRefFaces floor | 0.30 | **0.42** | same count of different-person pairs above the cut on CelebA (buffalo_l 5,132 pairs at 0.30) |
| reference consensus floor (server) | 0.20 | **0.30** | same, 304,744 pairs at 0.20 |

Expansion cut (CelebA sweep; grouping at 0.62; accept = the smallest cut with wrong items within budget for both ref counts):

| Expansion | accept | 3 refs recall | 8 refs recall |
|---|---|---|---|
| none | 0.494 | 0.864 | 0.929 |
| 0.56 | 0.836 | 0.189 (expansion runs away) | 0.192 |
| 0.58 | 0.575 | 0.867 | 0.886 |
| 0.60 | 0.527 | 0.897 | 0.923 |
| 0.62 | 0.527 | 0.885 | 0.918 |
| 0.64 | 0.526 | 0.871 | 0.914 |
| 0.70 | 0.521 | 0.849 | 0.910 |

At the rounded cuts (accept 0.53), 0.60 vs 0.62:

| Expansion | CelebA 3 refs (of 16,030) | CelebA 8 refs (of 12,255) | DigiFace 3 refs (of 20,610) | DigiFace 8 refs (of 19,110) |
|---|---|---|---|---|
| 0.60 | 0.895 (14,341), 56 wrong | 0.921 (11,288), 150 wrong | 0.912 (18,791), 11,524 wrong | 0.938 (17,922), 13,738 wrong |
| **0.62 (shipped)** | **0.882 (14,133), 55 wrong** | **0.916 (11,229), 142 wrong** | **0.901 (18,562), 7,332 wrong** | **0.931 (17,789), 7,891 wrong** |
| buffalo_l (0.55 / 0.40) | 0.976 (15,640), 62 wrong | 0.979 (12,001), 180 wrong | 0.994 (20,483), 60,038 wrong | 0.996 (19,029), 60,712 wrong |

- 0.62 instead of 0.60: 0.60 lets in more wrong people on both sets (DigiFace 11,524 vs 7,332 wrong items with 3
  refs; CelebA 56 vs 55, 150 vs 142), and 0.58 already needs a much stricter accept. 0.60 sits one grid step from
  that drift; the phone's Vision landmarks add noise the cut was not tuned on. The price is about 1 point of recall
  (CelebA 0.895 -> 0.882 with 3 refs, 0.921 -> 0.916 with 8).
- Different-person pairs above accept 0.53: CelebA 201 of 167.1M (buffalo_l at 0.40: 323), DigiFace 4,448 of 230.6M
  (buffalo_l: 77,615). DigiFace's rendered faces look far more alike than real people, so its wrong-item counts at
  buffalo_l's shipped cuts (about 60,000 per 300 queries) say little about users. CelebA is the real-face reference.
- Possible list (server only, [0.42, 0.53)), CelebA 3 refs: 985 right vs 4,896 wrong items (buffalo_l: 53 vs 4,713).
  It is a much noisier list than before, and it is never auto-added.
- What the switch costs: with the same number of wrong photos, AuraFace+flip finds 88 of every 100 of a person's
  other photos with 3 reference photos (buffalo_l 98), and 92 with 8 (buffalo_l 98). More reference photos help.
  The deep dive's raw-trace review (12/12 sampled misses were the right person; blur, gaze, sunglasses, profile) applies.
- Not measured: the phone itself (Vision landmarks, not SCRFD keypoints), real libraries (Reza's labels), and the
  cost of AuraFace's two ResNet-100 passes per face. FIRST_DEVICE_TEST step 13.

## 37. Video frame sampling sweep (10-08; eval/eval_video_sampling.py -> eval/video_sampling/analyze.log)
Why: the phone's every-2-s / max-40 frame rule was inherited from the server, never tested, and it is most of the cost
of indexing Reza's 36,497 iCloud-only videos (~332k frames at 2 s). Set: 522 Pexels clips (<10 s 83, 10-30 s 311,
30-80 s 128, NONE >= 80 s), 24 queries; truth = 9B judge on a 1-s frame grid (strict: >= 2 frames yes, 303 truth
videos; lenient: >= 1, 331). A video counts as found if any sampled frame ranks it (the phone judges one best frame).

| rule | frames/video | strict found (of 303) | <10 s (of 39) | paired vs 2 s (+/-, sign-test p) |
|---|---|---|---|---|
| poster only (t=0) | 1.0 | 191 (0.630) | 28 | +27 / -49, p = 0.02 (worse) |
| middle frame | 1.0 | 202 (0.667) | 31 | +27 / -38, p = 0.21 |
| fixed 3 | 3.0 | 216 (0.713) | 31 | +19 / -16, p = 0.74 |
| every 8 s | 3.3 | 214 (0.706) | 30 | +19 / -18, p = 1.00 |
| **every 4 s (adopted)** | 6.1 | **219 (0.723)** | 31 | +15 / -9, p = 0.31 |
| every 2 s (old) | 11.7 | 213 (0.703) | 30 | - |
| every 1 s | 21.6 | 218 (0.719) | 31 | +13 / -8, p = 0.38 |

Reading: from ~3 frames per clip upward, recall is flat within noise (2 s, 4 s, 1 s, fixed 3 all 0.70-0.72); more
frames add near-duplicates. Adopted every 4 s (half the 2-s cost, best point estimate). Fixed 3 is ~half again and
statistically the same, but long clips are untested (no clip >= 80 s; Reza has 2,143), so 4 s was kept. Poster-only
loses ~7 points (p = 0.02): the cover-first pass (M20) makes videos searchable early, the frames pass is still needed.

37b. Two-sweep frames pass (10-09, same harness + oracle; eval/video_sampling/analyze_sweeps.log). "fixed 3" above is 3
frames evenly over [0, d - 0.05] = START / MIDDLE / END, not t = 0 / 1/3 / 2/3. Strict truth, of 303:

| rule | frames/video | strict found | paired |
|---|---|---|---|
| cover + middle + end (= fixed 3; shipped sweep 1) | 3.0 | 216 | vs cover only +46 / -21 (191); vs every 4 s +13 / -16, p = 0.71 |
| cover + 1/3 + 2/3 (asked for, not shipped) | 3.0 | 211 | vs cover + middle + end +25 / -30, p = 0.59 |
| every 4 s + sweep 1's frames kept beside it (not shipped) | 6.6 | 217 | vs every 4 s +0 / -2, p = 0.50 |
| every 4 s (shipped sweep 2: sweep-1 frames reused only where they fall on its times) | 6.1 | 219 | - |

Reading: the middle + end layout costs the same 2 decoded frames as thirds and is the layout the 216 was measured on
(thirds 211; the 5-video gap is within noise). Extra frames beside the 4-s grid never added a video here (+0) and cost
2: they only change which frame wins the text score. So sweep 2 lands on exactly the every-4-s layout.

## 38. A smaller phone judge: compress the weights, not the photo (10-09; eval/compress/, scripts/sim_mlx_quant.py)
Problem: Qwen3-VL-4B-Instruct-4bit (mlx-community) needs ~3.6 GiB by the app's guard; the app has 3.2 GiB (M15). Image
resolution and image tokens unchanged (896 px); only weights compressed.
Footprint (safetensors headers; scripts/mlx_footprint.py): 3.094 GB = 2.881 GiB of weights = language layers 4-bit 2.044
GB + tied embeddings/lm_head 4-bit 0.219 + VISION TOWER IN bf16 0.831 (its Linear layers: 411 M weights, 0.822 GB).
Resident besides weights (estimates, not changed): KV cache 147,456 B per token (36 layers x K,V x 8 heads x 128 x 2 B);
a 896 px photo is 588 (4:3) - 784 (square) tokens -> ~0.09-0.12 GB; plus prefill transients (logits, vision attention).
The 3.6 GiB guard was set for Qwen3.5-4B and never measured for this model.
SIM EXACTNESS (correction to sections 32-35): our simulator rounded with plain min/max per group of 64. MLX anchors each
group on its larger-magnitude extreme and nudges the scale so 0.0 is exact. Against the REAL mlx-community tensors
(layer 0 q_proj, layer 20 down_proj): old sim 3.6% of weights identical, exact-MLX sim 76% (fp32-vs-bf16 compare; in the
job, embed_tokens bf16-vs-bf16: 99.3% identical, max diff 0.02). New baseline = the real checkpoint de-quantized
(--from-mlx). The phone's judge on the same eye labels (261 right / 141 wrong), right / wrong kept:
| judge weights | P>=0.7 | P>=0.95 | P>=0.99 | agree with 9B (of 11,088) | subject q. at 0.99 (of 139 r / 97 w) |
|---|---|---|---|---|---|
| old sim (sections 32-35) | 242 / 59 | 229 / 39 | 211 / 24 | 10,209 | 131 / 24 |
| REAL phone weights (q3vl4b_real) | 232 / 47 | 221 / 33 | 198 / 21 | 10,335 | 131 / 21 |
So the real phone judge keeps ~10 fewer real and ~12 fewer wrong photos at 0.7 than sections 32-35 reported (still well
above the Qwen3.5-4B q4 phone judge's 208 / 33 at 0.7 in the old sim). Same-checkpoint rerun noise: 1-2 photos per cell.
Candidates on the REAL language model (MLX affine, round to nearest, group 64; eval/compress/judge_report_real.txt):
| vision tower | saves | P>=0.7 | P>=0.95 | P>=0.99 | flips vs real at 0.95 (+kept r/w/u, -dropped r/w/u) |
|---|---|---|---|---|---|
| bf16 (now) | - | 232 / 47 | 221 / 33 | 198 / 21 | |
| 8-bit | 0.385 GB (0.359 GiB) | 231 / 45 | 221 / 33 | 201 / 22 | +1/2/1, -1/2/0 |
| 6-bit | 0.488 GB (0.455 GiB) | 232 / 47 | 218 / 33 | 197 / 21 | +1/3/2, -4/3/1 |
| **5-bit** | **0.539 GB (0.502 GiB)** | **232 / 44** | **222 / 33** | 200 / 26 | +4/3/5, -3/3/0 |
| 4-bit | 0.591 GB (0.550 GiB) | 230 / 50 | 212 / 36 | 195 / 25 | +2/8/4, -11/5/0 |
On the old-sim language model (eval/compress/judge_report.txt, vs 242/59, 229/39): embeddings 3-bit (saves 0.049 GB)
236/51, 218/33; language + embeddings 3-bit group 32 (saves 0.251 GB) 201/28, 181/17 = -41 real photos: rejected.
Contact sheet (vision 5-bit vs real, all 30 flipped eye-labeled photos at native ~500 px, viewed): every flip is a
borderline call, P moving across the cut by a step (0.56 <-> 0.73 at 0.7, 0.92 <-> 0.95 at 0.95), except two large
swings, both unsure/wrong: a boardwalk with a beach behind glass ("beach" 0.05 -> 0.96, unsure) and rubber ducks
pouring off a bridge ("food" 0.32 -> 0.99, wrong). Correct new drops: a dog on a sofa as "cat", a white house as
"church", a skyline reflected at dusk as "sunset". Real photos lost: a tiny boat at a quay edge (0.73 -> 0.50), and at
0.95 a cherry-tree street and a chenille plant ("flowers" 0.95/0.98 -> 0.92, still kept at 0.7). Real photos gained: a
team car with bikes on its roof, a church fair, kids on tricycles with a bicycle, a stunt car in a fireball, a dog's
paw. Rocky shores flip to "beach" in both directions (the judge's known looseness, section 34). Net: noise, no
direction.
Reza's heavier-vs-fit demo (shipped planner planner_4b27ball_q4merged, prob mode; his labels 267 H / 124 F; same plan
text in every run, checked): REAL weights 265 H + 5 F / 115 F + 2 H; vision 8-bit 265 + 5 / 114 + 2; 6-bit 265 + 5 /
109 + 2; **5-bit 137 H + 1 F / 119 F + 18 H, with 117 photos "not clearly either" (real: 4)**; 4-bit 257 + 6 / 114 + 2.
Old-sim language model: baseline 265 + 6 / 111 + 2, its rerun 265 + 7 / 111 + 2, vision 5-bit 264 + 9 / 109 + 2.
So 5-bit vision on the real weights breaks the person-look split (the old-sim 5-bit did not), and the bit-widths are not
monotonic (4-bit survives the demo). The person-look judgment is sensitive to vision rounding in a way the 402 eye
labels (objects/scenes) do not show. Not yet explained; would need the judge P's per photo (run caches) to trace.
DECISION (revised): 5-bit REJECTED. No candidate saves >= 0.5 GiB cleanly. Ranked: vision 8-bit is clean on both tests
(eye 231/45, 221/33; demo -1 fit) but saves 0.36 GiB, less than the 0.4 GiB gap to an UNMEASURED guard; vision 6-bit
saves 0.455 GiB (closes the gap) at -6 fit photos of 124 and -3 real at 0.95. Next: measure the real peak on the phone
(M25 step 1); if the true gap is <= 0.36 GiB ship 8-bit, else 6-bit with its cost stated. scripts/quantize_vision_mlx.py
takes --bits; mlx-swift-lm 3.32.x loads per-layer bits.


**Correction (cluster, 10-09 ~01:20) to the 5-bit demo reading above.** The 5-bit collapse (137 H; 117 "not clearly
either") is NOT a judge failure. Judge P(yes) vs the real baseline on the same 800 cached (photo, question) pairs,
|diff| 90th percentile: 8-bit 0.019, 6-bit 0.024, 5-bit 0.052, 4-bit 0.088 (monotone; 4-bit, with larger shifts,
split fine). Same plan text in every run. What broke is the PAIRING step (GMM split on logit(pA) - logit(pB)), which
flipped to a degenerate split on small noise: a product fragility independent of compression, being fixed and
re-scored offline on every saved demo run. The M25 default (8-bit if the measured gap allows, else 6-bit) stands
as the conservative choice; 5-bit is re-evaluated after the pairing fix.

## 39. Person-album pairing split: robust to a tied event (10-09; scripts/replay_pairing.py, engine._two_groups)
Problem: Reza's heavier-vs-fit demo on the 5-bit-vision judge split 137 H + 1 F / 119 F + 18 H with 117 "not clearly
either", while its judge scores barely moved from the real-weight run (265 + 5 / 115 + 2; |diff| p90 0.052).
REPRODUCED OFFLINE, no GPU: scripts/replay_pairing.py renders each pooled photo's judge crop once (Slurm CPU job),
hashes it like CachedJudge, and answers every judge call from the run's judge_cache.json (a stub judge). All 33 saved
demo runs (data/private/sample_runs/mode_*: 9B, Qwen3.5-4B q4, Qwen3-VL 2B/4B, vote, prob / hard / rating / 0-100 /
bipolar modes, 12 compression candidates; 392 face-matched photos each): 0 cache misses, cache P == saved pool P
exactly, and the replayed albums equal the saved album folders in 33/33 runs (realvis5: 138 / 137 / 117 reproduced).
CAUSE: the split is a 2-component Gaussian mixture (EM) on x = logit(pA) - logit(pB), each photo's x replaced by its
event's median (event = photos <= 3 h apart). One event holds 137 of the 392 photos, so 137 values are EXACTLY tied.
EM started at the quartiles; with 35% of the data tied, the 75th percentile IS the tie. A Gaussian on a point mass has
unbounded likelihood as its spread shrinks, so EM can fall into it: realvis5 ended at sd 0.01 (the floor) on the tie
vs sd 5.37 for everything else (log-lik -533, "better" than the real split's -1047). Posterior >= 0.9 then means
"in that event" (137 photos -> heavier) vs "far from it" (fit, incl. 18 heavier-era photos); the 75 heavier photos just
above the tie are in neither. The real-weight run (bf16 vision) from the same start escaped the tie (sd 1.37 / 2.33) on almost the same
data: which basin EM lands in flips on tiny noise. The same collapse was already in 4bq_prob (137 + 0 / 117 + 19, 119
unclear) and the three early vote runs (137 + 0 / 106-107 + 49).
FIX (engine._two_groups + FindPicsCore.twoGroups, same rule): two EM starts, the quartiles (old) and the exact 1-D
two-means split (best between-group variance; deterministic). A fit is degenerate if a component's spread hits the
floor (0.01) or >= 0.75 of a component's weight is one event (a group that is one event is a moment, not a lasting
look). Non-degenerate fits win, then higher likelihood; if every start is degenerate, the best of them (= old answer).
On the saved runs good components put <= 0.64 of their weight in one event, collapsed ones >= 0.79.
Rejected (measured on all runs): sd floor 0.1 x spread (realvis5 still collapses), 0.25 x (heavier empties on realvis5
and 4bq_prob); fit on one value per event (12 good runs lose up to 10 fit or 7 heavier photos; q3vl2b, vis4, ens4 fall
back to the rank margin); fit on raw per-photo values (fit album empties in 13 runs); "fall back to the rank margin when > 20% unclear" (realvis5 116 + 3 / 112 + 12, worse).
BEFORE -> AFTER, all 33 runs, from the judge caches (heavier: H + F / fit: F + H / not clearly either; Reza's labels
267 H / 124 F). Changed runs:
| run | before | after |
|---|---|---|
| q3vl4b_realvis5 (5-bit vision) | 137 + 1 / 119 + 18 / 117 | 265 + 5 / 114 + 0 / 7 |
| 4bq_prob (Qwen3.5-4B q4, build question) | 137 + 0 / 117 + 19 / 119 | 267 + 8 / 109 + 0 / 7 |
| 9b_r100 (already broken: 240 H in fit) | 22 + 2 / 116 + 240 / 11 | 22 + 0 / 122 + 245 / 2 |
The other 30 runs are identical (every count): q3vl4b real 265 + 5 / 115 + 2 / 4; realvis8 265 + 5 / 114 + 2 / 5;
realvis6 265 + 5 / 109 + 2 / 10; realvis4 257 + 6 / 114 + 2 / 12; vis8/6/5/4 265+6/115+2, 265+6/109+2, 264+9/109+2,
264+5/106+2; base_rerun 265+7/111+2; emb3 265+8/112+2; lm3g32 262+7/114+0; lmexact 265+5/115+0; q4pl 265+6/111+2;
q3vl_prob 265+6/111+2; q3vl2b 256+6/111+11; 9b_prob and 9b_prob_fact 267+8/116+0; 9b_prob_bi 267+19/105+0;
9b_rating 229+3/121+32; 9b_hard 214+0/124+53; 4bq_rating 236+9/114+28; 4bq_rating_bi 248+4/120+18; ens5 265+7/111+2;
ens4 260+22/42+0 (67 unclear); still collapsed, unchanged (every start degenerate): ens/ens2/ens3 137+0/106-107+49;
rank-margin fallback unchanged: 4bq_hard 63+0/0+0, 9b_rating_bi 75+0/66+10; 4bq_r100 174+34/72+0 (112 unclear).
NOISE TEST (data/private/audits/pair_fix/noise.py): each run's P's perturbed by logit noise N(0, s), 40 draws, s = 0.02 /
0.05 / 0.1; "collapse" = >= 50 photos unclear. The 25 runs not collapsed at s = 0: old 80 / 80 / 84 collapses of 1,000
draws per s (4bq_prob 40/40, realvis5 37-40/40, realvis4 7/40 at s = 0.1), new 0 / 0 / 0. The 5 runs collapsed at
s = 0 (4bq_r100, ens x3, ens4) stay collapsed in all draws under both. Worse anywhere: ens4 at s = 0.1, 2 of 40 draws
fall back to the rank margin (heavier min 123 vs 137 old; max wrong 40 vs 28).
Tests: Python 131 pass (+3: tied-event split, one-event rejection, two-means); FindPicsCore swift test 89/89 (pairing fixtures 7 -> 10 cases:
the 7 originals' expected answers unchanged; + a synthetic tie the old start collapses on, + a synthetic tie where every
start collapses, + a deterministic synthetic near-tie that only the one-event test catches: one 137-photo event tied
at 0 plus 40 photos within 0.1 (collapsed fit sd 0.027, above the floor; 0.78 of its weight in the tied event) is chosen
without events and rejected with them; split then exact, 237 / 120 / 0). Correction 10-09: an earlier version of this
fixture held an anonymised case built from a demo run's logit differences; removed (derived from private data).
5-BIT VISION RE-SCORED: with the fix the 5-bit demo is 265 H + 5 F / 114 F + 0 H (real weights 265 + 5 / 115 + 2), and
its eye labels were already equal (section 38). The demo no longer argues against 5-bit (saves 0.50 GiB); M25's choice
of bits waits on the phone's measured peak as before.
