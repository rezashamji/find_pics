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

## 40. find pics vs Apple Photos search, the real baseline (10-09; eval/score_apple_photos.py, MAC_INBOX M29)
Question: on the same photos and the same searches, does find pics find more of the real matches than the search
already on every iPhone/Mac, without showing more wrong ones? This decides whether the product is worth shipping.
Set: the RESULTS 34 test set (4 DISBench Flickr libraries, 7,886 photos, ~500 px), imported into Photos as ONE
library; 6 searches (dog, car, bicycle, beach, sunset, food). Truth: the RESULTS 34 blind eye labels (1,950) + earlier
eye labels, stratified estimator (eval_recall.score_systems: section 34's strata and labels, cells split by which systems returned
each photo; validation at the end of this section). Both systems are scored
by this one function. "unsure" = no match (sensitivity in eval/apple_photos/report.txt).
Package: data/public/apple_photos_test (7,886 original JPEG bytes as <item_id>.jpg + manifest.tsv; the label key stays
on the cluster). The Mac fetches the same bytes itself from YFCC100M's public S3 bucket (scripts/fetch_apple_photos_test.py,
sha256-checked against eval/apple_photos/manifest.tsv): no rsync, no 2FA.
Apple side: PENDING (M29). Three query styles: keyword ("dog"), "photos of a dog", "photos with a dog"
(scripts/apple_photos_search.sh: AppleScript `search for`; the UI search bar is compared on 2 queries).
find pics side (10-09, filled): the CURRENT phone pipeline. Judge = models/q3vl4b_real (the real mlx-community 4-bit
Qwen3-VL-4B checkpoint, de-quantized; RESULTS 38), re-run on all 7,886 x 6 = 47,316 (photo, question) pairs
(eval_recall.py gen --model --tag=q3vl4b_real, 8 Slurm shards); planner questions of RESULTS 34; cut =
FindPicsCore.judgeCutoff (0.99 for "Is this a photo of X?" = beach/sunset/food, 0.7 for the object questions).
Exhaustive = every photo judged. Fast = what the app shows first: round 1 of streamRounds with Search.swift's
parameters (head 150, +50 while >= 3% of the last 50 are yes, max 1,500; random tail sample 150), ranked by
PE-Core-B-16 (the phone's image-text model; eval/apple_photos/looks_b16.parquet), all 7,886 photos as one library.
Fast and exhaustive are scored jointly (one shared total; eval/apple_photos/report.txt). Unsure = no match:
| search | fast: returned / judge calls | fast: right / eye-labelled returns | fast: precision est. | fast: recall [95% boot; Bayes] | exhaustive: returned | exh.: right / labelled | exh.: precision est. | exh.: recall [95% boot; Bayes] |
|---|---|---|---|---|---|---|---|---|
| dog | 1,293 / 1,550 | 60/61 | 1.00 | 0.961 [0.918-0.984; 0.882-0.980] | 1,301 | 60/61 | 1.00 | 0.967 [0.924-0.990; 0.887-0.986] |
| car | 306 / 1,400 | 35/35 | 1.00 | **0.540** [0.470-0.601; 0.437-0.595] | 410 | 43/46 (2 unsure) | 0.99 | 0.717 [0.624-0.797; 0.576-0.775] |
| bicycle | 31 / 300 | 28/28 | 1.00 | **0.596** [0.574-0.608; 0.205-0.595] | 48 | 44/44 | 1.00 | 0.923 [0.889-0.941; 0.317-0.913] |
| beach | 309 / 750 | 30/44 (10 unsure) | 0.68 | 0.962 [0.895-0.996; 0.662-0.966] | 334 | 31/50 (12) | 0.65 | 0.995 [0.994-0.996; 0.693-0.988] |
| sunset | 167 / 550 | 29/38 (9) | 0.72 | 0.930 [0.839-0.985; 0.525-0.941] | 172 | 29/41 (10) | 0.70 | 0.930 [0.840-0.985; 0.526-0.947] |
| food | 151 / 550 | 28/41 (8) | 0.66 | 1.000 [0.943-1.000; 0.499-0.976] | 164 | 28/47 (12) | 0.61 | 1.000 [1.000-1.000; 0.506-0.982] |
| all 24 searches | 2,257 / 5,100 | | 0.91 | 0.854 (2,057 of est. 2,408 real) | 2,429 / 47,316 | | 0.90 | 0.909 (2,190 of 2,408) |
- Unsure as match: fast pooled recall 0.774, exhaustive 0.827 (car 0.419 / 0.559: unidentifiable vehicle specks).
- Fast mode's 20 tail-sample seeds move recall by <= 0.06 (bicycle 0.598-0.656), elsewhere <= 0.01.
- WHERE FAST MODE LOSES: car (-104 photos the exhaustive judge keeps) and bicycle (-17). Not the judge: those photos
  sit below the head, where the stop rule (< 2 yes in the last 50) ends round 1. It is not the phone's smaller image
  model either: PE-Core-L ranks them about the same (bicycle yes-photos in the top 150: B-16 31/48, L 34/48; car top
  600: 237 vs 267 of 410). Contact sheets at native ~500 px (eval/apple_photos/sheets/fastmiss_*, 12 random each,
  viewed): bicycle 9/12 clearly show a real bicycle (behind people, by a fence, on a lawn, a BMX trick, a parked
  trike), 3/12 too small to confirm at this size; car 9/12 clearly show a car (behind a woman, at a frame edge, parked
  in a harbour village, a taxi corner), 3/12 too small or unclear. Same failure as RESULTS 34: small things in
  busy scenes, now at the fast stage. The completeness note does say round 1 left most of the library unchecked.
- Real phone weights vs the old simulator (RESULTS 34, 0.7 everywhere): kept dog 1,301 vs 1,305, car 410 vs 461,
  bicycle 48 vs 52; beach/sunset/food at 0.99: 334 / 172 / 164 (old sim at 0.99: 348 / 208 / 165). Car recall 0.717
  (old sim 0.79): the real weights drop more small cars, matching RESULTS 38's "fewer real and fewer wrong kept".
- Cost on the phone (not measured here): judge calls above; at the ~1 photo/s Search.swift assumes, fast mode is
  ~5 min (bicycle) to ~26 min (dog), exhaustive ~2.2 h per search for 7,886 photos.
APPLE PHOTOS: PENDING (M29). When results_apple_*.tsv arrive: `python eval/score_apple_photos.py score` scores all
systems jointly (find pics numbers above can move slightly: the shared total is re-estimated with finer cells), then
view a contact sheet of Apple's returns that have no eye label (and of what each system finds that the other misses)
before any claim. Precision of a system's returns rests on few labels where it returns photos the RESULTS 34 judge
rejected (A/B/C strata): a blind label pass on a uniform sample of those returns would tighten it.
Scorer validation: on the RESULTS 34 judge's kept set score_systems == estimate() to 4 decimals for all 12 rows
(eval/apple_photos/selftest.log); at P >= 0.99 it reproduces RESULTS 35 (beach / sunset / food recall 0.995 / 0.971 /
1.000; precision 0.62 / 0.56 / 0.61 vs 0.59 / 0.56 / 0.66 there, which used plain Horvitz-Thompson; the cell
estimator here is bounded by the returned count).

## 41. Fast mode's stop rule, tuned offline (10-09; eval/tune_stop_rule.py -> eval/stop_rule/)
Question: RESULTS 40's fast mode kept car 306 and bicycle 31 of the 410 / 48 photos the same judge keeps when it checks
everything. Can a different round-1 stop rule get them back without making dog/food/beach searches much slower?
Method: no new judge calls. Replay of FindPicsCore.streamRounds on the stored answers of RESULTS 40 (real phone weights
q3vl4b_real on all 7,886 photos x 6 searches, judgeCutoff 0.99 / 0.7, PE-Core-B-16 ranking), 20 tail-sample seeds per
rule. The replay reproduces RESULTS 40 exactly on cost (heads 1,400 / 1,250 / 150 / 600 / 400 / 400; 5,100 judge
calls) and within seed noise on found (car 303-308 vs 303-310). Recall = RESULTS 34 eye-label truth (score_set, unsure
= no match), mean over the 20 seeds. Cost = judge calls, ~1 s each on the phone.
Why round 1 lost them (judge-yes positions in the B-16 ranking): bicycle had 27 yes in the top 50, 4 in 50-99, 0 in
100-149, then 5 in 150-299; the old rule ("stop when fewer than 2 of the last 50 are yes") stopped at 150. Car had
~7% yes at ranks 1,000-1,500; one 50-photo window with 1 yes stopped it at 1,250. A window of 50 at a 6% yes rate has
P(<= 1 yes) = 0.19, so over 20 windows a false stop is likely. Beyond that, car has 89 of its 410 yes past rank 1,500
(cap) and ~65 past rank 2,000 at < 3% density: those need deep judging (~1.5 min of judging per extra car found).
| rule (phone: head 150, chunk 50, tail sample 150) | judge calls, 6 searches | dog | car | bicycle | beach | sunset | food | pooled recall |
|---|---|---|---|---|---|---|---|---|
| old: stop if < 3% of last 50 yes, cap 1,500 | 5,100 (85 min) | 0.961 / 1,550 | 0.543 / 1,400 | 0.605 / 300 | 0.959 / 750 | 0.935 / 550 | 1.000 / 550 | 0.856 |
| A: < 3% of last 100, cap 2,000 | 6,000 (+18%) | 0.963 / 1,600 | 0.601 / 2,000 | 0.699 / 450 | 0.959 / 750 | 0.931 / 600 | 1.000 / 600 | 0.871 |
| **A + auto round 2 if the random check finds >= 3** (adopted) | 6,839 (+34%) | 0.963 / 1,600 | 0.632 / 2,839 | 0.699 / 450 | 0.959 / 750 | 0.931 / 600 | 1.000 / 600 | 0.879 |
| A + auto round 2 if >= 2 | 7,431 (+46%) | 0.963 / 1,685 | 0.647 / 3,258 | 0.699 / 450 | 0.959 / 838 | 0.931 / 600 | 1.000 / 600 | 0.882 |
| A + auto round 2 if >= 1 | 8,950 (+75%) | 0.963 / 1,856 | 0.671 / 3,886 | 0.714 / 568 | 0.962 / 1,147 | 0.931 / 674 | 1.000 / 820 | 0.889 |
| deep: < 2% of last 300, cap 3,000 | 8,650 (+70%) | 0.964 / 1,800 | 0.653 / 3,000 | 0.739 / 650 | 0.964 / 1,500 | 0.927 / 750 | 1.000 / 950 | 0.886 |
| exhaustive (reference) | 47,316 | 0.967 | 0.717 | 0.923 | 0.995 | 0.925 | 1.000 | 0.909 |
Cells: eye-label recall / mean judge calls. Precision unchanged (car 0.99, bicycle 1.00, beach/sunset/food 0.64-0.69).
- Adopted rule recovers 51% of the car gap (0.543 -> 0.632 of exhaustive's 0.717) and 30% of the bicycle gap (0.605 ->
  0.699 of 0.923) for +34% judge calls; dog / beach cost +3% / +0%, food / sunset +9%. Round 2 ran in 8/20 car seeds
  (the random 150 found >= 3 cars) and 0/20 for the other 5 searches. The car's bound shown to the person drops from
  "up to ~358 more" to ~186 (mean).
- Most of the gap is NOT recoverable by any stop rule at fast-mode cost: the 7 bicycles past rank 1,500 and ~65 cars
  past 2,000 sit at 0.1-3% density. Only a better fast ranking (per-crop / tile vectors for small objects; RESULTS 13
  tiles lifted christmas tree 0.61 -> 0.74 in round 1) or exhaustive mode finds them.
- Ranking model check: replaying with the server's PE-Core-L-14 look score instead of B-16 moves pooled
  judge-relative recall by <= 0.015 for all 312 rules swept (eval/stop_rule/sweep_audit4_l.log); rules rank alike
  (Spearman 0.95 on recall, 0.96 on calls).
Validation on libraries the rule was NOT tuned on (overfitting check): the 16 DISBench libraries of RESULTS 27 (30,273
photos, 192 searches; yes = Qwen3-VL-4B 16-bit on every photo, cut 0.7; ranking PE-Core-L, no B-16 vectors there).
Found = share of the judge's yes set, no eye labels (cost and found counts only):
| rule | judge calls, 192 searches | found / judge-yes | car | bicycle | flowers | church | dog | beach | food | auto round 2 |
|---|---|---|---|---|---|---|---|---|---|---|
| old | 73,810 | 11,635 / 12,723 (0.914) | 0.848 | 0.925 | 0.892 | 0.931 | 0.901 | 0.976 | 0.956 | - |
| A | 85,310 (+16%) | 12,034 (0.946) | 0.936 | 0.933 | 0.914 | 0.965 | 0.901 | 0.982 | 0.971 | - |
| **A + round 2 if >= 3** | 92,863 (+26%) | 12,177 (0.957) | 0.964 | 0.935 | 0.929 | 0.972 | 0.901 | 0.982 | 0.971 | 5.7% of searches (car 40%, flowers 10%, selfies 11%, rest <= 4%) |
| A + round 2 if >= 2 | 99,039 (+34%) | 12,252 (0.963) | 0.977 | 0.938 | 0.941 | 0.975 | 0.901 | 0.982 | 0.972 | 10.8% |
- Same direction, larger car gain on unseen libraries (0.848 -> 0.964). Dog calls +0%, beach +9%, food +14% there.
  Of all 312 rules swept (eval/stop_rule/sweep_audit4.log, sweep_ev16.log), none finds more of the judge's yes with
  fewer calls than the adopted rule on either set.
  Selfies stay bad (22 -> 46 of 153: these plans rank selfies by look; separate issue, RESULTS 27's default look).
- Viewed (native ~500 px, eval/stop_rule/sheets/gain_*): 12 random of the 80 cars and all 5 bicycles the adopted rule
  adds over the old one (car seed with round 2): bicycles 5/5 real (behind walkers, at a church, parked at a pier);
  cars 10/12 clearly a car (parked by a church, behind a truck, a taxi corner, a pickup, aerial streets), 2/12 too small
  to confirm (a beach village, a garden edge).
Implemented (phone and server share the stop rule): FindPicsCore StreamParams.headWindow (nil = old rule) +
autoRoundHits/autoRounds + fastModeWantsMore(); Search.swift sets headMax 2,000, headWindow 100, auto round 2 at >= 3
random hits; engine.Thresholds.head_window (server defaults unchanged). The bound stays honest after an automatic round:
its alpha share was fixed before round 1's sample (replay test: 34 phone fast-mode rounds incl. 4 auto, 0 overclaims;
old 170-round replay unchanged, 0 overclaims). Swift 92/92, pytest 132/132 (golden window case in both).
Worst case on a big library: round 1 <= 2,150 calls (~36 min, was 1,650 / ~28 min); with round 2 <= ~4,300 (~72 min),
but round 2's results stream in after round 1's, so the first answer is not delayed by it.

## 42. Tile vectors for fast mode's ranking (10-10; eval/eval_tiles_phone.py -> eval/tiles_phone/)
Question (PLAN step 3): fast mode misses small background objects because the one whole-photo PE-Core-B-16 vector
ranks them too deep (RESULTS 41: ~65 of car's 410 judge-yes past rank 2,000; 7 of 48 bicycles past 1,500). Do extra
vectors per photo, from tiles, rank them into the judged head, and what does that cost on the phone?
Method: no new judge calls. Same set, judge, cut and truth as RESULTS 40/41 (7,886 photos as one library, 6 searches,
q3vl4b_real P(yes) on every photo, RESULTS 34 eye labels via score_set, unsure = no match). New: each photo simulated as
the phone indexes it (docs/PHONE_PARITY.md: long side 480 Lanczos + JPEG q80; every crop cut from that 480 px image;
Pillow bilinear squash to 224; fp16 B-16): whole, 2x2 grid, 3x3 grid, centre half (= a 2x zoom; also the "centre crop"
of 2x2+centre) = 15 vectors per photo (GPU job fp_tiles_phone, 9 min). Photo score = max over the whole vector and the
tiling's tile vectors (= engine.look_scores' existing clip_tiles rule); weighted maxes also tried. The ADOPTED fast rule
(RESULTS 41: window 100, cap 2,000, auto round 2 at >= 3 random hits) replayed over each ranking, 20 seeds.
Baseline = "whole (phone sim)": the phone's real single vector. It is slightly worse than the 500 px original vector
RESULTS 40/41 used (pooled 0.873 vs 0.879; car 0.610 vs 0.632): RESULTS 41's numbers overstate the phone by ~0.006.
Judge-yes photos by rank cutoff (top 600 / 1,500 / 2,000 of the judge-yes set; past 2,000):
| ranking (vectors per photo) | dog (1,301) | car (410) | bicycle (48) | beach (334) | sunset (172) | food (164) |
|---|---|---|---|---|---|---|
| B-16, 500 px original (RESULTS 41) (1) | 594/1,295/1,297 | 237/321/344; 66 past | 39/41/45 | 308/331/334 | 169/172/172 | 158/162/163 |
| whole, phone sim (1) | 594/1,293/1,297 | 241/312/341; 69 past | 39/42/44 | 310/333/334 | 170/171/171 | 155/163/163 |
| **2x2** (5) | 591/1,297/1,297 | 257/332/353; 57 past | 45/46/46 | 313/330/333 | 170/172/172 | 155/163/164 |
| 3x3 (10) | 589/1,297/1,298 | 251/322/337; 73 past | 43/46/47 | 310/322/329 | 170/172/172 | 156/161/162 |
| 2x2 + centre (6) | 591/1,298/1,298 | 257/332/352; 58 past | 45/46/46 | 313/331/333 | 170/172/172 | 155/163/164 |
| centre 2x zoom (2) | 592/1,295/1,297 | 224/303/329; 81 past | 38/40/42 | 308/330/333 | 170/171/172 | 155/160/162 |
| 2x2+3x3+centre (15) | 589/1,298/1,298 | 250/321/339; 71 past | 43/47/47 | 310/325/329 | 170/172/172 | 156/160/163 |
Fast mode (adopted rule) replayed: eye-label recall / judge calls (mean of 20 seeds); pooled recall [seed min-max]:
| ranking | pooled | calls | dog | car | bicycle | beach | sunset | food |
|---|---|---|---|---|---|---|---|---|
| B-16, 500 px original (RESULTS 41) | 0.879 [0.870-0.894] | 6,839 | 0.963 / 1,600 | 0.632 / 2,839 | 0.699 / 450 | 0.959 / 750 | 0.931 / 600 | 1.000 / 600 |
| whole, phone sim | 0.873 [0.864-0.889] | 6,799 | 0.962 / 1,600 | 0.610 / 2,599 | 0.696 / 450 | 0.957 / 900 | 0.931 / 600 | 1.000 / 650 |
| **2x2** | **0.884** [0.882-0.900] | 6,773 | 0.965 / 1,600 | 0.629 / 2,373 | **0.814** / 500 | 0.995* / 900 | 0.933 / 650 | 1.000 / 750 |
| 3x3 | 0.880 [0.872-0.896] | 6,972 | 0.965 / 1,700 | 0.608 / 2,372 | 0.812 / 500 | 0.995 / 950 | 0.934 / 600 | 1.000 / 850 |
| 2x2 + centre | 0.885 [0.880-0.900] | 6,818 | 0.965 / 1,600 | 0.629 / 2,418 | 0.813 / 550 | 0.995 / 950 | 0.933 / 650 | 1.000 / 650 |
| centre 2x zoom | 0.869 [0.863-0.893] | 6,948 | 0.963 / 1,600 | 0.602 / 2,598 | 0.565 / 350 | 0.958 / 1,000 | 0.932 / 600 | 1.000 / 800 |
| 2x2+3x3+centre | 0.882 [0.873-0.897] | 7,283 | 0.965 / 1,700 | 0.613 / 2,533 | 0.830 / 550 | 0.995 / 1,000 | 0.934 / 600 | 1.000 / 900 |
| exhaustive (reference) | 0.909 | 47,316 | 0.967 | 0.717 | 0.923 | 0.995 | 0.925 | 1.000 |
Precision unchanged (same judge): car 0.99-1.00, bicycle 1.00, beach 0.64-0.65, sunset 0.69, food 0.61-0.63.
Judge-yes found (mean): car 345 -> 358 of 410, bicycle 36 -> 42 of 48, beach 314 -> 318 of 334, dog 1,293 -> 1,297.
*Beach 0.957 -> 0.995 rests on ONE eye-labelled photo (7298026758, a labelled real beach the 2x2 head now reaches,
standing for a cell of the estimator); treat it as "beach not worse", not as a beach gain.
- Weighted max (tiles discounted by 0.01-0.03, or 0.5 x whole + 0.5 x max): best pooled 0.888 (2x2 - 0.01, +4% calls),
  all within the seed spread of plain 2x2 and chosen on the test set itself: no robust gain; plain max kept.
- Why car barely moves (mechanism, counted): 2x2 lifts 35 of the 69 judge-yes cars that sat past rank 2,000 into the
  top 2,000, but pushes 23 others out, because the tile max also lifts 639 judge-no photos into the car top 2,000 (419-645
  across dog/car/bicycle) (a quarter of a busy street "looks like a car" more than the whole photo does). Net +12. Bicycles are rarer, so
  their gain survives the churn: 4 of 6 deep bicycles move in, 0 out. 3x3 tiles (160 px of a 480 px image, upsampled
  to 224) add no pixels and more noise: car worse than 2x2.
- Tiles cannot add detail here: the test photos are 500 px and the phone indexes a 480 px rendition, so a 2x2 tile is
  240 px of real image stretched to 224. They only remove clutter around the object. Untested: tiles cut from a
  higher-resolution read (the face upgrade already reads at 1280 px), which would add real pixels; this set cannot
  test that (originals are 500 px).
Validation on the 16 unseen libraries (RESULTS 27/41's ev16: 30,273 photos, 192 searches, judge-yes = Qwen3-VL-4B
16-bit on every in-scope photo; same phone simulation, whole + 2x2 only, GPU job fp_tiles16; found / judge-yes, 20 seeds):
| ranking | found | calls | car (3,104) | bicycle (963) | flowers (1,680) | dog (159) | beach (844) | food (1,273) | sunset (462) |
|---|---|---|---|---|---|---|---|---|---|
| whole, phone sim | 12,142 / 12,723 (0.954) | 94,679 | 0.969 | 0.925 | 0.932 | 0.859 | 0.971 | 0.964 | 0.962 |
| **2x2** | 12,205 (0.959) | 94,080 (-0.6%) | 0.964 | 0.951 | 0.951 | 0.884 | 0.977 | 0.968 | 0.958 |
| 2x2, gate "busy" top 25% | 12,059 (0.948) | 97,102 | 0.965 | 0.936 | 0.911 | 0.880 | 0.953 | 0.944 | 0.956 |
| PE-Core-L whole (RESULTS 41) | 12,177 (0.957) | 92,863 | 0.964 | 0.935 | 0.929 | 0.901 | 0.982 | 0.971 | 0.975 |
- Same direction on unseen libraries, same size: +63 of 12,723 judge-yes (+0.5 points) at equal judge calls; bicycle
  +25, flowers +32, dog +4, cat +3; car -14.5 (as on audit4: no car gain); sunset -2. 2x2 B-16 ~= whole PE-Core-L.
Viewed (native ~500 px, eval/tiles_phone/sheets/, seed 0, 2x2 vs whole phone sim): all 7 bicycles 2x2 adds: 6/7 clearly a
real bicycle (parked by a fence, at a church, by a statue, behind a food stall, behind a tent, mid-street), 1/7 too small
to confirm (a park); the 1 bicycle 2x2 drops is a real BMX among spectators. 12 random of the 45 cars it adds: 8/12
clearly a car (a silver car behind a sculpture, a red Chevrolet pickup by marathon runners, parked cars at a stadium,
a van at a street's end, a car corner at a substation, cars behind street musicians and a protester), 4/12 too small to
confirm (harbour and aerial views); the 12 cars it drops look the same (a harbour aerial, a car at a frame edge, a red
van): tiles trade some small cars for others.
Gates (tiles only on some photos, to cut cost): "busy" = whole-vector prompt score (busy street / wide town view vs
close-up / portrait; decided before any tile is computed), "novel" = 2x2 tiles least like the whole photo (saves storage
only). On audit4 busy top 25% kept most of the bicycle gain (0.813, pooled 0.881) but on ev16 it is WORSE than no tiles
(0.948 vs 0.954): mixing tiled and untiled photos in one ranking favours the tiled ones whatever they show. Novel gates
were worse on audit4 (0.866-0.884). No gate is recommended.
Phone cost (187k-photo library, PE-Core-B-16 1,024-d Float16 = 2 KB per vector; Core ML 8 ms per call cold, 35-41 ms
sustained when the phone is hot, MAC 10-07 14:46 / 17:33; + ~2 ms Pillow-exact resize per crop):
| tiling | extra vectors / photo | extra disk (mmap store) | extra index time, cold (8 + 2 ms per crop) | hot (~38 + 2 ms) |
|---|---|---|---|---|
| current (whole) | 0 (store 0.38 GB) | - | - | - |
| centre 2x zoom | 1 | 0.38 GB | ~31 min | ~2.1 h |
| **2x2** | 4 | 1.53 GB | ~2.1 h | ~8.3 h |
| 2x2 + centre | 5 | 1.92 GB | ~2.6 h | ~10.4 h |
| 3x3 | 9 | 3.45 GB | ~4.7 h | ~18.7 h |
The store is memory-mapped, so it is disk, not app memory; but a search scores every vector (5 x 187k dot products of
1,024 = ~1 G multiply-adds) and pages the whole 1.9 GB in when it is not cached (~1 s of flash reads per cold search).
Tiles can be built lazily: search works without them (max over what exists), so a charger-only pass after the main
index and the face upgrade, like FaceUpgrade, fits; the face upgrade's 1,280 px read could feed the tiles with real
pixels for only the Core ML cost.
Recommendation: do NOT add tiles to the phone now. The gain is real but small and narrow: +0.011 pooled eye recall on
audit4 (+0.005 judge-relative on ev16) at equal judge calls, almost all from bicycles (+12 points; 6 of the 12 missing
judge-yes bikes) and flowers; dog / beach / food / sunset not worse. It does NOT fix the gap this step was for: car
0.610 -> 0.629 (exhaustive 0.717), ~57 judge-yes cars still past rank 2,000, and no car gain on ev16. For that the
price (5x the vector store, +1.5 GB, +2-8 h of indexing) is too high. If adopted later, it is plain 2x2 + whole (max;
no 3x3, no centre crop, no gate), built lazily on the charger from the 1,280 px face-upgrade read, after measuring on
higher-resolution photos whether real pixels change the car result. The car gap stays with exhaustive mode / round 2
(RESULTS 41) or a different mechanism (detector crops) for now.

## 43. Fast mode against the phone's clock (10-10; eval/tune_stop_time.py -> eval/stop_time/)
Question: on Reza's iPhone 18 Pro the Qwen3-VL-4B judge runs ~0.67 photos/s sustained (MAC 10-10 13:56: 92 photos in the
first 91 s, then 0.82, 0.61, 0.52/s: thermal). RESULTS 41's rule (head 150, +50 while >= 3% of the last 100 head photos
are yes, cap 2,000, THEN 150 random photos of the rest, auto round 2 at >= 3 random hits) was tuned in judge calls at an
assumed 1/s, and its completeness bound appears only after the whole head. When does each rule show matches and a bound,
in minutes on the phone?
Method: no new judge calls. Same stored answers, truth and seeds as RESULTS 41/42 (audit4: q3vl4b_real on 7,886 photos x
6 searches, cut judgeCutoff, B-16 ranking, RESULTS 34 eye labels via score_set, unsure = no match; ev16: 192 searches,
judge-relative only), 20 tail-sample seeds. New: every rule is replayed as the ORDERED sequence of judge calls it makes,
so recall can be read at any time. Time = calls / 0.67 per s ("const"); "thermal" = the measured cumulative curve
above, 0.52/s after 14 min. Per-call overhead cannot be separated from the 0.67/s (each call = 896 px fetch + vision
encoder + prefill, measured only as a total); planner and ranking time before the first call are not included.
The old rule is mirrored exactly (same calls and found as tune_stop_rule.simulate, checked on 18 runs).
New round structure ("timed", FindPicsCore StreamParams.roundCalls > 0, Streaming.swift streamTimed):
- a uniform permutation of all positions is fixed before any label; the head is judged in rank order;
- round k ends after roundCalls x 2^(k-1) judge calls (a time budget: a new bound every doubling of time), or earlier
  when RESULTS 41's head rule is satisfied (same window/rate/cap); a round that ends on time tops the random sample up
  to `topup` photos, the round in which the head rule is satisfied tops it up to 150 (as before); after that one,
  rounds double the calls so far; fast mode keeps going while the head rule is not satisfied, plus one more round if
  the random sample found >= 3 (= RESULTS 41's auto round 2);
- the random sample of round k = every permutation item drawn so far that lies in the unchecked rest [head, n). Items
  are only skipped when they were inside the head when drawn, so it is the first m items of the permutation restricted
  to the rest; the head boundary, the draw times and the budgets depend on head labels and permutation positions, never
  on a label of the rest. Any reordering of the rest's positions leaves the run unchanged, so given m the sample is a
  uniform m-subset of the rest. Checked on 2,000 seeds (car, bicycle; eval/stop_time/uniform.txt): mean sample hits = sample size x
  rest's yes rate in every round (car round 1: 1.11 vs 1.16 expected, round 3: 0.76 vs 0.79; bicycle 0.06 vs 0.07 ...; 0 overclaims in 2,000
  seeds x 6-7 rounds;
  round 1 is uniform by construction, so ~0.04 is the noise level);
- error budget: round 1 gets a1 x alpha, rounds 2..K (1 - a1) x alpha / (K-1) each; K = 2 + ceil(log2(n / min(120,
  150))) is fixed by n before any label (every round but the satisfied one ends with calls >= its budget, so round k
  ends with >= 120 x 2^(k-2) calls; round K is made to judge everything). Union bound: every bound shown holds at once.
Sweep (audit4 and ev16; 600 rules first, then 65): round budget 60-200 calls, interleaving one random photo per 1-4 head
photos vs none, top-up 30-100, first head 60/100/150, extra round 0/1. Findings:
- INTERLEAVING LOSES. It spends random calls earlier, so matches come later: round 120, top-up 30, one random per 4
  head photos vs none: per-search mean recall at 3 min 0.379 vs 0.429, at 10 min 0.689 vs 0.715 (audit4,
  judge-relative), for a bound 0.6 min sooner (3.1 vs 3.7 min); eye-label pooled at 3 min 0.192 (top-up 50) vs 0.221.
  Its only gain is a higher final car recall (0.674 vs 0.621 eye) because its extra round fires more, at a p90 done
  time of 113 min vs 50.
- A SMALLER FIRST HEAD (60 or 100) changes nothing at 3/10 min (same calls in rank order) and does not end searches
  sooner: identical rows. Kept 150.
- What the early bound costs is just its random photos: 30 photos = 45 s of judging, taken at minute 3.
Chosen: roundCalls 120 (~3 min), top-up 30, no interleaving, a1 = 0.2 (FindPicsCore.phoneFastParams).
Eye-label recall of the matches ON SCREEN at each minute (audit4, 6 searches, mean of 20 seeds; pooled = all real
matches together, RESULTS 34 truth):
| rule, time model | 1 min | 3 min | 5 min | 10 min | 20 min | final | judge calls, 6 searches | first bound shown |
|---|---|---|---|---|---|---|---|---|
| old (RESULTS 41), 0.67/s | 0.092 | 0.221 | 0.306 | 0.442 | 0.633 | 0.879 | 6,839 | median 16.8 min, max 49.8 |
| **timed (chosen)**, 0.67/s | 0.092 | 0.221 | 0.282 | 0.428 | 0.621 | 0.876 | 6,552 | 3.7 min, every search |
| timed, top-up 50, a1 0.1, 0.67/s | 0.092 | 0.221 | 0.264 | 0.418 | 0.614 | 0.876 | 6,552 | 4.2 min |
| timed, round 90 calls, top-up 50 | 0.092 | 0.183 | 0.264 | 0.418 | 0.613 | 0.876 | 6,552 | 3.5 min |
| timed, interleave 1:4, top-up 50 | 0.076 | 0.192 | 0.249 | 0.390 | 0.560 | 0.889 | 8,459 | 3.6 min (max 3.7) |
| old, thermal clock | 0.127 | 0.271 | 0.356 | 0.466 | 0.614 | 0.879 | | median ~17, max ~60 min |
| **timed (chosen)**, thermal clock | 0.127 | 0.240 | 0.339 | 0.455 | 0.602 | 0.876 | | 2.7 min |
Per search, chosen vs old (eye recall at 3 / 10 / 20 min / final): dog 0.087/0.277/0.573/0.963 vs 0.087/0.297/0.593/0.963;
car 0.197/0.350/0.463/0.621 vs 0.197/0.365/0.469/0.632; bicycle 0.598/0.699/0.702/0.702 vs 0.598/0.699/0.699/0.699;
beach 0.450/0.952/0.957/0.957 vs 0.450/0.953/0.959/0.959; sunset 0.713/0.935/0.931/0.931 vs 0.713/0.931/0.931/0.931;
food 0.875/1.000 both (eye_const.txt; seeds differ slightly between rules, so +-0.004 is noise).
Judge-relative found / judge-yes, mean over searches (each search counts once), 3 / 5 / 10 / 20 min / final:
| set | old | chosen | first bound | done (median, p90) |
|---|---|---|---|---|
| audit4, 6 searches | 0.429 / 0.585 / 0.725 / 0.807 / 0.910 | 0.429 / 0.540 / 0.715 / 0.803 / 0.908 | 3.7 min vs median 16.8 (max 49.8) | 16.8 / 49.8 min, both |
| ev16, 192 searches (unseen) | 0.696 / 0.770 / 0.826 / 0.845 / 0.851 | 0.696 / 0.754 / 0.824 / 0.844 / 0.850 | 3.7 min vs median 8.7 (max 38.6) | 8.7 / 21.1 min, both |
- First match: median 1 judge call (~1.5 s after the judge starts) for every rule and search (the head starts at the
  best-ranked photo); first matches are not the problem. A dense search (dog, 1,301 matches) is capped by calls:
  at 0.67/s no rule can show more than ~120 photos in 3 min (dog 0.09 at 3 min, 0.30 at 10 min for every rule).
- Cost of the early bound: recall at 5 min -0.024 eye pooled (-0.045 per-search mean), at 10 min -0.014 (-0.010),
  at 20 min -0.012 (-0.004), final -0.003; the car's final eye recall 0.632 -> 0.621 (its extra round fires on a
  bigger, differently timed sample). Done times unchanged (median and p90 identical on both sets).
- THE FIRST BOUND IS COARSE. 30 random photos can only say "up to ~14% of the rest" (median first bound +1,105 for the
  sparse searches, +1,566 car, +2,643 dog, of ~7,766 unchecked). It tightens every round (~+250 at the end). The final
  bound is LOOSER than before: median +223-293 vs +156-188 (error budget split over up to K = 9 rounds instead of 1-2:
  alpha per round 0.005 vs 0.025; ev16 final bound 2.9% vs 2.2% of the library). Honest, but less informative.
Over-claims (the replay: every round of every seed, bound's recall lower limit vs true found / all judge-yes; 0 allowed):
| rule | audit4 fast | audit4 exhaustive | ev16 fast | ev16 exhaustive |
|---|---|---|---|---|
| old (RESULTS 41) | 0 / 128 | - | 0 / 3,938 | - |
| timed top-up 30, a1 0.5 | 0 / 466 | 0 / 900 | 2 / 10,044 | 2 / 18,658 |
| timed top-up 50, a1 0.5 | 0 / 506 | 0 / 940 | 5 / 10,684 | 5 / 19,298 |
| timed top-up 30, a1 0.1 / 0.05 | 0 / 466 | 0 / 900 | 1 / 2 of 10,044 | 1 / 2 of 18,658 |
| timed top-up 50, a1 0.1 | 0 / 506 | 0 / 940 | 0 / 10,684 | 0 / 19,298 |
| **timed top-up 30, a1 0.2 (chosen)** | 0 / 466 | 0 / 900 | **0 / 10,044** | **0 / 18,658** |
| timed round 60, interleave 1:3, top-up 20 | 1 / 677 | 1 / 1,080 | 2 / 14,729 | - |
Every overclaim was listed with how likely so bad a sample is (overlist.log): all are round-1 or round-2 bounds from
30-50 random photos while the rest still held 7-20% matches, e.g. 0 hits in 30 of a 58-photo rest holding 8
(P = 0.0016, alpha 0.025), 0 hits in 50 of 1,843 holding 133 (P = 0.022, alpha 0.025), 1 in 30 of 1,506 holding 310
(P = 0.008, alpha 0.009); the bound missed by 1-9 photos. These are the bound's stated risk, not a bug (a bug would
show P far below alpha). A 95% bound read early, from a small sample, while many matches remain, is right at its
nominal level; the old rule's 0/3,938 came from reading it late, when the rest was nearly empty. The chosen a1 = 0.2
gives 0 in this replay but was picked AFTER seeing it (a1 = 0.1 and 0.05 each showed 1-2): with more seeds an
overclaim at the stated rate will appear for this rule too, as it would for the old one.
Swift: streamTimed == the Python replay call for call (golden test, 3 public concepts x fast/exhaustive x 2 rules incl.
interleaving: rounds, sample sizes, hits, found, bound, first 80 calls); testTimedNeverOverclaims (phone params, 6
concepts x 5 seeds, fast + exhaustive): 0 of 425 rounds. FindPicsCore 101/101 (was 99 + 2 new). Old path unchanged (server,
STREAM 0/170, PHONE FAST 0/34).
No new photos are claimed as matches: every rule here returns photos from the same stored judge answers whose eye labels
were viewed in RESULTS 34/41/42; only the ORDER and the bound change.
Phone throughput (read-only source analysis of mlx-swift-lm 3.32.3 + the local Qwen3-VL-4B-Instruct-4bit checkpoint;
FLOP estimates, not measured): a call is compute-bound, ~1/3 vision tower (24 blocks x 1024 wide, full attention over
2,352 patches for a 4:3 photo / 3,136 square) and ~2/3 LLM prefill (36 layers, ~7.3 GFLOP per token x ~620 tokens);
image tokens = 588 for a 4:3 photo (896x672 after the aspect-preserving fit; 784 only for square), ~30 text tokens.
prepare() already returns the last token's logits (Qwen3VL.swift:1954; the .tokens branch in Judge.swift never runs).
- Batching several photos per forward pass: not supported (processor builds [1, L]; getRopeIndex assumes batch 1;
  deepstack indexes axis 1 with flat indices; no padding mask; the B=1 chunked path is the one that skips full-vocab
  logits). Even patched: <= 5-10% (prefill at ~300+ tokens per chunk is already compute-bound) for +90 MB KV per photo.
- KV-caching the text prompt across photos: the template puts the IMAGE FIRST (imagesThenVideosThenText), so the shared
  prefix is 4 tokens (<1%); text after the image depends on it (attention, and mRoPE positions offset by the grid).
- Worth doing: (1) fetch/decode the next photo while the GPU judges the current one (PhotoKit fetch is serial in
  Search.swift's loop; size unknown until timed); (2) reuse one photo's image KV for its exclude/filter questions
  (~95% off each extra question; only matters for searches with those questions); (3) fp16 vision tower (0-10%, check
  P(yes) agreement). The only large lever is fewer FLOPs per photo (smaller judge or visual-token pruning, each must be
  re-measured against the oracle); a lower image resolution is excluded by rule.
Device check pending: MAC_INBOX M37 (Release build; one fast search: first match / first bound / done, from the new
"SEARCH round ..." lines in Documents/memcheck.txt).

## 44. Searches against the phone's MEASURED clock and heat wall; image-prefix reuse; prefetch (10-11; eval/tune_stop_time.py heat -> eval/stop_time/heat*.txt)
Question: RESULTS 43 tuned fast mode for 0.67 photos/s. The Mac then measured the real curve (JOURNAL MAC 10-10 15:50 and
16:25, MAC_INBOX M37): photos judged per minute in 5-minute buckets of continuous judging on Reza's iPhone 18 Pro (the
Mac called them "2-minute"; 5 is right: 14 buckets x 5 min = 1,175 photos vs 1,181 logged in the 68-min run, and they
match that entry's 0-5 / 5-10 / 10-15 min rates):
  m4 = shipped checkpoint (bf16 vision tower): 35 26 15 12 12 14 19 18 16 16 18 17 14 3, then iOS halted charging and
       locked the phone at 68 min (the app is suspended: nothing more is judged). Cumulative: 105 photos by 3 min,
       305 by 10, 440 by 20, 755 by 40, 1,169 at the wall.
  m8 = 8-bit vision tower: 30 36 33 24 24 26 24 22 19 21 21 16 6 (log ends at 65 min; taken as its wall). 90 / 330 /
       615 / 1,095 / 1,510.
  fit4 = least-squares fit to m4's bucket means: r(t) = 15.5 + 33.4 exp(-t / 4.5 min) photos/min (0.82/s at t = 0, 0.26/s
       sustained). Same picture as the buckets (rows below). The same fit to m8 degenerates to a near-linear decline,
       so m8 is used as measured. Each clock is ONE run on one phone (n = 1 per checkpoint).
Method: no new judge calls. Same stored answers, truth and 20 seeds as RESULTS 41-43 (audit4: 6 searches on a 7,886-photo
library, eye labels of RESULTS 34; ev16: 192 unseen searches, judge-relative only). Each rule's ordered call sequence is
read off at minute t through the clock; "wall" = everything judged before the halt; "done" = when fast mode ends (inf if
the wall comes first). New rule knobs (Python sim_timed == Swift streamTimed): fastStopCalls (fast mode ends after the
first round that reaches it) and fastStopTopUp (that round's random sample is topped up to this many).
Sweep (heat.txt, 25 rules x 3 clocks x 2 sets; heat_stop.txt): round budget 60/90/120, top-up 20/30, budget 0/240/480/960;
then stop top-up 0/100/150. Findings:
- On the measured clock RESULTS 43's rule never ends for dense searches: 34% of audit4 search-seeds (all of dog and car)
  and 4.3% of ev16 ones are still running when the phone halts (m4; m8: 34% / 3.8%). Fast mode's head rule can run to
  2,000 calls; the phone does ~1,200 before iOS stops it.
- No round schedule wins outright: budget 90 calls / top-up 20 shows the first bound at 3.1 min instead of 4.3 (m4) but
  costs recall at 3 min (audit4 pooled 0.191 vs 0.215, ev16 0.512 vs 0.548) for +0.010 / +0.005 at 10 min; 60 calls
  (bound at 2.3 min) costs more. Kept 120 / 30.
- A call budget is what makes searches end. 480 ends every replay before the wall; 960 still leaves 1% at the wall and
  ends at ~53 min (m4). With 480 alone, the bound shown at the end is read off 30 random photos: beach found 304 of 334
  but showed "+1,204 may be missing". Topping that round up to 150 random photos gives +258 (car +1,656 -> +530, dog
  +2,382 -> +1,396; ev16 final bound p90 +231 -> +90 photos, unbudgeted +72) for ~+60 calls per audit4 search, +18 ev16.
CHOSEN (phoneFastParams): rounds unchanged (120 calls, top-up 30, a1 0.2) + fastStopCalls 480 + fastStopTopUp 150.
[REVISED 10-11 by 44b: the 480-call point is where the FIRST ANSWER is complete, not where the search ends; the search
keeps looking, thermally paced. The "chosen" rows below are the first answer.]
Judge-relative found / judge-yes, pooled | per-search mean (heat_final.txt):
| set, clock | rule | 3 min | 10 min | 20 min | 40 min | at the wall | first bound (med) | done med / p90 | done before the wall |
|---|---|---|---|---|---|---|---|---|---|
| audit4, m4 | RESULTS 43 | 0.215/0.396 | 0.427/0.657 | 0.525/0.730 | 0.676/0.795 | 0.861/0.864 | 4.3 min | 35.7 / wall | 0.658 |
| audit4, m4 | **chosen** | 0.215/0.396 | 0.427/0.657 | 0.525/0.730 | 0.553/0.745 | 0.553/0.745 | 4.3 min | 31.6 / 31.7 | **1.000** |
| audit4, m8 | RESULTS 43 | 0.190/0.362 | 0.448/0.674 | 0.611/0.770 | 0.830/0.853 | 0.940/0.896 | 5.0 min | 22.5 / wall | 0.658 |
| audit4, m8 | **chosen** | 0.190/0.362 | 0.448/0.674 | 0.553/0.745 | 0.553/0.745 | 0.553/0.745 | 5.0 min | 19.4 / 19.5 | **1.000** |
| ev16, m4 | RESULTS 43 | 0.548/0.672 | 0.819/0.805 | 0.885/0.828 | 0.931/0.843 | 0.947/0.848 | 4.3 min | 13.0 / 45.9 | 0.957 |
| ev16, m4 | **chosen** | 0.548/0.672 | 0.819/0.805 | 0.885/0.828 | 0.903/0.834 | 0.903/0.834 | 4.3 min | 13.0 / 31.8 | **1.000** |
| ev16, m8 | RESULTS 43 | 0.504/0.647 | 0.836/0.812 | 0.920/0.839 | 0.945/0.847 | 0.953/0.849 | 5.0 min | 10.6 / 29.4 | 0.962 |
| ev16, m8 | **chosen** | 0.504/0.647 | 0.836/0.812 | 0.903/0.834 | 0.903/0.834 | 0.903/0.834 | 5.0 min | 10.6 / 19.5 | **1.000** |
ev16 share of searches done by 10 / 20 / 30 / 40 min: chosen m8 0.42 / 0.985 / 0.999 / 1.000 (RESULTS 43: 0.42 / 0.836 /
0.906 / 0.936, 3.8% never); chosen m4 0.42 / 0.658 / 0.818 / 0.995 (RESULTS 43: ... 0.869, 4.3% never).
Eye-label recall of the matches on screen (audit4, RESULTS 34 truth, unsure = no match, mean of 20 seeds; eye_m*_heat*.txt):
| clock | rule | 3 min | 10 min | 20 min | 40 min | wall |
|---|---|---|---|---|---|---|
| m4 | RESULTS 41 (old) | 0.203 | 0.385 | 0.461 | 0.611 | 0.797 |
| m4 | RESULTS 43 | 0.203 | 0.368 | 0.449 | 0.599 | 0.785 |
| m4 | **chosen** | 0.203 | 0.368 | 0.449 | 0.475 | 0.475 |
| m8 | RESULTS 43 | 0.180 | 0.383 | 0.533 | 0.753 | 0.864 |
| m8 | **chosen** | 0.180 | 0.383 | 0.475 | 0.475 | 0.475 |
Where each audit4 search ends (median of 20 seeds; chosen vs RESULTS 43; minutes m4 / m8; found of judge-yes):
bicycle 20.8 / 13.6 (both rules, 36 of 48); food 31.6 / 19.4 (both, 153 of 164); sunset 31.6 / 19.4 (both, 168 of 172);
beach 31.6 / 19.4 vs 39.7 / 25.6 (304 vs 310 of 334); car 31.6 / 19.4 vs never (218 of 410 vs 297 at the m4 wall, 321 m8);
dog 31.6 / 19.4 vs never (464 of 1,301 vs 1,129 at the m4 wall, 1,294 m8).
WHAT THE BUDGET COSTS, plainly: nothing for the four sparse searches and nothing before minute ~20 (m8) / ~26 (m4) for any
search; the dense ones stop early: dog eye recall 0.345 instead of 0.839 at the m4 wall, car 0.388 vs 0.527. Those
matches came in minutes 20-68 of a throttled, hot phone that iOS then locked; now the search ends with an honest bound
(dog "+1,396 may be missing") and the note "Stopped after N photos to keep your phone cool; tap Look at everything to go
on" (exhaustive mode reuses every cached answer). Budget 960 would keep dog 0.687 / car 0.495 but end at ~53 min (m4).
Over-claims (every round of every seed, 0 allowed): chosen rule audit4 fast 0 / 360, exhaustive 0 / 920; ev16 fast
0 / 9,139, exhaustive 0 / 18,692 (overclaims_heat.txt). Uniform sample (2,000 seeds, car and bicycle, exhaustive, every
round): mean sample hits = size x the rest's yes rate (car r1 1.109 vs 1.155, r3 3.895 vs 3.954, r7 0.849 vs 0.866;
bicycle r1 0.063 vs 0.066 ...), 0 overclaims (uniform_heat.txt). Why it stays uniform: the budget and the top-up are
decided by the call count, which depends on head labels and permutation positions, never on a label of the rest.
Swift: phoneFastParams fastStopCalls 480, fastStopTopUp 150; golden fixture stream_timed.json + 2 budgeted rules (one
with interleaving), Swift == Python call for call; testFastStopEndsAfterTheSameRounds (rounds before the last identical
to the unbudgeted run; 20 of 30 public runs ended by the budget); testTimedNeverOverclaims 0 of 370 rounds.

THROUGHPUT (code in the app + FindPicsCore; device numbers pending, MAC_INBOX M39):
(a) IMAGE-PREFIX REUSE for a photo's follow-up questions (exclude / filter). mlx-swift-lm 3.32.3 supports it without a
    patch: Qwen3VL.prepare takes a warm cache (cache offset > 0) and continues from the "position anchor" = offset +
    the image's M-RoPE delta it returned in LMOutput.State (QwenVL.continuationAnchor / continuationResumeState); the
    library's own tests assert warm continuation == one cold prefill (Tests/MLXLMTests/ContinuationAssertions.swift,
    Qwen3VLContinuationTests). KVCacheSimple.trim(n) rewinds. Judge.pYes(keepImage:) prefills [header, image tokens,
    <|vision_end|>] once (vision encoder + ~590 tokens), keeps that cache with the CIImage object (held, so no other image
    can reuse its identity), and for each question appends only its ~30-token tail, then trims back. Guards: every
    layer a plain KVCacheSimple (Qwen3-VL; the hybrid Qwen3.5 falls back), prompt tokens start with exactly the kept
    prefix (FindPicsCore.tailAfterPrefix), else the old one-pass call. Search.swift also reads each photo ONCE for all
    its questions (it re-read it from PhotoKit for every follow-up). -checkKVReuse logs split vs one-pass P(yes) per call;
    -noKVReuse = old path.
    Expected: each follow-up ~0.03-0.05 of a call (30 vs ~620 tokens of prefill, no vision tower: FLOP estimate) plus no
    re-read. Applies only to searches with a follow-up: 161 filter + 218 exclude questions in 3,084 planned albums
    (eval/planners: ~5-12% of searches); none of the 192 everyday searches. Per judged photo, cost 1 + y -> ~1.05 + 0.05 y
    (y = yes share of judged photos: ev16 median 0.08 -> ~1.03x; audit4 median 0.36 -> ~1.3x; dense p90 0.93 -> ~1.8x;
    two follow-ups up to ~2.5x). The first question pays ~one extra forward over the weights (~2% est.) only when a
    follow-up is possible.
(b) PREFETCH. FindPicsCore.Prefetcher + streamRounds(lookahead:upcoming:) (the next 2 positions the run expects; 97.6%
    of next photos predicted, 10,386 of 10,645 calls on the public fixtures; the hint never changes the run, tested).
    Search.swift reads them (PhotoKit / video frame at 896 px, same function as before) while the GPU judges. Expected:
    hides the read: on Reza's library ~94% of 896 px reads are iCloud downloads (M4) at ~0.6 s each one at a time (M10)
    -> a cool-phone call of ~1.7 s becomes ~1.1 s (~1.5x) if that 0.6 s is inside the measured 1.7 s; when hot (~4 s per
    photo) ~1.15x; for photos already on the phone ~2-3%. Not measured: the read/model split was never timed on the phone
    (M37(3) open); new log line "PREFETCH ... waited X s for reads, Y s in N model calls"; -noPrefetch = old reads. Heat
    caveat: overlapping keeps the GPU busier, so if the phone is heat-limited the sustained gain shrinks toward zero; the
    gain is in the cool first ~10-15 min, which is where fast mode now spends most of its time.
8-BIT VISION TOWER AS DEFAULT: recommended. Quality equal within rerun noise (RESULTS 38: eye labels 231/45 vs 232/47 at
0.7, 221/33 both at 0.95; P(yes) |diff| p90 0.019; Reza's demo 265 heavier both, 114 vs 115 fit); peak memory 3.70 ->
3.50 GB; measured ~1.5x sustained rate after minute 15 (one run each, M37) -> in this replay fast mode ends by 19.5 min
(p90, ev16) instead of 31.8 (m4). Hosting: today the app downloads mlx-community/Qwen3-VL-4B-Instruct-4bit from Hugging
Face; the 8-bit checkpoint exists only on the Mac (side-loaded). Needs a public download location (Reza's decision): (1)
a Hugging Face model repo under his account, ~2.7 GB, Apache-2.0 allows redistribution with the licence and a note of
the change; the app only changes Judge.visionJudgeCandidateID; (2) later, Apple-hosted Background Assets once the paid
account is active (docs/APP_STORE_RESEARCH.md). Before switching: repeat the 8-bit vs bf16 heat runs from the same
starting temperature (n = 1 each so far).

## 44b. Keep looking after the first answer, paced by the phone's temperature (10-11; eval/tune_stop_time.py pace -> eval/stop_time/pace.txt)
Why: ending dense searches at 480 calls repeats what Reza rejected for the video pass on 10-10 ("capping it caps one of our
goals, which is accuracy"): dog eye recall 0.839 -> 0.345 at the wall. Now the 480-call round is the FIRST ANSWER
(bound with its 150-photo top-up, results on screen, album done), and the search keeps going by itself.
Design (FindPicsCore/ThermalPace.swift thermalPace + ThermalPacer; Search.swift paceGate; AppModel.keepLooking):
- after every album's first answer, each album is run again in exhaustive mode with the same seed: its first ~600
  answers come from the judge's cache (no reads, no model calls), then new photos are judged;
- before every model call after the first answer: ProcessInfo.thermalState .nominal / .fair -> judge; .serious ->
  pause, re-read every 20 s, status "Pausing to let your phone cool; it goes on by itself."; .critical -> stop, status
  says so (searching again skips every photo already checked while find pics stays open). "Stop looking" button; a new
  search cancels it at the current photo (and waits for it, so two searches never share the judge); a generation
  counter keeps late updates of a stopped search out of the next one's results.
- the bound: the continuation runs exhaustive mode's rounds, already replayed for this exact rule at 0 overclaims
  (audit4 0 / 920, ev16 0 / 18,692 rounds; RESULTS 44). The first answer is exactly fast mode's rounds (asserted in the
  replay: the exhaustive call order starts with the fast order, every case and seed; Swift testFirstAnswerIsWhereFastModeStops).
Replay of the pacing. ASSUMPTION, NOT MEASURED (M39 (5) measures it): the first answer runs flat out on the measured curve
(m4 / m8, RESULTS 44); after it, and after the minute the Mac saw throttling start (m4 15, m8 25 min), the phone judges
at its measured hot floor (mean of the later buckets, halting bucket left out: m4 15.6/min, m8 21.3/min) for a share
`duty` of the time and pauses for the rest ("the clock stops while hot"), and the pauses keep iOS from halting it.
duty 1.00 = never needs to pause (optimistic), 0.50 = judge 5 min / cool 5 min, 0.33 = cool twice as long as it judges.
Found / judge-yes (pooled | per-search mean; audit4 6 searches + eye-label pooled recall, ev16 192 searches; 20 seeds):
| set | clock | first answer then STOP (44) | keep looking, duty 1.00 | duty 0.50 | duty 0.33 |
|---|---|---|---|---|---|
| audit4 judge-rel. 20 / 40 / 60 / 90 min | m4 | 0.525 / 0.553 / 0.553 / 0.553 | 0.525 / 0.620 / 0.767 / 0.953 | 0.525 / 0.590 / 0.664 / 0.773 | 0.525 / 0.576 / 0.626 / 0.703 |
| audit4 EYE 20 / 40 / 60 / 90 | m4 | 0.449 / 0.475 / 0.475 / 0.475 | 0.449 / 0.539 / 0.682 / 0.865 | 0.449 / 0.509 / 0.581 / 0.688 | 0.449 / 0.496 / 0.545 / 0.619 |
| audit4 judge-rel. | m8 | 0.553 / 0.553 / 0.553 / 0.553 | 0.561 / 0.772 / 0.952 / 0.970 | 0.561 / 0.700 / 0.796 / 0.939 | 0.561 / 0.673 / 0.739 / 0.837 |
| audit4 EYE | m8 | 0.475 / 0.475 / 0.475 / 0.475 | 0.482 / 0.687 / 0.863 / 0.881 | 0.482 / 0.616 / 0.710 / 0.851 | 0.482 / 0.590 / 0.654 / 0.750 |
| ev16 judge-rel. | m4 | 0.885 / 0.903 / 0.903 / 0.903 | 0.888 / 0.937 / 0.967 / 0.992 | 0.887 / 0.926 / 0.946 / 0.966 | 0.887 / 0.920 / 0.937 / 0.953 |
| ev16 judge-rel. | m8 | 0.903 / 0.903 / 0.903 / 0.903 | 0.914 / 0.967 / 0.990 / 1.000 | 0.914 / 0.954 / 0.971 / 0.989 | 0.914 / 0.949 / 0.961 / 0.976 |
(audit4 pooled judge-relative per-search means are in pace.txt.) For reference, the unpaced RESULTS 43 rule ran into the
halt: eye 0.785 (m4) / 0.864 (m8) at the wall, then nothing more.
Read: under every duty assumed, keeping on looking beats stopping from minute ~20 on, and with the 8-bit tower at duty
0.5 the eye recall at 90 min (0.851) is near the unpaced run's wall value (0.864) without running into the halt, IF
pausing really prevents it. Everything after the first answer depends on two unmeasured things: how long the phone
needs to cool at .serious (duty), and whether pausing at .serious keeps it out of "Charging On Hold". M39 (5).
FindPicsCore swift test 109/109 (+3: thermalPace per state, pacer pause/transition bookkeeping, first answer = where fast
mode stops and the continuation then judges everything).

## 45. Where is the fast (vector) stage good enough on its own? (10-11; eval/eval_vector_enough.py -> eval/vector_enough/; MAC_INBOX M38 -> M40)
Why: on Reza's phone "food photos" took 149 min (4,305 judge calls), "beach" 151 min. Apple Photos answers in < 1 s from
precomputed labels. Every photo already has a PE-Core-B-16 vector; ranking by it takes milliseconds. Question: for which
searches is the vector ranking's FIRST SCREEN already as right as the judge's, can the phone tell before any judge call,
and how few judge calls fix the rest? No new judge calls: stored answers of RESULTS 40-44 (audit4: real phone weights,
judgeCutoff 0.99 / 0.7) and RESULTS 27/41/42 (ev16: Qwen3-VL-4B 16-bit, cut 0.7 everywhere = looser than the phone).
NEW EYE LABELS: 746 photos labelled blind from native-pixel sheets (eval/vector_enough/labels_*.txt; key.json not read
before labelling; 8 labelling subagents on the RESULTS 34 criteria, written down in LABEL_PROTOCOL.txt; I checked 2 sheets
myself (beach_01, food_05, car_09: 18/18 same call)): (a) a CENSUS of every unlabelled photo on the audit4 first screens
(vector top 50 and judged top 50 of the combined library, each library's vector top 20 and judged top 20): 526 photos,
so those screens are 100% eye-labelled; (b) ev16: per search kind 10 + 10 uniform draws from the pooled vector top 50,
split by the judge's answer (220 photos). These labels are NOT added to RESULTS 34's strata (they are not uniform draws).
Fast mode = phoneFastParams (RESULTS 44), its results in the order it finds them; "judged top N" = its first N yes.

(1a) audit4 as ONE 7,886-photo library (as RESULTS 40-44). Precision / recall [95% bootstrap] by RESULTS 34's estimator
(unsure = no match); top 20 / 50 precision from the census below is the reliable one (the estimator imputes stratum
rates there). Calls = judge calls before N judged photos are on screen; minutes on the measured m4 clock (RESULTS 44).
| search | N | vector top N: precision | recall | judged top N: precision | recall | calls (min m4) |
|---|---|---|---|---|---|---|
| dog (1,301 judge-yes) | 100 | 0.98 | 0.073 | 1.00 | 0.074 | 103 (2.9) |
| | 200 | 0.99 | 0.147 | 1.00 | 0.149 | 230 (7.1) |
| | 500 | 0.99 | 0.370 [0.353-0.379] | 1.00 (461 = first answer) | 0.343 | first answer 601 |
| car (410) | 100 | 0.99 | 0.176 | 1.00 | 0.178 | 101 (2.9) |
| | 200 | 0.75 | 0.265 [0.229-0.298] | 1.00 | 0.355 [0.307-0.400] | 414 (17.8) |
| | 500 | 0.49 | 0.433 | 1.00 (217) | 0.386 | first answer 601 |
| bicycle (48) | 50 | 0.54 | 0.519 | 1.00 (37 = first answer) | 0.712 [0.685-0.713] | first answer 450 |
| | 100 / 200 / 500 | 0.31 / 0.17 / 0.08 | 0.596 / 0.654 / 0.750 | 1.00 (37) | 0.712 | |
| beach (334) | 100 | 0.85 [0.54-1.00] | 0.397 | 0.85 | 0.397 | 100 (2.9) |
| | 200 | 0.72 [0.50-0.88] | 0.672 | 0.74 | 0.696 | 236 (7.3) |
| | 500 | 0.41 [0.28-0.51] | 0.952 [0.844-1.000] | 0.66 (304) | 0.937 | first answer 601 |
| sunset (172) | 100 | 0.89 | 0.680 | 0.90 | 0.690 | 116 (3.3) |
| | 200 | 0.63 | 0.961 [0.891-0.975] | 0.72 (169) | 0.926 | first answer 600 |
| | 500 | 0.26 | 0.997 | 0.72 (169) | 0.926 | |
| food (164) | 100 | 0.81 [0.70-0.88] | 0.835 | 0.80 | 0.830 | 120 (3.4) |
| | 200 | 0.48 [0.41-0.57] | 1.000 [0.833-1.000] | 0.57 (153) | 0.908 | first answer 600 |
| | 500 | 0.19 | 1.000 | 0.57 (153) | 0.908 | |
(full table with every interval: eval/vector_enough/report_audit4.txt.) Read: down to N = 100, vector-only and judged are
the same photos and the same precision for dog, car, beach, sunset and food in this pooled library (the judge said yes
to 88-100 of the vector top 100). Past ~N = 200 the judge's filtering is what keeps precision: car 0.75 -> 1.00 at 200,
beach 0.41 -> 0.66 at 500, food 0.48 -> 0.57 at 200 (vector recall there is HIGHER: 1.000 vs 0.908; the judge's 0.99
cut drops some real food). Bicycle is the exception everywhere: vector precision 0.54 already at 50.
(1b) FIRST SCREEN, census by eye, same pooled library. Hybrid = vector top S shown at once; the judge checks the vector top
c; the screen = its yes photos first, then unchecked vector ranks c+1... (c = 0 is vectors only). [95% Clopper-Pearson]
| S | vectors only | c = 25 | c = 50 | c = 100 | c = 200 | judge-verified |
|---|---|---|---|---|---|---|
| 20, all 6 searches | 111/120 = 0.93 [0.86-0.97] | 115/120 = 0.96 | 116/120 = 0.97 | 116/120 | 116/120 | 116/120 = 0.97 [0.92-0.99] |
| 50, all 6 searches | 263/300 = 0.88 [0.83-0.91] | 266/299 = 0.89 | 273/292 = 0.93 | 273/285 = 0.96 | 277/286 = 0.97 | 290/298 = 0.97 [0.95-0.99] |
Per search at 20: dog 20/20 vs 20/20, car 20/20 vs 20/20, beach 18/20 vs 18/20, sunset 20/20 vs 20/20, food 19/20 vs
19/20, bicycle 14/20 vs 19/20 (the whole gap). At 50: bicycle 27/50 vs 47/48, sunset 46 vs 49, food 46 vs 48, dog 48 vs 50.
(1c) BUT ONE PERSON'S LIBRARY IS WHAT THE PHONE SEARCHES. Per library (4 x 6 audit4 searches, census): vector top 20 =
272/480 right by eye (0.57), the judged first screen 318/370 (0.86; it holds fewer photos when a library has fewer than
20 judge-yes). The pooled library hid this: dog is 1,267 judge-yes in one library and 4, 5, 25 in the others.
(1d) ev16, vector-only precision per search kind (16 unseen libraries, pooled over libraries; judge-relative = share the
judge says yes to; eye = yes-share x eye rate(yes) + no-share x eye rate(no) from the stratified sample [95% Jeffreys]):
| search | judge-yes per library (median) | top 20: judge-rel. | top 20: eye est. | top 50: judge-rel. | top 50: eye est. | eye rate in judge-yes / judge-no |
|---|---|---|---|---|---|---|
| car | 176 | 0.88 | 0.79 [0.55-0.87] | 0.78 | 0.70 [0.49-0.79] | 9/10 / 0/10 |
| boat | 44 | 0.76 | 0.79 [0.62-0.84] | 0.61 | 0.65 [0.52-0.74] | 10/10 / 1/10 |
| night | 80 | 0.73 | 0.65 [0.47-0.74] | 0.64 | 0.58 [0.41-0.67] | 9/10 / 0/10 |
| food | 44 | 0.71 | 0.64 [0.45-0.73] | 0.54 | 0.48 [0.35-0.59] | 9/10 / 0/10 |
| beach | 38 | 0.77 | 0.54 [0.30-0.71] | 0.57 | 0.40 [0.24-0.55] | 7/10 / 0/10 |
| flowers | 96 | 0.86 | 0.52 [0.27-0.74] | 0.70 | 0.42 [0.22-0.61] | 6/10 (2 unsure) / 0/10 |
| bicycle | 28 | 0.49 | 0.49 [0.40-0.59] | 0.33 | 0.33 [0.28-0.46] | 10/10 / 0/10 |
| church | 57 | 0.65 | 0.33 [0.16-0.53] | 0.50 | 0.25 [0.13-0.43] | 5/10 (3) / 0/10 |
| dog | 9 | 0.31 | 0.31 [0.26-0.46] | 0.15 | 0.15 [0.13-0.32] | 10/10 / 0/10 |
| sunset | 20 | 0.60 | 0.24 [0.10-0.44] | 0.41 | 0.16 [0.07-0.35] | 4/10 (1) / 0/10 |
| cat | 4 | 0.32 | 0.16 [0.09-0.32] | 0.18 | 0.09 [0.05-0.27] | 5/10 / 0/10 |
| selfies | 2 | no looks in these plans: no vector ranking | | | | |
Read: per library, vectors alone fill the first screen with mostly-right photos only where the library holds many clear
matches (car, boat, night). Two limits that are not the vector's fault: (i) a library with 9 dogs cannot give 20 right
photos (dog 0.31 = 9/20 + misses); (ii) at the 0.7 cut the ev16 judge's own scene answers are often wrong by eye
(sunset 4/10, church 5/10, cat 5/10 incl. 4 wild cats, flowers 6/10), so for those kinds "agrees with the judge" is a weak
target and the judge-verified screen is not much better (below). With 10 draws per stratum these intervals are wide.

(2) THE RULE, computable in milliseconds before any judge call (FindPicsCore VectorFirst.swift):
  vectors are enough for the first screen  <=>  the search is a plain look (object "Is there a real X anywhere ...?",
  subject "Is this a photo of X?" or no question; no person / selfie / moment / exclude / filter; looks present)
  AND the 20th-best in-scope fast score (PE-Core-B-16 mean look - mean avoid) >= T.
Target: >= 18 of the vector top 20 are judge-yes ("enough"; base rate 68/216). 216 searches on 20 libraries (audit4 24 +
ev16 192). Every threshold fitted LEAVE-ONE-LIBRARY-OUT (19 libraries in, 1 out, 20 folds):
| rule (LOLO) | accuracy | said "vectors" | of which truly enough | missed "enough" | vector top-20 judge-rel. when "vectors" |
|---|---|---|---|---|---|
| always judge | 148/216 = 0.685 | 0 | - | 68/68 | - |
| always vectors | 84/216 = 0.389 | 200 | 68/200 | 0 | 0.640 |
| top score s1 >= T | 150/216 = 0.694 | 12 | 7/12 | 61 | 0.825 |
| gap s20 - median >= T | 172/216 = 0.796 | 58 | 41/58 | 27 | 0.893 |
| s20 >= T (max accuracy) | 167/216 = 0.773 | 35 | 27/35 | 41 | 0.926 |
| s20 >= T per kind (object / subject / fact) | 173/216 = 0.801 | 39 | 32/39 | 36 | 0.928 |
| **s20 >= T, safe (lowest T with >= 90% enough in training)** | **172/216 = 0.796** | **32** | **28/32** | **40** | **0.958** |
Chosen: the safe s20 rule (fewest wrong "vectors" calls; per-kind thresholds gain 1 point and need a kind label the
phone does not have for new query types). LOLO thresholds 0.2067-0.2083; fitted on all 20 libraries T = 0.2074 (in the
app). The 4 wrong "vectors" calls: two church, one beach, one car search with 14-17 of 20 judge-yes.
Eye consequences of the LOLO decisions (first screen of 20):
  said vectors: audit4 4 searches, vector screen 78/80 = 0.97 = judge-verified 78/80; ev16 28 searches, eye est. 0.61 vs
  judge-verified 0.64 (both low because 22 of the 28 are church/beach/sunset/flowers searches, see (1d)).
  said judge: audit4 20 searches, vector screen 194/400 = 0.48 [0.44-0.54] vs judge-verified 240/290 = 0.83 [0.78-0.87]
  (the judge shows MORE right photos, 240 vs 194, and 50 wrong instead of 206); ev16 164 searches, eye est. 0.43 vs 0.79.
Honest limits: it says "vectors" for only 15% of these searches (32/216), and misses 40 of the 68 that were enough: the
score that makes it safe is high. Fact queries (night: 7/16 enough) and new phrasings were not separable by one threshold
(night uses 4 looks averaged, a different score scale) -> the app keeps them on the judge (question-form gate).
PER PHOTO, the same score applied to each photo (likelyAtOnce): show at once every vector top-20 photo whose own score
>= t (t = lowest with >= 90% judge-yes above it; LOLO 0.2000-0.2054, all libraries 0.2036): 1,575 of 3,995 top-20 slots
filled at once (159 of 200 searches show >= 1 photo, 41 a full 20); judge-relative precision 1,409/1,575 = 0.89
[0.88-0.91]; eye audit4 census 154/172 = 0.90 [0.84-0.94]; ev16 eye est. 0.64 (by kind: boat 0.90, food 0.90, night 0.87,
car 0.84, bicycle 0.81, dog 0.80, beach 0.70, flowers 0.59, sunset 0.49, cat 0.43, church 0.42 - the last four are
where the 0.7-cut judge itself is 4-6/10 right by eye).

(3) BUDGET for the 168 searches the rule sends to the judge (LOLO). The judge checks the vector head in rank order (as
fast mode does now). "Judge's first screen" = its first min(20, judge-yes) photos. Minutes on m4 / m8 (RESULTS 44):
| judge calls | minutes (m4 / m8) | hybrid screen of 20, judge-rel. precision (mean) | judge's own first screen complete | share of it checked (mean) |
|---|---|---|---|---|
| 0 (vectors only) | 0 | 0.579 | - | - |
| 25 | 0.7 / 0.8 | 0.668 | 72/168 | 0.78 |
| 50 | 1.4 / 1.7 | 0.711 | 106/168 | 0.85 |
| 100 | 2.9 / 3.3 | 0.740 | 123/168 | 0.90 |
| 200 | 6.0 / 6.4 | 0.757 | 137/168 | 0.93 |
- Twenty judge-verified photos (searches with >= 20 judge-yes, 94/168): median 25 calls (0.7 min), p90 78, max 240.
  ev16 alone: median 22, p90 ~60.
- The hybrid's precision stays < 0.8 at any budget because 74 of these 168 libraries hold < 20 judge-yes: a full screen
  of 20 cannot be right. The fix is not more calls: show only the checked photos (and the per-photo "likely" ones) and
  let the screen be shorter.
- Stopping criterion with the honest bound (Completeness.swift, unchanged): the first screen is final when 20 are checked
  yes OR round 1's bound is out (120 head + 30 random = 150 calls, ~4.3 min m4), whichever first: median 60 calls
  (1.7 min), p90 150 (~4.3 min). ~50 calls is "instant-feeling" for 106/168; 150 covers p90; 200 adds 14 searches.

(4) RECOMMENDED SHAPE (numbers above; per one person's library, not the pooled one):
  t < 1 s: rank by vectors (dot products: ms; mmap of the 0.38 GB store, cold ~0.2-0.4 s - ESTIMATE, not measured on the
    phone). Show likelyAtOnce photos (score >= 0.2036) under "Likely matches - checking": 0.89 judge-relative / 0.90 eye
    (audit4) precision, a full screen in 41/200 searches, >= 1 photo in 159/200. When firstScreenDecision says .vectors
    (15% of searches here) the whole first screen is vector-only at 0.958 judge-relative, eye 78/80 = judge's 78/80.
  t ~ 0.7 min (25 calls) median / ~2.2 min (78) p90: 20 judge-checked photos for searches that have them (unchanged
    from today's fast mode: the first match already appears after ~1 call, RESULTS 43). Checked-yes photos move to the
    top with a check mark; checked-no photos leave the "likely" row.
  t ~ 4.3 min (150 calls, m4): the honest bound (unchanged). Then the first answer (480 calls, ~20-30 min) and the
    thermally paced keep-looking (RESULTS 44/44b), unchanged.
  Wording: "Checked" (the AI looked at the photo) vs "Likely" (similar by image search, not yet checked). "Checked" is
  not "right": the judge's checked beach/sunset/food photos are 0.57-0.72 right by eye in the pooled library (1a).
  What this does NOT change: the time to the bound and to the first answer, recall past the first screen, and the 149-min
  food run (that was the keep-looking continuation over 147,600 in-scope photos). Not measured here: whether to run the
  keep-looking pass at all for searches the rule marks .vectors - the deep-rank numbers (1a) say vectors alone keep recall
  (food top 200: 1.000 vs judged 0.908) but at half the precision past rank ~150.
Swift: FindPicsCore VectorFirst.swift (firstScreenDecision, likelyAtOnce, kthLargest, isAnywhereObjectQuestion);
VectorFirstTests (6). App wiring left to the Mac (MAC_INBOX M40).

## 46. Scene searches: why "food photos" is ~half wrong, and the subject cut 0.99 -> 0.999 (10-11; eval/vector_enough/false_positives_scene.txt, eval/judge_cut_sweep/; MAC_INBOX M44)
Why: RESULTS 45 (1a) put whole-set eye precision of the phone's judged answers at food 0.57, beach 0.66, sunset 0.72 vs
dog/car/bicycle 1.00. No new model runs: stored real-phone-weight P(yes) (eval/recall_audit/scores_q3vl4b_real) + the
RESULTS 34 blind labels + the RESULTS 45 census. Unsure = no match unless stated.
(1) WHAT THE WRONG ONES ARE (every labelled judge-yes / eye-no photo viewed at native pixels; I checked 2 sheets, 8/8
same call; labelled set leans to top ranks): food 38/118, beach 36/126, sunset 21/114 not a match. Kinds: the subject's
SETTING without the subject 40/95 (diners with incidental plates, fried-chicken storefront, puppies eating, beach-trip
people with no beach in frame); look-alike scenes 30/95 (rocky coast / lake shore as beach, golden-hour light as
sunset); label strictness: of 45 'unsure' judge-yes photos 15 are matches and 13 borderline (no_match: 0/50 flipped).
All three questions are the subject form "Is this a photo of X?".
(2) THE CUT IS IN THE WRONG PLACE. Eye precision by stored P: food 0/10 in [0.99, 0.999), 1/13 in [0.999, 0.9999),
65/68 at P == 1.0 (fp32); beach 0/10, 0/7, ... 86/92 at 1.0; sunset 5/18, 6/10, 14/16, 45/47, ... . Whole judged set,
RESULTS 34 estimator over P-bin cells (each stratum split by P; census exact), bootstrap 95%:
| cut (logit gap) | beach P / R | sunset P / R | food P / R |
|---|---|---|---|
| 0.99 (4.6), was | .62 [.55-.74] / .995 | .74 [.65-.83] / .921 | .60 [.54-.65] / .931 |
| 0.999 (6.9), NOW | .67 [.59-.78] / .995 | .83 [.73-.93] / .884 | .68 [.60-.72] / .931 |
| 0.9999 (9.2) | .72 / .995 | .91 / .802 | .78 / .922 |
| 0.99999 (11.5) | .76 / .990 | .91 / .613 | .87 / .912 |
| exactly 1.0 (>= 17.5) | .88 / .976 | .81 / .118 | .94 / .733 |
Object questions keep 0.7: dog/car/bicycle precision .98-1.00 there; 0.99 would cost car ~112 of 564 real photos.
The per-subject optimum differs (beach ~1.0, food ~0.999999, sunset 0.999-0.9999), and the planner writes any subject,
so one modest global step: 0.999. Church / flowers (older labels, simulated weights) lose 6/28 and 1/23 at 0.999.
No cut fixes beach: 28 of its 230 P == 1.0 photos are wrong with full certainty -> question wording (next).
(3) FAST MODE AT 0.999 (eval/judge_cut_sweep/fastmode_replay.txt; phoneFastParams, audit4, 20 seeds): the stop rule
fires at the same head for all three; first answer pooled precision .672 -> .734, recall .958 -> .950 (sunset .931 ->
.916 the largest drop). With unsure = match: sunset .747 -> .633, food -.032, beach -.016. The shown recall lower bound
drops by up to .036 (sunset). ev16 cannot be replayed (no stored P).
(4) PRECISION OF P ON THE PHONE: Judge.swift pOf = py/(py+pn) from an fp32 full-vocabulary softmax, = sigmoid(logit
gap). Stored gaps sit on a 0.25 grid (bf16 logits); fp32 P saturates at gap 17.5; 0.999 = gap 6.9 is far from both.
A logit-gap rule is the cleaner form (and skips the softmax); phone-vs-cluster logit agreement is not measured.
ADOPTED 10-11: FindPicsCore JudgeCutoff.subjectPhotoAccept 0.999 (JudgeCutoffTests), eval/score_apple_photos.py default.
Not measured: EnsembleJudge (also gets the strict cut on the phone), church/flowers/cat on real weights.

## 47. Scene question wording: beach gets its own question, food a stricter cut (10-12; eval/eval_scene_questions.py -> eval/scene_questions/; MAC_INBOX M44 (b))
Why: RESULTS 46 left beach unfixable by any cut (28 of its 230 P == 1.0 photos wrong). Can another WORDING of the subject
question fix food / beach / sunset? Public DISBench only. Judge = Qwen3-VL-4B at the phone's real 4-bit weights, the
exact pipeline of eval/recall_audit/scores_q3vl4b_real (native ~500 px, P = py/(py+pn)); every wording on ALL 7,886
photos of the 4 RESULTS 34 libraries (+726 calls on RESULTS 35 (2) labelled photos in other libraries): 142,674 calls.
Pipeline check: re-judged q0 vs stored P, keep decision at 0.999 identical for 7,883/7,886 (beach), 7,884/7,886 (food),
7,883/7,886 (sunset). Truth = RESULTS 34 estimator + R45 census (exact stratum) + 401 NEW blind eye labels (V3,
labels_extra.txt: every photo any best-cut rule or q0>=0.999 kept without a label, + 79 decoys, labelled before the key
was read). 88-94% of every kept set in the tables has an exact eye label; recall still rests on the RESULTS 34 strata.
Unsure = no match. All rules of one search scored jointly (one shared total; finer cells than RESULTS 46, so q0 moves a
little vs RESULTS 46). Full tables: eval/scene_questions/report_tables.txt; per-photo P: p.tsv.
Wordings (X = food / a beach / a sunset): q0 "Is this a photo of X?" (planner subject form) | g_main "Is X the main
subject of this photo?" | g_vis "Is X clearly visible in this photo?" | f_meal "Is food (a meal, dish or snack) the main
subject of this photo?" | b_sandsea "Is a beach (sand next to the sea) clearly visible in this photo?" | others in
report.txt. "& not people" = AND "Are people the main subject of this photo?" < 0.5.

(1) PER SEARCH (V3; precision [95% bootstrap] / recall [95% bootstrap] (95% Bayesian); n = kept of 7,886)
FOOD (est. 105.6 real)
| rule | n | precision | recall |
|---|---|---|---|
| q0 >= 0.999 (was) | 144 | .70 [.68-.72] | .957 [.892-.978] (.490-.916) |
| q0 >= 0.9999 (ADOPTED) | 125 | .80 [.77-.81] | .947 [.872-.967] (.479-.894) |
| g_main >= 0.999 | 90 | .94 [.91-.95] | .805 [.738-.822] (.406-.757) |
| f_meal >= 0.99 | 107 | .86 [.83-.87] | .871 [.805-.886] (.443-.827) |
| g_vis >= 0.9999 | 121 | .75 [.72-.76] | .862 [.798-.878] (.437-.818) |
| q0 >= 0.9999 & not people | 88 | .88 [.84-.88] | .729 [.670-.743] (.367-.685) |
q0 0.999 -> 0.9999 drops 19 photos (exact labels: 10 no_match, 8 unsure, 1 match). g_main drops 16 real food photos
(viewed: dumplings, bagel with coffee, fish counter, buffet, BBQ, a woman eating from a full plate): stricter than the
label protocol, not just on the failure kinds.
BEACH (est. 219.1 real)
| rule | n | precision | recall |
|---|---|---|---|
| q0 >= 0.999 (was) | 308 | .69 [.68-.70] | .968 [.932-.973] (.658-.937) |
| q0 >= 0.9999 | 284 | .75 [.73-.76] | .968 [.930-.973] (.656-.935) |
| g_vis >= 0.999 (ADOPTED) | 245 | .85 [.83-.86] | .949 [.910-.953] (.641-.914) |
| g_vis >= 0.9999 (best F1) | 233 | .87 [.85-.88] | .926 [.887-.929] (.625-.891) |
| b_sandsea >= 0.999 | 224 | .88 [.86-.89] | .899 [.861-.903] (.605-.864) |
| g_main >= 0.7 | 246 | .83 [.81-.84] | .931 [.895-.938] (.631-.902) |
| q0 >= 0.9999 & not people | 201 | .75 [.74-.77] | .689 [.664-.698] (.468-.671) |
g_vis >= 0.999 vs q0 >= 0.999: drops 67 (40 no_match, 21 unsure, 6 match), adds 4 (2 match). What it drops (viewed) is
the RESULTS 46 failure set: open sea with boats and no shore, palm crowns, a parasail, chalk cliffs, rocky coast, lake
boats. "Sand" wordings lose real pebble / dark-sand shores (b_sandsea 15 real vs g_vis 6). The people veto is useless.
SUNSET (est. 134.1 real): q0 >= 0.999 (KEPT) 146 kept, .79 / .857; g_main >= 0.99 .81 / .902 (net +6 real of ~134,
within noise); g_vis >= 0.999 .94 / .619; the spelled-out sunset wordings are worse than q0 at every cut.
OTHER LIBRARIES (RESULTS 35 (2) labels, right kept / wrong kept at 0.999, q0 | g_main | g_vis): food 10/9 | 7/2 | 9/11;
beach 14/0 | 8/0 | 13/1; sunset 4/1 | 4/0 | 3/0; flowers 18/2 | 7/0 | 17/3; cat 11/0 | 7/0 | 11/4; church 17/1 | 11/0 |
20/0. g_main loses ~half the real flowers / church / cat; g_vis keeps more wrong cats and food. Neither generic template
is safe for every X -> no generic planner rule; per-word exceptions only, for the 3 measured words.

(2) FAST MODE REPLAY (eval/fastmode_cut_replay.py --r47 -> eval/scene_questions/fastmode_r47_tables.txt; Slurm CPU job
52319733; no model runs). Shipped phoneFastParams, audit4 (one 7,886-photo library), PE-Core-B-16 rank, 20 tail seeds,
V3 eye labels, each first answer scored on its own (so the totals differ slightly from (1)). today = q0 at 0.999 for
beach / sunset / food (stored P); adopted = beach g_vis at 0.999 (P from p.tsv, via the new FINDPICS_P_OVERRIDE in
tune_stop_rule.audit4), food q0 at 0.9999, sunset unchanged. Means over 20 seeds:
| search | rule | judge-yes | head rule fires at | head at 1st answer | calls | found | eye P | eye R | R (unsure=match) | shown recall lower bound | exhaustive P / R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| beach | today | 309 | 600 | 451 | 601 | 288.4 | .72 | .954 | .867 | .499 | .69 / .986 |
| beach | adopted | 245 | 550 | 451 | 601 | 240.0 | .85 | .935 | .787 | .482 | .85 / .967 |
| food | today | 146 | 450 | 450 | 600 | 137.2 | .73 | .991 | .646 | .329 | .69 / 1.000 |
| food | adopted | 125 | 450 | 450 | 600 | 121.1 | .82 | .981 | .615 | .313 | .80 / .990 |
| sunset | both | 147 | 450 | 450 | 600 | 144.2 | .80 | .878 | .651 | .347 | .78 / .877 |
Pooled (3 searches): precision .743 -> .826, recall .940 -> .929; 1,801 judge calls either way; minutes to the first
answer unchanged (31.6 m4 / 19.4 m8). dog / car / bicycle identical (sanity).
FLAGS: first-answer eye recall drops are beach -0.019 and food -0.010, under the 0.03 flag. FLAGGED: beach's head rule
alone would fire earlier (head 600 -> 550: fewer yeses at the new question), but the first answer ends at the 480-call
budget at head 451 under both rules, so the first answer's calls and head are unchanged; it only moves where later rounds
switch from the head to random sampling (44b continuation: beach eye recall at 40 / 90 min .954 / .967 -> .935 / .953,
precision .71 -> .85). With unsure = match the recall cost is larger (beach -0.079, food -0.031): the new rules drop
mostly 'unsure' photos (beach-trip close-ups, sand details), which a user may count as beach / food photos.

(3) ADOPTED 10-12 (code; both languages identical):
- Grounding: an album judge question "Is this a photo of (a/the) beach(es)?" (any photo word) becomes "Is a beach
  clearly visible in this photo?" (converse.ground BEACH_QUESTION; Grounding.swift beachSubjectQuestion). Anchors,
  filter / exclude questions and longer beach questions ("a beach trip", "a sunset on a beach") are untouched. Parity
  fixture regenerated: 4 of 2,300 plans change, all "Is this a photo of a beach?" -> the new question; Swift
  GroundingTests.testMatchesPython 0 mismatches.
- Cut: JudgeCutoff.swift judgeCutoff / eval/score_apple_photos.py judge_cutoff: "Is this a photo of food?" -> 0.9999
  (foodSubjectAccept; FINDPICS_FOOD_CUT replays), the beach question -> 0.999 (isSubjectPhotoQuestion includes it, so
  VectorFirst eligibility is unchanged for beach), every other subject 0.999, object / other questions 0.7.
- Tests: tests/test_converse.py test_beach_subject_wording_and_scene_cuts; JudgeCutoffTests.testSceneWordExceptions;
  GroundingTests.testBeachSubjectWording.
Not measured: any other scene word (mountains, snow, lake, city at night...): the table must not grow by guessing. One
set (4 Flickr libraries, ~500 px). Phone P vs cluster P near 1.0 still unmeasured (RESULTS 46 (4)); a 0.9999 cut needs
float32 P (Judge.swift pOf is fp32) or a logit-gap rule. EnsembleJudge also gets these cuts on the phone (strictSubjects).

## 48. Learned head + cosine BLEND as fast mode's ORDER (10-12; eval/learned_head/scale_report.txt (M48-scale), eval/learned_head/{ship_replay,head_ship}.py -> eval/learned_head/scale/ship_*; FindPicsCore HeadRanker.swift)
Why: fast mode asks the slow phone judge (~2-4 s a photo) in rank order; a better order reaches the same matches with
fewer judge calls. M48 trained a small network ("head") on stored PE-Core-B-16 photo vectors + the judge question's text
vector to predict the judge's P(yes). Public DISBench / Open Images only; every number below is a replay of STORED judge
answers (only the order changes), shipped stop rule (phoneFastParams = rank_replay SHIPPED), 20 tail seeds.
Head = MLP on [img, txt, img*txt, |img - txt|] (4,096) -> 1,024 -> 256 -> 1, trained on teacher soft labels.

(1) M48-SCALE (scale_report.txt; 11,661 generated questions x 400 photos = 4.66M phone-judge labels, ~13 productive A100-h)
Head ALONE on unseen everyday concepts stays BELOW cosine at every size (prior suppression of neighbourhoods it only saw
as "no": boat AP .21 vs .78); it beats cosine on unseen planner questions whose neighbourhood is covered. Teacher-relative
macro AUC / AP / P@R.8 (3-seed means); eye = audit4 pooled precision / recall at the judge's own count:
| run | EV16 head | A4 head | eye head | JD held-out top-24 AUC cos / head |
|---|---|---|---|---|
| cosine | .944/.692/.463 | .967/.748/.586 | .814/.791 | .634 |
| N2400 / N5000 / N10000 (generated Q) | AP .595 / .602 / .617 | AP .709 / .716 / .709 | .770/.740, .787/.760, .788/.766 | .716 / .728 / .736 |
| JDonly (2,412 real planner Q + OI15) | .954/.736/.496 | .968/.742/.608 | .808/.785 | - |
BLEND = z(cosine) + z(logit head), per search, equal weights fixed before testing (blend.py): better than cosine on every
test. EV16 AUC/AP/P@R.8 | A4 | A4 eye P / R:
| run | EV16 | A4 | eye |
|---|---|---|---|
| cosine | .944/.692/.463 | .967/.748/.586 | .814/.793 |
| N10000 blend | .956/.724/.510 | .975/.792/.647 | .847/.825 |
| JDonly blend | .961/.763/.559 | .979/.796/.637 | .848/.822 |
| phone judge | 1 | 1 | .933/.909 |
As fast mode's order (rank_replay.py, 3-seed-mean head, z pooled over libraries): judge calls to reach the cosine order's
first-answer found: audit4 3,453 -> 2,306 (JDonly blend, -33%), 2,657 (N10000 blend, -23%), head alone 4,393 (+27%); ev16
without selfies 70,687 -> 45,911 (-35%), 54,907 (-22%), 88,429 (+25%). Verifier: no kind qualifies. Raw checks viewed
(scale_report): boat_head (models above real yachts), car_blend_gain (6/9 clearly cars).

(2) WHAT SHIPS AND THE PHONE COST (decision)
The question is fixed for a search, so the head's first layer folds once per search: W1 [img | txt | img*txt] x =
(W_img + W_prod diag(txt)) img + (W_txt txt + b1) = ONE 1,024 x 1,024 matrix per photo. |img - txt| is elementwise and
nonlinear in img: it cannot be folded and costs a second 1,024 x 1,024 product per photo.
| head | file (fp16) | per photo | 187,084-photo library |
|---|---|---|---|
| JDonly (B = 4, with |img - txt|) | 8,919,844 B | 2.36 M MAC (4.7 MFLOP) | 0.88 TFLOP |
| JDonly-na (B = 3, retrained: same script, data, seeds; train_scale.py `JDonly-na`) | 6,822,692 B | 1.31 M MAC (2.6 MFLOP) | 0.49 TFLOP |
Both are one Accelerate sgemm pass over the stored vectors, on the order of a second per search (estimate, not measured on
the device); the judge calls this saves are ~2-4 s EACH. Replay (z per replay case = the phone, shipped baselines; seeds
0/1/2 of each head, same 20 tail seeds; `-r` = a re-run of JDonly reproduces it exactly):
| plan | head | audit4 calls to match (3,453) | ev16 calls to match (70,687) |
|---|---|---|---|
| M48 replay plan (food 0.9999, old beach) | JDonly s0 / s1 / s2 | 2,503 / 2,444 / 2,318 (-28 / -29 / -33%) | 47,459 / 49,116 / 48,683 (-33 / -31 / -31%) |
| same | JDonly-na s0 / s1 / s2 | 2,698 / 2,804 / 2,611 (-22 / -19 / -24%) | 48,908 / 48,249 / 48,119 (-31 / -32 / -32%) |
| RESULTS 47 plan | JDonly s0 / s1 / s2 | 2,723 / 2,599 / 2,299 (-21 / -25 / -33%) | 48,076 / 50,300 / 49,406 (-32 / -29 / -30%) |
| same | JDonly-na s0 / s1 / s2 | 2,880 / 2,882 / 2,830 (-17 / -17 / -18%) | 49,551 / 49,032 / 48,760 (-30 / -31 / -31%) |
DECISION: keep |img - txt| (ship JDonly). On ev16 the variant is within seed noise; on audit4 it is worse than every
JDonly seed in both plans (best variant seed -24% vs worst JDonly seed -28%; -18% vs -21%), i.e. outside the noise. The
~0.4 TFLOP it saves per 187k-photo search is worth far less than the judge calls it gives back.
Shipped = JDonly seed 0 (weights_JDonly_mlp.pt), the file the M48 report named. It is NOT the scale report's number: that
was the 3-seed ENSEMBLE with pooled z (-33% / -35%); single seeds s0/s1/s2 give -28/-29/-33% (audit4) and -33/-31/-31%
(ev16, phone z); seed 0 is the weakest on audit4. An ensemble would be 3 x 8.9 MB (> 10 MB: hosting) and 3x the compute
for +6 (M48 plan) / +11 (RESULTS 47 plan) points over seed 0 on audit4 (6 searches) and +1 to +2 on ev16 (176 cases):
not now.

(3) THE RESULTS-47 PLAN (eval/learned_head/scale/ship_replay_r47_*.txt; head_ship.py rescore -> scores_*_r47.parquet)
Beach asks "Is a beach clearly visible in this photo?" (judge P from eval/scene_questions/p.tsv g_vis via
FINDPICS_P_OVERRIDE, AND the head reads that question's text; cosine 0.957 to the old question's vector), food at 0.9999.
ev16 stores only 0.7 yes sets of the old questions, so there only the head's beach text changes. Shipped head (seed 0),
phone z:
| set | baseline order | baseline calls / first-answer recall | blend calls to match | blend first-answer recall | 80% / 90% of judge-yes in rank order |
|---|---|---|---|---|---|
| audit4 (6 searches, 1 x 7,886-photo library) | shipped (B-16 on originals) | 3,453 / .537 | 2,723 (-21%) | .542 | 3,754 -> 3,248 / 6,498 -> 5,451 |
| audit4 | phone's B-16 cosine (phone-simulated vectors) | 3,453 / .534 | 2,532 (-27%) | .542 | 3,964 -> 3,248 / 6,725 -> 5,451 |
| ev16 w/o selfies (176 library x search cases) | shipped (PE-Core-L-14 server order) | 70,687 / .911 | 48,076 (-32%) | .918 | 30,517 -> 25,981 / 51,402 -> 44,211 |
| ev16 w/o selfies | phone's B-16 cosine | 72,003 / .902 | 41,544 (-42%) | .918 | 32,076 -> 25,981 / 54,808 -> 44,211 |
Per search, audit4 vs the phone's cosine (calls to match): beach 601 -> 607 (no gain: the new, stricter yes set is the part
cosine already ranks first; with the old question beach went 601 -> 399), food 600 -> 360, bicycle 450 -> 239, car 601 ->
418, dog 601 -> 554, sunset 600 -> 354. So the RESULTS 47 plan costs the blend ~6 points on audit4 (-28% -> -21% vs
shipped), all from beach; ev16 unchanged (-33% -> -32%). The earlier ev16 baseline was the server's L-14 order; against the
order the phone actually uses (B-16) the saving is larger (-42%).
z scope: the phone z-scores over ONE library's in-scope photos. On ev16 (single-library cases) per-library vs pooled z:
48,076 vs 47,984 (no difference). audit4 replays 4 libraries as one library; z-scoring each sub-library separately there
is NOT what the phone does and loses (dog recall .36 -> .05: one library holds most dogs), shown in ship_replay_*.txt as
`perlib` audit4 rows: ignore them.
Phone text tower is int8 (pe_core_text_..._int8): simulated (every text weight > 2,048 elements, symmetric per output
channel) cosine to fp .99987-.99992 on all 13 questions; head logit Spearman .9995-.9999, max |d logit| .10-.20; replay
audit4 2,710 (-22%), ev16 48,092 (-32%): same as fp.

(4) SHIPPED CODE (phone) AND PARITY
- File: ios/FindPicsApp.swiftpm/Sources/Models/learned_head.fphead (8,919,844 B, in git; sha256 0b87cc44...638f7e).
  Layout (head_ship.py docstring): "FPHEAD01", u32 version/D/B/H1/H2/metaLen, provenance JSON, pad 16, W1 f16 [1024][4096]
  with the training standardisation folded in (W/sd; b1 - W.mu with the fp16-rounded W), b1 f32, W2 f16 [256][1024], b2,
  w3, b3 f32. Export check vs the torch fp32 model on all 47,316 audit4 rows: max |d logit| 0.0013, Spearman 1.000000,
  top-200 identical in all 6 searches.
- Blend rule on the phone, per search: cos = lookScores of the in-scope photos (unchanged; VectorFirst keeps using these
  raw scores), head = FoldedHead.itemLogits (max over a video's frames, like lookScores; untested: DISBench has no videos),
  clipped to +-13.8155 (= logit of 1e-6, as the cluster); key = (cos - mean)/std + (head - mean)/std, population std over
  the finite values of THAT search's in-scope photos; std 0 -> 1; a photo without a vector stays last.
- FindPicsCore/HeadRanker.swift: HeadRanker(contentsOf:) / fold(question:) -> FoldedHead; FoldedHead.itemLogits(units:
  unitItem:nItems:block:gemm:) and logits(_:count:into:gemm:); zScores; blendScores(cosine:headLogits:). Matrix products
  go through a GemmNT closure (default MatrixMath.gemmNT = cblas_sgemm on Apple, a plain loop on Linux).
- Golden test HeadRankerTests.testGoldenParityWithPython: the app's head file on 3 audit4 searches (car, bicycle, beach
  with the RESULTS 47 question) x 500 public audit4 photos (fp16 vectors, as the index stores them; near-ties < 2e-4 in a
  top-201 replaced so float32 vs float64 cannot swap an order); Python reference = head_ship.HeadFile (float64).
  RESULT (Linux swift test, plain-loop gemm, float32): max |d cosine| 3.5e-7, |d head logit| 4.7e-6, |d blended key|
  9.3e-6 (bar 1e-3); top-200 order identical in all 3 searches. testFoldedEqualsUnfoldedWithAbsBlockAndCustomGemm: a
  tiny B = 4 head written in the file format, folded == unfolded network (incl. a video = max over frames, a photo
  without a vector = -inf), and a custom GemmNT closure is used for every product (3 per block). swift test 130/130
  (Slurm 52411005; then the known exit segfault in isBareSelfieRequest); pytest 137/137 (Slurm 52410845).
- Python: tests/test_head_ship.py (file round trip + fold == unfolded for B = 3 and 4; blend rule). Not wired into the
  server engine: its indexes are PE-Core-L-14 (the head reads B-16 vectors), and the cluster's fast-mode replays take
  scores from parquet; head_ship.py is the reference implementation and fixture generator.
Not measured: on-device time and memory (HeadRanker holds W1/W2 as Float32, ~18 MB, + 8 MB per folded search); person /
face-set searches, videos, and any query outside these 18 everyday searches with the blend; seed choice (seed 0 fixed in
advance, the weakest on audit4).

## 49. Complex-query benchmark, dev: code fixes for the dev failure kinds (A) and planner training on TRAIN truth programs (B) (10-12; eval/complex_bench/{plan_system,run_system,compare,sft_from_truth}.py, train_cb{,2}.sh -> eval/complex_bench/runs/cmp/)
Test split (20 libraries) untouched: not built, read, trained on or scored. TRAIN = training data only; DEV (350 requests,
13 libraries) = selection. The judge is Qwen3.5-9B in every run; only the planner and the code change.

HOW RUNS WERE COMPARED (new: the old protocol could not separate a fix from noise)
- The plan stage runs alone (plan_system.py); every planner then meets the same 9B judge stage (run_system.py --plans).
- Same model output wherever the prompt did not change (--reuse of the earlier run's raw outputs): two runs of IDENTICAL
  code + the 9B gave different final plans on 62/350 requests (vLLM output depends on batch composition / GPU type).
- Judge answers cached per (photo pixels, question) and shared by every run (--cache): a request whose plan did not
  change gets exactly the same answer set, so "A - base" is the effect of the code alone.
- Noise, for reading the table: the old code re-run vs the original baseline (runs/dev_all): ALL +0.003 [-0.007, +0.015]
  (95% bootstrap over requests), 13 requests better / 13 worse by > 0.05; f moved +0.031 and h +0.022 by noise alone.

RESULT (final-round F1 vs the candidate answer sets; c, g, h are provisional truth; runs/cmp/compare_all.txt)
  planner:          9B       9B       9B       old 4B    old 4B   old 4B   B1 4B    B1 4B    B2 4B    B2 4B
  code / weights:   orig     rerun    A        old code  A        A 4-bit  A        A 4-bit  A        A 4-bit
  a objects (61)    .771     .773     .773     .764      .764     .739     .714     .738     .728     .748
  b attributes (62) .637     .633     .633     .658      .658     .662     .632     .637     .635     .631
  c person (50)     .467     .471     .471     .498      .498     .505     .500     .465     .484     .498
  d time/place (64) .587     .575     .598     .707      .715     .687     .658     .651     .691     .685
  e multi-hop (17)  .599     .598     .598     .441      .441     .481     .604     .533     .626     .604
  f vague (33)      .545     .576     .576     .618      .618     .606     .596     .655     .564     .581
  g two albums (36) .663     .668     .706     .691      .706     .713     .731     .743     .714     .726
  h follow-ups (27) .451     .473     .524     .515      .515     .467     .494     .516     .528     .562
  ALL (350)         .605     .608     .620     .640      .644     .634     .627     .631     .633     .641
  ALL verified(237) .642     .643     .649     .677      .679     .668     .653     .662     .663     .667
  "Baseline" (report.txt) = column 1: it plans with the 9B. "Old 4B" = the distilled planner adapter of RESULTS 31, meant
  for the phone (MAC 10-12 19:55: it had never been transferred - the app planned with the BASE 4B, not measured here -
  and M59 bundles it now).

A. CODE FIXES (Python converse.py / engine.py + Swift Grounding / TextDates / Search; on origin as a67b1f7, eff7628)
  1. No future dates: a time phrase without a year whose dates land after today moves back whole years (the planner wrote
     this year's November in July). "November 27th" / "the 27th of November [2012]" are now computed in code.
  2. "N years ago" inside a moment phrase ("the day we hiked in Denali two years ago"): when the rest of the phrase has no
     calendar word, the dates are that period (that year), not exactly today minus N years.
  3. "add X too / also include X / add X back": the new thing becomes an ALTERNATIVE (one "or" question) to the album's
     latest condition (its filter if it has one), never one more AND condition.
  4. "actually show me all ...": a filter carried over unchanged is dropped; "all the <X> photos" with the album's
     question still the narrowed one -> X alone; "... from that day/night/trip" -> anchor = the album the person was
     looking at, window = that day / week / trip.
  5. Places: words naming a KIND of place (city, county, region, ...) become optional when the full name matches no photo
     ("New York City" vs "..., New York County, New York"); a judge question that only names the album's own place ("Is
     this a photo of New York City?" with place New York City) is dropped (GPS answers it; the judge kept 0 of 133).
  Parity: grounding fixtures regenerated: 16 of 2,300 grounded plans change, all dates (13 future dates moved back, 3
  month-day phrases now computed in code); new follow-up fixtures 874 (866 distillation follow-ups + 8 dev-kind cases; 10
  changed by the rules, 3 of them distillation follow-ups); dates fixture +78 month-day cases (0 old answers changed).
  Swift GROUND 0/2,300 and FOLLOWUP 0/874 mismatches; swift test 134/134 XCTest + 26 swift-testing; pytest 142/142.
  30 scripted conversations on the first A commit: 9B 30/30, old 4B LoRA 30/30, old 4B LoRA on 4-bit weights 30/30
  (the place-question rule came after; it changes 0 of the 2,300 fixture plans; B's gates ran on the final code).
  Dev, 9B planner, everything else identical: ALL .608 -> .620, +0.012 [+0.004, +0.022]; 8 requests better by > 0.05,
  0 worse (one -0.048). Fixed: "photos from the 7th of December vs the 21st" 0 -> .81 and "our Cuba trip in late May"
  0 -> .66 (future dates); "the day we hiked in Denali two years ago" 0 -> .31 and "the day we went to Niagara Falls three
  years ago" 0 -> .46 (were one exact day); "the parade || just the float || add the ones with the singer too" 0 -> .89
  (was float AND singer: nothing); "sea lions on the dock || only crowded || actually show me all the sea lion photos
  from that day" .24 -> .78; "New York City versus Boston" .30 -> .51, "Bali photos from August and the Lombok photos
  from September" .63 -> .98, "photos from Slab City" .96 -> 1.00 (place questions). Regressed: "photos of dinner || just
  the ones at the winery || actually include the restaurant in Murphys too" .05 -> 0 (the OR joined a place-like filter
  onto an album whose place "the winery" matches nothing). Not fixed: "the concert on November 27th" (truth 2012; the
  latest past Nov 27 is 2013 - a year-less date means the latest one by convention, wrong here; train shows ~1 in 4
  year-less dates meant an older year); "the wedding three years ago" (the planner drops the wedding: a planner error).
  With the distilled 4B planner the same code changes ALL only +0.003 (3 better, 0 worse): it rarely made these errors.

THE DISTILLED 4B PLANNER WAS ALREADY BETTER THAN THE BASELINE'S: old 4B LoRA .644 vs 9B .620 on the same code and judge
(+0.024 [+0.001, +0.044]; d +0.117 [+0.050, +0.190], e -0.157 [-0.306, -0.038]). So B is judged against the old 4B LoRA.

B. PLANNER TRAINING ON TRAIN TRUTH PROGRAMS
  Data: eval/complex_bench/queries_train_cb/train.json = 486 TRAIN requests from 18 of 24 train libraries (build.py
  FP_CB_ONLY_SPLITS=train; the 320 earlier ones byte-identical; 6 libraries' oracle jobs were still running - not waited
  for). A truth program knows the library (exact days, full address names, anchor photo ids), so it is NOT copied: the
  teacher (Qwen3.5-27B-FP8, the distillation data's teacher) writes the plan from the planner's own prompt (build_prompt:
  words, today, people, the current plan for follow-ups) with the program as a hidden answer key and rules for what may
  come from the key (structure, wording) and what may not (dates, days, places the words do not state). Training row =
  the planner prompt WITHOUT the key -> the teacher's JSON. Each target is grounded as the product does (ground +
  follow_up_edits) and checked in code against the key (album count, person, anchor + window, exclusion, "or" for
  need_any, dates containing the key's, place words said and in the key's address); a conversation stops at its first
  failing turn. Rows: data/public/distill/cb_train{,_v2}/ (not in git).
  Recipe = RESULTS 31: LoRA r16/alpha32 on the text layers, lr 1e-4, 2 epochs, accumulation 8; data = the 4,426 old 27B
  distillation rows + the new rows x3 (the 30 gate conversations are in neither). 4 s/step on an RTX PRO 6000 with fla
  kernels (overlay .cache/fla_overlay; was 17.6 s on an A100 without them - GPU and kernels both changed): 1,461 steps in
  1.6 h; the script now checkpoints every 50 steps and resumes.
  B1 (first rules): 472 of 558 turns pass (a 73, b 79, c 66, d 68, e 14, f 30, g 38, h 104). Fails: place word not in the
  key's address 35 ("Santorini" vs Thira), dates not containing the key's 25 (year-less "June 27th" whose truth is an
  older year), no valid output 19, time words but no dates 13. Eye check of 25 distinct passing targets (words + key +
  grounded plan): 22 right, 3 acceptable but weaker (a wedding asked as "cake or reception table", a snowy hike asked as
  snow only, "Is there the sea"), 0 wrong. Gate 30/30 (fp16) and 30/30 (4-bit). Dev ALL .627 vs .644: -0.017 [-0.037,
  +0.004], 57 better / 70 worse; a -0.050 [-0.088, -0.013], d -0.057 [-0.108, -0.010], e +0.163, g +0.025.
  Why B1 lost (aggregate plan shapes on dev, then the worst requests): my rule "a second need -> filter_question" copied
  the key's operationalisation, so ONE described thing became two AND-ed questions (filter rate in targets b .39, c .56,
  f .87 vs .19 in the old data; dev plans b .03 -> .44): "that door in Buenos Aires with the lion heads" -> door + "lion
  heads above the door?" (1.00 -> 0); "the guy getting shaved" -> "person sitting or lying down?" + "razor?" (1.00 ->
  .33). And "a named event becomes the anchor" turned trips into windows: "photos from our Venice trip" -> anchor +
  same_event, 299 returned vs 48 (.98 -> .28). B1's wins: e anchors asked as "Is there X in this photo?" with no
  moment-restating question (koala mascot .10 -> 1.00, colonial drummers 0 -> 1.00).
  B2 (rules revised ONCE after B1's dev result; so B2's dev numbers are optimistic): one question for one described
  thing, filter only for a separately stated condition; an event given only by the key's dates is asked about, anchors
  only when the key has one or the words ask for everything around a moment; 5 targets whose key wants everything in the
  window had a moment-restating question removed. 465 of 560 turns pass; target filter rate 0 outside follow-ups (h .41),
  anchors d .14. Dev plan shapes back to the old adapter's (filter b .00, c .02). Gate 30/30 (fp16) and 30/30 (4-bit).
  Dev ALL .633 vs old .644: -0.011 [-0.030, +0.008], 46 better / 57 worse; e +0.185 [+0.029, +0.368], h +0.013, g +0.008;
  a -0.036 [-0.071, -0.006], f -0.055, d -0.024, b -0.023. Remaining losses are wording, not structure: "that picture
  from Whistler where we're wearing pink helmets" -> "... pink helmets in a snowy mountain landscape?" (1.00 -> 0); "the
  photos of the trail maps" -> "Is there a real trail map anywhere ..." instead of "Is this a photo of a trail map?"
  (.79 -> .30); "the safari ... spotted dogs" gained place "safari" (.67 -> 0).
  4-bit gate (scripts/merge_lora_into.py on models/qwen35_4b_mlx4sim): B1 .631 vs fp16 .627 (+0.004 [-0.016, +0.022]);
  B2 .641 vs fp16 .633 (+0.008 [-0.007, +0.023]): same within noise. On the phone's weights B2 vs the RESULTS 31 adapter:
  .641 vs .634, +0.007 [-0.012, +0.027], 48 better / 50 worse (e +0.123, h +0.095 [+0.002, +0.204], b -0.030).
  DECISION: no swap. Neither retrained adapter beats the RESULTS 31 adapter on dev (fp16 -0.011 / -0.017; 4-bit +0.007 /
  -0.003, all inside noise); the app should load models/planner_4b27ball_mlx_adapter (what M59 bundles), not B. What B did show: training on truth programs
  fixes multi-hop moments (e +0.16 to +0.19 in both variants, the one consistent gain), and copying a key's
  question split costs more than it teaches. Models kept: models/planner_4bcb{,2}_{lora,merged,q4merged}.
  GPU: 16.6 GPU-h (search 11.9 incl. 13 model loads per configuration, training 2 x 1.7, planning + gates + teacher 1.4).

## 49b. The planner the phone actually used until 10-12 (BASE Qwen3.5-4B, 4-bit), and why category c fails (10-12; runs/cmp/compare_base4bq4.txt, eval/complex_bench/c_diag/labels.txt, sim_factcut.py, sim_judgecut.py)
Dev only (test split untouched). Method as 49: plan stage alone (plan_system.py), same 9B judge stage with the shared
judge cache, compare.py paired bootstrap over requests. Weights = models/qwen35_4b_mlx4sim (scripts/sim_mlx_quant.py:
mlx-community/Qwen3.5-4B-4bit's affine 4-bit, group 64, as RESULTS 31), greedy, no thinking, the prompt Planner.swift
builds (byte-identical to build_prompt), no adapter. Code = track A + one parity fix (below). ~1.6 GPU-h.

PARITY BUG FOUND FIRST: the base 4B writes "with_people": null (12 of 474 raw outputs). Plan.swift decodes null as the
default (decodeIfPresent ?? []); Python's pydantic rejected the plan, so 3/350 requests failed outright and 6 more got a
different plan from a retry. Python now does what the phone does (converse.Album/Plan: null -> default for looks, avoid,
with_people, media, want, notes, unknown_people, albums); pytest 142/142; plans re-made with --reuse (474/474 outputs
reused, 0 errors). The adapter never writes these nulls (0/421 and 0/425 outputs), so its runs are unchanged.

1. BASE 4B vs THE ADAPTER M59 BUNDLES, both on the phone's 4-bit weights (final-round F1, provisional truth for c, g, h)
  cat (n)              adapter q4   base q4    base - adapter [95% CI]       better / worse by > 0.05
  a objects (61)         .739        .745      +0.006 [-0.033, +0.051]          8 / 6
  b attributes (62)      .662        .618      -0.043 [-0.083, -0.012]          4 / 12
  c person+look (50)     .505        .505      -0.000 [-0.052, +0.052]         10 / 9
  d time/place (64)      .687        .581      -0.106 [-0.185, -0.032]          7 / 21
  e multi-hop (17)       .481        .343      -0.138 [-0.348, +0.075]          3 / 6
  f vague (33)           .606        .612      +0.006 [-0.089, +0.095]          7 / 6
  g two albums (36)      .713        .660      -0.053 [-0.127, +0.027]          4 / 7
  h follow-ups (27)      .467        .393      -0.074 [-0.225, +0.072]          7 / 12
  ALL (350)              .634        .590      -0.043 [-0.071, -0.016]         50 / 79
  ALL verified (237)     .668        .620      -0.047 [-0.082, -0.015]         29 / 51
  (9B planner, same code: .620.) Nothing returned: base 29 requests, adapter 23.
  30-conversation gate on the same weights + code: base 23/30 (adapter 30/30). Failures: "with Dad" and the restaurant
  exclusion dropped; "only outdoors" became an identity filter; cat + videos + Paris; "10 best from Japan" (Japan as a
  judge question, dates = today); "not the blurry ones" only as an avoid word; three albums for "make it two albums";
  "Italy 2019 except selfies" asks "Is this a selfie?" (keeps only selfies), Italy dropped.
  Where the base loses (worst requests, plans read side by side): it restates dates/places INSIDE the judge question,
  which the judge cannot see in pixels ("Is this a photo from Christmas Day 2002?" 1.00 -> 0, "our trip in October 2012"
  -.96, "the beach in Dar es Salaam in April 2009" -.92); empty one-day windows (date_to = date_from; the end is
  exclusive); places as questions instead of the place field ("Dubai 2011 vs Sydney 2010" -.49); self-contradicting
  filters ("black and white photos" + filter "Is this photo in color?" -.94); exclusions inverted ("only the ones with
  rickshaws" -> exclude rickshaws -.94).
  READING: bundling the adapter (M59) buys +0.043 ALL on the phone's own weights [+0.016, +0.071]; ~60% of it is time/
  place (d +0.106) and attributes (b +0.043), e/g/h are also lower with the base (CIs include 0); c, a, f do not move.
  Every device plan before M59 came from the base model.

2. CATEGORY c (person + look): 20 of the 23 dev failures (F1 < 0.5, adapter fp16 run), every sampled photo viewed at
  native pixels (DISBench <= 500 px = what both judges see; 7 random returned-not-truth, 6 random truth-not-returned, 2 hits
  per request), with the 9B's cached P(yes) for the exact image + question (1,333/1,333 found). Per-request labels:
  eval/complex_bench/c_diag/labels.txt. PRIMARY stage per request:
    judge accepts a partial match (colour/garment, hat type, singing vs holding a mic, "party", subject)    7/20
    engine person-look rule (keeps the top half of the person's photos even when the judge says no)        4/20
    truth bug: build.py drops person_need when person is null (truth = every photo of a hidden day/place)   4/20
    "the guy / the hiker ..." whose truth is one moment the words never name                                3/20
    plan (place inside the judge question: goalie #41 "... in a rink in Anaheim?"; cause NOT verified)       1/20
    truth: oracle too lenient (smoking: 0/6 truth-not-returned photos show smoking; truth = 261 of 1,945)  1/20
    face filter                                                                                              0/20
  Evidence: RULE requests (blue surgical cap, yellow tie, holding a beer, pink helmet): 0/28 viewed returned-not-truth
  photos show the item, judge P < 0.5 on 27/28 (max 0.68), and the boxed face was the named person in 27/28 (1 crop
  ambiguous); each album is about half the person's face matches (51/102, 49/98, 38/75, 119/238 by the oracle's face
  matches) because engine.py (and App.swift ~1379) keep
  rel > 0.5 OR (fact and P >= 0.7): the relative rule meant for looks ("heavier") also applies to facts ("wearing a pink
  helmet"). TRUTH BUG: 6 of 50 dev c requests have it (saris, underwear, coconuts, red jacket, gold panning, white
  birds): F1 .184 there vs .549 on the other 44; viewed truth-not-returned photos showed the described thing 0/24. JUDGE:
  the wrong photos score .70-.94, inside the hits' range; a stricter cut does not separate them (cut 0.8 simulated on
  all object/scene albums: c +0.004 [-0.021, +0.027], ALL -0.001, f hits 27 -> 15).
  Hidden scope in general: 28/50 c truths carry days/place while the words have no time cue (F1 .439).

  WHICH SINGLE CHANGE MOVES c MOST
  - As measured, the TRUTH: build.py should ask person_need of the whole photo when person is null (re-run the oracle
    for those 6 requests). It is a benchmark bug, not a product error; dropping the 6 alone reads c .549 instead of .505.
    The bug is in build.py, so it applies to every split's c (test not read).
  - In the product, the person-look RULE: facts use the judge's answer alone (P >= 0.7), the top half stays for LOOK_WORDS.
    Simulated from cached answers (exact: the rule only removes photos; eval/complex_bench/sim_factcut.py): c .505 -> .534
    on the phone's weights (+0.029 [-0.017, +0.082], 4 better / 2 worse), fp16 .498 -> .542 (+0.044 [-0.010, +0.108]),
    ALL +0.004 / +0.006. Pink helmet .02 -> 1.00, surgical cap .43 -> 1.00, beer .10 -> .50 (one hit scored .56), yellow
    tie .77 -> .87. The worst "loss" (Sam with sunglasses on her head .79 -> .30) is a truth artifact: 0/9 of the truth
    photos the cut drops show sunglasses on her head (worn over the eyes, or glasses), nor do the 4 non-truth photos it
    keeps: truth and judge are both wrong there.
    NOT implemented: it changes the phone's person albums ("photos of me at the beach" keeps half of my photos today);
    check on the eye-labelled person sets before shipping (Python engine.py stream_album + Swift App.swift together).
  - The judge's leniency is the biggest product class (7/20) but has no single knob (see the cut result above).

## 49c. Benchmark fix (person_need with no named person) and the person-album rule: FACTS by the judge alone, LOOKS stay relative (10-12; eval/complex_bench/{build,oracle,common}.py, c_diag/{s5,s6}*.py; eval/person_rule_49c.py -> eval/vector_first_person/rule49c.txt; engine.is_look_question / person_keep; FindPicsCore isLookQuestion / personAlbumFound)
Dev only for every score (test split: rebuilt, byte-identical, never scored).

1. BENCHMARK BUG FIXED. build.py skipped `person_need` when `person` was null, and oracle.py never asked those questions,
  so the truth was every photo of the album's hidden day/place. Now the question is asked of the WHOLE photo ("Is the
  person wearing a sari?" -> "Is a person wearing a sari?", common.whole_photo_q). 7 requests have it, all dev c (test 0,
  train 0); the 27B-FP8 oracle answered only the 8 new questions on the 2 libraries (oracle.py --users; 14,973 answers,
  ~6 min on one GPU). Rebuild: test.json byte-identical, dev changes only these 7; reproduction check first (old code ->
  test and dev byte-identical to the committed files).
  | request | truth before -> after | eye check, native pixels (c_diag/truthfix_labels.txt) |
  |---|---|---|
  | women in saris | 127 -> 39 | 36 sampled: 16 clear, 8 small/borderline, 12 not (tiny people at temples, saris drying, a sadhu) |
  | the guy in his underwear | 61 -> 1 | 1/1 |
  | the woman carrying coconuts | 23 -> 4 | coconuts 0/4; a load on the head 3/4 (oracle lenient) |
  | the guy in the red jacket at base camp | 42 -> 19 | 19/19 viewed: 16 yes, 2 small (one a woman), 1 no |
  | panning for gold in the stream | 30 -> 4 | 4/4 |
  | feeding the white birds at the zoo | 46 -> 0 | request DROPPED (empty answer) |
  | the guy in the red jacket skiing | dropped (1,149) -> 311 | newly KEPT; 12 of 36 sampled viewed: skiing 11/12, red jacket 3/12, red backpack only 6/12 (lenient) |
  Photos the fix REMOVED from a truth: 0/36 sampled show what the request describes (1 borderline).
  RE-SCORED (cached plans + cached 9B judge; the one newly kept request planned and searched once with the same weights;
  that re-planning reproduced the other 349 cached plans exactly, both weights):
  - adapter 4-bit (the phone's planner): dev c .505 (old truth) -> .552 (new truth), 50 requests each (-1 +1). Paired on
    the 49 requests in both: .516 -> .548, +.032 [+.003, +.075], 4 better / 0 worse. ALL .634 -> .640.
  - adapter fp16: c .498 -> .541; paired .508 -> .538.
  - The 6 truth-bug requests of 49b (adapter 4-bit): mean F1 .184 (6) -> .533 (the 5 still in the set; saris .36 -> .77,
    base camp .49 -> .72, gold .24 -> 1.00, underwear .03 -> .17: 11 returned for 1 truth photo, coconuts 0 -> 0: nothing
    returned); fp16 .169 -> .494. The newly kept skiing request: .758 (fp16 .721).
  Remaining c truth weakness: the oracle is lenient on attributes (red jacket <- red backpack; coconuts <- any load on the
  head; saris <- tiny figures). c stays "provisional".

2. PRODUCT RULE CHANGED (Python engine.person_keep + make_exclusive; Swift FindPicsCore personAlbumFound + App.swift).
  A person album's judge question is a LOOK if it says look/appear/seem (old LOOK_WORDS) OR names a relative body/mood
  trait (heavier, fitter, slimmer, thinner, leaner, overweight, out of shape, fit, tired, happier, ...; or heavy/slim/
  thin/athletic... + build/frame/physique/body/figure). Bare "heavy", "thin", "older" are NOT looks ("carrying a heavy
  bag", "a woman and an older man together" are facts). LOOK -> top half of the person's own photos (unchanged).
  FACT -> judge P >= 0.7 alone (was: top half OR P >= 0.7). Pairing ("A vs B" of one person) now applies to two LOOKS
  only: two facts ("wearing glasses" / "not wearing glasses") keep their own yes photos.
  Tested on the 30 conversations' plans: 93 person albums with a question across the 10 gate runs (31 distinct
  questions): 54 looks (every heavier/fit album, every planner's wording) and 39 facts (smiling with friends, beach,
  weddings, Mom and Dad at Christmas). Changed classification vs the old test: the 6 "a person with a heavy/fit build"
  questions the 9B and base-4B planners write were FACTS before (no "look" word) and are LOOKS now. Shared fixture
  ios/FindPicsCore/Tests/FindPicsCoreTests/Fixtures/look_questions.json (69 cases: every gate question, IMDB / demo /
  vector_first_person / dev-c questions, tricky facts), checked by pytest and swift test.
  Side effect in Swift only: App.swift paired ANY two albums of one person, including two plain "photos of Dad" albums
  with no question (no scores -> rank margin over nothing -> both albums empty; read from the code, not run). Python
  never did. The look check removes it.

  (a) eval/vector_first_person: 32 DISBench people, 7 fact conditions, phone judge (Qwen3-VL-4B phone weights, red box) on
  every photo. Each rule keeps whole strata of a person's photos; 290 photos labelled blind at native pixels (key read
  after labels), stratified: judge-yes top/bottom half, judge-no top half (P 0.1-0.7 / P < 0.1 by look-vector), judge-no
  bottom half. Estimated over all 4,010 photos per condition; unsure = not a match; [95% bootstrap within strata].
  | condition | OLD: kept, precision, recall, F1 | FACT (new): kept, precision, recall, F1 |
  |---|---|---|
  | beach | 2,012, .085, .838, .155 | 311, .500, .757, .602 |
  | eating | 2,015, .028, 1.00, .055 | 113, .500, 1.00, .667 |
  | hat | 2,067, .357, 1.00, .526 | 723, 1.00, .980, .990 |
  | outdoors | 2,611, .762, .988, .860 | 2,260, .873, .981, .924 |
  | smiling | 2,649, .842, .879, .860 | 2,316, .958, .875, .914 |
  | snow | 2,012, .014, 1.00, .027 | 55, .500, 1.00, .667 |
  | sunglasses | 2,028, .310, 1.00, .473 | 738, .830, .975, .897 |
  | ALL 7 | 15,394, .379 [.337, .413], .942 [.855, 1.00], .541 [.489, .578] | 6,516, .885 [.780, .964], .930 [.846, .988], .907 [.838, .962] |
  Estimated right photos 5,841 -> 5,768 of 6,204 (-73, 1.2%) while wrong photos 9,553 -> 748. The loss is mostly beach
  (judge P 0.1-0.7 in the person's top half: 1/8 real); eating and snow: 0 real of 22 + 22 viewed in the strata the
  new rule drops. Unsure counted as a match: OLD F1 .587, FACT .891. Per-condition CIs are wide (6-8 labels/stratum).
  SMILING IS A FACT, measured, not a look: LOOK rule (top half only) .844 / .687 / .758 - it drops 584 judge-yes smiles in
  the bottom half (5/6 real); FACT .914 > OLD .860 > LOOK .758. "Tired" is a look under both codes: unchanged.
  6/290 boxes were a dog's face, all the same "person" 47642109@N04#2 (M45's face-group purity issue), labelled no_match.

  (b) LOOKS UNCHANGED. IMDB-WIKI eras (RESULTS 10/18, 239 face-filtered crops, cached 9B P): albums before vs after, for
  the cached questions, the adapter's wording ("looking heavier/fit") and the 9B/base wording ("a heavy/fit build"):
  identical in 5/6 settings, every PAIRED setting (heavier 82 heavy-era / 22 lean, fit 75 lean / 3 heavy). The one change
  is a SINGLE album worded "with a heavy build": before it was treated as a fact (top half OR P >= 0.7: heavier 96/43,
  fit 103/45), now as a look (85/35, 96/24 = the same albums as the "look" wording; by era 0.69 -> 0.71 and 0.70 -> 0.80
  right). The owner's 33 saved heavier-vs-fit demo runs (private; scripts/replay_pairing.py on cached judge answers,
  counts in data/private/audits/pair_fix49c_*.csv): all 7 questions they used classify as looks; 33/33 runs give
  identical album counts before and after.

  (c) Complex bench dev, cached 9B judge, the search stage REPLAYED with the product code (run_system.py --judge=cache:
  no model; the old code reproduced the 12 affected requests' albums exactly, 12/12 in both runs, 0 cache misses):
  | run | c (50) | g (36) | ALL (350) |
  |---|---|---|---|
  | adapter 4-bit, fixed truth | .552 -> .580, +.029 [-.015, +.082], 4 / 2 | .713 -> .728, +.015 [.000, +.041], 3 / 0 | .640 -> .646, +.006 [-.001, +.014] |
  | adapter fp16, fixed truth | .541 -> .585, +.044 [-.011, +.108], 4 / 2 | .706 -> .722, +.015 [.000, +.041], 3 / 0 | .650 -> .658, +.008 [.000, +.018] |
  | adapter 4-bit, old truth | .505 -> .534 (= 49b's simulation), +.029 [-.017, +.082] | +.015 | .634 -> .639 |
  g gains: the three fact pairs (beard / clean-shaven .27 -> .64, sunglasses / none .75 -> .85, glasses / none .78 -> .86)
  are no longer forced apart by rank. The 2 worse c requests, truth photos the rule drops viewed (c_diag/
  rule_lost_labels.txt): "Sam wearing sunglasses" 0/12 show sunglasses, "sunglasses on her head" 0/12 on her head (worn
  over the eyes): the truth is wrong there; "with the tennis racket" 3/4 and "holding a beer" 1/1 are real misses of the
  judge (small or partial racket). Over all 29 viewed: 25 truth errors, 4 judge misses.

  ADOPTED: (a) F1 .541 -> .907 (precision .379 -> .885, recall .942 -> .930), (b) looks unchanged in every paired album
  and for the look wording (the single "heavy build" album now follows the look rule), (c) c +.029 / +.044, g +.015.
  Tests: pytest 145/145 (+3: fixture parity, fact vs look keep, fact pairs not paired; 4 pairing tests now use real look
  questions); FindPicsCore swift test 136 XCTest + 26 swift-testing, exit 0, no segfault. App.swift not compiled here
  (Mac). Python and Swift share the regexes and the 69-case fixture.

## 50. Food photos by the OWNER's notion, and a shop-front veto (10-12; eval/eval_food_notion.py -> eval/food_notion/report.txt, notion_tables.txt, veto_tables.txt; MAC_INBOX M62)
Why: RESULTS 45-47 measured food precision against blind labellers who count meals-with-people and drinks as "not food".
Reza answered "Is this a food photo?" for 35 public audit4 photos (eval/food_notion/reza_answers.tsv; key
data/public/food_notion_35/key.json): 25 DISPUTED (judge yes, labellers no_match/unsure) + 10 controls. THE OWNER'S NOTION
RESTS ON THESE 25 SELECTED ANSWERS: F1 people at a meal 10/11, F2 food shop / sign / van with no food 0/7, F3 pets eating
1/1, F4 drinks 2/2, F5 edible thing not served 1/2, F6 produce display 1/1, F7 food event 1/1 (F3/F6/F7 on one answer
each); controls 9/10 (the no: string-tied steamed crabs in front of diners).
Method: a tagged (false_positives_scene.txt) not-a-match label counts as w = his yes-rate for its kind; untagged labels keep
their label; every table re-scored exactly as made (eval_recall.score_systems(weight=...); labellers' column reproduces
every published number). Sensitivities: all kinds at the 95% CP low/high end; "+ my tags" = 22 untagged not-a-match labels
at P >= 0.99 that I viewed and tagged with the same kinds (eval/food_notion/my_kind_tags.tsv).
(1) precision / recall, labellers -> owner (CP-end range) [owner + my tags]:
| rule (kept of 7,886) | labellers | owner | owner + my tags |
|---|---|---|---|
| R45 fast first answer, cut 0.99 (153) | .57 / .908 | .84 (.67-.94) / .865 | .84 / .865 |
| R46 cut 0.99 (164) / 0.999 (146) / 0.9999 (125) | .60/.931, .68/.931, .78/.922 | .86/.906, .87/.818, .92/.741 | same |
| R47 q0 >= 0.99 (164) | .63 / .971 | .78 (.68-.83) / .955 | .87 / .959 |
| R47 q0 >= 0.999 (144, was) | .70 / .957 | .83 (.75-.87) / .886 | .90 / .869 |
| **R47 q0 >= 0.9999 (125, SHIPPED)** | **.80 / .947** | **.89 (.83-.91) / .829 (.80-.90)** | **.94 / .784** |
| R47 g_main >= 0.999 (90) / f_meal >= 0.99 (107) | .94/.805, .86/.871 | .96/.640, .91/.727 | .96/.580, .92/.661 |
Read: by his notion food search is not half wrong (shipped .89, old fast answer .84); what he rejects is the shop front
with no food. But the strict cut costs HIM recall: of the disputed photos the shipped rule keeps he says yes to 10/13, of
those it removes (P in [0.99, 0.9999)) 6/12. Owner F1: 0.99 .860 / 0.999 .855 / 0.9999 .859 (tags only), .914 / .885 /
.853 (+ my tags): the R47 food cut wins only by the labellers' notion. His recall is an UPPER BOUND: he was asked only
about judge-accepted photos; tagging 48 judge-rejected meals/drinks/pets labels at his rates (my_kind_tags_rejected.tsv)
moves his total 149 -> 198 and shipped recall to .589. Precision is the robust part.
(2) SHOP-FRONT VETO, keep = shipped rule AND P(yes) >= cut; phone-weight pipeline of eval_scene_questions gen on 164
audit4 photos (q0 >= 0.99) + 50 R35 (2) other-library food photos (fp_foodveto, 642 calls, 0.07 GPU-h; shipped-question
recheck 163/164 identical at 0.9999). Cut chosen on one item_id-parity half, reported on the other (both ways):
v_real "Is real food or a drink visible in this photo (not just a sign, menu or shop front)?" -> 0.01 on both halves;
v_any "Can you see any actual food or drink in this photo, even a small amount, rather than only a restaurant, shop front,
sign, menu or food van?" -> 0.01 / none (unstable). Every shop front in the shipped set (6) gets P < 0.5 from both.
| audit4, held out | kept | labellers P / R | owner P / R | owner + my tags |
|---|---|---|---|---|
| shipped | 125 | .800 / .947 | .891 / .829 | .935 / .784 |
| shipped & v_real >= 0.01 | 120 | .825 / .938 | .920 / .822 | .966 / .777 |
| shipped & v_any (out-of-fold) | 115 | .826 / .900 | .921 / .788 | .964 / .744 |
| exploratory: q0 >= 0.999 & v_real >= 0.01 | 136 | .735 / .947 | .861 / .872 | .940 / .857 |
Viewed, every drop: v_real removes a pizzeria front, an ice-cream van, a neon hotpot street, charity boxes, and (wrong) a
shelf of chip bags; in other libraries a festival dessert sign, a pub bread board, a vegetable garden (right/wrong/unsure
kept 10/5/5 -> 10/2/5). v_any also throws away a Cheerios box and a Lay's bag (controls he said YES to), three stockfish
photos and a pie crust: "actual food" reads as not-packaged / not-raw.
DECISION: veto NOT shipped. +.025/+.029 precision for -.009/-.007 recall (4 wrong for 1 real of 125, inside the CIs), one
more judge call per kept photo, fragile wording; the 0.9999 cut already removed 5 of 8 tagged shop fronts. The bigger
lever for the owner is the cut; "0.999 + v_real >= 0.01" is better for him (+ my tags .940/.857 vs .935/.784) and worse for
the labellers (.735 vs .800): a candidate found after looking. Next: Reza answers the 13 unanswered photos of the
[0.999, 0.9999) band (6 answered: 3 yes, 3 shop fronts no); switch to 0.999 + v_real only if he calls most non-shop ones food.

## 51. Complex bench dev, e multi-hop and h follow-ups: failure kinds by cause, and code fixes (10-13; eval/complex_bench/eh_diag/{labels.txt,anchors.log,probe_absent.log}, runs/cmp/{compare_51,attribution_51,per_fix_51}.txt; agent.one_day_anchors, converse.earlier_year_if_empty + ground "photos from a moment", engine.place_parts; FindPicsCore oneDayAnchors / earlierYearIfEmpty / placeParts / Grounding; App.swift, Moments.swift)
Dev only; the test split was not built, read or scored. Planner = what the phone ships (the 10-05 adapter on the phone's
4-bit weights, models/planner_4b27ball_q4merged, prompt = Planner.swift's); judge = Qwen3.5-9B with the shared answer
cache; baseline = 49c (fact rule, fixed truth): runs/cmp/r49c_A4oldq4_new_truthnew. Method as 49: plan stage alone
(plan_system.py --reuse: 3/350 plans changed, 2 new model outputs, 0 errors), search stage with the 9B + cache
(r51_search), then the FINAL code replayed with --judge=cache (r51f_search: no model, 0 misses, 350/350; albums identical
to r51_search except the 3 requests of the rejected rule below), compare.py paired bootstrap over requests. 1.0 GPU-h.

1. WHY THEY FAIL. All 27 dev e/h requests with final F1 < 0.5 (e 11/17, h 16/27), one primary cause each, from plans,
  traces, the anchor stage's hits recovered from the judge cache (every hit's P and day, the truth anchor photo's P) and
  photos viewed at native pixels where judge vs truth was in question (eh_diag/labels.txt):
  | cause | e | h | what happens |
  |---|---|---|---|
  | anchor matched several days | 5 | 1 | "the day I saw THE alpaca": the judge's extra hits on other days (P .70-.87) each open a whole day; the moment's own day scored highest in 5/5 e cases (.92-.99); the dog (h) is on 4 days at .99 |
  | anchor not found | 4 | 0 | no photo reaches .7: truth anchor .44 (colonial fife-and-drum band), .62 ("Iron Man" doll), .35 (bike in TINSEL the words call "Christmas lights"); Everest: the anchor asks about the peak and its place matched 1 photo |
  | moment restated as the album's question | 1 | 1 | "the day I took that selfie" -> "Is this a selfie?" (1 of 63); "the hospital when the baby was born" -> "a baby being born" (25 of 133; 6/8 missed photos viewed are newborns in the hospital) |
  | planner follow-up edit | 0 | 3 | turn 2 ignored "against the Dallas Stars", turn 3 kept it; "include the judo ones too" kept "muddy" (4/4 missed = judo on dry mats); "on land too, just not in the water" -> "flying or on land" (8/8 missed = gulls on rails/seaweed/piers) |
  | year-less date, wrong year | 0 | 2 | "Cortina in June" said in July 2014 -> June 2014 (no photo there; trip June 2012); "the November trip to Java" -> Nov 2013 (has photos; trip Nov 2009) |
  | place list | 0 | 1 | "Burgos and León" needed every word in ONE address: Burgos only (3/4 missed photos viewed = León churches) |
  | same_place window | 1 | 0 | same_place = the identical address string (4 photos); truth = 25 photos of the department |
  | benchmark truth | 0 | 8 | stripped place/days left truths that ignore the words (Borobudur turn 3 = all September; Europe/Dubrovnik/Zagreb truth = stone walls/trams, 4/4 returned-not-truth viewed are Zagreb/Dubrovnik); "parade" truth = a whole day of hiking and lunch (0/8 missed show a parade); concert = "a stage" (7/8 returned-not-truth viewed are concert photos); "include the bobcats too" truth dropped the lions |
  h is mostly a TRUTH problem: 8/16 h failures are benchmark errors (h stays provisional); of the 19 product failures
  across e+h the top kinds are anchors (6 several days + 4 not found) and follow-up edits by the planner (3).

2. FIXES (Python + Swift; FindPicsCore functions with mirrored tests; App.swift / Moments.swift call them, not compiled here)
  A. ONE-DAY ANCHORS (agent.one_day_anchors / oneDayAnchors): a same_day moment whose confident hits fall on several days
     keeps only the day(s) whose best hit is within 0.05 of the best hit overall; days the judge cannot tell apart all
     stay (the dog), other windows unchanged (a trip spans days: Edinburgh's 5 trip days score .80-.97). Dev same_day
     anchors with hits on 2+ days: 8 recovered from the cache (anchors.log; 5 narrowed by the rule), 0 truth days
     dropped; a 9th (taxidermy bear, not recoverable: cache miss) went .94 -> 1.00. The 0.05 band was set after seeing
     these dev P values. Swift reads the anchor P from the judge cache (id|question).
  B. YEAR-LESS DATE, EMPTY WINDOW (converse.earlier_year_if_empty / earlierYearIfEmpty, at search time after place_or_look):
     a date that recurs yearly (month, season, holiday, month-day; "last winter" counts, "this June", "two years ago",
     "last week", any year do not) whose window holds no photo here (with the album's place and media) moves back whole
     years to the latest one that has photos; the album says so. Fires only on an empty window, so it can only replace
     "nothing". (Java is not fixed: Nov 2013 has photos.) After the dev runs: "New Year's ..." counts as yearly (its
     "year" no longer disqualifies it); no dev plan has such a date, so the runs are unaffected.
  C. PLACE LISTS (engine.place_parts / placeParts + scopeMask): "A and B", "A or B", "A & B" match a photo at ANY of them.
  D. PHOTOS FROM A MOMENT (converse.ground / Grounding.ground): "(the other) photos from the day/trip/night ... (I|we) ..."
     with an anchor, no person and no narrowing word ("only", "just", "except", "without"...): an album question made
     only of the request's own words restates the moment and is dropped (everything in the window). Did NOT reach the
     pink-vest selfie (the planner made it a person album "me") or the hospital ("baby being born": "born" vs "being").
  REJECTED. CLOSEST MOMENT: a same_day moment with no hit >= .7 would use its best photo if P >= .3. On dev it turned the
     3 not-found e requests 0 -> 1.00 each (e .481 -> .812). FALSIFIED by a probe (eh_diag/probe_absent.py, 9B, 52
     probes): each of the 13 dev e moments asked in 4 OTHER dev libraries, where it should not exist: best P >= .3 in
     34/52, >= .5 in 25/52, >= .7 in 15/52 (median .47). The dev moments' own P (.35, .44/.50, .62) sit inside that range,
     so no floor separates "the moment, missed" from "no such moment". 8 highest absent probes viewed: 6/8 do not show the
     moment (a race car for "koala mascot" .88, a bar for "taxidermy bear" .88, hay bales for "bouncy castle" .84), 1/8 is
     a real bouncy castle (.995), 1/8 unclear (an orange inflatable shape behind a couple, .82). Not adopted.
     Side finding: 15/52 probes already reach the .7 cut (6 of the 8 highest viewed are not the moment), so a moment the
     library lacks is often "found" on some other day; the one-day rule does not change that.

3. RESULT (dev, final-round F1, paired vs the 49c baseline [95% bootstrap], better / worse by > 0.05; compare_51.txt)
  | cat (n) | 49c | RESULTS 51 | diff [95% CI] | better / worse |
  |---|---|---|---|---|
  | a objects (61) | .739 | .755 | +.016 [+.000, +.048] | 1 / 0 |
  | b attributes (62) | .662 | .662 | +.000 | 0 / 0 |
  | c person+look (50) | .580 | .580 | +.000 | 0 / 0 |
  | d time/place (64) | .687 | .722 | +.035 [+.001, +.080] | 3 / 0 |
  | e multi-hop (17) | .481 | .636 | +.154 [+.037, +.298] | 5 / 0 |
  | f vague (33) | .606 | .606 | +.000 | 0 / 0 |
  | g two albums (36) | .728 | .728 | +.000 | 0 / 0 |
  | h follow-ups (27) | .467 | .506 | +.040 [+.000, +.096] | 3 / 0 |
  | ALL (350) | .646 | .666 | +.020 [+.009, +.033] | 12 / 0 |
  | ALL verified (237) | .668 | .692 | +.025 [+.010, +.043] | 9 / 0 |
  Nothing returned: 23 -> 19 requests. Per fix alone (per_fix_51.txt; touched requests take the new F1, others the old):
  one-day 6 requests, ALL +.008 [+.002, +.015], e +.154 [+.039, +.290] (wardrobe .26 -> 1.00, alpaca .27 -> 1.00, white
  sports car .44 -> 1.00, bouncy castle .46 -> 1.00, taxidermy bear .94 -> 1.00; Denali d .31 -> .34); year-back 5, ALL
  +.009 [+.002, +.019] (November 27th concert a 0 -> .98, snow day last winter d 0 -> .94, Cortina h 0 -> .63, Phoenix
  zoo d 0 -> .57, Melbourne g 0 -> 0: its truth puts "winter" in July); place lists 2, ALL +.002 (Tokyo and Osaka d .16 ->
  .83, Burgos and León h .25 -> .35); photos-from-a-moment 3 plans, ALL +.001 (Edinburgh h .53 -> .87, Everest e 0 -> 0,
  Denali shared). 1 request changed with no rule (Venice d .19 -> .22: a new judge answer on another GPU; noise).
  No request got worse in any category; every fix ADOPTED (A-D). g did not move: its 6 failures are not these kinds.

4. CHECKS. 30 scripted conversations on the 4-bit adapter (eval_planners.py, FP_PLANNERS_ONLY_CONV): 30/30 (2 plans differ
  from 49's gate only in "looks" wording: vLLM output, not the rules). Grounding fixtures regenerated (ground_fixtures.py
  now also takes 8 shipped-planner dev outputs, eval/complex_bench/ground_dev_cases.json, so rule D is compared): the 2,300
  distillation plans unchanged (fixture_diff: 0 changed), +8 cases; FOLLOWUP 874 unchanged. Swift GROUND 0/2,308 and
  FOLLOWUP 0/874 mismatches; pytest 149/149 (+4); swift test XCTest 140 (+4) + swift-testing 29, 0 failures, exit 0.

5. NOT FIXED, for the next round: anchor not found (4 e; no score floor works, see the probe: needs a better anchor
  question, e.g. the words' own nouns asked separately, or the person picking the moment photo); planner follow-up edits
  (3 h); the pink-vest person album; same_place as an exact address string (use GPS distance); the shoe sculpture
  ("the trip where I saw X" = same_event; the one-day rule is same_day only); h truths (8/16 wrong: h needs a truth fix
  before it can rank fixes).
