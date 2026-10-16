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

## 52. Complex bench: the h (follow-up) TRUTHS fixed - causes, compile rules, a 27B rewrite of every h program, a fresh blind audit, dev h re-scored (10-13; eval/complex_bench/{build,gen,common,oracle,audit3}.py, h52/{diag,regen_summary}.py + *.txt, programs_h2/, audit3/, runs/cmp/r52_*)
Benchmark construction, all splits (test truths rebuilt; the system was NOT run, scored or looked at on test). Every other
category is byte-identical to a rebuild with the old code (checked per split); only h and the `verified` flag of h change.
queries/train.json now covers all 24 train libraries (651 requests; the committed file had the 12 built on 10-12).

1. WHY h TRUTHS WERE WRONG. Read every dev + train h program (57 + 30 conversations); four systematic causes. Counts on the old
  truth (106 h requests, h52/diag.txt, word heuristics, so indicative): a turn returning >= 25% of the library 8, a stripped
  constraint while the words name a place 16, a follow-up that contradicts its own edit 7, place-OR not expressible 6.
  (1) COMPILE. build.py's 10-12 strip dropped a place/date constraint unless the constraint's own words were in the request,
      but the generator had encoded places the owner NAMES with other words: "photos from Laos" = a date window (no time word
      -> dropped -> turn 1 = the whole library, 1,945 photos); "photos of Borobudur" = days (-> whole library); "Hampi" =
      "Hosapete taluk"; "trip to Bangladesh" = "Dhaka"; "Machu Picchu" vs the address word "Machupicchu"; "at the winery" =
      "Paso Robles" (turn 2 = whole library). Root: the digest lists ONE address level per day (common.city), so the
      generator never saw "Laos" (252 addresses contain it) or "Hampi" (74). Also: the place matcher turned "León" into the
      word "le" (accents stripped to spaces; every category has this; fixed for h only).
  (2) GENERATION, edits: a follow-up's program drops what the conversation keeps ("include the bobcats too, ... sleeping on
      the rocks" = the sleeping bobcat only, lions lost; "keep only the ones from September" = all of September; "just the
      ones at the winery" = Paso Robles with the dinner condition lost).
  (3) GENERATION, events: an event as a whole day ("photos from the parade" = a day of hiking and lunch) or as objects it may
      contain (concert = "Is there a stage?", which misses close-ups and crowd shots).
  (4) SCHEMA: one album per turn and `place` = all words, so "these OR those" could not be written: "also add the ones from
      Zagreb" became stone walls, ocean and trams; "add Philadelphia" dropped Chicago.

2. FIXES
  A. COMPILE (build.py, FP_CB_HFIX=1, category h only; FP_CB_HFIX=0 FP_CB_H2= reproduces the old test/dev files byte-for-byte):
     a place constraint stays when the words name ANY place (an address name of the library at any level with >= 2 photos,
     a landmark of the photo facts, or "the winery / hospital / zoo ..."); a date/day/hour constraint stays when the words
     carry a time cue (as before) OR name a place the album's place fields do not cover (then the dates are its encoding);
     place names match accent-, case- and space-insensitively; new field place_any (any of several places); a turn may hold
     several albums = their union. Checks that DROP a request: a follow-up whose words only narrow returns > 10% photos
     outside the previous turn, or one that only widens loses > 10% of it ("actually"/"wait" open most messages of every
     kind, so they are not a replace cue); a message that states a year compiles to an album with no date.
  B. GENERATION (gen.py regen_h; Qwen3.5-27B-FP8, thinking on, the original generator): all 171 h conversations' turn
     programs rewritten with the SAME words, given the digest + every address name at every level (new PLACE NAMES list) +
     the draft programs + 7 rules (carry the conversation forward; "also/too" = union; named places by their address name;
     an event = its days + a direct question, never a whole day or proxy objects; a trip = its days or place; no unstated
     date/place; questions no narrower than the words). The prompt's examples are generic (no library's own words). 170/171
     parsed (1 train rewrite failed: original kept); 36 hit the 14k-token thinking budget and were rewritten without thinking.
     147/171 changed >= 1 turn (377/510 turns), 27 turns use several albums, 20 use place_any, 118 new question texts. The
     oracle answered only those: 208,576 answers on 49 libraries (oracle.py via common.load_programs, programs_h2/ overlay).
  C. h ORACLE CUT 0.9 -> 0.5 and h VERIFIED, by a rule written into AUDIT_PROTOCOL.txt before any label was read (cut moves
     only if 0.5 beats 0.9 on F1 on BOTH the test-only and the dev-only requests; verified only if precision >= .8 and
     recall >= .7). Event questions ("Is this photo taken at a parade?") are answered conservatively by the 27B: at 0.9 the
     parade day had 3 parade photos and 9 float photos.
  Result: h test 49 -> 47, dev 27 -> 27 (5 out, 5 in), train 30 (12 libraries) -> 62 (24). h drops (35/171): edits change
  nothing 21 (rule unchanged from 10-12; it removes "undo" conversations), edit check 7, empty 4, stated year lost 3.

3. BLIND AUDIT, before vs after (audit3/; 20 FRESH h requests, 12 test + 8 dev, none of the 10-12 audit's 20; 293 photos at
  native pixels, 263 labelled + 30 unsure; 4 Claude subagents, never shown the key; the main agent labelled 4 of the requests
  blind: 67/67 agreement, unsure excluded). Strata = membership pattern over the four versions + in/out-of-scope hard and
  random non-answers; stratified estimate, 95% bootstrap over requests:
  | truth version | requests | precision | recall | F1 | est. right / wrong / missed |
  |---|---|---|---|---|---|
  | h1 old (as committed) | 20 | .89 [.72, .99] | .80 [.68, .90] | .84 | 773 / 93 / 195 |
  | h1c old programs + new compile rules | 19 | .98 [.93, .99] | .81 [.68, .91] | .89 | 713 / 18 / 166 |
  | h2 rewritten programs, cut 0.9 | 20 | .98 [.93, 1.00] | .66 [.55, .79] | .79 | 641 / 14 / 328 |
  | **h2 cut 0.5 = NEW TRUTH** | 18 | **.93 [.84, .97]** | **.94 [.89, .98]** | **.94** | 899 / 63 / 61 |
  Paired on the 17 requests every version keeps: old .98 / .81 -> new .93 / .93. The old truth's precision loss sat in one
  request ("library photos with the ceilings": 70 photos only the old truth held, 0/6 right; the edit check rejects its old
  program, the rewrite keeps the request with a different set).
  Test only (12 / 11 requests): old .85 / .80 -> new .91 / .94 (cut 0.9: .97 / .59). Dev only (8 / 7): old .97 / .80 ->
  new .97 / .92 (cut 0.9: 1.00 / .78). Cut chosen on one half, reported on the other: even half picks 0.5 (F1 .94 vs .85),
  odd half reports .94 / .91; odd picks 0.5 (.93 vs .72), even reports .93 / .96. h meets the bar: VERIFIED.
  Photos only the cut-0.5 truth holds are right 23/38 (that is its precision cost); in-scope hard non-answers matched 3/58,
  out-of-scope 1/32, random 0/31.
  The 8 dev TRUTH errors of RESULTS 51: 5 fixed (winery/Murphys = dinner at either place; bobcats = lions + sleeping
  bobcats; concert = "taken at a concert" 20 -> 75 photos; Europe/Dubrovnik/Zagreb = places, 285 -> 187; parade: turn 1 =
  the parade photos instead of the whole 73-photo day, final turn 12 -> 11 photos, 9 shared), 3 removed (Borobudur: final = first turn; Hampi: empty once "ruins" is carried
  forward; Laos: at cut 0.5 the "buffalo" turn adds 1 photo the "animal" turn lacks, 1 of 9 > 10% slack).

4. DEV h RE-SCORED (cached plans + 9B judge cache; the 5 requests new to the truth planned and searched once per code
  version with the shipped 4-bit adapter: runs/cmp/r52_new, r52_new2; h52/dev_h_rescore.txt per request)
  | dev h, final-round F1 (n) | 49c baseline | RESULTS 51 code | diff [95% CI] | better / worse |
  |---|---|---|---|---|
  | old truth (27) | .467 | .506 | +.040 [.000, +.095] | 3 / 0 |
  | new truth (27) | .573 | .621 | +.048 [.000, +.113] | 3 / 0 |
  | the 22 in both truths | .472 -> .542 | .520 -> .601 | | |
  The same three requests improve under both truths (Edinburgh castle + golf .32 -> .67 on the new truth, Cortina June 0 ->
  .71, Burgos and León .53 -> .78): RESULTS 51's h gain stands on the sound truth. The truth fix itself moves the system's
  h score .506 -> .621 (Europe trip .40 -> .99, concert .39 -> .88, parade .47 -> .62, dog day .23 -> .48). Dev ALL .654 ->
  .674 (+.020 [+.009, +.035]; 349 requests, the 49c run lacks one c request); ALL verified (now with h, 264) .658 -> .685
  (+.027 [+.012, +.045]). Remaining dev h product failures (new truth, F1 < .5): winery dinner 0, hockey goalie 0, Java
  market 0, zoo cats .07, hospital baby .31, Zermatt hut .33, seagulls .39, muddy rugby + judo .46, dog day .48.

5. g ("one album far too narrow") is NOT a compile bug: g has no strip (STRIP_CATS = d,h) and its programs compile as
  written. Of the 44 in-scope non-answers the 10-12 audit labelled as belonging to an album, 34 are in the 7 g requests whose
  programs test an event or activity through proxy objects, mostly AND-ed (roller derby = helmet AND rink, flag football =
  field AND flag, concerts = stage OR microphone stand, the wedding = dress / groom / place card, Celtics games = basketball
  court, beach = ocean AND beach), 10 are oracle/person misses in 4 others. Same kind as h cause (3): the fix is the same
  rewrite (gen.py regen_h with rule 3 for g's albums) + the oracle + a g audit. Not done here.

6. CAVEATS. The rewrite, the oracle and the product judge are all Qwen3.5 (errors correlate). The compile rules and the
  prompt were designed after reading dev + train programs and the 8 dev truth errors; 8 of the 20 audited requests are dev
  (test-only numbers above). Labels are Claude subagents', not Reza's; one labeller checked a hockey schedule on the web.
  The cut was chosen on the same labels by a pre-written rule (halves agree). The 10% edit-check slack drops requests on
  1-photo oracle noise in small sets (Laos). sft_from_truth.py does not know place_any or several albums per h turn: check
  before the next planner training on train truths. ~3.7 GPU-h (6 x 27B rewrite + oracle ~35 min, 2 system runs ~17 min).

## 53. Complex bench: the g (two albums) TRUTHS rewritten - cause, a 27B rewrite of every g program, a fresh blind audit, dev g re-scored; sft_from_truth.py reads the RESULTS 52/53 programs (10-13; eval/complex_bench/{build,gen,common,audit4,sft_from_truth}.py, g53/{diag.py,diag.txt,regen_summary.txt}, programs_g2/, audit4/, runs/cmp/r53_*; tests/test_sft_from_truth.py)
Benchmark construction, all splits (test truths rebuilt; the system was NOT run, scored or looked at on test). Every other
category is byte-identical to the RESULTS 52 files (checked per split).

1. WHY g ALBUMS WERE TOO NARROW. Not compile: g has no strip (STRIP_CATS = d,h), and build.py's RESULTS 52 place matcher
  (accent/case/space-insensitive, place_any) + a new g check (the words state a year but no album has a date) change NO kept
  g truth on any split (FP_CB_GFIX=0 reproduces the old files byte-for-byte; =1 gives the same g answers). So it is
  generation. Old g programs (157 requests, g53/diag.txt, word heuristics): an event/activity named but tested only through
  objects 18, an album AND-ing >= 2 object questions for a named event 18 (roller derby = helmet AND rink, concert = stage
  AND microphone AND crowd, wrestling = ring AND mat AND crowd, the wedding = bride/groom AND church/reception), dates the
  words never state 26 ("My Dubai photos from 2011" = one day and no Dubai; "the orchids vs the pies" = one day in
  Philadelphia), places the words never name 21 ("New York" = Manhattan, "Barcelona" = Barcelones), a metadata question 1
  ("Is the address in Orange County?"). The 10-12 audit: 34 of g's 44 missed photos sit in 7 requests of the proxy kind.

2. FIX. gen.py regen_g: the 27B (thinking on, the original generator) rewrote both albums of all 171 g requests with the SAME
  words, given the digest + every address name (PLACE NAMES) + the draft + 7 rules (two albums in the named order, no
  overlap; an event/activity = ONE direct question + its day(s) when one occurrence is meant, never AND-ed proxy objects;
  named places by their address name, no district or day the words do not say, no metadata questions; a shared condition
  in both albums; no unstated date/place; questions no narrower than the words). Generic examples (no library's words).
  171/171 parsed (16 without thinking after the budget); 161 changed (310 albums); albums with >= 2 AND-ed questions 94 ->
  38; albums with dates 259 -> 157; 183 new questions; the oracle answered only those (325,647 answers;
  4 helper jobs took the tails of slow H100 shards). Overlay programs_g2/ via common.load_programs (FP_CB_G2="" = old).
  New g (heuristics): proxy-only events 18 -> 2, AND-ed proxies 18 -> 1, unstated dates 26 -> 2, unstated places 21 -> 5,
  metadata questions 1 -> 0. g counts: test 54 -> 58, dev 36 -> 35 (3 out, 2 in),
  train 67 -> 62 (dev out: hockey in LA vs Anaheim now 473 photos > 25% of the library; two pairs overlap 22% / 100%).

3. BLIND AUDIT before vs after (audit4/; 20 FRESH g requests, 13 test + 7 dev, none of the 10-12 audit's 20; 469 photos at
  native pixels, 442 labelled album0/album1/neither + 27 unsure; 5 Claude subagents, never shown the key, reference crops for
  Sam/Alex; the main agent labelled 5 of the requests blind: 86/86 agreement, unsure excluded). Rule written before labels:
  the g cut moves 0.5 -> 0.3 only if 0.3 wins F1 on both test-only and dev-only; VERIFIED needs precision >= .8 and recall
  >= .7.
  | truth version | precision | recall | F1 | est. right / wrong / missed |
  |---|---|---|---|---|
  | g1 old | .80 [.65, .93] | .68 [.46, .83] | .73 | 1,762 / 432 / 846 |
  | **g2 rewritten, cut 0.5 = NEW TRUTH** | **.79 [.67, .92]** | **.84 [.75, .93]** | **.82** | 2,190 / 569 / 409 |
  | g2 at cut 0.3 | .75 [.62, .89] | .86 [.77, .95] | .80 | 2,228 / 744 / 375 |
  Test only (13): old .83 / .56 (F1 .67) -> new .79 / .79 (.79); cut 0.3 .70 / .81 (.75). Dev only (7): old .79 / .76 (.77)
  -> new .80 / .89 (.84); cut 0.3 .78 / .89 (.84). Halves: even old .77/.71 -> new .79/.85, odd old .86/.63 -> new .80/.84.
  DECISIONS (by the rule): cut stays 0.5 (0.3 loses on test, ties on dev); g stays PROVISIONAL: recall .68 -> .84 at
  unchanged precision (.80 -> .79, just under the .8 bar). Where the new truth is still wrong (per-request estimates):
  "snowy mountain landscapes from our 2009 ski trip vs 2012" (the rewrite widened the trip to the whole year: 255 wrong
  photos, the old trip window had 236), "dinner photos vs the stone ruins" ("Are there stone ruins?" over the whole
  library: 141 wrong; the old program was one day each: 5 right, 227 missed), words that describe a group taken as AND-ed
  conditions ("rock concerts with the bands on stage and the drum kits" = concert AND band AND drum kit: 127 missed),
  mutual "not" between canal and river (Oxford: 17 missed).

4. DEV g RE-SCORED (cached plans + 9B judge; the 2 requests new to the truth planned and searched once per code version with
  the shipped 4-bit adapter: runs/cmp/r53_new): 49c baseline and RESULTS 51 code give IDENTICAL g albums on every request,
  so g is .728 / .728 on the old truth (36) and .776 / .776 on the new truth (35), diff .000 (compare_53*.txt; RESULTS 51
  never touched g). Dev ALL on the new truths: 49c .658 -> 51 .679 (+.021 [+.009, +.034]; 348 requests).

5. sft_from_truth.py (planner SFT from TRAIN truths) now reads the RESULTS 52/53 programs: the key shows "place_any" and the
  teacher is told that several key albums for one message = their UNION (one album with an "or" question / a place list
  "A and B", or one album per part); the check accepts a union planned as one album (check_union: person, exclusion, an
  "or" question when the parts' conditions differ, a place list covering >= 2 of the parts' places, dates containing every
  part) or pairwise as before, and counts place_any words as the key's places ("and"/"or" allowed). Not run (it needs the
  27B teacher on a GPU); tests/test_sft_from_truth.py (6 cases on synthetic plans) passes; pytest 155/155.

6. CAVEATS. Same model family for rewrite, oracle and judge. The rewrite still ANDs descriptive words and widened one trip
  to a year: g needs a second pass (or a code check: a group named by "trip"/"event" must not be a whole year) before it can
  be verified. Labels are Claude subagents'; labellers disagreed with the words' readings on a few requests (tent vs whole
  Ledyard wedding; "zen garden"; photos with both colours), noted in their label files. ~5.5 GPU-h.

## 54. MILESTONE on the complex bench TEST split, run once: the phone on 10-12 morning (S0) vs what ships tonight (S1) (10-13; eval/complex_bench/m54/{systems,plan_job,search_job,submit_search,collect}.sh -> eval/complex_bench/runs/m54/{compare_test,compare_dev}.txt, m54_<split>_<sys>/score.txt)
RULES KEPT: each system was run ONCE on test; nothing in product code, prompts, cuts or rules was changed as a result; no
individual test request was read (aggregate per-category numbers and the error count only). Test = 548 requests in 20
libraries never used to tune the product; truths as of RESULTS 53 (VERIFIED a, b, d, e, f, h; PROVISIONAL c, g).
SYSTEMS (same 9B judge, Qwen3.5-9B, for both; Python mirror of the phone):
  S0 "phone on 10-12 morning": BASE Qwen3.5-4B on the phone's 4-bit weights (models/qwen35_4b_mlx4sim, no adapter), product
     code a95b214 (src/ identical to origin/main at 10-12 14:00; before track A a67b1f7; old person rule "top half OR P >= 0.7")
     + ONE patch: the RESULTS 49b parity fix (a null field = its default, as Plan.swift always decoded it). Without it the
     Python mirror rejects plans the phone ran: 18 of S0's 698 test raw outputs carry such a null (dev 13/498; S1 0).
  S1 "shipped tonight": the 10-05 planner adapter merged into the same 4-bit weights (models/planner_4b27ball_q4merged),
     code 64709ef (track A + RESULTS 49c person rule + RESULTS 51 fixes; the phone gets the RESULTS 51 Swift with M65).
  Code from sparse git worktrees .cache/wt_m54_s0 (patched) / .cache/wt_m54_s1.
METHOD (as 49-53): plan stage alone (plan_system.py, greedy; test reuses nothing), search stage run_system.py; each library
shard ran S1 then S0 in one job so both read one judge cache (same photo + question -> same answer); score.py final-round
F1 vs the candidate sets; compare.py paired bootstrap over requests (2,000 resamples). Dev re-run the same way for the
reference: S1 dev reproduces RESULTS 51-53 exactly (349/349 requests, every category diff .000; 446/446 plan outputs
reused); S0 dev is NEW (this exact system was never run on dev: 49b's .590 was the base 4B with track A code, old truths).

| cat | test n | flag | test S0 | test S1 | S1 - S0 [95% CI] | better / worse | dev n | dev S0 | dev S1 | dev S1 - S0 [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| a objects/scenes | 95 | verified | .657 | .724 | +.067 [+.024, +.120] | 19 / 8 | 61 | .744 | .755 | +.011 [-.043, +.066] |
| b attributes | 94 | verified | .596 | .604 | +.008 [-.017, +.034] | 10 / 11 | 62 | .608 | .662 | +.054 [+.016, +.100] |
| c person+look | 80 | provisional | .536 | .624 | +.088 [+.044, +.139] | 26 / 7 | 50 | .552 | .580 | +.029 [-.048, +.105] |
| d time/place/event | 100 | verified | .490 | .647 | +.157 [+.088, +.226] | 42 / 14 | 64 | .559 | .722 | +.163 [+.078, +.258] |
| e multi-hop | 16 | verified | .459 | .588 | +.129 [-.029, +.303] | 7 / 3 | 17 | .343 | .636 | +.292 [+.073, +.518] |
| f vague 1 photo | 58 | verified | .670 | .697 | +.028 [-.043, +.099] | 14 / 11 | 33 | .612 | .606 | -.006 [-.096, +.084] |
| g two albums | 58 | provisional | .614 | .729 | +.115 [+.039, +.199] | 18 / 8 | 35 | .665 | .776 | +.111 [+.011, +.216] |
| h follow-ups | 47 | verified | .494 | .583 | +.089 [-.021, +.200] | 22 / 10 | 27 | .448 | .621 | +.173 [+.030, +.325] |
| **ALL** | 548 | | **.575** | **.656** | **+.081 [+.058, +.104]** | 158 / 72 | 349 | .595 | .679 | +.084 [+.054, +.116] |
| **ALL verified** | 410 | | **.578** | **.653** | **+.075 [+.046, +.103]** | 114 / 57 | 264 | .595 | .685 | +.090 [+.055, +.129] |
Better / worse = requests whose F1 moved by > 0.05. ALL final precision / recall: test S0 .601 / .704, S1 .682 / .764; dev
S0 .613 / .727, S1 .694 / .791. ERRORS (requests failing to run, scored F1 0): test S0 0/548, S1 1/548 (a plan-stage
ValueError in one h conversation); dev 0/349 both. Nothing returned: test S0 53, S1 27; dev S0 32, S1 18. f target photo
returned: test S0 50/58, S1 52/58; dev 28/33, 27/33.

READING. On libraries the work never touched, what ships tonight beats the phone's 10-12 morning stack by +.081 F1
[+.058, +.104] (ALL, 548) and +.075 [+.046, +.103] on the verified categories (410); the dev estimate (+.084) held. Per
category: d time/place +.157 on test and +.163 on dev (the largest, CI clear of 0 on both); g +.115 / +.111 (provisional
truth); c +.088 on test (dev +.029, CI includes 0; provisional truth); a +.067 on test (dev +.011, inside noise). b's dev
gain (+.054) did NOT reproduce on test (+.008 [-.017, +.034]); f is flat on both splits; e (16 requests) and h (47) are up
on test but their CIs include 0. Absolute level: S1 is .656 on test vs .679 on dev (S0 .575 vs .595).
WHAT THIS DOES NOT SAY. S1 - S0 bundles three changes (planner adapter, track A + RESULTS 51 code, the 49c fact rule); dev
attributions are in RESULTS 49b (adapter +.043 on the phone's weights), 49c and 51; test was not split. "Truth" = Qwen3.5-27B
candidate sets (audited precision ~.8-.95 per category; same model family as the judge); c and g are provisional. Test
libraries were used in BUILDING the benchmark (cut audits 10-12, the RESULTS 52/53 audits: 12 + 13 test requests labelled
by eye, the system never run), never in tuning the product. Python mirror: the phone's Swift had parity fixtures with it
(GROUND / FOLLOWUP 0 mismatches) but was not run here. ~5.6 GPU-h.

## 55. Complex bench dev, category b (attribute / colour / style): failures by cause, and a pixel veto for photo-level looks TESTED and NOT adopted (10-13; eval/complex_bench/b55/{dump,sheets,fit,veto_eval,pixel_rule,anchor_probe}.py, labels.txt + key.txt, ranking.txt, fit.txt, thresholds.json, veto_eval.txt, compare_prereg.txt)
Dev only; the test split stays closed after RESULTS 54 (nothing run, read or scored on test). System = the shipped stack as
run in RESULTS 54 (m54 dev S1: 10-05 planner adapter on the phone's 4-bit weights + 64709ef code = origin/main src/; 9B
judge). Evidence per photo (dump.py): the S1 dev plans replayed with the CACHED 9B answers only (no model; final round
identical to the m54 run on 62/62 b requests), the oracle's P for every truth question, the PE-Core look score and rank,
pixel statistics. Every b plan is one album with no filter/exclude question, so the final round is exactly "judged
photos with P >= 0.7" and any per-photo rule can be replayed exactly.

1. WHY DEV b FAILS. 14 of 62 dev b requests have final F1 < 0.5; 11 of the 14 are precision failures (returned >> truth).
  Photos viewed at native pixels (DISBench <= 500 px = what the judge sees), sheets SHUFFLED and labelled blind before
  the key was read (labels.txt, key.txt): per request up to 6 returned-not-truth (FP), 6 truth-not-returned (FN), 2 hits.
  125 photos: FP 78 = 14 match / 48 no_match / 16 unsure; FN 23 = 7 match / 12 no_match / 4 unsure; hits 24 = 14 / 5 / 5.
  | request (F1) | primary cause | evidence |
  |---|---|---|
  | overexposed photos of the mountains (.01; 183 returned / 1 truth) | JUDGE | oracle "overexposed?" < .8 on 182/182 FP (median .29); FP 0/6 match by eye (normal snow scenes); the single truth photo is not right either (0/1) |
  | wide skyline shots in the smog (.03; 143 / 2) | JUDGE (+ truth's hidden day) | FP 1/6 match (hikers in front of a hazy city, a clear skyline); 139/141 FP outside the truth's one day |
  | my blurry photos (.18; 104 / 10) | JUDGE | oracle "blurry?" < .8 on 94/94 FP (median .27); FP 0/6 match (fog, dusk, soft); hits 2/2 |
  | women in the red traditional wraps (.19; 65 / 8) | JUDGE | FP 0/6 match (parade costumes, multicolour kangas, a souk) + 2 unsure; the colour is there, the garment is not |
  | the dark night shots of the city (.29; 36 / 6) | JUDGE | FP 1/6 match: theme-park parade floats and fireworks accepted as "a city at night" |
  | the city shots with the golden sunlight (.34; 39 / 8) | JUDGE | FP 0/6 match (midday sun, plain daylight) + 1 unsure |
  | b&w photos of the church interiors (.44; 14 / 4) | JUDGE | FP 0/6 match: an exterior portal, organ pedals, a sepia staircase; 3 toned-monochrome interiors unsure |
  | wedding pictures with the purple lights (.46; 9 / 4) | JUDGE | FP 0/6 match: blue-violet uplights at another wedding (3 unsure); purple-hue pixel share hits .011 vs FP .001 |
  | the blurry action shots (.19; 25 / 39) | TRUTH | FN 0/6 match (sharp players with bokeh, long-exposure waterfalls: the program asks only "motion blur?"); FP 0/7 match too |
  | the pictures with that teal water (.22; 45 / 9) | TRUTH (hidden day) | 39/39 FP outside the truth's one day at Anna Bay; FP 4/6 match (Trevi, a fountain, Niagara) |
  | b&w photos from the London gigs (.32; 175 / 33) | TRUTH | the program ANDs "Is there a stage?", which the words never say; FP 6/6 match |
  | foggy reservoir from that rainy hike in March (.32; 30 / 7) | TRUTH (wrong day) | the rainy hike is 03-29 (umbrellas, raindrops); the truth takes 03-10..03-20; FP 2/6 match (03-29 reservoir in mist) |
  | smooth water from that river trip (.00; 30 / 11) | PLAN (anchor) | "Is this a photo from a river trip?" scores max .651 on the Mostar day (0/47 photos >= .7; anchor_probe.txt) while 20 photos of a 2011 campsite pass: the window opens on the wrong events, the 11 truth photos are never judged |
  | the photos with Mickey Mouse (.29; 1 / 6) | PLAN (code) | ground()'s "real X ... (not a drawing, painting, statue, toy, model or picture of one)" rewrite: costumed Mickey scores .44-.62; FN 4/5 match |
  Totals: JUDGE 8, TRUTH 4, PLAN 2, RANKING 0. The judge's errors are looks accepted loosely: photo quality (overexposed,
  blurry) 2, a colour present on the wrong thing or a neighbouring hue (red garments, blue-violet as purple, sunlight as
  golden) 3, toned/exterior as "black and white church interiors" 1, scene (theme park as city, hikers as a skyline) 2.
  RANKING cannot cause a final-round miss (stream mode judges every in-scope photo). Measured anyway (ranking.txt): AUC of
  the PE-Core look score for truth vs other judged photos, colour words median .923 (27 requests; weakest red neon sign
  .679, magenta lighting .758), black and white / sepia median .988 (one plan with no looks: .500, first answer .59 vs
  final .95), night .965, photo quality .621 (overexposed), .714 (motion blur), .775 (blurry). Cosine ranks colours; it
  does not rank blur or exposure, which matters on the phone where the first answer judges only the head.
  A stricter judge cut is not the fix: dev b .662 at 0.7, .663 at 0.8, .589 at 0.9, .455 at 0.95 (exact replay).
  Why b's dev gain (RESULTS 54: S1 - S0 +.054) did not hold on test (+.008): 2 dev requests carry +.028 of it, both scope
  errors of the base 4B (a self-contradicting filter "Is this photo in color?" on "all my black and white photos" 0 -> .94;
  "late December" planned in the future 0 -> .77); the rest is +/- wording noise.

2. FIX TESTED FOR THE TOP CAUSE: photo-level looks from pixels, as a veto on the judge's yes. Rule written in the journal
  BEFORE the train data was read; thresholds fitted on TRAIN libraries only (24 libraries, pixel stats of 45,982 photos,
  fit.txt), scored once on dev:
  - "black and white" (in black and white, a b&w photo/picture/shot, monochrome, greyscale; not with sepia): at most
    T = .321 of pixels with chroma > 30/255 (97.5th percentile over 299 TRAIN truth photos of 15 black-and-white requests;
    median 0, 90th .025: the 27B truth already counts toned and some colour photos as b&w);
  - "overexposed / blown out": at least T = .0037 of pixels with luminance > 250/255 (10th percentile over 411 TRAIN
    photos the facts tag overexposed; AUC .860 vs untagged);
  - "blurry / blurred / out of focus / motion blur": mean absolute luminance step <= 13.38 (90th percentile over 63 TRAIN
    truth photos of 6 blurry requests; AUC only .671).
  RESULT (veto_eval.txt, compare_prereg.txt; 14 dev requests name such a look: 13 b + 1 d; 2 sepia b requests excluded by the rule): dev b .662 -> .661, -.001
  [-.005, +.003], 1 better / 1 worse; every other category .000; ALL -.000 [-.001, +.001]. Per look: black and white
  +.001 on b (church interiors .44 -> .50, two "all my b&w" +.01/+.03, one -.03: 2 truth photos removed), overexposed
  +.0002 (183 -> 79 returned, still 1 truth photo), blurry -.002 (blurry action .19 -> .07: the veto removes 4 truth
  waterfalls that by eye are not blurry action shots). NOT ADOPTED (the pre-written bar: dev b must improve). Product code
  unchanged; the rule lives in eval/complex_bench/b55/pixel_rule.py, not ported to Swift (no code change -> no parity or
  gate run needed).
  Sensitivity (post hoc, not a candidate): a strict "no colour" b&w cut (<= .01-.10 coloured pixels) LOSES on b (-.013),
  because truth and judge both call toned photos black and white. Of the 12 disputed photos with > 5% coloured pixels
  (viewed blind, bwcolour in labels.txt): the 6 the judge added are colour 4 (desaturated nave, calendar with a red
  sticker, yellow boarding pass, sign on green grass) + toned 2; the 6 in the truth are sepia-toned 3, a colour photo of a
  b&w placard 1, and the d request's 2 colour photos of a black-and-white CAT ("the black and white cat photos": the
  truth reads the cat's fur, the planner wrote "a cat in black and white" and the judge said yes anyway). Overexposed
  only helps at >= .2 clipped (fitted to the single dev truth photo); blur loses at every threshold tried (3-13).

3. READING. b's failures are mostly the judge's loose reading of looks and wrong truths, and pixels settle only the
  definitional part (no colour at all, blown highlights), where the 27B truth is itself loose: a pixel rule cannot gain
  against a truth that counts sepia as black and white, and dev holds only 4 photo-quality requests. The colour failures
  are not pixel problems (the red is in the photo, on a parade costume). Next, by size: (a) truth: b keeps hidden
  days/places by design (RESULTS 49's v4 strip rule), which makes "that teal water" one day of one beach - and two
  programs AND an unstated condition (stage) or drop one (action); a b truth pass like RESULTS 52/53 would move b more
  than any product rule tested here; (b) plan: the anchor that is never found (the RESULTS 51 class: the moment's own
  photos score .65) and the "real X (not a ... toy, model)" rewrite on characters, which are depictions by nature
  (1 dev request; a fix needs a character list or a planner flag, not a regex on capitals: "a real Eiffel Tower" should
  still reject souvenir models); (c) ranking: pixel chroma orders b&w truth photos at AUC .908-1.000 (9 dev requests, median .995),
  no better than PE-Core when the plan has looks (.915-1.000); it matters only for a plan with NO looks (1 dev plan:
  PE-Core .500, chroma 1.000; its first answer .59 vs final .95) - a first-answer effect this final-round benchmark
  cannot score; a "looks must not be empty for b&w" planner/grounding check would do the same without pixels. No model was run: CPU work (cache replay, pixel stats) on GPU-partition
  allocations, ~1.7 GPU-h held with the GPU idle.

## 56. Complex bench: the b (attribute / colour / style) TRUTHS rewritten - cause, a 27B rewrite, a fresh blind audit, b cut 0.8 -> 0.9, b now PROVISIONAL; the shipped stack's dev b level re-measured (10-13; eval/complex_bench/{build,gen,common,audit5}.py, b56/{diag.py,diag.txt,regen_summary.txt,sys_job.sh,dev_b_rescore.txt}, programs_b2/, audit5/, runs/cmp/r56_*)
Benchmark construction, all splits (test truths rebuilt; the system was NOT run, scored or looked at on test - closed after
RESULTS 54). Every other category is byte-identical to the RESULTS 55 files (checked per split); only b and its flag change.

1. WHY b TRUTHS WERE WRONG. Not compile: b has no strip (the 10-12 v4 decision kept hidden days/places for a/b/c/g), and
  build.py's place matcher + stated-year check (B_FIX, as g in RESULTS 53) change NO b truth (FP_CB_BFIX=0 byte-identical,
  =1 the same b answers). So generation. Old b programs (272 requests, b56/diag.py, word heuristics): an album with dates
  the words never state 76, with a place they never name 54, pinned to ONE day with no day word 77 ("the pictures with
  that teal water" = one day at Anna Bay; "the city shots with the golden sunlight" = one day in Melbourne), a need question
  sharing no word with the request 22 ("Is there a stage?" in "the black and white photos from the London gigs", "Is there
  a tree or bush?" in "those hazy mountain shots"); and dropped words ("the blurry action shots" = "motion blur?" only).

2. FIX. gen.py regen_b (the RESULTS 52/53 recipe): the 27B (thinking on) rewrote the album of all 285 b programs with the
  SAME words, given the digest + every address name + the draft + 7 rules (every stated part tested and nothing else; photo
  looks asked of the photo; "the"/"that" alone names no occasion; a named trip/event/place/time = its days or place names;
  literal visible questions). Generic examples (the first draft of the prompt quoted a dev request, replaced before any
  job ran). 285/285 parsed (14 without thinking after the budget); 236 albums changed; albums with dates 140 -> 30, with a
  place 96 -> 42; 170 new question texts; the oracle answered only those (303,996 answers; fp_cb56_regen_0..7).
  New b at cut 0.9 (heuristics): unstated dates 76 -> 4, unstated places 54 -> 3, one-day pins 77 -> 7, unstated need
  22 -> 15 (mostly a sports question for "action"). Counts: test 94 -> 91, dev 62 -> 64, train 116 -> 113.

3. BLIND AUDIT before vs after (audit5/; 20 FRESH b requests, 13 test + 7 dev, none of the 10-12 audit's b requests; 250
  photos at native pixels, 229 labelled + 21 unsure; 4 Claude subagents, never shown the key; the main agent labelled 4 of
  the requests blind: 44/45 agreement). Rules written before the sample was drawn (AUDIT_PROTOCOL.txt): (1) the rewrite
  replaces the old programs only if its F1 >= the old F1; (2) cut 0.8 -> 0.9 only if 0.9 wins F1 on test-only AND
  dev-only; (3) VERIFIED only if the adopted version reaches precision >= .8 and recall >= .7.
  | truth version | precision | recall | F1 | est. right / wrong / missed |
  |---|---|---|---|---|
  | b1 old (as verified since 10-12) | .56 [.42, .79] | .81 [.69, .91] | .66 | 230 / 182 / 55 |
  | b2 rewritten, cut 0.8 | .60 [.49, .74] | .89 [.78, .97] | .72 | 255 / 169 / 30 |
  | **b2 rewritten, cut 0.9 = NEW TRUTH** | **.73 [.62, .90]** | **.79 [.64, .90]** | **.75** | 223 / 84 / 61 |
  Test only (13): old .68 / .83 (F1 .747), rewrite .63 / .86 (.727), rewrite at 0.9 .81 / .76 (.786). Dev only (7): old
  .44 / .77 (.559), rewrite .57 / .93 (.710), at 0.9 .64 / .82 (.716). Halves: even old .42/.81 -> new .62/.79; odd old
  .83/.80 -> new .88/.79. DECISIONS (by the rules): (1) rewrite adopted (.72 >= .66); (2) cut 0.9 (test .786 vs .727, dev
  .716 vs .710 - a .006 margin on dev); (3) b loses VERIFIED: precision .73 < .8 (the OLD truth measures .56 on this
  sample; the 10-12 audit's .78 was 20 other requests). Old-truth errors here: hazy mountain shots (tree/bush + a date
  range: 96 wrong), close-up magenta flowers (22 wrong in every version: the oracle accepts non-close-ups and pink),
  glacier landscapes (19), turquoise water (13). Still wrong in the new truth: blue wedding lights as "purple lights"
  (11), "the photo with the white tent" (10), magenta close-ups (7); misses grow at 0.9 for reflections (14) and glacier
  landscapes (14). b2 at 0.9 drops "the blurry rugby action shots" (empty).

4. THE SHIPPED STACK'S DEV b LEVEL (S1 of RESULTS 54: cached plans + 9B judge cache; the 3 requests new to the truth planned
  and searched once with S1, runs/cmp/r56_new; b56/dev_b_rescore.txt): old truth .662 (62 requests; reproduces RESULTS 54)
  -> NEW truth .662 (64; P .602, R .898); on the 61 requests in both .672 -> .666. With the rewrite at cut 0.8 it would be
  .724 (65): the level depends on the truth cut, and the cut was decided by the rule above on a .006 dev margin. Per
  request the rewrite moves F1 both ways (teal water .22 -> .84, red wraps .19 -> .52, smog skyline .03 -> .42; dark night
  shots .73 -> .35, London gigs b&w .32 -> .13, purple stage lights .67 -> .29). RESULTS 54's test b row was scored on the
  OLD b truth (then flagged verified); it is not re-scored (test closed).

5. CAVEATS. Same model family for rewrite, oracle and judge. b now joins c and g as PROVISIONAL: looks (colour names,
  "close-up", "purple" vs blue) are where the 27B oracle and the labellers part. Labellers measured pixel colour/sharpness
  for some calls (b005 b&w, b018 magenta, b012 hue) - allowed, no web. The session stopped at ~06:30 during scoring and
  resumed at 23:05 (labels, sheets, oracle and builds were complete; scoring re-run, results unchanged by the stop).
  ~4.2 GPU-h.

## 57. Planner track B3: the 4B LoRA retrained on SFT targets from the CORRECTED TRAIN truths - NOT shipped (dev ALL-verified +.007 [-.016, +.033] on the phone's 4-bit weights) (10-14; eval/complex_bench/b57/{sft_job,plan_job,search_job,train_job}.sh, {inspect_targets,fix_targets,make_mix,plan_shapes,per_request}.py, eyecheck.txt, fixes.log, targets_v3.txt -> runs/cmp/r57_{base,q4,fp16}{,_search}, compare_57.txt)
Train for training, dev for selection only; the test split was not built, read, run or scored (closed after RESULTS 54).
Truths: verified a, d, e, f, h; PROVISIONAL b (RESULTS 56), c, g. Ship rule written in JOURNAL.md BEFORE any B3 dev
number: ship iff (1) ALL-verified (202 dev requests) B3-4bit minus shipped-4bit has its 95% CI above 0 OR is >= +.020,
AND (2) no verified category's CI lies entirely below 0, AND (3) the 30-conversation gate is 30/30 on the 4-bit weights.

1. DATA. Train truths = queries/train.json as built for RESULTS 56 (643 requests, 24 libraries; h2/g2/b2 programs, checked
  equal to the overlays). sft_from_truth.py --rules=v2 (RESULTS 49 B2's rules; teacher Qwen3.5-27B-FP8 writes the plan from
  the planner's own prompt with the truth program as a hidden key; grounded with the shipped code and checked against the
  key): 749 turns, 635 pass (a 92/114, b 106/113, c 86/88, d 80/117, e 19/29, f 45/58, g 54/62, h 153/168; B2 had 465/560
  on 18 libraries). Fails: place word not in the key's address 43 ("Santorini" vs Thira), dates not containing the key's
  29, no valid output 19, time words but no dates 13.
2. EYE CHECK BEFORE TRAINING (b57/eyecheck.txt; 25 random passing targets, words + key + grounded plan, raw read too):
  18 right, 7 acceptable but weaker, 0 wrong. Weaker: "biking" asked as "a bicycle", "beach" as "sand and ocean", 2002-2005
  for "2002 2003 and 2005", "snowy hike" without "hike", and 3 of one kind: an anchor + window the key does not have next to
  the album's own question ("waterfall shots from the jungle trip" -> anchor "Is this a photo from a jungle trip?" +
  same_event). Then a search of ALL 635 targets for the B1 kinds: AND-ed descriptions ABSENT (filter_question in 0 targets
  outside h; B1 had b .39 / c .56 / f .87; h filters are follow-up narrowings, 0 on turn 1); anchors the key lacks 48 (B2
  21). Three systematic conversion errors, fixed in CODE before training (fix_targets.py, every change in fixes.log):
  F1 unstated exclusions (17 albums, 12 rows): the key check demands exclusion parity and the g truths make the two albums
     mutually exclusive by construction ("My dog photos versus my cat photos" -> cat album excludes dogs; the words never
     ask) -> exclusion removed when the words carry no negation / "alone" cue (not on a turn later turns build on).
  F2 anchors the key lacks next to the album's own question (17, turn 1): the B1 "trip as a window" kind (the check lets it
     pass whenever the key has days or a place). Dev evidence before training: the shipped adapter's own plans of this shape
     in a/b/f score .00-.67 (8 requests) -> anchor + window removed, the album's question kept. NOT touched: 31 anchor-only
     "photos from our trip to X" plans (d, and h conversations built on them) = the shipped adapter's plan shape for trips
     (dev d, 14 such requests, F1 .73 = d overall).
  F3 moment restated (9): the key wants everything in a window but the target also asks "Is this a photo from a curling
     match?" -> judge question removed (B2 fixed 5 such by hand; ground()'s RESULTS 51 rule D catches only some phrasings).
  + 1 target whose notes cite "the answer key" dropped. Result: 634 rows (534 requests, 24 libraries), 37 changed.
3. TRAINING. Mix as B2: the 4,426 old 27B distillation train rows + 634 x3 = 6,328 rows (data/public/distill/
  planner_cbmix3.jsonl, not in git). RESULTS 31 recipe (LoRA r16/alpha32 on the text layers, lr 1e-4, 2 epochs, accum 8):
  1,582 steps, 4.0 s/step on an RTX PRO 6000 (fla overlay), final loss ~.045. Merged fp16 (models/planner_4bcb3_merged) and
  into the phone's 4-bit weights (scripts/merge_lora_into.py on models/qwen35_4b_mlx4sim -> models/planner_4bcb3_q4merged).
4. GATE (eval_planners.py, 30 scripted conversations, shipped code): B3 4-bit 30/30, B3 fp16 30/30 (shipped adapter 4-bit,
  same run: 30/30).
5. DEV (351 requests, 13 libraries; code = the shipped 64709ef src via the .cache/wt_m54_s1 worktree, = HEAD src; plan stage
  alone, plan_system.py, 0 errors in all three; search run_system.py with the 9B judge and the shared answer cache; score.py
  final-round F1 vs the candidate sets; compare.py paired bootstrap over requests, 2,000 resamples). Reference = the shipped
  10-05 adapter on the 4-bit weights re-planned on the current dev.json (449/449 raw outputs reused from m54 dev S1 +
  r56_new: it reproduces RESULTS 54's dev S1 on every unchanged category exactly, b .662 as RESULTS 56).
  | cat (n) | flag | shipped 4-bit | B3 4-bit | B3 4-bit - shipped [95% CI] | better / worse | B3 fp16 | B3 fp16 - shipped [95% CI] |
  |---|---|---|---|---|---|---|---|
  | a objects/scenes (61) | verified | .755 | .754 | -.001 [-.032, +.040] | 7 / 9 | .748 | -.007 [-.042, +.026] |
  | b attributes (64) | provisional | .662 | .657 | -.005 [-.036, +.023] | 9 / 10 | .667 | +.005 [-.027, +.036] |
  | c person+look (50) | provisional | .580 | .576 | -.004 [-.050, +.043] | 7 / 7 | .571 | -.009 [-.061, +.042] |
  | d time/place/event (64) | verified | .722 | .713 | -.008 [-.037, +.014] | 9 / 5 | .727 | +.006 [-.029, +.039] |
  | e multi-hop (17) | verified | .636 | .712 | +.077 [-.069, +.266] | 2 / 2 | .689 | +.053 [-.125, +.236] |
  | f vague 1 photo (33) | verified | .606 | .625 | +.020 [-.050, +.091] | 5 / 3 | .661 | +.055 [-.011, +.133] |
  | g two albums (35) | provisional | .776 | .821 | +.045 [+.007, +.086] | 7 / 1 | .821 | +.046 [+.009, +.087] |
  | h follow-ups (27) | verified | .621 | .627 | +.006 [-.049, +.068] | 6 / 5 | .627 | +.006 [-.068, +.076] |
  | **ALL (351)** | | .679 | .686 | +.007 [-.009, +.025] | 52 / 42 | .691 | +.012 [-.006, +.031] |
  | **ALL verified (202)** | | .692 | .700 | **+.007 [-.016, +.033]** | 29 / 24 | .706 | +.014 [-.014, +.042] |
  Better / worse = requests whose F1 moved by > 0.05. Nothing returned 18 -> 14 (fp16 15); f target photo returned 27 -> 29
  of 33 (both). 4-bit vs fp16 of B3: ALL -.005 [-.019, +.009] (same within noise, as B1/B2).
6. DECISION (by the pre-written rule): NOT SHIPPED. (1) fails: ALL-verified +.007, CI includes 0 and below +.020; (2) holds
  (no verified CI below 0); (3) holds (30/30). No MLX export; the app keeps the 10-05 adapter (planner_4b27ball). Models
  kept: models/planner_4bcb3_{lora,merged,q4merged}.
7. WHAT MOVED (per request, plans side by side: b57/per_request.py on r57_base vs r57_q4). Plan SHAPES barely changed
  (b57/plan_shapes.py, final-turn albums: filters outside h ~0 in both; f anchors .12 -> .03, d anchors .30 -> .20, h
  exclusions .27 -> .23); 348/351 final plans differ in text. The gains: e "anchor not found" moments (RESULTS 51's class)
  re-worded and found (colonial drummers 0 -> 1.00, Everest Base Camp with prayer flags 0 -> 1.00); g two-album plans
  without the old adapter's cross filters / restated place questions (Sam hospital gown vs blue cap .50 -> .91: the old
  plan AND-ed each album with the other's condition; Trinidad vs Viñales place-only .79 -> .96; elephants vs monkeys .69 ->
  .96); f trip anchors dropped (rooster weather vane 0 -> .67). The losses: judge-question WORDING the 9B answers
  differently, structure unchanged ("Is this a photo of a trail map?" -> "Is there a trail map?", which grounding turns into
  "a real trail map ... (not ... picture of one)": .79 -> .30, the same loss B2 had; "surfing with the sunset" as one AND
  question .67 -> .11; "Mickey Mouse" -> "a Mickey Mouse" .33 -> 0; "the Disney photos from the end of February" window
  02-25 -> 02-27 .74 -> 0), one e moment turned into a plain question (bouncy castle day 1.00 -> .52). So B3 trades
  wording both ways at the request level; the one CI-clear gain is g (provisional truth, partly the g2 rewrite's own
  convention), and e (17 requests) / f (33) are up inside noise - the same picture as B2 (RESULTS 49) on cleaner truths.
8. CAVEATS / NOT DONE. One training seed, one mix ratio (x3), not swept (dev is for selection; a sweep would spend it). The
  fixes F1-F3 were decided from train targets plus the shipped adapter's DEV plan shapes for the F2 kind (aggregate per
  shape, 8 requests) before B3 was trained. Same model family for teacher, truths, oracle and judge. The "real X (not a
  picture of one)" grounding rewrite hurts things that ARE pictures or characters (trail map, Mickey Mouse): a product-code
  candidate, not a planner one. ~3.7 GPU-h (27B targets 0.2, training 1.8, plan stages + gates 0.3, search 1.4).

## 58. Training the phone judge from a stronger open teacher: the TEACHER CHECK fails on the eye labels, so no distillation (10-14; eval/judge_teach/{eyeset,judge_eye,gate,sheet}.py -> eval/judge_teach/gate_teachers_{v0,v1,think,unsure_nomatch,397b}.txt; built, not run: questions.py, pairs.py, label.py, train.py, merge.py)
Goal: make the phone judge (Qwen3-VL-4B, the phone's real 4-bit language tensors + 8-bit tower = models/q3vl4b_v8) stricter
on the kinds that dominate its errors (RESULTS 34/46/49b/55: looks and colours accepted loosely, partial person-attribute
matches, subject vs setting, look-alike scenes) without losing recall, by LoRA on the vision tower AND the language model
from a stronger teacher. Step 1 (pre-registered gate, JOURNAL 10-14 00:30): a teacher must beat the phone AND the 9B on
precision at equal recall on the disputed kinds, on the existing blind eye labels; otherwise distillation cannot help.

EYE SET (eyeset.py; existing labels only, the product's questions and cuts of today): 4,373 rows. Held-out libraries
(section 26's 8 incl. audit4 + all complex_bench dev/test) 3,698 rows; ev16 (the libraries a training pool may use) 675.
Groups: D1 = 6 subject searches (beach "clearly visible", food, sunset, flowers, cat, church), D2 = 7 person facts with the
red box (M45 + 49c sheets), D3 = complex dev b (b55, 14 requests, question = the shipped plan's), OBJ = dog/car/bicycle/boat.
Judges run with the product pipeline (VLLMJudge prob mode: 896-px thumbnail, JPEG q88, P = py/(py+pn)).
RULE: per search, R* = the phone's recall at its product cut (unsure dropped); every judge (phone included) takes its
highest threshold reaching R*; wrong photos kept summed over the group; paired bootstrap within search (1,000x).

(1) PRECISION AT EQUAL RECALL, held-out, D1+D2+D3 pooled (27 searches, 642 match rows)
| judge | precision | vs phone [CI] | D1 subjects | D2 person facts | D3 cb_b |
|---|---|---|---|---|---|
| phone q3vl4b_v8 (real 4-bit + 8-bit tower) | .941 | - | .942 | .949 | .897 |
| phone q3vl4b_real (bf16 tower) | .937 | -.004 [-.012, +.004] | .935 | .949 | .929 |
| Qwen3.5-9B (bf16) | .935 | -.006 [-.047, +.017] | .944 | .940 | .788 |
| Qwen3.5-27B-FP8 | .950 | +.009 [-.049, +.026] | .946 | .969 | .963 |
| Qwen3.8-27B-FP8 (Aug 2026) | .920 | -.022 [-.062, .000] | .928 | .895 | .867 |
| Qwen3.5-122B-A10B-FP8 (TP 2) | .944 | +.003 [-.063, +.026]* | .950 | .959 | .812 |
| ENS mean logit (27B + 122B) | .953 | +.012 [-.051, +.036]* | .952 | .969 | .929 |
| Qwen3.5-27B, thinking 2,048 tok | .884 vs .948** | -.064 [-.119, -.030]* | | | |
| Qwen3.8-27B, thinking | .859 vs .948** | -.090 [-.407, -.065]* | | | |
| Qwen3.5-397B-A17B, GPTQ-Int4 (TP 4)*** | .917 | -.025 [-.071, +.015]* | .922 | .895 | .897 |
| ENS mean logit (27B + 122B + 397B) | .940 | -.001 [-.057, +.032]* | .942 | .931 | .929 |
*99.4% CIs (Bonferroni over the 8 pre-registered candidates). **thinking ran past 2,048 tokens on 324 (27B) / 24 (3.8-27B)
of 4,373 rows (no answer); those rows are dropped for every judge in that comparison (3,143 held-out rows, phone .948). ***The pre-registered FP8 checkpoint (406 GB)
needs 4 H200 or 8 RTX PRO 6000 in one node and did not start in ~1 h; the Int4 release (236 GB) raced next to it ran first
(a stated deviation; 4-bit weights cost a large model little, but it is not the FP8 model).
Sensitivity, unsure = no match (the labellers' protocol): phone .820, 27B .850 (+.029 [-.055, +.049]), 122B .822, ENS
.837: same verdict. Within-search AUC is .994-.996 for every non-thinking judge (the labelled sets hold many easy no's).

(2) WHERE THEY DIFFER (equal recall, wrong photos kept, held-out): food phone 12 -> 27B 4 (9B 7); sunset 11 -> 7 (9B 4);
beach 3 -> 3 (9B 7); flowers 2 -> 2; cat 0 -> 0; church 1 -> 11 (the 27B's P is compressed: real churches .82-.97,
non-churches .80-.91; the phone gives real churches 1.000); person beach 5 -> 2, outdoors 0 -> 1; D3 3 -> 1. Viewed at
native pixels (.cache/tmp/r58_sheets, 24 tiles): where the 27B rejects a phone yes (P < .5) it is right 8 of 12 (a
dessert-bar sign as "food", a non-beach portrait as "at the beach", fog as "blurry", no golden light, two "eating"
crops where nobody eats, a campsite as "smooth water", a sepia staircase as "b&w church interior"), wrong 2 (a sunset over
surf as "beach clearly visible", a memorial with flowers), unsure 2; where it accepts a phone no (P >= .9), the phone
also leans yes in 10 of 12 (P .94-.999, just under the .999 cut; the 27B says .90-.96): a scale difference, not a
different reading. The 397B's
own extra yeses (phone no, 397B P >= .5; 12 of 220 viewed): 1 match, 5 no_match (a family dinner table as "food", a
daytime sky over a monument and a daytime sky reflection as "sunset", a flooded picnic table as "beach", a sharp walking
dog as "blurry due to motion"), 6 unsure: the same setting / look-alike errors the phone is criticised for.

(3) SCALE. At the PRODUCT cuts the big models keep almost no subject photo (9B 0 of 502 D1 matches, 27B 0, 122B 20,
3.8-27B 93): their P for real sunsets / food sits at .9-.97, the phone's at .999-1.000. The cuts (.999 / .9999) are the
phone judge's own scale; any distilled judge must be trained on a REMAPPED target (teacher order, phone scale) or the
cuts must move. Thinking does the opposite: after reasoning the answer is near-binary and much LOOSER (D1 at the cut:
27B-think keeps 498 right / 210 wrong vs the phone's 472 / 32).

VERDICT (pre-registered rule): no candidate beats the phone and the 9B at equal recall on the disputed kinds; the best,
ENS(27B+122B), is +.012 [-.051, +.036]; model size does not move it (9B .935, 27B .950, 122B .944, 397B .917; phone
.941). The labelled photos were drawn from the phone's and the 9B's outputs (plus stratified rejections), so a teacher's
own extra yeses are mostly unlabelled and do not count against it: the test favours the teachers, and they still do not
win. So the teacher is not better on what the existing eye labels measure, and distillation from these teachers cannot
be expected to make the phone stricter where it matters. STOPPED before teacher labelling and training (no student exists; the
ship criterion and the disagreement-audit amendment were written but never evaluated). The phone judge (q3vl4b_v8) stays.

Leads (not tested, would need their own pre-registration on fresh labels): the 27B is better on food (12 -> 4) and
sunset (11 -> 7) and on person facts at the .7 cut (27B +.053 precision / +.029 recall, 122B +.054 / +.038, ENS3 +.061 /
+.029; none significant), but worse on church; a per-kind teacher (food/sunset/person facts only) could be tested with a fresh blind audit of phone-vs-teacher
disagreements on held-out libraries (the eye set holds only 75 phone-kept wrong photos in the disputed groups, so it
cannot resolve gains below ~.03). The b (attribute/colour) group has 35 match rows: too small to say anything.

Built and kept (public data only): 5,160 training questions in the product's forms (318 object, 180 subject, beach, 390
red-box person, 55 photo-level looks, 4,216 planner-written from the distillation plans + complex_bench TRAIN plans and
TRAIN oracle questions); 332,456 (question, photo[, face]) pairs from the 16 ev16 DISBench libraries (the 656 eye-labelled
photos there and 331 near-duplicates of them, PE-Core-L cosine >= .92, excluded), the Open Images + IMDB test library (39 eye-checked photos
excluded) and 17,594 Commons iPhone photos; 67,540 judge images packed exactly as the product feeds the judge (6.35 GB,
data/public/judge_teach/, git-ignored). train.py (vision + language LoRA, soft BCE + optional logit-gap term, hard pairs
x3, DDP) and merge.py (8-bit tower, runtime language adapter, re-quantized 4-bit variant, delta-kept report) are written
but were never run. GPU: 3.36 GPU-h (eye-set runs of 10 judges: phone x2, 9B, 27B x2, 3.8-27B x2, 122B on 2 GPUs, 397B
on 4; candidate pairs) + 1.4 CPU-job hours (pairs, packing, downloads). Downloaded for this test (lab storage, unused
now): Qwen3.8-27B-FP8 31 GB, Qwen3.5-122B-A10B-FP8 127 GB, Qwen3.5-397B-A17B-FP8 406 GB (never served), -GPTQ-Int4 236 GB.

## 59. Can the query-time judge go? FACTS written once at index time + a small trained SCORER at query time (10-14; eval/facts_scorer/{data,train,scorers,predict,replay,eval_photo,eval_eye,eval_eye_dev,judge_truth,oracle_plan,phone_facts,index_cost,sheets}.py -> eval/facts_scorer/{photo_main,photo_zs,photo_B_phone,e2e_main,e2e_B,e2e_phone_{ce,ens,tv},eye_qn,eye_B,eye_dev_B,plan_agreement,index_cost}.txt)
Question (decides the architecture): answer a search with NO per-photo judge at query time. Index time = a VLM writes
structured facts per photo (the 27B's complex_bench schema: caption, objects, people with clothing, scene, activities,
text, colours, lighting, quality/style, event...) + the stored PE-Core-B-16 vector; query time = a small trained scorer
maps (judge question, stored data) -> P(yes) for every photo. Public DISBench complex_bench only. TEST closed (not read,
run or scored). Scorers train ONLY on TRAIN libraries (27B oracle P as soft target); choices on held-out TRAIN fold 0
(6 libraries); DEV = evaluation. Reference judge = the shipped 9B (Qwen3.5-9B).
Disclosures: (i) the question-wording fix (QNORM below) was decided from the TRAIN-library eye sets before any dev replay
was scored; (ii) "B" (metadata + planner-wording training data) was designed AFTER reading the worst dev requests of the
first replays, so B's dev numbers are optimistic in the RESULTS 49 B2 sense (training data still TRAIN-only); (iii) the
small writers' decoding settings were picked on 256 dev photos by JSON parse rate and length only (no labels).

SCORERS (query-time cost per 187,084-photo search, ESTIMATES, not measured on the phone)
  ce12   cross-encoder MiniLM-L12 (33M params) reading "question [SEP] facts text" per photo: ~190 tokens x 42 MFLOP =
         ~8 GFLOP per photo -> ~1.5 PFLOP per question = minutes on the phone, NOT milliseconds (top-2,000 rerank ~16 TFLOP).
  towervec12  two towers: facts text -> MiniLM-L12 vector ONCE at index time (768 B/photo fp16 = 144 MB); question ->
         vector once per search; MLP over [f, q, f*q, v, t, v*t] (v, t = the phone's B-16 image / text vectors); first
         layer folds per search (RESULTS 48): ~0.8 M MAC per photo = ~0.3 TFLOP per search (~0.3 s, same estimate basis as 48).
  tower12 (facts only), vec (B-16 vectors only = RESULTS 48's input retrained on these labels), zero-shot (BM25 on facts,
         MiniLM cosine, best single fact phrase, PE-Core cosine), ens = equal-weight logit average of ce12 + towervec12.
  B      = facts text + the photo's own place (reverse-geocoded GPS) and month, AND 1,307,619 extra TRAIN pairs: the 27B
         oracle on the 617 questions the SHIPPED planner + grounding actually ask for the 643 TRAIN requests (23 of 24
         libraries; one file was corrupted by two writers and dropped).
  QNORM  grounding rewrites every plain object question to "Is there a real X anywhere in this photo (not a drawing,
         painting, statue, toy, model or picture of one)?" (38% of dev judge calls); 0 of 1,189 training questions have
         that form, and every first scorer scored ~0 F1 on the eye sets' object searches (dog / car / bicycle / boat).
         QNORM strips the parenthetical + "anywhere" for the scorer (-> "Is there a real X in this photo?", a train form).

1. PER PHOTO, DEV truth questions vs the 27B oracle (1,288,577 pairs: oracle P >= .8 match 89,901, < .2 non-match; the
   band between dropped; 732 questions with both). The 9B judge was run on exactly these pairs (product prompt/encoding).
   | method | AUC | macro AUC | P | R | F1 at .7 | precision at the 9B's recall |
   |---|---|---|---|---|---|---|
   | 9B judge (pixels) | 1.000 | .999 | .979 | .976 | .978 | .979 |
   | ce12B (27B facts + meta) | .989 | .972 | .975 | .789 | .872 | .370 |
   | ce12 | .989 | .971 | .980 | .775 | .865 | .360 |
   | towervec12B | .988 | .974 | .926 | .770 | .841 | .366 |
   | towervec12 | .986 | .975 | .925 | .764 | .837 | .338 |
   | tower12 (facts only, two towers) | .974 | .955 | .923 | .724 | .812 | .187 |
   | vec (B-16 vectors only) | .980 | .965 | .881 | .689 | .773 | .292 |
   | zero-shot: best fact phrase / MiniLM / BM25 / PE-Core cosine | .936 / .877 / .847 / .875 | | | | .537 / .302 / .208 / .284 | |
   Read: the facts scorers are as PRECISE as the judge (.975-.980) but find ~79% of its matches; to reach the judge's
   recall their precision falls to .37. Zero-shot matching of words against facts does not work (F1 <= .54): training
   is required. Held-out TRAIN fold (selection): ce12 .859, ce6 .856, towervec12 .821, tower12 .791, vec .773.
   By question kind (ce12B F1 vs the 9B's): scene/place .968 vs .986, object .925 vs .983, activity .866 vs .978,
   text/sign .830 vs .953, photo look (blur, b&w, haze, light) .810 vs .976, person attribute without box .769 vs
   .980, colour .755 vs .942, red-box person look .736 vs .977 (1,599 pairs), COUNT .468 vs .984 (10,395 pairs; recall
   .306). The facts' gaps, viewed at native px (24 disagreements, .cache/tmp/r59_sheets/ce12_*): scorer misses = details
   the facts never wrote (thatched huts behind the Maasai, trees reflected in a canal, water drops on metal, white neck
   rings, a castle named only as "church spire"; 2 of 12 were judge/oracle stretches); scorer false yes = a colour or
   word in the facts bound to the wrong thing ("purple" stripes on a puppet -> purple lighting, a pink coat -> red
   jacket, "white jersey" on a skater -> the goalie, brick wall -> stone wall, "climbing" hikers -> bouldering).

2. END TO END, DEV (351 requests; the shipped stack's plans runs/cmp/r57_base unchanged; EVERY judge call answered by the
   scorer through the patched engine._judge_rows - the patch reproduces r57_base_search exactly when it serves the 9B
   answers: 351/351, every category .000; final-round F1, paired bootstrap; ref 9B ALL .679, verified .692)
   | cat (n) | 9B | ce12 raw wording | ce12 QNORM | towervec QNORM | ens QNORM | ce12B | towervec12B | ensB |
   |---|---|---|---|---|---|---|---|---|
   | a objects (61) | .755 | .542 | .711 | .735 | .735 | .763 (+.008) | .750 | .762 |
   | b attributes (64) | .662 | .448 | .608 | .544 | .638 | .669 (+.007) | .605 | .662 |
   | c person+look (50) | .580 | .432 | .546 | .514 | .582 | .574 (-.007) | .473 | .576 |
   | d time/place (64) | .722 | .619 | .630 | .730 | .680 | .728 (+.006) | .734 | .735 |
   | e multi-hop (17) | .636 | .783 | .783 | .542 | .783 | .724 (+.088) | .441 | .665 |
   | f vague 1 photo (33) | .606 | .556 | .678 | .522 | .634 | .755 (+.149 [+.021, +.281]) | .378 | .612 |
   | g two albums (35) | .776 | .596 | .641 | .722 | .685 | .700 (-.075 [-.153, +.002]) | .704 | .705 |
   | h follow-ups (27) | .621 | .331 | .350 | .500 | .353 | .487 (-.134 [-.250, -.028]) | .513 | .488 |
   | **ALL (351)** | **.679** | .525 | .619 | .619 | .644 | **.683 (+.004 [-.026, +.034])** | .608 (-.071 [-.100, -.041]) | .667 (-.012 [-.040, +.017]) |
   | ALL verified (202) | .692 | .561 | .638 | .651 | .654 | .711 (+.019 [-.022, +.062]) | .626 | .684 |
   Nothing returned: 9B 18, ce12B 19, towervec12B 31; f target returned 27 / 28 / 16. Without QNORM or B every scorer
   loses .13-.19 (object questions in the product's wording score ~0). On the planner's own questions the scorers agree
   with the 9B far less than on truth questions (plan_agreement.txt, 418,257 judged pairs, label = 9B >= .7: ce12 QNORM
   F1 .671, towervec .706; the template questions .55-.56) - which is what B fixed.
   Where ce12B still loses (worst dev requests, read with the judge's and the scorer's yes counts): OR-questions that
   follow-ups build ("inside a house or in the grass" 0/27 vs "inside a house" 9/27; the 9B 17/26), relations and bound
   attributes ("white sports car next to a taxi", "pink gear vs orange gear" 0/1,867 vs the 9B's 272, "groom with a yellow
   flower on his lapel", "pie with the writing on it"), trip membership the pixels show ("taken during a safari" 21/855 vs
   399/855), and anchors of e requests whose one photo the facts describe differently ("the blue wardrobe": 0/900).
   CAVEAT, the big one: the benchmark's truth IS the 27B oracle, and the scorer is trained to imitate the 27B, so it has an
   edge the 9B lacks here. Hence the eye checks below.

3. EYE LABELS (truth = blind labels; unsure = no match; cut .7)
   | set | 9B judge | phone judge (stored) | ce12B | ce12 QNORM | towervec12B | towervec12 QNORM | vec |
   |---|---|---|---|---|---|---|---|
   | E1 = RESULTS 45 sample (746 photos, 427 match, TRAIN libraries, fold models), P / R / F1 | .851 / .991 / .916 | - | .878 / .927 / .902 | .877 / .920 / .898 | .836 / .869 / .852 | .830 / .902 / .864 | .841 / .857 / .849 |
   | E2 = RESULTS 34 stratified (1,950 photos, 220 match, 4 TRAIN libraries; weighted population estimate) | .843 / .947 / .892 | .835 / .929 / .879 | .854 / .851 / .853 | .862 / .834 / .848 | .855 / .755 / .802 | .843 / .773 / .807 | .803 / .577 / .671 |
   | dev requests, system level (478 photos in 32 requests: b55, audit3/4/5 dev parts; NOT uniform) | .628 / .833 / .716 | - | .799 / .606 / .689 | - | .747 / .655 / .698 | .663 / .631 / .646 | - |
   Per search (E2 F1, 9B / ce12B): dog .99 / .93, car .86 / .81, bicycle .92 / .93, food .78 / .77, beach .72 / .70,
   sunset .63 / .67. Read: against EYES the facts scorer is as precise as the judge but recalls less (E2 .851 vs .947);
   the gap sits in dog and car (objects the facts leave out when they are not the subject). Fold models: a library's
   photos are scored only by a model that never trained on it.

4. PHONE-SIZE FACTS (step 2; the gate "close to the judge" is met by ce12B: ALL +.004)
   Writers at the phone's simulated 4-bit weights (sim_mlx_quant), same prompt, TRAIN+DEV (70,580 photos each); scorers
   retrained in the B configuration on each writer's facts (TRAIN only) and evaluated on DEV the same way:
   | writer (decoding) | out tokens / photo mean (p90) | at token cap | JSON parsed | per photo F1 ce12B / towervec12B | END TO END ce12B / ensB / towervec12B (ALL; 9B .679) |
   |---|---|---|---|---|---|
   | 27B (reference) | ~243 | - | - | .872 / .841 | .683 / .667 / .608 |
   | Qwen3-VL-4B (rep 1.05) | 271 (331) | 2.0% | 98.0% | .847 / .829 | .593 / .614 / .601 |
   | Qwen3-VL-4B SHORT keyword line (rep 1.05, cap 160) | 126 (160) | 52% | n/a | .847 / .824 | .600 / .623 / .604 |
   | Qwen3-VL-2B (rep 1.1) | 404 (700) | 32% | 66% | .809 / .826 | .565 / .598 / .585 |
   | Qwen3.5-0.8B (rep 1.1) | 421 (700) | 13% | 84% (own schema) | .792 / .819 | .497 / .555 / .587 |
   | (vectors only, no facts) | 0 | | | - / .773 | - / - / .564 |
   Paired vs 27B facts, ce12B ALL: 4B -.089 [-.117, -.063], 4B short -.083 [-.112, -.057], 2B -.118, 0.8B -.186; best
   phone row ensB 4B-short -.056 [-.087, -.025] vs the 9B. The per-photo drop is small (.872 -> .847) but the "that one
   photo" (f .755 -> .401) and multi-hop anchor (e .724 -> .369) requests collapse: the 4B names the distinctive thing
   differently or not at all (15 f targets the 27B-facts scorer found and the 4B-facts scorer lost; 6 viewed at native
   px, f_4b_miss_0.jpg, the rest read from the facts text: "red emergency box" -> "red mailbox", "gold float" ->
   "golden archway", hyenas -> "wild boars", devil horns + yellow vests -> "orange vests", rooster weather vane -> "crow
   on a weather vane"). Greedy decoding is unusable for small
   writers: Qwen3-VL-2B loops on 79% of photos (one phrase to the 700-token cap), 4B on 19%; repetition penalty 1.05 fixes 4B
   (255/256 parse in the sweep), not 2B. A 27B-trained ce12 reading 4B facts without retraining: per photo .813 (vs .865).
   INDEX COST on the phone (index_cost.txt; tokens MEASURED; seconds ESTIMATED: per photo = 1.7 s cool / 4.0 s hot
   (the measured judge call, RESULTS 44) + out tokens / r; r = decode speed, NEVER MEASURED on the iPhone: assumed 15 or
   30 tok/s for the 4B, halved hot; 2B x1.8, 0.8B x3.5):
   | writer | s/photo cool (r=15 / 30) | hours for 187,084 photos, cool | hot |
   |---|---|---|---|
   | 4B full facts | 19.8 / 10.7 | 1,027 / 558 | 2,086 / 1,147 |
   | 4B short line | 10.1 / 5.9 | 525 / 306 | 1,080 / 644 |
   | 2B | 16.7 / 9.2 | 866 / 477 | 1,763 / 985 |
   | 0.8B | 9.7 / 5.7 | 505 / 297 | 1,042 / 625 |
   For scale: one judge pass over the whole library at the measured cool rate = 89 h. Facts at these lengths are 13-87 days
   of continuous phone compute for Reza's library: not shippable as written.

5. VERDICT
   - The INFORMATION is mostly there: a small reader over 27B facts (+ the photo's place/month), trained on the
     product's own question wording, matches the 9B end to end on dev (ALL .683 vs .679, +.004 [-.026, +.034]) and is
     within -.007 or better on a, b, c, d, e, f; it still loses g (-.075) and h (-.134). Against eyes it is as precise as
     the judge but recalls less (E2 F1 .853 vs .892; R .851 vs .947). Partly the benchmark favours it (truth = 27B).
   - NOT in milliseconds: that reader is a transformer pass per photo per question (~1.5 PFLOP per 187k-photo search).
     The millisecond scorer (two towers, ~0.3 TFLOP/search) loses -.071 (ALL .608), -.11 to -.23 on c, e, f.
   - NOT with phone-written facts: 4B facts lose another -.083 to -.089 end to end (f and e collapse), 2B/0.8B more, and
     writing them costs an estimated 300-1,000+ phone-hours for 187k photos (decode speed unmeasured).
   - So the product cannot drop the query-time judge today. It can SKIP it for: plain object / scene / activity
     requests and time/place requests (per photo scene .968, object .925 for ce12B) and use the
     scorer as the ranker and first screen everywhere. (d: .71-.75 for every B scorer on 27B or 4B facts vs the 9B's
     .722; a: ce12B +.008, towervec12B -.005.) Requests that still need pixels at query time: COUNTING (per-photo
     F1 .47), a colour or attribute bound to a specific thing or person ("pink gear", "yellow flower on his lapel", red-box
     person looks .74), relations ("next to the taxi"), small background details (thatched roof, reflections, text on a
     pie), photo-level looks (haze, reflection, sunset light: .81), the one-photo (f) and anchor (e) requests on phone-
     written facts, and follow-up OR/negation questions (h).
   - TRAIN NEXT (in order): (1) measure decode tok/s of the 4B/2B on the iPhone (MAC_INBOX) - it decides whether any
     generated facts are affordable; (2) a cascade instead of a replacement: towervec over the library (~0.3 s), the
     facts cross-encoder on the top ~2,000 (~16 TFLOP, seconds), the judge only for the kinds above (route by question
     kind) - replays of this are CPU-only with the harness built here; (3) a phone facts writer DISTILLED from the 27B
     facts in the short keyword format (the 4B short line already equals the 4B full JSON per photo at half the tokens),
     targeting ~60 tokens; (4) planner/grounding: split follow-up "A or B" questions into parts (max P) and route trip
     membership to the place/date metadata instead of a judge question.
   GPU: ~34 GPU-h (9B on 1.38M dev pairs 13 jobs, 27B on planner questions, 30 scorer trainings, 5 x 70,580 phone facts,
   ~70 replay shards). Code: the shipped stack m54 S1 (.cache/wt_m54_s1 = 64709ef; RESULTS 60's later grounding change is
   not in these replays). Not pushed.

## 59b. The CASCADE: millisecond scorer everywhere, facts reader on the top K, the 9B judge only for the kinds that need pixels (10-14; eval/facts_scorer/{route,cascade,cascade_report}.py, cascade_job.sh -> eval/facts_scorer/cascade_report{,_j}.txt, replay/cascade_{27b,4b}{,_j}/*/compare.txt)
Dev only (test closed). Code = main WITH RESULTS 60 grounding (frozen copy .cache/wt_r59b/src at c1a67a9); plans = the
shipped planner's r60v2_base (351 requests); REFERENCE = the all-judge shipped stack r60v2_base_search (ALL .680). The
harness serving the 9B's cached answers reproduces it exactly (351/351; every category .000; passthrough_r60).
DESIGN, PRE-REGISTERED (JOURNAL 10-14 ~03:55, before any cascade number): every engine judge call is routed by
route.py from the question text the engine passes (+ the red-box crop flag):
  JUDGE (9B; cached answers, a lazily loaded 9B for cache misses) if: red-box person, count, text/sign, photo look
  (blur, b&w, close-up, reflection, haze, light, sunset, night ...), colour word, person word, relation (next to,
  behind, on top of ...), small/background detail, "or" composition (the grounding tail's own "or" ignored);
  otherwise the SCORER CASCADE: towervec12B P on every in-scope photo; the facts reader ce12B replaces it on photos with
  towervec P >= the K-th largest of that question's in-scope photos (in-scope = what the engine asks about in a first
  towervec-only pass); the rest keep towervec P. K = 0 / 500 / all (2,000 and 5,000 exceed every dev library,
  1,778-1,990 photos, so they equal "all") and phone-scaled K_lib = ceil(K x N_lib / 187,084) = ~6 / ~21 / ~51.
  Facts: 27B (ce12B_full + towervec12B_full) and the 4B phone writer (ce12B_q3vl4bq4r + towervec12B_q3vl4bq4r), RESULTS 59.
  Acceptable = ALL diff 95% CI lower bound >= -.03.
Routing on dev (pass A, all judge calls): scorer 43.0%; judge: colour 22.4%, photo look 18.3%, person word 10.2%,
text/sign 1.9%, or 1.5%, red-box 1.3%, count 0.7%, relation 0.6% (on TRAIN plan questions, read before: 47% / 53%).

| cat (n) | 9B all-judge | 27B: K=0 | 27B: K~21 phone-scaled | 27B: K=all | 4B: K=0 | 4B: K~21 | 4B: K=all | POST HOC 27B: K=all + judge top 100 | POST HOC 4B: same |
|---|---|---|---|---|---|---|---|---|---|
| a objects (61) | .755 | .763 | .784 | .790 | .747 | .739 | .745 | .791 | .746 |
| b attributes (64) | .667 | .675 | .675 | .672 | .672 | .672 | .669 | .675 | .668 |
| c person+look (50) | .580 | .588 | .585 | .578 | .590 | .572 | .564 | .585 | .565 |
| d time/place (64) | .722 | .748 | .728 | .738 | .754 | .728 | .721 | .738 | .721 |
| e multi-hop (17) | .636 | .665 | .783 | .783 | .663 | .546 | .604 | .783 | .604 |
| f vague 1 photo (33) | .606 | .592 | .604 | .604 | .623 | .614 | .614 | .622 | .632 |
| g two albums (35) | .776 | .750 | .740 | .753 | .736 | .720 | .741 | .729 | .717 |
| h follow-ups (27) | .619 | .561 | .571 | .566 | .565 | .543 | .528 | .559 | .524 |
| **ALL (351)** | **.680** | .682 (+.002 [-.013, +.018]) | .688 (+.008 [-.006, +.024]) | .690 (+.010 [-.007, +.028]) | .681 (+.001 [-.012, +.015]) | .663 (-.017 [-.037, +.003]) | .665 (-.015 [-.036, +.006]) | .690 (+.011 [-.009, +.030]) | .664 (-.016 [-.039, +.006]) |
| ALL verified (202) | .692 | .695 | .708 | .712 | .697 | .673 | .675 | .715 | .678 |
| judge calls / search: mean, median | 1,188, 1,495 | 718, 82 | 717, 75 | 714, 69 | 719, 78 | 718, 72 | 714, 66 | 55, 65 | 54, 62 |
| share of all-judge calls | 1 | .604 | .603 | .601 | .605 | .604 | .601 | .046 | .045 |
| facts-reader pairs / search | - | 0 | 12 | 447 | 0 | 12 | 449 | 447 | 449 |
| est. phone min / search: mean, median | 47.5, 59.8 | 28.7, 3.3 | 28.7, 3.0 | 28.6, 2.8 | 28.8, 3.1 | 28.7, 2.9 | 28.6, 2.8 | 2.2, 2.7 | 2.2, 2.5 |
Other pre-registered rows (cascade_report.txt): 27B K~6 .685, K~51 .685, K=500 .690; 4B K~6 .667, K~51 .662, K=500 .665.
Phone minutes = judge calls / 25 per min + towervec 1.6 MFLOP and facts reader 8.1 GFLOP per photo-question at an
assumed 1 TFLOP/s (ESTIMATE): the scorers are < 1% of the time in every row (K=all: 447 reader pairs ~ 3.6 s). Judge
cache misses: 759 / 518 photos in the 27B / 4B first passes (new scopes), answered by the real 9B.
Searches with no judge call at all: 154/351 under every cascade (all-judge: 17); judge calls p90 1,945 (K=all) vs 1,974
(all-judge) - the heavy searches are the colour / photo-look / person ones, which the pre-registered cascade still judges
over the whole scope; POST HOC top-100: p90 102, max 242.

READING
- Pre-registered verdict: with 27B facts EVERY cascade is acceptable and none is worse than the all-judge stack (ALL
  +.002 to +.010; a +.008..+.035, e +.03..+.15; h -.048..-.058 and g -.023..-.040 are the losses, CIs mostly spanning 0
  for g). With 4B phone facts only K=0 (ms scorer + judge) is acceptable (+.001 [-.012, +.015]); adding the 4B facts
  reader makes it worse (K=all -.015, lower bound -.036): on phone facts the reader's errors outweigh its help (e .663
  -> .604, h .565 -> .528), on 27B facts it helps (a .763 -> .790, e .665 -> .783).
- But the pre-registered cascade saves only ~40% of judge calls (mean 1,188 -> ~715 per search): the median search
  needs ~70 calls instead of 1,495 (154 of 351 need none), yet the colour / photo-look / person searches still judge
  their whole scope. POST HOC (designed after the rows above were scored, not pre-registered): the judge sees only the
  top 100 of a pixel-kind question's in-scope photos by towervec P, the rest keep towervec P -> judge calls 55 per search
  (4.6% of all-judge), est. 2.2 min instead of 47.5, with 27B facts ALL +.011 [-.009, +.030] (a +.036, h -.060, g -.047),
  with 4B facts -.016 [-.039, +.006] (h -.095). M = 300: 144 calls (12%), 27B +.011, 4B -.014.
- What the cascade loses, read off the requests and one sheet (native px, cascade_diff_0.jpg, 6 tiles: 3 judge-only and
  3 cascade-only returns, all 6 truth): trip membership asked as a question the scorer answers ("Is this a photo taken
  during a safari?", "...from a trip out west?") - when it says no, the follow-up's pixel question (elephants, the cactus
  silhouette) never runs (h30 .96 -> .05, f26 1.00 -> 0); what it wins: e anchors the facts reader finds ("the musicians in
  colonial uniforms playing drums" 0 -> 1.00, "Everest Base Camp with the prayer flags" 0 -> 1.00, giraffes at the theme
  park 0 -> .92).
- SCALE CAVEAT. These are 1,778-1,990-photo libraries and the final round (every in-scope photo judged). Ratios transfer
  to Reza's 187k library, absolute counts do not: the all-judge final round there is ~100x the calls, and "top 100" on a
  dev library is ~5% of it (~9,800 photos at 187k). The phone-scaled K rows (~6 / ~21 / ~51 reader photos per dev
  library = 500 / 2,000 / 5,000 at 187k) say the reader's top-K budget can be that small with 27B facts (+.005..+.008)
  but not with 4B facts (-.013..-.018).
- VERDICT. The cascade keeps the all-judge stack's F1 on dev while sending 43% of judge calls to scorers (pre-registered)
  or ~95% (post hoc, top-100 judge). With the phone's own 4B facts the only clean configuration is K=0 (ms scorer +
  judge for pixel kinds; the 4B facts reader adds nothing). Next: pre-register the top-M judge budget as a fraction of the
  scope and test it on TRAIN libraries before shipping; route trip membership to place/date metadata (the cascade's main
  h loss). K=0 needs no on-phone facts reader, only towervec + a facts vector per photo (144 MB for 187k) - but those
  vectors still come from facts WRITTEN at index time (RESULTS 59's 300-1,000 phone-hours estimate stands); whether the
  image vector alone suffices inside the cascade (vec scorer at K=0) was not tested.
~1.7 GPU-h (scorers on GPU, judge from cache; 1,277 cache misses answered by a lazily loaded 9B). Not pushed.

## 59c. Cascade, part 2: a judge BUDGET as a fraction of the scope + trip questions from place metadata (pre-registered, tested on TRAIN, confirmed on dev), and the image-vector-only cascade (no facts at index time) (10-14; eval/facts_scorer/{route.py budget/TRIP/place_candidates, cascade.py r59c, cascade_report.py, alljudge_job.sh} -> eval/facts_scorer/c59c_{train,dev,dev_4bshort}.txt)
Test closed. Code = main with RESULTS 60 (frozen .cache/wt_r59b/src). Plans = the shipped planner (10-05 adapter, 4-bit):
dev r60v2_base; TRAIN = its 643 TRAIN plans (RESULTS 59) re-grounded on CPU with the RESULTS 60 code (3 change, 0 missing).
REFERENCES: all-judge shipped stack (9B on every in-scope photo): dev r60v2_base_search ALL .680; TRAIN NEW
(train_alljudge, 24 GPU jobs) ALL .657; the harness serving those cached answers reproduces both exactly (351/351,
643/643). Scorers on TRAIN libraries are FOLD models (never trained on that library); dev uses the full-TRAIN models.
RULE R59c, PRE-REGISTERED and committed (32326d0) before any number below, on top of RESULTS 59b's routing, K = 0:
  (a) TRIP route: a question matching trip / vacation / holiday / tour / visit / travel / "taken in|at|during|on" /
      "photo from" / "during a|the|our|my" is answered from the photo's place metadata when a capitalised name in it
      matches >= 1 photo of the library (engine.place_mask): P .99 there, .01 elsewhere; no matching name -> the judge.
  (b) JUDGE BUDGET: each judge-routed question is judged only on its top min(n, max(50, ceil(5% of n))) in-scope photos
      by the millisecond score; the rest keep the millisecond P.
  Scorers: vec = image vectors only (PE-Core-B-16 photo + question vectors, NO facts; trained on the same TRAIN pairs
  incl. the planner-question oracle pairs), towervec12B on 4B phone facts, towervec12B on 27B facts.
  Acceptable = lower 95% bound of the ALL difference >= -.03.

1. TRAIN (the test of the rule; 643 requests; all-judge ALL .657; judge calls / search mean 1,247, median 1,586 = est.
   49.9 / 63.4 phone minutes at 25 calls/min)
   | cat (n) | all-judge | vec K=0 | vec R59c | 4B facts K=0 | **4B facts R59c** | 27B facts K=0 | 27B facts R59c |
   |---|---|---|---|---|---|---|---|
   | a objects (114) | .682 | .630 | .625 | .657 | .653 | .655 | .648 |
   | b attributes (113) | .601 | .591 | .561 | .601 | .583 | .598 | .589 |
   | c person+look (88) | .662 | .634 | .626 | .646 | .655 | .646 | .644 |
   | d time/place (117) | .595 | .564 | .620 | .595 | .645 (+.050 [+.009, +.094]) | .585 | .642 |
   | e multi-hop (29) | .804 | .786 | .754 | .780 | .778 | .794 | .792 |
   | f one photo (58) | .615 | .580 | .590 | .554 | .588 | .595 | .606 |
   | g two albums (62) | .759 | .691 | .708 | .723 | .738 | .723 | .732 |
   | h follow-ups (62) | .693 | .600 | .559 | .646 | .629 (-.063 [-.107, -.022]) | .658 | .632 |
   | **ALL (643)** | **.657** | .617 (-.040 [-.053, -.027]) | .617 (-.040 [-.056, -.023]) | .636 (-.021 [-.034, -.009]) | **.645 (-.012 [-.025, +.002])** | .639 (-.018 [-.030, -.007]) | .645 (-.012 [-.026, +.002]) |
   | acceptable? | | NO | NO | NO (-.034) | **YES** | borderline (-.030) | YES |
   | judge calls / search: mean, median | 1,247, 1,586 | 703, 21 | 47, 50 | 703, 21 | 47, 50 | 703, 21 | 47, 50 |
   | est. phone min / search: mean, median | 49.9, 63.4 | 28.1, 0.8 | 1.9, 2.0 | 28.1, 0.8 | 1.9, 2.0 | 28.1, 0.8 | 1.9, 2.0 |
2. DEV (confirmation; 351 requests; all-judge .680; 1,188 / 1,495 calls = 47.5 / 59.8 min)
   | cat (n) | all-judge | vec K=0 | vec R59c | 4B facts K=0 | **4B facts R59c** | 27B facts K=0 | 27B facts R59c |
   |---|---|---|---|---|---|---|---|
   | a (61) | .755 | .721 | .716 | .747 | .742 | .763 | .757 |
   | b (64) | .667 | .665 | .626 | .672 | .646 | .675 | .651 |
   | c (50) | .580 | .589 | .582 | .590 | .583 | .588 | .588 |
   | d (64) | .722 | .717 | .741 | .754 | .761 | .748 | .757 |
   | e (17) | .636 | .660 | .660 | .663 | .663 | .665 | .665 |
   | f (33) | .606 | .561 | .558 | .623 | .614 | .592 | .614 |
   | g (35) | .776 | .711 | .719 | .736 | .738 | .750 | .752 |
   | h (27) | .619 | .569 | .555 | .565 | .559 | .561 | .558 |
   | **ALL (351)** | **.680** | .660 (-.019 [-.038, -.001]) | .655 (-.024 [-.046, -.003]) | .681 (+.001 [-.012, +.015]) | **.675 (-.005 [-.022, +.014])** | .682 (+.002 [-.014, +.018]) | .680 (+.000 [-.019, +.018]) |
   | acceptable? | | NO | NO | yes | **YES** | yes | yes |
   | judge calls / search: mean, median | 1,188, 1,495 | 719, 83 | 50, 50 | 719, 78 | 50, 50 | 718, 82 | 50, 50 |
   | est. phone min / search: mean, median | 47.5, 59.8 | 28.8, 3.3 | 2.0, 2.0 | 28.8, 3.1 | 2.0, 2.0 | 28.7, 3.3 | 2.0, 2.0 |
   Exploratory, dev only (no TRAIN fold check): towervec on the 4B SHORT keyword line (126 vs 271 output tokens per photo,
   RESULTS 59): K=0 .674 (-.006 [-.024, +.012]); R59c .668 (-.012 [-.033, +.009]), i.e. just below the bar.
   Phone minutes = judge calls / 25 per min + scorer compute (< 1% everywhere, ESTIMATE as in 59b). Judge-cache misses
   (photos no all-judge run had asked about) were answered by a lazily loaded 9B (TRAIN K=0: 321-1,145 per run; R59c 18-56).

3. READING
- THE DECISIVE ONE: the image-vector-only cascade does NOT hold: TRAIN -.040 (lower bounds -.053 / -.056), dev -.019 /
  -.024 (lower -.038 / -.046) under either routing; it loses most on h (TRAIN -.09 / -.13), g (-.05 / -.07) and a (-.05).
  The same cascade with a facts vector from the PHONE-SIZE 4B writer passes on TRAIN (R59c -.012 [-.025, +.002]) and dev
  (-.005 [-.022, +.014]). So with today's scorers the phone DOES need index-time facts; the image vector alone is ~.03
  short (TRAIN R59c .617 vs .645).
- THE RULE: R59c (judge budget max(50, 5% of scope) + trip metadata) passes on TRAIN and dev with 4B or 27B facts while
  using 3.8-4.3% of the all-judge stack's judge calls: 47-50 per search, est. ~2 min instead of ~50. It beats plain K=0
  on TRAIN (4B: -.012 vs -.021) at 1/15 of the judge calls; on dev it is .006 below K=0 (inside noise).
- Trip route: d +.050 [+.009, +.094] (TRAIN, 4B) and +.039 (dev): place metadata answers "from our Maui trip", "taken in
  Cuba / Trinidad / Viñales", "during a trip to Bangladesh"; on dev its place-matched answers agree with the 9B on
  13,158 / 15,248 photos (.863). Sheet (trip_meta_0.jpg, 6 disagreements, native px): metadata right on 3 (a Monterey
  sea lion the 9B could not place, a Buenos Aires door the 9B called Venice, a Ticino ski slope the 9B called Zermatt),
  wrong on 3: two COMPOUND questions whose visual condition the route dropped ("a photo with a sepia tone taken in Spain"
  -> every Spain photo, here two colour photos) and one place spelled differently in the address ("Pagsanjan Falls" vs
  "Pinagsanjan"). The compound flaw is visible in b (TRAIN -.018, dev -.021 vs all-judge under 4B R59c; K=0 b is
  .000/+.005): a rule written before the data, kept as is. Fix for the next rule: metadata only when the question has
  no condition left after the place phrase; otherwise metadata AND the judge for the rest.
- h stays the loss in every configuration (TRAIN -.035..-.13; dev -.05..-.06): follow-up "or" questions and conditions
  judged under a 50-photo budget inside a trip's scope.
- SCALE: the budget is a fraction, so on Reza's 187k library a whole-library pixel question gets 9,355 judge calls
  (~6 h at 25/min) for its complete answer; the 2-minute figure is for ~1,900-photo libraries. The phone's first answer
  (480 calls) is unchanged by this; what changes is that 96% of the per-photo judging becomes millisecond scoring.
- VERDICT: ship-candidate shape = millisecond scorer over a 4B-written facts vector + PE-Core vector (towervec12B),
  judge only for pixel kinds within a 5%-of-scope budget, trip questions from place metadata (with the compound fix).
  Open: the index-time facts cost (RESULTS 59: 300-1,000+ est. phone-hours for 187k at 4B; decode speed never measured on
  the iPhone) - the short keyword line (half the tokens) is .006 behind the full facts on dev but has no TRAIN check.
~7.6 GPU-h (TRAIN all-judge reference 24 jobs, fold scorers 9, cascades 76 shards). Not pushed.

## 60. The "real X (not a drawing, painting, statue, toy, model or picture of one)" rewrite skips things that ARE made images, figures or characters - ADOPTED (rule v2; dev shipped planner ALL +.001 [-.000, +.003], 3 requests move) (10-14; converse.DEPICTED_THING / asks_for_depiction, FindPicsCore Grounding.swift asksForDepiction; eval/complex_bench/r60/{plan_job.sh,plan_replay.py,changed_qids.py,merge.py,sheets.py,check_r26.py,ground_cases.json} -> runs/cmp/r60{,v2}_{base,q4}*, r60_train_{base,new}_search, compare_60{,_b3}.txt)
Dev for selection; the test split was not built, run or scored (closed after RESULTS 54).

1. THE RULE THAT HURT. converse.ground (Swift Grounding) rewrites every "Is there X (anywhere) in this photo?" to "Is there
  a real X anywhere in this photo (not a drawing, painting, statue, toy, model or picture of one)?" (RESULTS 26 fix 1),
  unless the conversation contains one of RESULTS 26's art words (drawing, statue, toy, poster, sign, logo ...). When X
  itself IS one of the excluded kinds the question contradicts itself: "a real trail map (not a drawing ...)" (RESULTS 57:
  .79 -> .30 under B3's wording), "real Mickey Mouse (not a statue, toy ...)" (RESULTS 55: costumed Mickey P .44-.62,
  4/5 truth photos missed), "a real yellow rubber duck (not a ... toy)" (TRAIN: returned NOTHING).
2. THE EXEMPTION (written before any dev score; JOURNAL 10-14 ~02:30). The rewrite is also skipped when the HEAD NOUN of
  any "X and Y" / "X or Y" part of the question's thing (the words before with / of / on / wearing / visible ...) is a
  made image (map, poster, sign, tattoo, logo, picture, mural, comic, sticker ...), a figure (doll, puppet, mannequin,
  rubber duck, mascot, character ...), a creature that is not real (dinosaur, dragon, unicorn, mermaid, fairy, angel,
  ghost ...) or a named character (~60: Mickey Mouse, Elmo, Santa Claus, Spider-Man ...), or when the head carries an
  art word ("a toy car", "a stuffed animal"). Kept: "a person wearing an Elmo costume", "a van with a logo", "a tattoo
  artist", "the Santa Monica pier", "an inflatable bouncy castle", "the Eiffel Tower" (a souvenir model is still not it).
  Not a capital-letter rule: a lower-case "mickey mouse" is exempt, "Santa Monica" is not.
3. v1 -> v2 (a deviation from the pre-registration, stated in the journal before v2's numbers were computed). v1 also
  listed reflections and real objects that carry writing (menu, label, plaque, whiteboard, text). That misread my own
  principle: the tail names drawing, painting, statue, toy, model, picture - made images - and a reflection, a menu or a
  whiteboard is a real thing, so the tail does not contradict it. The data agreed for reflections: WITH the tail F1 was
  higher on 9 of 10 dev request-runs ("photos with reflections in the water", 5 requests x 2 planners: shipped mean
  -.014, B3 -.048 without it) and 4 of 5 TRAIN requests (~-.03 each; the 5th went 0 -> .22 from an empty answer). Sheets
  (dev b9, b6, native px): both wordings accept junk in those libraries (the tail version food and drinks, the plain one
  sunsets and a bar counter); of 9 tiles in b6 the plain one adds 2 clear lake reflections (both outside the truth) and
  drops 3 wrong. v2 = v1 minus those words, nothing else. v1 by the letter of the pre-registered rule would also have
  passed (ALL +.001 [-.001, +.003]) but with 1 request better and 7 worse.
4. DEV (351 requests; shipped 10-05 adapter on the 4-bit weights, its r57 raw outputs reused: 451 raw outputs, 2 new, 0
  errors; 9B judge + shared jcache; changed requests searched, the rest identical by construction - same plans, same
  cached answers). Plans that change: shipped 3/351, B3 3/351.
  | request | shipped 10-05 adapter | B3 adapter (RESULTS 57, not shipped) |
  |---|---|---|
  | b "the photos with Mickey Mouse" | .333 -> .667 (R .20 -> 1.00, P 1.00 -> .50) | 0 -> .588 |
  | a "the photos of the trail maps" | (not rewritten: "Is this a photo of a trail map?") | .302 -> .800 (R .18 -> .86) |
  | h Zermatt "looking at the map" (3 turns) | .326 -> .286 | 0 -> 0 |
  | h castle, then "include the golf course too" | .667 -> .652 | (unchanged) |
  | **ALL (351)** | .679 -> .680, **+.001 [-.000, +.003]** | .686 -> .689, +.003 [.000, +.008] |
  | per category | b +.005 [.000, +.016]; h -.002 [-.006, .000]; a c d e f g .000 | a +.008 [.000, +.024]; b +.009 [.000, +.028]; rest .000 |
  The two h losses are knock-ons, not the exemption's own answers: Zermatt's turn-1 question ("a map") is now asked
  plainly, which changes the history the planner sees, and its turn-3 answer AND-ed "a group standing together" onto
  "people eating inside a hut" (sheet: 2 of 6 dropped tiles are the hut meals the truth wants); the castle follow-up's
  old question carried the tail TWICE ("a real real castle ... (not ...) or a golf course ... (not ...)"), the new one
  once (2 photos dropped, 1 in the truth: Mons Meg on the castle wall).
  Sheets viewed (native px, labelled before the key): Mickey, 6 of the 9 added: 2 costumed Mickey (= the 2 truth tiles),
  3 not (Cheshire cat float, Woody with a kid in a Mickey cap, a parade float), 1 unsure (Electrical Parade, Minnie); truth
  recall 1/5 -> 5/5. Trail maps (B3), 6 of 42 added: 4 trail-map boards (= the 4 truth tiles), a compass
  sign, a signpost.
5. TRAIN (24 libraries; not used for any choice above; 9B judge both ways on the requests v1 changed, 8): v2 keeps 3 of
  them - rubber ducks in the river 0 -> .905 (0 -> 23 returned), the giant worm image on a screen f 0 -> .154 (target
  found), "a person in a green Packers shirt and a map" f .667 -> .182 (target found both ways; 2 -> 10 returned: the
  "and a map" part triggers the exemption on a person question - the known cost of judging every listed thing).
6. RESULTS 26 HOLDS (check (2)): every question RESULTS 26 measured (photos with a dog / car / bicycle / boat; Open Images
  Dog, Cat, Bicycle, Horse, Guitar, Cake, Pizza, Coffee cup, Wine glass, Sunglasses; the 96 everyday plans) grounds
  byte-identically under old and new code (r60/check_r26.py, 24 distinct pairs, 16 with the tail) -> the same judge
  answers -> its eye-labelled gain is unchanged (8/8 wrong + 2/2 unsure dropped, 0/21 right; 10/12 rejected OI
  "positives" depictions: stuffed dogs, rocking horse, statues ...).
7. PARITY / GATE. Grounding fixtures regenerated (old code reproduces the committed fixture 0/2,308): 3 of 2,308
  distillation plans change, all heads that are made images or figures (a garden gnome; "a watermark, logo, text overlay
  or brand mark"; "a clear photo of a cat's collar tag"), + 16 hand cases (r60/ground_cases.json: 7 exempt, 7 keep the tail, 2 decided by older rules);
  FOLLOWUP 874 unchanged. Swift (same four patterns, string-identical to Python): GROUND 0/2,324, FOLLOWUP 0/874, XCTest
  141 (+1 testDepictionsKeepThePlainQuestion) + swift-testing 29, 0 failures. pytest 156/156 (+1). 30-conversation gate
  on the 4-bit adapter: v1 30/30 with all 30 plans identical to RESULTS 57's gate; v2 30/30 (gpu_test GPU; 2 plans differ from RESULTS
  57's gate in "looks" wording only = vLLM output on another GPU, questions identical); the v2 dev plan stage on the GPU
  equals the CPU replay 351/351.
8. CAVEATS. The dev gain is one request (Mickey) on the shipped planner and two on B3; the mechanism is clear on the
  sheets and on TRAIN (rubber ducks), but the dev CI includes 0. b is a PROVISIONAL truth (RESULTS 56). The word lists
  are hand-written (images / figures / creatures / ~60 characters): a character not on the list still gets "real X";
  a person conjoined with a made image ("... and a map") loses the tail. ~0.8 GPU-h (plan stages 0.2, dev searches 0.2,
  TRAIN searches 0.3, gate 0.1) + CPU jobs (swift test x3, pytest x3).

## 61. A SHORT-FORMAT facts writer distilled from the 27B for the phone: the format costs nothing, the students come within a hair of the bar but none passes (10-14; eval/facts_writer/{short_format,short_rewrite,train_writer,merge_q4,write_facts,quality,r61_table}.py, launch_{writer,student}.sh -> eval/facts_writer/{r61_table,r61_quality}.txt, r61_{train,dev}_{short27b,s4b,s2b,s08}.txt)
Why: RESULTS 59c's cascade passes with the 4B phone writer's facts, but those facts cost ~271 output tokens per photo
(est. 560-1,500 phone-hours for Reza's 187k photos). Test closed; TRAIN for training; dev + TRAIN (cross-fit) for
evaluation. PRE-REGISTERED (JOURNAL 10-14 ~05:10, f97aee6) before any number, incl. the SHIP RULE: a writer is acceptable
iff the R59c cascade's ALL difference vs the all-judge stack has lower 95% bound >= -.03 on BOTH TRAIN (643) and dev
(351); ship = the smallest acceptable model, then the fewest output tokens.

FORMAT (short_format.py): one line "scene: ..; objects: ..; colours: <colour + the thing it belongs to>; people: N - per
person clothing / colour / action; activity: ..; text: ..; look: <photo-look words>; event: ..", empty fields omitted.
STEP 1 - the format alone: the 27B rewrote its own JSON facts into it, TEXT ONLY, for all 70,455 TRAIN+DEV photos with
facts (mean 75 output tokens vs ~243 for the JSON; p90 100; 0.5% at the 120 cap; ~104 s per library on one GPU).
R59c cascade with towervec12B retrained on the short lines (+ place/month; TRAIN fold models): TRAIN -.011 [-.024,
+.002], dev +.000 [-.017, +.018] - the same as the 27B JSON (-.012 / .000). The format loses nothing measurable.
STEP 2 - students (LoRA r16 on language attn / linear-attn / MLP + vision blocks / mergers, on the phone's 4-bit-sim
weights; input = the phone's 448 px rendition, JPEG q80, + "Search facts for this photo, one line."; target = the 27B
short line; 1 epoch, lr 1e-4, bs 8 x 2). CROSS-FIT: student A trains on TRAIN folds {0,1} (~23k photos), B on {2,3};
each writes facts for the other half, A also for dev - so every evaluated facts line comes from a student that never saw
that photo. Phone form = LoRA merged in fp32 and the language model re-quantized to MLX 4-bit (exact rounding), vision
bf16; served greedily, repetition penalty 1.05, max 120 tokens. Training ~20-29 min per student on one RTX PRO 6000;
val token loss (A) 4B .687, 2B .752, 0.8B .819 - still falling at the end of the epoch (4B: .759 at step 400).
Writing speed on the cluster: the 2B ~125 photos/s (vLLM, batch 256).

| writer | size | out tokens / photo mean (p90) | repetition | objects F1 vs 27B (dev) | TRAIN R59c diff [95% CI] | dev R59c diff [95% CI] | acceptable | est. phone index hours, 187,084 photos, at 10 / 20 / 30 tok/s |
|---|---|---|---|---|---|---|---|---|
| 27B JSON (server; RESULTS 59c) | 27B | ~243 | - | - | -.012 [-.026, +.003] | .000 [-.018, +.018] | yes | 1,351 / 720 / 509 |
| 27B short line (step 1) | 27B | 75 (100) | .024 | 1 | -.011 [-.024, +.002] | .000 [-.017, +.018] | yes | 480 / 284 / 219 |
| Qwen3-VL-4B zero-shot JSON, 4-bit (59c) | 4B | 271 (331) | (19% looped greedy; rep 1.05: 2%) | - | -.012 [-.026, +.001] | -.005 [-.023, +.012] | yes | 1,497 / 793 / 558 |
| Qwen3-VL-4B zero-shot keyword line (59c, dev only) | 4B | 126 (160) | (52% at its 160 cap) | - | - | -.012 [-.033, +.009] | - | 743 / 416 / 306 |
| **student Qwen3-VL-4B, 4-bit** | 4B | **72 (99)** | .024 | .59 | -.016 [-.031, -.003] | -.012 [-.030, +.007] | **no (by .0006 / .0003)** | **462 / 275 / 213** |
| student Qwen3-VL-2B, 4-bit | 2B | 71 (97) | .051 | .54 | -.023 [-.038, -.009] | -.010 [-.030, +.009] | no (TRAIN) | 457 / 273 / 211 |
| student Qwen3.5-0.8B, 4-bit | 0.8B | 69 (94) | .029 | .54 | -.029 [-.045, -.014] | -.011 [-.031, +.008] | no | 448 / 268 / 208 |
CIs are exact bootstrap bounds (2,000 resamples, seed 0, as compare.py): 4B student TRAIN lower -.0306, dev -.0303 (the
rounded -.030 hides a miss); 2B dev -.0296 passes, its TRAIN -.0378 does not. Index hours = 187,084 x (1.7 s + tokens / r)
- 1.7 s = the phone's measured judge call (vision tower + short prompt, RESULTS 44, cool); the students' prompt is 152
tokens (image + 9 words) vs 788 for the zero-shot JSON writer, so the zero-shot rows are UNDER-estimated; r is unmeasured
(M70). Repetition = at the 120 cap or a word 3-gram repeated >= 3 times ("kilt, kilt, kilt ..." to the cap).
Per-photo quality vs the 27B short line (r61_quality.txt; dev 24,564 photos; TRAIN out-of-fold 45,891 similar): 4B /
2B / 0.8B objects P .62 / .57 / .57, R .58 / .51 / .52; colour words F1 .81 / .75 / .79, colour+thing phrases .37 /
.32 / .34 (exact phrase match: strict); scene .59 / .55 / .55; people count within 1 .87 / .84 / .82; visible text
shares a word .75 / .66 / .66. Sheet (r61_students_0.jpg, 6 random dev photos, native px, all three lines read): the 4B
lines are as right as the 27B's on 4 (a parrot where the 27B says "red bird", the red plaid collar the 27B missed) and
wrong on 2 details - text misread ("TONY MACDONALD" for AMY MACDONALD, "CHAMPAGNE AND BUBBLE") and one invented action
("golf course, swinging bat" for a man holding a stick).
Per category, R59c F1 (TRAIN; all-judge / 27B short / 4B zero-shot JSON / 4B student / 2B / 0.8B): a .682 / .661 / .653
/ .657 / .636 / .618; b .601 / .593 / .583 / .578 / .578 / .577; c .662 / .641 / .655 / .645 / .638 / .638; d .595 / .643 /
.645 / .634 / .636 / .641; e .804 / .804 / .778 / .783 / .755 / .756; f .615 / .579 / .588 / .583 / .584 / .553; g .759 /
.731 / .738 / .728 / .719 / .719; h .693 / .629 / .629 / .635 / .631 / .621. The students lose to the 27B's own short
line mostly on a (objects: -.004 / -.025 / -.043) and e/f (the one distinctive thing), as RESULTS 59's phone writer did.

READING
- The short format is free: 75 tokens instead of ~243 with no measurable end-to-end loss (27B source).
- Distillation works in kind: a 4B student writes the format in 72 tokens (3.8x fewer than the zero-shot 4B's JSON, 5.2x
  shorter prompt) and lands .004-.007 below the zero-shot 4B on the cascade - within noise of it - but its TRAIN and dev
  lower bounds are -.0306 / -.0303, so by the pre-registered rule NOTHING SHIPS from this round. The 2B and 0.8B students
  write equally short lines and are .007-.013 worse on TRAIN; on dev all three are within .002 of each other.
- Index cost: the 4B student at 20 tok/s would be ~275 phone-hours for 187k photos (est.), vs ~790 for the zero-shot
  JSON writer - and below ~10 tok/s nothing here is affordable. The fixed 1.7 s per photo (vision + prefill) alone is
  88 h; tokens matter less once the line is ~70 tokens. Smaller models only pay off if their vision + prefill time is
  well under the 4B's (unmeasured; M70 should time all three on the iPhone).
- What limits the students is data, not the format: each was trained on half the TRAIN photos (cross-fit, ~23k), one
  epoch, and its validation loss was still falling. NEXT (cheap, all public): label 10x more photos with the 27B short
  line (text rewrite of 27B facts ~1 s / photo / GPU; facts themselves need the 27B VLM pass) from public sets outside
  the benchmark (Open Images / the other DISBench data the benchmark does not use), train each size on all TRAIN + that
  data for 2-3 epochs, keep the same ship rule; and measure decode tok/s + vision time per size on the iPhone (M70).
~10.3 GPU-h (27B rewrite 8 jobs, 6 students, 6 phone-form + writing jobs, 20 scorer trainings, ~100 cascade shards).
Not pushed.

## 62. A FIRST ANSWER for person searches: FACT = fast mode inside the person's photos (bound kept), LOOK = the event walk (pre-registered 1e9756a; 10-14; src/findpics/person_first.py, engine Thresholds.person_first; FindPicsCore PersonFirst.swift + Pairing.twoGroups(weights:); eval/person_first/{r62,fixtures}.py -> eval/person_first/{facts,looks_public,coverage}.txt; private replay numbers in data/private/audits/r62_looks_private.txt; MAC_INBOX M72)
Problem (10-14 07:00, Mac M67): a person album with a condition judged EVERY face-matched photo before showing anything
(PersonSearch.swift `for k in ident`, engine.stream_album the same): 41,307 photos for the owner, ~34-38 h per album at
the measured 2.4 s judge + 0.7-1.0 s warm read (M67; 2.36 s cold). Rule (pre-registered before any number):
- FACT (is_look_question false: the judge's yes alone, RESULTS 49c): order the person's photos by the look vector (COS)
  or its learned-head blend (BLEND, RESULTS 48), run the phone's timed rounds exactly (phoneFastParams; first answer
  where fast mode stops, i.e. at the latest the round reaching 480 calls + a 150-photo random check; Clopper-Pearson
  bound every round), then keep looking (paced) to the end, which IS today's album (every photo judged, P >= 0.7).
- LOOK (relative): judge in a walk fixed before any answer (fnv1a64("seed|id") order, <= 3 judged photos per event
  first, events = 3 h gaps; a pair asks both questions of a photo, 2 calls). After any number of calls a photo of a
  reached event takes its event's median of the judged photos; the pair split is fitted on the judged photos, each
  weighted by event size / judged in its event, likelihoods scaled back to the judged count for the BIC; no clear
  split -> rank margin on the judged photos; a single look keeps the top half by the same weighted rank. Events not
  reached = "not sorted yet". With every photo judged both are the full rules exactly.

(A) FACT. eval/vector_first_person (32 DISBench people, phone judge Qwen3-VL-4B on every photo, red box), 7 facts.
F1 = each person alone (224 searches, 62-282 photos), F2 = all 32 as ONE 4,010-photo scope (7), F3 = random pools of 8 /
16 people (20 draws each; 280). 20 tail seeds per search. Judge-relative (precision vs judging everything is 1 by
construction: the first answer is judge-yes photos only).
| set | order | searches | photos (median) | first answer calls median / max | judge-yes found of all (recall) | calls to first screen (median) |
|---|---|---|---|---|---|---|
| F1 | COS / BLEND | 224 | 97 | 97 / 282 (= everything; n < 480) | 6,516 of 6,516 (1.000) | 21 / 21 (RANDOM 44) |
| F2 | COS | 7 | 4,010 | 602 / 604 | 1,982 of 6,516 (.304) | 22 |
| F2 | BLEND | 7 | 4,010 | 602 / 604 | 2,120 (.325) | 21 |
| F2 | RANDOM | 7 | 4,010 | 602 / 929 | 1,000 (.153) | 113 |
| F3_8 | COS / BLEND / RANDOM | 140 | 931 | 607 / 605 / 610 | .791 / .822 / .630 of 31,892 | 25 / 23 / 139 |
| F3_16 | COS / BLEND / RANDOM | 140 | 2,004 | 603 / 603 / 604 | .496 / .531 / .301 of 64,372 | 23 / 22 / 133 |
F2 per condition, BLEND recall at the first answer (602 calls of 4,010): snow .947 (of 55), beach .801 (311), eating
.779 (113), sunglasses .524 (738), hat .498 (723), outdoors .225 (2,260), smiling .205 (2,316): rare conditions are
nearly complete, common ones are capped by the budget (~500 matches in 600 calls) and the bound says so. Judging
everything = 4,010 calls. EYE (RESULTS 49c strata rates, blind labels, unsure = no match), F2 pooled: first answer
precision .839 (BLEND) / .835 (COS) vs judging everything .885; eye recall .286 / .267 vs .930.
Cost on the phone: median 603 calls (p90 611, max 987) = ~20 min on the measured m8 clock (RESULTS 44), ~33 min at
3.3 s/call serial (M67) - vs ~34-38 h today.
D1 FAILED AS PRE-REGISTERED: 47 of 74,285 rounds (COS + BLEND) showed a bound below the judge-yes still unjudged (42 of
20,440 runs = 0.21%; RANDOM 28 of 10,220). 43 of the 78 overclaiming rounds (all orders) are round 1 (30-photo random
check, alpha .01), the rest rounds 2-6; most in dense conditions (smiling, outdoors, hat, sunglasses) of the pooled
sets; most misses are a few % over the bound (F2 sunglasses COS round 3: 393 still hiding vs bound 388), the worst
37% (F3_16 hat COS round 1: 395 vs 288, 0 hits in its 30-photo check). Diagnosis: eval/person_first/over_diag.py. The bound's own guarantee (every bound shown
holds simultaneously with 95%: at most 5% of runs may show any miss) holds with a wide margin (0.17-0.24% of runs, per
order). My D1 ("0 overclaims in every round") is the RESULTS 43/44 replay convention, which a 95% bound cannot keep
as the replay grows; I wrote a criterion stricter than the guarantee. Shipped anyway, flagged: it is the identical
bound ordinary searches already show (same code, same parameters), and the alternative is 34 h with no answer. If
Reza wants the replay convention kept, the post-hoc lever is firstRoundAlphaShare / the first round's random check
(not tested here; it would need fresh person sets, e.g. the 4 unjudged DISBench shards of M45).
D2: BLEND. First-answer judge-yes found over F2 + F3 (287 searches): BLEND 62,540 vs COS 59,113; paired BLEND better in
245, worse in 33, same in 9. (The phone blends; the Python engine's person_first orders by the look cosine, as the
server's ordinary searches do: no learned head on the server.)

(B) LOOK, PAIR "me looking heavier" vs "me looking fit".
L1 Reza's 393-photo labelled sample (private; numbers only; 392 face-matched photos in 71 events; 784 calls = every photo,
both questions), replayed from the saved runs' pools. 20 walk seeds; precision / recall of each first-answer album
against the SAME album from judging everything; labels = his era labels (heavier: H found + F leak / fit: F found + H leak).
Judge-everything albums: phone judge (q3vl4b_real) 265 + 5 / 115 + 2 (4 neither); 9B (9b_prob) 267 + 8 / 116 + 0.
| run | variant | calls | photos sorted | agreement | heavier prec / rec | fit prec / rec (worst seed prec) | labels |
|---|---|---|---|---|---|---|---|
| phone | walk3 (shipped) | 200 | .908 | .826 | .999 / .828 | .912 / .824 (.382) | 220.3 + 3.4 / 96.2 + 17.3 |
| phone | walk3 (shipped) | 480 | 1.000 | .991 | 1.000 / .999 | 1.000 / .974 (.992) | 264.6 + 5.0 / 113.1 + 0.9 |
| phone | walk3 (shipped) | 1,000 | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 265 + 5 / 115 + 2 |
| phone | walk1 | 200 / 480 | .851 / 1.000 | .801 / .987 | .997 / .814, .999 / .997 | .982 / .771, .998 / .966 | |
| phone | uniform (no cap) | 200 / 480 | .841 / .960 | .760 / .951 | 1.000 / .812, .999 / .971 | .942 / .636, .998 / .904 | |
| phone | judged only | 200 / 480 | .255 / .612 | .224 / .605 | .998 / .143, 1.000 / .563 | .933 / .393, .999 / .688 | |
| 9B | walk3 | 200 / 480 | .942 / 1.000 | .921 / .980 | .997 / .935, 1.000 / .973 | .987 / .887, .997 / .996 | 252.2 + 5.7 / 103.0 + 1.3; 260.4 + 7.2 / 115.5 + 0.3 |
Curve, phone judge, walk3 (calls: sorted, agreement): 80: .51, .40; 160: .80, .74; 240: .94, .87; 320: 1.00, .97;
400: 1.00, .99; 480: 1.00, .99; 720: 1.00, 1.00.
D3 PASS: at 480 calls both albums' precision vs judging everything 1.000 / 1.000 (>= .95). All 30 saved runs (bipolar
excluded) at 480: 22 pass .95 on both albums; the 8 that do not are runs whose judge-everything albums are already
broken or odd (the collapsed ensembles ens/ens2/ens3/ens4, the 0-100 rating runs 9b_r100/4bq_r100, 9b_hard, 9b_rating).
CHECK: with every photo judged the partial rule = engine._split_pair / rank margin in 30 / 30 runs, 0 photo differences.
Early updates are NOT trustworthy: at 200 calls the fit album was wrong in some walk seeds (worst .382; mean 16.2
heavier-era photos in fit). So the phone shows the pair's albums only from the first answer (480 calls) on.
L2 IMDB-WIKI eras (Pratt 99 / Hill 74 / Rogen 66 crops, cached 9B, NO dates: every photo its own event, so propagation and
the cap do nothing): at 25% of the calls heavier / fit precision vs everything .918 / .897, at 50% .959 / .951, recall
= the share judged; every photo judged: 0 differences of 239 from the engine.
(B') LOOK, SINGLE album (not gated by D3; the pre-registered rule applies). Event propagation trades agreement with the
eventual per-photo album for coverage: Reza 'heavier' alone, phone judge, 200 calls: walk3 precision vs everything .779
(judged-only .985) but by his labels 221.3 H + 14.1 F vs the judge-everything single album's 176 H + 20 F (events are
era-pure; the per-photo top half is not). Public 'tired' (32 people, 1,534 events, judge-relative only - 0/16 judge-yes
photos were clearly tired by eye, M45): at 10% of the calls walk3 .673 vs judged-only .891 precision vs everything,
at 50% .791 vs .948. A single look's first answer therefore changes more on the way to the final album than a pair's.
Not changed here; the cleaner fix (pre-register next) is to give single looks the event median in the FULL rule too,
which matched his labels better in this replay.

COVERAGE (label-free; share of photos whose event was reached). DISBench whole libraries (57; median 1,963 photos, 152
events): 240 walked photos -> .945 (p10 .823). A 41k proxy for the owner's set (21 DISBench libraries' events together;
median 40,439 photos, 3,185 events): 240 photos (a pair's 480 calls) .270 [.229-.330 over 10 draws], 480 (a single look's
first answer) .413, 1,000 .601, 2,000 .788, 5,000 .964. So on his library the pair's first answer would sort ~1/4 of
his photos into heavier / fit / neither, at the precision above; the rest arrives as it keeps looking. His real event
structure is unknown here; the Mac measures it with zero judge calls (M72 (a)).
Tests: pytest 168/168 (+12 in tests/test_person_first.py: stream_timed = sim_timed call for call, weights 1 change no
bit, walk order, single and pair with everything judged = the full rules, stream_look cadence, fixture parity, engine
person_first FACT + LOOK end at the judge-everything album). FindPicsCore swift test 150 XCTest + 29 swift-testing,
0 failures (login node, nice; the copy diffed equal to the repo); PersonFirstTests (9) against Fixtures/person_first.json
(public + synthetic only; 11 walks, 58 single, 51 pair, 7 fact orders, 3 streams): single 0 / 4,191 and pair 0 / 6,604
photo mismatches, update sequences equal; PairingTests unchanged (weights nil).
LIMITS: F2/F3 pool different people's photos into one scope (the order, rounds and bound are tested at 1-4k photos;
nothing public holds one person's 41k photos); L1 is one person's 4-month curated sample with eras 3 years apart (RESULTS
25 caveat); the 4.8 / 3.3 s per call are the overnight M67 numbers, not measured with prefetch on the person path.

## 62b. A COVERAGE-FIRST walk for LOOK person searches (events largest first, one photo each): twice the photos sorted at the first answer, same precision - NOT adopted as pre-registered (the precision bar failed for EVERY walk); shipped as an opt-in (pre-registered f47db8d; 10-14; eval/person_first/r62b.py -> eval/person_first/r62b_public.txt, data/private/audits/r62b_private.txt; person_first.walk_order(by_size=), FindPicsCore lookWalkOrder(bySize:))
Why: the Mac's census of his face set (M72(a)): 41,307 photos in 5,764 events, median 3, largest 517, 1,689 singletons.
A pair's 480-call first answer judges 240 photos. Walks: SIZE = events largest first, pass 1 one photo per event, pass 2
photos 2-3 of every event, then the rest; WALK3 = shipped random walk, cap 3; WALK1 = random, cap 1.
DEVIATION before any scored number (journal 21:30): the walk key is now fmix64(fnv1a64("seed|id")) for every walk. Plain
FNV-1a sorted ids that differ only in their last characters next to each other (the synthetic "e00012p003" ids); real
ids were unaffected (his sample: same-event neighbours .143 vs .134 random; DISBench .016 vs .013), so RESULTS 62 stands.
Stand-in for his sizes (until M73 brings the real histogram): 1,689 singletons + 4,075 events of 2 + floor(Y), Y lognormal
mu 1.025 sigma 1.475 (deterministic quantiles, largest = 517; exactly 41,307 photos, median 3).
COVERAGE (share of photos whose event was reached; label-free; calls = pair calls):
| library | walk | 480 | 1,000 | 2,000 | 5,000 | calls to 50% / 90% |
|---|---|---|---|---|---|---|
| his sizes (best fit) | size | .381 | .520 | .670 | .869 | 906 / 5,862 |
| his sizes | walk3 | .192 | .315 | .459 | .678 | 2,408 / 13,268 |
| his sizes | walk1 | .214 | .355 | .527 | .798 | 1,810 / 7,186 |
| his sizes, lightest / heaviest tail that keeps median 3, mean 9.72 | size | .310 / .412 | .447 / .549 | .609 / .691 | .849 / .873 | |
| | walk3 | .145 / .218 | .254 / .343 | .395 / .485 | .628 / .696 | |
| 41k DISBench proxy (3 draws: 2,714-3,478 events) | size | .407-.536 | .599-.709 | .805-.865 | .976-.995 | 408-700 / 2,408-2,938 |
| | walk3 | .218-.315 | .365-.466 | .545-.654 | .808-.865 | 1,120-1,694 / 5,984-7,438 |
| | walk1 | .242-.363 | .429-.559 | .682-.776 | .962-.994 | 828-1,236 / 3,014-3,718 |
ALBUMS on a library with his sizes, answers BOOTSTRAPPED from his sample's 71 events (private, numbers only; 5 draws x 5
seeds; "correct" = sorted AND the label judging everything gives, over all 41,307 photos):
| walk | calls | sorted | correct | precision among sorted | heavier prec vs full | fit prec vs full | heavier: H + F photos / fit: F + H |
|---|---|---|---|---|---|---|---|
| size | 480 | .381 | .342 | .896 | .990 | .948 | 6,242 + 745 / 6,249 + 125 |
| walk3 | 480 | .197 | .180 | .910 | .988 | .966 | 3,369 + 329 / 3,332 + 74 |
| walk1 | 480 | .213 | .195 | .914 | .992 | .969 | 3,584 + 341 / 3,586 + 99 |
| size | 2,000 | .670 | .615 | .918 | .993 | .968 | 10,839 + 1,297 / 11,206 + 186 |
| walk3 | 2,000 | .455 | .422 | .929 | .995 | .963 | 7,487 + 816 / 7,890 + 138 |
| size | 5,000 | .869 | .791 | .911 | .990 | .971 | 14,122 + 1,746 / 14,140 + 207 |
| walk3 | 5,000 | .674 | .628 | .932 | .993 | .970 | 10,995 + 1,297 / 11,577 + 184 |
His sample itself (392 photos, 71 events, 20 seeds): at 480 calls all three judge the same photo set, so size = walk3
(.984 correct, heavier / fit precision 1.000 / 1.000, by his labels 263.1 + 5.0 / 112.2 + 0.7); at 240 calls size .733 /
walk3 .728 / walk1 .836 correct, fit album heavier-era photos 14.4 / 37.8 / 1.2; calls to 50% / 90% sorted 10 / 78,
23 / 162, 18 / 92.
PUBLIC parametric answers (event era +-1.5 + N(0, .8), photo noise t3, 10% uninformative): every walk fails - precision
among sorted .516 / .538 / .553 (size / walk3 / walk1) at 480 calls, .42-.48 at 5,000, while judging everything puts
.973-.985 of photos in their era's album. Cause: one judged photo per event is too noisy here for the two-group split to
be clear (the four inspected runs had no split; BIC gain -15 to 0), so the first answer falls back to the rank margin on
the judged photos, which disagrees with the event-median full albums. On his real answers the split is clear.
DECISION (pre-registered): (i) correct at 480, size beats walk3 and walk1 on both answer models (bootstrap .342 vs .180 /
.195; parametric .052 vs .025 / .028) - YES; (ii) size's precision among sorted >= .95 - NO (.896 bootstrap, .516
parametric); (iii) his sample at 480: 1.000 / 1.000, wrong-album photos 5.7 vs 5.7 - YES. -> SIZE WALK NOT ADOPTED, walk3
stays the default. The bar in (ii) was set without measuring the shipped walk against it: walk3 fails it too (.910 /
.538), so (ii) cannot tell the walks apart. What the numbers do say: size sorts 1.9x the photos at the first answer and
reaches 50% / 90% sorted in 906 / 5,862 calls instead of 2,430 / 13,350, at 1-1.5 points less precision among sorted
(heavier album 10.7% fit-era photos vs 8.9%). Shipped as an OPT-IN (walk_order(by_size=True) / lookWalkOrder(bySize:),
parity-tested); RECOMMENDATION to the coordinator / Reza: switch the default (one argument in M72's streamPersonLook call).
TWO CAVEATS FOR M72 that hold for every walk: on a library with his event sizes, the first answer's sorted photos agree
with the eventual albums ~90% of the time (bootstrap), not the 98-100% of his 393-photo sample; and with a noisier judge
the first answer may have no clear split at all (parametric), in which case it shows only rank-margin guesses for the
judged photos. Not changed here; a pre-registered follow-up should test showing "not sorted yet" instead of rank-margin
guesses before the end.
Tests: pytest 170/170 (+2: walk key mixes the last bytes; size walk order); swift test 150 XCTest + 29 swift-testing,
0 failures (fixture regenerated: 21 walks incl. 9 by size and 2 on end-differing ids, 4 streams incl. one by size;
single 0 / 4,163, pair 0 / 7,124 mismatches). Product default unchanged (bySize false).

## 59d. Cascade, part 3: compound trip questions, OR/AND parts, budget extension, judged exclusions - R59d ACCEPTABLE on TRAIN and dev and +.012 / +.010 over R59c (h -.063 -> -.015 TRAIN); the all-judge stack gains (a)+(b) too (dev +.011 [+.001, +.022]) -> ADOPTED in the product (Python findpics.compose + engine, Swift QuestionTree.swift, parity 0/660) (10-15; pre-registered 443c36c; eval/facts_scorer/{route.py R59d header, cascade.py DVARS, r59d/{diag_ms,diag_roles,design_check,compose_fixture,sheets,launch}.py/.sh} -> eval/facts_scorer/c59d_{train,dev}_{4b,27b}{,_vs_r59c}.txt, c59d_dev_aj_vs_aj0.txt, c59d_dev_product.txt; MAC_INBOX M74)
Test closed; TRAIN folds + dev only. Same plans, scorers, facts, judge cache and references as 59c (TRAIN all-judge .657,
dev .680; R59c re-run inside the same jobs reproduces 59c: TRAIN .645, dev .675). New judge questions (parts,
remainders) answered by a lazily loaded 9B (TRAIN 2,540 / dev 1,981 photo-questions for R59d with 4B facts; 21,545 for
the all-judge variant). RULE R59d (route.py header, committed before any number) = R59c +
  (a) trip route only when nothing but frame / trip-kind / date words is left after the place phrase; else place metadata
      AND the routed remainder; tolerant place names (accents/case; edit distance <= 2 on words >= 6 chars only when exact
      matching finds nothing and no plain word of the name matches alone; place-kind / feature words optional last);
  (b) "A or B" split into parts routed separately (max), "A and also B" / two question sentences (min);
  (c) from TRAIN h failures only (diag_roles.py, scorer P on every TRAIN all-judge pair): (c1) judge-budget EXTENSION -
      one more block of max(50, 5% of scope) while the last block was >= 50% judge matches (TRAIN h: budget-limited
      questions hold 5,098 9B matches, the budget reached 2,151); (c2) exclude questions always judged (scorer F1 vs 9B
      on exclusions h .80 / g .69; "Is the bottle on a kitchen counter?" scorer yes 24/27, 9B 2/27).
  Disclosure: (c) and the wording details of (a)/(b) were designed on TRAIN (questions, R59c per-request scores), so
  TRAIN is in-sample for them; dev was not read before scoring.

1. CASCADE, 4B phone facts (final-round F1; diff vs all-judge [95% paired bootstrap over requests])
   | cat | TRAIN all-judge | R59c | **R59d** | +a | +b | +c | dev all-judge | R59c | **R59d** | +a | +b | +c |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | a | .682 | .653 | .658 | .654 | .652 | .658 | .755 | .742 | .745 | .742 | .743 | .744 |
   | b | .601 | .583 | .601 | .589 | .582 | .595 | .667 | .646 | .672 | .677 | .634 | .651 |
   | c | .662 | .655 | .663 | .655 | .660 | .659 | .580 | .583 | .584 | .583 | .583 | .584 |
   | d | .595 | .645 | .650 | .645 | .647 | .648 | .722 | .761 | .774 | .772 | .763 | .763 |
   | e | .804 | .778 | .778 | .778 | .778 | .778 | .636 | .663 | .663 | .663 | .663 | .663 |
   | f | .615 | .588 | .588 | .588 | .588 | .588 | .606 | .614 | .614 | .640 | .588 | .614 |
   | g | .759 | .738 | .755 | .738 | .739 | .753 | .776 | .738 | .751 | .738 | .740 | .751 |
   | h | .693 | .629 | .677 | .628 | .660 | .670 | .619 | .559 | .575 | .549 | .561 | .580 |
   | **ALL** | .657 | .645 (-.012 [-.026, +.002]) | **.658 (+.001 [-.013, +.014])** | .646 | .649 | .655 | .680 | .675 (-.005 [-.023, +.013]) | **.685 (+.006 [-.010, +.022])** | .684 | .671 | .680 |
   | acceptable (lower >= -.03) | | yes | **YES** | | | | | yes | **YES** | | | |
   | judge calls / search mean, median (p90, max) | 1,247, 1,586 | 47, 50 (100, 201) | 66, 50 (187, 802) | 47 | 46 | 71 | 1,188, 1,495 | 50, 50 (99, 296) | 80, 50 (189, 1,883) | 51 | 49 | 81 |
   R59d - R59c, paired: TRAIN ALL +.012 [+.008, +.018] (52 better / 10 worse by > .05), h +.048 [+.015, +.087], b +.018
   [+.005, +.036], g +.017 [+.006, +.031], c +.008, a +.005, d +.005; dev ALL +.010 [-.001, +.022] (23 / 11), b +.026
   [-.009, +.068], h +.017 [-.038, +.075], d +.013, g +.013 [+.002, +.030]. h vs all-judge: TRAIN -.063 -> -.015, dev
   -.060 -> -.044. Per item vs R59c (TRAIN / dev): (a) +.001 / +.010 [+.002, +.020] (dev b +.032: "sepia/black and white
   ... from <place>"); (b) +.004 [+.001, +.007] / -.004 [-.010, +.001] (TRAIN h +.031; dev b -.012 (3 requests), f -.026
   (1 request; cause not read)); (c) +.010 [+.006, +.014] / +.005 [+.002, +.009] (h +.041 / +.021, g +.016
   / +.014). Cost: mean judge calls +40% TRAIN (47 -> 66), +60% dev (50 -> 80), median unchanged (50); the extension
   judges a whole scope when every block is >= 50% yes (dev max 1,883): still 5.3% / 6.7% of all-judge calls.
   27B facts (reference): TRAIN R59d .659 (+.002 [-.012, +.015] vs all-judge; R59c -.012), h .696 (+.004); dev .692
   (+.012 [-.005, +.030]; R59c .000); R59d - R59c +.013 [+.008, +.019] / +.012 [+.001, +.023].
2. ALL-JUDGE STACK + (a) + (b), dev (every leaf judged on its whole scope; aj0 = the all-judge stack through the same
   harness = the reference on 349/351 requests, the other 2 differ by a harness artifact, ALL -.001)
   | cat | reference | aj0 | + R59c trip route | + (b) parts | **+ (a) + (b)** |
   |---|---|---|---|---|---|
   | a | .755 | .755 | .767 | .754 | .766 |
   | b | .667 | .667 | .642 | .657 | .663 |
   | c | .580 | .580 | .579 | .580 | .579 |
   | d | .722 | .722 | .744 | .722 | .755 |
   | e | .636 | .636 | .636 | .636 | .636 |
   | f | .606 | .606 | .579 | .598 | .598 |
   | g | .776 | .776 | .800 | .776 | .800 |
   | h | .619 | .612 | .613 | .647 | .649 |
   | ALL | .680 | .679 | .680 (+.001) | .679 (-.001) | **.691 (+.011 [+.001, +.022]; vs aj0 +.011 [+.002, +.022])** |
   | judge calls / search | 1,188 | 1,188 | 1,133 | 1,224 | 1,169 |
   Pre-registered adoption bar (ALL > 0, lower >= -.01, no category below -.02: b -.004, c -.002, f -.008) MET. The
   metadata-only trip route alone would cost b -.026 (compound questions) and f -.027; the compound fix recovers most.
   Parts help h (+.028 with or without the trip route); 21 requests better / 9 worse.
3. PRODUCT (adopted): findpics.compose (the route.py rules) behind engine._judge_rows (COMPOSE_QUESTIONS, env FP_COMPOSE=0
   turns it off); FindPicsCore QuestionTree.swift (decompose / evaluate / per-photo evaluateOne). Dev replay of the
   product code on the cached judge (run_system --judge=cache): 321 requests replay without a cache miss in both runs
   (30 error with the composition off, 19 on: anchors the reference judged on images the cache does not hold - the same
   with FP_COMPOSE=0, so not this change); on those 321, composition off = the reference 321/321, composition on = the
   harness's aj59d 320/321 (the one exception is the aj0 harness artifact). Swift vs Python on 660 benchmark judge
   questions with their libraries' place texts: 0/660 mismatches (+ hand cases); pytest 176/176, swift test 152 + 29, 0 failures. Product guard added AFTER scoring: "X or
   not?" / "or anything else" is one question, never split (0 TRAIN/dev questions have such an alternative: no number
   moves). App wiring = MAC_INBOX M74 (not compiled here).
4. RAW TRACES. Sheet dev R59c-only vs R59d-only returned photos (native px, 12 + 6 tiles viewed): R59c-only photos are
   in the truth 105/1,188 (colour photos returned for "my black and white photos from India", a dark non-sepia photo for
   "those sepia photos from the Spain trip", a teddy bear on a beach for "reflections in the water"); R59d-only 597/1,717
   (an elephant for the safari follow-up, a boat and buoy reflected in water, a green-lit concert stage); by eye 10/12
   R59d-only tiles agree with the truth label (2 debatable: a skier with a red backpack for "that red jacket", a cloud
   scene for "dark night shots").
5. READING. The two diagnosed causes were real and fixable in text: compound trip questions (b, dev +.032 from (a)
   alone) and OR follow-ups judged under one budget (TRAIN h +.031 from (b)). The largest single gain is (c): the 5%
   budget was too small exactly where a follow-up narrows a trip/place scope to mostly matches; extending it while the
   judge keeps saying yes costs +19 / +30 mean calls. h is still the weakest category on dev (-.044 vs all-judge, CI
   spans 0). Ship-candidate shape unchanged (towervec over 4B facts + judge for pixel kinds), now R59d.
~9 GPU-h (50 replay shards incl. new 9B answers); CPU: product replays, pytest, swift test. Not pushed.

## 61b. Ten times the distillation data: 313k public photos labelled by the 27B, 2-epoch students - per-photo facts get better, the end-to-end bar still is not met: NOTHING SHIPS (10-14/15; eval/facts_writer/{extra_data,label_extra,train_writer2,r61b_table}.py, launch_{train,ckpt,writer}.sh, relaunch_waiters.sh -> eval/facts_writer/{r61b_table,r61b_quality}.txt, r61b_{train,dev}_*.txt)
Same pre-registered SHIP RULE as RESULTS 61 (f97aee6, unchanged): a writer is acceptable iff the R59c cascade's ALL
difference vs the all-judge stack has lower 95% bound >= -.03 on BOTH TRAIN (643) and dev (351); ship = the smallest
acceptable model, then the fewest output tokens. 61b design pre-registered before any 61b number (JOURNAL 10-14 ~06:05,
b42dd7c). Test closed. Decision on the FINAL checkpoints; intermediate checkpoints are the learning curve only.

DATA (all public, outside the benchmark; extra_data.py). 300,000 Open Images V4 train photos (every one CC BY 2.0 per the
dataset's list; random sample, seed 0, of 1,743,042 after dropping the 642 that ARE DISBench photos by Flickr id and any
by the 57 DISBench users) + the 17,594 Wikimedia Commons iPhone photos already on disk (Commons hosts free licences only);
stored like the benchmark's (500 px, JPEG q92). Near-duplicates of any of the 109,467 DISBench photos (64-bit dHash,
Hamming <= 3) dropped: 3,778, plus 237 duplicates inside the set; a sheet of 4 dropped pairs shows unrelated photos (dHash
collisions), i.e. the check is conservative and the Flickr-id exclusion is the real guard. Kept 313,816 (Open Images
296,343, Commons 17,473). LABELS: the same two 27B steps as the TRAIN targets (VLM JSON facts with the facts.py prompt,
then the text-only short-line rewrite): 313,323 short lines, mean 74.9 tokens (TRAIN targets 75); 24 GPU jobs at ~3
photos/s each (RTX PRO 6000). 6 random labelled photos viewed at native px (r61b_labels.jpg): 5 right, 1 sport misnamed
("rugby pitch" for football).
STUDENTS (train_writer2.py, DDP): same LoRA / input / target as RESULTS 61, base = the phone's 4-bit-sim weights; each
student = its cross-fit TRAIN half (~22.6k photos) + all extra photos (313k), 2 epochs (671,264 photos seen), lr 1e-4
linear decay; 4B-A / 2B-A on 4 GPUs (~5 h), 4B-B / 2B-B on 2 GPUs (~9 h: the lab account's GPU cap); the 0.8B students
were dropped ("if time": the cap). Facts for every evaluated photo still come from a student that never saw it (A writes
TRAIN folds {2,3} + dev, B writes folds {0,1}); phone form = LoRA merged + language re-quantized to MLX 4-bit; each
writer's millisecond scorer retrained on its own facts (TRAIN only; fold models for TRAIN).
Validation token loss (held-out TRAIN-half photos / extra photos), 4B-A: 3.05 -> .664 (100k) -> .621 (200k) -> .586 (400k)
-> .569 (final); 2B-A: .720 -> .668 -> .631 -> .612; B students final 4B .590 / .559, 2B .634 / .600; RESULTS 61's 1-epoch
half-TRAIN students: 4B .687, 2B .752.

LEARNING CURVE + SHIP TABLE (r61b_table.txt; exact bootstrap bounds as compare.py; ALL difference vs the all-judge stack)
| size | writer | photos seen per student | out tokens mean (p90) | repetition | objects F1 vs 27B (dev) | TRAIN R59c diff [95% CI] | dev R59c diff [95% CI] | acceptable | est. index hours, 187,084 photos, at 10 / 20 / 30 tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 27B | short line (61, reference) | - | 75 (100) | .024 | 1 | -.0113 [-.0242, +.0020] | +.0002 [-.0174, +.0181] | yes | 480 / 284 / 219 |
| 4B | RESULTS 61 student | 22,600 | 72 (99) | .030 | .59 | -.0160 [-.0306, -.0025] | -.0116 [-.0303, +.0072] | no | 462 / 275 / 213 |
| 4B | 61b checkpoint | 100,000 | 72 (98) | .029 | .62 | -.0192 [-.0328, -.0050] | -.0068 [-.0255, +.0125] | (curve) | 464 / 276 / 213 |
| 4B | 61b checkpoint | 200,000 | 72 (98) | .020 | .63 | -.0231 [-.0378, -.0083] | -.0074 [-.0261, +.0114] | (curve) | 465 / 277 / 214 |
| 4B | 61b checkpoint | 400,000 | 73 (98) | .018 | .63 | -.0206 [-.0347, -.0063] | +.0006 [-.0174, +.0190] | (curve) | 469 / 279 / 215 |
| **4B** | **61b FINAL** | **671,264** | **75 (100)** | **.021** | **.64** | **-.0183 [-.0318, -.0046]** | **-.0109 [-.0309, +.0096]** | **no** | **476 / 282 / 218** |
| 2B | RESULTS 61 student | 22,600 | 71 (97) | .047 | .54 | -.0226 [-.0378, -.0085] | -.0103 [-.0296, +.0091] | no | 457 / 273 / 211 |
| 2B | 61b checkpoint | 100,000 | 82 (120) | .241 | .37 | -.0257 [-.0398, -.0107] | -.0096 [-.0291, +.0106] | (curve) | 515 / 302 / 231 |
| 2B | 61b checkpoint | 400,000 | 76 (104) | .065 | .56 | -.0196 [-.0339, -.0052] | -.0030 [-.0210, +.0157] | (curve) | 481 / 285 / 219 |
| **2B** | **61b FINAL** | **671,264** | **77 (108)** | **.084** | **.55** | **-.0228 [-.0376, -.0085]** | **-.0101 [-.0279, +.0083]** | **no** | **489 / 288 / 222** |
| 0.8B | RESULTS 61 student | 22,600 | 69 (94) | .031 | .54 | -.0290 [-.0447, -.0135] | -.0112 [-.0305, +.0084] | no | 448 / 268 / 208 |
Index hours = 187,084 x (1.7 s measured judge call + tokens / r), r unmeasured (M70). Repetition = at the 120 cap or a
word 3-gram repeated >= 3 times. Objects F1 = set F1 of the objects field vs the 27B's line on 24,564 dev photos
(r61b_quality.txt also has colours, scene, people count, text; TRAIN out-of-fold numbers agree).
Per category (TRAIN; all-judge / 27B short line / 4B 61 / 4B 61b final / 2B 61b final): a .682 / .661 / .657 / .640 /
.643; b .601 / .593 / .578 / .589 / .582; c .662 / .641 / .645 / .645 / .642; d .595 / .643 / .634 / .635 / .630; e .804 /
.804 / .783 / .788 / .740; f .615 / .579 / .583 / .578 / .565; g .759 / .731 / .728 / .723 / .732; h .693 / .629 / .635 /
.628 / .629.

READING
- By the unchanged rule NOTHING SHIPS: the 4B final misses on TRAIN (lower -.0318) and on dev (-.0309); the 2B final on
  TRAIN (-.0376). 10x the data (and 30x the photos seen) did not move the end-to-end number: the 4B curve is flat within
  noise from 23k to 671k photos seen (TRAIN -.016 / -.019 / -.023 / -.021 / -.018; dev -.012 / -.007 / -.007 / +.001 /
  -.011), while the facts themselves did improve (objects F1 vs the 27B .59 -> .64, colour phrases .37 -> .41, scene .59 ->
  .65, repetition .030 -> .021; validation token loss .687 -> .569).
- Why the end-to-end number does not follow: the cascade itself sits at -.011 [-.024, +.002] on TRAIN even with the 27B's
  own short lines, so a writer has ~.005 of room before its lower bound crosses -.03, about the size of the noise between
  two runs here (the 4B checkpoints span -.016..-.023 on TRAIN with no trend). The 4B final's remaining loss vs the 27B
  line sits in a (objects; TRAIN .640 vs .661) and, on dev, e / f (the one distinctive thing; .667 vs .722, .559 vs .636).
- The 2B got WORSE at repeating itself with more training (repetition .047 -> .084, and .24 mid-schedule at 100k):
  greedy decoding of a 2B on this format loops; the 4B does not (.021).
- Cost: unchanged by the data (~73-77 tokens): est. 476 / 282 / 218 phone-hours at 10 / 20 / 30 tok/s for the 4B, vs
  793 at 20 tok/s for the zero-shot 4B JSON writer that DOES pass (RESULTS 59c).
- NEXT (what would move the decision, by size): (1) the cascade's own TRAIN gap (-.011 with 27B facts): fix the 59c
  compound-trip flaw (b) and the h follow-up losses - every writer gains that headroom; (2) measure decode tok/s and vision
  + prefill time per size on the iPhone (M70) - it decides whether the zero-shot 4B JSON writer (passes, ~3x the tokens) is
  affordable at all; (3) a 2B needs anti-loop decoding or training (e.g. stop on a repeated item) before it is usable.
~68 GPU-h (27B labelling 24 x ~1.2 h, students 4 x 4-GPU / 2-GPU for 5-9 h, 7 checkpoint evaluations x ~32 jobs). The
login-node waiters died twice with the parent session (07:20-16:45, 17:10-19:25); training ran on in Slurm and the
idempotent launchers resumed the evaluation; nothing was re-run or re-decided because of the gaps. Not pushed.

## 61c. The existing short-line writers under the R59d cascade: ALL acceptable; by the pre-registered rule the 2B student (61b final) ships (10-15; pre-registered 6c9962f; eval/facts_writer/r61c_launch.sh -> eval/facts_writer/{r61c_table,r61c_train_*,r61c_dev_*}.txt)
Test closed. PRE-REGISTERED (JOURNAL, 6c9962f) before any number: re-evaluate the EXISTING writers - 27B short line
(reference, server only), 4B RESULTS-61 student, 4B and 2B 61b finals (671k photos seen) - under the R59d cascade
(route.py / cascade.py as of 500ade1, variant r59d) with NO retraining (same cross-fit facts, same per-writer millisecond
scorers: TRAIN fold models, full model for dev); bar unchanged from f97aee6: acceptable iff the ALL difference vs the
all-judge stack (TRAIN 643, dev 351) has exact-bootstrap lower 95% bound >= -.03 on BOTH; ship = the smallest acceptable
model (2B < 4B < 27B), then the fewest mean output tokens. R59c re-run in the same jobs reproduces RESULTS 61 / 61b to the
4th decimal on all four writers (reproduction check). One dev request (24413182@N00:e19, 27B short line) failed on a
vLLM engine-init error of the lazily loaded 9B (infrastructure, F1 0); its records were removed and that shard re-run
(log kept as cascade_3.log.enginefail); no other errors (100 shards).

| writer | size | out tokens / photo mean (p90) | TRAIN R59d ALL (diff [95% CI]) | dev R59d ALL (diff [95% CI]) | judge calls / search mean, median (TRAIN; dev) | R59c (TRAIN / dev diff) | acceptable under R59d | est. phone index hours, 187,084 photos, at 10 / 20 / 30 tok/s |
|---|---|---|---|---|---|---|---|---|
| 27B short line (reference) | 27B | 75 (100) | .660 (+.0032 [-.0087, +.0165]) | .688 (+.0078 [-.0101, +.0256]) | 65, 50; 80, 50 | -.0113 / +.0003 | yes | 480 / 284 / 219 |
| 4B student, RESULTS 61 (22.6k seen) | 4B | 72 (99) | .655 (-.0017 [-.0151, +.0124]) | .678 (-.0019 [-.0201, +.0167]) | 65, 50; 79, 50 | -.0160 / -.0116 | yes | 462 / 275 / 213 |
| 4B student, 61b final (671k) | 4B | 75 (100) | .646 (-.0113 [-.0252, +.0031]) | .680 (+.0002 [-.0198, +.0209]) | 64, 50; 81, 50 | -.0183 / -.0109 | yes | 476 / 282 / 218 |
| **2B student, 61b final (671k)** | **2B** | **77 (108)** | **.646 (-.0110 [-.0254, +.0030])** | **.677 (-.0024 [-.0213, +.0154])** | **65, 50; 79, 50** | -.0228 / -.0101 | **YES -> SHIP** | **489 / 288 / 222** |
All-judge reference: TRAIN .657 (1,247 judge calls / search mean, 1,586 median), dev .680 (1,188 / 1,495); the R59d
cascade uses ~5-7% of those calls (median 50). Index hours = 187,084 x (1.7 s measured judge call + tokens / r), decode
rate r unmeasured (M70); a 2B decodes faster per token than a 4B, which this column does not credit.
Per category under R59d (TRAIN; all-judge / 27B line / 4B-61 / 4B-61b / 2B-61b): a .682 / .667 / .662 / .636 / .646; b .601 /
.609 / .600 / .598 / .598; c .662 / .652 / .653 / .658 / .650; d .595 / .650 / .642 / .647 / .638; e .804 / .804 / .783 /
.788 / .740; f .615 / .579 / .583 / .578 / .565; g .759 / .751 / .749 / .728 / .746; h .693 / .690 / .686 / .646 / .675.
Dev: a .755 / .746 / .726 / .745 / .747; b .667 / .676 / .669 / .667 / .671; c .580 / .582 / .581 / .592 / .590; d .722 /
.768 / .760 / .767 / .754; e .636 / .722 / .718 / .667 / .604; f .606 / .640 / .568 / .559 / .615; g .776 / .741 / .759 / .757
/ .754; h .619 / .554 / .578 / .576 / .540.

VERDICT (by the rule, decided before the numbers): every writer clears the bar under R59d; the smallest acceptable model
is the 2B 61b final student -> SHIP it as the phone's facts writer (77 output tokens / photo, est. 288 phone-hours for
187k photos at an assumed 20 tok/s, ~3x below the zero-shot 4B JSON writer's 793). R59d added +.011..+.020 to every
writer (it removed the cascade's own TRAIN gap), which is what moved them over the bar - the writers themselves are the
same as in 61 / 61b.
CAVEATS, stated, not decision changes: (1) the 2B is the weakest writer on e (TRAIN .740 vs .788 for the 4B, dev .604) and
its outputs loop on 8.4% of photos (61b) - the phone should stop generation on a repeated list item, and M70 must
measure its real decode speed; (2) the 4B RESULTS-61 student is the strongest student here (TRAIN -.002, dev -.002) at
the fewest tokens (72) - the rule picks size first; (3) truth = 27B-oracle candidate sets; dev / TRAIN only. ~2 GPU-h.
Not pushed.

## 61d. Preparing the shipped 2B facts writer: ONE product adapter (dev reproduces 61c: +.000 [-.018, +.018]), an MLX checkpoint bit-identical to the measured phone form, real-MLX parity at vLLM's own noise level, and REPEAT-STOP v2 adopted (10-15/16; pre-registered 0df08e1 + v2 072f8cb; src/findpics/facts_line.py, FindPicsCore FactsLine.swift + FactsLineTests; eval/facts_writer/{to_mlx,mlx_cmp,r61d_repeat,r61d_parity_{prep,mlx,vllm,cmp},r61d_table,r61d_fix_errors,facts_line_fixture}.py, r61d_{prod_job,prod_launch,launch_rs,parity_job}.sh -> eval/facts_writer/{r61d_table,r61d_repeat_v1,r61d_repeat_v2,r61d_repeat_p2b_v2,r61d_quality,r61d_quality_p2b,r61d_parity_t2bA,r61d_parity_p2b}.txt, r61d_{train,dev}_*.txt)
Test closed. Bar unchanged (f97aee6): R59d cascade ALL difference vs the all-judge stack, exact bootstrap lower 95% >= -.03.

(1) PRODUCT WEIGHTS (pre-registered 0df08e1). The 61b fold adapters each miss half of TRAIN; the app ships ONE adapter
p2b_all = the 61b recipe unchanged (LoRA r16 + vision r16 on the 4-bit-sim Qwen3-VL-2B, 2 epochs, lr 1e-4 linear, 4 GPUs)
on ALL non-test training data: 45,691 TRAIN photos + 313,023 extra public photos (717,440 seen, 22,796 s on 4 H100;
held-out validation loss TRAIN-photos .611 / extra .616 vs 61b 2B-A .612 / .619). USE CONDITION: dev facts through R59d with
the 61c scorer unchanged must land inside the 61c dev CI [-.0213, +.0154]. RESULT: dev ALL .680 vs all-judge .680, +.0001
[-.0182, +.0176] (n=351) -> INSIDE -> the product adapter is used. (TRAIN not scored: it is in the training data.) Per
category dev (all-judge / product): a .755/.745, b .667/.669, c .580/.591, d .722/.753, e .636/.606, f .606/.632, g .776/.751,
h .619/.564; judge calls / search mean 80, median 50 (6.7% of all-judge). Its facts are better than the 61b student's on
the same 24,564 dev photos (vs the 27B line): objects F1 .60 vs .55, colour words .81 vs .75, scene .61 vs .57, people
count within 1 .87 vs .80, text .76 vs .70; 74.1 tokens (p90 100) vs 78.5; at the 120 cap 136/24,598 vs 1,509/24,598.
20 raw product lines read next to the 27B's (seeded): all keep the field format; content errors are species / scene
names (a "baboon" for a monkey, "bowling alley" for an exhibition hall, "stone barn" for a fort) and invented OCR text.

(2) MLX FORM. Changed BEFORE any parity number from the pre-registered adapter to MERGED 4-bit weights: mlx-swift-lm
3.32.3's Qwen3VL exposes only the language model's layers to LoRA (`loraLayers`), so the student's VISION LoRA cannot load
as an adapter, and the 2B base is not otherwise on the phone. eval/facts_writer/to_mlx.py writes mlx-community's exact
layout (1,019 tensors, U32 codes + bf16 scales / biases, channels-last patch embed). Checks: (a) converting Qwen's bf16
2B with CPU arithmetic reproduces the real mlx-community/Qwen3-VL-2B-Instruct-4bit file bit for bit (2,181,296,128 /
2,181,296,128 elements); on CUDA the same formula differs in ~0.3% of codes (so every merge_q4.py phone form is within +-2
codes of true MLX rounding on ~0.3% of weights); (b) merge_q4.py rounds the TIED embedding twice (embed_tokens, then
lm_head = the same tensor), so to_mlx.py reproduces that double rounding: the shipped MLX checkpoint de-quantizes to the
measured vLLM phone form EXACTLY (2,127,532,032 / 2,127,532,032 weights, for the 61b student and for p2b_all).
Shipped file: eval/facts_writer/models/p2b_all_final_mlx/ (1.7 GB; not in git).
PARITY, 50 public dev photos (seed 0), real MLX (mlx[cuda12] 0.32.3 + mlx-vlm 0.7.4 on an H100; MLX's CPU backend needed
1,815 s for ONE photo) vs the batched vLLM phone-form lines; noise floor = the same vLLM at batch 1. Preprocessing
identical (pixel_values and prompt ids, 3/3 photos, max diff 0; prompt token counts 50/50).
| writer | candidate | exact lines | objects F1 vs vLLM | colours | scene | people | tokens mean (vLLM) | objects F1 vs 27B (vLLM) |
|---|---|---|---|---|---|---|---|---|
| product p2b_all | MLX, repetition context = all (vLLM semantics) | 19/50 | .96 | .87 | 1.00 | .90 (n=25) | 71.2 (71.6) | .62 (.63) |
| product p2b_all | MLX, repetition context 20 (MLX / mlx-swift default) | 11/50 | .95 | .80 | 1.00 | .87 | 70.6 | .62 |
| product p2b_all | vLLM batch 1 (noise floor) | 26/50 | .96 | .91 | .99 | .89 | 72.4 | .63 |
| 61b 2B-A | MLX, context all | 11/50 | .92 | .87 | .92 | .89 (n=22) | 75.9 (75.6) | .53 (.54) |
| 61b 2B-A | MLX, context 20 | 7/50 | .90 | .78 | .92 | .88 | 73.4 | .53 |
| 61b 2B-A | vLLM batch 1 (noise floor) | 21/50 | .94 | .89 | .93 | .89 | 74.7 | .53 |
Reading: real MLX diverges from vLLM by near-tie flips ("22.09.2014" vs "22-09-2014", "pillow" vs "blanket"), a little
more often than vLLM diverges from itself across batch sizes, and with the same quality vs the 27B (50 photos). One
difference is not noise: MLX's default repetition-penalty window (last 20 tokens) is not what was measured - the app must
set repetitionContextSize to cover prompt + output (e.g. 4096). Free-text OCR fields are the most fragile (1/50 product
photo: entirely different invented sign text between runtimes).

(3) REPEAT-STOP (findpics.facts_line; Swift FactsLine, golden fixture 558 cases incl. streamed prefixes; pytest 179/179,
swift test 155 + 29, 0 failures). v1 (pre-registered): stop when a completed item repeats within a ";" field. Measured
on the 61b 2B final's 70,580 cross-fit TRAIN+dev lines it caught only 705/5,933 looping lines and cut 752 non-looping
ones: the dominant loop is FORMAT COLLAPSE into an unnamed ";"-list ("mask; gloves; shoes; pants; shirt; mask; ..."), which
v1 resets on every ";", and its non-loop cuts were legitimate duplicates ("person in dark clothing walking" x2). v2 was
pre-registered after seeing those lines, before any v2 number: fields start only at "name:", stop at an item's 3rd
occurrence. Decision rule: adoptable iff the bar passes on TRAIN and dev and the loop rate falls; adopt the lower loop rate.
| 2B writer | tokens mean (p90) | loop rate (quality.rep) | at 120 cap | TRAIN R59d diff [95%] (n=643) | dev R59d diff [95%] (n=351) | bar |
|---|---|---|---|---|---|---|
| 61b final, no stop (61c) | 77.0 (108) | .084 | 4,150/70,580 | -.0110 [-.0254, +.0030] | -.0024 [-.0213, +.0154] | pass |
| + v1 (scorers retrained) | 76.3 (106) | .075 | 3,806 | -.0087 [-.0226, +.0054] | +.0055 [-.0113, +.0233] | pass |
| **+ v2 (scorers retrained) ADOPTED** | **74.9 (103)** | **.054** | **1,512** | **-.0068 [-.0203, +.0069]** | **+.0070 [-.0100, +.0242]** | **pass** |
| PLACEBO: no stop, scorers retrained at seed 1 | 77.0 (108) | .084 | 4,150 | -.0042 [-.0176, +.0089] | +.0098 [-.0076, +.0279] | pass |
| PRODUCT p2b_all, no stop (61c scorer) | 74.1 (100) | .033 | 136/24,598 dev | - | +.0001 [-.0182, +.0176] | dev: pass |
| PRODUCT p2b_all + v2 (v2 scorer) | 74.1 (100) | .033 | 127/24,598 | - | +.0054 [-.0125, +.0241] | dev: pass |
v2 facts F1 vs the 27B unchanged to 2 decimals in every field (v1 lost activity .56 -> .54 and text .70 -> .68 on dev).
Reading: (i) the cascade-level effect of repeat-stop is NOT distinguishable from scorer-retraining noise - retraining the
millisecond scorer on identical facts with another seed moved dev by +.012 [+.003, +.023] paired, as much as v1 / v2; what
v2 measurably buys is decode work (at-cap lines -64%, -2.1 tokens/photo) at no facts cost; (ii) on the PRODUCT adapter
repeat-stop barely matters: it fires on 14/24,598 dev lines (its .033 "loop" rate is mostly quality.rep's 3-gram rule
firing on honest people lists) - more in-domain data cured most of the 61b student's looping; v2 stays on as a guard.
INFRASTRUCTURE: 15/643 v1 TRAIN, 4 + 1 dev and 10 placebo TRAIN requests first failed because the lazily loaded 9B judge
cannot init on A100-SXM4-40GB (vLLM "max_num_seqs 256 exceeds available Mamba cache blocks 201"); the first v1 TRAIN
number (-.025) was this artifact. Errored records removed, shards re-run on H100 / RTX (logs *.enginefail), A100-40GB
removed from the 61d launchers; every number above has 0 errored requests.
VERDICT: ship p2b_all as the facts writer in the MLX form above (conditional on device speed, M70 / M76 unchanged), with
repeat-stop v2 at decode and repetition penalty 1.05 over the full prompt + output. ~25 GPU-h (training 4 x 6.3 h, ~100
cascade shards, parity). Not pushed.

## 64. Write the facts line ONCE per near-duplicate group: the cascade does not notice (TRAIN and dev pass at every cosine down to .90, 20% of writer calls saved), but the shared lines are often WRONG for the member by eye (11/30 strict at .90) - NOT ADOPTED as pre-registered (pre-registered da3f15f; 10-16; eval/dedup64/{group,launch.sh,table,sheets}.py -> eval/dedup64/{train_table,dev_table}.txt, data/saved_s*.txt, sheets/pairs_s*.txt, sheets/lost_train_s0.90.txt)
Test closed; TRAIN + dev only; public bench photos. RULE G(s): per library, capture order; a photo joins an earlier
representative iff 0-10 min after it, PE-Core-L-14-336 cosine >= s, and <= 1 km when both have GPS (leader grouping, no
chaining); the member gets the rep's writer line, keeps its own place/month prefix and image vector. No retraining (61d
scorers, plans, judge cache). Per-photo baselines = 61d (TRAIN: cross-fit 61b 2B + repeat-stop v2; dev: product p2b_all +
v2). Bar: (A) shared - per-photo lower 95% >= -.01, (B) shared - all-judge lower 95% >= -.03, both splits; s = the smallest
grid value passing on TRAIN; dev once at that s; ship also needs >= 27/30 eye-checked pairs to be true near-duplicates
whose shared line is right for the member.

| s | writer calls saved TRAIN | dev | TRAIN shared - per-photo [95%] (n=643) | TRAIN shared - all-judge | dev shared - per-photo (n=351) | dev shared - all-judge | judge calls / search mean, median (TRAIN; dev) |
|---|---|---|---|---|---|---|---|
| inf (repro) | 0/45,982 | - | +.0002 [-.0000, +.0006] | -.0066 [-.0201, +.0071] | - | - | 65, 50 |
| .98 | 900/45,982 (2.0%) | 388/24,598 (1.6%) | +.0002 [-.0002, +.0007] | -.0065 [-.0203, +.0072] | - | - | 65, 50 |
| .96 | 2,926 (6.4%) | 1,432 (5.8%) | +.0001 [-.0005, +.0006] | -.0067 [-.0203, +.0069] | - | - | 65, 50 |
| .94 | 5,230 (11.4%) | 2,597 (10.6%) | +.0001 [-.0007, +.0010] | -.0066 [-.0202, +.0070] | - | - | 65, 50 |
| .92 | 7,438 (16.2%) | 3,817 (15.5%) | -.0000 [-.0011, +.0010] | -.0068 [-.0204, +.0069] | - | - | 65, 50 |
| **.90 (selected)** | **9,551 (20.8%)** | **4,977 (20.2%)** | **+.0003 [-.0007, +.0015]** (12 better / 10 worse by > .05) | **-.0064 [-.0201, +.0072]** | **-.0002 [-.0026, +.0019]** (4 / 3) | **+.0052 [-.0132, +.0243]** | 65, 50; 80, 50 |
Per-photo baselines: TRAIN ALL .6503 (-.0068 [-.0203, +.0069] vs all-judge), dev .6852 (+.0054 [-.0125, +.0241]); 0 errored
requests anywhere. Writer calls saved per library at .90: TRAIN median 17.8% (min 3.4%, max 46.5%, 24 libraries), dev median
20.0% (4.5%-40.7%, 13 libraries). Bars A and B pass at every s on TRAIN and at .90 on dev.

EYE CHECK (pre-registered ship condition; 30 (rep, member) pairs at s = .90, seed 0, 15 TRAIN + 15 dev; the bench originals
are 500 px on the long side, shown upscaled to 900 px - no higher resolution exists): 11/30 true near-duplicates whose shared
line is right for the member; 9/30 same moment and subject, with a minor line error for the member (people count 1 vs 2,
"walking" for a lying jaguar, clouds for a clear-sky shot); 9/30 same place / event but a DIFFERENT shot with material line
errors (other room of Versailles, other street in Bergen, other cove at Port Campbell, empty pews described as "people: 10");
1/30 an EDITED COPY (sepia vs colour version of one car photo: the colour member gets "look: sepia, black car"); 0/30
different subjects. 11/30 (or 20/30 counting minor errors) < 27/30 -> FAIL -> NOT ADOPTED, by the rule.
LOST TRUTH (TRAIN, .90): member photos that were truth and returned per-photo but not shared 158, gained 151 (wrong photos
dropped 220, added 227; per-photo truth returned 20,759): the churn is symmetric. 4 of 8 lost cases viewed: 3 are different
shots whose rep line lacks the member's asked attribute (climber in a red helmet vs the rep's blue-helmet climber; magenta vs
red backdrop; tower over a reflecting pond vs tower without the pond); 1 is a true near-duplicate (both lines say "red
collar"), lost by ranking churn.
EXPLORATORY, NOT pre-registered (after the decision): 20 pairs at s = .96 (seed 1, 10 + 10): 15/20 strict near-duplicates, 3/20
minor line errors, 2/20 material (the other facade of the Sagrada Familia 5.8 min later; a grey-sky vs blue-sky shot of the
same shuttle carrier). Higher cosine cleans groups but saves only ~6%.

REPRESENTATIVENESS of the bench's duplicate rate for a phone camera roll: UNKNOWN. The DISBench libraries are Flickr
UPLOADS (owner-selected, 500 px), not camera rolls with bursts, Live Photo stills, screenshots and retakes. No public
smartphone camera-roll dataset with a near-duplicate rate was found; the only number found is film-era (Kodak consumer
database, Jaimes et al. ACM MM 2002: ~19% of images per roll are duplicates or near-duplicates). Reza's library was not used.
If a phone library had the bench's rate, G(.90) would cut the 61c index estimate 222-489 phone-h to ~176-387 phone-h (x 0.79);
the real phone rate may be higher (bursts), which is also exactly where groups are cleanest (dt <= 1 min: 8/12 strict at .90).

READING. (1) The end-to-end bench metric is blind to this change: even at .90 the ALL difference is +-.000 with CIs inside
+-.003, because a member's own image vector still goes to the scorer and pixel kinds still go to the 9B; the bench cannot
tell a correct shared line from a wrong one at this rate. (2) The eye check shows the lines ARE often wrong for the member
(different shot of the same place; colour edits, which PE-Core cosine barely separates). A user asking for "the magenta
lighting" or "the red helmet" would lose real matches, and gain wrong ones, at the margins of a burst. (3) Candidate for a
future pre-registered test (not run): tighter time (<= 1 min) + cosine >= .96 + a pixel colour-statistics guard against
edited copies, measured by a member-vs-rep facts agreement on TRAIN (cheap) rather than only the blind end-to-end bar.
~2 GPU-h (85 cascade shards). Not pushed.

## 63. Complex bench truth pass for b, c, g on dev + TRAIN: a direct 27B judgment of the owner's words on every photo, a 122B third opinion on every disagreement, 420 photos by eye - only REMOVALS pass the flip rule; g becomes VERIFIED on the corrected truth, b and c stay PROVISIONAL; the shipped stack's ranking vs all-judge is unchanged (pre-registered 6c0b74a; 10-16; eval/complex_bench/t63/{direct,t63}.py, job.sh, rescore.sh -> t63/{units,report}.txt, labels_main.txt, key.json, queries/{dev,train}.json, rescore/{truth_*,compare_*}.txt)
Test split closed (no request, truth or image read). Rules written and committed before any answer was seen (AUDIT_PROTOCOL.txt
"ADDED 10-16"). Unit = (request, album, photo), every photo of the library (named-person albums: photos where the bench's face
match finds the person, red box drawn; identity not re-derived).
- T = current truth (program + 27B oracle). D = Qwen3.5-27B-FP8 asked the WORDS directly ("photo taken <weekday date time>,
  at <address>. Today is <today>. Someone ... asked: "<words>". Does this photo belong in the answer / in the FIRST|SECOND
  group?"), P >= .5, on 911,395 units (412 requests: b 177, c 138, g 97; ~1.5 GPU-h). L = Qwen3.5-122B-A10B-FP8, same prompt,
  on every D != T unit + the agreed-strata eye sample (29,905 units, tp=2).
- Strata per category (dev + TRAIN pooled): A11 T=D=1; FADD T=0,D=1,L=1 (flip candidate, add); KADD T=0,D=1,L=0; FREM T=1,
  D=0,L=0 (flip candidate, remove); KREM T=1,D=0,L=1; agreed-out N1/N2/N3 by D's P [.2,.5) / [.05,.2) / <.05.
- Eye: I labelled 420 sampled units myself on 70 shuffled sheets at native pixels (<= 500 px, what every judge sees; crops
  enlarged where small), blind to stratum / truth / judges, labels committed (b0efd34) BEFORE key.json was read; 23 unsure.
  One rule detail added before the key was read: a g photo labelled "both" counts as a match for each album.

| category | eye: A11 / FADD / KADD / FREM / KREM / N1 / N2 / N3 (match / labelled) | flip add (bar .70) | flip remove (bar .70) | eye agreement on disputed units: T / D (27B) / L (122B) | corrected truth P [95%] / R [95%] | VERIFIED |
|---|---|---|---|---|---|---|
| b | 13/15, 14/30, 3/15, 5/29, 7/15, 0/15, 0/10, 0/10 | 14/30 = .47 NO | 24/29 = .83 REMOVE | 40/89, 49/89, 57/89 | .86 [.67, 1.00] / .52 [.42, .64] | no (recall) |
| c | 9/15, 10/26, 0/15, 2/29, 2/13, 0/15, 0/10, 0/10 | 10/26 = .38 NO | 27/29 = .93 REMOVE | 35/83, 48/83, 54/83 | .59 [.33, .85] / .68 [.52, .80] | no (precision) |
| g | 13/15, 10/27, 0/15, 1/21, 4/12, 0/15, 0/10, 0/10 | 10/27 = .37 NO | 20/21 = .95 REMOVE | 37/75, 38/75, 49/75 | .86 [.66, .99] / .82 [.75, .90] | YES (point estimates, as pre-registered) |

Same estimate for the other versions (P / R): old truth b .85 / .53, c .51 / .69, g .81 / .83; D alone b .52 / 1.00, c .38 /
.98, g .55 / .99; L on disputes b .64 / .89, c .50 / .99, g .70 / 1.00. Photos FLIPPED (removed; nothing added): b dev 46 of
2,231 truth units, TRAIN 73 of 3,931; c dev 388 of 1,791, TRAIN 179 of 1,876; g dev 168 of 4,507, TRAIN 640 of 8,130. Two TRAIN
requests become empty and are dropped from the corrected file (21435131@N06:g28, 24736216@N07:c10).

EFFECT ON THE SHIPPED STACK (R59d cascade + 2B 61b writer + repeat-stop v2 = c61d_*_t2b_final_rs2_xfit/r59d; same runs
re-scored, no new product run; final-round F1, paired over requests):
| split | b old -> new | c old -> new | g old -> new | ALL old -> new | shipped - all-judge, ALL: old truth / new truth | ALL verified (new flags a,d,e,f,g,h): shipped - all-judge |
|---|---|---|---|---|---|---|
| dev (351 -> 351) | .673 -> .671 (-.002 [-.014, +.006]) | .586 -> .602 (+.016 [+.007, +.026]) | .766 -> .779 (+.013 [+.001, +.034]) | .687 -> .690 (+.003 [+.000, +.006]) | +.007 [-.010, +.024] / +.006 [-.010, +.023] | .714 vs .706: +.007 [-.016, +.032] (237 req) |
| TRAIN (643 -> 641) | .603 -> .605 (+.002) | .664 -> .691 (+.027 [+.009, +.049]) | .746 -> .749 (+.003) | .651 -> .655 (+.004 [+.002, +.008]) | -.007 [-.021, +.007] / -.007 [-.021, +.008] | .661 vs .671: -.010 [-.029, +.010] (441 req) |
The all-judge reference moves the same way (dev c +.014, g +.015; TRAIN c +.024): the correction raises absolute levels a
little and changes NO comparison made so far (61c / 61d decisions unchanged in sign and CI).

READING. (1) The direct judges are lenient, the programs are conservative: D says yes to 2-3x the truth (b dev 7,423 vs
2,231), and the photos only D + L add are right 14/30, 10/26, 10/27 by eye - so additions do not pass and the truths' main
error is MISSES, which no rule here may fix: b's estimated recall is .52 (FADD alone holds ~3,600 right photos the truth
lacks; eye-confirmed examples: b&w London gig shots, purple / magenta / blue stage light, reflections, motion blur). (2) Where BOTH judges reject a
truth photo they are almost always right (71/79 by eye over b, c, g): those are program errors ("shots of people smoking": 10
sampled truth photos with no cigarette; "smooth water from that river trip": rapids; the wrong group or date in g). The
removal errors are real too (b: 4 wedding-day close-ups both judges rejected, 5/29). (3) c fails on precision even
where everyone agrees (A11 9/15): the truth AND both judges accept loose readings of people's looks ("Sam in the beige
cardigan": 3 agreed photos show a white dress, a white top, a beige blouse); c needs a program rewrite + a person-look audit, not more judges. (4) Of the three
opinions on disputed photos the 122B agrees with the eye most (160/247 vs 27B 135/247, truth 112/247) - still only 65%.
CAVEATS. g VERIFIED rests on point estimates (P lower 95% bound .66); the pre-registered rule used point estimates as RESULTS
52/53/56 did. All three opinions are Qwen3.5 (correlated). The eye labeller is me (Claude), one pass, no second labeller; 23
unsure (7 are "Chicago winter 2010" = Jan-Feb 2011?). The N3 stratum (<.05, ~620k units) had 0/30 matches, so its misses are
estimated as 0 (upper bound not 0). The corrected files live in t63/queries/; queries/ (the bench the other tracks score
against) is NOT replaced here - adopting t63/queries + FP_CB_VERIFIED=a,d,e,f,g,h is a coordinator decision (it moves every
c/g number by the amounts above). No product change. ~2.2 GPU-h (27B 16 GPUs ~1.4 h wall incl. queue; 122B 8 x 2 GPUs ~15 min).

## 65. Search while facts are PARTIAL: photos without a facts line must be scored by the image-vector scorer (V), not by the current "no description" fallback (+.026 curve area); newest-first stays the writing order; the cascade bar is met from 25% coverage - ADOPTED: policy V, order N (pre-registered 0414301; 10-16; eval/partial65/{orders,cascade65,table,sheets}.py, job.sh, launch.sh -> eval/partial65/table.txt, sheets/lost_{0,25}_N_V.txt)
Why: the shipped 2B writer needs est. 222-489 phone-hours for 187k photos, so for days only part of a library has facts.
Test closed; TRAIN (24 libraries, 643 requests) + dev (13, 351). No new writer runs, no retraining: R59d cascade (frozen
.cache/wt_r59b/src), shipped plans, judge cache (+ lazy 9B), 61d facts (TRAIN cross-fit 61b 2B + v2; dev product p2b_all
+ v2), towervec tvw_t2b_final_rs2_xfit_h{fold}/_full, vecB_h{fold}/_full. Coverage c = in EACH library the first
ceil(c x n) photos of the writing order have a line. Orders: N newest-first (capture time; undated last), R random, E one
photo per event first (3 h gap events, newest events first, round-robin). Policies for photos WITHOUT a line: A = current
code (towervec reads the Scorer default text "no description" + the image vector), V = image-vector-only scorer vecB in
the same ranking and judge budget, VS = V + a separate judge budget for the no-facts photos. Reproduction: 100% = the 61d
baseline (TRAIN .6505 vs .6503, dev .6852 vs .6852). 0 errored requests in all 39 configs x 994 requests.

ALL F1 diff vs all-judge [95%], judge calls / search mean, median (TRAIN all-judge .657 / 1,247 calls; dev .680 / 1,188):
| coverage | order | policy | TRAIN diff | TRAIN calls | dev diff | dev calls |
|---|---|---|---|---|---|---|
| 0% | - | A (current) | -.065 [-.083, -.046] | 63, 50 | -.071 [-.100, -.044] | 76, 50 |
| 0% | - | V | -.025 [-.040, -.009] | 64, 50 | -.013 [-.034, +.008] | 79, 50 |
| 10% | N | A | -.062 [-.080, -.043] | 63, 50 | -.061 [-.088, -.033] | 76, 50 |
| 10% | N | **V** | -.021 [-.036, -.006] | 65, 50 | -.011 [-.031, +.011] | 79, 50 |
| 10% | N | VS | -.022 [-.036, -.007] | 83, 50 | -.011 [-.031, +.011] | 99, 59 |
| 10% | R / E | V | -.025 / -.022 | 64, 50 | -.012 / -.015 | 79, 50 |
| 25% | N | A | -.049 [-.066, -.031] | 62, 50 | -.047 [-.072, -.021] | 76, 50 |
| 25% | N | **V** | -.014 [-.028, +.000] | 64, 50 | -.008 [-.027, +.013] | 80, 50 |
| 25% | N | VS | -.013 [-.026, +.001] | 79, 50 | -.007 [-.027, +.013] | 96, 72 |
| 25% | R / E | V | -.020 [-.034, -.005] / -.021 [-.035, -.006] | 64-65, 50 | -.009 / -.011 | 79, 50 |
| 25% | R / E | A | -.052 / -.044 | 62, 50 | -.058 / -.045 | 76-77, 50 |
| 50% | N | A | -.038 [-.055, -.021] | 63, 50 | -.034 [-.058, -.010] | 78, 50 |
| 50% | N | **V** | -.012 [-.026, +.002] | 64, 50 | -.004 [-.025, +.016] | 79, 50 |
| 50% | N | VS | -.012 [-.025, +.002] | 72, 50 | -.001 [-.022, +.019] | 88, 62 |
| 50% | R / E | V | -.017 / -.017 | 65, 50 | -.006 / -.005 | 79, 50 |
| 75% | N | A | -.025 [-.040, -.009] | 63, 50 | -.020 [-.042, +.002] | 78, 50 |
| 75% | N | **V** | -.007 [-.021, +.007] | 65, 50 | -.002 [-.022, +.017] | 80, 50 |
| 75% | N | VS | -.008 [-.021, +.006] | 79, 50 | -.001 [-.021, +.019] | 94, 53 |
| 75% | R / E | V | -.011 / -.014 | 65, 50 | +.002 / -.004 | 80, 50 |
| 100% | - | - | -.007 [-.020, +.007] | 65, 50 | +.005 [-.013, +.024] | 80, 50 |
Full 39-row table (every order x policy, ALL F1 and CIs): eval/partial65/table.txt.

PRE-REGISTERED DECISIONS (curve area = trapezoid over coverage, per request; pooled = TRAIN + dev, 994 requests):
- D1 POLICY: V - A pooled +.026 [+.019, +.033] (TRAIN +.025, dev +.029) PASS; VS - A +.027 [+.020, +.034] PASS; VS - V
  +.0005 [-.0006, +.0016] -> **V** (VS costs +8..+20 mean judge calls for nothing measurable).
- D2 ORDER (under V): R - N -.002 [-.005, +.000]; E - N -.003 [-.007, -.000] (E is WORSE) -> **N, newest-first, stays**.
  Under the current policy A the ranking is reversed (E best, area -.032 vs N -.036): spreading facts over events only
  helps when no-facts photos are scored badly.
- D3 READINESS (N, V; lower 95% >= -.03 on both splits): 0% no (TRAIN lower -.040), 10% no (-.036), **25% PASS** (-.028 /
  -.027), 50 / 75 / 100% pass. 25% of 187k = est. 56-122 phone-hours with the 61c rate assumptions (unmeasured decode speed).

EYE CHECK (pre-registered; 4 requests per level drawn at seed 65 among those dropping > .10 vs 100% (0%: 99 such requests,
25%: 73); lost truth photos at 900 px, upscaled from the bench's 500 px originals; all 8 draws happened to be TRAIN):
- 0% (V): 14/14 lost tiles are real matches. Chihuly glass close-ups for "the glass museum visit" (4/4), jaguar mother and
  cubs for "the jaguars from the zoo in early December" (4/4), whisky sample bottles in front of a Christmas tree for "the
  Christmas whisky bottle photos" (4/4, F1 .667 -> 0), gift-opening on a sofa for "the Christmas photos from Regina"
  (2/2). Cause by eye: the subject is visible, but the request also needs context (a venue, a month, Christmas, a label)
  that the facts line carried (place + month prefix, "christmas tree", label text) and the image vector does not.
- 25% (N, V): 11/12 lost tiles are matches (1 doubtful truth: a city panorama for "the teal building"; a taco close-up for
  "only the ones with the big tables" is debatable). Every lost tile had NO facts at 25% - they are older photos (Mexico
  City restaurants 2013, Peachtree City kids' soccer 2011, Cuban teal facades 2012). The 4th request (apple press) lost
  nothing; it added 2 wrong photos.

READING
- The current fallback is the worst choice: towervec fed "no description" scores a no-facts photo LOWER than the image
  vector alone does in end-to-end F1 (0%: -.065 vs -.025 TRAIN). Likely mechanism, NOT measured here: towervec was
  trained only on photos with real lines, so a constant out-of-distribution text distorts its P for those photos (ranking
  and judge budget); vecB was trained for exactly the image-only input.
- Newest-first beats random and event-first under V, but part of that is the bench: 29% (TRAIN) / 31% (dev) of truth
  photos are in the newest 25% of their library (25% expected by chance; E's first 25% holds 22%). Under R the readiness
  level would be 75% (TRAIN lowers 25% -.034, 50% -.031, 75% -.025). So "acceptable from 25%" holds for requests that
  target recent photos; for old events it is the 0% row (-.025 TRAIN) until their facts exist. A real camera roll's
  question mix is UNKNOWN.
- A metadata-only text for no-facts photos (place + month prefix, which the phone has from day 1) was NOT tested; the 0%
  failures (venue, month) suggest it may help. Not pre-registered here: candidate for a later test.
DECISION (by the pre-registered rules): the phone scores photos without a facts line with the image-vector scorer (vecB),
in the same ranking and judge budget (V); the writer keeps going newest-first; the app shows a "still reading your photos
(N% done) - results may miss older photos" notice until 25% of the library has facts. Product wiring is spec only (the
app does not run the facts scorer yet; no Mac inbox item tonight). ~3 GPU-h (37 jobs, 39 configs each). Not pushed.

## 66. Search while facts are PARTIAL, part 2: a place + month line for photos without a writer line (P) is WORSE than the image-vector scorer (V); the logit blend P+V ties V - NOT ADOPTED, V stays (pre-registered 8d29a80; first result on the RESULTS-63 truth; 10-16; eval/partial66/{rescore63,prefix,cascade66,table66,sheets66}.py, job.sh, launch.sh -> eval/partial66/{base63,table}.txt, sheets/{lost_0_N_V,lost_25_N_V,vs_0_N_P}.txt)
Why: in RESULTS 65 the 0% losses were real matches whose request needed venue / month context that the writer line's
"place: ... taken: Month YYYY." prefix carried. The phone has place + capture month for every photo from day 1.
Truth: RESULTS 63 corrected dev/TRAIN adopted first (queries/{dev,train}.json = t63/queries, old files in queries_pre63/,
verified a,d,e,f,g,h; test closed, not read). Baselines re-scored on it (base63.txt, no new runs): all-judge TRAIN .6617 (641
requests) / dev .6836 (351); 61d TRAIN .6550, dev (TRAIN-writer xfit) .6900 = RESULTS 63's t63/rescore/new per request
(4/4 overlapping runs identical); dev product 61d = 65's 100% = .6886. 65's own conclusions hold on the new truth (N+V first
passes the bar at 25%: lowers -.028 / -.029).
Setup: 65's frozen stack and newest-first order; coverage 0/10/25/50% (100% = 65's 100_N_A, shared). Policies for photos
without a line: V = vecB (adopted in 65); P = a line made only of the photo's prefix (built exactly as the writer lines'
prefix: 70,580/70,580 lines start with it; every photo has a month, 69,134 a place), read by the same towervec + image
vector; PV = equal-weight logit average of P and V. Reproduction: 66's V = 65's V, identical returned sets 637-638/641
TRAIN and 348-350/351 dev per level, ALL F1 within .0003. 0 errored requests in 12 configs x 992 requests.

ALL F1 diff vs all-judge [95%] (judge calls / search mean, median: 63-64, 50 TRAIN and 77-80, 50 dev for every row):
| coverage | V (current spec) TRAIN / dev | P (prefix line) TRAIN / dev | PV (blend) TRAIN / dev |
|---|---|---|---|
| 0% | -.024 [-.040, -.009] / -.014 [-.034, +.007] | -.044 [-.061, -.026] / -.050 [-.075, -.026] | -.020 [-.035, -.004] / -.019 [-.040, +.002] |
| 10% | -.021 [-.036, -.006] / -.011 [-.031, +.010] | -.042 / -.040 | -.019 / -.016 |
| 25% | -.014 [-.028, +.000] / -.008 [-.029, +.013] | -.030 / -.027 | -.013 [-.027, +.001] / -.011 [-.031, +.010] |
| 50% | -.011 [-.025, +.002] / -.005 [-.025, +.015] | -.022 / -.022 | -.010 / -.012 |
| 100% (shared) | -.007 [-.021, +.008] / +.005 [-.013, +.024] | same | same |

PRE-REGISTERED DECISIONS (curve area, trapezoid 0/10/25/50/100, pooled 992 requests):
- D1: P - V pooled -.0125 [-.0182, -.0072] (TRAIN -.011, dev -.016) FAIL; PV - V -.0006 [-.0041, +.0028] (TRAIN +.001, dev
  -.004) FAIL; (PV - P +.012 [+.008, +.016]) -> **V stays**.
- D3 (V): 0% no (TRAIN lower -.040), 10% no (-.036), **25% PASS** (-.028 / -.029). P first passes only at 100%; PV too
  (dev 25% lower -.031, 50% -.034).
Where P loses (0%, P - V per category, TRAIN / dev): h follow-ups -.109 / -.119, a -.031 / -.051, d -.021 / -.029, g -.009 /
-.042; c +.012 / -.022. Mean recall at 0%: TRAIN .719 -> .687, dev .750 -> .706; precision flat (.684 -> .684 TRAIN).

EYE CHECK (pre-registered, adopted policy V; requests drawn seed 66 among those dropping > .10 vs 100%; lost truth tiles at
900 px, upscaled from 500 px):
- 0% (100 such requests; 12 tiles over 5 requests, 2 requests lost no truth = they added wrong photos): 10/12 real matches -
  tigers for "add the tigers back" (4/4, incl. a white tiger), Peachtree City kids' soccer May 2011 for the g "soccer match in
  Peachtree City that same year" (4/4), a yellow shop facade and a selfie against a yellow wall (2/2); 2 doubtful truth (a
  yellow boat railing and a bridge seen from that boat, "bright yellow wall"). Cause: subject visible; the request names a
  place / year (soccer) or a follow-up turn (tigers) that the image vector alone misses.
- 25% (73 requests; 13 tiles over 6): 11/13 real matches - resort pool in Punta Cana Feb 2012 "last year" (4/4), snow patches
  in Harrisburg Jan 2013 (3/3, small amounts of snow), feather headdress (1/1), boy with sled (1/1), beige cardigan (2/4; 2
  doubtful truth: a tan jacket, a cream top in a close-up - c is PROVISIONAL). 11/13 lost tiles had NO line at 25% (older
  photos), 2 had a line (cardigan).
- Not pre-registered (diagnostic, since P failed): 12 photos P returns at 0% that V does not (3 requests): tricycle race in the
  Mission District, 3 truth gains + 1 non-truth that IS a race photo (truth miss); "yaks on the trail": 4/4 trail / Nepal-trip
  photos with no yak; "google glass unboxing": 2 people wearing Glass (not unboxing), a phone case, a parade. The prefix
  pulls in the same trip / place without the subject.

READING
- Giving a no-facts photo only its place + month makes the towervec scorer WORSE than ignoring text (V): it loses recall
  across categories, most on follow-ups (h) and objects (a). Likely mechanism, NOT measured: towervec was trained on full
  lines, where place/month are a small part; a line that is ONLY place/month is out of distribution, and the place match
  dominates the score, so in-trip photos without the subject take the judge budget (the yak / Glass sheets show exactly
  that) while true matches drop below it. The context the 65 losses needed is real, but this route does not deliver it.
- The blend PV recovers V's level (pooled area -.0006) and helps a/c/e on TRAIN, but not on dev; no net gain.
- Untested alternatives (not run here, would need their own pre-registration): place/month as a metadata FILTER (the R59d
  trip route already does this for named places), or training vecB with a metadata input.
DECISION (by the pre-registered rules): no change. Photos without a facts line stay on vecB (V); newest-first; "still
reading your photos" notice until 25%. ~1 GPU-h (37 jobs x 12 configs). Not pushed.

## 67. Complex bench c (person + look) truths: WHY they are wrong (the oracle's cut 0.5 accepts looks that are not there; 115/137 programs pin unstated days/places), a pre-registered 27B rewrite at cut 0.5 / 0.9, a fresh blind eye audit - NEITHER version passes the bar (R5 P .53 / R .81, R9 .80 / .51); by the pre-written rule R5 is the candidate and "adopt" (F1 .644 vs .622) but the paired difference is noise (+.023 [-.133, +.166]); re-scored on it, shipped - all-judge on c drops to -.039 (pre-registered fec56c8; 10-16; eval/complex_bench/t67/{diag,diag_t63,regen,t67,paired}.py, job.sh, rescore.sh -> t67/{diag,build,report,paired}.txt, labels_main.txt, key.json, programs_c2/, queries/{dev,train}.json, rescore/*.txt)
Test split closed (no request, truth or image read). Dev + TRAIN c only (137 requests: dev 50, TRAIN 87).
DIAGNOSIS (t67/diag.txt; the RESULTS 63 c eye labels, already unblinded, used only here):
- 44 truth photos wrong by eye: 36 = the 27B oracle said yes at P .50-.89 to a literal question whose thing is NOT in the photo
  ("Is a person smoking?" .53-.80 on 15 photos with no cigarette; "white hat" .50-.71; "beige cardigan" .56-.59); 4 = a near-miss
  look at P >= .9 (red as pink x2, climbing helmet as hat, football jersey as jacket); 2 = named person, the look is on someone
  else in the photo (person questions say "the person", never "the person in the red box", while the photo shows several);
  2 = proxy question ("Elmo suit" -> "a person in a red costume"; "marshals in orange suits" -> any orange jumpsuit).
- By the lowest oracle P of the program's questions, labelled truth photos are right 1/24 at [.5, .7), 3/18 at [.7, .9), 9/15
  at >= .9. c is the only look category still compiled at cut 0.5 (b moved to 0.9 in RESULTS 56).
- 10 right photos outside the truth: 5 = an unstated day / date window / place pin, 4 = oracle no on a stated thing, 1 = a
  relational question ("Is the parrot on the woman's chest?" .11 on the right photo).
- Census of the 137 programs: 115 pin an unstated date or place (81 dates with no time word, 84 places never named; c was
  exempted from the strip in 10-12 because stripping then added mostly wrong photos - the pins were hiding the cut's leniency),
  46 carry a question with no content word of the request, 44 split one described person's look over separate whole-photo
  questions (binding lost), 19 are Sam/Alex albums (named-person A11 units in RESULTS 63: 7/15, 5 of them wrong).
FIX (pre-registered): the 27B rewrote every c program with the words fixed (REGEN_C, 11 rules: one bound question per person,
"the person in the red box" enforced, the owner's own look words, no added condition, no unstated date/place, a named occasion
= one direct question + its day, me/us = people in the photo, "real", well_defined flag); oracle on the new questions only
(~200k answers, 8 shards, ~25 min); compiled at cut 0.5 (R5) and 0.9 (R9). Drops: R5 3 (all "empty album"), R9 13; the
rewriter flagged no request ill-defined. Truth units dev T 1,403 -> R5 2,956 / R9 987; TRAIN 1,697 -> 2,743 / 1,393.
VERIFICATION: 144 fresh units (seed 67; none from RESULTS 63), strata = membership pattern over T / T9 (old program at cut .9) /
R5 / R9 plus out-of-all by the RESULTS 63 direct 27B P; labelled blind by me at native pixels (DISBench <= 500 px; crops
enlarged), Sam/Alex units with 3 reference crops; labels committed (4b5424d) before key.json was read; 54 match / 84 no_match / 6 unsure.
| version (requests) | precision [95%] | recall [95%] | F1 | named-person in-version right | other in-version right |
|---|---|---|---|---|---|
| T current truth (137) | .69 [.54, .82] | .57 [.40, .81] | .623 | 8/11 | 34/56 |
| T9 = T at cut .9 (137) | .83 [.63, .98] | .44 [.30, .63] | .572 | 3/4 | 23/30 |
| R5 rewrite, cut .5 (134) | .53 [.41, .66] | .81 [.58, 1.00] | .644 | 10/12 | 34/57 |
| R9 rewrite, cut .9 (124) | .80 [.62, .94] | .52 [.36, .75] | .630 | 8/10 | 19/25 |
Paired (same labels, 134 requests): R5 - T precision -.154 [-.274, -.022], recall +.246 [+.109, +.397], F1 +.023 [-.133, +.166];
R9 - T F1 -.002 [-.095, +.087]; T9 - T F1 -.050 [-.122, +.030] (t67/paired.txt; reported, not part of the rule).
DECISION (pre-written rule): candidate = R5 (F1 .644 vs R9 .630); VERIFIED no (P .53 < .80); rule outcome "recommend adopt"
(.644 > .622 on the same requests). My reading: the F1 gain is not distinguishable from zero; R5 buys recall with precision.
t67/queries/{dev,train}.json = queries/ with c replaced by R5 (dev 50 rewritten, TRAIN 84 rewritten + 3 dropped; c verified=False).
EFFECT ON THE SHIPPED STACK (existing runs re-scored: c61d_*_t2b_final_rs2_xfit/r59d and the all-judge references; final-round F1):
| split | shipped c: current -> t67 | all-judge c: current -> t67 | shipped - all-judge on c: current / t67 | ALL shipped - all-judge: current / t67 |
|---|---|---|---|---|
| dev (50 c) | .602 -> .604 (+.002 [-.072, +.078], +13/-18) | .594 -> .643 (+.049 [-.026, +.125]) | +.008 [-.017, +.033] / -.039 [-.065, -.015] | +.006 / -.000 [-.017, +.017] |
| TRAIN (87 -> 84 c) | .682 -> .670 (-.012 [-.066, +.039]) | .684 -> .707 (+.022) | -.002 [-.028, +.020] / -.037 [-.062, -.016] | -.007 / -.011 [-.026, +.002] |
On R5 the shipped stack's c gap to the all-judge reference becomes significant on both splits (-.04): R5's added recall is
photos the all-judge stack finds and the cascade does not. ALL-request comparisons stay inside their CIs.
READING. (1) The c truth error is mostly the TRUTH ORACLE, not the program text: at cut 0.5 the 27B accepts absent looks on
36/44 wrong photos, and after the rewrite (bound questions, no pins) the same leniency remains (R5 wrong photos by eye: pink
backpack / smoking / magenta coat / conical hat at P .5-.9). (2) The hidden day/place pins were masking that leniency; removing
them (R5) raises recall .57 -> .81 but floods in loose yeses. (3) At cut 0.9 precision reaches the bar (.80-.83) but recall falls
to .44-.52: the 27B's P is not separable enough for looks at any single cut - the 0.5/0.9 trade-off is the measured frontier
here. (4) The rewrite's "named occasion = one direct question" lost right photos (jousting knights, the Lombok hike, the Star
Wars party: 6/10 P1100 units right). c cannot reach VERIFIED with a yes/no 27B oracle + one cut; options for the coordinator:
keep T (c stays PROVISIONAL, current comparisons stand), adopt R5 per the rule (comparisons on c move by -.04), or treat c
as eye-labelled-only (score c on audited photos). CAVEATS: one labeller (me), one pass, 6 unsure; 144 units over 12 strata
(10-12 per stratum: wide CIs, recall CI widths ~.4); 18 "borderline" notes; the out-of-all strata hold 222k units with 1/48
matches (O1 1/12 drives most of every version's missed-photo estimate); named-person universe = the bench's face match
(identity of 1 unit not checkable); test never touched. ~3.5 GPU-h (8 x H100/RTX for regen + oracle).

## 68. The index-time-facts search path PORTED to FindPicsCore with golden parity: the R59d cascade + policy V as pure logic (0 mismatches on 2,976 recorded dev + TRAIN request-configs), the towervec head and vecB in pure Swift (max |P diff| 2.6e-6 vs PyTorch fp32 on 75,142 photo-questions), the towervec text encoder exported for MLX (real-MLX cosine >= .999996) and its WordPiece tokenizer in Swift (0/1,418) - app NOT wired (Mac) (pre-registered 581250b; 10-16; eval/cascade_port/{INVENTORY.md, record65.py, job.sh, launch.sh, fixture.py, heads.py, encoder.py, encoder_mlx.py} -> encoder_mlx_*.txt, fixture.log; FindPicsCore CascadeRoute.swift, FactsScorer.swift, BertTokenizer.swift + tests)
Why: the phone app has none of the search path RESULTS 59d / 61c / 61d / 65 adopted (millisecond scorer over index-time facts,
judge only for pixel kinds under a 5% budget, trip metadata, OR/AND parts, extension, judged exclusions, vecB for photos without
a line). No new measurement of search quality: this is a port, judged only by parity. Test split closed; no Mac inbox item.

(1) INVENTORY (eval/cascade_port/INVENTORY.md): pure logic = facts prefix (place parts + UTC month), WordPiece, scorer question,
question tree (QuestionTree.swift, existing), leaf route kinds + trip-no-place + exclude, policy V, budget max(50, ceil 5%),
extension blocks (>= 50% judge P >= .7), threshold, reference-photo calls, pass-A scope bookkeeping. Learned: towervec head
(LN + MLP over [f, q, f*q, 32v, 32t, 32v*32t], 2,230,529 params), vecB (same MLP over the image half, 1,639,169), towervec's
fine-tuned MiniLM-L12 encoder (33,360,000; mean pool incl. CLS/SEP, unnormalised, 384-d), PE-Core-B-16 towers (existing),
the 2B writer (RESULTS 61d MLX).

(2) PURE LOGIC: CascadeSession (recordScope = harness pass A, answer = pass B). Golden = the measured harness itself:
record65.py = cascade65.py + logging (configs 0_N_V, 25_N_V, 100_N_A; 37 GPU shards). Recorder reproduction: returned sets
identical to RESULTS 65's runs 2,955/2,976 (the 21 others = run-to-run bf16 GPU nondeterminism, same level as RESULTS 66's
rerun 637-638/641 + 348-350/351). Swift vs recording, pre-registered bar 0 mismatches:
| set | (request, config) | pass-B calls | photo answers bit-equal | judged sets | extension blocks | pass-A scopes |
|---|---|---|---|---|---|---|
| dev | 1,053 | 4,073 | 1,353,893 / 1,353,893 | 1,053 / 1,053 (83,721 pairs) | 0 mismatches | 0 mismatches |
| TRAIN | 1,923 | 7,654 | 2,595,133 / 2,595,133 | 1,923 / 1,923 (123,667 pairs) | 0 | 0 |
BAR MET. Covered: 262 subset calls, 177 exclude calls, 475 requests with a budget extension; NOT covered: reference-photo
calls (0 in the benchmark runs; ported from the code, untested). Bundle fixture (always run): 29 stratified request-configs,
3.6 MB, 0 mismatches; full set with FP_CASCADE_FULL=data/public/cascade_port/full (421 MB, not in git).
Scope finding for the Mac: of 1,984 judge leaves, pass B asked exactly pass A's scope in 1,832, a strict subset in 39, and
photos outside it in 113 (the threshold still applies to them); 0 leaves lacked a pass-A scope. The harness's pass A answers
pixel-kind questions with the judge on EVERY photo (cached in the benchmark): the app cannot afford that, so it needs its own
scope source (e.g. a scorer-only dry run of the plan, or the album's candidate set) - a design decision, NOT measured here.
Facts prefix: 1,506/1,506 cases equal (factsPrefix).

(3) SMALL MODELS (.fphead = header JSON + float32; towervec_head 8.9 MB, vecB 6.6 MB, eval/cascade_port/models/, not in git).
ScorerHead (reference forward) + FoldedScorerHead (per-question folded first layer, batch path = vDSP_mmul on Apple, loops
elsewhere). Real heads on 2 dev libraries, inputs computed by PyTorch fp32 on CPU (f, q from the encoder; v stored; t PE text
fp32), 3,705 photos, 70 questions, 75,142 photo-questions: max |P diff| towervec 1.4e-6, vecB 2.6e-6 (bar <= 1e-5: MET).
Downstream, the recorded requests of those 2 libraries (165 request-configs, 243,554 answers) replayed with each score source:
| Swift fp32 vs | answers not bit-equal | other side of P = .7 | judged sets differ | judge answers the run needs that the recording lacks |
|---|---|---|---|---|
| PyTorch fp32 | 199,606 (max 2.4e-6) | 0 | 0 / 165 | 32 (Swift-score run; recorded-score run 0) |
| recorded bf16-GPU scores (what every measured number used) | 219,185 in 666 calls | 29 | 55 / 165 | 32 |
The pre-registered wording "0 differences in pass-B outputs" FAILS as written and was a mis-specified bar: scorer-routed
answers ARE the scores, so two fp32 implementations can never be bit-equal there; at the decision layer (judged sets, side of
.7) Swift = PyTorch fp32 0/165 and 0/243,554. The real finding is fp32 vs bf16: the phone's fp32 changes which photos the judge
sees in 55/165 request-configs (ties/threshold edges; 32 photo-questions would need judge calls the recording never made) and
flips 29 answers across .7. The F1 effect of that is NOT measured (would need a cascade rerun in fp32 with the 9B).
Not measured: fp16 STORAGE of the facts vector (768 B/photo plan) and fp16 PE text vectors on the phone.

(4) TEXT ENCODER: eval/cascade_port/models/towervec_encoder (HF BERT dir, fp32 safetensors 134 MB) and towervec_encoder_fp16
(67 MB), + vocab/tokenizer, modules.json, 1_Pooling (mean). Checks vs the original PyTorch fp32 vectors (1,000 seeded dev facts
lines, 400 recorded scorer questions): HF reload fp32 cosine 1.000000 (max |diff| 1.1e-6); HF fp16 weights computed fp32
.999999; REAL MLX 0.32.3 (CPU, a Python mirror of mlx-swift-lm MLXEmbedders Bert.swift with its key sanitize, weights loaded
strict): fp32 facts min 1.000000 / questions 1.000000, fp16 weights + fp16 compute facts min .999996 / questions .999997 (bar
>= .999: MET). One trap, measured: MLXEmbedders adds the token-type embedding ONLY if tokenTypeIds is passed; without it
cosine falls to min .720 / mean .926 (200 lines) - the app MUST pass zeros. BertTokenizer.swift = the HF fast tokenizer on
1,418 texts (1,000 facts lines, 400 questions, 18 hand cases: accents, CJK, emoji, control chars, full-width, ligatures,
120-char words), 106,673 tokens: 0 differ. Core ML conversion not attempted (MLX path verified instead).

WHAT THE MAC STILL HAS TO DO (spec, no inbox item tonight): load towervec_encoder(_fp16) with MLXEmbedders BertModel, run
it on BertTokenizer.factsInput(prefix + writer line) at index time (store f, 384 floats) and questionInput(scorerQuestion) per
search, tokenTypeIds = zeros, mean pool, no normalise; ship the two .fphead files; build a CascadeSession per search (towervec
via FoldedScorerHead for photos with a line, vecB without) around the existing judge; give it a leaf scope (recordScope)
before answering; wire PlaceIndex per photo set. Not compiled with Apple frameworks here (vDSP path untested).
Tests: swift test 162 XCTest + 29 swift-testing, 0 failures (env-gated full sets run separately, all pass); pytest 179/179. ~2 GPU-h (recording) + CPU jobs. Not pushed.
