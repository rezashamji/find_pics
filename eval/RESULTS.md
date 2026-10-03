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
