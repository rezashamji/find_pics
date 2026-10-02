# 04: Recall estimation and platform access

Research date: 2026-10-02. Scope: (A) how find_pics can honestly report how many items it examined and a statistically valid lower bound on recall; (B) read-only access to Apple, Google and Android libraries, and on-device Mac throughput.

**Tags used on every claim**
- `[DOCUMENTED: Sx]`: stated in source Sx (see the Sources list at the end).
- `[INFERENCE]`: my derivation or reasoning from documented facts. Where arithmetic matters, it is shown.
- `UNVERIFIED`: I could not confirm it.

**How the numbers were produced**
- Every numeric bound in A4 comes from exact binomial and hypergeometric computations I ran, with Monte Carlo over 200–300 replications.
- I checked my QBCB implementation against Table 1 of Lewis et al. (S1): it reproduces r = 14, 22, 30 and 37 → j = 14, 21, 28 and 34.

**How Part B was gathered**
- Part B facts came from parallel research passes that fetched primary pages.
- I re-checked these directly: the icloudpd release and its Advanced Data Protection (ADP) requirement, the osxphotos release, and the Google scope removal.

---

## 0. The mechanism first: why recall is hard to certify

Notation (used throughout):
- N: library size.
- R: the retrieved set, i.e. what we show.
- U: everything else ("unretrieved pool"). N_U = |U|.
- T_R: number of true matches inside R.
- M: number of true matches hiding in U ("misses").

Recall:

$$\rho = \frac{T_R}{T_R + M}$$

**What is countable and what is not.** T_R is countable, because someone can look at every item in R. M is not, because nobody looks at all of U. So any honest recall claim reduces to one question: **how confidently can we say M is small?**

**What we report.** We report a one-sided lower confidence bound (LCB). An LCB is a number ρ_L computed from a random sample. Over the randomness of the sampling, P(ρ ≥ ρ_L) ≥ 1 − α. "Recall ≥ 0.92 with 95% confidence" means ρ_L = 0.92 with α = 0.05.

**Why low prevalence makes this expensive** (prevalence = the fraction of a pool that are true matches).

Concrete example:
- 300 bread photos in 150,000 photos, and suppose we already found 285.
- The 15 misses are spread over about 149,700 photos, a rate of 1 in 10,000.
- A random sample of 1,000 photos from U will almost surely contain zero bread photos.
- Zero hits in 1,000 does not mean "no misses". It only proves the miss rate is below about 3/1,000, which is about 450 photos. That is useless.

**The zero-hit law** ("rule of three"). If you inspect n random items from a pool of size N_U and find 0 matches, the exact one-sided 95% upper bound on the number of matches in the pool is:

$$M^U \approx N_U \cdot \frac{\ln(1/\alpha)}{n} = N_U\cdot\frac{3.0}{n}$$

[DOCUMENTED: S26 for the Clopper–Pearson definition; the multipliers below are my exact computation]

If you see k > 0 hits, the multiplier grows. These are the exact Clopper–Pearson upper bounds × n at α = 0.05 (computed):

| hits k | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| multiplier | 3.00 | 4.74 | 6.29 | 7.75 | 9.15 | 10.51 | 11.84 |

At α = 0.05/3 (Bonferroni over three strata, explained in A4), the multipliers are 4.09, 6.05, 7.75 and 9.34 for k = 0..3.

**So certifying at most m remaining misses in a pool of N_U items costs about N_U·ln(1/α)/m labels** [INFERENCE from the formula above]:
- Bread case: certifying recall ≥ 0.92 means m ≤ about 24, so n ≈ 149,700 × 3 / 24 ≈ 18,700 labels.
- This is not a weakness of one method. A 2026 paper proves a matching lower bound: any valid audit certifying fewer than m missed items "must inspect on the order of N₀/m excluded-pool labels". It also proves that "no procedure using only labels from inside the candidate set can certify any non-trivial bound on the missed mass: the audit must sample the excluded pool." [DOCUMENTED: S9]
- Consequence for find_pics [INFERENCE]: checking only the results the user sees can never certify recall. Some labeler must look at a sample of what was not shown, and the number of looks scales with (size of the unshown pool) / (number of misses you're willing to tolerate).

The rest of Part A is about three things: (1) which published procedures turn that sample into a valid bound, (2) how to spend labels where misses actually are (stratification), and (3) how to use a VLM judge as the cheap labeler without lying.

---

## Part A: Recall estimation

### A1. E-discovery / TAR recall certification

Vocabulary:
- **TAR** (technology-assisted review): a human-in-the-loop workflow where a classifier ranks documents and people review the top of the ranking, used in legal e-discovery and systematic reviews.
- **Stopping rule**: the decision of when to stop reviewing.
- **Certification rule**: a stopping rule that comes with a statistical guarantee on recall.
- **Heuristic rule**: a stopping rule without such a guarantee. [DOCUMENTED: S1 §2]
- **One-phase TAR**: train-and-review interleaved, which is our situation: retrieve, look, refine. **Two-phase TAR**: train a classifier, then review a cutoff. [DOCUMENTED: S1 §1]
- **Standoff rule**: a rule that works regardless of how documents were selected. **Interventional rule**: a rule that changes the selection process. [DOCUMENTED: S1 §2]

#### A1.1 Elusion test (sampling the discard pile)

Elusion means the fraction of the discard pile (U) that is relevant [DOCUMENTED: S1 §3; S7].

Mechanism:
1. Draw a simple random sample of n items from U.
2. Have them labeled and find k positives.
3. Compute

$$\hat e = \frac{k}{n},\qquad \hat M = \hat e\, N_U,\qquad \hat\rho = \frac{T_R}{T_R + \hat e N_U}$$

The e-discovery recall formula is "total number of relevant documents found / the Eluded documents + the total number of relevant documents found" [DOCUMENTED: S7].

**Valid one-sided version.** Replace ê with its exact upper bound:
- Binomial: e^U = Clopper–Pearson upper.
- Hypergeometric (exact for a finite pool): M^U = max{K : P_HG(X ≤ k; N_U, K, n) > α}.

Then:

$$\rho_L = \frac{T_R}{T_R + M^U}$$

This is a valid (1 − α) LCB under three conditions [INFERENCE]:
- (i) T_R is exact, meaning every item in R was verified.
- (ii) The sample was drawn once, with n fixed in advance.
- (iii) The labels are correct.

**How it fails in practice:**
- **Low prevalence produces zero hits, which produces "near-perfect recall" claims.** In large low-prevalence collections, a sample sized for ±2.5–3% margin "is unlikely to yield any positive examples ... Therefore, the Elusion test predicts almost perfect recall ... when that may not be the case." [DOCUMENTED: S7]
- **Measured overestimates.** In an Irish litigation study, a vendor tool's elusion-test recall was compared against a blind "confusion test" (independent re-review of stratified samples of both retrieved and unretrieved sets). Elusion overestimated recall by 36, 29, 20 and 25 percentage points on four requests. By the confusion test, the tool never reached 75% recall. [DOCUMENTED: S7]
  - Example from Table V, request 9: the original assessors' elusion test said recall 0.91; the confusion test said 0.55; an independent assessor re-reviewing the *same* elusion sample got 0.40. [DOCUMENTED: S7 Table V]
- **Assuming R is perfectly coded.** Elusion "presupposes all documents marked as relevant on the first pass are accurately coded". Mislabeled positives in R inflate T_R. [DOCUMENTED: S7]
- **Reviewer bias.** The same reviewers often do the elusion review knowing the items are "supposed to be" non-responsive, which suppresses found positives. [DOCUMENTED: S7]
- **Sequential bias (re-testing until it passes).** "Repeated PET" practice: stop, sample, and if the test fails, continue and re-test. Lewis et al.: "all RPET approaches suffer from sequential bias induced by multiple testing: the process is more likely to stop when sampling fluctuation gives an over-optimistic estimate." [DOCUMENTED: S1 §6.1]
- A typical e-discovery elusion sample is "2,395, targeting a margin of error of 2.5% at a confidence level of 95%" [DOCUMENTED: S7]. The normal-approximation formula gives n ≈ 1.96²·0.25/0.02² ≈ 2,401 for ±2%, so the quoted "2.5%" looks inconsistent [INFERENCE].

Lessons for find_pics [INFERENCE from S7]:
- Label audit items blind: mix R-items and U-items so the labeler can't tell provenance.
- Verify R as well as U.
- Never re-sample until a test passes.

#### A1.2 PET, QPET, QBCB: the "quantile method" (Lewis, Yang & Frieder, CIKM 2021)

**Setup.**
- Draw a random sample of the whole collection before (or independently of) the review.
- The positives in that sample are D, with r of them. They are a simple random sample of all positives.
- Each positive has an "A-rank": the position at which the review/ranking finds it.
- Recall at a stopping point equals the fraction of the positive population found by then. So certifying recall ≥ t means certifying that the stopping point is at or past the t-quantile of the positives' A-ranks. [DOCUMENTED: S1 §3]

**PET rule** (stop when the sample-based plug-in recall j/r ≥ t): **invalid, biased low.** Example from the paper: with every document relevant and goal 0.5, the expected recall at the PET stopping point is (1/2)(n/(n+1))((N+1)/N) < 0.5 [DOCUMENTED: S1 §3.2].

**QPET**: a quantile point estimate (Hyndman–Fan Q7). It is a point estimate, not a bound [DOCUMENTED: S1 §3.3].

**QBCB** (Quantile Binomial Confidence Bound) **gives a valid one-sided 1 − α guarantee.** Stop when the j-th sample positive is found, where j is the smallest integer with

$$\sum_{k=0}^{j-1}\binom{r}{k}t^{k}(1-t)^{r-k} \;\ge\; 1-\alpha$$

[DOCUMENTED: S1 eq. 3]
- This is the binomial approximation to a hypergeometric expression (eq. 2).
- "For values of t greater than 0.5 ... the j chosen using the binomial approximation will never be less than the one chosen using the hypergeometric." In other words, it is conservative. [DOCUMENTED: S1 §4.1]

Table 1 of S1 (t = 0.80, 95%): r = 14 → j = 14; 22 → 21; 30 → 28; 50 → 45; 158 → 135; 457 → 380 [DOCUMENTED: S1 Table 1]. My computation for other targets, at 95%, gives the minimum number of sample positives r for a given number of tolerated sample misses (r − j):

| recall target t | all sample positives found | 1 may be missed | 2 missed | 3 missed |
|---|---|---|---|---|
| 0.80 | r = 14 | r = 22 | r = 30 | r = 37 |
| 0.90 | 29 | 46 | 61 | 76 |
| 0.92 | **36** | 58 | 77 | 95 |
| 0.95 | 59 | 93 | 124 | 153 |

**The low-prevalence cost of QBCB** [INFERENCE]: r is the number of positives in a random sample, so you must randomly sample about r / prevalence items.
- "Photos of the user" (2,000/150,000 = 1.33%): r = 36 needs about 2,700 random labels.
- Bread (300/150,000 = 0.2%): r = 36 needs about 18,000 random labels.
- That matches the zero-hit law in §0. The two different methods hit the same about-N/m wall, which is what S9's lower bound predicts.

**Target rule** (Cormack & Grossman): sample until 10 positives are found (the "target set"), then review until all 10 are found.
- It is valid and "a special case of the QBCB rule" [DOCUMENTED: S1 §6.3].
- 10 targets certify recall ≈ 0.74 at 95%. Nine suffice for 0.70 [DOCUMENTED: S1 §6.3].
- "Requiring that every positive sample document be found also means a single coding error would have large consequences." [DOCUMENTED: S1 §6.3]

**Practical finding: a larger certification sample lowers total cost.** "incurring a larger sampling cost to reduce excess recall leads to lower total cost in almost all scenarios" [DOCUMENTED: S1 abstract].
- Mechanism: with a tiny sample, you must overshoot the recall goal to be confident, and the last few percent of recall are the most expensive to review [DOCUMENTED: S1 §5].
- Relevance to find_pics [INFERENCE]: QBCB tells you how deep down the ranked list to show results to certify recall. That makes it a principled way to choose the result-list cutoff.

**Countdown / "indirect method" (BIR)**: estimate total positives from a pre-sample and stop when that many are found. It is "seriously flawed": it can give recall > 1 and ignores sampling variation [DOCUMENTED: S1 §6.2]. Callaghan & Müller-Hansen show BIR-based stopping achieves ≥95% recall with work savings in "only 23% of cases" [DOCUMENTED: S4].

#### A1.3 Knee and Budget methods (Cormack & Grossman, SIGIR 2016): heuristics, no guarantee

The knee method looks at the gain curve (cumulative positives found vs. items reviewed) and stops when the slope before a "knee" is much steeper than the slope after it.

TARexp implementation, per review round i out of n rounds, where rel(i) = positives found by round i [DOCUMENTED: S3, code]:

$$\text{slope ratio}(i) = \frac{rel(i)/i}{\big(1 + rel(n) - rel(i)\big)/(n-i)},\qquad \text{stop if } \max_i \text{ratio} \ge 156 - \min(rel(n), 150)$$

- **Budget** adds: (ratio ≥ 6 and enough documents reviewed relative to 10·N/rel), or 75% of the collection reviewed [DOCUMENTED: S3].
- These rules give no confidence statement. Cormack & Grossman frame quality as "the probability of achieving the threshold" and describe augmenting TAR "to achieve guaranteed reliability, for a quantifiable level of additional review effort" [DOCUMENTED: S6 abstract]. The knee rule's thresholds are empirical constants [INFERENCE from S3 code].
- **Role for find_pics** [INFERENCE]: a decent signal for "when to stop running the expensive VLM down the ranked list". Never use it as the reported recall claim.

#### A1.4 Quant and QuantCI (Yang, Lewis & Frieder, DocEng 2021): model-based heuristics

Use the classifier's calibrated probabilities p_i:

$$\hat\rho = \frac{\sum_{i\in\text{reviewed}} p_i}{\sum_{i} p_i}$$

Variance (TARexp implementation), with K = Σ_{reviewed} p_i and V = Σ_{unreviewed} p_i [DOCUMENTED: S3 code; S2]:

$$\widehat{\mathrm{Var}} = \frac{K^2}{(K+V)^4}\sum_i p_i(1-p_i) + \frac{1}{(K+V)^2}\sum_{i\in \text{rev}} p_i(1-p_i)$$

- QuantCI stops when ρ̂ − c·SD ≥ target.
- The paper presents both as heuristics that are "accurate at hitting a range of recall targets" [DOCUMENTED: S2].
- **Not a certificate**: they are only as good as the calibration of p_i in the unreviewed region, which is exactly where the model is least trustworthy [INFERENCE].
- **Relevance** [INFERENCE]: a SigLIP or face-cluster score turned into a probability (e.g., by isotonic regression on labeled audit items) gives a cheap "expected misses" estimate, Σ_{U} p_i. Report it as a model estimate, never as the guarantee.

#### A1.5 Callaghan & Müller-Hansen 2020: hypergeometric stopping test

The paper sets a target recall τ_tar and a confidence level α, and tests [DOCUMENTED: S4]:

$$H_0:\ \rho < \tau_{tar}$$

Variables:
- After ML-prioritised screening, ρ_AL relevant documents have been found.
- Then n further documents are drawn at random from the N remaining, and k relevant are found.
- ρ_seen = ρ_AL + k.

The least-favourable number of relevant documents remaining under H₀ is:

$$K_{tar} = \Big\lfloor \frac{\rho_{seen}}{\tau_{tar}} - \rho_{AL} + 1 \Big\rfloor,\qquad p = P(X\le k),\ X\sim \mathrm{Hypergeom}(N, K_{tar}, n)$$

Stop, i.e. reject H₀, if p < α [DOCUMENTED: S4]. "This gives the maximum probability of observing k for all values of K compatible with our null hypothesis" [DOCUMENTED: S4].

- **Validity**: exact when the n documents are a genuine random sample from the remainder. That is the paper's "random sampling" variant. [DOCUMENTED: S4]
- **"Ranked quasi-sampling" variant**: treats already-screened, ML-prioritised batches as if random. The authors acknowledge these are "not in fact a random sample" and argue it is conservative if prioritised documents are at least as likely relevant as random ones. [DOCUMENTED: S4] This is an assumption, not a guarantee [INFERENCE].
- **Results**: both criteria "achieve the target threshold of 95% in more than 95% of cases". Work savings were about 17% (quasi-sampling) and 15% (random sampling) [DOCUMENTED: S4].
- **Worked check in our setting** [INFERENCE, computed]: bread scenario (A4) after the judge census of the top strata. 296 positives found; tail N = 147,600; random n = 20,000.
  - k = 0: K_tar = 26, p = 0.023, so recall ≥ 0.92 is certified at 95%.
  - k = 1: p = 0.103, so it is not certified.
- **Implementations**: `buscarpy` (Python, v0.0.2) and `buscarR` [DOCUMENTED: S5]. TARexp has a `CHMHeuristicsStoppingRule` that applies the hypergeometric test to past batches (the quasi-sampling form) [DOCUMENTED: S3 code].

#### A1.6 Webber 2013: recall confidence intervals with stratified sampling

Recall estimated from samples of both the retrieved and the unretrieved segments.

Stratified yields [DOCUMENTED: S8 eqs. 4–5]:

$$\hat R_s = N_s\, r_s/n_s,\qquad \hat R_T = \sum_{s \in T}\hat R_s$$

The recall estimator 1/(1 + R̂₀/R̂₁) is a function of a ratio, so it is biased, "serious for extreme prevalences and small sample sizes" [DOCUMENTED: S8 §2.1].

Methods compared:
- **Normal/Wald**: "poor coverage in many circumstances" [DOCUMENTED: S8].
- **Koopman ratio-of-binomials**: better, but "inaccurate on small populations" [DOCUMENTED: S8].
- **Recommended**: beta-binomial posteriors on retrieved and unretrieved yield with half priors (α = β = 0.5), and a Monte Carlo distribution of recall. It is "accurate in all conditions examined" [DOCUMENTED: S8 §3.6, Table].
  - This is approximately calibrated (Bayesian with good frequentist coverage in their simulations), not a finite-sample guarantee [INFERENCE].

**Allocation advice**: "low precision and high recall causes wide intervals, and favors an allocation of samples towards the unretrieved segment. A 20%:80% sample allocation to the retrieved and unretrieved segments gives reasonably close to minimal interval width across the scenarios" [DOCUMENTED: S8 §4.3].

**Why stratification works**: "Estimate accuracy is improved by stratification if prevalence differs between strata, and further still if more samples are allocated to strata that have mediate prevalence" [DOCUMENTED: S8 §2.2].

#### A1.7 Newer work (2024–2026)

- **Anthony & Salehzadeh Nobari (arXiv, 2026-07-23), "Finite-Sample Coverage Audits for High-Recall Candidate Generation"** [DOCUMENTED: S9]:
  - Uses binomial/hypergeometric inversion.
  - Converts a missed-mass bound into recall via a "two-pool design".
  - Proves the about-N₀/m lower bound.
  - Its validity discipline: "the candidate generator ... and the audit rule are fixed before the certification labels are examined."
  - It is the closest published match to the find_pics problem: a candidate stage whose misses are lost downstream.
- **Fletcher & Stevenson (arXiv, 2026-06)**: two new heuristic stopping methods that monitor decision sufficiency rather than recall. The abstract states no recall guarantee [DOCUMENTED: S10].
- **Li & Kanoulas, TOIS 2020**: "Sampling from an Adaptive Distribution to Estimate Residual Relevant Documents" [DOCUMENTED: S11, citation via S7]. This is importance-sampling-based estimation of misses. UNVERIFIED: its exact estimator and CI construction (paper not fetched).

#### A1.8 Validity summary

| Method | Valid one-sided recall bound? | Conditions | Labels needed at 0.2% prevalence for recall ≥ 0.92 @95% |
|---|---|---|---|
| PET (plug-in, stop when estimate ≥ t) | No, biased [S1] | n/a | n/a |
| Repeated elusion until pass (RPET) | No, sequential bias [S1] | n/a | n/a |
| Elusion test, once, CP/hypergeometric upper bound | Yes [INFERENCE] | fixed n, blind labels, R fully verified | about 18,700 random U labels with 0 hits (§0) |
| QBCB / Target | Yes [S1] | sample of positives drawn at random from the whole collection | about 36/0.002 ≈ 18,000 random labels |
| Callaghan–Müller-Hansen, random-sampling variant | Yes [S4] | truly random post-stop sample | about 20,000 if tail is clean (A1.5 check) |
| Callaghan–Müller-Hansen, quasi-sampling variant | Only under a ranking assumption [S4] | prioritised ≥ random | n/a |
| Knee / Budget | No, heuristic [S3, S6] | n/a | n/a |
| Quant / QuantCI | No, model-based heuristic [S2] | calibrated scores | n/a |
| Webber beta-binomial | Approximate (good coverage in simulation) [S8] | stratified samples | similar order |
| Stratified exact bounds (A4) | Yes [INFERENCE] | per-stratum exact bounds plus a union bound | concentrates labels; the tail still costs about N_tail·ln(1/α)/m |

---

### A2. Stratified and importance sampling when prevalence is very low

**Stratification** (split U into bands by model score, then sample each band separately).

Estimator for misses and its variance, without-replacement within strata (standard survey sampling; S8 §2.2 for the yield form):

$$\hat M = \sum_h N_h\frac{k_h}{n_h},\qquad \widehat{\mathrm{Var}}(\hat M) = \sum_h N_h^2\Big(1-\frac{n_h}{N_h}\Big)\frac{\hat p_h(1-\hat p_h)}{n_h-1}$$

**Neyman allocation** (the variance-minimising split of a fixed budget n): n_h ∝ N_h·√(p_h(1−p_h)), using prior guesses of p_h from calibrated scores [INFERENCE: textbook result; consistent with S8 "allocate to mediate prevalence"].

**Why misses concentrate near the cutoff** [INFERENCE]. Retrieval scores are monotone-ish in match probability, so the band just below the cutoff (ranks |R|+1 … 2|R|) has prevalence 10–100× the deep tail. Spending labels there finds misses, which also lets you add them to the results. Spending the same labels in the deep tail mostly certifies emptiness.

**Census strata need no α.** If a stratum is fully labeled (a census), its count is exact, so no confidence budget is spent on it [INFERENCE]. This is the main design lever in A4.

**Horvitz–Thompson (HT)**: weight each sampled item by 1/(its inclusion probability). With item i included with probability π_i:

$$\hat M_{HT} = \sum_{i\in s}\frac{y_i}{\pi_i},\qquad \widehat{\mathrm{Var}}_{\text{Poisson}} = \sum_{i\in s}(1-\pi_i)\frac{y_i^2}{\pi_i^2}$$

The variance form is for Poisson sampling, where each item is included independently [INFERENCE: standard survey-sampling result].

- Choose π_i ∝ (calibrated score)^γ, e.g. γ = 1/2. Why 1/2: with no judge prediction (f = 0), the active-inference optimum in A3.3, π ∝ √E[(Y−f)²|x], becomes √p(x) for binary Y [INFERENCE from S17].
- Mix with uniform ("defensive" mixture) so no weight explodes: π_i = (1−τ)π_i^{score} + τ·n/N_U [INFERENCE].

**Active testing (Kossen, Farquhar, Gal & Rainforth, ICML 2021).** Pick test points with an acquisition distribution q, then de-bias with the LURE estimator [DOCUMENTED: S12 eqs. 3–4]:

$$\hat R_{LURE} = \frac{1}{M}\sum_{m=1}^{M} v_m L(f(x_{i_m}), y_{i_m}),\qquad v_m = 1 + \frac{N-M}{N-m}\Big(\frac{1}{(N-m+1)q(i_m)} - 1\Big)$$

The optimal proposal is q*(i) ∝ E_{p(y|x_i)}[L] ("sample in proportion to the true loss"), and the estimator "will always be unbiased" [DOCUMENTED: S12 §3.1]. For recall, the "loss" is y_i (is it a miss?), so q ∝ calibrated P(match) [INFERENCE].

**Sawade, Landwehr & Scheffer, NeurIPS 2010, "Active Estimation of F-Measures"**: instances drawn from an instrumental distribution, with an analysis that "leads to an optimal sampling distribution that minimizes estimator variance" [DOCUMENTED: S13]. UNVERIFIED: the exact optimal form (not re-derived here).

**Poms et al. 2021, "Low-Shot Validation"**: targets rare categories (≤0.1% positives, the find_pics regime) [DOCUMENTED: S14].
- Method: importance sampling with a self-normalised estimator Ĝ = Σ w_j ℓ_j / Σ w_j. Isotonic-calibrated scores are re-fit as labels arrive.
- CIs: delta-method variance.
- Results: "up to 10× fewer labels", and F1 estimated "with variance of 0.005 using as few as 100 labels".

**The catch with importance sampling for certification** [INFERENCE, derivation].
- Draw with replacement from q and define Z = y_I/(N_U q_I). Then E[Z] = p_U (the miss rate).
- With a defensive mixture, q_i ≥ τ/N_U, so 0 ≤ Z ≤ 1/τ. Any finite-sample bounded-mean CI is then valid (Hoeffding, empirical-Bernstein, or betting CIs [S24]).
- When no misses are drawn, all Z = 0. The best possible upper bound behaves like (1/τ)·ln(1/α)/n, so the worst-case certificate is 1/τ times worse than plain uniform sampling.
- So importance sampling buys low variance in the point estimate where the score is informative. It does not escape the N/m certification cost for regions the score calls "safe". This is consistent with S9's lower bound.

Recent titles (UNVERIFIED: content not read): "Prediction-Powered Active Testing" (arXiv 2607.08347) [S28]; "Active Testing of LLMs via Approximate Neyman Allocation" (arXiv 2605.10075) [S29].

---

### A3. When the labeler is a model (a VLM judge)

Here f(x) means the judge's label or probability, Y means the human label, n is the number of human-labeled items, and N is the number of judge-only items.

#### A3.1 Prediction-Powered Inference (PPI; Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, Science 2023)

PPI gives valid CIs when a small labeled set is supplemented by ML predictions on a large set, "without making any assumptions about the machine-learning algorithm". More accurate predictions give narrower intervals [DOCUMENTED: S15].

Mean estimator (the form implemented in `ppi_py`) [DOCUMENTED: S19 code]:

$$\hat\theta^{PP} = \underbrace{\frac{1}{N}\sum_{j=1}^{N} f(\tilde X_j)}_{\text{judge average}} \;-\; \underbrace{\frac{1}{n}\sum_{i=1}^{n}\big(f(X_i)-Y_i\big)}_{\text{"rectifier": judge bias measured on human items}},\qquad \hat\theta^{PP} \pm z_{1-\alpha/2}\sqrt{\frac{\hat\sigma_f^2}{N}+\frac{\hat\sigma^2_{Y-f}}{n}}$$

**Mechanism.** The judge supplies volume. The human sample estimates the judge's average error and subtracts it. The CI width depends on Var(Y − f), the disagreement rate, not Var(Y).

#### A3.2 PPI++ (Angelopoulos, Duchi & Zrnic, 2023)

PPI++ adds a power-tuning weight λ that automatically adapts to prediction quality [DOCUMENTED: S16]:

$$\hat\theta_\lambda = \bar Y_n + \lambda\big(\bar f_N - \bar f_n\big)$$

In `ppi_py`, "Setting `lam=1` recovers PPI with no power tuning, and setting `lam=0` recovers the classical CLT interval"; the default estimates λ from data [DOCUMENTED: S19 code docstring].

For the mean, the variance-minimising weight is λ* = Cov(Y,f) / [(1 + n/N)·Var(f)] [INFERENCE: control-variate algebra]. The derivation: Var(θ̂_λ) = Var(Y)/n + λ²Var(f)(1/n + 1/N) − 2λ·Cov(Y,f)/n; set the derivative to zero.

#### A3.3 Active statistical inference (Zrnic & Candès, ICML 2024)

Decide per item whether to ask a human, with probability π(x), and keep the judge's answer otherwise [DOCUMENTED: S17]:

$$\hat\theta^{\pi} = \frac1n\sum_{i=1}^n\Big[f(X_i) + \big(Y_i - f(X_i)\big)\frac{\xi_i}{\pi(X_i)}\Big],\quad \xi_i\sim\mathrm{Bern}(\pi(X_i))$$

$$\mathrm{Var}(\hat\theta^{\pi}) = \frac1n\Big[\mathrm{Var}(Y) + \mathbb E\Big[(Y-f(X))^2\Big(\frac{1}{\pi(X)}-1\Big)\Big]\Big]$$

- Optimal rule: π_opt(x) ∝ √(E[(Y−f(X))² | X = x]). In words: send humans where the judge is likely wrong.
- Practical rule: mix with uniform, π^{(τ)} = (1−τ)π + τπ^{unif}.
- CIs are asymptotically valid [DOCUMENTED: S17].

#### A3.4 Other related work

- **Stratified PPI** (Fisch et al., 2024): stratify and allocate human labels per stratum. This helps most when "the performance of the autorater varies across different conditional distributions" [DOCUMENTED: S18].
- **Generalized PPI for binary classifier evaluation** (Zou, Witten & Williamson, arXiv 2026-02): PPI for TPR (= recall), FPR and AUC [DOCUMENTED: S20]. UNVERIFIED: small-sample behaviour.
- **IR-specific work**: "Reliable Confidence Intervals for Information Retrieval Evaluation Using Generative A.I." (KDD 2024) [S21]. UNVERIFIED: methods (title only).

#### A3.5 Three ways these break in the find_pics regime

1. **CLT intervals collapse at zero events** [INFERENCE].
   - PPI, PPI++ and active inference CIs are normal-approximation intervals ("Coverage holds as n → ∞" [DOCUMENTED: S17]).
   - Suppose a human sample of 400 judge-negatives contains 0 disagreements. Then σ̂²_{Y−f} = 0 and the rectifier term vanishes. The CI claims zero uncertainty about judge misses. That is exactly the situation that is typical at 0.1% prevalence.
   - Fix: in low-count strata use exact (Clopper–Pearson / hypergeometric) bounds. Reserve CLT-based PPI for strata with tens of disagreements.
2. **PPI can be worse than ignoring the judge** [INFERENCE, computed]. Take a deep-tail stratum with true match rate p = 10⁻⁴ and a judge with Se = 0.9 and Sp = 0.9995.
   - Var(Y − f) = FN + FP − (FP−FN)² ≈ 5.1×10⁻⁴.
   - Classical p(1−p) ≈ 1.0×10⁻⁴.
   - So the PPI variance is 5× worse. The judge's false-positive rate (5×10⁻⁴) exceeds the prevalence.
   - PPI++'s λ → 0 is the documented escape hatch (classical interval).
3. **`ppi_py` detail** [DOCUMENTED: S19 code, main branch]. In `ppi_mean_ci`, when `lam=None` the function re-calls itself without passing `alternative`, so a requested one-sided interval silently becomes two-sided. The effect is conservative (a 1−α/2 one-sided bound), not anti-conservative [INFERENCE].

#### A3.6 Correcting for a noisy judge: Rogan–Gladen and why to avoid it here

Rogan & Gladen (1978) correct an apparent prevalence p̂ from an imperfect test with sensitivity Se (the fraction of true matches the judge says "yes" to) and specificity Sp (the fraction of non-matches it says "no" to) [DOCUMENTED: S22]:

$$\hat\pi = \frac{\hat p + Sp - 1}{Se + Sp - 1}$$

- Exact CIs exist for known Se/Sp (Reiczigel et al. 2010).
- Lang & Reiczigel (2014) give CIs with estimated Se/Sp [DOCUMENTED: S23].

**At low prevalence the estimate is dominated by Sp error** [INFERENCE, computed]. True p = 0.001, Se = 0.9, Sp = 0.9995, so observed judge-yes rate = 0.001399. Plugging in slightly wrong Sp values:

| assumed Sp | 0.9999 | 0.9995 (true) | 0.9990 | 0.9985 |
|---|---|---|---|---|
| RG estimate of p | 0.00144 | 0.00100 | 0.00044 | −0.00011 |

A 0.1-point error in Sp, which needs about 10⁴ labeled negatives to detect, swings the miss estimate from 144% of truth to negative.

**What to do instead** [INFERENCE]:
1. **Have a human confirm every judge-positive.** They are few, and you want to add them to the results anyway. This makes Sp irrelevant for the confirmed set: the residual error is only judge false negatives.
   - Computed effect on a near-cutoff stratum (p = 0.003, Se = 0.9, Sp = 0.999), per-item variance of the human-checked correction term:
     - classical: 3.0×10⁻³
     - PPI rectifier: 1.3×10⁻³
     - FN-only rectifier after confirming judge-positives: 3.0×10⁻⁴
   - That is a 10× reduction in human labels for the same SE.
2. **Estimate the judge's false-negative rate on the right population**: a random sample of judge-negatives from the U strata, not judge sensitivity on easy, already-found positives.
   - Retriever misses are likely the hard cases (small face, back of head, occluded bread), so the judge's sensitivity on them is plausibly lower than on found positives.
   - Transferring a validation-set Se to U would therefore overstate recall. That is anti-conservative.
3. Assessor disagreement is real even among humans: "Assessor error in stratified evaluation" (Webber et al., CIKM 2010) [S27, title via S7]. Write the match rubric down before labeling (e.g., does a photo-of-a-photo of dad count?).

---

### A4. A recall-audit design for find_pics, with worked numbers

The numbers below are worked illustrations under assumed miss distributions, chosen to show how the bound responds to each budget. Strata boundaries, α split and budgets are design choices, and the miss counts per stratum in the examples are assumptions, not predictions.

#### A4.1 The audit logic

**Step 0. Freeze before looking** [DOCUMENTED rationale: S1 §6.1, S9]. Fix all of these before any audit label exists:
- the written match rubric
- the retriever/model versions
- strata boundaries
- α and its split
- sample sizes and the random seed

Peeking and adding samples until the bound passes breaks validity. If incremental sampling is wanted, use an anytime-valid confidence sequence for sampling without replacement (Waudby-Smith & Ramdas 2020) [DOCUMENTED: S24]. A confidence sequence is a sequence of intervals that all hold simultaneously, so you can stop whenever you like. Python package: `confseq` 0.0.11 [DOCUMENTED: S25].

**Step 1. Numerator.**
- Verify every item in R, so T_R is exact and α₁ = 0.
- If R is too large, sample it and use a Clopper–Pearson lower bound T_R^L at level α₁.

**Step 2. Stratify U by retrieval-score rank.** For example: U1 = the next |R| ranks, U2 = the next 4|R|, U3 = the tail. This follows S8 §2.2 on stratification; the geometric widths are [INFERENCE].

**Step 3. Judge labels (compute budget).**
- Census U1 and U2 with the VLM.
- Draw a simple random sample n₃ from U3, sized by n₃ ≈ N₃·ln(1/α₃)/m₃, where m₃ is the number of tail misses you can afford to leave uncertified.

**Step 4. Human labels (attention budget).**
- (a) Confirm every judge-positive; confirmed ones join the delivered set D = R ∪ confirmed finds.
- (b) A blind random sample of judge-negatives from U1 and U2, interleaved with R items so provenance is hidden (S7 lesson).
- (c) Record judge–human agreement on all human-seen items. Report Se and Sp with Clopper–Pearson intervals.

**Step 5. Bounds.** Use exact per-stratum upper bounds, union-bounded so that Σ_h α_h + α₁ = α. Let k_h be the positives found in stratum h. Then:

$$M^U_{\text{remaining}} = \sum_{h\ \text{sampled}}\Big(\mathrm{HGupper}(k_h; N_h, n_h, \alpha_h) - k_h\Big),\qquad \rho_L(D) = \frac{|D\cap \text{true}|}{|D\cap\text{true}| + M^U_{\text{remaining}}}$$

The validity argument [INFERENCE]: recall is monotone in each count, so the union bound gives P(ρ ≥ ρ_L) ≥ 1 − Σα.

**Building blocks** (k hits in n draws; F_HG is the hypergeometric CDF):

Clopper–Pearson (exact) one-sided upper bound on a proportion. For k = 0 it is exactly 1 − α^{1/n} ≈ ln(1/α)/n:

$$p^U = \mathrm{Beta}^{-1}(1-\alpha;\ k+1,\ n-k)$$

Clopper–Pearson lower bound, used for T_R^L or for judge sensitivity:

$$p^L = \mathrm{Beta}^{-1}(\alpha;\ k,\ n-k+1)$$

Finite-pool exact upper bound on the count of matches in a stratum of N_h items:

$$\mathrm{HGupper}(k;N_h,n_h,\alpha_h) = \max\{K : F_{HG}(k; N_h, K, n_h) > \alpha_h\}$$

[standard definitions; S26 for Clopper–Pearson]

Wilson score interval. It is good for two-sided intervals at moderate counts. At k = 0 its upper end is slightly *above* Clopper–Pearson (computed: n = 400 gives 0.0095 vs 0.0092 for the 97.5% end). Its coverage is not guaranteed at small k, so use Clopper–Pearson for certificates:

$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}$$

[INFERENCE: standard formula; numbers computed]

Two versions are reported:
- **Judge-referenced LCB**: U-strata labels come from the VLM, so the statement is recall as judged by the VLM.
- **Hybrid human-referenced LCB**: add the human-sample upper bounds on judge-misses within U1 and U2. The tail term stays judge-referenced. Human-certifying a 135k-item tail needs about 3·135,000/m human labels, which is infeasible for any useful m. State this explicitly.

**Step 6. Point estimate and two-sided interval**, if wanted alongside the LCB:
- Ratio estimator: ρ̂ = T̂_R / (T̂_R + M̂).
- Delta-method variance, with independent strata:

$$\widehat{\mathrm{Var}}(\hat\rho)\approx\frac{\hat M^2\,\widehat{\mathrm{Var}}(\hat T_R)+\hat T_R^2\,\widehat{\mathrm{Var}}(\hat M)}{(\hat T_R+\hat M)^4}$$

  From ∂ρ/∂T = M/(T+M)² and ∂ρ/∂M = −T/(T+M)² [INFERENCE].
- Stratified bootstrap (resample within strata) fails the same way as the CLT when a stratum has 0 hits, because every replicate is 0 [INFERENCE].
- Webber's half-prior beta-binomial Monte Carlo is the documented best two-sided choice [DOCUMENTED: S8].
- One-sided tip for library CIs: the upper end of a **two-sided 90%** Clopper–Pearson interval is the **one-sided 95%** upper bound. `scipy.stats.binomtest(k,n).proportion_ci(confidence_level=0.90, method='exact')`; `statsmodels ... proportion_confint(..., alpha=0.10, method='beta')` [DOCUMENTED: S26].

#### A4.2 Worked example A: "photos of the user", N = 150,000, true matches ≈ 2,000

**Assumed truth:**
- |R| = 2,400 with T_R = 1,860 (precision 0.775, recall 0.93).
- U1 = ranks 2,401–4,800 (2,400 items, 100 misses, 4.2%).
- U2 = ranks 4,801–14,400 (9,600 items, 30 misses, 0.31%).
- U3 = 135,600 items (10 misses, 0.007%).

α = 0.05. All bounds are exact hypergeometric. "LCB(R)" is the bound on recall of R; "LCB(D)" is the bound on the delivered set after adding confirmed finds. Values are the median (10th percentile) over 200–300 simulated audits. Empirical coverage was 94–100%; 94% is within Monte Carlo error of the nominal 95%, because exact bounds are conservative by construction.

| Design (labels per U1 / U2 / U3) | Labels | LCB(R) | LCB(D) |
|---|---|---|---|
| Human-only, stratified 300 / 300 / 400 | 1,000 | 0.52 (0.50) | 0.52 |
| Plain elusion SRS of U, n = 2,000 | 2,000 | 0.80 (0.74) | n/a |
| Plain elusion SRS of U, n = 20,000 | 20,000 | 0.90 (0.88) | n/a |
| Judge: census U1, 2,000 of U2, 5,000 of U3 | 9,400 | n/a | **0.93 (0.90)** |
| Judge: census U1 + U2, 5,000 of U3 | 17,000 | 0.90 | **0.96 (0.94)** |
| Judge: census U1 + U2, 10,000 of U3 | 22,000 | 0.91 | **0.97 (0.96)** |

Reading the table [INFERENCE]:
- **Human-only certification is hopeless at this prevalence.** The tail term alone is 135,600 × 4.09/400 ≈ 1,390 possible misses.
- **Plain elusion** at 20,000 labels certifies only 0.90, because it spends labels uniformly.
- **The stratified judge audit certifies more for fewer labels**, because it does two things at once: it finds and adds the near-cutoff misses (D's true recall ≈ 0.98–0.996), and it spends confidence only on the tail.

**Hybrid human-referenced version** (assume judge Se = 0.9 on U-positives, so the judge finds about 117 of the 130 misses in U1 ∪ U2). Components:
- Humans confirm about 145 judge-positives.
- 400 human labels on U1 judge-negatives (2,300 items) find k = 2 judge misses → upper bound 41.
- 1,000 human labels on U2 judge-negatives (9,570 items) find 0 → upper bound 37.
- 10,000 judge labels on the tail find 0 → upper bound 53 (each at α/3).

$$\rho_L^{H} = \frac{1979}{1979 + (41-2) + 37 + 53} = 0.939$$

So **"recall ≥ 0.92 at 95%" is reachable with about 1,550 human glances plus about 22,000 VLM calls.** With only 400 human labels on U2 judge-negatives (upper bound 95) it drops to 0.914. With a 20,000-item tail sample it rises to 0.951. [computed]

#### A4.3 Worked example B: "bread", N = 150,000, true matches ≈ 300

**Assumed truth:**
- |R| = 400 with T_R = 270 (precision 0.675, recall 0.90).
- U1 = 400 items (18 misses).
- U2 = 1,600 items (8 misses).
- U3 = 147,600 items (4 misses).

| Design (labels per U1 / U2 / U3) | Labels | LCB(R) | LCB(D) |
|---|---|---|---|
| Human-only 300 / 300 / 400 | 1,000 | 0.15 | 0.16 |
| Plain elusion SRS of U, n = 20,000 | 20,000 | 0.81 (0.74) | n/a |
| Judge: census U1 + U2, 5,000 of U3 | 7,000 | 0.71 | 0.77 (0.68) |
| Judge: census U1 + U2, 10,000 of U3 | 12,000 | 0.80 | 0.88 (0.82) |
| Judge: census U1 + U2, 20,000 of U3 | 22,000 | 0.85 | **0.94 (0.90)** |
| Judge: census U1 + U2, 30,000 of U3 | 32,000 | 0.85 | **0.94 (0.92)** |

**Exact tail arithmetic** (judge-referenced, D = 296 + k):

| Tail sample n₃ | Tail hits k | Remaining-miss bound | LCB |
|---|---|---|---|
| 10,000 | 0 | ≤ 42 | 0.876 |
| 20,000 | 0 | ≤ 20 | 0.937 |
| 20,000 | 1 | ≤ 32 | 0.903 |
| 30,000 | 0 | ≤ 13 | 0.958 |
| 30,000 | 1 | ≤ 20 | 0.937 |

**Hybrid human-referenced version.** Humans check:
- all 384 U1 judge-negatives (census; 2 judge misses found and added)
- 800 of the 1,592 U2 judge-negatives (0 found → upper bound 5)

LCB_H = 0.908 with a 20,000 tail, 0.934 with 30,000, and 0.949 with 40,000. So **"recall ≥ 0.92 for bread" needs about 1,200 human glances plus about 32,000 VLM calls.** For a 0.80 target, about 8,000 tail calls suffice when k = 0 (LCB 0.848). [computed]

#### A4.4 Cost and wall-clock [INFERENCE, using the throughput numbers in B8]

**VLM call counts:**
- 22,000–32,000 calls: about 2–9 h on an M3 Max-class Mac, at a planning range of 0.33–1.5 s per image for a 4B VLM at 448 px; roughly double on an M2 Pro.
- 9,400 calls (example A's cheaper design): about 1–4 h.

**So a certified ≥ 0.92 claim for a rare concept is an "audit mode" job (overnight), not an interactive default.** The interactive default can report:
- (i) the exact examined counts
- (ii) a weaker certified bound from a small tail sample
- (iii) a clearly labeled model-based estimate, Σ_{U} p̂_i (Quant-style)

**Human glances.** 1,000–1,500 thumbnail checks in a grid UI are plausibly 15–45 minutes for clear-cut negatives. UNVERIFIED: no measured source.

#### A4.5 What "examined" should mean in the report [INFERENCE]

Report three separate exact counts, never one blended number:
1. **Scored**: items the embedding model scored, normally all N.
2. **Judged**: items the VLM looked at, split as census strata plus random tail sample.
3. **Human-verified**: items a person looked at.

Each recall bound must name its ruler: "as judged by the VLM" or "as judged by you". The judge's measured agreement with the human (Se, Sp with CIs on the human-checked items) belongs next to any judge-referenced number.

#### A4.6 Pitfalls checklist [INFERENCE unless tagged]

- Do not re-test until pass [S1 §6.1].
- Do not let labelers know an item's provenance [S7].
- Do not assume R is clean; verify it [S7].
- Do not use CLT/PPI intervals in strata with 0–5 events (A3.5).
- Do not transfer judge sensitivity from easy positives to misses (A3.6).
- Do not refit the retriever on audit labels and then reuse those same labels for certification. The generator and audit rule must be fixed before the certification labels are examined [S9].
- Near-duplicates and bursts (20 shots of the same moment) make "items" non-independent in meaning but not in sampling. The bound is still valid for item counts. Consider also reporting by event/burst if the user cares about moments, not files.

---

### A5. Open-source implementations

| Package | What it gives | Status |
|---|---|---|
| `ppi-python` (`ppi_py`) | `ppi_mean_ci`, `ppi_mean_pointestimate`, `ppi_mean_pval`, quantiles, OLS/GLM, cross-PPI, power analysis. Takes `lam` (PPI++), `w`/`w_unlabeled` sample weights (usable for IPW/stratified designs) and `alternative` ('two-sided', 'larger', 'smaller') | v0.2.3, 2024-12-22 on PyPI; repo commits to 2026-04 [DOCUMENTED: S19] |
| `tarexp` | TAR experiment framework. Stopping rules: Knee, Budget, ReviewHalf, BatchPrec, Rule2399, Quant(nstd), CHMHeuristics. **No QBCB** | v0.1.4, 2024-03-18 [DOCUMENTED: S3] |
| `buscarpy` / `buscarR` | Callaghan–Müller-Hansen hypergeometric stopping criterion | v0.0.2 (2023-10); repo pushed 2025-10; a Svelte web app updated 2026-01 [DOCUMENTED: S5] |
| `confseq` | Confidence sequences / uniform boundaries | v0.0.11, 2023-01 [DOCUMENTED: S25] |
| `scipy.stats` / `statsmodels` | Exact Clopper–Pearson (`binomtest(...).proportion_ci(method='exact')`; `proportion_confint(method='beta')`), Wilson; `scipy.stats.hypergeom` for exact finite-pool bounds | [DOCUMENTED: S26] |
| `asreview` | Systematic-review screening tool | v3.0.8, 2026-06 [DOCUMENTED: S30]. UNVERIFIED: which stopping rules ship in core |

I did not find a library that does the full stratified exact-bound recall certificate of A4 [INFERENCE]. The pieces are about 50 lines on top of `scipy.stats.hypergeom`.

---

## Part B: Platform access (read-only is a hard requirement)

### B6. Apple

#### osxphotos (RhetTbull), Mac

**Version and OS support**
- Latest release is v0.77.2 (2026-09-27): "Fix shared albums on macOS 26+, system library detection on macOS 27" [DOCUMENTED: S40, checked via GitHub API].
- v0.77.0 (2026-09-20) added "initial macOS 27 search & media info support" [DOCUMENTED: S40].
- The README table lists Sequoia 15.x, Tahoe 26.x and 27.0 as supported [DOCUMENTED: S41].
- But PyPI text still says it does not "yet fully support 26.x" [DOCUMENTED: S42]. Treat Tahoe and 27 as working but young [INFERENCE].

**Read-only database access**
- The Photos SQLite database is opened with `sqlite3.connect(f"{dbpath.as_uri()}?mode=ro", uri=True)` [DOCUMENTED: S43].
- If it is locked (Photos.app open), osxphotos copies the DB plus its WAL/SHM files to a temp directory (`osxphotos_` prefix) and reads the copy [DOCUMENTED: S44].
- No delete API was found in osxphotos [INFERENCE from source review, S44–S46].
- PhotoScript's README states Photos' AppleScript "cannot ... delete a photo" [DOCUMENTED: S47].

**`osxphotos query --json`** emits the deep dict (`p.json(shallow=False)`) [DOCUMENTED: S48]. Fields relevant to search:

Identity and paths:
- `uuid`, `filename`, `original_filename`
- `date`, `date_added`, `date_modified`
- `path`, `path_edited`, `path_raw`, `path_live_photo`
- `path_derivatives`: "derivative (preview) images ... sorted by file size (largest first)"

Status flags:
- `ismissing` (original not on disk), `incloud`, `iscloudasset`, `hasadjustments`
- `intrash`, `hidden`, `favorite`

Content metadata:
- `keywords`, `albums`, `folders`, `title`, `description`
- `persons`, `person_info` (Photos' named faces)
- `labels`, `labels_normalized` ("labels applied to photo by Photos image categorization")
- `ai_caption` ("AI generated caption")
- `media_analysis`

Location and camera: `latitude`, `longitude`, `place`, `exif_info`.

Aesthetic score: `score` (ScoreInfo: `overall`, `curation`, `well_framed_subject`, ...). UNVERIFIED: the scale and meaning of `overall`.

`search_info` subfields [DOCUMENTED: S49]:
- Labels and places: `labels`, `place_names`, `streets`, `neighborhoods`, `city`, `state`, `country`, `bodies_of_water`, `landmarks`, `venues`, `venue_types`
- Time: `month`, `year`, `season`, `times_of_day`, `holidays`
- Events and people: `activities`, `events`, `human_actions`, `pets`, `age_groups`
- Media and text: `media_types`, `detected_text`, `document_types`, `captured_by_me`, and more

Many of these are populated only on newer macOS versions [INFERENCE].

Sources for the field list: [DOCUMENTED: S45, S50]. Why this matters for find_pics [INFERENCE]: these fields are free, independent signals, especially `persons` for "photos of dad". They can define strata or cheap pre-filters. They are Apple's model outputs, so they are not ground truth.

**Export flags** [DOCUMENTED: S50, quotes]
- `--download-missing`: "uses Applescript to interact with Photos to export the photo which will force Photos to download from iCloud if the photo does not exist on disk."
- `--use-photos-export`: "Force the use of AppleScript or PhotoKit to export even if not missing."
- `--use-photokit`: a direct Photos interface; "Highly experimental alpha feature."
- `--preview-if-missing`: "Export preview image generated by Photos if the actual photo file is missing."
  - This copies the existing lower-resolution derivative and does not trigger a download [INFERENCE].

**Commands and flags that write, to ban in a read-only tool** [DOCUMENTED: S41, S50–S53]:

| Command or flag | What it writes |
|---|---|
| `import` | Adds photos to the library |
| `batch-edit` | Changes title, description and keywords. Has `--dry-run`, and "--undo cannot undo album changes" |
| `timewarp` | "directly modifies your Photos library database ... may corrupt, damage, or destroy your Photos library" |
| `push-exif` | "will modify the original files in the Photos library" |
| `add-locations` | Writes location data |
| `sync` | Syncs metadata and albums between libraries |
| `query --add-to-album` | Adds results to an album |
| `export --add-exported-to-album` / `--add-skipped-to-album` / `--add-missing-to-album` | Create albums and add to them |

`export --cleanup` deletes files only in the export directory, not in the library [DOCUMENTED: S50].

**Read-only commands**: `query`, `export` without the album flags, `albums`, `persons`, `labels`, `keywords`, `places`, `info`, `dump`, `inspect`, `uuid`, `dbversion` [INFERENCE from descriptions].

#### Adding results back as an album (the only write find_pics should ever do)

**photoscript** [DOCUMENTED: S54, S47]
- `PhotosLibrary().create_album(name, folder=None)` creates an album.
- `Album.add(photos)` adds library photos to it.
- `delete_album` "deletes album (but does not delete photos in the album)".
- `Album.remove()` works by creating a new same-name album, copying all but the removed photos, and deleting the original. That destroys album identity. Avoid it.

**osxphotos `PhotosAlbum`** wraps photoscript with `add`, `update`, `append` and `extend`, and has no remove or delete methods [DOCUMENTED: S55].

**PhotoKit**
- `PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle:)` creates an album, and `addAssets(_:)` adds to it [DOCUMENTED: S56, S57].
- Deletion is a separate API: `PHAssetChangeRequest.deleteAssets(_:)` [DOCUMENTED: S58].
- Apple: "For each call to the performChanges ... method, Photos shows an alert asking the user for permission to edit the contents of the photo library." [DOCUMENTED: S56]
- UNVERIFIED: whether album-only changes actually prompt in current OS versions, and whether deleteAssets always goes to Recently Deleted.

**How to guarantee no deletion** [INFERENCE]
- (1) Code-level ban list, enforced by a CI grep/lint: `deleteAssets`, `deleteAssetCollections`, `removeAssets`, photoscript `delete_album` / `delete_folder` / `Album.remove`, and every osxphotos write command above.
- (2) The only write path is: create a new album with a unique, prefixed name, then add. Album membership is a reference, not a move or copy.
- (3) Search reads only from the `mode=ro` DB or the temp copy.
- (4) Require a Time Machine/APFS backup before the first album write.

#### PhotoKit on iOS (future native app)

**Reading images**
- Fetch with `PHAsset.fetchAssets(with:options:)` [DOCUMENTED: S59].
- `PHImageManager.requestImage(for:targetSize:contentMode:options:resultHandler:)` returns an image "at, or near, the size you specify", possibly first as a degraded version [DOCUMENTED: S60].
- `PHCachingImageManager` is "optimized for batch preloading large numbers of assets" [DOCUMENTED: S61].
- With `isNetworkAccessAllowed = true` (the default is false), Photos downloads iCloud-only images [DOCUMENTED: S62].

**Access levels.** `PHAccessLevel` has only `addOnly` and `readWrite`. There is **no read-only level** [DOCUMENTED: S63], so read-only must be enforced in code [INFERENCE].

**Limited library.**
- `.limited` status (iOS 14+) means the app sees only user-selected assets [DOCUMENTED: S64, S65].
- iOS 17 added a full-access prompt showing photo count and samples, and periodic reminders [DOCUMENTED: S66].
- Consequence [INFERENCE]: in limited mode a recall claim must be scoped to "the photos you shared with the app".

**Background processing**
- `BGProcessingTask` runs "for minutes" when idle, can be interrupted, and `requiresExternalPower` is available [DOCUMENTED: S67, S68].
- iOS 26 `BGContinuedProcessingTask` [DOCUMENTED: S69]:
  - It starts from a user action, shows progress in a Live Activity, and can be cancelled.
  - Background GPU needs `requiredResources = .gpu` plus an entitlement.
  - It fits a user-tapped "index my library" job [INFERENCE].

**Apple's own labels and People are not exposed to third-party iOS apps.**
- The `PHAsset` API lists no label, person or caption properties [DOCUMENTED by absence: S59, S70].
- An iOS app must compute its own (Vision / Core ML) [INFERENCE].
- osxphotos gets them only by reading the Mac's Photos.sqlite.

#### icloudpd (iCloud Photos Downloader)

**Maintenance status**
- Latest release is v1.32.3 (2026-05-30): "fix: restore 2FA for Apple's updated auth flow (2026+)". The previous release was v1.32.2 (2025-09-02) [DOCUMENTED: S71, checked via GitHub API].
- The README's first line is "Looking for MAINTAINER for this project" (issue #1305, 2026-01-06) [DOCUMENTED: S72].
- Open auth issues: #1366 "2fa not working again?" (2026-08-06), and #1376 HTTP 410 (2026-09-12) [DOCUMENTED: S73].
- Status: works intermittently; maintenance at risk. UNVERIFIED: whether it works today.

**ADP blocks it.**
- Prerequisites are "Enable Access iCloud Data on the Web" and "Disable Advanced Data Protection"; otherwise Apple returns ACCESS_DENIED [DOCUMENTED: S72, checked].
- Consequence [INFERENCE]: using icloudpd forces the user to weaken their iCloud encryption, which is a real privacy cost.

**Sizes.** `--size original|medium|thumb|adjusted|alternative` (repeatable). Unavailable sizes fall back to original unless `--force-size` is set. `--live-photo-size` also exists [DOCUMENTED: S74, S75].

**Safe flags**
- `--dry-run`: "no changes to local storage or iCloud remote storage are made".
- `--only-print-filenames`.
- `--list-albums`.

[DOCUMENTED: S75]

**Destructive flags to ban** [DOCUMENTED: S75, S76]

| Flag | Effect |
|---|---|
| `--delete-after-download` | Deletes in iCloud (moves to Recently Deleted). Deprecated but present |
| `--keep-icloud-recent-days N` | Turns on "Move" mode, which deletes older iCloud assets after download |
| `--auto-delete` | Deletes **local** files removed in iCloud |
| `--set-exif-datetime` | Modifies downloaded files |

**No structural read-only guarantee.** It authenticates with full Apple ID credentials, which by themselves allow deletion [INFERENCE]. If it is used at all: wrap it with a flag allowlist and default to copy mode.

#### Safest Apple path [INFERENCE]

On the Mac:
1. Use osxphotos as a library to read metadata (`mode=ro`).
2. Read pixels from files already on disk: originals where `ismissing` is false, else the largest `path_derivatives` preview.
3. Do not use `--download-missing` or Photos.app automation for indexing. It makes Photos.app download originals and grow the library. That is not destructive, but it is a side effect.
4. The single permitted write is create-album-then-add, behind explicit user confirmation.

icloudpd is a last resort (ADP conflict, fragile auth, delete-capable credentials). UNVERIFIED: whether Terminal/Python needs Full Disk Access to read the Photos library bundle on current macOS.

### B7. Google and Android

#### Google Photos Library API after 2025-03-31

**What was removed and what remains**
- Removed scopes: `photoslibrary.readonly`, `photoslibrary.sharing`, `photoslibrary`.
- Remaining scopes: `photoslibrary.appendonly`, `photoslibrary.readonly.appcreateddata`, `photoslibrary.edit.appcreateddata`.
- Listing and search methods "can now only be used with albums and media items created by your app".
- "Apps needing users to select from their entire library should migrate to the new Google Photos Picker API."
- [DOCUMENTED: S80, checked]

**Consequences**
- A third-party app cannot read a user's whole Google Photos library through the API anymore [DOCUMENTED: S80].
- rclone: "From March 31, 2025 rclone can only download photos it uploaded." [DOCUMENTED: S81]

**Picker API**
- Mechanism: create a session, get a `pickerUri`, the user picks items in the Google Photos UI, poll until `mediaItemsSet`, then call `mediaItems.list` [DOCUMENTED: S82].
- `maxItemCount` "Defaults to 2000 ... Values above 2000 will be coerced to 2000" [DOCUMENTED: S83].
- `baseUrl` lasts 60 minutes and needs an OAuth bearer token. The `=d` download keeps EXIF "except the location metadata" [DOCUMENTED: S84].
- **Unsuitable for 150k** [INFERENCE]: that is at least 75 manual sessions, with GPS stripped.

**Ambient API.** For TVs and photo frames; requires partner-program acceptance and Google review [DOCUMENTED: S85, S86]. Not a route for a personal tool [INFERENCE].

#### Google Takeout: the only complete export

**What is preserved**
- The embedded file metadata keeps the original timestamp.
- Google-side extras go to JSON sidecars.
- Archive size can go up to 50 GB per file.
- [DOCUMENTED: S87]

**Layout: duplicates across folders.** Year folders ("Photos from 2020") plus album folders hold duplicates [DOCUMENTED: S88]. Dedupe by content hash [INFERENCE].

**Sidecar naming changed in late 2024.** `X.jpg.json` became `X.jpg.supplemental-metadata.json`, truncated (e.g. `.supplemental-metada.json`, `.supplementa`). Bad matching produced bogus dates [DOCUMENTED: S89].

**Sidecar fields** [DOCUMENTED: S90, S91]
- `title`, `description`, `imageViews`
- `creationTime`, `photoTakenTime{timestamp, formatted}`
- `geoData`, `geoDataExif` (note: not "geoDataExp")
- `people[{name}]`: named face groups only, an inference for unnamed ones
- `url`, `googlePhotosOrigin`
- The schema varies by export age [S91].
- UNVERIFIED: `appSource`.

**Edited copies.** `-edited` copies are exported. Sources disagree on whether they have their own sidecar (UNVERIFIED).

**Tools**
- Xentraxx/GooglePhotosTakeoutHelper: "Files will be moved from the input folder during processing ... keep the original ZIPs as backup!" A `--keep-input` flag works on a copy [DOCUMENTED: S92]. Never run it on the only copy [INFERENCE].
- immich-go reads Takeout zips directly and has `--people-tag` [DOCUMENTED: S93].

#### Android on-device

**Read permissions.** `READ_MEDIA_IMAGES` / `READ_MEDIA_VIDEO` (Android 13+), queried via `ContentResolver` on MediaStore [DOCUMENTED: S94].

**Partial access.** Android 14's `READ_MEDIA_VISUAL_USER_SELECTED` returns only user-selected items [DOCUMENTED: S95].

**Location.** It is hidden unless the app has `ACCESS_MEDIA_LOCATION` and calls `setRequireOriginal()` [DOCUMENTED: S94].

**Writes and deletes need consent**
- On Android 11+, edits or deletes of others' media go through `createWriteRequest`, `createDeleteRequest` or `createTrashRequest`, and "users see a dialog that requests their consent" [DOCUMENTED: S94].
- Exception: the special `MANAGE_MEDIA` permission allows changes "without per-file prompts" [DOCUMENTED: S94].
- Guarantee [INFERENCE]: never declare `MANAGE_MEDIA` or `MANAGE_EXTERNAL_STORAGE`, and never call the `create*Request` APIs.

**Play policy.** The "Photo and Video Permissions" policy allows `READ_MEDIA_*` only when pickers aren't enough for core functionality, with a declaration form. Enforcement ran 2025-01-22, with full compliance by 2025-05-28 [DOCUMENTED: S96, S97]. This only matters for Play distribution [INFERENCE].

**Photo picker.** No permission is needed, and it includes cloud media providers [DOCUMENTED: S98]. Cloud-only Google Photos items are not in MediaStore [INFERENCE from S98; UNVERIFIED via the CloudMediaProvider javadoc].

**Background indexing.** WorkManager, with `setForeground()` for long work [DOCUMENTED: S99].

#### Safest Google/Android path [INFERENCE]

- **Cloud library**: Takeout treated as an immutable input. Read the zips (or a copy), match sidecars by `title` with tolerant suffix rules, dedupe by hash, and take dates from `photoTakenTime`, never mtime.
- **Phone-local photos**: MediaStore with `READ_MEDIA_*` only.
- **No API path writes back** to the user's full Google library, which is itself a safety property.

### B8. On-device Mac execution

**Frameworks**
- **mlx-vlm** supports Qwen2/2.5-VL, Qwen3-VL, Gemma 3, SmolVLM, PaliGemma, Moondream, Florence2, Idefics3 and others, with 4/8-bit quantization and continuous batching [DOCUMENTED: S100].
- **Qwen2.5-VL pixel cap**: its default `max_pixels` is "12× the training resolution", so you must cap pixels yourself or per-image cost explodes [DOCUMENTED: S101].
- **llama.cpp (libmtmd)** supports Gemma 3, Qwen2/2.5-VL, SmolVLM and others [DOCUMENTED: S102]. Historically its Metal vision encoding was slow (5.4 s vs 0.12 s CUDA in one report) [DOCUMENTED: S103].
- **ONNX Runtime CoreML EP**: a "will compile and save to disk every time" cost unless a cache dir is set, plus op restrictions [DOCUMENTED: S104]. Immich users hit "CoreML does not support input dim > 16384" [DOCUMENTED: S105].
- **Core ML (rclip)**: about 180 img/s for CLIP ViT-B/32 on M1 Max via coremltools, vs 160 for PyTorch. rclip moved back from ONNX Runtime because Apple Silicon indexing regressed [DOCUMENTED: S106].

**Measured numbers**

| Model / task | Hardware / path | Number | Source |
|---|---|---|---|
| SigLIP2-SO400M-384 embedding | M4, MLX | ~5 img/s; "8/4-bit quantization was slower" (compute-bound) | [DOCUMENTED: S107] |
| ViT-L/14-336 encoder | M1 Max, Neural Engine | 127 ms/img | [DOCUMENTED: S108] |
| CLIP ViT-B/32 | M1 Max, Core ML | 180 img/s | [DOCUMENTED: S106] |
| SigLIP ViT-L-16-256 | M1 Air, CPU (Immich) | ~1.2 img/s | [DOCUMENTED: S105] |
| Qwen3-VL-4B 4-bit, one image | M4 Max, vllm-mlx | 0.8 s @224², 1.2 s @448², 1.8 s @768² | [DOCUMENTED: S109] |
| 7B LLM + FastViTHD, 256 visual tokens | M1 Max | 641 ms time-to-first-token | [DOCUMENTED: S108] |

Image GFLOPs (open_clip profile) [DOCUMENTED: S110]:

| Model | GFLOPs per image |
|---|---|
| ViT-B-16-SigLIP | 35 |
| SO400M-224 | 220 |
| SO400M-384 | 670 |

**Wall-clock estimates** [INFERENCE: FLOPs at 50–75% of estimated peak, calibrated on the M4 5 img/s point; see S107, S110]
- Embedding 150k with SO400M-384: about 5.5–8 h on M2 Pro, about 2.3–4 h on M3 Max.
- SO400M-224: about 3× faster.
- ViT-B/16: tens of minutes, at which point JPEG/HEIC decode likely becomes the bottleneck.

**VLM yes/no judge**
- 4B at 448 px: about 0.33 s/img (FLOP estimate) to about 1–1.5 s/img (anchored on the measured 1.2 s on M4 Max) on Max-class chips. Roughly double on Pro-class.
- So 5,000 candidates take about 0.5–2 h, and an A4 audit of 22–32k calls takes about 2–9 h on Max-class.
- Output is one token, so cost ≈ vision encoder plus prefill of the visual tokens. Capping resolution is the main lever: Qwen2.5-VL uses 28 px/token, so 448² ≈ 256 tokens; Qwen3-VL uses 32 px/token [DOCUMENTED: S111, S112].

**2025–2026 developments**
- **M5 + MLX**: Apple reports time-to-first-token speedups of 3.3–4.1× vs M4 (e.g. 8B 4-bit: 3.97×), and only 1.19–1.27× for decode. MLX needs macOS 26.2+ to use the GPU Neural Accelerators [DOCUMENTED: S113]. Prefill-bound VLM judging should gain about 3–4× [INFERENCE].
- **Foundation Models framework**: text-only for developers in the 26 cycle. At WWDC26 Apple said "the on-device model is also gaining Vision capabilities", with image attachments in iOS/macOS 27 [DOCUMENTED: S114]. UNVERIFIED: shipping status and throughput.
- **Vision framework**: `VNClassifyImageRequest` (1,303 labels in revision 1) and `VNGenerateImageFeaturePrintRequest` (image-image similarity) [DOCUMENTED: S115, S116]. There is no public CLIP-style text-image embedding API (absence in the WWDC26 sessions reviewed). UNVERIFIED: a global absence across all of Apple's docs.
- **PE-Core**: no MLX/Core ML port or Apple Silicon numbers found. UNVERIFIED.

---

## Sources

**Part A**
- S1. Lewis, Yang, Frieder. "Certifying One-Phase Technology-Assisted Reviews." CIKM 2021. https://arxiv.org/abs/2108.12746 ; https://dl.acm.org/doi/10.1145/3459637.3482415
- S2. Yang, Lewis, Frieder. "Heuristic Stopping Rules for Technology-Assisted Review." DocEng 2021. https://arxiv.org/abs/2106.09871 ; https://dl.acm.org/doi/abs/10.1145/3469096.3469873
- S3. TARexp (Yang et al.), repo and stopping-rule source. https://github.com/eugene-yang/tarexp ; https://raw.githubusercontent.com/eugene-yang/tarexp/main/tarexp/component/stopping.py ; PyPI `tarexp` 0.1.4 ; paper https://arxiv.org/abs/2202.11827
- S4. Callaghan, Müller-Hansen. "Statistical stopping criteria for automated screening in systematic reviews." Systematic Reviews 9:273 (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7700715/
- S5. buscarpy / buscarR / buscar-app. https://github.com/mcallaghan/buscarpy ; https://mcallaghan.github.io/buscarR/ ; PyPI `buscarpy`
- S6. Cormack, Grossman. "Engineering Quality and Reliability in Technology-Assisted Review." SIGIR 2016. https://dl.acm.org/doi/10.1145/2911451.2911510
- S7. O'Halloran, McManus, Harbison, Grossman, Cormack. "Comparison of Tools and Methods for Technology-Assisted Review" (2024). https://ediscoverytoday.com/wp-content/uploads/2024/01/Comparison-of-Tools-and-Methods-for-Technology-Assisted-Review.pdf ; https://link.springer.com/chapter/10.1007/978-3-031-64359-0_9
- S8. Webber. "Approximate Recall Confidence Intervals." ACM TOIS 31(1), 2013. https://arxiv.org/abs/1202.2880 ; https://dl.acm.org/doi/10.1145/2414782.2414784
- S9. Anthony, Salehzadeh Nobari. "Finite-Sample Coverage Audits for High-Recall Candidate Generation: Certification and Learning-Theoretic Design." arXiv 2607.21480 (2026). https://arxiv.org/abs/2607.21480
- S10. Fletcher, Stevenson. "Confidence-Based Stopping Methods for Systematic Reviews." arXiv 2606.15380 (2026). https://arxiv.org/abs/2606.15380
- S11. Li, Kanoulas. "When to Stop Reviewing in TAR: Sampling from an Adaptive Distribution to Estimate Residual Relevant Documents." ACM TOIS 38(4):41, 2020. doi:10.1145/3411755 (citation via S7)
- S12. Kossen, Farquhar, Gal, Rainforth. "Active Testing: Sample-Efficient Model Evaluation." ICML 2021. https://proceedings.mlr.press/v139/kossen21a.html
- S13. Sawade, Landwehr, Scheffer. "Active Estimation of F-Measures." NeurIPS 2010. https://papers.nips.cc/paper/3999-active-estimation-of-f-measures
- S14. Poms et al. "Low-Shot Validation: Active Importance Sampling for Estimating Classifier Performance on Rare Categories." 2021. https://arxiv.org/abs/2109.05720
- S15. Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. "Prediction-powered inference." Science 382:669 (2023), doi:10.1126/science.adi6000. https://arxiv.org/abs/2301.09633
- S16. Angelopoulos, Duchi, Zrnic. "PPI++: Efficient Prediction-Powered Inference." 2023. https://arxiv.org/abs/2311.01453
- S17. Zrnic, Candès. "Active Statistical Inference." ICML 2024. https://arxiv.org/abs/2403.03208
- S18. Fisch et al. "Stratified Prediction-Powered Inference for Hybrid Language Model Evaluation." 2024. https://arxiv.org/abs/2406.04291 (venue UNVERIFIED)
- S19. ppi_py. https://github.com/aangelopoulos/ppi_py ; source https://raw.githubusercontent.com/aangelopoulos/ppi_py/main/ppi_py/ppi.py ; PyPI `ppi-python` 0.2.3
- S20. Zou, Witten, Williamson. "Generalized Prediction-Powered Inference, with Application to Binary Classifier Evaluation." arXiv 2602.10332 (2026). https://arxiv.org/abs/2602.10332
- S21. "Reliable Confidence Intervals for Information Retrieval Evaluation Using Generative A.I." KDD 2024. https://dl.acm.org/doi/10.1145/3637528.3671883 (title only)
- S22. Rogan, Gladen. "Estimating prevalence from the results of a screening test." Am J Epidemiol 107:71–76 (1978). Formula via https://epitools.ausvet.com.au/trueprevalence ; https://www.rdocumentation.org/packages/epiR/versions/0.9-19/topics/epi.prev
- S23. Lang, Reiczigel. "Confidence limits for prevalence of disease adjusted for estimated sensitivity and specificity." Prev Vet Med (2014). https://www.sciencedirect.com/science/article/abs/pii/S0167587713002936 ; Reiczigel et al. 2010 exact limits (ResearchGate 41668726)
- S24. Waudby-Smith, Ramdas. "Confidence sequences for sampling without replacement." NeurIPS 2020. https://arxiv.org/abs/2006.04347
- S25. `confseq` on PyPI (0.0.11). https://pypi.org/project/confseq/
- S26. SciPy `binomtest` docs, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html ; statsmodels `proportion_confint`, https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- S27. Webber, Oard, Scholer, Hedin. "Assessor error in stratified evaluation." CIKM 2010. doi:10.1145/1871437.1871508 (citation via S7)
- S28. "Prediction-Powered Active Testing." arXiv 2607.08347 (title only)
- S29. "Active Testing of Large Language Models via Approximate Neyman Allocation." arXiv 2605.10075 (title only)
- S30. `asreview` on PyPI (3.0.8). https://pypi.org/project/asreview/

**Part B: Apple**
- S40. osxphotos releases. https://github.com/RhetTbull/osxphotos/releases
- S41. osxphotos README. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/README.md
- S42. osxphotos on PyPI. https://pypi.org/project/osxphotos/
- S43. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/sqlite_utils.py
- S44. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/photosdb/photosdb.py
- S45. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/photoinfo.py
- S46. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/photokit.py
- S47. PhotoScript README. https://raw.githubusercontent.com/RhetTbull/PhotoScript/master/README.md
- S48. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/cli/print_photo_info.py
- S49. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/searchinfo.py
- S50. osxphotos CLI docs. https://rhettbull.github.io/osxphotos/cli.html ; API https://rhettbull.github.io/osxphotos/API_README.html
- S51. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/cli/timewarp.py
- S52. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/cli/push_exif.py
- S53. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/cli/batch_edit.py ; https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/cli/query.py
- S54. https://raw.githubusercontent.com/RhetTbull/PhotoScript/master/photoscript/__init__.py
- S55. https://raw.githubusercontent.com/RhetTbull/osxphotos/main/osxphotos/photosalbum.py
- S56. https://developer.apple.com/tutorials/data/documentation/photokit/requesting-changes-to-the-photo-library.json
- S57. https://developer.apple.com/tutorials/data/documentation/photos/phassetcollectionchangerequest/addassets(_:).json
- S58. https://developer.apple.com/tutorials/data/documentation/photos/phassetchangerequest/deleteassets(_:).json
- S59. https://developer.apple.com/tutorials/data/documentation/photos/phasset.json
- S60. https://developer.apple.com/tutorials/data/documentation/photos/phimagemanager/requestimage(for:targetsize:contentmode:options:resulthandler:).json
- S61. https://developer.apple.com/tutorials/data/documentation/photos/phcachingimagemanager.json
- S62. https://developer.apple.com/tutorials/data/documentation/photos/phimagerequestoptions/isnetworkaccessallowed.json
- S63. https://developer.apple.com/tutorials/data/documentation/photos/phaccesslevel.json
- S64. https://developer.apple.com/tutorials/data/documentation/photos/phauthorizationstatus/limited.json
- S65. https://developer.apple.com/tutorials/data/documentation/photokit/delivering-an-enhanced-privacy-experience-in-your-photos-app.json
- S66. WWDC23 "What's new in privacy." https://developer.apple.com/videos/play/wwdc2023/10053/
- S67. https://developer.apple.com/tutorials/data/documentation/backgroundtasks/bgprocessingtask.json
- S68. https://developer.apple.com/tutorials/data/documentation/backgroundtasks/bgprocessingtaskrequest/requiresexternalpower.json
- S69. https://developer.apple.com/tutorials/data/documentation/backgroundtasks/bgcontinuedprocessingtask.json ; https://developer.apple.com/tutorials/data/documentation/backgroundtasks/performing-long-running-tasks-on-ios-and-ipados.json
- S70. https://developer.apple.com/tutorials/data/documentation/photos/phassetcollectionsubtype.json
- S71. icloudpd releases and CHANGELOG. https://github.com/icloud-photos-downloader/icloud_photos_downloader/releases ; https://raw.githubusercontent.com/icloud-photos-downloader/icloud_photos_downloader/master/CHANGELOG.md
- S72. icloudpd README and maintainer issue. https://github.com/icloud-photos-downloader/icloud_photos_downloader ; https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues/1305
- S73. icloudpd issues. https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues/1366 ; https://github.com/icloud-photos-downloader/icloud_photos_downloader/issues
- S74. https://icloud-photos-downloader.github.io/icloud_photos_downloader/size.html
- S75. https://icloud-photos-downloader.github.io/icloud_photos_downloader/reference.html
- S76. https://icloud-photos-downloader.github.io/icloud_photos_downloader/mode.html

**Part B: Google / Android**
- S80. Google Photos API updates. https://developers.google.com/photos/support/updates ; release notes https://developers.google.com/photos/support/release-notes
- S81. rclone Google Photos backend. https://rclone.org/googlephotos/ ; https://github.com/rclone/rclone/issues/8434
- S82. https://developers.google.com/photos/picker/guides/get-started-picker
- S83. https://developers.google.com/photos/picker/reference/rest/v1/sessions
- S84. https://developers.google.com/photos/picker/guides/media-items
- S85. https://developers.google.com/photos/ambient/guides/about ; https://developers.google.com/photos/ambient/guides/configure-your-app
- S86. https://developers.google.com/photos/partner-program/overview
- S87. https://support.google.com/photos/answer/3024190 ; https://support.google.com/accounts/answer/3024190
- S88. https://kb.uconn.edu/space/IKB/26398359570/Use+Google+Takeout+to+Export+Google+Photos ; https://metadatafixer.com/learn/google-takeout-duplicate-photos-explained
- S89. https://github.com/TheLastGimbus/GooglePhotosTakeoutHelper/issues/353
- S90. https://forum.photostructure.com/t/does-photostructure-ingest-google-photos-takeout-geodata/740
- S91. https://github.com/RhetTbull/osxphotos/issues/2228 ; https://metadatafixer.com/learn/google-takeout-json-files-explained ; https://plutophotos.com/blog/google-takeout-json-files/
- S92. https://github.com/Xentraxx/GooglePhotosTakeoutHelper
- S93. https://github.com/simulot/immich-go ; https://github.com/simulot/immich-go/blob/main/docs/upload-commands-overview.md
- S94. https://developer.android.com/training/data-storage/shared/media
- S95. https://developer.android.com/about/versions/14/changes/partial-photo-video-access
- S96. https://support.google.com/googleplay/android-developer/answer/14115180
- S97. https://support.google.com/googleplay/android-developer/answer/15800983
- S98. https://developer.android.com/training/data-storage/shared/photo-picker
- S99. https://developer.android.com/develop/background-work/background-tasks/persistent

**Part B: On-device Mac**
- S100. https://github.com/Blaizzy/mlx-vlm
- S101. https://github.com/Blaizzy/mlx-vlm/issues/1175
- S102. https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/multimodal.md
- S103. https://github.com/ggml-org/llama.cpp/issues/14527
- S104. https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- S105. https://github.com/immich-app/immich/discussions/10454
- S106. https://mikhalevi.ch/rclip-3-better-and-6x-faster-semantic-image-search/
- S107. https://huggingface.co/tillknuesting/siglip2-so400m-probe-mlx
- S108. FastVLM (Apple). https://arxiv.org/html/2412.13303
- S109. vllm-mlx. https://arxiv.org/html/2601.19139v1
- S110. https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/model_profile.csv
- S111. https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- S112. https://github.com/QwenLM/Qwen3-VL/blob/main/README.md
- S113. https://machinelearning.apple.com/research/exploring-llms-mlx-m5 ; https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- S114. WWDC26 sessions. https://developer.apple.com/videos/play/wwdc2026/241/ ; https://developer.apple.com/videos/play/wwdc2026/237/ ; https://developer.apple.com/videos/play/wwdc2026/326/
- S115. https://developer.apple.com/documentation/vision/vnclassifyimagerequest ; https://gist.github.com/ktustanowski/56c0d7541813868fed4aceb60ab5d149
- S116. https://developer.apple.com/documentation/vision/vngenerateimagefeatureprintrequest
