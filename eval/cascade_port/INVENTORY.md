# RESULTS 68 (1): inventory of the index-time-facts search path (shipped R59d cascade + policy V + repeat-stop)

What the phone must do to answer one engine judge call `(question, crop flag, rows, face rows)` the way the measured
harness did (eval/partial65/cascade65.py, variant r59d, config `<cov>_N_V`; RESULTS 59d / 61c / 61d / 65 / 66).
Status column: SWIFT = already in FindPicsCore before tonight; R68 = ported tonight (with its parity test); MAC = needs an
Apple framework, spec only.

## A. Index time (once per photo)

| # | component | kind | what it does | status |
|---|---|---|---|---|
| A1 | facts WRITER (Qwen3-VL-2B student p2b_all, merged 4-bit MLX) | learned, 2B VLM | photo -> one short facts line (~74 tokens) | MAC (RESULTS 61d: MLX checkpoint `eval/facts_writer/models/p2b_all_final_mlx/`, 1.7 GB; repetition penalty 1.05 over prompt + output) |
| A2 | repeat-stop v2 | pure logic | stop decoding at an item's 3rd occurrence inside a `name:` field | SWIFT (FactsLine.swift, 558 golden cases) |
| A3 | facts PREFIX | pure logic | `"place: <address parts without digits, de-duplicated>. taken: <Month YYYY in UTC>. "` + writer line = the text towervec reads (eval/facts_scorer/data.py --meta=1) | R68 (FactsPrefix in CascadeRoute.swift) |
| A4 | WordPiece tokenizer (BERT uncased, all-MiniLM-L12-v2 vocab, 30,522 tokens) | pure logic | facts text -> ids: `[CLS] + ids[:254] + [SEP]` (Tok.single, max_len 256) | R68 (BertTokenizer.swift) |
| A5 | facts TEXT ENCODER = towervec's own fine-tuned MiniLM-L12 (`tvw_t2b_final_rs2_xfit_full/weights.pt`, keys `enc.*`) | learned transformer | BERT, 12 layers, hidden 384, 12 heads, FFN 1,536, GELU (erf), post-LN, 33.36 M params; output = MEAN over all tokens (incl. CLS/SEP) of the last hidden state, NOT normalised, 384-d | R68: exported to HF/MLX safetensors + numerically verified in real MLX (Python); MAC runs it (MLXEmbedders BertModel or Core ML) |
| A6 | stored per photo | data | facts vector f (384, fp16 = 768 B; 144 MB for 187k), has-facts flag, the PE-Core-B-16 image vector v (1024, already stored by the app) | MAC (IndexStore field) |

## B. Query time, once per search

| # | component | kind | what it does | status |
|---|---|---|---|---|
| B1 | scorer question | pure logic | `scorer_question`: drop a `PERSON[..]::` prefix, "the person in the red box" -> "a person", drop " in the red box", strip (QNORM is OFF in every shipped run) | R68 |
| B2 | question vector q for towervec | learned (A5's encoder) | `[CLS] + ids[:64] + [SEP]`, same mean pool | MAC (same model as A5) |
| B3 | question vector t | learned (PE-Core-B-16 text tower) | unit 1024-d; the app already computes it for vector search | SWIFT/MAC (existing) |

## C. Query time, per judge call (all pure logic except C3)

| # | component | kind | rule | status |
|---|---|---|---|---|
| C1 | question tree | pure logic | trip/place questions from place metadata (tolerant names, remainder AND), "A or B" = max, "A and also B" / two sentences = min (R59d a, b) | SWIFT (QuestionTree.swift, 0/660) |
| C2 | leaf route | pure logic | route.py RULES: red-box person, count, text/sign, photo look, colour, person attribute, relation, small/background detail, "or" -> JUDGE; a trip question whose place matches nothing -> JUDGE ("trip-no-place"); a plan's exclude_question -> JUDGE (R59d c2); else SCORER | R68 |
| C3 | millisecond scorer, policy V | learned, small | photo WITH a facts line -> towervec P; WITHOUT -> vecB P (RESULTS 65) | R68 (pure-Swift forward, see D) |
| C4 | judge budget | pure logic | M = min(n, max(50, ceil(5% of n))) over the leaf's in-scope photos n, ordered by millisecond P (stable) | R68 |
| C5 | budget extension | pure logic | judge the next block of M while >= 50% of the last block had judge P >= .7 (R59d c1) | R68 |
| C6 | threshold | pure logic | t = millisecond P of the last judged photo (-inf if the scope is exhausted); every photo of a call with P >= t gets the judge's P, the rest keep the millisecond P; no pass-A scope -> budget over that call's rows, no extension | R68 |
| C7 | leaf scope | pure logic, ENGINE-DEPENDENT | the harness learns each leaf's scope (union of rows the engine asked about it) from a first pass A in which scorer kinds are scored and judge kinds are judged on everything; pass B is the measured cascade | R68 implements both passes' bookkeeping; the app needs a scope source (see RESULTS 68 "what remains") |
| C8 | reference-photo calls (ref_img) | pure logic | judge kind -> judge every row, scorer kind -> millisecond P (no budget) | R68 |
| C9 | the 9B / phone judge | learned VLM | unchanged | SWIFT/MAC (existing) |

## D. The learned models

| model | file | architecture | params | input -> output | phone path |
|---|---|---|---|---|---|
| towervec (`tvw_t2b_final_rs2_xfit_full`) head | weights.pt `head.*` | LN(384) on f and on q; x = [f, q, f*q, 32v, 32t, 32v*32t] (4,224) -> Linear 512 -> GELU(erf) -> Linear 128 -> GELU -> Linear 1 -> sigmoid | 2,230,529 | (f, q, v, t) -> P(yes) | pure Swift (R68). Per search the first layer folds: W1 x = (W_f + W_fq diag(q~)) f~ + (W_v + W_vt diag(32t)) 32v + W_q q~ + W_t 32t + b, so per photo 512 x 1,408 + 512 x 128 + 128 = 0.79 M MAC (0.15 TMAC for 187k photos) |
| towervec encoder | weights.pt `enc.*` | = A5 | 33,360,000 | text -> 384-d | MLX Swift (MLXEmbedders `BertModel` loads the exported HF safetensors as is: its `sanitize` renames HF keys) or Core ML (coremltools conversion runs on Linux, prediction only on Apple) |
| vecB (`vecB_full`) | weights.pt `head.*` | x = [32v, 32t, 32v*32t] (3,072) -> 512 -> GELU -> 128 -> GELU -> 1 -> sigmoid | 1,639,169 | (v, t) -> P(yes) | pure Swift (R68); folds the same way: 512 x 1,024 per photo |
| PE-Core-B-16 text / image towers | models/coreml/pe_core_*_PE_Core_B_16*.mlpackage | existing | existing | existing | existing (vector search) |
| facts writer (2B) | p2b_all_final_mlx | Qwen3-VL-2B 4-bit | ~2 B | photo -> line | MAC (RESULTS 61d) |
TRAIN numbers use fold models (`_h{fold}`); the product ships the `_full` models (trained on all TRAIN libraries).

## E. Numerics to know
- Every measured cascade number (RESULTS 59d-66) scored with bf16 autocast on a GPU. The phone will run fp32 (head) and
  fp16/fp32 (encoder). RESULTS 68 (3) measures how many cascade decisions that changes.
- Judge-match test is `Double(P) >= 0.7` on the Float32 judge answer (Python compares float(float32) with 0.7); a Swift
  `Float >= 0.7` would differ at P = 0.7 exactly (0.7 rounds to 0.699999988 in Float32).
- Threshold ties: every photo with millisecond P >= t is judged, so ties at t enlarge the judged set beyond M.
