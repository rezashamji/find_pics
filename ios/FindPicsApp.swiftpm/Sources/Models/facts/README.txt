Index-time facts search models (RESULTS 61d-69). All trained on PUBLIC data only (DISBench TRAIN libraries + public photos).
towervec_encoder_fp16/  33M-param BERT (MiniLM-L12) text encoder, HF/MLX layout; load with MLXEmbedders BertModel and pass
                        tokenTypeIds = zeros (without them cosine drops to .72). Mean pool, no normalisation, 384-d.
towervec_head.fphead    head for photos WITH a facts line (FindPicsCore FactsScorer / FoldedScorerHead).
vecB.fphead             head for photos WITHOUT a facts line yet (policy V, RESULTS 65).
The facts WRITER (Qwen3-VL-2B student, 1.7 GB merged 4-bit MLX) is NOT here: hosting is Reza's decision. For plumbing and
speed use mlx-community/Qwen3-VL-2B-Instruct-4bit (same architecture and speed).
