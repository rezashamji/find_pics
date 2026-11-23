# Third-party data and models in the app
- GeoNames city and country tables (rg_cities1000.csv via reverse_geocoder; countries.tsv): CC BY 4.0, https://www.geonames.org/
- CLIP BPE vocabulary (bpe_simple_vocab_16e6.txt): MIT, OpenAI CLIP / open_clip.
- PE-Core (Meta Perception Encoder) image/text towers: Apache-2.0.
- Qwen3.5 and Qwen3-VL (planner / photo judge, downloaded on first use): Apache-2.0.
- Face recognizer AuraFace-v1 (fal, https://huggingface.co/fal/AuraFace-v1, glintr100): Apache-2.0. Modified by
  find pics: converted from ONNX to Core ML (fp16), and the graph embeds each face and its mirror image and averages
  them. The model repository ships no NOTICE file (checked 10-07), so there is no NOTICE text to carry. The vendor says
  it was trained on "a commercial dataset"; that claim is not auditable (eval/face_free_deepdive.md).
- InsightFace buffalo_l (w600k_r50, NON-COMMERCIAL): NOT in the app since 10-07; server dev option only.
- swift-transformers, swift-huggingface (Hugging Face): Apache-2.0.
- Transitive Swift packages (Package.resolved; checked 11-23 in the checkouts): swift-jinja, swift-crypto (NOTICE:
  "Copyright 2019 The SwiftCrypto Project"), swift-asn1 (NOTICE: "Copyright 2022 The SwiftASN1 Project"),
  swift-collections, swift-numerics: Apache-2.0. EventSource (Copyright 2025 Mattt), yyjson (Copyright (c) 2020
  YaoYuan): MIT. swift-syntax (Apache-2.0) runs only at build time (macros) and is not in the app.
- Shown in the app: About (AboutView.swift) lists all of these and carries the Apache-2.0 text and the MIT notice.
  Keep both lists in sync.
- MLX, mlx-swift and mlx-swift-lm (on-device model runtime): MIT, Apple ml-explore.
- Apple Vision (face detection) and Core ML: system frameworks, no bundled third-party code.
