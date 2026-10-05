# find_pics iPhone app: design (10-05)

Goal (Reza): "on my phone, click something and type, like the Photos search bar". Everything runs on the phone; photos
never leave it. Target device: iPhone 18 Pro (12 GB RAM, iOS 27). Build: Xcode 27 (needs macOS Tahoe 26.6+).

## Pieces and where each runs
| step | on the phone | status |
|---|---|---|
| read the library | PhotoKit (`PHAsset`, `PHImageManager`), read-only; album writes = create + add only, after a tap | to write |
| image vectors (search by meaning) | PE-Core-B-16 image tower, Core ML (Neural Engine), 94-186 MB | converted, unverified on device |
| text vectors | PE-Core-B-16 text tower, Core ML, 355-709 MB | converted, unverified |
| faces | Apple Vision face detection + ArcFace embedding (Core ML conversion of InsightFace w600k_r50) | to convert |
| planner (words -> plan JSON) | Qwen3.5-4B (4-bit MLX) + distilled planner LoRA adapter (PEFT format: `MLXLMCommon/Adapters/LoRA/PEFTAdapter.swift`) | training |
| judge (yes/no per photo) | Qwen3.5-4B or 9B 4-bit via mlx-swift-lm `MLXVLM/Models/Qwen35.swift`; 9B needs ~5.5 GB (fits the 12 GB phone's ~6 GB default app budget only barely; ~9 GB with the increased-memory entitlement) | to measure on device |
| plan grounding, scope, ranking, pairing, bursts, dates | `ios/FindPicsCore` (Swift, Linux-tested against the Python on golden fixtures) | partly ported |

## Modes
- Fast: image vectors rank everything (milliseconds), the judge checks the top few hundred: answer in seconds-minutes.
- Exhaustive ("more accurate, takes longer"): the judge checks every in-scope photo, in the background while charging;
  results grow as it goes (same streaming rounds as the Python engine).
- First run: index the library once (vectors + faces), in the background while charging, like Apple's own analysis.

## Privacy promise (checkable)
"Your photos never leave your phone. After the one-time model download, the app makes no network requests."
Enforce: the only network code is the model download; no analytics SDKs.

## Open questions to measure on the device
Index speed per 1,000 photos; judge seconds per photo (4B vs 9B); memory peak; battery per 1,000 judged photos;
Apple's on-device Foundation Model (image input since iOS 27) as an alternative judge/planner.
