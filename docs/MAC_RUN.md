# Running find_pics entirely on a Mac (Apple Silicon)

Status: **written but untested on real Mac hardware.** It was built and tested on Linux GPUs; the Mac-specific pieces
(MPS device for the image/face models, the MLX judge, osxphotos album writing) follow their libraries' documented
APIs but have not been run. Report problems as issues.

What runs where:
| step | Mac backend | expected speed (estimate from research/04) |
|---|---|---|
| image vectors (PE-Core) | PyTorch MPS | a few images/s -> 150k photos in roughly 3-8 h, once |
| faces (InsightFace) | onnxruntime (CPU / CoreML) | similar order |
| planner + judge | MLX, `mlx-community/Qwen3.5-4B-4bit` (set FP_MLX_VLM) | 0.3-1.5 s per judged image |
| albums | osxphotos PhotosAlbum (create + add only) | instant |

Faster option: index on any Linux GPU machine (one-time), copy `fp_index/` to the Mac, and only run `ask` locally.

Steps:
1. Export (read-only): [MAC_EXPORT.md](MAC_EXPORT.md)
2. `uv venv && source .venv/bin/activate && uv pip install -e ".[mac]"`
3. `findpics scan ~/fp_export ~/fp_index --metadata ~/fp_export/library_metadata.json`
4. `findpics index ~/fp_index`
5. `findpics ask ~/fp_index "..." --out ~/fp_albums --me "<your name in Photos>"`. Read the printed plan, open
   `~/fp_albums/index.html`, mark mistakes, then rerun with `--apple-apply` to create the albums in Photos.
