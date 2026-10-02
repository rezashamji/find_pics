#!/bin/bash
# Builds envs/fp entirely inside find_pics/. Idempotent.
set -e
source /n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/env.sh
cd $FP_ROOT
[ -d envs/fp ] || uv venv --python 3.12 envs/fp
source envs/fp/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate open_clip_torch timm sentencepiece protobuf \
  insightface onnxruntime-gpu opencv-python-headless pillow pillow-heif av \
  numpy pandas pyarrow scikit-learn hdbscan scipy statsmodels \
  fastapi uvicorn jinja2 rich tqdm pydantic pytest exifread huggingface_hub datasets
python -c "import torch, transformers, open_clip, insightface, onnxruntime; print('torch', torch.__version__, 'cuda build', torch.version.cuda); print('ort providers', onnxruntime.get_available_providers())"
echo SETUP_OK
