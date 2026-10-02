"""Model wrappers. Names are configurable; defaults chosen in research/00_SYNTHESIS.md."""
from __future__ import annotations

import os
import numpy as np
import torch

DEFAULT_CLIP = os.environ.get("FP_CLIP_MODEL", "hf-hub:timm/PE-Core-L-14-336")
DEFAULT_FACE = os.environ.get("FP_FACE_MODEL", "buffalo_l")


def _as_tensor(out):
    """transformers >=5 returns a ModelOutput from get_*_features; older versions return the tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for k in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


class ImageTextEncoder:
    """Image and text towers of a CLIP/SigLIP-style model (HF transformers). Outputs L2-normalized vectors."""

    def __init__(self, name: str = DEFAULT_CLIP, device: str | None = None, dtype=torch.float16):
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.dtype = dtype if self.device != "cpu" else torch.float32
        self.name = name
        self.is_siglip = "siglip" in name.lower()
        self.openclip = name.startswith("hf-hub:")
        if self.openclip:  # e.g. hf-hub:timm/PE-Core-L-14-336 (Meta Perception Encoder via open_clip)
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name)
            self.model = self.model.to(self.device, self.dtype).eval()
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            from transformers import AutoModel, AutoProcessor
            self.model = AutoModel.from_pretrained(name, dtype=self.dtype).to(self.device).eval()
            self.proc = AutoProcessor.from_pretrained(name)

    @torch.inference_mode()
    def images(self, ims) -> np.ndarray:
        if self.openclip:
            x = torch.stack([self.preprocess(im) for im in ims]).to(self.device, self.dtype)
            f = torch.nn.functional.normalize(self.model.encode_image(x).float(), dim=-1)
            return f.cpu().numpy().astype(np.float16)
        x = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        f = _as_tensor(self.model.get_image_features(pixel_values=x))
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy().astype(np.float16)

    @torch.inference_mode()
    def texts(self, texts: list[str]) -> np.ndarray:
        if self.openclip:
            t = self.tokenizer(texts).to(self.device)
            f = torch.nn.functional.normalize(self.model.encode_text(t).float(), dim=-1)
            return f.cpu().numpy().astype(np.float32)
        kw = dict(padding="max_length", max_length=64, truncation=True) if self.is_siglip else dict(padding=True, truncation=True)
        t = self.proc(text=texts, return_tensors="pt", **kw).to(self.device)
        f = _as_tensor(self.model.get_text_features(**t))
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy().astype(np.float32)

    def logit_params(self):
        """SigLIP scores = sigmoid(scale * cos + bias): lets us read cosine as a calibrated-ish probability."""
        s = getattr(self.model, "logit_scale", None)
        b = getattr(self.model, "logit_bias", None)
        return (float(s.exp()) if s is not None else 100.0, float(b) if b is not None else 0.0)


class FaceEncoder:
    """Face detection + identity embedding (InsightFace: SCRFD detector + ArcFace recognizer)."""

    def __init__(self, name: str = DEFAULT_FACE, det_size=(640, 640), gpu_id: int = 0):
        import onnxruntime as ort
        try:  # load CUDA 12 / cuDNN 9 from the nvidia-*-cu12 pip packages so ORT's CUDA provider works next to a cu13 torch
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False)
        except Exception as e:
            print("onnxruntime preload_dlls failed:", e)
        from insightface.app import FaceAnalysis
        root = os.environ.get("INSIGHTFACE_HOME")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name=name, root=root, providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=gpu_id, det_size=det_size, det_thresh=0.5)
        self.providers = {k: m.session.get_providers()[0] for k, m in self.app.models.items()}
        print("insightface providers:", self.providers, flush=True)

    def faces(self, pil_im):
        """-> list of dict(bbox[x1,y1,x2,y2], det_score, emb[512] normalized)."""
        arr = np.asarray(pil_im)[:, :, ::-1]  # RGB->BGR for insightface
        out = []
        for f in self.app.get(arr):
            e = f.normed_embedding.astype(np.float16)
            out.append(dict(bbox=[float(v) for v in f.bbox], det_score=float(f.det_score), emb=e))
        return out
