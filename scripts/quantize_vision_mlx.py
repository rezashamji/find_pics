"""Make the phone judge smaller: quantize the VISION TOWER of an mlx-community Qwen3-VL 4-bit checkpoint (which ships it
in bf16), leaving every language-model tensor byte-identical. Run on the Mac (needs `pip install mlx`), not the cluster.
Why a script and not mlx_vlm.convert: convert always skips vision modules (mlx_vlm.utils.skip_multimodal_module), and
re-converting from the HF weights would also re-quantize the language model. mlx-swift-lm (3.32.x) loads the result:
Load.swift quantizes every Linear whose "<path>.scales" is present, with bits/group from config "quantization"
per-layer entries (BaseConfiguration.PerLayerQuantization).
Layers quantized: vision_tower.blocks.N.attn.{qkv,proj}, .mlp.linear_fc{1,2}, merger.linear_fc{1,2},
deepstack_merger_list.N.linear_fc{1,2} (411 M weights in the 4B). patch_embed (Conv3d) and pos_embed stay bf16.
Usage: python scripts/quantize_vision_mlx.py <mlx-community 4-bit dir> <out dir> [--bits 4] [--group 64]
       e.g. huggingface-cli download mlx-community/Qwen3-VL-4B-Instruct-4bit --local-dir q3vl4b
            python scripts/quantize_vision_mlx.py q3vl4b q3vl4b_vis4 --bits 4
Check: the printed sizes; then python -m mlx_vlm.generate --model q3vl4b_vis4 --image <a photo> --prompt "Describe."
"""
import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

SUFFIX = (".attn.qkv.weight", ".attn.proj.weight", ".mlp.linear_fc1.weight", ".mlp.linear_fc2.weight",
          ".linear_fc1.weight", ".linear_fc2.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=4); ap.add_argument("--group", type=int, default=64)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(src / "model.safetensors.index.json"))
    wmap = dict(idx["weight_map"])
    # mlx-community/Qwen3-VL-4B-Instruct-4bit ships ONE consolidated model.safetensors but an index that still
    # names two shards, so the index's file list does not exist on disk (MAC 10-10). Use the files that are
    # actually there; the weight names are the same either way.
    if any(not (src / f).exists() for f in set(wmap.values())):
        present = sorted(f.name for f in src.glob("*.safetensors"))
        if len(present) != 1:
            raise SystemExit(f"index names {sorted(set(wmap.values()))} but {present} are on disk")
        wmap = {k: present[0] for k in wmap}
        print(f"index named missing shards; using {present[0]}", flush=True)
    cfg = json.load(open(src / "config.json"))
    q = dict(cfg["quantization"])
    before = after = 0
    layers = []
    for f in sorted({v for v in wmap.values()}):
        w = mx.load(str(src / f))
        new = {}
        for k, v in w.items():
            if k.startswith("vision_tower.") and k.endswith(SUFFIX) and v.ndim == 2 and v.shape[-1] % a.group == 0:
                wq, s, b = mx.quantize(v, group_size=a.group, bits=a.bits, mode="affine")
                p = k[:-len(".weight")]
                new[k], new[p + ".scales"], new[p + ".biases"] = wq, s, b
                wmap[p + ".scales"] = wmap[p + ".biases"] = f
                q[p] = {"group_size": a.group, "bits": a.bits}
                layers.append(p)
                before += v.nbytes; after += wq.nbytes + s.nbytes + b.nbytes
            else:
                new[k] = v
        mx.save_safetensors(str(out / f), new, metadata={"format": "mlx"})
        print(f, "done", flush=True)
    cfg["quantization"] = q; cfg["quantization_config"] = q
    json.dump(cfg, open(out / "config.json", "w"), indent=4)
    idx["weight_map"] = dict(sorted(wmap.items()))
    idx.setdefault("metadata", {})["total_size"] = sum((out / f).stat().st_size for f in set(wmap.values()))
    json.dump(idx, open(out / "model.safetensors.index.json", "w"), indent=4)
    for f in src.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name not in ("config.json",
                                                                               "model.safetensors.index.json"):
            shutil.copy(f, out / f.name)
    print(f"quantized {len(layers)} vision layers to {a.bits}-bit g{a.group}: {before / 1e9:.3f} GB -> "
          f"{after / 1e9:.3f} GB (saved {(before - after) / 1e9:.3f} GB); "
          f"checkpoint now {sum(p.stat().st_size for p in out.glob('*.safetensors')) / 1e9:.3f} GB")


if __name__ == "__main__":
    main()
