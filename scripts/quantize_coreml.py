"""8-bit weight quantization of a Core ML package (size for the phone). Linux can build it but not RUN it, so output
fidelity must be checked on macOS/iOS. Usage (envs/coreml + module load gcc/13.2.0-fasrc01):
  python scripts/quantize_coreml.py models/coreml/pe_core_text_PE_Core_B_16.mlpackage"""
import sys
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto

src = Path(sys.argv[1]); dst = src.with_name(src.stem + "_int8.mlpackage")
m = ct.models.MLModel(str(src), skip_model_load=True)
cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=2048))
q = cto.linear_quantize_weights(m, config=cfg)
q.save(str(dst))
size = lambda p: sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
print(f"{src.name}: {size(src):.0f} MB -> {dst.name}: {size(dst):.0f} MB")
