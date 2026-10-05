Put the Core ML image/text encoders here before building (they are too big for git):
  pe_core_image.mlpackage  <- models/coreml/pe_core_image_PE_Core_B_16_int8.mlpackage
  pe_core_text.mlpackage   <- models/coreml/pe_core_text_PE_Core_B_16_int8.mlpackage
The judge/planner (Qwen3.5 4-bit MLX) downloads once on first launch (the app's only network use).
