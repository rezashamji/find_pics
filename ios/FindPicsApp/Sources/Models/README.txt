Put the Core ML models here before building (too big for git; they are on the cluster in models/coreml/):
  pe_core_image.mlpackage   <- models/coreml/pe_core_image_PE_Core_B_16_int8.mlpackage   (photo scanner)
  pe_core_text.mlpackage    <- models/coreml/pe_core_text_PE_Core_B_16_int8.mlpackage    (text side of the scanner)
  face_buffalo_l.mlpackage  <- models/coreml/face_buffalo_l.mlpackage                     (faces; NON-COMMERCIAL weights:
                               personal / dev use only until a shippable face model is chosen)
The judge / request reader (Qwen3.5 4-bit, MLX) downloads once on first launch: the app's only network use.
