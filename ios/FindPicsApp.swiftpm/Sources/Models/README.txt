Put the Core ML models here before building (too big for git; they are on the cluster in models/coreml/):
  pe_core_image.mlpackage   <- models/coreml/pe_core_image_PE_Core_B_16_int8.mlpackage   (photo scanner)
  pe_core_text.mlpackage    <- models/coreml/pe_core_text_PE_Core_B_16_int8.mlpackage    (text side of the scanner)
  face_auraface.mlpackage   <- models/coreml/face_auraface.mlpackage                      (faces: fal AuraFace-v1, Apache-2.0,
                               crop + mirror averaged inside the graph; scripts/convert_face_coreml.py)
Do NOT put face_buffalo_l.mlpackage here: InsightFace buffalo_l weights are non-commercial (server dev option only).
Which face model the app expects: FindPicsCore FaceProfile.shipped -> FaceEngine.resource (Faces.swift).
The judge / request reader (Qwen, 4-bit, MLX) downloads once on first use: the app's only network use.
