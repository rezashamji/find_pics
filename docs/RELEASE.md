# Releasing find pics (App Store) — what Apple will ask, and our answers

## Privacy label (App Store Connect -> App Privacy)
- "Do you or your third-party partners collect data from this app?" -> **No** ("Data Not Collected").
  True only while: no analytics/crash SDK that sends data, no server calls with photo-derived data. The only network
  use is the one-time model download (public files from Hugging Face / Apple-hosted assets; nothing about the user is sent).
- Permission texts (already in ios/FindPicsApp.swiftpm/Package.swift):
  - Photos read/write: "find pics searches your photos ON this phone. Nothing is uploaded."
  - Add to album: "find pics can save a search result as a new album (only when you tap Save)."
- Limited photo access: the app searches only the shared photos and cannot create albums (iOS rule); it says so.

## Review notes to paste for App Review
- "All photo analysis runs on the device. The app downloads on-device AI models once (~3 GB) after asking the user,
  with the size shown (guideline 4.2.3). The models are data, not code (guideline 2.5.2). No photo or derived data
  leaves the device. The app never deletes or edits photos; it only creates albums when the user taps Save."

## Blockers before submitting
1. Face model licence: DONE 10-07 (Reza's decision). The app ships fal AuraFace-v1 (Apache-2.0) + flip averaging;
   buffalo_l (non-commercial) is no longer bundled (docs/BUILD_ON_MAC.md deletes it from Sources/Models). Cost:
   fewer of a person's photos found (RESULTS 36). buffalo_l can come back under a paid InsightFace licence: calibrate,
   add its FaceProfile row, set FaceProfile.shipped; the phone re-embeds faces itself (FaceMigration).
   Still open: (a) Self-check's face.png is a DigiFace-1M render (non-commercial research data): remove the face
   check or replace the image with one we may ship before an App Store build; (b) the About screen lists the direct
   dependencies only: add the transitive Swift packages' licences (ios/ATTRIBUTIONS.md); (c) Apache-2.0 covers
   copyright, not biometric-privacy law (BIPA / GDPR apply to on-device face matching whatever the model).
2. Paid Apple Developer Program ($99/year): needed for TestFlight, App Store, Apple-hosted model downloads.
3. Phone measurements (docs/FIRST_DEVICE_TEST.md): memory budget, indexing time, which judge (Qwen / Apple).

## Licences shipped in the app (About screen, AboutView.swift = ios/ATTRIBUTIONS.md)
- Apache-2.0 (full text in the app): AuraFace-v1 (fal; we state our changes: Core ML conversion + mirror averaging,
  section 4(b)), PE-Core (Meta), Qwen3.5 / Qwen3-VL (downloaded), swift-transformers / swift-huggingface.
  NOTICE files: AuraFace-v1's repository has none (file list checked 10-07), so section 4(d) asks nothing more for it.
- MIT (notice in the app): MLX Swift (Copyright (c) 2023 ml-explore), MLX Swift LM (Copyright (c) 2024 ml-explore),
  CLIP BPE vocabulary. CC BY 4.0: GeoNames (attribution line in the app).

## TestFlight (friends try it before the App Store)
1. Join the Apple Developer Program with Reza's Apple ID.
2. In Xcode: Product -> Archive -> Distribute App -> App Store Connect -> Upload.
3. App Store Connect -> TestFlight -> add testers by email (up to 10,000 external testers; external builds get a
   short Beta App Review first).
4. Testers install the TestFlight app and accept the invite. Collect: wrong photos in albums (screenshots), how long the
   first library read took, battery, crashes.

## Supported devices (to decide after the phone test)
Apple Intelligence (for Apple's model) needs iPhone 15 Pro or newer. Our Qwen judge needs ~3 GB of app memory: 8 GB
iPhones (15 Pro / 16 / 17) give apps less than 12 GB ones; measure before promising support.
