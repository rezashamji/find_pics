# Releasing find pics (App Store) — what Apple will ask, and our answers

## Privacy label (App Store Connect -> App Privacy)
- "Do you or your third-party partners collect data from this app?" -> **No** ("Data Not Collected").
  True only while: no analytics/crash SDK that sends data, no server calls with photo-derived data. The only network
  use is the one-time model download (public files from Hugging Face / Apple-hosted assets; nothing about the user is sent).
- Permission texts (already in ios/FindPicsApp/Package.swift):
  - Photos read/write: "find pics searches your photos ON this phone. Nothing is uploaded."
  - Add to album: "find pics can save a search result as a new album (only when you tap Save)."
- Limited photo access: the app searches only the shared photos and cannot create albums (iOS rule); it says so.

## Review notes to paste for App Review
- "All photo analysis runs on the device. The app downloads on-device AI models once (~3 GB) after asking the user,
  with the size shown (guideline 4.2.3). The models are data, not code (guideline 2.5.2). No photo or derived data
  leaves the device. The app never deletes or edits photos; it only creates albums when the user taps Save."

## Blockers before submitting
1. Face model licence (buffalo_l is non-commercial): buy the InsightFace commercial licence or switch to a
   commercial-OK model (eval/face_commercial.md, eval/face_free_deepdive.md). Reza decides.
2. Paid Apple Developer Program ($99/year): needed for TestFlight, App Store, Apple-hosted model downloads.
3. Phone measurements (docs/FIRST_DEVICE_TEST.md): memory budget, indexing time, which judge (Qwen / Apple).

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
