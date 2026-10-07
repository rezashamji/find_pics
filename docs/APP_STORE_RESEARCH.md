# App Store research: shipping on-device open models (researched 2026-10-06)

Legend: **[V]** = verified on a primary Apple page (URL given). **[S]** = secondary source (blog, GitHub, forum post by a non-Apple developer). **[U]** = could not verify; treat as a hypothesis to measure.

---

## A. Model delivery

### App size limits
- **Max uncompressed iOS app: 4 GB** (deployment target iOS 9+). Executable `__TEXT` total: 80 MB. [V] https://developer.apple.com/help/app-store-connect/reference/app-uploads/maximum-build-file-sizes
  - Same page: "Use Background Assets to host larger assets"; Apple-hosted asset pack sizes are shown separately on the product page as an "Up to" size.
- **Cellular download limit: 200 MB default prompt, user-overridable.** Settings > App Store > App Downloads: "Always Allow" / "Ask If Over 200 MB" / "Always Ask" (since iOS 13). [V-ish] Apple Support user guide https://support.apple.com/en-sa/guide/iphone/iph3dfd91de/ios ; history [S] https://www.macrumors.com/how-to/download-large-apps-over-cellular-iphone-ipad/
  - **[U]** Whether Apple-hosted asset packs obey the same 200 MB cellular prompt: not stated in any Apple doc I found.

### Background Assets / Apple-Hosted Background Assets (the official path for multi-GB weights)
- **Apple hosts it, included in the Developer Program membership (no extra cost).** "Apple-Hosted Background Assets hosts up to 200GB of compressed assets ... You can use the service to host ... machine learning models ..." [V] https://developer.apple.com/documentation/backgroundassets/downloading-apple-hosted-asset-packs
- **Limits: 200 GB asset-pack total per app record, 200 asset packs.** Total = sum over packs of each pack's max size across live/TestFlight versions; email warning at 80 %. [V] https://developer.apple.com/help/app-store-connect/reference/app-uploads/apple-hosted-asset-pack-size-limits
  - **[U]** No per-pack size limit is documented (a third-party summary said "100 packs"; Apple's page says 200).
- **OS: iOS 26+** for Apple-hosted packs; localized packs iOS 27+. [V] https://developer.apple.com/help/app-store-connect/manage-asset-packs/overview-of-apple-hosted-asset-packs
- **Download policies** (per pack, in manifest): `essential` (downloaded during install, counts in App Store progress bar, ready before first launch; can be first-install only), `prefetch` (starts at install, may finish after), `onDemand` (only when you call the API). API: `AssetPackManager.shared.ensureLocalAvailability(of:)`, `statusUpdates(forAssetPackWithID:)`, `contents(at:)`, `descriptor(for:)` (file descriptor, i.e. mmap-able), `remove(assetPackWithID:)`. Needs a downloader app extension + shared App Group. [V] WWDC25 "Discover Apple-Hosted Background Assets" https://developer.apple.com/videos/play/wwdc2025/325/ and the docs page above.
  - Apple note: even `essential` packs can be missing on first launch after network dropouts; call `ensureLocalAvailability` anyway. [V]
- **App Review:** "Asset packs must be reviewed before they can be tested externally in TestFlight or made available for users on the App Store." After the app is approved, new pack versions can ship **without a new app build**. [V] https://developer.apple.com/help/app-store-connect/manage-submissions-to-app-review/submit-apple-hosted-asset-packs
- Self-hosting via Background Assets is also supported (`BAManifestURL`, `ManagedDownloaderExtension`). [V] https://developer.apple.com/documentation/backgroundassets
- Precedent: Apple itself told developers to ship Foundation Models LoRA adapters (~160 MB each) via Background Assets, not the bundle. [V] https://developer.apple.com/apple-intelligence/foundation-models-adapter/

### On-Demand Resources
- **Deprecated as of iOS 27** ("support will be removed in future releases. Migrating to Background Assets is recommended"). Limits on iOS 18+: 8 GB per pack, 70 GB hosted, 4 GB bundle. [V] https://developer.apple.com/help/app-store-connect/reference/app-uploads/on-demand-resources-size-limits
- Conclusion: do not build on ODR.

### Is downloading multi-GB model weights after install allowed?
- **Guideline 2.5.2**: apps "may not ... download, install, or execute code which introduces or changes features or functionality of the app". [V] https://developer.apple.com/app-store/review/guidelines/#2.5.2
  - Model weights are data, not code, and Apple's own docs list "machine learning models" as an intended Apple-hosted asset type [V, link above]. **[U]** No guideline text says "ML weights are not code" explicitly; the reasoning risk is if a downloaded model *adds a feature the reviewed app did not have*. Keep the feature set fixed at review time; weights only change quality.
- **Guideline 4.2.3(ii)**: "If your app needs to download additional resources in order to function on initial launch, disclose the size of the download and prompt users before doing so." [V] same page. -> Need an in-app size disclosure + consent screen before any non-essential download (essential packs download with the install itself, so the App Store "Up to" size is the disclosure).
- **Precedent (shipping App Store apps that download GB-scale weights from Hugging Face):**
  - **PocketPal AI** (App Store id6502579498, open source): downloads GGUF files directly from `https://huggingface.co/${hfModelId}/resolve/main/${filename}`; HF token only sent to `huggingface.co`. Its iOS entitlements include `increased-memory-limit` and `extended-virtual-addressing`. [V, read the code] https://github.com/a-ghorbani/pocketpal-ai (`src/config/urls.ts`, `src/services/deviceRules/parse.ts`, `ios/PocketPal/PocketPal.entitlements`)
  - **Locally AI** (now "Locally AI by LM Studio", acquired Apr 2026): MLX models from the `mlx-community` org on Hugging Face. [S] https://simonwillison.net/2025/Sep/21/locally-ai/ , https://apps.apple.com/us/app/locally-ai-by-lm-studio/id6741426692
  - **Private LLM**: **[U]** could not find where it hosts models.
  - Rules about HF hosting: none in Apple guidelines specifically. Practical constraints: model licence must allow redistribution/use; HF rate limits and gated repos (needs token) are a reliability risk; Apple-hosted packs avoid both and are reviewed once.

---

## B. Memory

### Entitlements (`com.apple.developer.kernel.increased-memory-limit`, `...extended-virtual-addressing`)
- Apple's capability table (columns: ADP = paid Developer Program, ADEP = Enterprise, "Apple Developer" = free account): **Extended Virtual Addressing: ADP yes, ADEP yes, free NO. "Increased Debugging Memory Limit": ADP, ADEP, free NO.** The plain "Increased Memory Limit" capability **is not listed in the table at all.** [V, parsed the table] https://developer.apple.com/help/account/reference/supported-capabilities-ios
- Contradicting field report: an Xcode **Personal Team** (free) build kept `increased-memory-limit` and got ~6 GB vs ~3.3 GB without it, on iPhone 15 Pro Max, iOS 27.0. [S] https://github.com/SideStore/SideStore/issues/1616
- **Net:** paid program = both entitlements available. Free team = increased-memory-limit *reportedly* works, extended-virtual-addressing officially not. **[U]** -> test on Reza's own device. (Irrelevant for App Store distribution, which requires the paid program anyway.)
- Apple's wording on increased-memory-limit: "only available on some device models. Call `os_proc_available_memory` to determine the amount of memory available"; app must behave correctly if the extra memory isn't granted. [V] https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.increased-memory-limit
- Extended virtual addressing = larger address space (for mmap of large files), not more RAM. [V] https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.kernel.extended-virtual-addressing

### Measured per-app limits (no Apple-published table exists)
| Device (RAM) | Without entitlement | With increased-memory-limit | Source |
|---|---|---|---|
| iPhone 15 Pro Max (8 GB), iOS 27 | ~3.3 GB | ~6 GB | [S] SideStore #1616 |
| "iPhone 17 Pro" (article says 8 GB; Apple spec is 12 GB, so device/RAM is doubtful) | Gemma E4B load crashed | load OK, peak 4.65 GB (EVA alone also OK, 4.78 GB) | [S] https://zenn.dev/mtfum/articles/ios_memory_entitlements?locale=en |
| iPhone 16 Pro Max (8 GB) | — | ~6144 MB (os_proc_available_memory + phys_footprint) | [U] search snippet, primary not found |
| 12 GB iPhones | **[U]** no reliable number found | **[U]** | — |
- iOS 18+ adds jetsam reason `vm-compressor-space-shortage`: compressed memory counts too; a 16 GB iPad gave an app ~13 GiB of real footprint. [V, Apple DTS reply] https://developer.apple.com/forums/thread/777370
- **Mechanism to remember:** `os_proc_available_memory()` returns bytes left before *this process* hits its jetsam limit (not device free RAM). Limit ≈ `os_proc_available_memory() + current phys_footprint`. Read-only mmapped weights are "clean" pages and are cheaper than dirty heap; Apple: "Memory that the app allocates at runtime doesn't initially contribute ... When the app writes to the allocated memory, it becomes dirty". [V] https://developer.apple.com/documentation/xcode/reducing-your-app-s-memory-use
- Action: log `os_proc_available_memory()` at launch on the iPhone 18 Pro (12 GB) with/without entitlement; that is the only trustworthy number for our device.

---

## C. Apple Foundation Models framework (iOS 27, WWDC26)

### Image input: **YES (iOS 27+)**
- `Attachment` struct, introduced iOS 27. Initializers: `init(_:orientation:)` (CGImage), `init(imageURL:orientation:)`, `init(_:)` (data content), plus `.label(_:)`. Accepts CGImage, CIImage, CVPixelBuffer, image file URLs (WWDC26 also lists UIImage/NSImage). [V] https://developer.apple.com/documentation/foundationmodels/attachment
- Exact usage from Apple's article:
  ```swift
  let response = try await session.respond(
      generating: ImageLabel.self,
      options: GenerationOptions(samplingMode: .greedy)
  ) {
      "Choose the label that best represents the following image:"
      Attachment(image)
  }
  ```
  "The framework performs the necessary scaling and color conversions before passing an image to the model." [V] https://developer.apple.com/documentation/foundationmodels/analyzing-images-with-multimodal-prompting
- "Arbitrary image sizes are allowed, but ... larger images will consume more tokens and incur more latency." **No tokens-per-image number published.** [V] https://developer.apple.com/videos/play/wwdc2026/241/ -> measure with `SystemLanguageModel.tokenCount(for:)` / `response.usage`.
- Tools can receive an `ImageReference` argument and resolve it in the transcript; built-in Vision tools `OCRTool`, `BarcodeReaderTool`. [V] same article.

### Probabilities / logprobs: **NO**
- `LanguageModelSession.Response` exposes only `content`, `rawContent`, `transcriptEntries`, `usage` (token counts). [V] https://developer.apple.com/documentation/foundationmodels/languagemodelsession/response
- `GenerationOptions` = `samplingMode` (`.greedy`, `.random(top:seed:)`, `.random(probabilityThreshold:seed:)`), `temperature`, `maximumResponseTokens`, `toolCallingMode`. Nothing returns a probability. [V] https://developer.apple.com/documentation/foundationmodels/generationoptions
- `LanguageModelSession.Usage` = token counts only (input total/cached, output total/reasoning). [V] WWDC26 241.
- **Consequence for ranking by P(yes):** the API gives a hard label, not a score. Only indirect estimates are possible (e.g. repeated random-sampling frequency, or asking for a numeric `@Guide(.range(...))` rating) — both are not calibrated probabilities. **[U]** whether the new `LanguageModelExecutor` path (bring-your-own model: `CoreAILanguageModel`, `MLXLanguageModel`) could surface logprobs: no field for it in the documented executor/channel types; for an open model you'd just run it directly.

### Guided generation (`@Generable`)
- Uses **constrained sampling**: "Constrained sampling prevents the model from producing malformed output". Supports Bool, Int, Float, Double, Decimal, Array, String, enums, nested structs; `DynamicGenerationSchema` for runtime schemas. [V] https://developer.apple.com/documentation/foundationmodels/generating-swift-data-structures-with-guided-generation
- `@Guide` constraints (`GenerationGuide`): `anyOf`, `constant`, `count`, `element`, `minimum`, `maximum`, `minimumCount`, `maximumCount`, `pattern` (regex), `range`. [V] https://developer.apple.com/documentation/foundationmodels/generationguide
- Schema text is sent to the model and costs context tokens (`includeSchemaInPrompt:` can turn it off). [V] https://developer.apple.com/documentation/foundationmodels/managing-the-context-window
- Apple tip: use `.greedy` for classification so it always picks the most likely label. [V] multimodal article.

### Context window: **conflicting, 4,096 vs 8,192**
- Docs article: "Apple's on-device foundation model has a context window of 4096 tokens per session". [V] https://developer.apple.com/documentation/foundationmodels/managing-the-context-window
- WWDC26 session 241 code sample: `print(model.contextSize) // 8192`. [V] https://developer.apple.com/videos/play/wwdc2026/241/
- Private Cloud Compute model: 32K (off-device; not usable under our on-device rule). [V] WWDC26 241.
- -> Read `SystemLanguageModel.default.contextSize` at runtime (iOS 26.4+). Images count against it. Session throws `LanguageModelError.contextSizeExceeded`; one fresh session per photo avoids accumulation.

### Models / devices
- iOS 27 on-device variants: `core3` ("AFM 3 Core") and `coreAdvanced3` ("AFM 3 Core Advanced"), via `SystemLanguageModel.variant`. [V] https://developer.apple.com/documentation/foundationmodels/systemlanguagemodel/variant
- AFM 3 Core = ~3B dense; Core Advanced = 20B sparse, 1-4B active, "natively multimodal". [V] https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models . **[U]** which iPhones get Core Advanced (Apple storage note: "Up to 14 GB for iPhone 17 Pro and iPhone 17 Pro Max" vs 8 GB others suggests Pro models get the bigger one).
- Apple Intelligence on iOS 27: iPhone 15 Pro / 15 Pro Max and iPhone 16 models or later. [V] https://support.apple.com/en-us/121115 . Apple Intelligence must be enabled; check `SystemLanguageModel.default.availability`.
- **[U]** whether image input works on every Apple-Intelligence device or only some variants.

### Rate limits / background
- Apple (WWDC25 group lab summary, posted by Apple on the forums): **foreground: no rate limit** unless the device is under heavy load (camera, game mode); **background: a budget; exceeding it throws a rate-limited error**; throughput varies with thermals/battery. [V] https://developer.apple.com/forums/thread/791086
- One request at a time per `LanguageModelSession`. [S]
- `BGContinuedProcessingTask` (iOS 26) lets a user-started job continue in background, but background **GPU** (`com.apple.developer.background-tasks.continued-processing.gpu`, paid program only) is reported unsupported on all iPhones (iPads M3+ only); no Apple reply. [V entitlement page] https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.background-tasks.continued-processing.gpu ; [S] https://developer.apple.com/forums/thread/816774
- Practical: plan the exhaustive scan as a foreground job (screen on) with resume-from-checkpoint.

### Custom adapters (LoRA): **effectively NO on iOS 27**
- Apple's toolkit page: "Version 26.0.0 is the last release of this toolkit and is not compatible with macOS, iOS, iPadOS, or visionOS 27 and later." Deployment needed the Foundation Models Adapter entitlement requested by the Account Holder of a paid membership; ~160 MB per adapter; one adapter per exact base-model version. [V] https://developer.apple.com/apple-intelligence/foundation-models-adapter/
- The iOS 27 `SystemLanguageModel` docs no longer list `Adapter` / `init(adapter:)` (the doc URLs for them now fail). [V, by absence]
- iOS 27 alternative: wrap an open model you export yourself (`CoreAILanguageModel`, `MLXLanguageModel`) behind the same `LanguageModelSession` API. [V] WWDC26 241, https://developer.apple.com/documentation/foundationmodels/updates/foundationmodels

---

## D. App Store privacy

### "Data Not Collected"
- "'Collect' refers to transmitting data off the device in a way that allows you and/or your third-party partners to access it for a period longer than what is necessary to service the transmitted request in real time." Data processed only on device is not collected. "If you derive anything from that data and send it off device, the resulting data should be considered separately." [V] https://developer.apple.com/app-store/app-privacy-details/
- Requirements to honestly claim it: no photo/embedding/thumbnail/result/query leaves the device; **no third-party SDK that sends anything** (you are responsible for SDKs' practices); no crash/analytics SDK of your own (Crash Data and Performance Data are disclosable types). Apple's own opt-in crash sharing via App Store Connect is generally treated as not collected by the developer **[U]**. Downloading model weights from Apple/HF sends only a request (IP used transiently) -> falls under the real-time exception [V, same page]. Private Cloud Compute FM calls would be off-device processing: **[U]** how Apple wants that labelled; avoid it.
- Guideline 5.1.2(vi): data from "Photo APIs" / facial mapping "may not be used for marketing, advertising or use-based data mining". [V] https://developer.apple.com/app-store/review/guidelines/#5.1.2

### Info.plist keys
- `NSPhotoLibraryUsageDescription`: required for any API with **read or read/write** access (fetching assets, creating albums). [V] https://developer.apple.com/documentation/bundleresources/information-property-list/nsphotolibraryusagedescription
- `NSPhotoLibraryAddUsageDescription`: for **add-only** access (`PHAccessLevel.addOnly`). [V] https://developer.apple.com/documentation/bundleresources/information-property-list/nsphotolibraryaddusagedescription
- Creating an album = `PHAssetCollectionChangeRequest.creationRequestForAssetCollection(withTitle:)` inside a `performChanges` block; adding = `init(for:)` + `addAssets`. This needs `.readWrite` authorization (add-only cannot fetch or create albums). [V for API] https://developer.apple.com/documentation/photos/phassetcollectionchangerequest ; [U] explicit "add-only can't create albums" sentence not found, but you can't fetch existing assets to add without read access.
- `PHPhotoLibraryPreventAutomaticLimitedAccessAlert` = YES to suppress the once-per-launch "select more photos" prompt; present `PHPhotoLibrary.shared().presentLimitedLibraryPicker(from:)` from your own button instead. [V] https://developer.apple.com/documentation/photos/phphotolibrary/presentlimitedlibrarypicker(from:)
- Missing usage string = crash on access. [V] next link.

### Limited library behaviour
- User can pick "Select Photos"; status is `.limited` from `authorizationStatus(for: .readWrite)` (the old no-argument API reports `.authorized` even when limited). Fetches return only the selected assets. **"You can't create or fetch user albums"** in limited mode; assets your app creates are auto-added to the selection. Observe changes with `PHPhotoLibraryChangeObserver`. [V] https://developer.apple.com/documentation/photokit/delivering-an-enhanced-privacy-experience-in-your-photos-app
- Consequence: in limited mode the product can search only the selected subset and cannot write the result album; the UI must say so.

## D. Licences of the bundled models (10-07)

- **Face recognizer: fal AuraFace-v1, Apache-2.0** (Reza's decision 10-07; replaces InsightFace buffalo_l, whose
  weights are non-commercial). [V] model card https://huggingface.co/fal/AuraFace-v1 (`license: apache-2.0`; "trained
  on commercially and publicly available data sources to enable its usage in commercial setting"; usage is the standard
  insightface pipeline: SCRFD detection, 5-point norm_crop to 112x112, ArcFace-style recognizer, RGB, (x-127.5)/127.5).
  The repository's files (fetched tree, 10-07): .gitattributes, LICENSE.md, README.md and the ONNX models; **no NOTICE
  file**, so Apache-2.0 section 4(d) adds nothing; sections 4(a)-(c) need the licence text in the app and a statement of
  our changes (Core ML conversion, mirror averaging): both are in the About screen.
- What Apache-2.0 does NOT settle: the vendor's training-data claim is not auditable, and biometric-privacy law
  (BIPA, GDPR art. 9) applies to face matching on the device regardless of the model's licence. [U] whether App Review
  asks about face recognition specifically; the privacy label stays "Data Not Collected" because nothing leaves the phone.
- Quality cost, measured (RESULTS 36, CelebA, people.py exactly, equal wrong items): see eval/face_free_deepdive.md and
  RESULTS 36 for the recalibrated cuts.
