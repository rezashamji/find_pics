# App Store materials (DRAFTS, 2026-11-22)

Drafts for Reza to edit and approve. Nothing here has been published or entered into App Store Connect. Items marked
OWNER need Reza's account, money or legal judgement.

## Listing

**Name:** find pics
**Subtitle (30 chars max):** Search your photos, privately

**Promotional text (170 max):**
Type what you're looking for in your own words. find pics searches your library on your iPhone, and nothing is uploaded.

**Description:**
find pics searches your photo library the way you'd describe a photo to a friend.

Type "food photos from last summer", "my dog at the beach", "selfies with Mom" or "me looking tired" and find pics
finds those photos and videos. Save any result as an album with one tap.

Private by design
- Every photo is read and searched on your iPhone. Nothing is uploaded, ever.
- find pics never deletes or edits a photo. It only adds albums, and only when you tap Save.

Honest answers
- Each answer says how complete it is, for example "found at least 92%, with 95% confidence", based on a
  random check of the photos it didn't look at closely.

Gets better while you sleep
- While your iPhone charges, find pics quietly describes your photos, so searches get faster and more accurate.

People, pets and things
- Pick a face once and search for that person: "me at the beach", "Dad in a suit".
- Show it a few photos of your pet or a favourite thing, then find every photo of it.

Requirements: iPhone 15 Pro or newer. A one-time download of about 6 GB of on-device AI (Wi-Fi recommended), plus
1.8 GB later on Wi-Fi while charging.

**Keywords (100 chars max, comma-separated):**
photo search,find photos,private,on-device,albums,people,pets,gallery,AI search,organize

**Category:** Photo & Video (primary), Productivity (secondary)
**Age rating:** 4+ (no objectionable content; the app only shows the user's own photos)
**Price:** OWNER (free is simplest: no tax or banking forms)

## App Review notes
Use the text in docs/RELEASE.md ("Review notes to paste for App Review"); it was updated 11-22 for the ~6 GB + 1.8 GB
downloads. Add a demo note: the reviewer's test device needs photos in its library; the app is useful only with a
library of a few hundred photos or more. Expect questions about face recognition and download size.

## App Privacy ("nutrition label")
- Data collection: **Data Not Collected** (no analytics, no crash reporting, no accounts; the only network traffic is
  downloading the AI models from Hugging Face).
- Tracking: none. The privacy manifest (Sources/PrivacyInfo.xcprivacy) declares the same.

## Export compliance
Already answered in the app: ITSAppUsesNonExemptEncryption = NO (HTTPS and Apple system crypto only).

## Privacy policy (needs a public URL - OWNER)
A privacy policy URL is required even when no data is collected. Draft text below; it could be hosted as a GitHub
Pages page or any simple website. The biometric section needs a legal check (BIPA in Illinois, GDPR "special
category" data in the EU) before publishing.

> **find pics privacy policy** (effective [date])
>
> find pics searches the photos and videos on your iPhone. It does this entirely on your device.
>
> **What we collect:** nothing. find pics has no accounts, no analytics, no advertising and no crash reporting.
> Your photos, and everything find pics computes from them, stay on your iPhone and are never sent to us or anyone
> else.
>
> **What find pics stores on your iPhone:** an index of your library (numbers that describe what is in each photo),
> short written descriptions of your photos, and, if you choose to search for a person, numbers that describe their
> face so that other photos of them can be found. This is stored inside the app on your device and is deleted when
> you delete the app.
>
> **Face recognition:** find pics only compares faces when you pick a person to search for. Face data never leaves
> your device and is not used for any other purpose. [LEGAL REVIEW: retention, consent wording, BIPA/GDPR.]
>
> **Network use:** find pics downloads its on-device AI models from Hugging Face (huggingface.co) after asking you.
> Hugging Face receives a normal web request for those files (like visiting a website); no photo or personal data is
> included. If your photos are stored in iCloud, iOS may download them to your iPhone so find pics can read them.
>
> **Your photos are never changed:** find pics never deletes or edits photos. It only creates albums when you tap Save.
>
> **Children:** find pics does not knowingly collect information from anyone, including children under 13.
>
> **Contact:** [support email - OWNER]

## Support URL (OWNER)
Any page with a way to reach Reza (an email address is enough). Could be the same site as the privacy policy.

## Screenshots (OWNER: needs photos that may be shown publicly)
Required: 6.9" iPhone screenshots (1320 x 2868), at least 3. Suggested set, each on a demo library of photos Reza
owns or has rights to (not real friends' faces without consent):
1. A search typed in plain words with results ("dogs at the beach").
2. The completeness line under a result ("found at least 92%...").
3. Saving a result as an album.
4. The privacy message ("Your photos never leave this phone").

## Open-source credits still to add to About (AboutView.swift)
ios/ATTRIBUTIONS.md lists the direct ones. Licences checked 11-22 from the package checkouts:

| Package | Licence |
|---|---|
| mlx-swift, mlx-swift-lm (ml-explore) | MIT |
| swift-transformers, swift-huggingface, swift-jinja (Hugging Face) | Apache-2.0 |
| swift-collections, swift-crypto, swift-asn1, swift-numerics, swift-syntax (Apple) | Apache-2.0 |
| EventSource (Mattt) | MIT |
| yyjson (YaoYuan) | MIT |

swift-syntax only runs at build time (macros) and is not in the shipped binary; listing it is harmless.

## Before submitting
- [ ] OWNER: App Store Connect record (name, bundle id com.rezashamji.findpics, SKU, language).
- [ ] OWNER: privacy policy + support URLs live.
- [ ] OWNER: legal view on face matching (BIPA/GDPR).
- [ ] Replace or remove SelfCheck/face.png (DigiFace-1M, non-commercial) - e.g. a photo of Reza with his OK.
- [x] Add the transitive credits above to AboutView (done 11-23).
- [ ] Minimum iOS and device rule for iPhone 15 Pro or newer (8 GB memory).
- [ ] Screenshots.
- [ ] Archive a STORE build (no FINDPICS_DEV), upload, TestFlight internal test, then submit.
