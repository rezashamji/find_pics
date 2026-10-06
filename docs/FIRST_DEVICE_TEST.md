# First test on the iPhone (after the app builds and runs)

Goal: answer the questions the cluster cannot. Each step says what to send back (a screenshot is fine).

## 0. Memory: the app needs the increased-memory-limit entitlement (one-time)
Symptom (seen 10-06 on the iPhone 18 Pro): "This iPhone lets an app use 2.4 GB of memory; find pics needs about
3.6 GB." iOS caps a third-party app's memory below total RAM by default; the 4-bit 4B judge needs ~3.1 GB. The fix is
the entitlement `com.apple.developer.kernel.increased-memory-limit` (unrestricted: works with a free Personal Team,
no App ID capability; 8 GB+ phones -> ~6 GB cap). Two ways to get it in:
- Command line, no GUI (what the Mac loop uses): `bash scripts/build_device_entitled.sh` builds for device and
  re-signs the .app with the entitlement (Xcode's automatic signing can't add it via a raw entitlements file, and
  AppleProductTypes has no capability for it, so we re-sign; the kernel honours it from the signature). Then install
  on a connected, unlocked, trusted iPhone with `xcrun devicectl device install app --device <UDID> "<path>"`.
- Xcode GUI: select the app target -> Signing & Capabilities -> "+ Capability" -> "Increased Memory Limit", then Run.

## 1. Does the phone compute the same as the server? (2 min)
Tap **Self-check** (top right). Send the screen: image, text and face vectors vs the server's (cosine near 1.0 = same).

## 2. How much memory does an app get on your iPhone 18 Pro? (1 min)
The Self-check screen will show "app memory available" (os_proc_available_memory). Send the number.
Why: our best photo judge (the 9B) needs ~6 GB of weights; other 12 GB iPhones give apps ~6.1 GB. This number decides
whether the 9B can ever run on the phone or the phone uses the 4B / Apple's model.

## 3. First library read (leave it plugged in)
The progress line shows "Reading your library once: N of M". Note the time it started and finished, and the battery
level before/after, and whether an orange line says "N photos could not be read" (send N). Why: nobody has measured
how long the one-time indexing takes on a real phone, and photos kept only in iCloud may be invisible to the search.

## 4. Same searches, three models (10 min)
Top-left **Model** menu: Qwen / Apple 1-10 rating / Apple yes-no. For each model, run:
- `food photos`
- `selfies`
- `photos of me looking heavier vs photos of me looking fit`
Send: the counts per album, how long each took, and a screenshot of the first screen of results.
Why: Apple's model gives no probabilities. On the cluster, a yes/no-only judge broke the heavier/fit split
(53 heavier photos landed in "fit"); a 1-10 rating mostly worked (32). The phone test shows what Apple's model does.

## 5. Anything that looked wrong
Screenshot it. Wrong photos in an album are the most useful thing you can send.

## 6. A specific pet, person-free thing or place (3 min)
Bottom bar: **Find a specific pet or thing…** -> pick 1-3 photos of it (e.g. your car, a pet, your building) -> name
and what it is -> Find it. Send: how many it found and a screenshot of the first results (are they the same one?).

## 7. Follow-ups (2 min)
Search `all my photos with bread`, then type `drop the sandwiches and burgers`, then `actually keep the sandwiches`.
Send a screenshot of the plan note at the top after each message (it should end with only burgers excluded).
