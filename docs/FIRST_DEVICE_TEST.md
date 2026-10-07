# First test on the iPhone (after the app builds and runs)

Goal: answer the questions the cluster cannot. Each step says what to send back (a screenshot is fine).

## 0. Memory: the app needs the increased-memory-limit entitlement (one-time)
Symptom (seen 10-06 on the iPhone 18 Pro): "This iPhone lets an app use 2.4 GB of memory; find pics needs about
3.6 GB." iOS caps a third-party app's memory below total RAM by default; the 4-bit 4B judge needs ~3.1 GB. The fix is
the entitlement `com.apple.developer.kernel.increased-memory-limit` (8 GB+ phones -> ~6 GB cap).

MEASURED 10-06 22:5x (MAC): the re-sign route does NOT install, and the "unrestricted, no App ID capability" claim
that used to be in this section is WRONG. What actually happens:
- `bash scripts/build_device_entitled.sh` works as written: it builds and the entitlement really is in the signature
  (`codesign -d --entitlements :-` lists it).
- But `xcrun devicectl device install app` then fails with
  `0xe8008015 (A valid provisioning profile for this executable was not found.)` /
  `IXUserPresentableErrorDomain error 14`. Reason: the embedded profile
  ("iOS Team Provisioning Profile: com.rezashamji.findpics", team YYP85AQ2C5) grants only
  `application-identifier`, `keychain-access-groups`, `get-task-allow`, `com.apple.developer.team-identifier`.
  `installd` validates the signature's entitlements AGAINST the profile, so an entitlement that is only in the
  signature makes the install fail. The kernel honouring it at runtime is irrelevant if the app cannot be installed.
- Passing it as a real entitlements file so automatic provisioning would request it also fails, at build time:
  `error: Entitlement com.apple.developer.kernel.increased-memory-limit not found and could not be included in
  profile` — i.e. this team's automatic provisioning will not put the capability in the profile.
So the entitlement needs an Apple-side change (Increased Memory Limit capability on the App ID / a team that may
have it). That is a Reza item, not a shell item; see "MAC NEEDS REZA" at the top of JOURNAL.md.
- Xcode GUI (untried, the remaining candidate): select the app target -> Signing & Capabilities -> "+ Capability" ->
  "Increased Memory Limit", then Run. Worth a try because Xcode registers the capability on the App ID by name
  instead of just copying a key out of an entitlements file.

Without the entitlement: steps 1, 2, 3 below need no judge model and work normally. Any Qwen row in step 4 (and the
follow-up planner in step 7) cannot run, because even ONE 4-bit 4B model (~3.1 GB) exceeds the 2.4 GB cap. Apple's
own model runs out-of-process, so the "Apple 1-10 rating" / "Apple yes-no" rows in step 4 are not affected.

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
Top-left **Model** menu: Qwen / Qwen3-VL photo judge / Two-model vote / Apple 1-10 rating / Apple yes-no. For each model, run:
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

## 8. New photos appear without restarting (2 min)  [added 10-07]
With the app open on the search screen, take 2 photos with the Camera app (or save 2 images), come back. A line
"Adding new photos to the search: N of 2" should appear and go away; then search for what is in them. Delete one of
them in Photos and search again: it must be gone from the results. Send: did both happen, and how long it took.

## 9. Indexing on the charger (overnight)
Leave the phone on the charger, locked, on Wi-Fi overnight with the app in the background (not force-quit). Next
morning open the app: the orange "Not searchable yet" line should have shrunk (or be gone). Send that line before and
after. (Mac: first check the Info.plist keys, docs/MAC_SESSION.md; without them this step cannot work.)

## 10. Photos stored only in iCloud (5 min, if "Optimize iPhone Storage" is on)
The orange line now splits what searches cannot see: "N stored only in iCloud wait for Wi-Fi", "N could not be
downloaded", "N could not be read", and "N are indexed from a smaller copy". On Wi-Fi with the app open, a grey line
"Downloading photos stored only in iCloud to read them: k of N" should count up. Send both lines, on cellular and on
Wi-Fi. Then search for something from an old trip (likely iCloud-only) and check its photos open sharp when tapped.

## 11. "Who is X?" suggests someone (2 min)
Search `photos of me at the beach` (first time, or after deleting the app's data). The sheet should say "Is this
you?" with ONE suggested row of faces on top (the face in most of your front-camera selfies), a Yes button, the other
rows under "No? Tap the right row", and "Add a photo of you". Send: was the suggestion you? Then try `photos of Mom`
(or any name): the suggestion must not be you. Try "Add a photo of them" with 1-3 photos of that person: the search
should run with them.

## 12. A named pet or thing asks for photos (3 min)
Search `my dog Max at the beach` (use your pet's real kind and name, or `my blue car`, `my guitar`). The app should
NOT search for any dog: it should show "Show me Max: pick 1-3 clear photos of Max." Pick 1-3 photos; the same search
runs again and the album says "Best matches first..." and "Only photos that also pass ..." (the beach part). Then
search `Max sleeping`: it must not ask again. Send: the first screen of results (are they YOUR pet / thing?).
