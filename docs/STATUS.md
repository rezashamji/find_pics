# find pics — where things stand (plain version; updated 2026-10-06 ~04:00)

## What it is
An iPhone app: type what you want ("me heavier vs me fit", "food photos", "my dog Max at the beach") and it finds those
photos and videos, entirely on the phone. Nothing is uploaded. Target: the App Store, for other people.

## How well it works (measured, server version)
- Everyday searches on 24 real people's photo libraries, checked by eye: about 3 in 4 results are right
  (91/132 on 16 new libraries; 80/110 on 8 others). Strong: dogs, cats, bicycles, flowers, boats, cars. Weak: selfies,
  sunsets, food (photos where food is merely present), animal look-alikes (a tiger for "cat").
- Your heavier-vs-fit demo: all 267 heavier photos in "heavier", 116 of 124 fit photos in "fit", none crossed over.
- Understanding requests: phone-size and server models both pass 30 of 30 test conversations (also on the phone's
  compressed weights).
- Selfies on your own sample: 79 of 86 results are real selfies (the 4 wrong ones were video-call screenshots, now
  excluded), and 22 of 23 previously missed selfies are found.
- A specific pet / thing / place from example photos: ~3 in 4 of the top matches are the same one.

## What the phone needs (and status)
| part | status |
|---|---|
| app code (search, people, pets/things, places, follow-ups, Apple-model switch) | BUILDS for iPhone and Simulator (10-06); core logic tested against the server (22 test groups pass); first screens checked in the Simulator |
| photo judge that fits in iPhone memory (~6 GB per app) | 4B model (3 GB) works but misses more than the server's 9B; training it to imitate the 9B did NOT help (same accuracy at equal strictness); testing other small models (Gemma 4 E4B, Qwen3-VL-4B) and Apple's model next |
| request planner | 4B + a 56 MB add-on: 30/30 on the phone's compressed weights |
| Apple's built-in model as an alternative | code written; no probabilities, so the heavier-vs-fit split needs a rating mode; must be tested on the phone |
| face model you may sell | NOT solved: best free option is ~7 points worse; your call (buy licence / free model) |
| App Store paperwork | privacy label, review notes, download consent, icon: done; paid developer account: your call |

## Your next steps
1. Xcode: open `~/find_pics/ios/FindPicsApp.swiftpm`, pick your iPhone, press Run, send me any red errors.
2. On the phone: follow `docs/FIRST_DEVICE_TEST.md` and send me the numbers.
3. Decide when you can: face model licence; paid Apple developer account ($99/yr).
4. When Apple's data copy of your library arrives: tell me (the final exam).
