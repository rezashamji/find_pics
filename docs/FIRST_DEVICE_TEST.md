# First test on the iPhone (after the app builds and runs)

Goal: answer the questions the cluster cannot. Each step says what to send back (a screenshot is fine).

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
