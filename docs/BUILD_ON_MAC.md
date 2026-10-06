# Putting find pics on your iPhone (first time, ~45 min of your time)

What you need: your MacBook Pro (M2 Max), your iPhone 18 Pro and its cable.

## 1. Update the Mac (once, ~1 hour mostly waiting)
Back up first if you normally do. Then System Settings -> General -> Software Update -> the newest macOS (macOS 27; Tahoe 26.6+ also works).
(Apple's rule: Xcode 27 is needed for iOS 27 phones, and it only installs on Tahoe 26.6+.)

## 2. Install Xcode (once)
Mac App Store -> search "Xcode" -> Get. It is big (10+ GB). Open it once and let it install its components.
Xcode -> Settings -> Accounts -> "+" -> Apple ID -> sign in with your normal Apple ID (free is fine for your own phone).

## 3. Get the code and the models onto the Mac (Terminal on the Mac)
    cd ~ && git clone https://github.com/rezashamji/find_pics.git
    mkdir -p ~/find_pics/ios/FindPicsApp.swiftpm/Sources/Models && cd ~/find_pics/ios/FindPicsApp.swiftpm/Sources/Models
    rsync -av --progress rshamji@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/models/coreml/pe_core_image_PE_Core_B_16_int8.mlpackage/ pe_core_image.mlpackage/
    rsync -av --progress rshamji@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/models/coreml/pe_core_text_PE_Core_B_16_int8.mlpackage/ pe_core_text.mlpackage/
    rsync -av --progress rshamji@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/models/coreml/face_buffalo_l.mlpackage/ face_buffalo_l.mlpackage/
    rsync -av rshamji@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/ios/FindPicsApp.swiftpm/Sources/SelfCheck/face.png ../SelfCheck/
    rsync -av --progress rshamji@login.rc.fas.harvard.edu:/n/holylfs06/LABS/mzitnik_lab/Users/rshamji/find_pics/models/planner_4b27ball_mlx_adapter/ planner_adapter/
(The last line, 56 MB, is the trained planner add-on; without it the app still works with the base model.)
(The rsync lines ask for your cluster password + 2-factor code, like before. About 600 MB in total.)

## 4. Open the app in Xcode
Xcode -> File -> Open -> ~/find_pics/ios/FindPicsApp.swiftpm (the folder itself; Xcode opens it as an app project). Wait while Xcode downloads the
packages it needs (MLX) the first time.

## 5. Put it on the phone
1. Plug the iPhone into the Mac; on the phone tap "Trust".
2. iPhone: Settings -> Privacy & Security -> Developer Mode -> On (the phone restarts). If you do not see it, do step 3
   below once and it appears.
3. In Xcode's top bar, pick your iPhone as the run destination, then press the Play button.
4. First time only: Xcode may ask for a "team": pick your name (Personal Team).
5. On the phone: Settings -> General -> VPN & Device Management -> your Apple ID -> Trust.

If Xcode shows red errors: select all the error text (or take a screenshot) and paste it to Claude. Expect a few on the
first build (the app was written on Linux, where Apple's frameworks cannot be compiled).

## 6. First run on the phone
- Allow access to Photos.
- The on-phone AI downloads once (a few GB, on Wi-Fi). After that nothing leaves the phone.
- Tap "Self-check" (top right) and send Claude the numbers: they show whether the phone computes the same as the server.
- Then it reads your library once (keep it plugged in; the first time takes a while), then: type a search.

Note: with a free Apple ID the app stops opening after 7 days; plug in and press Play again to renew.
The face model in this build is for personal/testing use only (its license is non-commercial).
