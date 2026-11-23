// Developer tooling switch (release cleanup 11-22).
//
// ON in Debug builds and in builds made by scripts/build_device_entitled.sh, which passes -D FINDPICS_DEV: every
// measurement hook the Mac relies on keeps working there (Documents/next_query.txt and its #commands, -runQueries,
// -timeIndex, Documents/memcheck.txt, the Model menu, Self-check, the food-notion button).
// OFF in any other Release build - an Xcode archive for TestFlight / the App Store - so a user never sees a raw
// engine menu or a debug screen, and the app never grows a log file in their Documents.
// To try the store build on the phone: FINDPICS_STORE=1 bash scripts/build_device_entitled.sh
enum DevTools {
    #if DEBUG || FINDPICS_DEV
    static let enabled = true
    #else
    static let enabled = false
    #endif
}
