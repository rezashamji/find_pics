import XCTest
@testable import FindPicsCore

final class NightPassTests: XCTestCase {
    private let wifi = NetworkPath(connected: true, expensive: false, constrained: false)
    private let cellular = NetworkPath(connected: true, expensive: true, constrained: false)

    func testQueueIsNewestFirstAndSkipsDescribedAndFailed() {
        let photos = [NightPassPhoto(id: "old", taken: 100), NightPassPhoto(id: "new", taken: 300),
                      NightPassPhoto(id: "done", taken: 400), NightPassPhoto(id: "bad", taken: 350),
                      NightPassPhoto(id: "mid", taken: 200)]
        let q = nightPassQueue(photos, described: { $0 == "done" }, skipped: ["bad"])
        XCTAssertEqual(q, ["new", "mid", "old"])
    }

    /// Undated photos go last, and ties keep a fixed order so a resumed pass continues where it stopped.
    func testUndatedLastAndStableTies() {
        let photos = [NightPassPhoto(id: "b", taken: nil), NightPassPhoto(id: "z", taken: 5),
                      NightPassPhoto(id: "a", taken: nil), NightPassPhoto(id: "y", taken: 5)]
        XCTAssertEqual(nightPassQueue(photos, described: { _ in false }), ["y", "z", "a", "b"])
    }

    func testEverythingDescribedLeavesNothing() {
        XCTAssertEqual(nightPassQueue([NightPassPhoto(id: "a", taken: 1)], described: { _ in true }), [])
    }

    func testCheckpointByCountOrTimeButNeverEmpty() {
        XCTAssertFalse(nightPassShouldSave(linesSinceSave: 0, secondsSinceSave: 9_999))
        XCTAssertFalse(nightPassShouldSave(linesSinceSave: 199, secondsSinceSave: 299))
        XCTAssertTrue(nightPassShouldSave(linesSinceSave: 200, secondsSinceSave: 1))
        XCTAssertTrue(nightPassShouldSave(linesSinceSave: 3, secondsSinceSave: 300))
    }

    func testRunsOnlyWhenEverythingAllows() {
        XCTAssertNil(nightPassBlock(consented: true, writerOnPhone: false, network: wifi, freeGB: 5,
                                    thermal: .fair, appInUse: false))
        XCTAssertEqual(nightPassBlock(consented: false, writerOnPhone: true, network: wifi, freeGB: 5,
                                      thermal: .nominal, appInUse: false), .noConsent)
        XCTAssertEqual(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 5,
                                      thermal: .nominal, appInUse: true), .appInUse)
        XCTAssertEqual(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 5,
                                      thermal: .critical, appInUse: false), .tooHot)
        XCTAssertNil(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 5,
                                    thermal: .serious, appInUse: false))
        XCTAssertEqual(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 2.2,
                                      thermal: .nominal, appInUse: false), .lowMemory)
    }

    /// The 1.7 GB writer downloads only where indexing may download (Wi-Fi, not Low Data Mode); once it is on the
    /// phone the network no longer matters.
    func testWriterDownloadNeedsWifiOnlyTheFirstTime() {
        XCTAssertEqual(nightPassBlock(consented: true, writerOnPhone: false, network: cellular, freeGB: 5,
                                      thermal: .nominal, appInUse: false), .writerNotDownloadable)
        XCTAssertNil(nightPassBlock(consented: true, writerOnPhone: true, network: cellular, freeGB: 5,
                                    thermal: .nominal, appInUse: false))
    }

    /// os_proc_available_memory reports 0 where it is not supported (the simulator): that is "unknown", not "none".
    func testUnknownMemoryDoesNotBlock() {
        XCTAssertNil(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 0,
                                    thermal: .nominal, appInUse: false))
    }

    func testNote() {
        XCTAssertEqual(nightPassNote(described: 148_451, photos: 148_451), "")
        XCTAssertEqual(nightPassNote(described: 0, photos: 0), "")
        XCTAssertTrue(nightPassNote(described: 1_885, photos: 148_451).contains("1885 of 148451 (1%)"))
    }

    /// 11-23: the phone sat unlocked with the app open all night and every charger grant was refused.
    func testAppLeftOpenAndIdleIsNotBusy() {
        XCTAssertFalse(nightPassAppBusy(onScreen: true, searching: false, secondsSinceUse: 3_600))
        XCTAssertTrue(nightPassAppBusy(onScreen: true, searching: false, secondsSinceUse: 60))
        XCTAssertTrue(nightPassAppBusy(onScreen: false, searching: true, secondsSinceUse: 3_600))
        XCTAssertFalse(nightPassAppBusy(onScreen: false, searching: false, secondsSinceUse: 10))
    }

    /// 11-23 04:18: GPU work from the background crashed the app; the pass must not try.
    func testNeverRunsInTheBackground() {
        XCTAssertEqual(nightPassBlock(consented: true, writerOnPhone: true, network: wifi, freeGB: 5,
                                      thermal: .nominal, appInUse: false, inBackground: true), .noGPUInBackground)
    }

    /// 11-23: pause when warm instead of ending the turn; stop only when critical.
    func testThermalStepPausesWhenWarm() {
        XCTAssertEqual(nightPassThermalStep(.nominal), .judge)
        XCTAssertEqual(nightPassThermalStep(.fair), .judge)
        XCTAssertEqual(nightPassThermalStep(.serious), .wait(seconds: 15))
        XCTAssertEqual(nightPassThermalStep(.critical), .stop)
    }
}
