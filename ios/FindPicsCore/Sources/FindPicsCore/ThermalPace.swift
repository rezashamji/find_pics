// Thermal pacing of a search AFTER its first answer (RESULTS 44b; Reza's rule 10-10: a cap that drops accuracy is
// wrong, run the long part in parallel / in the background instead).
//
// The first answer (fast mode: ~630 judge calls at most, phoneFastParams) runs flat out and is shown as complete. Then
// the search keeps going by itself, through the same rounds exhaustive mode runs (so every bound it shows is one the
// overclaim replay already checked), but only while the phone is cool enough:
//   .nominal / .fair  -> judge the next photo
//   .serious          -> pause, re-read the state every `poll` seconds (MEASURED 10-10: judging flat out took an
//                        iPhone 18 Pro from 35 to 12-15 photos/min in 15 min, and after 68 min iOS halted charging
//                        and locked the phone)
//   .critical         -> stop judging; the person is told, and searching again picks up from the answers kept.
import Foundation

/// ProcessInfo.ThermalState without Foundation's Apple-only type (same order, same meaning).
public enum ThermalLevel: Int, Sendable, Comparable, CaseIterable {
    case nominal = 0, fair, serious, critical
    public static func < (a: ThermalLevel, b: ThermalLevel) -> Bool { a.rawValue < b.rawValue }
}

public enum PaceStep: Equatable, Sendable {
    case judge
    /// Do not judge now; read the thermal state again after this many seconds.
    case wait(seconds: Double)
    case stop
}

/// The decision for the next photo. `secondsSinceCheck`: how long ago the state was last read; while paused the next
/// read is due `poll` seconds after the last one, so a caller that just read it waits the full poll.
public func thermalPace(_ level: ThermalLevel, secondsSinceCheck: Double, poll: Double = 20) -> PaceStep {
    switch level {
    case .nominal, .fair: return .judge
    case .serious: return .wait(seconds: min(poll, max(1, poll - max(0, secondsSinceCheck))))
    case .critical: return .stop
    }
}

/// thermalPace with bookkeeping for the device log (MAC_INBOX M39): every change of state with its time, how many
/// pauses, and how long they lasted. `now`: seconds on any monotonic clock.
public struct ThermalPacer: Sendable {
    public let poll: Double
    public private(set) var lastLevel: ThermalLevel?
    public private(set) var transitions: [(at: Double, level: ThermalLevel)] = []
    public private(set) var pauses = 0, pausedSeconds = 0.0
    private var lastCheck: Double?, pauseStart: Double?

    public init(poll: Double = 20) { self.poll = poll }

    public mutating func step(_ level: ThermalLevel, now: Double) -> PaceStep {
        if level != lastLevel { transitions.append((now, level)); lastLevel = level }
        let s = thermalPace(level, secondsSinceCheck: lastCheck.map { now - $0 } ?? 0, poll: poll)
        lastCheck = now
        switch s {
        case .wait:
            if pauseStart == nil { pauseStart = now; pauses += 1 }
        case .judge, .stop:
            if let p = pauseStart { pausedSeconds += now - p; pauseStart = nil }
        }
        return s
    }

    /// Seconds of the pause still running (0 when judging).
    public func pausedNow(_ now: Double) -> Double { pauseStart.map { now - $0 } ?? 0 }
}
