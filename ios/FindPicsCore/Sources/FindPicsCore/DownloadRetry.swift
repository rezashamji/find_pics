// The one-time model download must survive a flaky connection (MAC 10-10 02:40: the 2.83 GB judge took 16 manual
// restarts, every failure NSURLErrorDomain -1005 "network connection lost" from Hugging Face's Xet CDN; the app showed
// a raw NSError and stopped). Transient network errors are retried with backoff; anything else is shown as before.
import Foundation

/// NSURLError codes worth retrying: timed out, cannot find / connect to host, connection lost, not connected,
/// international roaming off, data not allowed (cellular off for the app) - the last two wait for Wi-Fi to return.
public let transientURLErrorCodes: Set<Int> = [-1001, -1003, -1004, -1005, -1009, -1018, -1020]

/// Seconds to wait before attempt `attempt + 1`, or nil to give up and show the error.
public func downloadRetryDelay(attempt: Int, urlErrorCode: Int?, maxAttempts: Int = 40) -> Double? {
    guard let c = urlErrorCode, transientURLErrorCodes.contains(c), attempt >= 1, attempt < maxAttempts else { return nil }
    return min(60, 2 * pow(1.5, Double(attempt - 1)))
}
