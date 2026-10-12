import Testing
@testable import FindPicsCore

// A bare "selfies" is answered by the front-camera tag alone, with no judge call (10-11: the judge version was
// still unfinished after 11 minutes on a warm phone). Anything more than "selfies" still needs the judge.
@Suite struct BareSelfieTests {
    @Test func bareRequests() {
        #expect(isBareSelfieRequest("selfies"))
        #expect(isBareSelfieRequest("my selfies"))
        #expect(isBareSelfieRequest("show me all my selfie photos"))
        #expect(isBareSelfieRequest("Selfies!"))
    }

    @Test func requestsWithMoreInThem() {
        #expect(!isBareSelfieRequest("selfies at the beach"))
        #expect(!isBareSelfieRequest("selfies with mum"))
        #expect(!isBareSelfieRequest("selfies where I look tired"))
        #expect(!isBareSelfieRequest("selfies from 2019"))   // a date is content: the date filter must be planned
    }

    @Test func notASelfieRequestAtAll() {
        #expect(!isBareSelfieRequest("photos of me"))
        #expect(!isBareSelfieRequest("food photos"))
        #expect(!isBareSelfieRequest(""))
    }

}
