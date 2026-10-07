// "Who is X?": suggest the face group, then ask to confirm ("Is this you?"). iOS gives apps no access to Apple's
// People names, so the suggestion comes from the library itself: for the owner, the group most present in
// front-camera photos (selfies, EXIF lens model); for anyone else, the largest group not already named.
import Foundation

func unit(_ v: [Float]) -> [Float] { let n = sqrt(dot(v, v)) + 1e-8; return v.map { $0 / n } }

public func isOwnerWord(_ person: String) -> Bool { ["me", "i", "myself"].contains(normText(person)) }

/// "Is this you?" / "Is this Mom?"
public func whoQuestion(_ person: String) -> String { isOwnerWord(person) ? "Is this you?" : "Is this \(person)?" }

/// Groups that already belong to a named person: mean over the group's faces of the best cosine to that person's
/// references >= accept (same accept as face grouping).
public func assignedGroups(groups: [[[Float]]], named: [[[Float]]], accept: Float = 0.55) -> Set<Int> {
    var out = Set<Int>()
    for (g, faces) in groups.enumerated() where !faces.isEmpty {
        for refs in named where !refs.isEmpty {
            let R = refs.map(unit)
            let m = faces.map { f in let u = unit(f); return R.map { dot(u, $0) }.max()! }.reduce(0, +) / Float(faces.count)
            if m >= accept { out.insert(g); break }
        }
    }
    return out
}

/// Front-camera photos needed before "most present in selfies" beats "largest group" for the owner.
public let minSelfiesForSuggestion = 3

/// The group to suggest, or nil when every group is taken. `groupItems`: item numbers per group (groups sorted
/// largest first, as faceGroups returns them); `itemCamera`: "front" / "back" / "" per item number.
public func suggestGroup(groupItems: [[Int]], itemCamera: [String], forOwner: Bool, assigned: Set<Int>) -> Int? {
    let free = groupItems.indices.filter { !assigned.contains($0) }
    guard let largest = free.first else { return nil }
    if !forOwner { return largest }
    var best = largest, bestN = -1
    for g in free {
        let n = Set(groupItems[g]).filter { $0 >= 0 && $0 < itemCamera.count && itemCamera[$0] == "front" }.count
        if n > bestN { best = g; bestN = n }
    }
    return bestN >= minSelfiesForSuggestion ? best : largest
}

/// The other groups to list under the suggestion: unnamed ones first (largest first), then the named ones.
public func otherGroupsOrder(count: Int, suggested: Int?, assigned: Set<Int>) -> [Int] {
    let rest = (0..<count).filter { $0 != suggested }
    return rest.filter { !assigned.contains($0) } + rest.filter { assigned.contains($0) }
}

/// "Add a photo of them": one face per picked photo. With 2+ photos, the face that best matches faces in the OTHER
/// photos (the person they all share); if nothing matches (best mean cosine < 0.3) or with one photo, the largest.
public func pickRefFaces(photos: [[(px: Double, emb: [Float])]]) -> [[Float]] {
    let ps = photos.filter { !$0.isEmpty }
    return ps.indices.map { p -> [Float] in
        let largest = ps[p].indices.max { ps[p][$0].px < ps[p][$1].px }!
        guard ps.count >= 2 else { return ps[p][largest].emb }
        var best = largest, bestS: Float = -2
        for (k, f) in ps[p].enumerated() {
            let u = unit(f.emb)
            let s = ps.indices.filter { $0 != p }.map { q in ps[q].map { dot(u, unit($0.emb)) }.max()! }.reduce(0, +) / Float(ps.count - 1)
            if s > bestS { bestS = s; best = k }
        }
        return ps[p][bestS >= 0.3 ? best : largest].emb
    }
}
