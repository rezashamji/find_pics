"""Can the app find "which face is you" without Apple's People names (privacy.apple.com copy has none)?
people.face_groups on the test library (19,218 items; IMDB photos are named imdb_<identity>_<n>, so each IMDB item's
main person is known; Open Images items have unknown people). Per top group: majority identity, purity (share of the
group's IMDB items with that identity; non-IMDB items count as wrong), and how many of that identity's items the group
covers before/after expand_refs (what the product would then use as references). CPU only.
Usage: python eval/eval_face_groups.py [index_dir]"""
import json
import re
import sys
import time
from collections import Counter

import numpy as np


def main():
    sys.path.insert(0, "src")
    from findpics import store
    from findpics.people import expand_refs, face_groups, item_person_scores
    idx = store.load(sys.argv[1] if len(sys.argv) > 1 else "data/public/index_testlib")
    ident = idx.items.path.str.extract(r"/imdb_(\d+)_\d+\.")[0].to_numpy()
    size = Counter(x for x in ident if isinstance(x, str))
    t0 = time.time(); G = face_groups(idx, top=12); dt = time.time() - t0
    out = []
    for g in G:
        ids = [ident[i] for i in g["items"]]
        maj, n_maj = Counter(x for x in ids if isinstance(x, str)).most_common(1)[0] if any(isinstance(x, str) for x in ids) else (None, 0)
        refs = expand_refs(idx, idx.face_emb[g["faces"]], accept=0.55, rounds=3)
        sc, _ = item_person_scores(idx, refs)
        got = np.where(sc >= 0.55)[0]
        tp = sum(ident[i] == maj for i in got)
        out.append(dict(items=len(g["items"]), identity=maj, identity_items=size.get(maj, 0), purity=round(n_maj / len(ids), 3),
                        covered_group=n_maj, after_expand=dict(found=int(len(got)), correct=int(tp),
                                                                precision=round(tp / max(len(got), 1), 3),
                                                                recall=round(tp / max(size.get(maj, 1), 1), 3))))
        print(json.dumps(out[-1]), flush=True)
    print(f"{len(G)} groups in {dt:.0f}s; identities by size: {size.most_common(8)}")
    json.dump(dict(groups=out, seconds=dt), open("eval/results_face_groups.json", "w"), indent=1)


if __name__ == "__main__":
    main()
