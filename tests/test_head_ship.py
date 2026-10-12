"""RESULTS 48: the learned head's phone file (.fphead) and the fast-mode blend, numpy reference
(eval/learned_head/head_ship.py, the golden generator for FindPicsCore HeadRankerTests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "learned_head"))
import head_ship as HS  # noqa: E402


def _relu(x):
    return np.maximum(x, 0)


def test_file_round_trip_and_fold_equals_unfolded(tmp_path):
    rng = np.random.default_rng(0)
    for B in (3, 4):
        D, H1, H2 = 8, 5, 3
        W1 = (rng.integers(-64, 64, (H1, B * D)) / 64).astype(np.float16)    # exact in fp16
        b1, W2 = rng.normal(size=H1), (rng.integers(-64, 64, (H2, H1)) / 64).astype(np.float16)
        b2, w3, b3 = rng.normal(size=H2), rng.normal(size=H2), 0.25
        f = tmp_path / f"h{B}.fphead"
        HS.write_head(f, W1, b1, W2, b2, w3, b3, {"test": B}, D=D)
        H = HS.HeadFile(f)
        assert (H.D, H.B, H.H1, H.H2) == (D, B, H1, H2) and H.meta == {"test": B}
        t = rng.normal(size=D); t /= np.linalg.norm(t)
        X = rng.normal(size=(6, D))
        feats = [X, np.broadcast_to(t, X.shape), X * t, np.abs(X - t)][:B]
        want = _relu(_relu(np.hstack(feats) @ W1.astype(float).T + b1.astype(np.float32)) @ W2.astype(float).T
                     + b2.astype(np.float32)) @ w3.astype(np.float32) + np.float32(b3)
        np.testing.assert_allclose(H.logits(X, H.fold(t)), want, rtol=1e-6, atol=1e-6)


def test_blend_is_zscore_sum_with_clip_and_missing_vectors():
    cos = np.array([0.1, 0.3, 0.2, -np.inf])
    lg = np.array([40.0, -3.0, 0.0, 2.0])
    b = HS.blend(cos, lg)
    zc = (cos[:3] - cos[:3].mean()) / cos[:3].std()
    h = np.array([HS.LOGIT_CLIP, -3.0, 0.0])
    np.testing.assert_allclose(b[:3], zc + (h - h.mean()) / h.std())
    assert b[3] == -np.inf
    assert np.all(HS.zscore([5.0, 5.0]) == 0)          # std 0 -> divide by 1
