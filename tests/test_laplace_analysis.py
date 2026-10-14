"""Synthetic posterior checks, independent of the unavailable fitted model artifact."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vision_backend.model.block_head")

from marslabeler.inference.laplace import checkpoint_sha256, make_laplace_score_fn
from tests.test_block_model_analysis import bundle


def posterior(tmp_path, bundle, variance):
    bundle.checkpoint_path = tmp_path / "model.pt"
    bundle.checkpoint_path.write_bytes(b"test checkpoint identity")
    head = bundle.model.seg.decoder.head
    artifact = {
        "format": "mars-spatial-laplace-diag-v1",
        "head_name": "seg.decoder.head",
        "checkpoint_sha256": checkpoint_sha256(bundle.checkpoint_path),
        "posterior_variance": {
            "weight": torch.full_like(head.weight, variance),
            "bias": torch.full_like(head.bias, variance),
        },
    }
    path = tmp_path / "model.laplace.pt"
    torch.save(artifact, path)
    return path, artifact


def test_zero_posterior_variance_has_zero_epistemic_uncertainty(bundle, tmp_path):
    path, _ = posterior(tmp_path, bundle, 0.)
    batch = np.full((1, 4, 4), 120, dtype=np.uint8)
    score = make_laplace_score_fn(bundle, path)(batch, None)
    np.testing.assert_allclose(score, 0, atol=1e-6)


def test_posterior_draws_produce_spatial_map_without_backbone_reruns(bundle, tmp_path):
    path, _ = posterior(tmp_path, bundle, 20.)
    batch = np.tile(np.arange(4, dtype=np.uint8) * 60 + 10, (1, 4, 1))
    before = {key: value.clone() for key, value in bundle.model.state_dict().items()}
    score_fn = make_laplace_score_fn(bundle, path)
    first = score_fn(batch, None)
    assert bundle.model.seg.calls == 1
    assert first.shape == batch.shape
    assert np.isfinite(first).all() and np.all((first >= 0) & (first <= 1))
    assert first.max() > .01 and np.ptp(first) > .001
    np.testing.assert_array_equal(first, score_fn(batch, None))
    for name, value in bundle.model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_laplace_rejects_different_model_training_prior_and_invalid_variance(bundle, tmp_path):
    path, artifact = posterior(tmp_path, bundle, 1.)
    bundle.checkpoint_path.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="different checkpoint"):
        make_laplace_score_fn(bundle, path)
    artifact["checkpoint_sha256"] = checkpoint_sha256(bundle.checkpoint_path)
    artifact["posterior_variance"]["weight"][0, 0] = -1
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="nonnegative"):
        make_laplace_score_fn(bundle, path)
    torch.save({"precision": {}, "mean": {}}, path)
    with pytest.raises(ValueError, match="training prior"):
        make_laplace_score_fn(bundle, path)
