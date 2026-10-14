"""Stage-4 integration uses the trained block head and the matching NPCA vector."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
block_head = pytest.importorskip("vision_backend.model.block_head")

from marslabeler.inference.modelio import (
    ModelBundle, load_model_bundle, make_prediction_analysis_fn,
    make_uncertainty_score_fn, _forward_logits,
)
from marslabeler.inference.local_npca import classifier_weight_vectors


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.head = torch.nn.Conv2d(2, 2, 1)
        self.decoder.head_dropout = torch.nn.Identity()
        self.calls = 0
        with torch.no_grad():
            self.decoder.head.weight.zero_()
            self.decoder.head.bias.copy_(torch.tensor([5., -5.]))

    def decode_features(self, image):
        self.calls += 1
        return torch.cat([image, image.square()], dim=1)


@pytest.fixture
def bundle():
    # Reverse pixel -> block ordering and add a third class without a pixel head.
    model = block_head.BlockClassifier(TinySegmenter(), [1, 0, None], attn_hidden=2, tile_size=4)
    with torch.no_grad():
        model.class_bias[2] = 20.
    return ModelBundle(model.eval(), "simmim", 3, 4, False, torch.device("cpu"),
                       Path("model.pt"), {}, block_classifier=True)


def test_one_pass_preserves_block_vote_and_segmentation_class_mapping(bundle):
    batch = np.full((2, 4, 4), 120, dtype=np.uint8)
    batch[1, :2, :2] = 0
    expected = _forward_logits(bundle, batch, None, None).detach()
    bundle.model.seg.calls = 0
    bundle.temperature = 2.0
    predict, embeddings, confidence, votes, logits = make_prediction_analysis_fn(bundle)
    pixels = predict(batch, None)
    assert bundle.model.seg.calls == 1
    assert votes == [2, 2]  # newly learned block class; never pixel majority
    assert np.all(pixels == 1)  # segmentation channel 0 -> block class 1
    np.testing.assert_allclose(confidence, (expected / 2).softmax(1).max(1).values.numpy())
    np.testing.assert_allclose(logits[0], expected.numpy(), rtol=1e-6)
    assert embeddings[0].shape == (2, bundle.model.fc.in_features)
    weights = classifier_weight_vectors(bundle.model, [2])
    np.testing.assert_array_equal(weights[2], bundle.model.fc.weight[2].detach().numpy())
    assert not bundle.model.fc._forward_pre_hooks
    assert not bundle.model.seg.decoder.head._forward_hooks


def test_stage4_checkpoint_loads_entire_state_and_run_directory(bundle, tmp_path, monkeypatch):
    from vision_backend.training import builders
    monkeypatch.setattr(builders, "build_simmim_segmentation_model", lambda config: TinySegmenter())
    torch.save({
        "block_state": bundle.model.state_dict(), "seg_model_config": {},
        "head_config": {"attn_hidden": 2, "dropout": 0.1, "tile_size": 4},
        "classes": {"seg_index": [1, 0, None]},
    }, tmp_path / "best.pt")
    loaded = load_model_bundle(tmp_path, device="cpu")
    assert loaded.block_classifier
    assert loaded.num_classes == 3 and loaded.model.num_seg_classes == 2
    assert loaded.required_stride == 4
    for name, value in bundle.model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], value)
    assert not loaded.model.training


def test_mahalanobis_uses_block_tiling_and_retains_spatial_distances(bundle):
    from vision_backend.uncertainty.malahanobis import ClassGaussianStats, MahalanobisStats
    stats = MahalanobisStats({0: ClassGaussianStats(torch.zeros(2), torch.eye(2))}, 2, 2.)
    batch = np.tile(np.arange(8, dtype=np.uint8) * 30 + 10, (1, 8, 1))
    score = make_uncertainty_score_fn(bundle, stats, normalize=False)(batch, None)
    x = batch.astype(np.float32) / 127.5 - 1
    expected = np.sqrt(x ** 2 + x ** 4)
    np.testing.assert_allclose(score, expected, atol=1e-6)
    assert np.unique(score).size > 1
    scaled = make_uncertainty_score_fn(bundle, stats)(batch, None)
    np.testing.assert_allclose(scaled, np.clip(expected / 2, 0, 1), atol=1e-6)


def test_local_npca_ranks_matching_class_blocks_and_excludes_absent_classes():
    from marslabeler.inference.local_npca import build_local_gallery
    from vision_backend.pc_align.neural_pca import NeuralPCAResult
    from tests.test_inference_engine import make_block

    blocks = [make_block(f"b{i}", i * 8, 0) for i in range(3)]
    embeddings = np.array([[1., 0.], [3., 1.], [100., 2.]], dtype=np.float32)
    bases = {c: NeuralPCAResult(c, torch.zeros(2), torch.eye(2), torch.ones(2), 3)
             for c in (0, 1)}
    weights = {c: np.array([2., 1.], dtype=np.float32) for c in (0, 1)}
    result = build_local_gallery(embeddings, blocks, bases, weights,
                                 eligible_block_ids={0: {"b0", "b1"}})
    assert list(result) == [0]  # no blocks predicted as class 1
    assert [item.block_id for item in result[0][0]] == ["b1", "b0"]
    assert [item.score for item in result[0][0]] == [6., 2.]
    assert result[0][0][0].x_px == 8  # navigation still points at the winning block
