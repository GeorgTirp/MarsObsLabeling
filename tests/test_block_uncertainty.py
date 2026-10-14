"""Stage-4 block classifiers: training input stretch, calibration and per-block uncertainty."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from marslabeler.inference.engine import OXIA_P1_P99, quantize_to_uint8
from marslabeler.inference.modelio import read_temperature

torch = pytest.importorskip("torch")
pytest.importorskip("vision_backend.uncertainty.block_laplace")

from marslabeler.inference.blockanalysis import (  # noqa: E402
    BlockFeatures,
    block_laplace_scores,
    block_mahalanobis_scores,
)
from tests.test_block_model_analysis import bundle  # noqa: E402,F401
from tests.test_mainwindow_inference import qapp  # noqa: E402,F401

MODEL_DIR = Path(__file__).resolve().parents[2] / "v2_final_lr3_e12"


def test_oxia_stretch_matches_training_crops():
    """Same mapping as AI4ExoMars to_uint8(stretch="oxia"): p1/p99 -> 58/178, 0 = nodata only."""
    raw = np.array([[0, 100, 330, 799, 2000, 331]], dtype=np.uint16)
    out = quantize_to_uint8(raw, (330.0, 799.0), stretch="oxia")
    unit = (raw.astype(np.float64) - 330) / (799 - 330)
    expected = np.clip(np.rint(OXIA_P1_P99[0] + unit * 120), 1, 255).astype(np.uint8)
    expected[raw == 0] = 0
    np.testing.assert_array_equal(out, expected)
    assert out[0, 0] == 0 and out[0, 1] == 1  # dark terrain must not become nodata
    assert (out[0, 2], out[0, 3]) == (58, 178)
    # The full stretch, used for Stage-3 models, is unchanged.
    np.testing.assert_array_equal(quantize_to_uint8(raw, (330.0, 799.0))[0, 2:4], [0, 255])


def test_block_classifier_tensor_uses_its_input_stretch(bundle):  # noqa: F811
    from marslabeler.inference.modelio import _to_tensor

    raw = np.array([[[0, 330], [799, 2000]]], dtype=np.uint16)
    bundle.input_stretch = "oxia"
    tensor = _to_tensor(bundle, raw, (330.0, 799.0))
    expected = np.array([[0, 58], [178, 255]], dtype=np.float32) / 127.5 - 1
    np.testing.assert_allclose(tensor[0, 0].numpy(), expected)


def test_temperature_only_calibrates_the_checkpoint_it_was_fitted_for(tmp_path):
    checkpoint = tmp_path / "best.pt"
    assert read_temperature(checkpoint) == 1.0
    (tmp_path / "eval.json").write_text(json.dumps(
        {"checkpoint": "checkpoints/run/best.pt", "temperature": 1.6}))
    assert read_temperature(checkpoint) == pytest.approx(1.6)
    assert read_temperature(tmp_path / "last.pt") == 1.0
    (tmp_path / "eval.json").write_text(json.dumps({"checkpoint": "best.pt", "temperature": -1}))
    assert read_temperature(checkpoint) == 1.0


def synthetic_features(n=40, d=3, k=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    fc = torch.nn.Linear(d, k)
    phi = torch.randn(n, d, generator=generator)
    offset = torch.randn(n, k, generator=generator)
    with torch.no_grad():
        logits = offset + fc(phi)
    weight = torch.cat([fc.weight.detach(), fc.bias.detach()[:, None]], dim=1)
    features = BlockFeatures([f"b{i}" for i in range(n)], phi.numpy(), logits.numpy(), weight.numpy())
    return features, phi, logits, weight


def save_laplace(path, phi, logits, weight, prior_precision):
    from vision_backend.uncertainty.block_laplace import LastLayerLaplace, save_last_layer_laplace

    save_last_layer_laplace(LastLayerLaplace.fit(phi, logits, weight, prior_precision=prior_precision), path)
    return path


def test_block_laplace_is_normalised_entropy_and_reduces_to_map_without_posterior_spread(tmp_path):
    features, phi, logits, weight = synthetic_features()
    tight = save_laplace(tmp_path / "tight.laplace.pt", phi, logits, weight, 1e12)
    scores = block_laplace_scores(features, tight)
    p = logits.softmax(1)
    expected = -(p * p.log()).sum(1) / math.log(3)
    np.testing.assert_allclose([scores[f"b{i}"] for i in range(len(features))], expected.numpy(), atol=2e-3)

    wide = save_laplace(tmp_path / "wide.laplace.pt", phi, logits, weight, 0.01)
    progress = []
    first = block_laplace_scores(features, wide, progress_cb=lambda done, total: progress.append(done))
    assert first == block_laplace_scores(features, wide)  # seeded, so the map is reproducible
    assert all(0 <= value <= 1 for value in first.values())
    assert np.mean(list(first.values())) > expected.mean().item()  # posterior spread adds entropy
    assert progress[-1] == len(features)
    assert block_laplace_scores(features, wide, should_cancel=lambda: True) == {}


def test_block_laplace_rejects_a_posterior_fitted_for_another_head(tmp_path):
    features, phi, logits, weight = synthetic_features()
    path = save_laplace(tmp_path / "model.laplace.pt", phi, logits, weight + 1, 1.0)
    with pytest.raises(ValueError, match="different block head"):
        block_laplace_scores(features, path)


def test_block_mahalanobis_is_relative_to_the_reference_distance(tmp_path):
    from vision_backend.uncertainty.malahanobis import ClassGaussianStats, MahalanobisStats, save_stats

    features, phi, _, _ = synthetic_features()
    path = tmp_path / "model.uncertainty.pt"
    save_stats(MahalanobisStats({0: ClassGaussianStats(torch.zeros(3), torch.eye(3))}, 3, 2.0), path)
    scores = block_mahalanobis_scores(features, path)
    np.testing.assert_allclose([scores[f"b{i}"] for i in range(len(features))],
                               (phi.norm(dim=1) / 2).numpy(), rtol=1e-5)

    # Pixel-level stats (decoder features) are not interchangeable with phi.
    save_stats(MahalanobisStats({0: ClassGaussianStats(torch.zeros(4), torch.eye(4))}, 4, 2.0), path)
    with pytest.raises(ValueError, match="block-level"):
        block_mahalanobis_scores(features, path)


def test_block_features_file_belongs_to_one_model_and_grid(tmp_path):
    features, *_ = synthetic_features(n=3)
    path = tmp_path / "OBS.block_features.npz"
    features.save(path, "sig-a")
    loaded = BlockFeatures.load(path, "sig-a", {"b0", "b1", "b2", "b3"})
    assert loaded.block_ids == features.block_ids
    np.testing.assert_array_equal(loaded.phi, features.phi)
    np.testing.assert_array_equal(loaded.logits, features.logits)
    np.testing.assert_array_equal(loaded.head_weight, features.head_weight)
    assert BlockFeatures.load(path, "sig-b", {"b0", "b1", "b2"}) is None
    assert BlockFeatures.load(path, "sig-a", {"b0", "b1"}) is None
    assert BlockFeatures.load(tmp_path / "missing.npz", "sig-a", {"b0"}) is None
    assert not list(tmp_path.glob(".*.tmp*"))


def test_worker_scores_kept_features_without_loading_the_model(qapp, tmp_path, monkeypatch):
    from vision_backend.uncertainty.malahanobis import ClassGaussianStats, MahalanobisStats, save_stats

    import marslabeler.inference.modelio as modelio
    from marslabeler.inference.blockanalysis import MAHALANOBIS_DISPLAY_MAXIMUM
    from marslabeler.ui.uncertaintydialog import UncertaintyWorker
    from tests.test_inference_engine import make_block

    features, phi, _, _ = synthetic_features(n=3)
    save_stats(MahalanobisStats({0: ClassGaussianStats(torch.zeros(3), torch.eye(3))}, 3, 1.0),
               tmp_path / "best.uncertainty.pt")
    monkeypatch.setattr(modelio, "load_model_bundle", lambda *a, **k: pytest.fail("model loaded"))
    blocks = [make_block("b0", 0, 0), make_block("b2", 8, 0)]  # b1 retired as nodata since
    worker = UncertaintyWorker(None, blocks, 8, tmp_path / "best.pt", {
        "method": "mahalanobis", "block_classifier": True, "block_features": features,
    })
    results, errors = [], []
    worker.finished_ok.connect(results.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert not errors
    assert results[0] == pytest.approx({"b0": float(phi[0].norm()), "b2": float(phi[2].norm())}, rel=1e-5)
    assert worker.maximum == MAHALANOBIS_DISPLAY_MAXIMUM


@pytest.mark.skipif(not (MODEL_DIR / "best.laplace.pt").exists(), reason="v2_final_lr3_e12 not present")
def test_real_artifacts_reproduce_the_model_card():
    """Scores from the shipped files match the model card's own test-set checks."""
    data = np.load(MODEL_DIR / "block_features.npz")
    weight = data["fc_weight_bias"].astype(np.float32)

    def features(split):
        phi = data[f"{split}_phi"].astype(np.float32)
        logits = data[f"{split}_offset"] + phi @ weight[:, :-1].T + weight[:, -1]
        return BlockFeatures([str(i) for i in range(len(phi))], phi, logits.astype(np.float32), weight)

    test = features("test")
    assert (test.logits.argmax(1) == data["test_label"]).mean() == pytest.approx(0.680, abs=1e-3)
    laplace = block_laplace_scores(test, MODEL_DIR / "best.laplace.pt")
    entropy = np.array([laplace[i] for i in test.block_ids])
    wrong = test.logits.argmax(1) != data["test_label"]
    ranks = entropy.argsort().argsort() + 1
    auroc = (ranks[wrong].sum() - wrong.sum() * (wrong.sum() + 1) / 2) / (wrong.sum() * (~wrong).sum())
    assert auroc == pytest.approx(0.727, abs=0.01)  # model card: Laplace entropy (MC) 0.727

    oxia = features("oxia")
    distance = block_mahalanobis_scores(oxia, MODEL_DIR / "best.uncertainty.pt")
    assert np.median(list(distance.values())) == pytest.approx(2.14, abs=0.01)  # model card: 2.14
