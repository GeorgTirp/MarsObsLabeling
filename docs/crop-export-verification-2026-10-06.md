# Crop/label export verification — 6 October 2026

The GUI's **Export Labels** button now produces a portable crop/label dataset.
The export includes current in-memory edits, native PNGs, a CSV pairing manifest,
explicit class/target mappings, preprocessing metadata, a full Parquet label
snapshot, a coarse GIS raster, loading instructions, and checksums. Each export
has a fresh timestamped directory under `exports/<observation>/`.

## Verification results

The complete repository suite passed: **389 passed, 0 failed, 0 skipped** in
37.05 seconds. The 96 warnings are rasterio notices from synthetic test rasters
with identity geotransforms. `git diff --check` passed.

The new export tests exercise both uint8 and uint16 imagery and cover:

- Actual Qt Export Labels button, real background worker, and unsaved labels.
- Exact source pixels, partial edge crops, CSV quoting, and class assignments.
- Stable target mapping with missing classes, non-contiguous IDs, and a separate
  configured model-channel mapping.
- Copying the dataset to another folder and loading without its source image.
- Shuffled PyTorch batches, correct crop/target pairing, padding/validity masks,
  finite loss and gradients, and an optimizer update.
- Rejection of targets that do not fit a requested classifier head.
- Missing, truncated, swapped, extra, or modified files and altered manifests.
- A deliberately changed class assignment with recomputed checksums, caught by
  comparison with the independent Parquet label snapshot.
- Cancellation, simulated full disk, incorrect PNG encoding, and archive failure.
- Re-export after clearing/relabeling: no stale crops, prior export unchanged.
- Failed GUI export leaves the user's current labels available.

## Real observation exercise

Source: `ESP_092084_1525_RED.JP2`, 13,432 × 27,301, uint16, 0.5 m/pixel.

The verification script opened the actual observation, clicked six image blocks,
assigned class hotkeys, exercised autosave and clear/undo/redo/relabel, closed and
reopened the session, then clicked **the actual GUI Export Labels button**. It
copied the resulting folder and verified it with the same reader available to
training users.

All six exported 512 × 512 images matched independent native raster reads:
**1,572,864 pixels checked; maximum absolute error 0; uint16 retained.** CSV class
names/IDs, crop coordinates, and coarse GeoTIFF classes/alignment agreed.

`BlockCropDataset` loaded the copied folder into a shuffled `(6, 1, 512, 512)`
batch. Every target matched its block ID. A small CPU block classifier completed
a forward pass, finite cross-entropy loss (2.844468), finite backpropagation, and
an optimizer update. This validates the exported classification interface; it
does not demonstrate fine-tuning the existing Stage-3 segmentation architecture.

The six labels are mechanical QA assignments, not scientific annotations. All
trial sessions/exports are under `exports/crop-dataset-verification-20261006/`;
production annotation files were not changed.

Artifacts: [machine-readable verification](../exports/crop-dataset-verification-20261006/verification.json),
[test results](../exports/crop-dataset-verification-20261006/pytest-results.txt),
[crop contact sheet](../exports/crop-dataset-verification-20261006/verified_crops.png),
[portable sample dataset](../exports/crop-dataset-verification-20261006/probe/).

## Checks on every export

Before the destination folder is published, the exporter validates source
identity and block coordinates; reopens every PNG and compares its dtype and
pixels with the source crop; checks the CSV mapping against the archived label
snapshot; and records SHA-256 checksums. Failure or cancellation removes the
staging dataset. The prior dataset is never overwritten.

To check an exported folder again after transfer, without PyTorch or the source:

```bash
python -m marslabeler.model.cropdataset /path/to/export
```

The default tensor loader also verifies each crop checksum as it reads it.
Checksums detect accidental corruption or changes; they are not a guarantee
against storage loss or deliberate alteration of every copy of the metadata.

## Reproduce

```bash
PYTHONPATH=src:../AI4ExoMars QT_QPA_PLATFORM=offscreen \
  ../AI4ExoMars/.venv/bin/python -m pytest tests/ -q

PYTHONPATH=src:../AI4ExoMars QT_QPA_PLATFORM=offscreen \
  ../AI4ExoMars/.venv/bin/python scripts/verify_labeling_workflow.py \
  ../ESP_092084_1525_RED.JP2 --output exports/new-crop-verification
```

These are whole-block majority-class labels. Dense segmentation targets,
context imagery, train/validation splitting, model-head compatibility, and
scientific correctness of assigned labels remain separate concerns. The recorded
uint16 preprocessing is the existing observation-wide approximation used by
`mars-inference`, not a recreation of the DRG training-data pipeline.
