# Labeling workflow verification — 22 September 2026

The exercised labeling → save → reopen → export → training-batch workflow passes after fixes. The original implementation failed all 13 initial end-to-end regression scenarios. This verifies the mechanical integrity of the tested workflows; it is not a claim that all possible inputs, crashes, or scientific class assignments are correct.

## Real observation exercise

Source: `ESP_092084_1525_RED.JP2`, 13,432 × 27,301 pixels, native uint16, 0.5 m/pixel.

The verification script opened the actual observation with the real Qt loader, preprocessing worker, canvas, and keyboard controller (offscreen Qt platform). It clicked six tiles in separate parts of the observation, assigned class hotkeys, checked the three-action autosave, exercised clear/undo/redo/relabel, closed immediately after editing, and reopened the session. Both all six labels and the cursor were restored.

| Source pixel origin | Final class ID | Crop |
| --- | ---: | --- |
| (1536, 1024) | 1 | 512 × 512, uint16 |
| (6144, 5120) | 11 | 512 × 512, uint16 |
| (9728, 8192) | 12 | 512 × 512, uint16 |
| (9728, 15872) | 17 | 512 × 512, uint16 |
| (4096, 21504) | 6 | 512 × 512, uint16 |
| (12800, 25600) | 13 | 512 × 512, uint16 |

These are intentionally mechanical QA assignments, **not validated scientific terrain annotations**. They are stored under `exports/workflow-verification-20260922-final/labels/`, separately from production labels.

Every exported PNG was independently reopened and compared to a native-resolution rasterio read of the source window. All 1,572,864 exported pixels were exactly equal: maximum absolute error **0**, with uint16 preserved. The six CSV rows, filenames, class IDs, class names (including commas), and coarse GeoTIFF positions/IDs all agreed. The output contained exactly six crops.

The resulting CSV/PNG dataset was shuffled through a PyTorch DataLoader, with each tensor target checked against its source block ID. A small CPU classifier consumed a `(6, 1, 64, 64)` batch, computed finite cross-entropy (2.88985085), backpropagated finite gradients, and updated its weights. This tests the exported classification data interface, not model accuracy or full Stage-3 training.

Artifacts: [machine-readable checks](../exports/workflow-verification-20260922-final/verification.json), [crop contact sheet](../exports/workflow-verification-20260922-final/verified_crops.png), [reopened GUI](../exports/workflow-verification-20260922-final/gui_reopened.png).

## Fixed failures

- Recent edits were lost on close or when opening another observation. Labeling sessions now save at both transitions, and a failed save prevents closing/switching.
- Count-based autosave was checked only by the timer, the time threshold used the last edit rather than last save, and painting/selection/undo/redo could bypass dirty tracking. These paths now share save tracking; File → Save Labels and the platform save shortcut are available.
- Automatic nodata discovery polluted undo history, preventing one undo from undoing the user's label. Automatic skips no longer add undo steps.
- Arrow navigation across a panel boundary could display the old panel while the label cursor belonged to the next one. The canvas and preview now follow the actual cursor panel.
- Editing an existing label ignored `advance_on_edit: false`. The configured review behavior is now respected.
- Direct Parquet writes could truncate a previous save, and a missing cursor sidecar caused existing labels to be ignored. Saves now atomically replace the Parquet, with session metadata in that same snapshot; the JSON sidecar is optional for recovery.
- Export forced uint16 imagery into Pillow's 8-bit `L` mode and manually joined CSV fields. PNGs now retain native pixel values, and CSV fields are quoted properly.
- Export accepted another observation or bad coordinates, while raster reads could clamp invalid coordinates into unrelated pixels. New saves bind to the source file SHA-256 and georeferencing; export/resume validate identity and geometry and reject invalid, duplicate, or inconsistent records.
- Reusing export folders could retain orphaned crops. Probe exports now publish a complete set into a new/empty output directory.
- Map centroids and coarse export transforms omitted affine rotation/shear terms. They now use the full affine transform.
- Empty padding cells could be labeled despite having no crop. Keyboard/paint/selection paths now exclude them; class IDs that would wrap or collide in uint8 coarse exports raise an error.

## Automated coverage

Final full-suite result: **334 passed**, 0 failed, 0 skipped (69.37 seconds). The 96 warnings are rasterio notices from existing synthetic fixtures with identity geotransforms. `git diff --check` also passed.

`tests/test_labeling_workflow.py` adds 29 regression cases, including the real UI event path on a deterministic asymmetric uint16 raster with partial right/bottom edge tiles. It covers save/reload/export identity, all edit paths, autosave, panel navigation, same-name/same-size wrong images, legacy Parquet-only sessions, damaged sidecars, failed writes, corrupt extents/statuses, stale output folders, and affine geometry.

The existing undo test now passes. An existing Qt test used the unavailable `QKeyEvent.setAutoRepeat`; it now supplies the autorepeat flag through the event constructor. Startup help is scheduled on an actual observation open, avoiding modal help dialogs during hidden-window render tests.

Reproduce using the existing environment with torch, Qt, and rasterio installed:

```bash
cd MarsObsLabeling
PYTHONPATH=src:../AI4ExoMars QT_QPA_PLATFORM=offscreen \
  ../AI4ExoMars/.venv/bin/python -m pytest tests/ -q

PYTHONPATH=src:../AI4ExoMars QT_QPA_PLATFORM=offscreen \
  ../AI4ExoMars/.venv/bin/python scripts/verify_labeling_workflow.py \
  ../ESP_092084_1525_RED.JP2 --output exports/new-workflow-check
```

## Scope and remaining format constraints

- Existing production labels were not edited. The real-data labels are isolated QA artifacts.
- Class IDs are authoritative. Class names may be renamed, but reusing an ID for a different terrain concept is not automatically detectable.
- New saves require the original source bytes and georeferencing. Re-encoding the raster or modifying internal overviews changes its fingerprint; identical copies remain valid. Legacy saves without a source fingerprint have weaker identity checks.
- Probe PNG export supports uint8 and uint16; unsupported source dtypes are rejected instead of silently converted.
- The coarse display GeoTIFF is **not a direct Stage-3 training label raster**: it keeps 0-based IDs and uses 253/254/255 for abstain/nodata/unlabeled. The existing Stage-3 loader expects 1-based labels with 0 ignored, and uint8 imagery. Explicit remapping/preprocessing is required for that separate pipeline. The QA training step used the classification CSV/PNG pairs.
- Atomic replacement protects against an interrupted individual write; this exercise did not simulate power loss, storage hardware failure, or concurrent editing on another OneDrive client.
