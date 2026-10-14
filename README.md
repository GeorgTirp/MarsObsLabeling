# Mars Obs Labeling Tool

A fast, keyboard-driven GUI for block-level terrain labeling of HiRISE observations.

## Overview

This tool enables rapid supervised labeling of Mars terrain features on gigapixel HiRISE scenes. The labeler views a large panel of imagery, presses a single key for each block's majority terrain class, and the cursor auto-advances to the next block. Minimal mouse movement and maximum keyboard speed are the core design goals.

The desktop workspace uses a dark, imagery-centered layout inspired by
[MMGIS](https://nasa-ammos.github.io/MMGIS/): observation and save controls in the
header, panel navigation on the left, terrain classes beside the image, and a
resizable block inspector with view and analysis controls on the right. It runs
locally in PySide6 with the existing Python raster, labeling, and inference core.
Open JP2 or GeoTIFF imagery from **Open observation** (or Ctrl/Cmd+O). The toolbar
exposes zoom, **Fit panel**, and keyboard help; all labeling shortcuts remain
available. The labeler opens maximized with a large **Block preview** visible on
the right: this full native crop is the detail view used to decide the class.
Drag its divider to adjust the size. **Focus block** (F11) enlarges that preview,
keeping the class legend and labeling keys available; press it again to restore
the map and panels. Crops fit the available width and height without clipping or
changing their aspect ratio. Export, panel controls, and observation-wide progress
remain visible in both views.

**Spatial hierarchy:**
- **Panel**: a large square region (default 4096×4096 px) shown one at a time
- **Block**: the unit that gets a class label (default 512×512 px); a panel contains 8×8 = 64 blocks

## Installation

### Requirements
- Python ≥ 3.11
- GDAL ≥ 3.5 with JP2 support (JP2OpenJPEG or JP2ECW/Kakadu driver)

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

For GUI (M3+):
```bash
pip install -e '.[gui]'
```

For development:
```bash
pip install -e '.[dev]'
```

For inference (`mars-inference`), on top of `[gui]`:
```bash
pip install -e '.[infer]'
```
This pulls in `torch`. It does **not** pull in the AI4ExoMars model code
(`vision_backend`) — see [Inference](#inference-mars-inference) below.

## Quick Start

### Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Run the tool
```bash
mars-label                    # Open GUI, choose JP2 file
mars-label /path/to/file.jp2  # Open GUI with JP2 pre-loaded
```

### Run tests
```bash
pytest tests/ -v
```

### View CLI help
```bash
mars-label --help
```

## Inference (`mars-inference`)

`mars-inference` runs a trained AI4ExoMars segmentation or block-classification checkpoint over a HiRISE
observation and opens the predictions in the same window as `mars-label` — same
panels, same grid, same hotkeys — so a prediction is just a labeling session with a
head start. It is the ML counterpart to `mars-label`; nothing about labeling changes.

```bash
mars-inference /path/to/file.jp2 /path/to/model.pt
# A training run directory selects its best.pt:
mars-inference /path/to/file.jp2 ../v2_final_lr3_e12
```

What happens:
1. **Preprocessing** — same nodata/skip-detection pass as `mars-label` (progress bar); off-swath blocks are never sent through the model.
2. **Inference** — the checkpoint's model is reconstructed and run block-by-block over the observation, with a progress bar (`block N/M`). Stage-3 models use a majority vote of pixel predictions. Stage-4 models use their trained block classifier directly, including the newly learned classes.
3. **Review** — the window opens exactly like `mars-label`, with predicted blocks pre-colored by class. Relabel any block with the normal hotkeys; a block you edit gets `edit_count > 0` in the saved Parquet, so reviewed/edited predictions stay distinguishable from untouched model output.
4. **Save predictions** — press the header or sidebar's **Save predictions** button to persist the current (possibly reviewed) predictions.

### Caching

Predictions are saved to `configs/predictions/<model_stem>/<obs_id>.parquet` (+ a
`.session.json` sidecar), keyed by the checkpoint's filename stem — one subfolder per
model, so several models' predictions for the same image coexist. Re-running
`mars-inference` on the same `(image, model)` pair loads the cached result instead of
calling the model again; if the checkpoint file's size/mtime changed since the cache
was written, it's treated as stale and inference reruns automatically.

For a checkpoint called `best.pt`, the folder is `<run_directory>-best` to keep
different training runs separate. Saved predictions include confidence and the
observation's NPCA rankings. Old caches without summaries rebuild these automatically,
preserving reviewed labels. A changed NPCA sidecar also triggers rebuilding.
Pixel classification maps are temporary and require **Re-run inference** after
reopening a saved session. Uncertainty maps are always computed on demand.

### Continuing in labeling mode

Because predictions use the exact same Parquet + session-JSON format as labels, once
you've saved them you can review/edit that same file with `mars-label`: launch it,
then **File → Set Labels Folder...** and point it at
`configs/predictions/<model_stem>/`, then reopen the image. (You're actually already
in a fully editable labeling window right after `mars-inference` opens — this is only
useful for coming back to a saved prediction set in a later session.)

### Class mapping

The model's output channels are mapped to `classes.yaml` class ids via each class's
`id` (default) or an explicit `model_index:` field, when the model's channel order
doesn't match your class ids 1:1. See the comment in `configs/classes.yaml`.

### Model code (AI4ExoMars)

`mars-inference` reconstructs the model architecture from the checkpoint itself:
Stage-3 `model_state` / `config` and Stage-4 `block_state` / `seg_model_config` /
`head_config` / `classes` are supported. It uses the
`vision_backend` package from [AI4ExoMars](../AI4ExoMars). It looks for that package
already installed, then for a sibling `../AI4ExoMars` checkout; point it elsewhere
with `--ai4exomars-path` or `configs/app.yaml`'s `inference.ai4exomars_path`.

Stage-4 needs a recent backend containing `vision_backend/model/block_head.py`.
For `v2_final_lr3_e12`, the default view shows all 18 block classes. Its optional
pixel classification view uses the fine-tuned **14-class segmentation backbone**;
the four additional block classes have no pixel classifier. The UI distinguishes
these views rather than painting a block decision as a pixel prediction.

AI4ExoMars trains on 8-bit (uint8, 0 = nodata) imagery. Non-uint8 rasters (e.g.
16-bit HiRISE RDR) are handled too — see [Non-8-bit imagery](#non-8-bit-imagery)
below — but that's an approximation; uint8 input matching the training
convention is preferred when you have it.

### Performance: convert JP2s before inference

Raw JP2 windowed reads at native resolution are slow — roughly 100–250x slower
per block than a tiled GeoTIFF of the same data (measured: ~0.7s vs ~0.003s per
512×512 read). Since each block read happens on the CPU between GPU batches,
this is the #1 cause of `mars-inference` looking like "the GPU isn't being
used" — the GPU is just idle, waiting on JP2 decode. (Check with `nvidia-smi`:
low GPU utilization + high CPU + several GB allocated on the GPU means the model
*is* loaded and running on it — the bottleneck is upstream.)

Fix: convert once before running inference (a few minutes for a full HiRISE
scene; every read after that is ~250x faster, for both `mars-inference` and
`mars-label`):
```bash
python3 scripts/convert_for_inference.py /path/to/observation.JP2
# writes /path/to/observation.tif (tiled, block_size=512, deflate-compressed)
mars-inference /path/to/observation.tif model.pt
```
This mirrors what AI4ExoMars's own training pipeline does — it never reads JP2
directly either (`noahh_alignment/warp_drg.py` produces a tiled GeoTIFF up front
for exactly this reason).

### Non-8-bit imagery

Non-uint8 rasters get a *global* percentile stretch to 8-bit, computed once from
the whole image (never per-block — that would flatten real brightness
differences between blocks) before being fed to the model. This is a
best-effort approximation, not a reproduction of AI4ExoMars's DRG (Digital
Rectified Graphic) training-imagery pipeline, which involves cartographic
processing there's no exact recipe to replicate from raw calibrated data. Status
bar reports when it kicks in.

Stage-3 models get the 1st/99th percentiles mapped onto 0–255. Stage-4 block
classifiers get them mapped onto the Oxia training percentiles 58/178 instead
(clipped to 1–255, 0 kept for nodata), exactly as their expert training crops
were made (`to_uint8(..., stretch="oxia")` in AI4ExoMars). The 0–255 stretch has
about twice that contrast and quietly degrades a block classifier's predictions.

### Configuration

```yaml
inference:
  ai4exomars_path: null   # null -> auto-detect ../AI4ExoMars
  device: auto             # auto | cpu | cuda | mps
  batch_size: 4
  context_multiplier: 4    # context-branch models only: context crop = this * local window
  nodata_skip_threshold: 0.33  # blocks more nodata than this never reach the model
```

## Analysis Layers (predictions mode)

Separate spatial maps and a class summary are available in predictions mode.

### Mahalanobis distance and Laplace uncertainty

**Mahalanobis distance** and **Laplace uncertainty** each compute their map only
on the first click, then reuse it during that session.

#### Stage-4 block classifiers (one score per block)

Both maps score the 18-class block decision, from φ — the input of the block
head's last layer — which the prediction pass keeps for every block. A click
therefore costs about a millisecond per block, with no second model pass.
**Save predictions** also writes φ (`<obs>.block_features.npz`), so a cached
reload draws the maps without the model too. Only for a reload of predictions
saved before this file existed is φ extracted again, in one pass, on first click.

- **Laplace uncertainty** (`<checkpoint_stem>.laplace.pt`, the last-layer Laplace
  from AI4ExoMars `uncertainty/block_laplace.py`): Monte Carlo (1000 draws)
  predictive entropy divided by log(number of classes), 0–1. The best of the
  model's scores at flagging likely misclassifications.
- **Mahalanobis distance** (`<checkpoint_stem>.uncertainty.pt`, fitted on φ):
  distance to the nearest class Gaussian divided by the file's reference
  distance, the 95th percentile of the training blocks. Blue 0, yellow 1, red ≥ 2:
  above 1 is unlike nearly every training block (another site measures ≈ 2).

A run folder such as `v2_final_lr3_e12/` holds both files next to `best.pt`.
Its `segmenter.*` files describe the segmentation backbone alone and are not used.

#### Stage-3 segmentation models (pixel maps)

Both maps retain spatial variation
within blocks and align to the image's native pixel grid. Missing imagery stays
transparent. Maps are stored in temporary files to limit RAM and open file handles.
Pixel detail comes from the segmentation backbone's spatial features, upsampled
from decoder resolution; it does not imply an independent measurement at each
native HiRISE pixel or pixel predictions for the four new block classes.

Mahalanobis uses `<checkpoint_stem>.uncertainty.pt`, fitted for this checkpoint's
**spatial** feature distribution. It displays the raw distance to the nearest fitted
class Gaussian, using one fixed calibration scale across the image. Blue means
lower distance, red higher distance; it is not a probability of being wrong.
The backend's Stage-3 fitting script is `uncertainty/fit_gaussians.py`.

Laplace uses `<checkpoint_stem>.laplace.pt`. The supported spatial format is
`mars-spatial-laplace-diag-v1`: a dictionary with `checkpoint_sha256`, `head_name`
(`decoder.head`), and
`posterior_variance` containing the fitted diagonal weight/bias posterior variances
in the head's parameter shapes. The checkpoint is the posterior mean. The map
estimates normalized mutual information from 64 shared posterior classifier draws,
with one backbone pass per batch. Other fitted formats need a verified adapter;
a training Laplace/EWC prior is **not** this posterior. Missing or incompatible
calibration is reported explicitly. No fitted artifact is bundled with the GUI.

Separately, every `mars-inference` run also computes per-block **softmax
confidence** (`max_softmax`) during the prediction pass. For a Stage-4 model it
is calibrated: softmax(logits / T*), with T* read from the run folder's
`eval.json` (1.0 if that file is missing or names another checkpoint).
Confidence, coverage and the Neural-PCA gallery are ready immediately after
inference, without computing either uncertainty map.

### Class Summary

The Legend panel's **Class summary** button opens a per-class view with, for
every configured class:

- **Neural PCA gallery** — the top-activating image crops along each of the
  class's leading orthogonal feature directions ("what did the model learn
  about this class"). Loaded from `<checkpoint_stem>.npca.pt`, fitted offline by
  `AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py`; shown as a clearly
  labeled placeholder until that artifact exists.
  The observation's own blocks are ranked during inference using the stored bases.
  Stage-4 requires bases fitted to its 270-dimensional block-head input; these
  explain the learned linear contribution to its block logits (the segmentation
  prior is an additional contribution). A Stage-3 256-dimensional basis is not
  interchangeable. Unavailable projections show their reason in the summary.
- **Coverage** — % of this observation's blocks currently classified as this
  class. Always real: pure label-store arithmetic, works in plain `mars-label`
  sessions too (no model needed).
- **Avg. softmax confidence** (calibrated for Stage-4), **Mean Mahalanobis
  distance**, and **Mean Laplace uncertainty** — confidence is ready after
  inference; each uncertainty statistic becomes available once its corresponding
  map has been computed.

## Labeling Workflow

### Open an observation
1. Launch `mars-label`
2. Open observation, or File → Open observation (or pass path on command line)
3. Observation loads; first panel displays
4. Legend shows terrain classes; history panel lists all panels

The current block's assigned class is highlighted in orange with a checkmark in
the legend. It follows block navigation, label edits, and undo/redo, and scrolls
into view when needed. Unlabeled and no-data blocks have no highlighted class;
blocks marked Abstain highlight the Abstain row.

### Label blocks
Press the **hotkey** for the majority terrain class in the current block:
- **q** = Smooth bedrock → label + auto-advance to next block
- **w** = Fractured bedrock
- **e** = Boulder field
- **r** = Ripples/TARs
- **a** = Sand sheet
- **s** = Dust/mantled
- **d** = Crater interior
- **f** = Slope/scarp
- **Space** = Abstain/Skip (no label assigned) → auto-advance

Blocks auto-advance to the next unlabeled block in reading order.
Entering a panel from the panel list, overview or Page Up/Down selects its first
unlabeled, usable block. Completed panels remain available for review. Clicking
a particular block still selects that exact block.

The progress bar shows labeled blocks against all blocks inside the HiRISE image,
including partial edge blocks. Hover for skipped, no-data and remaining counts.
Skipped and no-data blocks do not count as labels; empty padding outside the image
is excluded from the total.

Choose **Edit → Customize class keys…** (also available in **Shortcuts**) to
change terrain and skip keys. Each key must be unique; reserved navigation/zoom
keys cannot be assigned. Use a single character, `space`, or leave a field blank
to label that class via the legend. Preferences persist across restarts and
observations in `app.shortcuts.json` beside the app config, which Git ignores.
They do not alter class IDs, labels, or the shared class definitions. **Restore
defaults**, then **Save**, returns to the keys in the class scheme.

### Navigate without labeling
- **← / →** = Move left / right
- **↑ / ↓** = Move up / down
- **PageUp / PageDown** = Previous / next panel
- **Home** = First block in current panel
- **Click block** = Jump to that block
- **Click panel in history** = Jump to that panel

### Edit existing labels
1. Navigate to a labeled block (arrow keys or click)
2. Press a different class hotkey
   - Overwrites the label (increments edit_count)
   - **Does NOT auto-advance** (review before moving)
3. Press arrow key to move to next block

### Undo / Redo
- **Ctrl+Z** = Undo last action (label, abstain, clear, edit)
- **Ctrl+Shift+Z** = Redo

### Clear a block
- **Backspace / Delete** = Clear current block back to unlabeled
- Turn on **Erase**, then drag with the left mouse button to clear multiple blocks.
  Turn it off or press a class key to return to labeling. Each stroke is one undo step.
- **Shift + drag**, then **Backspace / Delete** = Erase a rectangular selection
- Hold a class key and select blocks with the left mouse button to paint a region.

### Help
- **?** = Show keybinding cheat-sheet

## Persistence & Resume

**Windows `[Errno 9] Bad file descriptor` while labels are still visible:** keep
the app open and follow the [live-session rescue instructions](docs/windows-save-recovery.md)
before restarting to install the fix. They preserve the current snapshot in a
separate folder and verify it before reopening.

Every labeling action triggers autosave checks:
- **By label count**: every `autosave.every_n_labels` actions (default 25)
- **By time**: every `autosave.every_seconds` seconds (default 30)

Keyboard edits, undo/redo, drag painting, and selection fills all participate in
autosave. **File → Save Labels** (`Ctrl+S` / `Cmd+S`) saves immediately. Closing
the window or opening another observation saves the current labeling session;
if saving fails, closing/switching is stopped so the edits remain available.
Prediction sessions retain their explicit **Save Predictions** workflow.
Successful saves show a persistent labeled-block count, time, and destination
tooltip. A save failure opens an error dialog and leaves a visible failure
indicator; autosave errors are never confined to transient navigation messages.

When you reopen an observation:
- Session resumes at last cursor position
- All labels restored from Parquet
- Class names refreshed from the current `classes.yaml`; keep class IDs stable

If saved labels cannot be read or do not match the image, the warning offers
**Open without saved labels**. The observation opens with a fresh session, and
new labels save to the separate folder shown in the warning. Existing label
files remain intact. **Cancel** keeps the current observation open.

Saves atomically replace the Parquet file, embedding cursor, tile geometry, and
source-image identity in the same snapshot. A missing or damaged JSON sidecar
does not discard labels. New saves bind labels to the exact source file (SHA-256
plus georeferencing), rejecting a different/replaced image even with the same
filename and dimensions. Use the original source file when reopening/exporting:
re-encoding a JP2 as TIFF or changing its internal overviews changes this identity.
Legacy files without source identity can only be checked using their available
geometry and observation metadata.

## Export & Training

### Export crops and labels from the GUI

Click **Export Labels** to save the current labels, including edits that have not
been autosaved yet, as a complete dataset folder:

```text
exports/<observation>/dataset-<timestamp>/
  crops/              # Native PNG image per labeled block
  labels.csv          # Filename → class ID, target index, coordinates, dimensions
  classes.json        # Complete, explicit class/target mapping
  dataset.json        # Source identity, input scale, preprocessing, label origin
  labels.parquet      # Full label snapshot, including unlabelled/abstained blocks
  labels_coarse.tif    # GIS overview, one pixel per block
  integrity.json      # SHA-256 checksums for the exported files
  README.md           # Loading and verification instructions
```

Only blocks assigned a terrain class get a crop. Abstain, nodata, and unlabeled
blocks are excluded. PNGs preserve native **uint8 or uint16** pixels exactly,
including partial edge crops. Each label describes the whole block's majority
terrain. The original observation is not needed to read the exported folder.

Export runs with progress and cancellation. Newly generated PNGs are reopened
and compared pixel-for-pixel with their source windows. Every complete new
dataset is checked against its archived labels before publishing; failures and
cancellation leave no partial dataset.

Repeated exports of the same open observation reuse the last successful export:

- **Nothing changed:** the existing folder is verified against checksums retained
  in memory. The dialog reports **Already up to date**; no new folder or crops are
  created. Moving the cursor or saving the session does not count as a label edit.
- **Labels changed:** a complete new timestamped folder is created. Valid existing
  crops are copied without decoding the source window or encoding another PNG.
  Relabeling updates the label mapping; newly labeled blocks get new crops, and
  cleared/abstained/nodata blocks are excluded. The dialog reports reused and
  newly generated crop counts.
- **Old files are missing or damaged:** affected crops are regenerated from the
  original observation. Checksums come from the successful export held in memory,
  so rewriting an old folder's integrity manifest cannot hide swapped crops.

Previous exports remain untouched. Copied crops are independent files, so deleting
or editing an older export cannot damage a newer dataset. Reuse is scoped to the
current open observation in the current app session; restarting/reopening begins
with a fresh export. The original observation is still checked for changes.

### Load the folder for block classification

With PyTorch installed (`pip install -e '.[infer]'`):

```python
from torch.utils.data import DataLoader
from marslabeler.model.cropdataset import BlockCropDataset

dataset = BlockCropDataset("exports/OBS/dataset-<timestamp>")
batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
images = batch["image"]     # [N, 1, block_size, block_size], float32 in [-1, 1]
targets = batch["target"]   # [N], integer class indices
# logits = classifier(images)  # [N, dataset.num_classes]
# loss = torch.nn.functional.cross_entropy(logits, targets)
```

`class_id` retains the GUI's stable IDs. `target_index` is a contiguous mapping
recorded in `classes.json`, independent of which classes appear in one export.
The configured existing-model channel is recorded separately as `model_index`.
The loader also returns `block_id`, `class_id`, `original_size`, and `valid_mask`.
It checks each crop's checksum when reading it, pads edge crops on the right and
bottom, and never resizes the terrain. Check class order and ground sampling
scale when using a pretrained network; `num_classes=14` rejects exported targets
outside a 14-class classifier head.

The default input conversion is recorded in `dataset.json`: uint8 pixels are
normalized to [-1, 1]; uint16 input first uses one observation-wide percentile
stretch, matching the approximation used in `mars-inference`. This does not
reproduce the original DRG training preprocessing. A custom `transform` can
supply other intensity preprocessing while preserving crop geometry.

This is a **whole-block classification dataset**. AI4ExoMars Stage-3 segmentation
training still needs a deliberate adapter/loss for these block-level targets;
the coarse GeoTIFF is not a ready-made segmentation target raster. Context-branch
models also need surrounding imagery. Prediction-mode exports are marked
`prediction_review` and may include unreviewed predictions; `edit_count` records
edits, not scientific approval. Split train/validation data by observation or
separated regions to avoid spatial leakage.

### Verify a folder after copying it

This requires no PyTorch and no original observation:

```bash
python -m marslabeler.model.cropdataset /path/to/exported/folder
```

After reinstalling the package, the equivalent command is
`mars-verify-export /path/to/exported/folder`. Verification fails on missing,
changed, swapped, or extra files, invalid mappings, and disagreement between the
CSV and archived labels.

### Command-line crop export

The CLI uses the same exporter and validations as the GUI:

```bash
python scripts/export_labels.py /path/to/file.jp2 labels/OBS_ID.parquet \
  -o probe_set/
```

Choose a new or empty output directory. The CLI includes crops, manifests,
checksums, instructions, and the label snapshot; the GUI also includes the coarse
GeoTIFF. The coarse encoding stays unchanged: 0-based class IDs, 253=abstain,
254=nodata, 255=unlabeled.

### Verify the complete workflow on an observation

With GUI and inference dependencies installed, run:

```bash
QT_QPA_PLATFORM=offscreen python scripts/verify_labeling_workflow.py observation.JP2 \
  --output exports/workflow-check
```

This drives real Qt tile clicks and hotkeys, autosave, clear/undo/redo, close and
reopen, and the actual **Export Labels** button. It copies the resulting folder,
verifies its integrity, compares all six crops pixel-for-pixel to the source,
checks CSV/GeoTIFF alignment, and uses `BlockCropDataset` for a shuffled CPU
classification batch with a backward/optimizer step. It writes screenshots, a contact sheet,
and `verification.json`. Trial labels are mechanical QA examples, **not scientific
annotations**, and are isolated in the new output directory.

## Configuration

### `configs/app.yaml` — App behavior
```yaml
geometry:
  panel_size: 4096      # Panel display size
  block_size: 512       # Block label unit

navigation:
  advance_mode: next_unlabeled  # or next_sequential
  advance_on_edit: false        # No auto-advance when editing

skip:
  nodata_skip_threshold: 0.5    # Auto-skip if >50% nodata
  variance_skip_threshold: 0.0  # Disabled by default

autosave:
  every_n_labels: 25
  every_seconds: 30
```

### `configs/classes.yaml` — Terrain legend
Each class has: `id` (never renumber), `name`, `color` (hex), `hotkey` (single char).

## Command-line Tools

### Build overviews (for large JP2s)
```bash
python3 scripts/build_overviews.py /path/to/file.jp2
```

Significantly speeds up panel loading by building GDAL overviews.

### Export probe set
```bash
python3 scripts/export_labels.py file.jp2 labels/OBS.parquet -o probe_set/
```

## Acceptance Checklist

- ✅ Loads a 2 GB JP2 and displays without loading fully into RAM
- ✅ Legend shows each class's color, name, hotkey
- ✅ Panel canvas shows block grid overlay on imagery
- ✅ One keypress = label block, auto-advance (or abstain)
- ✅ Arrow keys move; labeled blocks can be reclassified
- ✅ Finishing a panel → jumps to next; history supports panel jumps
- ✅ Block/panel sizes are config values; class scheme is config-driven
- ✅ Labels persist (Parquet) with metadata; resume works; GeoTIFF exports aligned
- ✅ Entire workflow keyboard-only (no mouse required for labeling)

## Project Structure

```
mars-labeler/
  configs/
    app.yaml              # App configuration (geometry, behavior, export)
    classes.yaml          # Terrain class legend
  src/marslabeler/
    config.py             # YAML → typed dataclass config + validation
    classes.py            # ClassScheme: load/validate classes.yaml
    io/
      raster.py           # RasterSource: windowed/decimated JP2 reads
      overviews.py        # Overview detection and building
    model/
      grid.py             # Panel/block geometry and indexing
  tests/
    conftest.py           # Test fixtures (synthetic GeoTIFFs)
    test_config.py        # Config loading/validation tests
    test_classes.py       # Class scheme validation tests
    test_raster.py        # Raster I/O tests
    test_grid.py          # Grid geometry tests
  scripts/
    build_overviews.py    # CLI to build external overviews for JP2s
```

## Milestones

### ✅ M1 — IO core: config, classes, raster reading, geometry (COMPLETE)

**Components:**
- `config.py`: YAML → dataclass config with validation (block_size, panel_size, behavior)
- `classes.py`: ClassScheme (load/validate classes.yaml, hotkey mapping, reserved-key collision checks)
- `io/raster.py`: RasterSource (open JP2, windowed/decimated reads, nodata/variance stats)
- `model/grid.py`: Grid (panel/block geometry, index↔pixel↔map mappings, edge handling)
- Console entry: `mars-label --help`

**DoD:** 43 passing tests covering config validation, class scheme, raster reading, and grid geometry.

### ✅ M2 — Label store, session, persistence, export (COMPLETE)

**Components:**
- `model/labelstore.py`: In-memory label state, Parquet persistence, undo/redo stack
- `model/session.py`: Ties RasterSource + Grid + LabelStore; navigation logic (auto-advance, next-unlabeled, panel rollover)
- `model/export.py`: GeoTIFF export with correct geotransform, class metadata export
- Session persistence: Parquet + sidecar JSON for cursor/config resume

**DoD:** 37 passing tests (total 80). Label/abstain/edit/clear transitions work. Autosave → reopen restores cursor and labels. GeoTIFF opens in QGIS with correct geotransform. Multi-panel layouts tested.

### ✅ M3 — GUI shell & rendering (COMPLETE)

**Components:**
- `ui/render.py`: numpy→QImage, display stretch, grid/overlay/highlight compositing
- `ui/panelcanvas.py`: QGraphicsView with multi-layer composition
- `ui/sidepreview.py`: Native-res block preview
- `ui/legendpanel.py`: Class legend with colors/names/hotkeys
- `ui/historypanel.py`: Panel list with completion progress
- `ui/mainwindow.py`: Main window, layout, File→Open, status bar, worker threads

**DoD:** Read-only visualization complete. Open JP2 → see panel with grid, side preview, legend, history. Pre-seeded labels tinted. Worker thread keeps UI responsive. 24 tests (render + components).

### ✅ M4 — Interaction: keyboard labeling loop (COMPLETE)

**Components:**
- `ui/controller.py`: Keyboard input → Session mutations + UI callbacks
- Keyboard bindings: class hotkeys, arrows, PageUp/Down, Home, Backspace, Undo/Redo
- Auto-advance on label/abstain, no-advance on edit
- Autosave trigger integration (by label count or elapsed time)

**DoD:** Full labeling workflow with keyboard only. Label → auto-advance. Arrow keys navigate. Edit existing blocks. Undo/redo restores state. Panel jumps. Autosave on thresholds and resume correctly. 16 controller tests.

### 📋 M5 — Polish, QA, packaging, acceptance

Stats dialogs, probe-set export, README, keybinding cheat-sheet, performance tuning.

## Configuration

### `configs/app.yaml`

Controls app behavior:
```yaml
geometry:
  panel_size: 4096        # Panel size in pixels
  block_size: 512         # Block size; must be multiple of 32 and divide panel_size

navigation:
  advance_mode: next_unlabeled  # next_unlabeled | next_sequential
  advance_on_edit: false        # Don't auto-advance on label edits (review in place)

display:
  max_canvas_px: 1600          # Canvas size for GDAL decimation
  stretch_percentiles: [1, 99]  # Display contrast stretch (viewing only, not written to labels)

skip:
  nodata_skip_threshold: 0.5   # Auto-skip blocks with >50% nodata
  variance_skip_threshold: 0.0  # Disable low-variance skipping
  skip_low_variance: false

autosave:
  every_n_labels: 25       # Save after every N labels
  every_seconds: 30        # Save every 30 seconds

export:
  full_res: false          # Export full-resolution mask (vs. coarse grid)

labeler: null              # null → OS username; set to override
```

### `configs/classes.yaml`

Defines terrain class legend with colors and hotkeys:
```yaml
classes:
  - { id: 0,  name: "Smooth bedrock",    color: "#4C72B0", hotkey: "q" }
  - { id: 1,  name: "Fractured bedrock", color: "#DD8452", hotkey: "w" }
  # ... more classes

abstain:
  id: -1
  name: "Abstain/Skip (no label assigned)"
  color: "#000000"
  hotkey: "space"

nodata:
  id: -2
  name: "No data"
  color: "#222222"
```

## Development Notes

### Key Design Constraints (from planning)
- ✅ **Block size must be multiple of 32** (model stride)
- ✅ **Block size must divide panel size evenly**
- Reserved class IDs: -1 (abstain), -2 (nodata) — never renumber
- Never confuse display stretch with model normalization
- All label actions must be keyboard-accessible

### Testing

All tests are deterministic and use synthetic GeoTIFFs to avoid large file dependencies.

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_config.py -v

# Run a specific test
pytest tests/test_grid.py::test_grid_basic_dimensions -v
```

### Next Steps (M2)

1. Implement `LabelStore` (Parquet-based in-memory state + persistence)
2. Implement `Session` (binds RasterSource + Grid + LabelStore)
3. Implement undo/redo and autosave logic
4. Implement GeoTIFF export with aligned geotransform

See the top-level implementation plan for full details on each milestone.
