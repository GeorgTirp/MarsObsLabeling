# Mars Obs Labeling Tool

A fast, keyboard-driven GUI for block-level terrain labeling of HiRISE observations.

## Overview

This tool enables rapid supervised labeling of Mars terrain features on gigapixel HiRISE scenes. The labeler views a large panel of imagery, presses a single key for each block's majority terrain class, and the cursor auto-advances to the next block. Minimal mouse movement and maximum keyboard speed are the core design goals.

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

`mars-inference` runs a trained AI4ExoMars segmentation checkpoint over a HiRISE
observation and opens the predictions in the same window as `mars-label` — same
panels, same grid, same hotkeys — so a prediction is just a labeling session with a
head start. It is the ML counterpart to `mars-label`; nothing about labeling changes.

```bash
mars-inference /path/to/file.jp2 /path/to/model.pt
```

What happens:
1. **Preprocessing** — same nodata/skip-detection pass as `mars-label` (progress bar); off-swath blocks are never sent through the model.
2. **Inference** — the checkpoint's model is reconstructed and run block-by-block over the observation, with a progress bar (`block N/M`). Each block gets one class: the model's per-pixel argmax, majority-voted across the block.
3. **Review** — the window opens exactly like `mars-label`, with predicted blocks pre-colored by class. Relabel any block with the normal hotkeys; a block you edit gets `edit_count > 0` in the saved Parquet, so reviewed/edited predictions stay distinguishable from untouched model output.
4. **Save Predictions** — press the "💾 Save Predictions" button to persist the current (possibly reviewed) predictions.

### Caching

Predictions are saved to `configs/predictions/<model_stem>/<obs_id>.parquet` (+ a
`.session.json` sidecar), keyed by the checkpoint's filename stem — one subfolder per
model, so several models' predictions for the same image coexist. Re-running
`mars-inference` on the same `(image, model)` pair loads the cached result instead of
calling the model again; if the checkpoint file's size/mtime changed since the cache
was written, it's treated as stale and inference reruns automatically.

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

`mars-inference` reconstructs the model architecture from the checkpoint itself (it
expects the `{"model_state", "config": {"model": {...}}}` format written by
`AI4ExoMars/vision_backend/train_stage3_segmentation_finetune.py`) using the
`vision_backend` package from [AI4ExoMars](../AI4ExoMars). It looks for that package
already installed, then for a sibling `../AI4ExoMars` checkout; point it elsewhere
with `--ai4exomars-path` or `configs/app.yaml`'s `inference.ai4exomars_path`.

Imagery must be 8-bit (uint8, 0 = nodata) — the same convention AI4ExoMars trains on.

### Configuration

```yaml
inference:
  ai4exomars_path: null   # null -> auto-detect ../AI4ExoMars
  device: auto             # auto | cpu | cuda | mps
  batch_size: 4
  context_multiplier: 4    # context-branch models only: context crop = this * local window
```

## Analysis Layers (predictions mode)

Two extra ways to look at a model, beyond its class predictions. Both build on
analysis code in `AI4ExoMars/vision_backend` (`uncertainty/` and `pc_align/`).

### Uncertainty Heatmap

The **🌡 Uncertainty Heatmap** button (right panel, predictions mode only) swaps
the class-color overlay for a per-block heatmap (blue = confident, red =
uncertain) of *epistemic* uncertainty — Mahalanobis distance from the model's
feature representation to the nearest class's fitted Gaussian, a standard
out-of-distribution (OOD) detector. This needs a calibration artifact fitted
offline from a trained checkpoint (`AI4ExoMars/vision_backend/uncertainty/fit_gaussians.py`),
conventionally saved as `<checkpoint_stem>.uncertainty.pt` next to the checkpoint.
Until that exists for a given model, the button reports so clearly (with the
command to fit it) instead of guessing.

Separately, every `mars-inference` run also computes per-block **softmax
confidence** (`1 - max_softmax`) — no calibration needed, available for any
model — feeding the Summary window's confidence stat even before an uncertainty
artifact exists.

### Class Summary

The Legend panel's **📊 Summary** button opens a per-class window with, for
every configured class:

- **Neural PCA gallery** — the top-activating image crops along each of the
  class's leading orthogonal feature directions ("what did the model learn
  about this class"). Loaded from `<checkpoint_stem>.npca.pt`, fitted offline by
  `AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py`; shown as a clearly
  labeled placeholder until that artifact exists.
- **Coverage** — % of this observation's blocks currently classified as this
  class. Always real: pure label-store arithmetic, works in plain `mars-label`
  sessions too (no model needed).
- **Avg. softmax confidence** / **Avg. epistemic uncertainty** — mean per-block
  scores from the last inference / uncertainty-heatmap run in this window.
  "N/A" until that's been run at least once.

## Labeling Workflow

### Open an observation
1. Launch `mars-label`
2. File → Open JP2 (or pass path on command line)
3. Observation loads; first panel displays
4. Legend shows terrain classes; history panel lists all panels

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
- **Space** = Abstain (uncertain/mixed) → auto-advance

Blocks auto-advance to the next unlabeled block in reading order.

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

### Help
- **?** = Show keybinding cheat-sheet (future implementation)

## Persistence & Resume

Every labeling action triggers autosave checks:
- **By label count**: every `autosave.every_n_labels` actions (default 25)
- **By time**: every `autosave.every_seconds` seconds (default 30)

When you reopen an observation:
- Session resumes at last cursor position
- All labels restored from Parquet
- Warning if `classes.yaml` changed since last session

## Export & Training

### Export labeled blocks as probe set
```bash
python3 scripts/export_labels.py /path/to/file.jp2 labels/OBS_ID.parquet \
  -o probe_set/
```

Generates:
- `crops/` — PNG images (one per labeled block)
- `labels.csv` — Block coordinates and class IDs
- `classes.json` — Class metadata

### Export to GeoTIFF
Done automatically on exit (future: manual export button). Produces coarse-grid GeoTIFF aligned to source CRS, ready for QGIS.

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
  name: "Abstain"
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
