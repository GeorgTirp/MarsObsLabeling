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

## Quick Start

### Run tests
```bash
pytest tests/ -v
```

### View CLI help
```bash
mars-label --help
```

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
