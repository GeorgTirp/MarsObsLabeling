# Observation workspace redesign

The existing PySide6 application now uses a consistent dark visual system,
inspired by [MMGIS's imagery-centered workspace](https://nasa-ammos.github.io/MMGIS/).
This is a desktop interface redesign built on the existing Python core.

## Interface

- Observation identity, open, and save actions share a persistent header.
- The image toolbar exposes zoom, fit-to-panel, and shortcut help.
- Terrain classes use neutral rows, exact class-color swatches, and keycaps.
- Panel navigation uses compact progress indicators and a burnt-orange active state.
- The block inspector scales with the sidebar. Partial crops keep their aspect
  ratio; resizing uses a cached pixmap without another raster read.
- View/analysis actions and session actions have separate groups. The inspector
  scrolls when necessary, including when widened on a short screen.
- The class summary stays inside the main window, with styling consistent with
  the rest of the workspace. Toolbar zoom returns to the image first.
- The opening state explains how to begin and provides an open-observation action.

The four-pane splitter remains resizable. Keyboard labeling, region selection,
painting, undo/redo, autosave, explicit prediction saving, export, pixel/block
rendering, uncertainty, and Neural PCA navigation retain their existing handlers.
JP2 and GeoTIFF are both exposed in the file picker. Scientific class colors,
saved formats, model code, and native source pixels are unchanged.

## Validation

111 tests passed across layout, UI components, class interactions, labeling and
persistence workflows, inference controls, prediction rendering, Neural PCA
navigation, summary, prediction dialog, and raster extent checks. The inference
checks used the existing AI4ExoMars environment with PyTorch installed. These are
regression tests, not a new trained-model inference benchmark.

Added checks cover the first-show 1440 × 900 window geometry in both modes and
the real mouse/keyboard path from summary through toolbar zoom/fit to labeling
and saving. Visual review used the existing ESP_092084_1525_RED observation at
1440 × 900 and 1920 × 1080, including the empty state, overview, and summary.
Colored blocks in review screenshots are illustrative QA assignments held in
memory, not scientific labels. Production labels were not edited.

## Performance follow-up

The processing architecture remains in Python. Raster access, label/session
storage, and inference already have separate modules. A future performance pass
can avoid rereading unchanged panel imagery after each label, refresh progress
only for affected panels, and prefetch native block previews. A browser frontend
would be a separate migration requiring explicit workflow-parity work.
