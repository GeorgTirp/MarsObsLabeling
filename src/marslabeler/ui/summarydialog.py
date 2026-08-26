"""Class summary window: per-class neural-PCA gallery + coverage/confidence/uncertainty.

Opened from the Legend panel's Summary button. For each configured class, shows:

- **Neural PCA**: the top-activating image crops for each of the class's leading
  orthogonal feature directions ("what did the model learn about this class"),
  loaded from a `<checkpoint_stem>.npca.pt` gallery artifact
  (`AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py`). Shown as a placeholder
  layout when that artifact doesn't exist yet (no trained/calibrated model).
- **Coverage**: fraction of this observation's blocks currently classified as this
  class -- always available, computed directly from the label store (no model
  needed), so this is real in both mars-label and mars-inference sessions.
- **Avg. softmax confidence** / **Avg. epistemic uncertainty**: mean per-block
  scores from the last inference run in this window, if any (see MainWindow's
  `block_confidence` / `block_uncertainty`).
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QPushButton,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from marslabeler.classes import ClassScheme
from marslabeler.model.session import Session
from marslabeler.ui.render import numpy_to_qimage

NPCA_COMPONENTS_SHOWN = 4
NPCA_THUMBNAILS_PER_COMPONENT = 3
# Thumbnails for the per-observation gallery are read from the open raster on
# demand (the offline artifact's are baked in at fit time).
LOCAL_THUMBNAIL_PX = 96


class ClickableBlockThumbnail(QLabel):
    """A per-observation thumbnail that emits its `block_id` when clicked.

    Separate from ClickableThumbnail because the payloads address different
    things: a training-corpus `source_id` names a pixel in another raster, while
    a `block_id` names a block of the observation already open.
    """

    clicked = Signal(str)  # block_id

    def __init__(self, block_id: str, parent=None):
        super().__init__(parent)
        self._block_id = block_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._block_id)
        super().mousePressEvent(event)


class ClickableThumbnail(QLabel):
    """A Neural-PCA gallery thumbnail that emits its provenance `source_id`
    when clicked -- MainWindow uses this to jump to that block's location
    (opening its source image first if a different one is currently loaded;
    see MainWindow._on_npca_thumbnail_clicked)."""

    clicked = Signal(str)  # source_id

    def __init__(self, source_id: str, parent=None):
        super().__init__(parent)
        self._source_id = source_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._source_id)
        super().mousePressEvent(event)


class ClassSummaryView(QWidget):
    """Per-class summary panel. Shown non-modally (MainWindow.show()s it,
    doesn't exec() it) so a Neural-PCA thumbnail click can navigate
    MainWindow to that block's location while this window stays open and
    visible alongside it."""

    thumbnail_clicked = Signal(str)  # source_id, forwarded from whichever ClickableThumbnail was clicked
    local_block_clicked = Signal(str)  # block_id, from the per-observation gallery
    close_requested = Signal()       # 'Back to map' pressed

    def __init__(
        self,
        classes_scheme: ClassScheme,
        session: Session,
        npca_gallery: dict | None = None,
        block_confidence: dict[str, float] | None = None,
        block_uncertainty: dict[str, float] | None = None,
        local_npca_gallery: dict | None = None,
        on_local_npca_clicked=None,
        parent=None,
    ):
        super().__init__(parent)
        self.classes_scheme = classes_scheme
        self.session = session
        self.npca_gallery = npca_gallery or {}
        self.block_confidence = block_confidence or {}
        self.block_uncertainty = block_uncertainty or {}
        self.local_npca_gallery = local_npca_gallery or {}
        if on_local_npca_clicked is not None:
            self.local_block_clicked.connect(on_local_npca_clicked)
        # Prefer this observation's own top-activating blocks when we have them:
        # clicking one navigates the image being labeled, whereas a training-corpus
        # thumbnail points into a different raster entirely.
        self.npca_source = "local" if self.local_npca_gallery else "training"

        outer = QVBoxLayout()
        self.setLayout(outer)

        # --- Header: title + return to the map ---
        header_row = QHBoxLayout()
        title = QLabel("Class Summary")
        title.setStyleSheet("font-weight: bold; font-size: 15px; color: #ffffff;")
        header_row.addWidget(title)
        header_row.addStretch()
        self.back_button = QPushButton("\u2190 Back to map")
        self.back_button.clicked.connect(self.close_requested)
        header_row.addWidget(self.back_button)
        outer.addLayout(header_row)

        # --- Gallery source selector (only meaningful when both exist) ---
        selector_row = QHBoxLayout()
        selector_label = QLabel("Neural-PCA examples from:")
        selector_label.setStyleSheet("color: #ccc; font-size: 11px;")
        selector_row.addWidget(selector_label)
        self.source_combo = QComboBox()
        self.source_combo.addItem("This observation", "local")
        self.source_combo.addItem("Training corpus", "training")
        self.source_combo.setCurrentIndex(0 if self.npca_source == "local" else 1)
        self.source_combo.setEnabled(bool(self.local_npca_gallery and self.npca_gallery))
        if not self.local_npca_gallery:
            self.source_combo.setToolTip(
                "Run inference on this observation to rank its own blocks."
            )
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        selector_row.addWidget(self.source_combo)
        selector_row.addStretch()
        outer.addLayout(selector_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Content wider than the column scrolls rather than forcing the
        # splitter to widen this pane.
        scroll.setMinimumWidth(0)
        outer.addWidget(scroll)

        container = QWidget()
        self._container_layout = QVBoxLayout()
        self._container_layout.setSpacing(12)
        container.setLayout(self._container_layout)
        scroll.setWidget(container)

        self._rebuild_sections()

    def _on_source_changed(self, index: int) -> None:
        self.npca_source = self.source_combo.itemData(index) or "training"
        self._rebuild_sections()

    def _rebuild_sections(self) -> None:
        layout = self._container_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        classes = sorted(self.classes_scheme.classes.values(), key=lambda c: c.id)
        for cls in classes:
            layout.addWidget(self._build_class_section(cls))
        layout.addStretch()

    # ------------------------------------------------------------------ #
    # Per-class stats (always computable: pure LabelStore query)
    # ------------------------------------------------------------------ #

    def _coverage(self, class_id: int) -> tuple[int, int]:
        """(count of blocks labeled/predicted as class_id, total non-nodata blocks)."""
        count = 0
        total = 0
        for record in self.session.labels.records.values():
            if record.status == "nodata":
                continue
            total += 1
            if record.class_id == class_id:
                count += 1
        return count, total

    def _mean_score(self, class_id: int, block_scores: dict[str, float]) -> float | None:
        values = [
            block_scores[block_id]
            for block_id, record in self.session.labels.records.items()
            if record.class_id == class_id and block_id in block_scores
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def _model_index_for_class(self, cls) -> int:
        """classes.yaml id -> model channel index (inverse of ClassScheme.model_index_to_id)."""
        return cls.model_index if cls.model_index is not None else cls.id

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #

    def _build_class_section(self, cls) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #1a1a1a; border: 1px solid #555; "
            "border-radius: 6px; padding: 8px; }"
        )
        layout = QVBoxLayout()
        frame.setLayout(layout)

        # --- Header: swatch + name ---
        header = QHBoxLayout()
        swatch = QLabel()
        pixmap = QPixmap(28, 28)
        pixmap.fill(QColor(cls.color))
        swatch.setPixmap(pixmap)
        swatch.setStyleSheet("border: 1px solid #666; border-radius: 3px;")
        header.addWidget(swatch)

        name = QLabel(cls.name)
        name.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffffff;")
        header.addWidget(name)
        header.addStretch()
        layout.addLayout(header)

        # --- Neural PCA row ---
        layout.addWidget(self._build_npca_row(cls))

        # --- Stats row ---
        layout.addWidget(self._build_stats_row(cls))

        return frame

    def _corpus_thumbnail(self, item):
        """Thumbnail baked into the offline artifact at fit time."""
        thumb = ClickableThumbnail(item.source_id)
        thumb.setPixmap(QPixmap.fromImage(numpy_to_qimage(item.thumbnail)))
        thumb.setToolTip(
            f"rank {item.rank}, score={item.score:.3f}\n{item.source_id}\n"
            "From the model's training corpus -- click to jump to this location\n"
            "in that image (opens it if a different one is loaded)"
        )
        thumb.clicked.connect(self.thumbnail_clicked)
        return thumb

    def _local_thumbnail(self, item):
        """Thumbnail read on demand from the observation currently open.

        Returns None if the window can't be read (edge/nodata block), so one bad
        crop degrades to a missing tile rather than an empty gallery.
        """
        from marslabeler.ui.render import apply_display_stretch

        try:
            data = self.session.raster.read_window(
                item.x_px, item.y_px, item.w_px, item.h_px,
                LOCAL_THUMBNAIL_PX, LOCAL_THUMBNAIL_PX,
            )
            if data.size == 0:
                return None
            stretched = apply_display_stretch(data, (1, 99))
        except Exception as exc:
            # One unreadable window shouldn't blank the whole gallery, but it must
            # not vanish silently either -- a systematic failure here would look
            # exactly like "the gallery stopped working".
            print(
                f"[summary] local thumbnail failed for {item.block_id} "
                f"@({item.x_px},{item.y_px}) {item.w_px}x{item.h_px}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return None

        thumb = ClickableBlockThumbnail(item.block_id)
        thumb.setPixmap(QPixmap.fromImage(numpy_to_qimage(stretched)))
        thumb.setToolTip(
            f"rank {item.rank}, score={item.score:.3f}\n"
            f"{item.block_id} -- panel {item.panel_idx}, "
            f"block {item.block_row},{item.block_col}\n"
            "In this observation -- click to jump there"
        )
        thumb.clicked.connect(self.local_block_clicked)
        return thumb

    def _build_npca_row(self, cls) -> QWidget:
        row = QWidget()
        grid = QGridLayout()
        grid.setSpacing(6)
        row.setLayout(grid)

        model_index = self._model_index_for_class(cls)
        local = self.npca_source == "local"
        class_gallery = (
            self.local_npca_gallery.get(model_index)
            if local
            else self.npca_gallery.get(model_index)
        )

        if not class_gallery:
            if local:
                text = (
                    "No blocks of this observation were predicted as this class, so "
                    "it has no local top-activating examples. Switch to \"Training "
                    "corpus\" to see what the model learned for it."
                )
            else:
                text = (
                    "Neural PCA gallery not available yet -- needs a trained model + a "
                    "calibration pass (AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py)."
                )
            placeholder = QLabel(text)
            placeholder.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
            placeholder.setWordWrap(True)
            grid.addWidget(placeholder, 0, 0)
            return row

        for component_idx in range(NPCA_COMPONENTS_SHOWN):
            col_widget = QWidget()
            col_layout = QVBoxLayout()
            col_layout.setSpacing(2)
            col_widget.setLayout(col_layout)

            title = QLabel(f"Component {component_idx + 1}")
            title.setStyleSheet("color: #8AB4F8; font-size: 10px; font-weight: bold;")
            col_layout.addWidget(title)

            thumbnails_row = QHBoxLayout()
            items = class_gallery.get(component_idx, [])
            if not items:
                missing = QLabel("--")
                missing.setStyleSheet("color: #666; font-size: 10px;")
                thumbnails_row.addWidget(missing)
            for item in items[:NPCA_THUMBNAILS_PER_COMPONENT]:
                widget = (
                    self._local_thumbnail(item) if local else self._corpus_thumbnail(item)
                )
                if widget is not None:
                    thumbnails_row.addWidget(widget)
            col_layout.addLayout(thumbnails_row)

            grid.addWidget(col_widget, 0, component_idx)

        return row

    def _build_stats_row(self, cls) -> QWidget:
        count, total = self._coverage(cls.id)
        coverage_pct = (100.0 * count / total) if total else 0.0

        avg_confidence = self._mean_score(cls.id, self.block_confidence)
        avg_uncertainty = self._mean_score(cls.id, self.block_uncertainty)

        confidence_text = f"{avg_confidence:.2f}" if avg_confidence is not None else "N/A -- run inference"
        uncertainty_text = (
            f"{avg_uncertainty:.2f}" if avg_uncertainty is not None else "N/A -- run Uncertainty Heatmap"
        )

        label = QLabel(
            f"Coverage: {coverage_pct:.1f}% ({count}/{total} blocks)   |   "
            f"Avg. softmax confidence: {confidence_text}   |   "
            f"Avg. epistemic uncertainty: {uncertainty_text}"
        )
        label.setStyleSheet("color: #cccccc; font-size: 11px; padding-top: 4px;")
        label.setWordWrap(True)
        return label


class ClassSummaryDialog(QDialog):
    """Free-floating window wrapper around :class:`ClassSummaryView`.

    MainWindow shows the summary in-window (as a page of its centre stack)
    because on macOS a non-modal child of a fullscreen/maximised window opens on
    another Space and never becomes visible. This wrapper is kept for callers
    that genuinely want a separate window, and forwards the view's signals and
    the handful of attributes callers introspect.
    """

    thumbnail_clicked = Signal(str)
    local_block_clicked = Signal(str)

    def __init__(
        self,
        classes_scheme: ClassScheme,
        session: Session,
        npca_gallery: dict | None = None,
        block_confidence: dict[str, float] | None = None,
        block_uncertainty: dict[str, float] | None = None,
        local_npca_gallery: dict | None = None,
        on_local_npca_clicked=None,
        parent=None,
    ):
        super().__init__(parent)
        self.view = ClassSummaryView(
            classes_scheme,
            session,
            npca_gallery=npca_gallery,
            block_confidence=block_confidence,
            block_uncertainty=block_uncertainty,
            local_npca_gallery=local_npca_gallery,
            on_local_npca_clicked=on_local_npca_clicked,
        )
        self.view.back_button.setVisible(False)  # a window closes by its own chrome
        self.view.thumbnail_clicked.connect(self.thumbnail_clicked)
        self.view.local_block_clicked.connect(self.local_block_clicked)
        self.view.close_requested.connect(self.close)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.setWindowTitle("Class Summary")
        self.resize(900, 700)

    def __getattr__(self, name: str):
        """Delegate anything not found on the dialog to the wrapped view.

        Keeps the pre-split surface (`npca_source`, `source_combo`, `_coverage`,
        `_mean_score`, `_model_index_for_class`, ...) reachable on the dialog.
        Only called for genuine misses, so it never shadows QDialog's own API.
        """
        try:
            view = object.__getattribute__(self, "view")
        except AttributeError:  # during __init__, before self.view exists
            raise AttributeError(name) from None
        return getattr(view, name)
