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

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (
    QDialog,
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


class ClassSummaryDialog(QDialog):
    """Non-modal-ish (but exec()'d like other dialogs here) per-class summary window."""

    def __init__(
        self,
        classes_scheme: ClassScheme,
        session: Session,
        npca_gallery: dict | None = None,
        block_confidence: dict[str, float] | None = None,
        block_uncertainty: dict[str, float] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.classes_scheme = classes_scheme
        self.session = session
        self.npca_gallery = npca_gallery or {}
        self.block_confidence = block_confidence or {}
        self.block_uncertainty = block_uncertainty or {}

        self.setWindowTitle("Class Summary")
        self.resize(900, 700)

        outer = QVBoxLayout()
        self.setLayout(outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.setSpacing(12)
        container.setLayout(container_layout)
        scroll.setWidget(container)

        classes = sorted(classes_scheme.classes.values(), key=lambda c: c.id)
        for cls in classes:
            container_layout.addWidget(self._build_class_section(cls))
        container_layout.addStretch()

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

    def _build_npca_row(self, cls) -> QWidget:
        row = QWidget()
        grid = QGridLayout()
        grid.setSpacing(6)
        row.setLayout(grid)

        model_index = self._model_index_for_class(cls)
        class_gallery = self.npca_gallery.get(model_index)

        if not class_gallery:
            placeholder = QLabel(
                "Neural PCA gallery not available yet -- needs a trained model + a "
                "calibration pass (AI4ExoMars/vision_backend/pc_align/fit_neural_pca.py)."
            )
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
                thumb_label = QLabel()
                qimage = numpy_to_qimage(item.thumbnail)
                thumb_label.setPixmap(QPixmap.fromImage(qimage))
                thumb_label.setToolTip(f"rank {item.rank}, score={item.score:.3f}\n{item.source_id}")
                thumbnails_row.addWidget(thumb_label)
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
