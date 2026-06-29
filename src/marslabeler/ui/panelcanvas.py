"""Panel canvas: QGraphicsView displaying panel image with overlays."""

from typing import Optional, Callable

import numpy as np
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QPixmap, QImage, QColor, QPen, QBrush
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem

from marslabeler.ui.render import (
    numpy_to_qimage,
    apply_display_stretch,
    create_grid_overlay,
    create_block_overlay,
    create_current_block_highlight,
)


class PanelCanvas(QGraphicsView):
    """QGraphicsView for displaying a panel image with grid and label overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setBackgroundBrush(Qt.GlobalColor.black)

        # Layer management
        self.image_item: Optional[QGraphicsPixmapItem] = None
        self.grid_item: Optional[QGraphicsPixmapItem] = None
        self.label_overlay_item: Optional[QGraphicsPixmapItem] = None
        self.highlight_item: Optional[QGraphicsPixmapItem] = None
        self.selection_item: Optional[QGraphicsRectItem] = None

        # Current state
        self.canvas_width = 1600
        self.canvas_height = 1600
        self.block_width = 200  # Will be updated
        self.block_height = 200  # Will be updated
        self.blocks_per_row = 8
        self.blocks_per_col = 8

        # Interaction callbacks
        self.on_block_clicked: Optional[Callable[[int, int], None]] = None
        # Drag-paint: called for each block while the left button is held
        # (is_start=True on the initial press of the stroke)
        self.on_block_paint: Optional[Callable[[int, int, bool], None]] = None
        # Drag-paint finished (mouse released)
        self.on_block_paint_end: Optional[Callable[[], None]] = None
        # Marquee selection finished: (r0, c0, r1, c1) inclusive block range
        self.on_selection_made: Optional[Callable[[int, int, int, int], None]] = None
        self._last_paint_cell: Optional[tuple[int, int]] = None

        # Shift+drag marquee selection state
        self._selecting = False
        self._sel_anchor: Optional[tuple[int, int]] = None

    def set_panel_image(self, panel_data: np.ndarray, stretch_percentiles=(1, 99)) -> None:
        """
        Set the panel image (decimated to canvas size).

        Args:
            panel_data: Grayscale image array (will be displayed at canvas size)
            stretch_percentiles: (low, high) for display stretch
        """
        # Apply display stretch
        stretched = apply_display_stretch(panel_data, stretch_percentiles)

        # Convert to QImage
        qimage = numpy_to_qimage(stretched)

        # Scale to canvas size if needed
        if qimage.width() != self.canvas_width or qimage.height() != self.canvas_height:
            qimage = qimage.scaled(self.canvas_width, self.canvas_height, Qt.AspectRatioMode.IgnoreAspectRatio)

        # Update or create image item
        pixmap = QPixmap.fromImage(qimage)
        if self.image_item is None:
            self.image_item = self.scene.addPixmap(pixmap)
        else:
            self.image_item.setPixmap(pixmap)

        self._update_view_size()

    def set_grid(self, blocks_per_row: int, blocks_per_col: int) -> None:
        """
        Set grid parameters and create grid overlay.

        Args:
            blocks_per_row, blocks_per_col: Grid dimensions
        """
        self.blocks_per_row = blocks_per_row
        self.blocks_per_col = blocks_per_col
        self.block_width = self.canvas_width // blocks_per_row
        self.block_height = self.canvas_height // blocks_per_col

        # Create and set grid overlay
        grid_pixmap = create_grid_overlay(
            self.canvas_width,
            self.canvas_height,
            self.block_width,
            QColor(0, 255, 0),
            line_width=1,
        )

        if self.grid_item is None:
            self.grid_item = self.scene.addPixmap(grid_pixmap)
        else:
            self.grid_item.setPixmap(grid_pixmap)

    def set_label_overlay(self, block_data: np.ndarray, class_colors: dict[int, str]) -> None:
        """
        Set the label overlay (colored blocks for assigned classes).

        Args:
            block_data: 2D array of class IDs (blocks_per_col x blocks_per_row)
            class_colors: Dict mapping class_id to hex color
        """
        overlay_pixmap = create_block_overlay(
            self.canvas_width,
            self.canvas_height,
            self.block_width,
            self.block_height,
            block_data,
            class_colors,
            alpha=0.25,
        )

        if self.label_overlay_item is None:
            self.label_overlay_item = self.scene.addPixmap(overlay_pixmap)
        else:
            self.label_overlay_item.setPixmap(overlay_pixmap)

    def set_current_block_highlight(self, block_row: int, block_col: int) -> None:
        """
        Set the current block highlight.

        Args:
            block_row, block_col: Current block position in grid
        """
        highlight_pixmap = create_current_block_highlight(
            self.canvas_width,
            self.canvas_height,
            self.block_width,
            self.block_height,
            block_row,
            block_col,
            QColor(255, 255, 0),
            line_width=3,
        )

        if self.highlight_item is None:
            self.highlight_item = self.scene.addPixmap(highlight_pixmap)
        else:
            self.highlight_item.setPixmap(highlight_pixmap)

    def _update_view_size(self) -> None:
        """Update view to fit scene."""
        self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _block_at(self, pos) -> tuple[int, int]:
        """Map a widget position to a (block_row, block_col), clamped to the grid."""
        scene_pos = self.mapToScene(pos)
        block_col = int(scene_pos.x() / self.block_width)
        block_row = int(scene_pos.y() / self.block_height)
        block_col = max(0, min(block_col, self.blocks_per_row - 1))
        block_row = max(0, min(block_row, self.blocks_per_col - 1))
        return block_row, block_col

    def set_selection_rect(self, r0: int, c0: int, r1: int, c1: int) -> None:
        """Draw/update the yellow marquee covering blocks [r0..r1] x [c0..c1]."""
        x = c0 * self.block_width
        y = r0 * self.block_height
        w = (c1 - c0 + 1) * self.block_width
        h = (r1 - r0 + 1) * self.block_height

        if self.selection_item is None:
            pen = QPen(QColor(255, 255, 0), 2, Qt.PenStyle.SolidLine)
            brush = QBrush(QColor(255, 255, 0, 60))  # translucent yellow fill
            self.selection_item = self.scene.addRect(x, y, w, h, pen, brush)
            self.selection_item.setZValue(10)  # above overlays
        else:
            self.selection_item.setRect(x, y, w, h)

    def clear_selection(self) -> None:
        """Remove the marquee selection rectangle."""
        if self.selection_item is not None:
            self.scene.removeItem(self.selection_item)
            self.selection_item = None

    def mousePressEvent(self, event) -> None:
        """Shift+drag = marquee selection; plain = navigate + start paint stroke."""
        block_row, block_col = self._block_at(event.pos())
        self._last_paint_cell = (block_row, block_col)

        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift:
            # Begin a rubber-band selection
            self._selecting = True
            self._sel_anchor = (block_row, block_col)
            self.set_selection_rect(block_row, block_col, block_row, block_col)
            return

        self._selecting = False
        if self.on_block_clicked:
            self.on_block_clicked(block_row, block_col)
        # Begin a potential drag-paint stroke (handler no-ops if no class is held)
        if self.on_block_paint:
            self.on_block_paint(block_row, block_col, True)

    def mouseMoveEvent(self, event) -> None:
        """Extend the marquee, or drag-paint blocks while LMB is held."""
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        block_row, block_col = self._block_at(event.pos())

        if self._selecting and self._sel_anchor is not None:
            r0, r1 = sorted((self._sel_anchor[0], block_row))
            c0, c1 = sorted((self._sel_anchor[1], block_col))
            self.set_selection_rect(r0, c0, r1, c1)
            return

        cell = (block_row, block_col)
        if cell != self._last_paint_cell:
            self._last_paint_cell = cell
            if self.on_block_paint:
                self.on_block_paint(block_row, block_col, False)

    def mouseReleaseEvent(self, event) -> None:
        """Finalize the marquee selection, or end a drag-paint stroke."""
        if self._selecting and self._sel_anchor is not None:
            block_row, block_col = self._block_at(event.pos())
            r0, r1 = sorted((self._sel_anchor[0], block_row))
            c0, c1 = sorted((self._sel_anchor[1], block_col))
            self.set_selection_rect(r0, c0, r1, c1)  # keep it highlighted
            self._selecting = False
            self._sel_anchor = None
            if self.on_selection_made:
                self.on_selection_made(r0, c0, r1, c1)
            return

        self._last_paint_cell = None
        if self.on_block_paint_end:
            self.on_block_paint_end()

    def resizeEvent(self, event) -> None:
        """Handle resize to maintain fit."""
        super().resizeEvent(event)
        if self.scene and self.scene.items():
            self.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
