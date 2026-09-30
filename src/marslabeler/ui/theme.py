"""Shared visual language for the observation workspace and its dialogs."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


STYLESHEET = """
QWidget {
    background-color: #141b22; color: #e6edf3;
    font-family: "Helvetica Neue", "Segoe UI", sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog { background-color: #141b22; }
QLabel { background: transparent; border: none; }
QLabel[role="muted"] { color: #91a0b2; font-size: 11px; }
QLabel[role="section"] { color: #91a0b2; font-size: 10px; font-weight: 600; }
QLabel#brand { font-size: 16px; font-weight: 700; }
QLabel#brandMark { color: #b65d32; font-size: 26px; font-weight: 600; }
QLabel#modeBadge {
    color: #e6a079; background: #38281f; border: 1px solid #684731;
    border-radius: 4px; padding: 5px 10px; font-size: 10px; font-weight: 600;
}
QLabel#welcomeTitle { font-size: 26px; font-weight: 600; }
QLabel#welcomeEyebrow { color: #e6a079; font-size: 11px; font-weight: 600; }
QFrame#workspaceHeader { background: #10161d; border-bottom: 1px solid #2c3947; }
QFrame#viewToolbar { background: #19222c; border-bottom: 1px solid #2c3947; }
QWidget#inspector { background: #19222c; }
QWidget#emptyWorkspace { background: #0c1117; }
QPushButton, QToolButton {
    background: #202b36; border: 1px solid #344353; border-radius: 4px;
    padding: 7px 12px; font-weight: 500;
}
QPushButton:hover, QToolButton:hover { background: #2c3b48; border-color: #647a8e; }
QPushButton:pressed, QToolButton:pressed { background: #364959; }
QPushButton:checked, QToolButton:checked {
    background: #423024; border-color: #b65d32; color: #f0b993;
}
QPushButton:disabled, QToolButton:disabled { color: #637180; background: #19222c; border-color: #293440; }
QPushButton[role="primary"] { background: #a84f24; border-color: #b65d32; color: #fff4eb; font-weight: 600; }
QPushButton[role="primary"]:hover { background: #ae5528; }
QPushButton[role="primary"]:pressed { background: #8f401d; }
QPushButton[role="primary"]:disabled { color: #a18a7c; background: #413027; border-color: #413027; }
QPushButton[role="compact"] { padding: 5px 10px; }
QPushButton:focus, QToolButton:focus { border-color: #b65d32; }
QSplitter::handle { background: #2c3947; }
QSplitter::handle:hover { background: #b65d32; }
QScrollArea, QGraphicsView { border: none; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #3a4b5d; border-radius: 3px; min-height: 25px; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0; }
QScrollBar::handle:horizontal { background: #3a4b5d; border-radius: 3px; min-width: 25px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QStatusBar { background: #10161d; border-top: 1px solid #2c3947; }
QStatusBar QLabel { color: #91a0b2; font-size: 11px; padding: 3px 8px; }
QStatusBar::item { border: none; }
QMenuBar, QMenu { background: #141b22; }
QMenuBar::item { padding: 5px 10px; }
QMenu::item { padding: 7px 24px; }
QMenu::item:selected, QMenuBar::item:selected { background: #2c3b48; }
QMenu::separator { height: 1px; background: #2c3947; margin: 4px 8px; }
QToolTip { background: #24313e; color: #e6edf3; border: 1px solid #46596c; padding: 5px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
    background: #0f161d; border: 1px solid #344353; border-radius: 3px; padding: 6px;
    selection-background-color: #713d29;
}
QProgressBar { background: #10161d; border: 1px solid #2c3947; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background: #b65d32; border-radius: 2px; }
QTabWidget::pane { border: 1px solid #2c3947; }
QTabBar::tab { background: #19222c; padding: 9px 14px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e6a079; border-bottom-color: #b65d32; }
QHeaderView::section { background: #202b36; border: none; padding: 7px; }
QTableView, QListView { background: #141b22; alternate-background-color: #19222c; gridline-color: #2c3947; }
"""


def apply_theme() -> None:
    """Apply once per application, including unparented worker/help dialogs."""
    app = QApplication.instance()
    if app is None or app.property("marsWorkspaceTheme"):
        return
    app.setStyle("Fusion")
    palette = QPalette()
    for role, value in {
        QPalette.ColorRole.Window: "#141b22",
        QPalette.ColorRole.WindowText: "#e6edf3",
        QPalette.ColorRole.Base: "#0f161d",
        QPalette.ColorRole.AlternateBase: "#19222c",
        QPalette.ColorRole.Text: "#e6edf3",
        QPalette.ColorRole.Button: "#202b36",
        QPalette.ColorRole.ButtonText: "#e6edf3",
        QPalette.ColorRole.Highlight: "#713d29",
        QPalette.ColorRole.HighlightedText: "#e6edf3",
        QPalette.ColorRole.Link: "#e6a079",
    }.items():
        palette.setColor(role, QColor(value))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
    app.setProperty("marsWorkspaceTheme", True)
