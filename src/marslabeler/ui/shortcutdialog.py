"""Edit class shortcuts without changing class IDs or shared definitions."""

from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from marslabeler.classes import ClassScheme


class ShortcutDialog(QDialog):
    def __init__(self, scheme: ClassScheme, defaults: ClassScheme, parent=None):
        super().__init__(parent)
        self.scheme = scheme
        self.defaults = defaults
        self.selected_scheme = scheme
        self.setWindowTitle("Customize class keys")
        self.resize(560, 650)
        layout = QVBoxLayout(self)
        hint = QLabel(
            "Choose one character per class, or type 'space'. Leave a field blank "
            "to use the legend only. Keys are saved locally and apply to every observation. "
            "Navigation, zoom, help and undo shortcuts stay fixed."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        self.fields = {}
        for cls in [*scheme.classes.values(), scheme.abstain]:
            field = QLineEdit(cls.hotkey or "")
            field.setMaximumWidth(90)
            field.setAccessibleName(f"Key for {cls.name}")
            name = QLabel(cls.name)
            name.setWordWrap(True)
            name.setMinimumWidth(350)
            form.addRow(name, field)
            self.fields[cls.id] = field
            field.textChanged.connect(self._validate)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        reset = QPushButton("Restore defaults")
        reset.clicked.connect(self._restore_defaults)
        self.buttons.addButton(reset, QDialogButtonBox.ButtonRole.ResetRole)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._validate()

    def _restore_defaults(self):
        for cls in [*self.defaults.classes.values(), self.defaults.abstain]:
            if cls.id in self.fields:
                self.fields[cls.id].setText(cls.hotkey or "")

    def _validate(self):
        try:
            self.selected_scheme = self.scheme.with_hotkeys({
                cid: field.text() for cid, field in self.fields.items()
            })
        except ValueError as error:
            self.error_label.setText(str(error))
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)
        else:
            self.error_label.setText("")
            self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
