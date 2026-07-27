"""Console entry point for mars-label."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="mars-label",
        description="Fast keyboard-driven GUI for block-level terrain labeling of HiRISE observations",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to app config (default: configs/app.yaml)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        metavar="METERS",
        help=(
            "Classification tile resolution in meters/tile-side (e.g. 1.0). "
            "Converted to a pixel block size using the observation's native GSD "
            "(default: block_size from the app config, unchanged)"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "jp2_path",
        nargs="?",
        help="Path to JP2 file to label (optional, can open from GUI)",
    )

    args = parser.parse_args()

    # Launch GUI
    from PySide6.QtWidgets import QApplication
    from marslabeler.ui.mainwindow import MainWindow

    app = QApplication(sys.argv)

    if args.config:
        config_path = Path(args.config)
    else:
        # Try multiple paths for config
        potential_paths = [
            Path("configs/app.yaml"),
            Path(__file__).parent.parent.parent / "configs/app.yaml",
        ]
        config_path = next((p for p in potential_paths if p.exists()), Path("configs/app.yaml"))
    window = MainWindow(config_path, resolution_m=args.resolution)
    window.show()

    if args.jp2_path:
        window._load_observation(Path(args.jp2_path))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
