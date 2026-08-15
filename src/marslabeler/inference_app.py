"""Console entry point for mars-inference."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="mars-inference",
        description=(
            "Run a trained AI4ExoMars segmentation model over a HiRISE observation, "
            "then open the predictions in the same labeling UI for review/editing."
        ),
    )
    parser.add_argument("jp2_path", help="Path to the JP2/GeoTIFF observation to predict on")
    parser.add_argument("model_path", help="Path to the trained model checkpoint (.pt)")
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
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Override configs/app.yaml inference.device",
    )
    parser.add_argument(
        "--ai4exomars-path",
        type=str,
        default=None,
        help=(
            "Path to an AI4ExoMars checkout providing 'vision_backend', if it isn't "
            "installed and isn't a sibling '../AI4ExoMars' directory next to this repo. "
            "Overrides configs/app.yaml inference.ai4exomars_path."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    jp2_path = Path(args.jp2_path)
    model_path = Path(args.model_path)
    if not jp2_path.exists():
        parser.error(f"JP2 file not found: {jp2_path}")
    if not model_path.exists():
        parser.error(f"Model checkpoint not found: {model_path}")

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

    window = MainWindow(config_path, resolution_m=args.resolution, predictions_mode=True)
    if args.device:
        window.config.inference.device = args.device
    if args.ai4exomars_path:
        window.config.inference.ai4exomars_path = args.ai4exomars_path

    window.show()
    window.load_for_inference(jp2_path, model_path)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
