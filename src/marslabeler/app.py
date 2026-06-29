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

    # TODO: Launch the GUI (after M3)
    print("Mars Obs Labeler v0.1.0")
    if args.jp2_path:
        print(f"Will open: {args.jp2_path}")
    print("(GUI not yet implemented)")


if __name__ == "__main__":
    main()
