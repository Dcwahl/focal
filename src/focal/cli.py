"""Command-line interface for focus stacking - testing tool."""

import argparse
import sys
import time
from pathlib import Path

import cv2

from focal.core.stacker import FocusStacker, StackAlgorithm

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def get_image_paths(folder: Path) -> list[Path]:
    """Get sorted list of image paths from a folder."""
    paths = [
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]
    return sorted(paths)


def run_stack(
    image_paths: list[Path],
    algorithm: StackAlgorithm,
    output_path: Path,
    skip_alignment: bool = False,
) -> float:
    """
    Run focus stacking and save result.

    Returns elapsed time in seconds.
    """
    stacker = FocusStacker(algorithm=algorithm, skip_alignment=skip_alignment)

    start = time.perf_counter()
    result = stacker.stack(image_paths)
    elapsed = time.perf_counter() - start

    # Determine output format from extension
    ext = output_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext in (".tiff", ".tif"):
        cv2.imwrite(str(output_path), result)
    else:
        # Default to JPEG
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return elapsed


def main(argv: list[str] | None = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Focus stacking CLI for testing",
        prog="focal.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # stack command
    stack_parser = subparsers.add_parser("stack", help="Stack images from a folder")
    stack_parser.add_argument(
        "input",
        type=Path,
        help="Folder containing source images",
    )
    stack_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path (supports .jpg, .png, .tiff)",
    )
    stack_parser.add_argument(
        "--algorithm",
        choices=["laplacian", "wavelet"],
        default="laplacian",
        help="Stacking algorithm (default: laplacian)",
    )
    stack_parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both algorithms and save with suffixed names",
    )
    stack_parser.add_argument(
        "--align",
        action="store_true",
        default=None,
        help="Enable ECC alignment (default: on for wavelet, off for laplacian)",
    )
    stack_parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable ECC alignment",
    )

    args = parser.parse_args(argv)

    if args.command == "stack":
        return cmd_stack(args)

    return 1


def cmd_stack(args: argparse.Namespace) -> int:
    """Handle the stack command."""
    input_folder = args.input.resolve()
    output_path = args.output.resolve()

    # Validate input folder
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}", file=sys.stderr)
        return 1

    # Get image paths
    image_paths = get_image_paths(input_folder)
    if not image_paths:
        print(f"Error: No images found in {input_folder}", file=sys.stderr)
        return 1

    print(f"Found {len(image_paths)} images in {input_folder.name}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine alignment setting
    # --align forces on, --no-align forces off, otherwise use algorithm default
    def get_skip_alignment(algo_name: str) -> bool:
        if args.no_align:
            return True  # skip alignment
        if args.align:
            return False  # don't skip (i.e., do align)
        # Default: wavelet aligns, laplacian doesn't
        return algo_name == "laplacian"

    if args.compare:
        # Run both algorithms
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent

        for algo_name, algo_enum in [
            ("laplacian", StackAlgorithm.LAPLACIAN),
            ("wavelet", StackAlgorithm.COMPLEX_WAVELET),
        ]:
            out_file = parent / f"{stem}_{algo_name}{suffix}"
            skip_align = get_skip_alignment(algo_name)
            align_status = "no-align" if skip_align else "align"
            print(f"Running {algo_name} ({align_status})...")
            try:
                elapsed = run_stack(image_paths, algo_enum, out_file, skip_alignment=skip_align)
                print(f"  Saved: {out_file}")
                print(f"  Time: {elapsed:.2f}s")
            except Exception as e:
                print(f"Error running {algo_name}: {e}", file=sys.stderr)
                return 1
    else:
        # Run single algorithm
        algo_enum = (
            StackAlgorithm.COMPLEX_WAVELET
            if args.algorithm == "wavelet"
            else StackAlgorithm.LAPLACIAN
        )
        skip_align = get_skip_alignment(args.algorithm)
        align_status = "no-align" if skip_align else "align"
        print(f"Running {args.algorithm} ({align_status})...")
        try:
            elapsed = run_stack(image_paths, algo_enum, output_path, skip_alignment=skip_align)
            print(f"Saved: {output_path}")
            print(f"Time: {elapsed:.2f}s")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
