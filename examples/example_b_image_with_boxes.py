#!/usr/bin/env python3
"""
Example B: Single image with bounding boxes
Demonstrates image segmentation using spatial coordinates.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario B: Process single image with bounding boxes"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/images/cafe.png",
        help="Path to input image"
    )
    parser.add_argument(
        "--boxes",
        nargs="+",
        type=float,
        default=None,
        help="Bounding boxes in XYWH format (space-separated: x y w h)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_b",
        help="Output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling (handled by __globals.py)"
    )
    
    args = parser.parse_args()
    
    # Default box if none provided (center region)
    boxes = [[100, 100, 200, 200]] if args.boxes is None else [args.boxes]
    
    print("=" * 70)
    print("Example B: Single Image with Bounding Boxes")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Boxes: {boxes}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Process image with bounding boxes
    result = sam3.process_image(
        image_path=args.image,
        boxes=boxes,
        output_dir=args.output
    )
    
    if result.success:
        print(f"✓ Success! Generated {len(result.mask_files)} mask(s)")
        print(f"Output directory: {result.output_dir}")
        print(f"Object IDs: {result.object_ids}")
    else:
        print(f"✗ Failed: {result.errors}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
