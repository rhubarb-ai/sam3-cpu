#!/usr/bin/env python3
"""
Example D: Multiple images with bounding boxes
Demonstrates batch processing of images with spatial coordinates.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario D: Process multiple images with bounding boxes"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["assets/images/cafe.png", "assets/images/test_image.jpg"],
        help="Paths to input images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_d",
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
    
    # Default boxes for demonstration
    boxes = [[50, 50, 150, 150], [200, 200, 100, 100]]
    
    print("=" * 70)
    print("Example D: Batch Images with Bounding Boxes")
    print("=" * 70)
    print(f"Images: {args.images}")
    print(f"Boxes: {boxes}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Process multiple images with bounding boxes
    result = sam3.process_image(
        image_path=args.images,
        boxes=boxes,
        output_dir=args.output
    )
    
    if result.success:
        print(f"✓ Success! Generated {len(result.mask_files)} mask(s) across {len(args.images)} images")
        print(f"Output directory: {result.output_dir}")
        print(f"Object IDs: {result.object_ids}")
    else:
        print(f"✗ Failed: {result.errors}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
