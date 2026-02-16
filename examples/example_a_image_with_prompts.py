#!/usr/bin/env python3
"""
Example A: Single image with text prompts
Demonstrates basic image segmentation using text descriptions.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario A: Process single image with text prompts"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/images/truck.jpg",
        help="Path to input image"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["truck", "wheel"],
        help="Text prompts for segmentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_a",
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
    
    print("=" * 70)
    print("Example A: Single Image with Text Prompts")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Process image with text prompts
    result = sam3.process_image(
        image_path=args.image,
        prompts=args.prompts,
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
