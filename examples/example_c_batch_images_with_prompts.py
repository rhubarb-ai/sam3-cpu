#!/usr/bin/env python3
"""
Example C: Multiple images with text prompts
Demonstrates batch processing of images with text descriptions.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario C: Process multiple images with text prompts"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["assets/images/truck.jpg", "assets/images/groceries.jpg"],
        help="Paths to input images"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["truck", "food"],
        help="Text prompts for segmentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_c",
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
    print("Example C: Batch Images with Text Prompts")
    print("=" * 70)
    print(f"Images: {args.images}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Process multiple images with text prompts
    result = sam3.process_image(
        image_path=args.images,
        prompts=args.prompts,
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
