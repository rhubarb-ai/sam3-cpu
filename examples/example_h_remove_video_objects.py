#!/usr/bin/env python3
"""
Example H: Remove objects from video segmentation
Demonstrates removing unwanted objects by their IDs.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario H: Remove objects from video segmentation"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--object-ids",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Object IDs to remove"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_h",
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
    print("Example H: Remove Video Objects")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Object IDs to remove: {args.object_ids}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Remove objects from video segmentation
    result = sam3.remove_video_objects(
        video_path=args.video,
        object_ids=args.object_ids,
        output_dir=args.output
    )
    
    if result.success:
        print(f"✓ Success! Removed {len(args.object_ids)} object(s)")
        print(f"Output directory: {result.output_dir}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"✗ Failed: {result.errors}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
