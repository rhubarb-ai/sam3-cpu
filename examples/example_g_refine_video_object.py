#!/usr/bin/env python3
"""
Example G: Refine video object segmentation
Demonstrates refining an existing tracked object using additional point prompts.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario G: Refine video object with additional prompts"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=10,
        help="Frame index for refinement annotation"
    )
    parser.add_argument(
        "--object-id",
        type=int,
        default=1,
        help="Existing object ID to refine"
    )
    parser.add_argument(
        "--points",
        nargs="+",
        type=float,
        default=None,
        help="Refinement point coordinates (x1 y1 x2 y2 ...)"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=int,
        default=None,
        help="Point labels (1=positive, 0=negative)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_g",
        help="Output directory"
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="both",
        help="Propagation direction (default: both)"
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
    
    # Default refinement points if none provided
    if args.points is None:
        points = [[350, 190]]  # One positive refinement point
        labels = [1]
    else:
        if len(args.points) % 2 != 0:
            print("Error: Points must be in pairs (x y)")
            return 1
        points = [[args.points[i], args.points[i+1]] for i in range(0, len(args.points), 2)]
        labels = args.labels if args.labels else [1] * len(points)
    
    print("=" * 70)
    print("Example G: Refine Video Object Segmentation")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Frame: {args.frame}")
    print(f"Object ID: {args.object_id}")
    print(f"Refinement Points: {points}")
    print(f"Labels: {labels}")
    print(f"Direction: {args.direction}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Refine video object segmentation
    result = sam3.refine_video_object(
        video_path=args.video,
        frame_idx=args.frame,
        object_id=args.object_id,
        points=points,
        point_labels=labels,
        output_dir=args.output,
        propagation_direction=args.direction
    )
    
    if result.success:
        print(f"✓ Success! Generated {len(result.mask_files)} refined mask video(s)")
        print(f"Output directory: {result.output_dir}")
        print(f"Object IDs: {result.object_ids}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"✗ Failed: {result.errors}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
