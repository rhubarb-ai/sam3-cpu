#!/usr/bin/env python3
"""
Example E: Video with text prompts
Demonstrates video segmentation using text descriptions with automatic propagation.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario E: Process video with text prompts"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["person", "tennis racket"],
        help="Text prompts for segmentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_e",
        help="Output directory"
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward", "both"],
        default="both",
        help="Propagation direction (default: both)"
    )
    parser.add_argument(
        "--frame-from",
        type=str,
        default=None,
        help="Start frame/time (e.g., 100, 45.5, or 1:30)"
    )
    parser.add_argument(
        "--frame-to",
        type=str,
        default=None,
        help="End frame/time (e.g., 200, 90.5, or 3:00)"
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
    print("Example E: Video with Text Prompts")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Prompts: {args.prompts}")
    print(f"Direction: {args.direction}")
    if args.frame_from or args.frame_to:
        print(f"Frame range: {args.frame_from or 'start'} to {args.frame_to or 'end'}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Parse frame/time parameters
    frame_from = None
    frame_to = None
    
    if args.frame_from:
        # Try parsing as int (frame), fallback to string (time)
        try:
            frame_from = int(args.frame_from)
        except ValueError:
            try:
                frame_from = float(args.frame_from)
            except ValueError:
                frame_from = args.frame_from  # Keep as string for timestamp
    
    if args.frame_to:
        # Try parsing as int (frame), fallback to string (time)
        try:
            frame_to = int(args.frame_to)
        except ValueError:
            try:
                frame_to = float(args.frame_to)
            except ValueError:
                frame_to = args.frame_to  # Keep as string for timestamp
    
    # Process video with text prompts
    result = sam3.process_video_with_prompts(
        video_path=args.video,
        prompts=args.prompts,
        output_dir=args.output,
        propagation_direction=args.direction,
        frame_from=frame_from,
        frame_to=frame_to
    )
    
    if result.success:
        print(f"✓ Success! Generated {len(result.mask_files)} mask video(s)")
        print(f"Output directory: {result.output_dir}")
        print(f"Object IDs: {result.object_ids}")
        print(f"Metadata: {result.metadata}")
    else:
        print(f"✗ Failed: {result.errors}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())
