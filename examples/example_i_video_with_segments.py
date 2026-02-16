#!/usr/bin/env python3
"""
Example I: Video with segment-based prompts
Demonstrates processing different segments of a video with different prompts.
"""

import argparse
from pathlib import Path
from sam3 import Sam3


def main():
    parser = argparse.ArgumentParser(
        description="Scenario I: Process video with segment-based prompts"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4",
        help="Path to input video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/example_i",
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
    
    # Define segments with different prompts
    # Segment 1: First 2 seconds with text prompts
    # Segment 2: 2-5 seconds with point prompts
    segments = {
        "segments": [
            {
                "start_time_sec": 0.0,
                "end_time_sec": 2.0,
                "prompts": ["person"]
            },
            {
                "start_time_sec": 2.0,
                "end_time_sec": 5.0,
                "points": [[320, 180], [400, 200]],
                "labels": [1, 1]
            }
        ]
    }
    
    print("=" * 70)
    print("Example I: Video with Segment-based Prompts")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Number of segments: {len(segments['segments'])}")
    print(f"Output: {args.output}")
    print()
    
    for i, seg in enumerate(segments["segments"], 1):
        print(f"Segment {i}:")
        if "start_time_sec" in seg:
            print(f"  Time: {seg['start_time_sec']:.2f}s - {seg['end_time_sec']:.2f}s")
        else:
            print(f"  Frames: {seg['start_frame']} - {seg['end_frame']}")
        
        if "prompts" in seg:
            print(f"  Prompts: {seg['prompts']}")
        if "points" in seg:
            print(f"  Points: {len(seg['points'])} points")
    print()
    
    # Initialize SAM3
    sam3 = Sam3(verbose=args.verbose)
    
    # Process video with segments
    result = sam3.process_video_with_segments(
        video_path=args.video,
        segments=segments,
        output_dir=args.output
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
