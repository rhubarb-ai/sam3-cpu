#!/usr/bin/env python3
"""
Run all SAM3 examples (Scenarios A-I)
This script demonstrates all 9 segmentation scenarios in sequence.
"""

import argparse
import sys
from pathlib import Path


def run_example(name: str, script: str, args: list, verbose: bool = False, profile: bool = False):
    """Run a single example script."""
    import subprocess
    
    print("\n" + "=" * 80)
    print(f"Running Example {name}")
    print("=" * 80)
    
    cmd = [sys.executable, script] + args
    if verbose:
        cmd.append("--verbose")
    if profile:
        cmd.append("--profile")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Example {name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Example {name} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run all SAM3 examples (Scenarios A-I)"
    )
    parser.add_argument(
        "--video-resolution",
        choices=["480p", "720p", "1080p"],
        default="480p",
        help="Video resolution for examples (default: 480p)"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        default=[],
        help="Skip specific examples"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["a", "b", "c", "d", "e", "f", "g", "h", "i"],
        default=None,
        help="Run only specific examples"
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="results",
        help="Base output directory for all examples"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for all examples"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling for all examples"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running examples even if one fails"
    )
    
    args = parser.parse_args()
    
    # Video path based on resolution
    video_path = f"assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_{args.video_resolution}.mp4"
    
    examples_dir = Path(__file__).parent
    
    # Define all examples with their configurations
    examples = {
        "a": {
            "name": "A: Single Image with Text Prompts",
            "script": str(examples_dir / "example_a_image_with_prompts.py"),
            "args": ["--output", f"{args.output_base}/example_a"]
        },
        "b": {
            "name": "B: Single Image with Bounding Boxes",
            "script": str(examples_dir / "example_b_image_with_boxes.py"),
            "args": ["--output", f"{args.output_base}/example_b"]
        },
        "c": {
            "name": "C: Batch Images with Text Prompts",
            "script": str(examples_dir / "example_c_batch_images_with_prompts.py"),
            "args": ["--output", f"{args.output_base}/example_c"]
        },
        "d": {
            "name": "D: Batch Images with Bounding Boxes",
            "script": str(examples_dir / "example_d_batch_images_with_boxes.py"),
            "args": ["--output", f"{args.output_base}/example_d"]
        },
        "e": {
            "name": "E: Video with Text Prompts",
            "script": str(examples_dir / "example_e_video_with_prompts.py"),
            "args": ["--video", video_path, "--output", f"{args.output_base}/example_e"]
        },
        "f": {
            "name": "F: Video with Point Prompts",
            "script": str(examples_dir / "example_f_video_with_points.py"),
            "args": ["--video", video_path, "--output", f"{args.output_base}/example_f"]
        },
        "g": {
            "name": "G: Refine Video Object",
            "script": str(examples_dir / "example_g_refine_video_object.py"),
            "args": ["--video", video_path, "--output", f"{args.output_base}/example_g"]
        },
        "h": {
            "name": "H: Remove Video Objects",
            "script": str(examples_dir / "example_h_remove_video_objects.py"),
            "args": ["--video", video_path, "--output", f"{args.output_base}/example_h"]
        },
        "i": {
            "name": "I: Video with Segment Prompts",
            "script": str(examples_dir / "example_i_video_with_segments.py"),
            "args": ["--video", video_path, "--output", f"{args.output_base}/example_i"]
        }
    }
    
    # Determine which examples to run
    if args.only:
        examples_to_run = {k: v for k, v in examples.items() if k in args.only}
    else:
        examples_to_run = {k: v for k, v in examples.items() if k not in args.skip}
    
    print("=" * 80)
    print("SAM3 Examples Runner")
    print("=" * 80)
    print(f"Video resolution: {args.video_resolution}")
    print(f"Output base: {args.output_base}")
    print(f"Examples to run: {', '.join(examples_to_run.keys()).upper()}")
    print(f"Profiling: {'ENABLED' if args.profile else 'DISABLED'}")
    print(f"Continue on error: {args.continue_on_error}")
    print()
    
    # Run examples
    results = {}
    for key, example in examples_to_run.items():
        success = run_example(
            key.upper(),
            example["script"],
            example["args"],
            verbose=args.verbose,
            profile=args.profile
        )
        results[key] = success
        
        if not success and not args.continue_on_error:
            print("\n✗ Stopping due to error (use --continue-on-error to continue)")
            break
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} Example {key.upper()}: {examples[key]['name']}")
    
    print()
    print(f"Results: {successful}/{total} examples completed successfully")
    
    return 0 if successful == total else 1


if __name__ == "__main__":
    exit(main())
