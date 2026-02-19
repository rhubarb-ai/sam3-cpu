#!/usr/bin/env python3
"""
Test: New Modular API for Video Processing
Demonstrates the new Sam3API interface for video segmentation with chunking.
"""

import argparse
from pathlib import Path
from sam3 import Sam3API


def main():
    parser = argparse.ArgumentParser(
        description="Test new modular API with video processing"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["person"],
        help="Text prompts for segmentation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/test_modular_api_video",
        help="Output directory"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--chunk-spread",
        type=str,
        choices=["even", "default"],
        default="default",
        help="Chunking strategy"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Test: New Modular API - Video Processing")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Chunk spread: {args.chunk_spread}")
    print(f"Keep temp files: {args.keep_temp}")
    print()
    
    # Initialize new Sam3API
    api = Sam3API(
        output_dir=args.output,
        device=args.device
    )
    
    try:
        # Process video with text prompts
        print("Processing video with prompts...")
        result = api.process_video_with_prompts(
            video_path=args.video,
            prompts=args.prompts,
            chunk_spread=args.chunk_spread,
            keep_temp_files=args.keep_temp
        )
        
        print("\n✓ Success!")
        print(f"Video: {result['video_name']}")
        print(f"Output directory: {result['output_dir']}")
        print(f"Number of chunks: {result['num_chunks']}")
        print(f"Prompts processed: {result['prompts']}")
        print(f"Metadata: {result['metadata_path']}")
        
        # Print chunk results
        for chunk_result in result['chunks']:
            print(f"\nChunk {chunk_result['chunk_id']}:")
            print(f"  Chunk video: {chunk_result['chunk_video_path']}")
            print(f"  Prompts processed: {chunk_result['num_prompts']}")
            for prompt, prompt_result in chunk_result['prompts'].items():
                print(f"    Prompt '{prompt}': {prompt_result['num_objects']} object(s) found")
        
        # Clean up
        api.cleanup()
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
