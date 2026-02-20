"""
SAM3 CPU - Main Entry Point

Process images or videos with text prompts for segmentation.
"""

import argparse
import sys
from pathlib import Path
from sam3 import Sam3API


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 CPU - Segment Anything Model 3 for Images and Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with default settings
  python main.py --video assets/videos/sample.mp4 --prompts player ball

  # Process video with custom output directory
  python main.py --video sample.mp4 --prompts player ball --output results/my_test

  # Process image
  python main.py --image test.jpg --prompts person car

  # Keep temporary files for debugging
  python main.py --video sample.mp4 --prompts player --keep-temp
        """
    )
    
    # Input selection (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        '--video', 
        type=str,
        default='assets/videos/sample.mp4',
        help='Path to input video file (default: assets/videos/sample.mp4)'
    )
    input_group.add_argument(
        '--image', 
        type=str,
        help='Path to input image file (if specified, processes image instead of video)'
    )
    
    # Prompts
    parser.add_argument(
        '--prompts',
        nargs='+',
        type=str,
        default=['player'],  # Can add more like: 'ball', 'tennis-court'
        help='Text prompts for segmentation (default: player)'
    )
    
    # Output directory
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/<video_name> or results/<image_name>)'
    )
    
    # Keep temporary files
    parser.add_argument(
        '--keep-temp',
        action='store_true',
        help='Keep temporary chunk files for debugging (default: False)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if args.image:
        input_path = Path(args.image)
        if not input_path.exists():
            print(f"❌ Error: Image file not found: {args.image}")
            sys.exit(1)
        processing_mode = 'image'
    else:
        input_path = Path(args.video)
        if not input_path.exists():
            print(f"❌ Error: Video file not found: {args.video}")
            sys.exit(1)
        processing_mode = 'video'
    
    # Print configuration
    print("="*70)
    print("SAM3 CPU - Processing Configuration")
    print("="*70)
    print(f"Mode: {processing_mode.upper()}")
    print(f"Input: {input_path}")
    print(f"Prompts: {', '.join(args.prompts)}")
    if args.output:
        print(f"Output: {args.output}")
    print(f"Keep temp files: {args.keep_temp}")
    print("="*70)
    print()
    
    # Initialize API
    api = Sam3API()
    
    try:
        # Process based on mode
        if processing_mode == 'video':
            result = api.process_video_with_prompts(
                video_path=str(input_path),
                prompts=args.prompts,
                output_dir=args.output,
                keep_temp_files=args.keep_temp
            )
        else:  # image
            result = api.process_image_with_prompts(
                image_path=str(input_path),
                prompts=args.prompts,
                output_dir=args.output
            )
        
        print("\n" + "="*70)
        print("✅ Processing Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Always cleanup
        api.cleanup()


if __name__ == "__main__":
    main()