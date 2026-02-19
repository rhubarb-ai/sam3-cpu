#!/usr/bin/env python3
"""
Test: New Modular API for Image Processing
Demonstrates the new Sam3API interface for image segmentation.
"""

import argparse
from pathlib import Path
from sam3 import Sam3API


def main():
    parser = argparse.ArgumentParser(
        description="Test new modular API with image processing"
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
        default="results/test_modular_api",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Test: New Modular API - Image Processing")
    print("=" * 70)
    print(f"Image: {args.image}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize new Sam3API
    api = Sam3API(output_dir=args.output)
    
    try:
        # Process image with text prompts
        print("Processing image with prompts...")
        result = api.process_image_with_prompts(
            image_path=args.image,
            prompts=args.prompts
        )
        
        print("\n✓ Success!")
        print(f"Output directory: {result['output_dir']}")
        print(f"Total images processed: {result['total_images']}")
        print(f"Total prompts per image: {result['total_prompts']}")
        
        # Print results for each image
        for img_result in result['images']:
            print(f"\nImage: {img_result['image_name']}")
            for prompt, prompt_result in img_result['prompts'].items():
                print(f"  Prompt '{prompt}': {prompt_result['num_objects']} object(s) found")
                print(f"    Masks directory: {prompt_result['masks_dir']}")
        
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
