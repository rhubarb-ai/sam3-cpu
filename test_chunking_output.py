#!/usr/bin/env python3
"""
Test script to verify detailed chunking output is shown correctly.
This script just loads a large video to trigger chunking and verify the output format.
"""

import sys
from sam3 import Sam3

def test_chunking_output():
    """Test that chunking output shows detailed information"""
    
    # Use 480p video - full video should trigger chunking
    video_path = "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"
    
    print("\n" + "="*70)
    print("Testing Detailed Chunking Output")
    print("="*70)
    print(f"Video: {video_path}")
    print("This test will start processing to show detailed chunking info.")
    print("Press Ctrl+C after seeing the chunking details to exit.")
    print("="*70 + "\n")
    
    # Initialize SAM3 with verbose mode to see detailed output
    sam3 = Sam3(verbose=True)
    
    try:
        from pathlib import Path
        
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
        
        # Now trigger the chunking by starting to process
        print("Calling process_video_with_prompts to trigger chunking output...\n")
        
        result = sam3.process_video_with_prompts(
            video_path=video_path,
            prompts=["person"],
            propagation_direction="both"
        )
        
        print("\n" + "="*70)
        print("✅ Test completed successfully!")
        print("You should see detailed chunking information above including:")
        print("  - Video name")
        print("  - Resolution")
        print("  - FPS")
        print("  - Total frames")
        print("  - Available RAM/VRAM")
        print("  - Memory needed")
        print("  - RAM/VRAM usage percent")
        print("  - RAM-safe frames per chunk")
        print("  - Overlap")
        print("  - Stride")
        print("  - Number of chunks")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_chunking_output()
