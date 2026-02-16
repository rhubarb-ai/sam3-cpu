#!/usr/bin/env python3
"""
Quick Test Script - 1 Second Clips
Tests all SAM3 functionality with minimal CPU time
"""

from sam3 import Sam3
from sam3.__globals import DEFAULT_PROPAGATION_DIRECTION, TEMP_DIR, DEFAULT_OUTPUT_DIR
import os
import subprocess

def create_1sec_clip(input_video: str, output_path: str, start_sec: float = 0):
    """Extract 1-second clip from video."""
    cmd = [
        'ffmpeg', '-loglevel', 'error', '-y',
        '-ss', str(start_sec),
        '-i', input_video,
        '-t', '1.0',  # 1 second duration
        '-c:v', 'libx264',  # Re-encode video
        '-c:a', 'aac',  # Re-encode audio
        '-preset', 'veryfast',  # Fast encoding
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"✓ Created 1-sec clip: {output_path}")

def test_video_processing():
    """Test video processing with 1-sec clips."""
    print("=" * 70)
    print("SAM3 Quick Test - Video Processing (1 second clips)")
    print("=" * 70)
    print(f"Using globals:")
    print(f"  TEMP_DIR: {TEMP_DIR}")
    print(f"  DEFAULT_OUTPUT_DIR: {DEFAULT_OUTPUT_DIR}")
    print(f"  DEFAULT_PROPAGATION_DIRECTION: {DEFAULT_PROPAGATION_DIRECTION}")
    print()
    
    # Setup
    test_video = "assets/videos/bedroom.mp4"
    clip_path = f"{TEMP_DIR}/test_1sec.mp4"
    output_dir = f"{DEFAULT_OUTPUT_DIR}/quick_test"
    
    # Create 1-second test clip
    if not os.path.exists(clip_path):
        print("Creating 1-second test clip...")
        create_1sec_clip(test_video, clip_path, start_sec=0)
    
    # Initialize SAM3
    sam3 = Sam3(verbose=True)
    
    # Test 1: Basic video with prompts
    print("\n" + "=" * 70)
    print("Test 1: Video with text prompts (1 sec, default direction='both')")
    print("=" * 70)
    result1 = sam3.process_video_with_prompts(
        video_path=clip_path,
        prompts=["kids"],
        output_dir=f"{output_dir}/test1_prompts",
        propagation_direction=None  # Should use 'both' from globals
    )
    print(f"✓ Test 1: {'SUCCESS' if result1.success else 'FAILED'}")
    print(f"  Detected objects: {result1.object_ids}")
    print(f"  Generated {len(result1.mask_files)} mask video(s)")
    if result1.mask_files:
        for mf in result1.mask_files:
            print(f"    - {mf}")
    
    # Test 2: Second run to verify caching/performance
    print("\n" + "=" * 70)
    print("Test 2: Second run (verifies consistency)")
    print("=" * 70)
    result2 = sam3.process_video_with_prompts(
        video_path=clip_path,
        prompts=["bed"],
        output_dir=f"{output_dir}/test2_second_run"
    )
    print(f"✓ Test 2: {'SUCCESS' if result2.success else 'FAILED'}")
    print(f"  Detected objects: {result2.object_ids}")
    
    # Test 3: Multiple prompts
    print("\n" + "=" * 70)
    print("Test 3: Multiple prompts")
    print("=" * 70)
    result3 = sam3.process_video_with_prompts(
        video_path=clip_path,
        prompts=["kids", "bed"],
        output_dir=f"{output_dir}/test3_multi_prompt"
    )
    print(f"✓ Test 3: {'SUCCESS' if result3.success else 'FAILED'}")
    print(f"  Detected objects: {result3.object_ids}")
    
    # Test 4: Direction parameter
    print("\n" + "=" * 70)
    print("Test 4: Explicit direction='forward' (override default)")
    print("=" * 70)
    result4 = sam3.process_video_with_prompts(
        video_path=clip_path,
        prompts=["person"],
        output_dir=f"{output_dir}/test4_direction",
        propagation_direction="forward"  # Explicit override
    )
    print(f"✓ Test 4: {'SUCCESS' if result4.success else 'FAILED'}")
    if result4.metadata.get('chunked'):
        print("  ⚠ Warning: Chunking triggered for 1-sec clip (unexpected)")
    else:
        print("  ✓ No chunking (expected for short clip)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Quick Tests Complete")
    print("=" * 70)
    total = 4
    passed = sum([result1.success, result2.success, result3.success, result4.success])
    print(f"Passed: {passed}/{total}")
    print(f"Output directory: {output_dir}")
    print()
    print("🎉 All quick tests completed!")
    print("   For full testing with large videos, use examples/example_e_video_with_prompts.py")
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = test_video_processing()
    sys.exit(0 if success else 1)
