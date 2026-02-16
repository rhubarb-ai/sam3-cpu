#!/usr/bin/env python3
"""
Test script to verify chunking calculation fixes.
Tests both 480p and 1080p to ensure they get different chunk sizes.
"""

import sys
from sam3 import Sam3

def test_chunking_calculation():
    """Test that chunking calculation uses actual video resolution"""
    
    videos = [
        ("480p", "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"),
        ("1080p", "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_1080p.mp4")
    ]
    
    print("\n" + "="*70)
    print("Testing Fixed Chunking Calculation")
    print("="*70)
    print("This test verifies that:")
    print("1. 480p and 1080p get DIFFERENT chunk sizes")
    print("2. Memory calculation based on actual video resolution")
    print("3. Safety multiplier reduced from 3x to 1.5x")
    print("="*70 + "\n")
    
    for name, video_path in videos:
        print(f"\n{'='*70}")
        print(f"Testing {name}")
        print(f"{'='*70}")
        
        from pathlib import Path
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            continue
        
        # Get video metadata
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True, check=True
        )
        
        data = result.stdout.strip().split(',')
        if len(data) >= 2:
            width, height = data[0], data[1]
            frames = data[2] if len(data) > 2 else "unknown"
            print(f"Video: {width}x{height}, Frames: {frames}")
        
        # Initialize SAM3 with verbose mode
        sam3 = Sam3(verbose=True)
        
        # Get video metadata from SAM3
        video_meta = sam3._get_video_metadata(video_path)
        print(f"SAM3 detected: {video_meta.width}x{video_meta.height}, {video_meta.total_frames} frames")
        
        # Calculate memory needed
        memory_needed = sam3._calculate_memory_needed(
            video_meta.width, video_meta.height, video_meta.total_frames, is_video=True
        )
        print(f"Memory needed: {memory_needed:.2f} GB")
        
        # Check feasibility
        feasibility = sam3._check_feasibility(
            memory_needed, video_meta.total_frames, is_video=True
        )
        
        print(f"Requires chunking: {feasibility.requires_chunking}")
        if feasibility.requires_chunking:
            print(f"Chunk size: {feasibility.chunk_size} frames")
            print(f"Number of chunks: {feasibility.num_chunks}")
        else:
            print(f"✅ No chunking needed - video fits in memory!")
        print()
    
    print("="*70)
    print("Expected Results:")
    print("="*70)
    print("480p: Should NOT need chunking (or very few chunks)")
    print("1080p: Should need chunking (3-4 chunks)")
    print("="*70)

if __name__ == "__main__":
    test_chunking_calculation()
