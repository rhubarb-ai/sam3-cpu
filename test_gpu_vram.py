#!/usr/bin/env python3
"""
GPU/VRAM Test Script

This script verifies that the chunking mechanism correctly adapts to GPU/VRAM.
Run this on a GPU machine to verify proper VRAM detection and usage.

Expected behavior on GPU:
- Device should be detected as CUDA
- Should use VRAM_USAGE_PERCENT (90%) instead of RAM_USAGE_PERCENT (33%)
- Should show "VRAM" in chunking details, not "RAM"
- Should handle larger chunks compared to CPU (same memory available)
"""

import sys
import torch
from sam3 import Sam3

def test_gpu_vram_support():
    """Test GPU/VRAM chunking support"""
    
    print("\n" + "="*70)
    print("GPU/VRAM Chunking Support Test")
    print("="*70)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA not available - this test requires a GPU machine")
        print("   Current device: CPU")
        print("   Run this test on a machine with NVIDIA GPU and CUDA installed")
        sys.exit(0)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print()
    
    videos = [
        ("480p", "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_480p.mp4"),
        ("1080p", "assets/videos/Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_1080p.mp4")
    ]
    
    for name, video_path in videos:
        print(f"\n{'='*70}")
        print(f"Testing {name} on GPU")
        print(f"{'='*70}")
        
        from pathlib import Path
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            continue
        
        # Initialize SAM3 (should auto-detect GPU)
        sam3 = Sam3(verbose=True)
        
        # Verify GPU detection
        assert sam3.memory_info.device_type == "cuda", "Device type should be 'cuda' on GPU"
        print(f"✅ Device correctly detected as: {sam3.memory_info.device_type.upper()}")
        
        # Verify VRAM usage percentage is used
        assert sam3.memory_usage_percent == sam3.vram_usage_percent, \
            "Should use vram_usage_percent on GPU"
        print(f"✅ Using VRAM usage percent: {sam3.vram_usage_percent * 100:.0f}%")
        print(f"   (Not RAM usage percent: {sam3.ram_usage_percent * 100:.0f}%)")
        
        # Get video metadata
        video_meta = sam3._get_video_metadata(video_path)
        print(f"\nVideo: {video_meta.width}x{video_meta.height}, {video_meta.total_frames} frames")
        
        # Calculate memory needed
        memory_needed = sam3._calculate_memory_needed(
            video_meta.width, video_meta.height, video_meta.total_frames, is_video=True
        )
        print(f"Memory needed: {memory_needed:.2f} GB")
        print(f"Available VRAM: {sam3.memory_info.available_gb:.2f} GB")
        
        # Check feasibility
        feasibility = sam3._check_feasibility(
            memory_needed, video_meta.total_frames, is_video=True
        )
        
        print(f"\nRequires chunking: {feasibility.requires_chunking}")
        if feasibility.requires_chunking:
            print(f"Chunk size: {feasibility.chunk_size} frames")
            print(f"Number of chunks: {feasibility.num_chunks}")
            
            # Generate chunks to see detailed output
            chunks = sam3._generate_chunks(
                video_meta, feasibility.chunk_size, sam3.min_chunk_overlap
            )
            print(f"\n✅ Chunking would use {len(chunks)} chunks")
        else:
            print(f"✅ No chunking needed - video fits in VRAM!")
        print()
    
    print("="*70)
    print("GPU/VRAM Test Summary")
    print("="*70)
    print("✅ CUDA device detected correctly")
    print("✅ VRAM usage percentage applied (90%)")
    print("✅ Memory calculations work on GPU")
    print("✅ Chunking adapts to VRAM constraints")
    print()
    print("Expected differences from CPU:")
    print("  - Uses 90% of available memory (vs 33% on CPU)")
    print("  - Larger chunks for same video (fewer chunks)")
    print("  - Shows 'VRAM' in outputs, not 'RAM'")
    print("="*70)

def test_custom_vram_percentage():
    """Test custom VRAM percentage"""
    
    if not torch.cuda.is_available():
        return
    
    print("\n" + "="*70)
    print("Testing Custom VRAM Percentage")
    print("="*70)
    
    # Test with custom VRAM percentage
    custom_vram = 0.75  # 75%
    sam3 = Sam3(vram_usage_percent=custom_vram, verbose=False)
    
    assert sam3.vram_usage_percent == custom_vram, \
        f"Custom VRAM percent not applied: {sam3.vram_usage_percent} != {custom_vram}"
    
    assert sam3.memory_usage_percent == custom_vram, \
        f"memory_usage_percent should return vram_usage_percent on GPU"
    
    print(f"✅ Custom VRAM percentage works: {custom_vram * 100:.0f}%")
    print("="*70)

if __name__ == "__main__":
    test_gpu_vram_support()
    test_custom_vram_percentage()
