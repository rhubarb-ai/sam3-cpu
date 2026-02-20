"""
Test script to verify chunk overlap frames are identical.

This checks:
1. Original video: last frame of "chunk 0 region" vs first frame of "chunk 1 region"
2. Chunk videos: last frame of chunk_0.mp4 vs first frame of chunk_1.mp4
3. Mask videos: last frame of mask vs first frame of next mask
"""

import cv2
import numpy as np
from pathlib import Path
import json

def compare_frames(frame1, frame2, name1="Frame 1", name2="Frame 2"):
    """Compare two frames and return difference metrics."""
    if frame1.shape != frame2.shape:
        print(f"  Shape mismatch: {frame1.shape} vs {frame2.shape}")
        return False
    
    # Compute pixel-wise difference
    diff = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_different_pixels = np.sum(diff > 0)
    total_pixels = diff.size
    percent_different = (num_different_pixels / total_pixels) * 100
    
    print(f"\n  Comparing {name1} vs {name2}:")
    print(f"    Max pixel diff: {max_diff:.2f}")
    print(f"    Mean pixel diff: {mean_diff:.4f}")
    print(f"    Different pixels: {num_different_pixels}/{total_pixels} ({percent_different:.2f}%)")
    
    if max_diff == 0:
        print(f"    ✅ IDENTICAL (pixel-perfect)")
        return True
    else:
        print(f"    ❌ NOT IDENTICAL (lossy compression detected)")
        return False


def test_chunk_overlap():
    """Test if chunk overlap frames are identical."""
    
    print("="*70)
    print("CHUNK OVERLAP VERIFICATION TEST")
    print("="*70)
    
    # Paths
    results_dir = Path("results/sample")
    original_video = Path("assets/videos/sample.mp4")
    chunks_dir = results_dir / "temp_files" / "chunks"
    metadata_path = results_dir / "metadata" / "video_metadata.json"
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print(f"\nVideo: {metadata['video']}")
    print(f"Total frames: {metadata['nb_frames']}")
    print(f"FPS: {metadata['fps']}")
    
    # Get chunk boundaries
    chunks = metadata['chunks']
    print(f"\nChunks defined: {len(chunks)}")
    for chunk in chunks[:3]:  # Show first 3
        print(f"  Chunk {chunk['chunk']}: frames {chunk['start']}-{chunk['end']}")
    
    # Test 1: Original video overlap frames
    print("\n" + "="*70)
    print("TEST 1: Original Video - Overlap Frames Should Be IDENTICAL")
    print("="*70)
    
    cap_orig = cv2.VideoCapture(str(original_video))
    
    chunk_0_end = chunks[0]['end']  # Should be frame 24
    chunk_1_start = chunks[1]['start']  # Should be frame 24
    
    print(f"\nChunk 0 ends at frame {chunk_0_end}")
    print(f"Chunk 1 starts at frame {chunk_1_start}")
    
    if chunk_0_end == chunk_1_start:
        print(f"✅ Overlap confirmed: frame {chunk_0_end} should be identical in both chunks")
        
        # Read frame from original video
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, chunk_0_end)
        ret1, frame_overlap = cap_orig.read()
        
        if ret1:
            print(f"\n✅ Successfully read frame {chunk_0_end} from original video")
            print(f"   Shape: {frame_overlap.shape}")
        else:
            print(f"❌ Failed to read frame {chunk_0_end}")
            return
    else:
        print(f"❌ No overlap? Chunk 0 ends at {chunk_0_end}, Chunk 1 starts at {chunk_1_start}")
        return
    
    cap_orig.release()
    
    # Test 2: Chunk videos (if they exist)
    print("\n" + "="*70)
    print("TEST 2: Chunk Videos - Check For Lossy Compression")
    print("="*70)
    
    chunk_0_path = chunks_dir / "chunk_0" / "chunk_0.mp4"
    chunk_1_path = chunks_dir / "chunk_1" / "chunk_1.mp4"
    
    if not chunk_0_path.exists():
        print(f"\n⚠️  Chunk videos not found (they get deleted after processing)")
        print(f"   Expected: {chunk_0_path}")
        print(f"   This is normal - chunks are cleaned up after use")
    else:
        print(f"\n✅ Found chunk videos, testing...")
        
        cap_0 = cv2.VideoCapture(str(chunk_0_path))
        cap_1 = cv2.VideoCapture(str(chunk_1_path))
        
        # Get last frame of chunk_0
        total_frames_0 = int(cap_0.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_0.set(cv2.CAP_PROP_POS_FRAMES, total_frames_0 - 1)
        ret0, last_frame_chunk0 = cap_0.read()
        
        # Get first frame of chunk_1
        cap_1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, first_frame_chunk1 = cap_1.read()
        
        if ret0 and ret1:
            # Compare chunk frames to original overlap frame
            print(f"\n📊 Chunk 0 last frame vs Original frame {chunk_0_end}:")
            compare_frames(last_frame_chunk0, frame_overlap, 
                         f"Chunk 0 last frame", f"Original frame {chunk_0_end}")
            
            print(f"\n📊 Chunk 1 first frame vs Original frame {chunk_1_start}:")
            compare_frames(first_frame_chunk1, frame_overlap,
                         f"Chunk 1 first frame", f"Original frame {chunk_1_start}")
            
            print(f"\n📊 Chunk 0 last vs Chunk 1 first:")
            identical = compare_frames(last_frame_chunk0, first_frame_chunk1,
                                     "Chunk 0 last", "Chunk 1 first")
            
            if not identical:
                print(f"\n⚠️  LOSSY COMPRESSION DETECTED IN CHUNK VIDEOS!")
                print(f"   The chunks are re-encoded with libx264 CRF 23")
                print(f"   This causes pixel differences even for overlap frames")
        
        cap_0.release()
        cap_1.release()
    
    # Test 3: Mask videos
    print("\n" + "="*70)
    print("TEST 3: Mask Videos - Check For Lossy Compression")
    print("="*70)
    
    masks_dir = results_dir / "temp_files" / "chunks"
    chunk0_masks = masks_dir / "chunk_0" / "masks" / "player"
    chunk1_masks = masks_dir / "chunk_1" / "masks" / "player"
    
    if not chunk0_masks.exists() or not chunk1_masks.exists():
        print(f"\n⚠️  Mask videos not found")
        print(f"   Expected: {chunk0_masks}")
        return
    
    # Find matching objects (object_0, object_1, object_2)
    mask_files_0 = sorted(chunk0_masks.glob("object_*.mp4"))
    mask_files_1 = sorted(chunk1_masks.glob("object_*.mp4"))
    
    print(f"\nChunk 0 masks: {len(mask_files_0)} objects")
    print(f"Chunk 1 masks: {len(mask_files_1)} objects")
    
    # Test first mask
    if mask_files_0 and mask_files_1:
        mask0_path = mask_files_0[0]
        mask1_path = mask_files_1[0]
        
        print(f"\nTesting: {mask0_path.name} vs {mask1_path.name}")
        
        cap_mask0 = cv2.VideoCapture(str(mask0_path))
        cap_mask1 = cv2.VideoCapture(str(mask1_path))
        
        # Get last frame of first mask
        total_frames_m0 = int(cap_mask0.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_mask0.set(cv2.CAP_PROP_POS_FRAMES, total_frames_m0 - 1)
        ret0, last_mask_frame = cap_mask0.read()
        
        # Get first frame of second mask
        cap_mask1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret1, first_mask_frame = cap_mask1.read()
        
        if ret0 and ret1:
            # Convert to grayscale if needed
            if len(last_mask_frame.shape) == 3:
                last_mask_frame = cv2.cvtColor(last_mask_frame, cv2.COLOR_BGR2GRAY)
            if len(first_mask_frame.shape) == 3:
                first_mask_frame = cv2.cvtColor(first_mask_frame, cv2.COLOR_BGR2GRAY)
            
            # Compute IoU
            last_binary = (last_mask_frame > 0).astype(np.uint8)
            first_binary = (first_mask_frame > 0).astype(np.uint8)
            
            intersection = np.logical_and(last_binary, first_binary).sum()
            union = np.logical_or(last_binary, first_binary).sum()
            iou = intersection / union if union > 0 else 0.0
            
            print(f"\n📊 Mask IoU (last frame of chunk_0 vs first frame of chunk_1):")
            print(f"   IoU = {iou:.4f}")
            
            if iou < 0.95:
                print(f"   ⚠️  IoU < 0.95: LOSSY MP4 COMPRESSION DEGRADING MASKS")
                print(f"   The mp4v codec is causing mask quality loss")
                print(f"   This explains why we need threshold 0.6 instead of 0.9+")
            else:
                print(f"   ✅ IoU > 0.95: Masks are high quality")
            
            # Also check pixel-level
            compare_frames(last_mask_frame, first_mask_frame,
                         "Last mask frame", "First mask frame")
        
        cap_mask0.release()
        cap_mask1.release()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print("""
If overlap frames are NOT pixel-identical:

ROOT CAUSE:
  1. Chunk extraction uses libx264 with CRF 23 (lossy)
  2. Mask storage uses mp4v codec (lossy)
  3. Both cause quality degradation

SOLUTION:
  1. Extract chunks with lossless codec (ffv1 or png sequence)
  2. Store masks as PNG images (lossless, one file per frame)
  3. This will give IoU > 0.95 for actual matches

BENEFITS:
  - Pixel-perfect overlap frames
  - IoU threshold can be 0.9+ instead of 0.6
  - More accurate object tracking across chunks
  - Faster matching (PNG read is faster than video seek)
    """)


if __name__ == "__main__":
    test_chunk_overlap()
