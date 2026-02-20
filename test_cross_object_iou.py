"""Test cross-object IoU matching (object_i in chunk_0 vs all objects in chunk_1)"""
import cv2
import numpy as np
from pathlib import Path

# Paths
chunk0_dir = Path("/tmp/sam3-cpu/sample/chunks/chunk_0/masks/player")
chunk1_dir = Path("/tmp/sam3-cpu/sample/chunks/chunk_1/masks/player")

print("="*70)
print("CROSS-OBJECT IoU TEST (Finding Best Matches)")
print("="*70)

chunk0_objects = sorted([d for d in chunk0_dir.iterdir() if d.is_dir()])
chunk1_objects = sorted([d for d in chunk1_dir.iterdir() if d.is_dir()])

print(f"\nChunk 0 objects: {[d.name for d in chunk0_objects]}")
print(f"Chunk 1 objects: {[d.name for d in chunk1_objects]}")

# For each object in chunk_0, find best match in chunk_1
for obj0_dir in chunk0_objects:
    obj0_id = obj0_dir.name
    
    # Get last PNG from chunk_0
    pngs_0 = sorted(obj0_dir.glob("frame_*.png"))
    if not pngs_0:
        continue
    
    last_mask_0 = cv2.imread(str(pngs_0[-1]), cv2.IMREAD_GRAYSCALE)
    mask_0_bin = (last_mask_0 > 0).astype(np.uint8)
    
    print(f"\n{obj0_id} (chunk 0):")
    best_iou = 0
    best_match = None
    
    # Test against all objects in chunk_1
    for obj1_dir in chunk1_objects:
        obj1_id = obj1_dir.name
        
        # Get first PNG from chunk_1
        pngs_1 = sorted(obj1_dir.glob("frame_*.png"))
        if not pngs_1:
            continue
        
        first_mask_1 = cv2.imread(str(pngs_1[0]), cv2.IMREAD_GRAYSCALE)
        mask_1_bin = (first_mask_1 > 0).astype(np.uint8)
        
        # Compute IoU
        intersection = np.logical_and(mask_0_bin, mask_1_bin).sum()
        union = np.logical_or(mask_0_bin, mask_1_bin).sum()
        iou = intersection / union if union > 0 else 0.0
        
        print(f"  vs {obj1_id} (chunk 1): IoU = {iou:.4f}")
        
        if iou > best_iou:
            best_iou = iou
            best_match = obj1_id
    
    if best_match:
        print(f"  ✅ Best match: {obj0_id} -> {best_match} (IoU = {best_iou:.4f})")
        if best_iou >= 0.9:
            print(f"     ✅ EXCELLENT! Lossless implementation working!")
        elif best_iou >= 0.8:
            print(f"     ⚠️  Good (0.8-0.9)")
        else:
            print(f"     ❌ Poor (< 0.8)")

print("\n" + "="*70)
