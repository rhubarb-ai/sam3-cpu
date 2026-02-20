"""Quick IoU test on lossless PNG masks"""
import cv2
import numpy as np
from pathlib import Path

# Paths
chunk0_dir = Path("/tmp/sam3-cpu/sample/chunks/chunk_0/masks/player")
chunk1_dir = Path("/tmp/sam3-cpu/sample/chunks/chunk_1/masks/player")

print("="*70)
print("LOSSLESS PNG MASK IoU TEST")
print("="*70)

# Test each object
for obj_id in range(3):
    obj0_dir = chunk0_dir / f"object_{obj_id}"
    obj1_dir = chunk1_dir / f"object_{obj_id}"
    
    if not obj0_dir.exists() or not obj1_dir.exists():
        continue
    
    # Get last PNG from chunk_0
    pngs_0 = sorted(obj0_dir.glob("frame_*.png"))
    # Get first PNG from chunk_1
    pngs_1 = sorted(obj1_dir.glob("frame_*.png"))
    
    if not pngs_0 or not pngs_1:
        continue
    
    # Read masks
    last_mask_0 = cv2.imread(str(pngs_0[-1]), cv2.IMREAD_GRAYSCALE)
    first_mask_1 = cv2.imread(str(pngs_1[0]), cv2.IMREAD_GRAYSCALE)
    
    # Binarize
    mask_0_bin = (last_mask_0 > 0).astype(np.uint8)
    mask_1_bin = (first_mask_1 > 0).astype(np.uint8)
    
    # Compute IoU
    intersection = np.logical_and(mask_0_bin, mask_1_bin).sum()
    union = np.logical_or(mask_0_bin, mask_1_bin).sum()
    iou = intersection / union if union > 0 else 0.0
    
    print(f"\nObject {obj_id}:")
    print(f"  Last frame chunk_0: {pngs_0[-1].name}")
    print(f"  First frame chunk_1: {pngs_1[0].name}")
    print(f"  IoU = {iou:.4f}")
    
    if iou >= 0.9:
        print(f"  ✅ EXCELLENT! IoU >= 0.9 (lossless working!)")
    elif iou >= 0.8:
        print(f"  ⚠️  Good but could be better (0.8-0.9)")
    else:
        print(f"  ❌ Poor IoU < 0.8 (compression still an issue)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("With lossless H.264 (qp=0) chunks + PNG masks:")
print("  - Expected IoU: >= 0.9 for same object")
print("  - Previous lossy: 0.6-0.8")
print("="*70)
