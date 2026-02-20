"""
Test IoU matching between consecutive chunks

This script tests the IoU matching logic used by the PostProcessor
to understand why object mappings are not being created properly.
"""

import numpy as np
from pathlib import Path
from PIL import Image


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two binary masks.
    
    Args:
        mask1: First binary mask (0 or 1)
        mask2: Second binary mask (0 or 1)
    
    Returns:
        IoU score between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def load_png_mask(png_path: Path) -> np.ndarray:
    """Load PNG mask and convert to binary (0 or 1)."""
    try:
        pil_image = Image.open(png_path)
        mask_image = np.array(pil_image)
        # Convert to binary (0 or 1)
        return (mask_image > 0).astype(np.uint8)
    except Exception as e:
        print(f"  Error loading {png_path}: {e}")
        return None


def test_chunk_matching():
    """Test IoU matching between chunk_0 object_1 and all chunk_1 objects."""
    
    # Use results directory where masks are preserved
    chunk_0_dir = Path("/home/prashant/repo/public/sam3-cpu/results/sample/temp_files/chunks/chunk_0/masks/player")
    chunk_1_dir = Path("/home/prashant/repo/public/sam3-cpu/results/sample/temp_files/chunks/chunk_1/masks/player")
    
    if not chunk_0_dir.exists():
        print(f"❌ Chunk 0 directory not found: {chunk_0_dir}")
        return
    
    if not chunk_1_dir.exists():
        print(f"❌ Chunk 1 directory not found: {chunk_1_dir}")
        return
    
    # Get all object directories in chunk_0 and chunk_1
    chunk_0_objects = sorted([d for d in chunk_0_dir.iterdir() if d.is_dir() and d.name.startswith("object_")])
    chunk_1_objects = sorted([d for d in chunk_1_dir.iterdir() if d.is_dir() and d.name.startswith("object_")])
    
    print(f"\n{'='*80}")
    print(f"Testing IoU Matching: chunk_0/object_1 → chunk_1 (all objects)")
    print(f"{'='*80}\n")
    print(f"Chunk 0 objects: {len(chunk_0_objects)}")
    print(f"Chunk 1 objects: {len(chunk_1_objects)}")
    print(f"\nIoU Threshold in config: 0.9 (expecting matches > 0.9)")
    print(f"\n{'='*80}\n")
    
    # Focus specifically on object_1 from chunk_0
    object_1_dir = chunk_0_dir / "object_1"
    
    if not object_1_dir.exists():
        print(f"❌ object_1 not found in chunk_0")
        print(f"\nAvailable objects in chunk_0:")
        for obj_dir in chunk_0_objects:
            print(f"  - {obj_dir.name}")
        return
    
    # Process only object_1
    for chunk_0_obj_dir in [object_1_dir]:
        obj_0_id = chunk_0_obj_dir.name.split("_")[1]
        
        # Get last frame from chunk_0
        png_files_0 = sorted(chunk_0_obj_dir.glob("frame_*.png"))
        if not png_files_0:
            print(f"⚠️  No PNG files in {chunk_0_obj_dir.name}")
            continue
        
        last_frame_path = png_files_0[-1]
        last_frame_idx = last_frame_path.stem.split("_")[1]
        
        print(f"📍 chunk_0/{chunk_0_obj_dir.name} (last frame: frame_{last_frame_idx})")
        
        # Load last frame mask from chunk_0
        mask_0 = load_png_mask(last_frame_path)
        
        if mask_0 is None:
            print(f"  ❌ Failed to load mask from chunk_0")
            continue
        
        # Check if mask has any data
        nonzero_pixels = np.count_nonzero(mask_0)
        if nonzero_pixels == 0:
            print(f"  ⚠️  Mask is BLANK (all zeros) - cannot match!")
            print()
            continue
        
        print(f"  ✓ Loaded mask: shape={mask_0.shape}, non-zero pixels={nonzero_pixels}")
        
        # Compare with first frame of all objects in chunk_1
        print(f"\n  Comparing with chunk_1 objects:")
        print(f"  {'-'*60}")
        
        best_match = None
        best_iou = 0.0
        
        for chunk_1_obj_dir in chunk_1_objects:
            obj_1_id = chunk_1_obj_dir.name.split("_")[1]
            
            # Get first frame from chunk_1
            png_files_1 = sorted(chunk_1_obj_dir.glob("frame_*.png"))
            if not png_files_1:
                continue
            
            first_frame_path = png_files_1[0]
            first_frame_idx = first_frame_path.stem.split("_")[1]
            
            # Load first frame mask from chunk_1
            mask_1 = load_png_mask(first_frame_path)
            
            if mask_1 is None:
                continue
            
            nonzero_pixels_1 = np.count_nonzero(mask_1)
            
            # Compute IoU
            iou = compute_iou(mask_0, mask_1)
            
            status = ""
            if iou > 0.9:
                status = "✅ MATCH (IoU > 0.9)"
            elif iou > 0.5:
                status = "⚠️  CLOSE (0.5 < IoU < 0.9)"
            elif iou > 0:
                status = "❌ LOW"
            else:
                status = "❌ NO OVERLAP"
            
            if nonzero_pixels_1 == 0:
                status += " [BLANK MASK]"
            
            print(f"    vs chunk_1/{chunk_1_obj_dir.name} (frame_{first_frame_idx}): IoU={iou:.4f} {status}")
            
            if iou > best_iou:
                best_iou = iou
                best_match = obj_1_id
        
        print(f"  {'-'*60}")
        if best_match is not None:
            if best_iou > 0.9:
                print(f"  🎯 BEST MATCH: object_{best_match} with IoU={best_iou:.4f} ✅")
            else:
                print(f"  🔍 BEST MATCH: object_{best_match} with IoU={best_iou:.4f} (below threshold!)")
        else:
            print(f"  ❌ NO MATCH FOUND")
        
        print()
        print()


if __name__ == "__main__":
    test_chunk_matching()
