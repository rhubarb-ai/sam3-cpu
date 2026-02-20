#!/usr/bin/env python3
"""
Test object tracking chains to understand why only 1 object tracked across all chunks.
"""

import numpy as np
from pathlib import Path
from PIL import Image


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two binary masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def load_png_mask(png_path):
    """
    Load a PNG mask and convert to binary (0 or 1).
    """
    try:
        pil_image = Image.open(png_path)
        mask_array = np.array(pil_image)
        
        # Convert to binary
        binary_mask = (mask_array > 0).astype(np.uint8)
        return binary_mask
        
    except Exception as e:
        print(f"  Error loading {png_path}: {e}")
        return None


def test_object_pair(chunk_i, obj_i, chunk_j, base_dir, prompt="player"):
    """Test IoU between last frame of chunk_i/obj_i and first frames of all chunk_j objects."""
    
    chunk_i_dir = base_dir / f"chunk_{chunk_i}"
    chunk_j_dir = base_dir / f"chunk_{chunk_j}"
    
    if not chunk_i_dir.exists() or not chunk_j_dir.exists():
        return None
    
    # Load last frame from chunk_i/obj_i
    obj_i_dir = chunk_i_dir / "masks" / prompt / f"object_{obj_i}"
    if not obj_i_dir.exists():
        return None
    
    png_files_i = sorted(obj_i_dir.glob("frame_*.png"))
    if not png_files_i:
        return None
    
    last_png = png_files_i[-1]
    mask_i_last = load_png_mask(last_png)
    
    if mask_i_last is None:
        return None
    
    non_zero_i = np.count_nonzero(mask_i_last)
    if non_zero_i == 0:
        return None  # Blank mask
    
    # Test against all objects in chunk_j
    masks_dir_j = chunk_j_dir / "masks" / prompt
    object_dirs_j = sorted([d for d in masks_dir_j.iterdir() if d.is_dir()])
    
    results = []
    for obj_dir_j in object_dirs_j:
        obj_j = int(obj_dir_j.name.replace("object_", ""))
        
        png_files_j = sorted(obj_dir_j.glob("frame_*.png"))
        if not png_files_j:
            continue
        
        first_png = png_files_j[0]
        mask_j_first = load_png_mask(first_png)
        
        if mask_j_first is None:
            continue
        
        iou = compute_iou(mask_i_last, mask_j_first)
        non_zero_j = np.count_nonzero(mask_j_first)
        
        results.append({
            "obj_j": obj_j,
            "iou": iou,
            "pixels_j": non_zero_j
        })
    
    return {
        "chunk_i": chunk_i,
        "obj_i": obj_i,
        "chunk_j": chunk_j,
        "pixels_i": non_zero_i,
        "results": results
    }


def main():
    """Test the chain that should have been tracked."""
    
    base_dir = Path("results/sample/temp_files/chunks")
    
    print(f"\n{'='*90}")
    print(f"Testing Object Tracking Chain (IoU threshold = 0.9)")
    print(f"{'='*90}\n")
    
    # From id_mapping.json, we know:
    # chunk_0/object_0 → chunk_1/object_1 (recorded)
    # chunk_1/object_1 → ??? (NOT recorded - chain stopped!)
    
    print("🔍 TRACING THE RECORDED CHAIN:")
    print("-" * 90)
    
    # Test 1: chunk_0/object_0 → chunk_1 (recorded as 0→1)
    print("\n1️⃣  chunk_0/object_0 → chunk_1 (id_mapping shows: 0→1)")
    result = test_object_pair(0, 0, 1, base_dir)
    if result:
        print(f"   Source: chunk_{result['chunk_i']}/object_{result['obj_i']} ({result['pixels_i']} pixels)")
        best = max(result['results'], key=lambda x: x['iou'])
        print(f"   Best match: chunk_{result['chunk_j']}/object_{best['obj_j']} with IoU={best['iou']:.4f}")
        if best['iou'] > 0.9:
            print(f"   ✅ MATCH CONFIRMED (IoU > 0.9)")
        
        # Show top 3 matches
        sorted_results = sorted(result['results'], key=lambda x: x['iou'], reverse=True)[:3]
        print(f"\n   Top 3 candidates:")
        for r in sorted_results:
            status = "✅" if r['iou'] > 0.9 else "❌"
            print(f"     {status} object_{r['obj_j']}: IoU={r['iou']:.4f} ({r['pixels_j']} pixels)")
    
    # Test 2: chunk_1/object_1 → chunk_2 (where chain stops)
    print("\n2️⃣  chunk_1/object_1 → chunk_2 (chain stopped - no mapping recorded)")
    result = test_object_pair(1, 1, 2, base_dir)
    if result:
        print(f"   Source: chunk_{result['chunk_i']}/object_{result['obj_i']} ({result['pixels_i']} pixels)")
        best = max(result['results'], key=lambda x: x['iou'])
        print(f"   Best match: chunk_{result['chunk_j']}/object_{best['obj_j']} with IoU={best['iou']:.4f}")
        if best['iou'] > 0.9:
            print(f"   ⚠️  MATCH FOUND but NOT recorded!")
        else:
            print(f"   ❌ No good match (IoU < 0.9) - explains why chain stopped")
        
        # Show top 3 matches
        sorted_results = sorted(result['results'], key=lambda x: x['iou'], reverse=True)[:3]
        print(f"\n   Top 3 candidates:")
        for r in sorted_results:
            status = "✅" if r['iou'] > 0.9 else "❌"
            print(f"     {status} object_{r['obj_j']}: IoU={r['iou']:.4f} ({r['pixels_j']} pixels)")
    
    print(f"\n{'='*90}")
    print("🔍 TESTING UNRECORDED MAPPINGS:")
    print("-" * 90)
    
    # Test 3: chunk_0/object_1 → chunk_1 (NOT in id_mapping)
    print("\n3️⃣  chunk_0/object_1 → chunk_1 (NOT in id_mapping)")
    result = test_object_pair(0, 1, 1, base_dir)
    if result:
        print(f"   Source: chunk_{result['chunk_i']}/object_{result['obj_i']} ({result['pixels_i']} pixels)")
        best = max(result['results'], key=lambda x: x['iou'])
        print(f"   Best match: chunk_{result['chunk_j']}/object_{best['obj_j']} with IoU={best['iou']:.4f}")
        if best['iou'] > 0.9:
            print(f"   ⚠️  MATCH FOUND (IoU > 0.9) but NOT recorded in id_mapping!")
            print(f"   This suggests PostProcessor couldn't load this object's masks")
        else:
            print(f"   ✅ No match - correctly not recorded")
        
        # Show all matches
        sorted_results = sorted(result['results'], key=lambda x: x['iou'], reverse=True)[:5]
        print(f"\n   Top candidates:")
        for r in sorted_results:
            status = "✅" if r['iou'] > 0.9 else "❌"
            print(f"     {status} object_{r['obj_j']}: IoU={r['iou']:.4f} ({r['pixels_j']} pixels)")
    
    # Test 4: chunk_0/object_2 → chunk_1 (also NOT in id_mapping)
    print("\n4️⃣  chunk_0/object_2 → chunk_1 (also NOT in id_mapping)")
    result = test_object_pair(0, 2, 1, base_dir)
    if result:
        print(f"   Source: chunk_{result['chunk_i']}/object_{result['obj_i']} ({result['pixels_i']} pixels)")
        best = max(result['results'], key=lambda x: x['iou'])
        print(f"   Best match: chunk_{result['chunk_j']}/object_{best['obj_j']} with IoU={best['iou']:.4f}")
        if best['iou'] > 0.9:
            print(f"   ⚠️  MATCH FOUND (IoU > 0.9) but NOT recorded in id_mapping!")
        else:
            print(f"   ✅ No match - correctly not recorded")
        
        # Show top candidates
        sorted_results = sorted(result['results'], key=lambda x: x['iou'], reverse=True)[:5]
        print(f"\n   Top candidates:")
        for r in sorted_results:
            status = "✅" if r['iou'] > 0.9 else "❌"
            print(f"     {status} object_{r['obj_j']}: IoU={r['iou']:.4f} ({r['pixels_j']} pixels)")
    
    print(f"\n{'='*90}")
    print("📊 SUMMARY:")
    print("-" * 90)
    print("If unrecorded mappings show IoU > 0.9, it means:")
    print("  • Masks exist on disk")
    print("  • IoU computation would succeed") 
    print("  • But PostProcessor didn't check them")
    print("  • Likely cause: PostProcessor using old /tmp/ paths from metadata")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
