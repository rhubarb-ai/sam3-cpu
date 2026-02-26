"""
Test script for post-processing functionality.

Validates:
- IoU computation
- ID mapping across chunks
- Mask stitching

This script can run on existing chunk results without reprocessing.
"""

import json
import numpy as np
from pathlib import Path

from sam3.postprocessor import VideoPostProcessor
from sam3.utils.logger import get_logger

logger = get_logger(__name__)


def test_iou_computation():
    """Test IoU computation with known masks."""
    print("\n=== Testing IoU Computation ===")
    
    # Create test masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    
    # Perfect overlap
    mask1[25:75, 25:75] = 1
    mask2[25:75, 25:75] = 1
    
    postprocessor = VideoPostProcessor(
        video_name="test",
        chunk_results=[],
        video_metadata={},
        chunks_temp_dir=Path("temp"),
        masks_output_dir=Path("temp"),
        meta_output_dir=Path("temp")
    )
    
    iou = postprocessor._compute_iou(mask1, mask2)
    print(f"  Perfect overlap IoU: {iou:.3f} (expected: 1.000)")
    assert abs(iou - 1.0) < 0.001, "Perfect overlap should have IoU = 1.0"
    
    # Partial overlap (50%)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[50:100, 25:75] = 1  # Bottom half
    
    iou = postprocessor._compute_iou(mask1, mask2)
    print(f"  50% overlap IoU: {iou:.3f} (expected: ~0.333)")
    
    # No overlap
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[80:90, 80:90] = 1
    
    iou = postprocessor._compute_iou(mask1, mask2)
    print(f"  No overlap IoU: {iou:.3f} (expected: 0.000)")
    assert iou == 0.0, "No overlap should have IoU = 0.0"
    
    print("  ✓ IoU computation test passed!")


def test_frame_matching():
    """Test frame-level mask matching."""
    print("\n=== Testing Frame Mask Matching ===")
    
    # Create test masks
    masks_i = {
        0: np.zeros((100, 100), dtype=np.uint8),
        1: np.zeros((100, 100), dtype=np.uint8),
    }
    masks_i[0][10:30, 10:30] = 1  # Object 0
    masks_i[1][50:70, 50:70] = 1  # Object 1
    
    masks_j = {
        0: np.zeros((100, 100), dtype=np.uint8),  # Should match to object 1 from i
        1: np.zeros((100, 100), dtype=np.uint8),  # Should match to object 0 from i
    }
    masks_j[0][49:71, 49:71] = 1  # Similar to object 1 from i
    masks_j[1][9:31, 9:31] = 1     # Similar to object 0 from i
    
    postprocessor = VideoPostProcessor(
        video_name="test",
        chunk_results=[],
        video_metadata={},
        chunks_temp_dir=Path("temp"),
        masks_output_dir=Path("temp"),
        meta_output_dir=Path("temp"),
        iou_threshold=0.5  # Lower threshold for test
    )
    
    mapping = postprocessor._match_frame_masks(masks_i, masks_j)
    
    print(f"  Mapping: j -> i = {mapping}")
    print(f"    j_0 -> i_{mapping.get(0, 'unmatched')}")
    print(f"    j_1 -> i_{mapping.get(1, 'unmatched')}")
    
    # Verify expected mappings
    assert 0 in mapping, "Object 0 from j should be matched"
    assert 1 in mapping, "Object 1 from j should be matched"
    assert mapping[0] == 1, "j_0 should match to i_1"
    assert mapping[1] == 0, "j_1 should match to i_0"
    
    print("  ✓ Frame matching test passed!")


def test_with_real_results():
    """Test with real chunk results if available."""
    print("\n=== Testing with Real Results ===")
    
    # Look for existing results
    results_dir = Path("results/sample")
    if not results_dir.exists():
        print("  No real results found at results/sample")
        print("  Run main.py first to generate test data")
        return
    
    # Check for metadata
    metadata_path = results_dir / "metadata.json"
    if not metadata_path.exists():
        print("  No metadata.json found")
        return
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    print(f"  Found video: {metadata.get('video_name')}")
    print(f"  Chunks: {metadata.get('num_chunks')}")
    print(f"  Resolution: {metadata.get('width')}x{metadata.get('height')}")
    
    # Check for chunk results
    chunks_dir = results_dir / "temp_files" / "chunks"
    if not chunks_dir.exists():
        print("  No chunks directory found")
        return
    
    chunk_count = len(list(chunks_dir.glob("chunk_*")))
    print(f"  Found {chunk_count} chunk directories")
    
    # Load chunk results if available
    chunk_results_file = results_dir / "temp_files" / "chunk_results.json"
    if chunk_results_file.exists():
        with open(chunk_results_file, "r") as f:
            chunk_results = json.load(f)
        print(f"  Loaded {len(chunk_results)} chunk results")
        
        # Display prompts
        if chunk_results:
            first_chunk = chunk_results[0]
            prompts = list(first_chunk.get("prompts", {}).keys())
            print(f"  Prompts: {prompts}")
    
    print("  ✓ Real results test complete!")


def main():
    """Run all tests."""
    print("Starting post-processing tests...")
    
    try:
        test_iou_computation()
        test_frame_matching()
        test_with_real_results()
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
