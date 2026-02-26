#!/usr/bin/env python3
"""
Isolated PostProcessor test to debug object tracking.
Focuses on tracking object_1 from chunk_0 through all chunks.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent))

from sam3.postprocessor import VideoPostProcessor
from sam3.utils.logger import get_logger

logger = get_logger(__name__)


def load_chunk_results_from_metadata(chunks_dir: Path, num_chunks: int = 9) -> List[Dict[str, Any]]:
    """
    Load chunk results from metadata files.
    
    Returns:
        List of chunk result dictionaries (same format as ChunkProcessor returns).
    """
    chunk_results = []
    
    for i in range(num_chunks):
        chunk_dir = chunks_dir / f"chunk_{i}"
        if not chunk_dir.exists():
            logger.warning(f"Chunk {i} directory not found: {chunk_dir}")
            continue
        
        # Load player metadata
        player_metadata_path = chunk_dir / "metadata" / "player_metadata.json"
        
        if not player_metadata_path.exists():
            logger.warning(f"Player metadata not found: {player_metadata_path}")
            continue
        
        with open(player_metadata_path, "r") as f:
            player_metadata = json.load(f)
        
        # Build chunk result structure
        chunk_result = {
            "chunk_id": i,
            "chunk_index": i,
            "start_frame": i * 24,  # Approximate
            "end_frame": (i + 1) * 24,
            "total_frames": 25,
            "prompts": {
                "player": {
                    "object_ids": player_metadata["object_ids"],
                    "num_objects": player_metadata["num_objects"],
                    "masks_dir": player_metadata["masks_dir"]
                }
            }
        }
        
        chunk_results.append(chunk_result)
    
    logger.info(f"Loaded {len(chunk_results)} chunk results from metadata")
    return chunk_results


def trace_object_1_mappings(postprocessor: VideoPostProcessor):
    """
    Manually trace what should happen to object_1 from chunk_0.
    """
    print(f"\n{'='*90}")
    print("MANUAL TRACE: What SHOULD happen to chunk_0/object_1")
    print(f"{'='*90}\n")
    
    from PIL import Image
    import numpy as np
    
    def compute_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0
    
    def load_mask(png_path):
        try:
            img = np.array(Image.open(png_path))
            return (img > 0).astype(np.uint8)
        except:
            return None
    
    # Start with chunk_0/object_1
    current_chunk = 0
    current_obj = 1
    
    chunks_dir = Path("/tmp/sam3-cpu/sample/chunks")
    threshold = 0.9
    
    trace_chain = [(0, 1)]
    
    for next_chunk in range(1, 9):
        # Load last frame of current object
        current_dir = chunks_dir / f"chunk_{current_chunk}" / "masks" / "player" / f"object_{current_obj}"
        if not current_dir.exists():
            print(f"  ❌ chunk_{current_chunk}/object_{current_obj} directory not found")
            break
        
        last_png = sorted(current_dir.glob("frame_*.png"))[-1]
        mask_current = load_mask(last_png)
        
        if mask_current is None or np.count_nonzero(mask_current) == 0:
            print(f"  ❌ chunk_{current_chunk}/object_{current_obj} last frame is blank")
            break
        
        # Check all objects in next chunk
        next_masks_dir = chunks_dir / f"chunk_{next_chunk}" / "masks" / "player"
        if not next_masks_dir.exists():
            print(f"  ❌ chunk_{next_chunk} masks directory not found")
            break
        
        best_iou = 0.0
        best_obj = None
        
        for obj_dir in sorted(next_masks_dir.iterdir()):
            if not obj_dir.is_dir():
                continue
            
            obj_id = int(obj_dir.name.replace("object_", ""))
            first_png = sorted(obj_dir.glob("frame_*.png"))[0]
            mask_next = load_mask(first_png)
            
            if mask_next is None:
                continue
            
            iou = compute_iou(mask_current, mask_next)
            
            if iou > best_iou:
                best_iou = iou
                best_obj = obj_id
        
        if best_iou > threshold:
            trace_chain.append((next_chunk, best_obj))
            print(f"  ✅ chunk_{current_chunk}/object_{current_obj} → chunk_{next_chunk}/object_{best_obj} (IoU={best_iou:.4f})")
            current_chunk = next_chunk
            current_obj = best_obj
        else:
            print(f"  ❌ chain stopped at chunk_{current_chunk}/object_{current_obj} (best IoU={best_iou:.4f} < {threshold})")
            break
    
    print(f"\n  📊 EXPECTED CHAIN: {trace_chain}")
    print(f"  📊 CHAIN LENGTH: {len(trace_chain)} chunks\n")
    
    return trace_chain


def main():
    """
    Run PostProcessor in isolation and debug object_1 tracking.
    """
    print(f"\n{'='*90}")
    print("ISOLATED POSTPROCESSOR TEST - Debugging object_1 tracking")
    print(f"{'='*90}\n")
    
    # Paths
    chunks_dir = Path("/tmp/sam3-cpu/sample/chunks")
    output_dir = Path("results/sample")
    masks_output_dir = output_dir / "masks"
    meta_output_dir = output_dir / "metadata"
    
    # Load chunk results
    print("Step 1: Loading chunk results from metadata...")
    chunk_results = load_chunk_results_from_metadata(chunks_dir)
    
    if not chunk_results:
        print("❌ Failed to load chunk results")
        return
    
    print(f"  ✓ Loaded {len(chunk_results)} chunks")
    
    # Show what objects exist in each chunk
    print("\nStep 2: Object inventory per chunk:")
    for cr in chunk_results:
        obj_ids = cr["prompts"]["player"]["object_ids"]
        print(f"  chunk_{cr['chunk_id']}: {len(obj_ids)} objects - {obj_ids}")
    
    # Manually trace expected chain for object_1
    print("\nStep 3: Manual trace of object_1...")
    expected_chain = trace_object_1_mappings(None)
    
    # Load video metadata
    video_metadata_path = output_dir / "metadata" / "video_metadata.json"
    if video_metadata_path.exists():
        with open(video_metadata_path, "r") as f:
            video_metadata = json.load(f)
        print(f"\nStep 4: Loaded video metadata")
    else:
        video_metadata = {
            "video_name": "sample",
            "nb_frames": 200,  # Correct key name expected by PostProcessor
            "fps": 25.0,
            "width": 854,
            "height": 480
        }
        print(f"\nStep 4: Using default video metadata (200 frames)")
    
    # Initialize PostProcessor
    print(f"\nStep 5: Initializing PostProcessor...")
    print(f"  chunks_temp_dir: {chunks_dir.parent}")
    print(f"  masks_output_dir: {masks_output_dir}")
    print(f"  meta_output_dir: {meta_output_dir}")
    
    postprocessor = VideoPostProcessor(
        video_name="sample",
        chunk_results=chunk_results,
        video_metadata=video_metadata,
        chunks_temp_dir=chunks_dir.parent,
        masks_output_dir=masks_output_dir,
        meta_output_dir=meta_output_dir,
        iou_threshold=0.9
    )
    
    print(f"  ✓ PostProcessor initialized")
    print(f"  IoU threshold: {postprocessor.iou_threshold}")
    print(f"  Overlap frames: {postprocessor.overlap_frames}")
    
    # Run post-processing
    print(f"\n{'='*90}")
    print("Step 6: Running PostProcessor.process()...")
    print(f"{'='*90}\n")
    
    postprocessor.process(prompts=["player"])
    
    # Load and analyze results
    print(f"\n{'='*90}")
    print("Step 7: Analyzing results...")
    print(f"{'='*90}\n")
    
    id_mapping_path = meta_output_dir / "id_mapping.json"
    if id_mapping_path.exists():
        with open(id_mapping_path, "r") as f:
            id_mapping = json.load(f)
        
        print("Recorded mappings:")
        player_mappings = id_mapping["mappings"]["by_prompt"]["player"]
        for chunk_pair, mapping in player_mappings.items():
            print(f"  {chunk_pair}: {mapping}")
        
        # Check if object_1 mappings match expected
        print(f"\n{'='*90}")
        print("COMPARISON: Expected vs Actual")
        print(f"{'='*90}\n")
        
        print(f"Expected chain for object_1: {expected_chain}")
        
        # Try to follow object_1 in actual mappings
        actual_chain = [(0, 1)]
        current_chunk = 0
        current_obj = 1
        
        for next_chunk in range(1, 9):
            mapping_key = f"chunk_{current_chunk:03d}->chunk_{next_chunk:03d}"
            mapping = player_mappings.get(mapping_key, {})
            
            # Check if current_obj maps to something
            next_obj = mapping.get(str(current_obj))
            
            if next_obj is not None:
                actual_chain.append((next_chunk, int(next_obj)))
                print(f"  ✅ chunk_{current_chunk}/object_{current_obj} → chunk_{next_chunk}/object_{next_obj}")
                current_chunk = next_chunk
                current_obj = int(next_obj)
            else:
                print(f"  ❌ chunk_{current_chunk}/object_{current_obj} → chunk_{next_chunk}/??? (no mapping)")
                break
        
        print(f"\nActual chain in id_mapping: {actual_chain}")
        print(f"Expected chain from IoU test: {expected_chain}")
        
        if actual_chain == expected_chain:
            print(f"\n✅ SUCCESS: Chains match!")
        else:
            print(f"\n❌ MISMATCH: Chains don't match")
            print(f"   Expected {len(expected_chain)} chunks, got {len(actual_chain)} chunks")
    else:
        print("❌ id_mapping.json not found")
    
    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
