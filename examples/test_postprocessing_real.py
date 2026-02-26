"""
Test script for post-processing with existing results.

This script loads existing chunk results from results/sample and runs
the post-processing logic to generate stitched masks.
"""

import json
import sys
from pathlib import Path

from sam3.postprocessor import VideoPostProcessor
from sam3.utils.logger import get_logger

logger = get_logger(__name__)


def load_chunk_results(results_dir: Path):
    """
    Load chunk results from the results directory.
    
    Args:
        results_dir: Directory containing processed results.
    
    Returns:
        List of chunk result dictionaries.
    """
    chunks_dir = results_dir / "temp_files" / "chunks"
    
    if not chunks_dir.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return None
    
    # Find all chunk directories
    chunk_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])
    
    if not chunk_dirs:
        logger.error("No chunk directories found")
        return None
    
    logger.info(f"Found {len(chunk_dirs)} chunk directories")
    
    # Load results for each chunk
    chunk_results = []
    
    for chunk_dir in chunk_dirs:
        chunk_id = int(chunk_dir.name.split("_")[1])
        chunk_video_path = chunk_dir / f"chunk_{chunk_id}.mp4"
        
        # Load metadata
        metadata_dir = chunk_dir / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"  Chunk {chunk_id}: no metadata directory")
            continue
        
        # Get prompts from masks directory
        masks_dir = chunk_dir / "masks"
        if not masks_dir.exists():
            logger.warning(f"  Chunk {chunk_id}: no masks directory")
            continue
        
        prompt_dirs = [d for d in masks_dir.iterdir() if d.is_dir()]
        
        # Build prompt results
        prompt_results = {}
        
        for prompt_dir in prompt_dirs:
            prompt = prompt_dir.name
            
            # Load prompt metadata if available
            prompt_metadata_path = metadata_dir / f"{prompt}.json"
            if prompt_metadata_path.exists():
                with open(prompt_metadata_path, "r") as f:
                    prompt_metadata = json.load(f)
                
                prompt_results[prompt] = {
                    "prompt": prompt,
                    "num_objects": prompt_metadata.get("num_objects", 0),
                    "object_ids": prompt_metadata.get("object_ids", []),
                    "frame_objects": prompt_metadata.get("frame_objects", {}),
                    "masks_dir": str(prompt_dir),
                    "metadata_path": str(prompt_metadata_path)
                }
            else:
                # Infer from mask files
                mask_files = list(prompt_dir.glob("object_*.mp4"))
                object_ids = [int(f.stem.split("_")[1]) for f in mask_files]
                
                prompt_results[prompt] = {
                    "prompt": prompt,
                    "num_objects": len(object_ids),
                    "object_ids": object_ids,
                    "frame_objects": {},
                    "masks_dir": str(prompt_dir),
                    "metadata_path": None
                }
        
        chunk_result = {
            "chunk_id": chunk_id,
            "chunk_video_path": str(chunk_video_path),
            "prompts": prompt_results,
            "num_prompts": len(prompt_results)
        }
        
        chunk_results.append(chunk_result)
        
        logger.info(f"  Chunk {chunk_id}: {len(prompt_results)} prompts, "
                   f"{sum(p['num_objects'] for p in prompt_results.values())} total objects")
    
    return chunk_results


def main():
    """Run post-processing on existing results."""
    results_dir = Path("results/sample")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Run main.py first to generate test data")
        return 1
    
    print(f"\n=== Post-Processing Test on {results_dir} ===\n")
    
    # Load video metadata
    metadata_path = results_dir / "metadata" / "video_metadata.json"
    if not metadata_path.exists():
        print(f"❌ Video metadata not found: {metadata_path}")
        return 1
    
    with open(metadata_path, "r") as f:
        video_metadata = json.load(f)
    
    print(f"Video: {video_metadata.get('video')}")
    print(f"Resolution: {video_metadata.get('width')}x{video_metadata.get('height')}")
    print(f"Duration: {video_metadata.get('duration')}s")
    print(f"Frames: {video_metadata.get('nb_frames')}")
    print(f"FPS: {video_metadata.get('fps')}")
    print(f"Chunks defined: {len(video_metadata.get('chunks', []))}")
    print()
    
    # Load chunk results
    print("Loading chunk results...")
    chunk_results = load_chunk_results(results_dir)
    
    if not chunk_results:
        print("❌ Failed to load chunk results")
        return 1
    
    print(f"✓ Loaded {len(chunk_results)} chunks")
    
    # Get prompts
    prompts = []
    if chunk_results:
        prompts = list(chunk_results[0]["prompts"].keys())
    
    print(f"✓ Prompts: {prompts}\n")
    
    # Create post-processor
    print("Initializing post-processor...")
    postprocessor = VideoPostProcessor(
        video_name="sample",
        chunk_results=chunk_results,
        video_metadata=video_metadata,
        chunks_temp_dir=results_dir / "temp_files" / "chunks",
        masks_output_dir=results_dir / "masks",
        meta_output_dir=results_dir / "metadata"
    )
    
    print()
    
    # Run post-processing
    print("Running post-processing...")
    try:
        postprocessor.process(prompts)
        print("\n✅ Post-processing complete!")
        
        # Check output
        output_dir = results_dir / "masks"
        if output_dir.exists():
            print(f"\n📁 Output directory: {output_dir}")
            for prompt in prompts:
                prompt_dir = output_dir / prompt
                if prompt_dir.exists():
                    mask_files = list(prompt_dir.glob("object_*.mp4"))
                    print(f"  {prompt}: {len(mask_files)} stitched objects")
        
        # Check metadata
        meta_dir = results_dir / "metadata"
        if meta_dir.exists():
            print(f"\n📁 Metadata directory: {meta_dir}")
            meta_file = meta_dir / "id_mapping.json"
            
            # Show mapping
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    mapping = json.load(f)
                print(f"\n  ID Mapping file: id_mapping.json")
                print(f"    Num chunks: {mapping.get('num_chunks')}")
                print(f"    Overlap frames: {mapping.get('overlap_frames')}")
                print(f"    IoU threshold: {mapping.get('iou_threshold')}")
                print(f"    Prompts: {mapping.get('prompts')}")
                
                # Show sample by_chunk mapping
                by_chunk = mapping.get('mappings', {}).get('by_chunk', {})
                if by_chunk:
                    print(f"\n  Sample chunk-based mappings:")
                    for chunk_pair, prompt_mappings in list(by_chunk.items())[:2]:
                        print(f"    {chunk_pair}:")
                        for prompt, obj_mappings in prompt_mappings.items():
                            if obj_mappings:  # Only show if there are mappings
                                print(f"      {prompt}: {obj_mappings}")
                
                # Show sample by_prompt mapping
                by_prompt = mapping.get('mappings', {}).get('by_prompt', {})
                if by_prompt:
                    print(f"\n  Sample prompt-based mappings:")
                    first_prompt = list(by_prompt.keys())[0] if by_prompt else None
                    if first_prompt:
                        print(f"    {first_prompt}:")
                        for chunk_pair, obj_mappings in list(by_prompt[first_prompt].items())[:2]:
                            if obj_mappings:  # Only show if there are mappings
                                print(f"      {chunk_pair}: {obj_mappings}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Post-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
