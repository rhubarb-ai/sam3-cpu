#!/usr/bin/env python3
"""
Test Cross-Chunk Mask Injection on Pre-Split Video Chunks

Processes pre-existing video chunk files sequentially, injecting tracked masks
from chunk N into chunk N+1 for seamless object continuity.

This bypasses VideoProcessor/ChunkProcessor and drives the Sam3VideoDriver
directly, so you can see exactly what happens at each step.

Usage:
  # Default: processes assets/videos/private/feb4_camera10_chunks/
  python3 test_chunks_injection.py

  # Custom chunks folder and prompt:
  python3 test_chunks_injection.py \
      --chunks-dir assets/videos/private/feb4_camera10_chunks \
      --prompts person \
      --output results/feb4_camera10_injection_test

  # Multiple prompts:
  python3 test_chunks_injection.py --prompts person ball --keep-temp
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from sam3.drivers import Sam3VideoDriver
from sam3.logger import get_logger
from sam3.__globals import DEFAULT_PROPAGATION_DIRECTION

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_chunks(chunks_dir: Path) -> List[Path]:
    """Find and sort chunk video files in a directory."""
    patterns = ["chunk_*.mp4", "chunk_*.avi", "chunk_*.mov", "chunk_*.mkv"]
    chunks = []
    for pat in patterns:
        chunks.extend(chunks_dir.glob(pat))
    chunks = sorted(chunks)
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {chunks_dir}")
    return chunks


def get_video_info(video_path: Path) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "path": str(video_path),
        "name": video_path.name,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "nb_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def save_mask_png(mask: np.ndarray, path: Path):
    """Save a uint8 mask as grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path, format="PNG", compress_level=1)


def extract_frame_masks(
    result_prompt: dict,
    object_ids: set,
    frame_idx: int = None,
) -> Dict[int, np.ndarray]:
    """Extract masks from a specific frame of propagation results.
    
    Args:
        result_prompt: Dict[frame_idx -> output] from propagate_in_video.
        object_ids: Set of all detected object IDs.
        frame_idx: Frame to extract from. If None, uses the last frame.
    
    Returns:
        Dict mapping obj_id -> 2D uint8 mask (H x W, 0 or 255).
    """
    if not result_prompt:
        return {}

    if frame_idx is None:
        frame_idx = max(result_prompt.keys())

    output = result_prompt.get(frame_idx)
    if output is None:
        return {}

    out_obj_ids = output.get("out_obj_ids", [])
    if isinstance(out_obj_ids, np.ndarray):
        out_obj_ids = out_obj_ids.tolist()

    masks = {}
    for obj_id in object_ids:
        if obj_id in out_obj_ids:
            try:
                idx = out_obj_ids.index(obj_id)
                mask_bool = output["out_binary_masks"][idx]
                masks[obj_id] = (mask_bool.astype(np.uint8) * 255)
            except (IndexError, ValueError):
                pass
    return masks


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary/uint8 masks."""
    a = mask_a > 127 if mask_a.dtype == np.uint8 else mask_a.astype(bool)
    b = mask_b > 127 if mask_b.dtype == np.uint8 else mask_b.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union > 0 else 0.0


def match_and_remap_ids(
    result_prompt: dict,
    object_ids: set,
    prev_last_masks: Dict[int, np.ndarray],
    global_next_id: int,
    iou_threshold: float = 0.25,
):
    """
    Match this chunk's objects to previous chunk objects via IoU on the first frame,
    then remap all frame outputs so the same physical object keeps the same ID.
    
    Args:
        result_prompt: Dict[frame_idx -> output] from propagate_in_video.
        object_ids: Set of all detected object IDs in this chunk.
        prev_last_masks: Dict[obj_id -> uint8 mask] from previous chunk's last frame.
        global_next_id: Next available global ID for truly new objects.
        iou_threshold: Minimum IoU to consider a match.
    
    Returns:
        Tuple of:
        - remapped_result: result_prompt with IDs replaced
        - remapped_object_ids: set of remapped IDs
        - id_mapping: dict {old_chunk_id -> new_global_id}
        - updated global_next_id
    """
    if not result_prompt or not prev_last_masks:
        # No previous data — assign global IDs to all objects
        id_mapping = {}
        for oid in sorted(object_ids):
            id_mapping[oid] = global_next_id
            global_next_id += 1
        remapped_result = _apply_id_mapping(result_prompt, id_mapping)
        remapped_ids = set(id_mapping.values())
        return remapped_result, remapped_ids, id_mapping, global_next_id

    # Get first frame of this chunk
    first_frame_idx = min(result_prompt.keys())
    first_frame_masks = extract_frame_masks(result_prompt, object_ids, first_frame_idx)

    # Greedy IoU matching: for each new object, find best-matching previous object
    id_mapping = {}  # chunk_local_id -> global_id
    used_prev_ids = set()

    # Score all pairs
    pairs = []
    for new_id, new_mask in first_frame_masks.items():
        for prev_id, prev_mask in prev_last_masks.items():
            iou = compute_iou(new_mask, prev_mask)
            if iou >= iou_threshold:
                pairs.append((iou, new_id, prev_id))

    # Greedy match: best IoU first
    pairs.sort(reverse=True)
    for iou, new_id, prev_id in pairs:
        if new_id in id_mapping or prev_id in used_prev_ids:
            continue
        id_mapping[new_id] = prev_id
        used_prev_ids.add(prev_id)
        print(f"    Matched: chunk_obj_{new_id} -> global_obj_{prev_id} (IoU={iou:.3f})")

    # Assign fresh global IDs to unmatched objects
    for oid in sorted(object_ids):
        if oid not in id_mapping:
            id_mapping[oid] = global_next_id
            print(f"    New object: chunk_obj_{oid} -> global_obj_{global_next_id}")
            global_next_id += 1

    remapped_result = _apply_id_mapping(result_prompt, id_mapping)
    remapped_ids = set(id_mapping.values())
    return remapped_result, remapped_ids, id_mapping, global_next_id


def _apply_id_mapping(
    result_prompt: dict,
    id_mapping: Dict[int, int],
) -> dict:
    """Apply an ID mapping to all frames in result_prompt."""
    remapped = {}
    for frame_idx, output in result_prompt.items():
        out_obj_ids = output.get("out_obj_ids", [])
        if isinstance(out_obj_ids, np.ndarray):
            out_obj_ids_list = out_obj_ids.tolist()
        else:
            out_obj_ids_list = list(out_obj_ids)

        mapped_ids = [id_mapping.get(oid, oid) for oid in out_obj_ids_list]

        remapped_output = dict(output)
        remapped_output["out_obj_ids"] = np.array(mapped_ids, dtype=np.int64)
        remapped[frame_idx] = remapped_output
    return remapped


def save_chunk_masks(
    result_prompt: dict,
    object_ids: set,
    output_dir: Path,
    video_info: dict,
):
    """Save all masks for a chunk as PNGs."""
    h, w = video_info["height"], video_info["width"]
    total_frames = video_info["nb_frames"]

    for obj_id in object_ids:
        obj_dir = output_dir / f"object_{obj_id}"
        obj_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(total_frames):
        frame_output = result_prompt.get(frame_idx)

        for obj_id in object_ids:
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)

            if frame_output is not None:
                out_obj_ids = frame_output.get("out_obj_ids", [])
                if isinstance(out_obj_ids, np.ndarray):
                    out_obj_ids = out_obj_ids.tolist()
                if obj_id in out_obj_ids:
                    try:
                        idx = out_obj_ids.index(obj_id)
                        mask_bool = frame_output["out_binary_masks"][idx]
                        if mask_bool.any():
                            mask_uint8 = mask_bool.astype(np.uint8) * 255
                    except (IndexError, ValueError):
                        pass

            png_path = output_dir / f"object_{obj_id}" / f"frame_{frame_idx:06d}.png"
            save_mask_png(mask_uint8, png_path)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_chunks_with_injection(
    chunk_paths: List[Path],
    prompts: List[str],
    output_dir: Path,
    propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION,
) -> Dict[str, Any]:
    """
    Process pre-split video chunks sequentially with cross-chunk ID consistency.
    
    For each prompt:
      Chunk 0: detect + propagate, assign initial global IDs
      Chunk 1+: detect + propagate, then IoU-match first-frame masks against
                previous chunk's last-frame masks and remap IDs for consistency.
    
    Args:
        chunk_paths: Sorted list of chunk video paths.
        prompts: Text prompts for segmentation.
        output_dir: Where to save masks and metadata.
        propagation_direction: "forward", "backward", or "both".
    
    Returns:
        Summary results dictionary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CROSS-CHUNK MASK INJECTION PIPELINE")
    print("=" * 70)
    print(f"Chunks: {len(chunk_paths)}")
    print(f"Prompts: {prompts}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Get info for all chunks
    chunk_infos = []
    for cp in chunk_paths:
        info = get_video_info(cp)
        chunk_infos.append(info)
        print(f"  {info['name']}: {info['width']}x{info['height']}, {info['nb_frames']} frames, {info['fps']:.1f} fps")

    # Initialize driver (model loads once)
    print("\nLoading SAM3 model...")
    t0 = time.time()
    driver = Sam3VideoDriver()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    all_results = []

    try:
        for prompt in prompts:
            print(f"\n{'=' * 70}")
            print(f"PROMPT: '{prompt}'")
            print(f"{'=' * 70}")

            # Carry-forward masks between chunks for this prompt
            prev_last_masks: Optional[Dict[int, np.ndarray]] = None
            global_next_id = 0  # Global ID counter across chunks
            prompt_results = []

            for chunk_idx, (chunk_path, chunk_info) in enumerate(zip(chunk_paths, chunk_infos)):
                print(f"\n--- Chunk {chunk_idx}/{len(chunk_paths)-1}: {chunk_path.name} "
                      f"({chunk_info['nb_frames']} frames) ---")

                t1 = time.time()

                # Start session for this chunk
                session_id = driver.start_session(video_path=str(chunk_path))

                try:
                    # Step 1: Add text prompt (detects objects with chunk-local IDs)
                    print(f"  Adding prompt: '{prompt}'")
                    driver.add_prompt(session_id, prompt)

                    # Step 2: Propagate
                    print(f"  Propagating masks ({propagation_direction})...")
                    result_prompt, object_ids, frame_objects = driver.propagate_in_video(
                        session_id,
                        propagation_direction=propagation_direction,
                    )

                    elapsed = time.time() - t1
                    raw_ids = sorted(object_ids)
                    num_objects = len(object_ids)
                    print(f"  Detected {num_objects} object(s) (raw IDs: {raw_ids})")

                    # Step 3: IoU-based ID remapping for cross-chunk consistency
                    if chunk_idx == 0:
                        # First chunk: assign initial global IDs
                        id_mapping = {}
                        for oid in sorted(object_ids):
                            id_mapping[oid] = global_next_id
                            global_next_id += 1
                        result_prompt = _apply_id_mapping(result_prompt, id_mapping)
                        object_ids = set(id_mapping.values())
                        print(f"  Initial global IDs: {sorted(object_ids)}")
                    else:
                        # Match with previous chunk's objects via IoU
                        print(f"  Matching against previous chunk ({len(prev_last_masks)} objects)...")
                        result_prompt, object_ids, id_mapping, global_next_id = (
                            match_and_remap_ids(
                                result_prompt, object_ids,
                                prev_last_masks, global_next_id,
                            )
                        )
                        continued_ids = sorted(set(id_mapping.values()) & set(prev_last_masks.keys()))
                        new_ids = sorted(set(id_mapping.values()) - set(prev_last_masks.keys()))
                        print(f"  Continued from prev chunk: {continued_ids}")
                        if new_ids:
                            print(f"  Newly assigned IDs: {new_ids}")
                    
                    # Rebuild frame_objects with remapped IDs
                    frame_objects = {}
                    for fidx, output in result_prompt.items():
                        oids = output.get("out_obj_ids", [])
                        if isinstance(oids, np.ndarray):
                            oids = oids.tolist()
                        frame_objects[fidx] = oids

                    print(f"  Final object IDs: {sorted(object_ids)} ({elapsed:.1f}s)")

                    # Step 4: Save masks (with remapped IDs)
                    chunk_output_dir = output_dir / prompt / f"chunk_{chunk_idx:03d}"
                    save_chunk_masks(result_prompt, object_ids, chunk_output_dir, chunk_info)
                    print(f"  Saved masks to: {chunk_output_dir}")

                    # Step 5: Extract last-frame masks for carry-forward (now with global IDs)
                    prev_last_masks = extract_frame_masks(result_prompt, object_ids)
                    non_empty = sum(1 for m in prev_last_masks.values() if m.any())
                    print(f"  Carry-forward: {len(prev_last_masks)} masks ({non_empty} non-empty)")

                    prompt_results.append({
                        "chunk_idx": chunk_idx,
                        "chunk_name": chunk_path.name,
                        "num_objects": len(object_ids),
                        "object_ids": sorted(object_ids),
                        "id_mapping": {str(k): v for k, v in id_mapping.items()},
                        "num_frames": len(result_prompt),
                        "elapsed_sec": round(elapsed, 2),
                        "masks_dir": str(chunk_output_dir),
                    })

                finally:
                    driver.close_session(session_id)

            all_results.append({
                "prompt": prompt,
                "chunks": prompt_results,
            })

    finally:
        print("\nCleaning up driver...")
        driver.cleanup()

    # Save summary
    summary = {
        "num_chunks": len(chunk_paths),
        "prompts": prompts,
        "output_dir": str(output_dir),
        "results": all_results,
    }
    summary_path = output_dir / "injection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {summary_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for pr in all_results:
        print(f"\nPrompt: '{pr['prompt']}'")
        for cr in pr["chunks"]:
            mapping_str = ""
            if cr.get("id_mapping"):
                mapping_str = f", mapping={cr['id_mapping']}"
            print(f"  Chunk {cr['chunk_idx']} ({cr['chunk_name']}): "
                  f"{cr['num_objects']} objs in {cr['num_frames']} frames "
                  f"({cr['elapsed_sec']}s), IDs={cr['object_ids']}{mapping_str}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test cross-chunk mask injection on pre-split video chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default (feb4_camera10_chunks, prompt="player")
  python3 test_chunks_injection.py

  # Custom folder and prompts
  python3 test_chunks_injection.py \\
      --chunks-dir assets/videos/private/feb4_camera10_chunks \\
      --prompts person ball \\
      --output results/injection_test
        """,
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default="assets/videos/private/feb4_camera10_chunks",
        help="Directory containing chunk_*.mp4 files (default: assets/videos/private/feb4_camera10_chunks)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=str,
        default=["player"],
        help="Text prompts for segmentation (default: player)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/<folder_name>_injection)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["forward", "backward", "both"],
        default=DEFAULT_PROPAGATION_DIRECTION,
        help=f"Propagation direction (default: {DEFAULT_PROPAGATION_DIRECTION})",
    )

    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        print(f"ERROR: Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    chunk_paths = discover_chunks(chunks_dir)
    print(f"Found {len(chunk_paths)} chunk(s) in {chunks_dir}")

    output_dir = Path(args.output) if args.output else Path("results") / f"{chunks_dir.name}_injection"

    process_chunks_with_injection(
        chunk_paths=chunk_paths,
        prompts=args.prompts,
        output_dir=output_dir,
        propagation_direction=args.direction,
    )


if __name__ == "__main__":
    main()
