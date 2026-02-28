"""
SAM3 Post-Processor Module

Handles post-processing of video chunks including:
- Matching objects across chunk boundaries using IoU
- Building ID mappings across all chunks
- Stitching masks together with continuous frame numbering
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from sam3.utils.logger import get_logger
from sam3.__globals import (
    CHUNK_MASK_MATCHING_IOU_THRESHOLD,
    DEFAULT_MIN_CHUNK_OVERLAP
)

logger = get_logger(__name__)


class VideoPostProcessor:
    """
    Handles post-processing of chunked video segmentation results.
    
    Main responsibilities:
    - Match objects across chunk boundaries using IoU
    - Build ID mappings showing object flow across chunks
    - Stitch masks together removing overlap frames
    - Generate consolidated metadata
    
    Args:
        video_name: Name of the video being processed.
        chunk_results: List of chunk processing results.
        video_metadata: Video metadata (fps, width, height, etc.).
        chunks_temp_dir: Directory containing chunk outputs.
        masks_output_dir: Directory for final stitched masks.
        meta_output_dir: Directory for output metadata.
        iou_threshold: IoU threshold for matching masks (default from __globals).
    """
    
    def __init__(
        self,
        video_name: str,
        chunk_results: List[Dict[str, Any]],
        video_metadata: Dict[str, Any],
        chunks_temp_dir: Path,
        masks_output_dir: Path,
        meta_output_dir: Path,
        iou_threshold: float = CHUNK_MASK_MATCHING_IOU_THRESHOLD
    ):
        self.video_name = video_name
        self.chunk_results = chunk_results
        self.video_metadata = video_metadata
        self.chunks_temp_dir = Path(chunks_temp_dir)
        self.masks_output_dir = Path(masks_output_dir)
        self.meta_output_dir = Path(meta_output_dir)
        self.iou_threshold = iou_threshold
        
        # Ensure output directories exist
        self.masks_output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Infer overlap frames count from chunk boundaries
        self.overlap_frames = self._infer_overlap_from_metadata()
        if self.overlap_frames < 1:
            self.overlap_frames = DEFAULT_MIN_CHUNK_OVERLAP
        
        logger.info(f"PostProcessor initialized: {len(chunk_results)} chunks, "
                   f"{self.overlap_frames} overlap frames, IoU threshold={iou_threshold}")
    
    def _infer_overlap_from_metadata(self) -> int:
        """
        Infer overlap frames from chunk boundaries in metadata.
        
        If chunk_i ends at frame N and chunk_i+1 starts at frame M,
        then overlap = N - M + 1 (when N >= M).
        
        Returns:
            Number of overlap frames (at least 1).
        """
        chunks_info = self.video_metadata.get("chunks", [])
        
        if len(chunks_info) < 2:
            logger.debug("    Less than 2 chunks, using default overlap")
            return DEFAULT_MIN_CHUNK_OVERLAP
        
        # Check first two chunks
        chunk_0 = chunks_info[0]
        chunk_1 = chunks_info[1]
        
        end_0 = chunk_0.get("end")
        start_1 = chunk_1.get("start")
        
        if end_0 is None or start_1 is None:
            logger.debug("    Chunk boundaries not found, using default overlap")
            return DEFAULT_MIN_CHUNK_OVERLAP
        
        # Overlap is how many frames are shared
        # If end_0 = 24 and start_1 = 24, overlap = 1
        overlap = end_0 - start_1 + 1
        
        if overlap < 1:
            logger.debug(f"    Computed overlap={overlap}, using default")
            return DEFAULT_MIN_CHUNK_OVERLAP
        
        logger.debug(f"    Inferred overlap from chunk boundaries: {overlap} frames")
        return overlap
    
    def process(self, prompts: List[str]):
        """
        Main post-processing workflow.
        
        Args:
            prompts: List of prompts that were processed.
        """
        logger.info(f"Starting post-processing for {len(prompts)} prompt(s)")
        t_post_start = time.time()
        
        # Build mappings for all prompts
        all_mappings = {}
        all_iou_matrices = {}
        
        for prompt in prompts:
            logger.info(f"  Building mappings for prompt: '{prompt}'")
            prompt_mappings, prompt_iou = self._build_id_mappings(prompt)
            all_mappings[prompt] = prompt_mappings
            if prompt_iou:
                all_iou_matrices[prompt] = prompt_iou
        
        # Save combined mapping metadata
        logger.debug(f"    Saving combined mapping metadata...")
        self._save_combined_mapping_metadata(all_mappings, all_iou_matrices)
        
        # Stitch masks for each prompt
        for prompt in prompts:
            logger.info(f"  Stitching masks for prompt: '{prompt}'")
            self._stitch_masks_for_prompt(prompt, all_mappings[prompt])
        
        post_duration = round(time.time() - t_post_start, 3)
        logger.info(f"Post-processing complete in {post_duration:.1f}s")
    
    def _save_combined_mapping_metadata(self, all_mappings: Dict[str, Dict], all_iou_matrices: Dict[str, Dict] = None):
        """
        Save combined ID mappings for all prompts to a single file.
        
        Saves in dual structure:
        1. Chunk-based: "chunk_000->chunk_001": {"ball": {"0": "1"}, "player": {...}}
        2. Prompt-based: "ball": {"chunk_000->chunk_001": {"0": "1"}, ...}
        
        Args:
            all_mappings: Dictionary mapping prompts to their chunk mappings.
                         Format: {prompt: {chunk_pair: {j_id: i_id}}}
            all_iou_matrices: Optional IoU matrices from chunk matching.
        """
        metadata_path = self.meta_output_dir / "id_mapping.json"
        
        # Build chunk-based structure
        chunk_based = {}
        for prompt, prompt_mappings in all_mappings.items():
            for chunk_pair, mapping in prompt_mappings.items():
                if chunk_pair not in chunk_based:
                    chunk_based[chunk_pair] = {}
                # Convert int keys to string for JSON
                chunk_based[chunk_pair][prompt] = {str(k): str(v) for k, v in mapping.items()}
        
        # Build prompt-based structure (pivot)
        prompt_based = {}
        for prompt, prompt_mappings in all_mappings.items():
            prompt_based[prompt] = {}
            for chunk_pair, mapping in prompt_mappings.items():
                # Convert int keys to string for JSON
                prompt_based[prompt][chunk_pair] = {str(k): str(v) for k, v in mapping.items()}
        
        # Combined metadata
        metadata = {
            "video_name": self.video_name,
            "num_chunks": len(self.chunk_results),
            "overlap_frames": self.overlap_frames,
            "iou_threshold": self.iou_threshold,
            "prompts": list(all_mappings.keys()),
            "mappings": {
                "by_chunk": chunk_based,
                "by_prompt": prompt_based
            },
            "iou_matrices": all_iou_matrices if all_iou_matrices else None,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"      Saved combined mapping metadata to {metadata_path}")
    
    def _build_id_mappings(self, prompt: str) -> Tuple[Dict, Dict]:
        """
        Build ID mappings across all chunks for a prompt.
        
        Returns:
            Tuple of (prompt_mappings, iou_data).
            prompt_mappings: {chunk_pair: {j_id: i_id}} meaning chunk_j's object maps to chunk_i's object
            iou_data: {chunk_pair: iou_matrix_dict} captured from chunk processing
        """
        prompt_mappings = {}
        iou_data = {}
        
        # Process consecutive chunk pairs
        for i in range(len(self.chunk_results) - 1):
            chunk_i = self.chunk_results[i]
            chunk_j = self.chunk_results[i + 1]
            
            chunk_i_id = chunk_i["chunk_id"]
            chunk_j_id = chunk_j["chunk_id"]
            
            logger.debug(f"      Matching chunk {chunk_i_id} -> chunk {chunk_j_id}...")
            
            # Match objects between chunks
            mapping = self._match_chunks(chunk_i, chunk_j, prompt)
            
            logger.info(f"        Matched {len(mapping)} objects between chunks")
            
            # mapping is already in {i_id: j_id} format (chunk_i obj -> chunk_j obj)
            # No inversion needed - _match_chunks returns the correct format
            
            # Build mapping key
            mapping_key = f"chunk_{chunk_i_id:03d}->chunk_{chunk_j_id:03d}"
            prompt_mappings[mapping_key] = mapping

            # Capture IoU matrix from chunk_j if available
            prompt_result_j = chunk_j["prompts"].get(prompt, {})
            iou_mat = prompt_result_j.get("iou_matrix")
            if iou_mat:
                iou_data[mapping_key] = iou_mat
        
        return prompt_mappings, iou_data
    
    def _match_chunks(
        self,
        chunk_i: Dict,
        chunk_j: Dict,
        prompt: str
    ) -> Dict[int, int]:
        """
        Match objects between two consecutive chunks.
        
        First, deterministically matches objects that were injected from chunk_i into
        chunk_j (they share the same object IDs by construction). Then, for remaining
        unmatched objects, falls back to IoU-based matching on the overlap frames.
        
        Args:
            chunk_i: Previous chunk result.
            chunk_j: Current chunk result.
            prompt: Prompt being processed.
        
        Returns:
            Dictionary mapping chunk_i IDs to chunk_j IDs.
            Format: {i_id: j_id}
        """
        chunk_i_id = chunk_i["chunk_id"]
        chunk_j_id = chunk_j["chunk_id"]
        
        # Get object IDs for this prompt in both chunks
        prompt_result_i = chunk_i["prompts"].get(prompt, {})
        prompt_result_j = chunk_j["prompts"].get(prompt, {})
        
        object_ids_i = prompt_result_i.get("object_ids", [])
        object_ids_j = prompt_result_j.get("object_ids", [])
        
        if not object_ids_i or not object_ids_j:
            logger.info(f"          No objects to match (chunk_{chunk_i_id}: {len(object_ids_i)}, chunk_{chunk_j_id}: {len(object_ids_j)})")
            return {}
        
        logger.info(f"          Matching chunk_{chunk_i_id} ({len(object_ids_i)} objs) -> chunk_{chunk_j_id} ({len(object_ids_j)} objs) with {self.overlap_frames} overlap frame(s)")
        
        mappings = {}
        matched_j_ids = set()
        
        # Phase 1: Deterministic matching for injected objects
        # Objects injected from chunk_i into chunk_j retain the same IDs,
        # so any ID present in both chunks is a direct match.
        injected_ids_j = set(prompt_result_j.get("injected_object_ids", []))
        if injected_ids_j:
            set_i = set(object_ids_i)
            for obj_id in injected_ids_j:
                if obj_id in set_i and obj_id in object_ids_j:
                    mappings[obj_id] = obj_id  # Same ID in both chunks
                    matched_j_ids.add(obj_id)
                    logger.info(f"            ✅ Injected match: chunk_{chunk_i_id} obj_{obj_id} -> chunk_{chunk_j_id} obj_{obj_id} (deterministic)")
            
            if mappings:
                logger.info(f"          Phase 1: {len(mappings)} deterministic match(es) from injected objects")
        
        # Phase 2: IoU-based matching for remaining (newly detected) objects
        remaining_i = [id_i for id_i in object_ids_i if id_i not in mappings]
        remaining_j = [id_j for id_j in object_ids_j if id_j not in matched_j_ids]
        
        if remaining_i and remaining_j:
            logger.info(f"          Phase 2: IoU matching for {len(remaining_i)} remaining chunk_i objs vs {len(remaining_j)} chunk_j objs")
            
            # Load overlap frames from PNGs
            overlap_masks_i = self._load_overlap_frames_from_pngs(chunk_i, prompt, position='end')
            overlap_masks_j = self._load_overlap_frames_from_pngs(chunk_j, prompt, position='start')
            
            logger.info(f"          Loaded overlap masks: chunk_{chunk_i_id}={len(overlap_masks_i)} frames, chunk_{chunk_j_id}={len(overlap_masks_j)} frames")
            
            if overlap_masks_i and overlap_masks_j:
                num_overlap = min(len(overlap_masks_i), len(overlap_masks_j))
                
                for id_i in remaining_i:
                    best_iou = 0
                    best_id_j = None
                    
                    for id_j in remaining_j:
                        if id_j in matched_j_ids:
                            continue
                        
                        # Compute best (maximum) IoU across all overlap frames
                        ious = []
                        for frame_idx in range(num_overlap):
                            mask_i = overlap_masks_i[frame_idx].get(id_i)
                            mask_j = overlap_masks_j[frame_idx].get(id_j)
                            
                            if mask_i is not None and mask_j is not None:
                                iou = self._compute_iou(mask_i, mask_j)
                                ious.append(iou)
                        
                        best_iou_candidate = max(ious) if ious else 0.0
                        logger.debug(f"              IoU(obj_{id_i}, obj_{id_j}) = {best_iou_candidate:.4f} (max over {len(ious)} frames)")
                        
                        if best_iou_candidate > best_iou:
                            best_iou = best_iou_candidate
                            best_id_j = id_j
                    
                    if best_iou > self.iou_threshold:
                        mappings[id_i] = best_id_j
                        matched_j_ids.add(best_id_j)
                        logger.info(f"            ✅ IoU match: chunk_{chunk_i_id} obj_{id_i} -> chunk_{chunk_j_id} obj_{best_id_j} (IoU={best_iou:.3f})")
                    else:
                        logger.debug(f"            No match for obj_{id_i} (best IoU={best_iou:.4f} < threshold={self.iou_threshold})")
        
        return mappings
    
    def _load_overlap_frames_from_pngs(self, chunk: Dict, prompt: str, position: str = 'end') -> List[Dict[int, np.ndarray]]:
        """
        Load overlap frames masks from PNG images for a specific prompt.
        
        Args:
            chunk: Chunk result dictionary.
            prompt: Prompt name.
            position: 'end' to load last N frames, 'start' to load first N frames.
        
        Returns:
            List of dictionaries {object_id: mask_array} for each overlap frame.
        """
        chunk_id = chunk["chunk_id"]
        prompt_result = chunk["prompts"].get(prompt, {})
        object_ids = prompt_result.get("object_ids", [])
        masks_dir = prompt_result.get("masks_dir")
        
        if not masks_dir or not object_ids:
            return []
        
        masks_dir = Path(masks_dir)
        overlap_frames = []  # List of {obj_id: mask} dicts, one per frame
        
        logger.debug(f"          Loading {position} {self.overlap_frames} overlap frame(s) for chunk_{chunk_id}")
        
        for obj_id in object_ids:
            object_dir = masks_dir / f"object_{obj_id}"
            
            if not object_dir.exists():
                logger.warning(f"            Mask directory not found: {object_dir}")
                continue
            
            # Get all PNG files
            png_files = sorted(object_dir.glob("frame_*.png"))
            
            if not png_files:
                logger.warning(f"            No PNG files found in: {object_dir}")
                continue
            
            # Select overlap frames based on position
            if position == 'end':
                # Last N frames
                selected_pngs = png_files[-self.overlap_frames:]
            else:  # 'start'
                # First N frames
                selected_pngs = png_files[:self.overlap_frames]
            
            # Read selected frames
            for frame_idx, png_path in enumerate(selected_pngs):
                try:
                    # Use PIL for reading
                    pil_image = Image.open(png_path)
                    mask_image = np.array(pil_image)
                    if mask_image is not None and mask_image.size > 0:
                        mask_binary = (mask_image > 0).astype(np.uint8)
                        
                        # Ensure we have enough frame dicts in the list
                        while len(overlap_frames) <= frame_idx:
                            overlap_frames.append({})
                        
                        overlap_frames[frame_idx][obj_id] = mask_binary
                        
                        if frame_idx == 0:  # Log only first frame for brevity
                            nonzero = np.count_nonzero(mask_binary)
                            logger.debug(f"            ✓ chunk_{chunk_id}/object_{obj_id}: {len(selected_pngs)} frames, first has {nonzero} pixels")
                except Exception as e:
                    logger.warning(f"            Failed to read mask {png_path}: {e}")
        
        logger.info(f"          Loaded {len(overlap_frames)} overlap frame(s) from chunk_{chunk_id} ({len(object_ids)} objects)")
        return overlap_frames
    
    def _load_last_frame_masks(self, chunk: Dict, prompt: str) -> Dict[int, np.ndarray]:
        """
        Load last frame masks from PNG images for a specific prompt.
        Convenience wrapper for backward compatibility.
        
        Args:
            chunk: Chunk result dictionary.
            prompt: Prompt name.
        
        Returns:
            Dictionary {object_id: mask_array}.
        """
        overlap_frames = self._load_overlap_frames_from_pngs(chunk, prompt, position='end')
        # Return the last frame (most recent)
        return overlap_frames[-1] if overlap_frames else {}
    
    def _load_first_frame_masks(self, chunk: Dict, prompt: str) -> Dict[int, np.ndarray]:
        """
        Load first frame masks from PNG images for a specific prompt.
        Convenience wrapper for backward compatibility.
        
        Args:
            chunk: Chunk result dictionary.
            prompt: Prompt name.
        
        Returns:
            Dictionary {object_id: mask_array}.
        """
        overlap_frames = self._load_overlap_frames_from_pngs(chunk, prompt, position='start')
        # Return the first frame
        return overlap_frames[0] if overlap_frames else {}
    
    def _match_frame_masks(
        self,
        masks_i: Dict[int, np.ndarray],
        masks_j: Dict[int, np.ndarray]
    ) -> Dict[int, int]:
        """
        Match masks between two frames using IoU.
        
        Args:
            masks_i: Dictionary {object_id: mask} for frame i.
            masks_j: Dictionary {object_id: mask} for frame j.
        
        Returns:
            Dictionary mapping object IDs: {j_id: i_id}
        """
        mapping = {}
        matched_i_ids = set()
        
        # Compute IoU for all pairs
        iou_matrix = {}
        for j_id, mask_j in masks_j.items():
            for i_id, mask_i in masks_i.items():
                iou = self._compute_iou(mask_i, mask_j)
                if iou >= self.iou_threshold:
                    if j_id not in iou_matrix:
                        iou_matrix[j_id] = []
                    iou_matrix[j_id].append((iou, i_id))
        
        # Match greedily by highest IoU
        for j_id in sorted(iou_matrix.keys()):
            candidates = iou_matrix[j_id]
            if not candidates:
                continue
            
            # Sort by IoU descending
            candidates.sort(reverse=True)
            
            # Take highest IoU that hasn't been matched
            for iou, i_id in candidates:
                if i_id not in matched_i_ids:
                    mapping[j_id] = i_id
                    matched_i_ids.add(i_id)
                    break
        
        return mapping
    
    def _backtrack_matching(
        self,
        chunk_i_masks: List[Dict[int, np.ndarray]],
        chunk_j_masks: List[Dict[int, np.ndarray]],
        initial_mapping: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Backtrack to find matches for unmatched objects using earlier/later frames.
        
        Args:
            chunk_i_masks: List of mask dictionaries from chunk_i (end frames).
            chunk_j_masks: List of mask dictionaries from chunk_j (start frames).
            initial_mapping: Initial mapping from last frame of i to first frame of j.
        
        Returns:
            Enhanced mapping with backtracked matches.
        """
        mapping = initial_mapping.copy()
        matched_i_ids = set(mapping.values())
        
        # Find unmatched IDs in chunk_i last frame
        last_frame_i_ids = set(chunk_i_masks[-1].keys())
        unmatched_i_ids = last_frame_i_ids - matched_i_ids
        
        if not unmatched_i_ids:
            return mapping  # All matched
        
        logger.debug(f"          Backtracking for {len(unmatched_i_ids)} unmatched objects...")
        
        # TODO: Implement more robust feature-based matching in the future
        # For now, try earlier frames in chunk_i with later frames in chunk_j
        
        for i_id in unmatched_i_ids:
            best_match = None
            best_iou = self.iou_threshold
            
            # Try progressively earlier frames in chunk_i
            for frame_idx_i in range(len(chunk_i_masks) - 2, -1, -1):
                if i_id not in chunk_i_masks[frame_idx_i]:
                    continue
                
                mask_i = chunk_i_masks[frame_idx_i][i_id]
                
                # Try progressively later frames in chunk_j
                for frame_idx_j in range(1, len(chunk_j_masks)):
                    for j_id, mask_j in chunk_j_masks[frame_idx_j].items():
                        if j_id in mapping:
                            continue  # Already matched
                        
                        iou = self._compute_iou(mask_i, mask_j)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = j_id
                
                if best_match is not None:
                    break
            
            if best_match is not None:
                mapping[best_match] = i_id
                logger.debug(f"            Backtracked match: {i_id} -> {best_match} (IoU={best_iou:.3f})")
        
        return mapping
    
    def _compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IoU) between two binary masks.
        
        Args:
            mask1: Binary mask array.
            mask2: Binary mask array.
        
        Returns:
            IoU score in [0, 1].
        """
        # Ensure binary
        mask1 = mask1 > 0
        mask2 = mask2 > 0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def _load_chunk_overlap_masks(
        self,
        chunk: Dict,
        prompt: str,
        position: str = 'end'
    ) -> List[Dict[int, np.ndarray]]:
        """
        Load overlap frames masks from a chunk.
        
        Args:
            chunk: Chunk result dictionary.
            prompt: Prompt being processed.
            position: 'start' or 'end' - which overlap frames to load.
        
        Returns:
            List of dictionaries {object_id: mask} for each overlap frame.
        """
        chunk_id = chunk["chunk_id"]
        prompt_result = chunk["prompts"].get(prompt, {})
        object_ids = prompt_result.get("object_ids", [])
        
        logger.info(f"          Loading {position} overlap masks for chunk {chunk_id}, "
                   f"prompt '{prompt}', {len(object_ids)} objects")
        
        if not object_ids:
            logger.info(f"            No objects found for prompt '{prompt}' in chunk {chunk_id}")
            return []
        
        masks_dir = prompt_result.get("masks_dir")
        if not masks_dir:
            logger.info(f"            No masks_dir found for chunk {chunk_id}")
            return []
        
        masks_dir = Path(masks_dir)
        
        if not masks_dir.exists():
            logger.info(f"            Masks directory doesn't exist: {masks_dir}")
            return []
        
        # Load mask videos for each object
        overlap_masks = []
        
        for obj_id in object_ids:
            mask_video_path = masks_dir / f"object_{obj_id}.mp4"
            
            if not mask_video_path.exists():
                logger.warning(f"          Mask video not found: {mask_video_path}")
                continue
            
            # Open video
            cap = cv2.VideoCapture(str(mask_video_path))
            if not cap.isOpened():
                logger.warning(f"          Cannot open mask video: {mask_video_path}")
                continue
            
            # Get total frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine which frames to read
            if position == 'end':
                # Last N frames
                start_frame = max(0, total_frames - self.overlap_frames)
                frame_indices = range(start_frame, total_frames)
            else:  # 'start'
                # First N frames
                frame_indices = range(min(self.overlap_frames, total_frames))
            
            # Read frames
            object_masks = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert to binary mask
                    if len(frame.shape) == 3:
                        frame = frame[:, :, 0]  # Take first channel
                    mask = (frame > 0).astype(np.uint8)
                    object_masks.append(mask)
                else:
                    # Frame read failed, use empty mask
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    object_masks.append(np.zeros((height, width), dtype=np.uint8))
            
            cap.release()
            
            # Store masks organized by frame
            for i, mask in enumerate(object_masks):
                if i >= len(overlap_masks):
                    overlap_masks.append({})
                overlap_masks[i][obj_id] = mask
        
        return overlap_masks
    
    def _stitch_masks_for_prompt(self, prompt: str, prompt_mappings: Dict):
        """
        Stitch masks for a single prompt by following mapping chains.
        
        Args:
            prompt: Prompt name.
            prompt_mappings: Chunk-pair mappings for this prompt.
                           Format: {chunk_pair: {j_id: i_id}}
        """
        from sam3.utils.helpers import sanitize_filename
        
        safe_prompt = sanitize_filename(prompt)
        prompt_output_dir = self.masks_output_dir / safe_prompt
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        
        fps = self.video_metadata.get("fps", 25)
        width = self.video_metadata.get("width")
        height = self.video_metadata.get("height")
        total_video_frames = self.video_metadata.get("nb_frames")
        
        # Build chains: starting object_id -> list of (chunk_id, local_id)
        chains = self._build_chains_from_mappings(prompt, prompt_mappings)
        
        # Log chain summary
        tracked_chains = sum(1 for chain in chains.values() if len(chain) > 1)
        singleton_chains = sum(1 for chain in chains.values() if len(chain) == 1)
        logger.info(f"      Built {len(chains)} chains for '{prompt}': {tracked_chains} tracked across chunks, {singleton_chains} unmatched singletons")
        
        # Stitch each chain
        for start_obj_id, chain in chains.items():
            output_path = prompt_output_dir / f"object_{start_obj_id}.mp4"
            
            logger.debug(f"        Stitching chain starting with object_{start_obj_id}: {len(chain)} chunk(s)...")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
                isColor=False
            )
            
            if not writer.isOpened():
                logger.error(f"          Failed to create video writer for {output_path}")
                continue
            
            try:
                frames_written = 0
                
                # Process each segment in the chain
                for chain_idx, (chunk_id, local_id) in enumerate(chain):
                    # Find chunk result
                    chunk_result = self._get_chunk_result(chunk_id)
                    if chunk_result is None:
                        logger.warning(f"          Chunk {chunk_id} not found")
                        continue
                    
                    # Get mask directory path
                    prompt_result = chunk_result["prompts"].get(prompt, {})
                    masks_dir = prompt_result.get("masks_dir")
                    if not masks_dir:
                        logger.warning(f"          Masks dir not found for chunk {chunk_id}")
                        continue
                    
                    mask_object_dir = Path(masks_dir) / f"object_{local_id}"
                    if not mask_object_dir.exists():
                        logger.warning(f"          Mask directory not found: {mask_object_dir}")
                        continue
                    
                    # Copy frames from this segment
                    frames_copied = self._copy_frames_from_mask_pngs(
                        mask_object_dir, 
                        writer, 
                        skip_first_n=(self.overlap_frames if chain_idx > 0 else 0)
                    )
                    frames_written += frames_copied
                    logger.debug(f"          Copied {frames_copied} frames from chunk {chunk_id} object {local_id}")
                
                # Pad with black frames if needed
                if total_video_frames and frames_written < total_video_frames:
                    black_frames_needed = total_video_frames - frames_written
                    black_frame = np.zeros((height, width), dtype=np.uint8)
                    for _ in range(black_frames_needed):
                        writer.write(black_frame)
                    logger.debug(f"          Padded with {black_frames_needed} black frames")
            
            finally:
                writer.release()
            
            logger.debug(f"        Saved stitched mask to {output_path}")
    
    def _build_chains_from_mappings(self, prompt: str, prompt_mappings: Dict) -> Dict:
        """
        Build object ID chains by following mappings.
        
        Args:
            prompt: Prompt name.
            prompt_mappings: Mapping dictionary {chunk_pair: {j_id: i_id}}.
        
        Returns:
            Dictionary {start_obj_id: [(chunk_id, local_id), ...]}.
        """
        chains = {}
        processed = set()  # Track (chunk_id, local_id) to avoid duplicates
        
        # Get all object IDs from all chunks for this prompt
        all_objects = []  # List of (chunk_id, local_id)
        for chunk_result in self.chunk_results:
            chunk_id = chunk_result["chunk_id"]
            prompt_result = chunk_result["prompts"].get(prompt, {})
            object_ids = prompt_result.get("object_ids", [])
            for local_id in object_ids:
                all_objects.append((chunk_id, local_id))
        
        # Start chains from chunk_0 objects
        first_chunk = self.chunk_results[0]
        first_chunk_id = first_chunk["chunk_id"]
        prompt_result = first_chunk["prompts"].get(prompt, {})
        first_chunk_objects = prompt_result.get("object_ids", [])
        
        logger.debug(f"        Starting chains from chunk_{first_chunk_id}: {len(first_chunk_objects)} objects")
        
        for start_obj_id in first_chunk_objects:
            chains[start_obj_id] = [(first_chunk_id, start_obj_id)]
            processed.add((first_chunk_id, start_obj_id))
            
            # Follow the chain through mappings
            current_chunk_id = first_chunk_id
            current_obj_id = start_obj_id
            
            while True:
                # Find next chunk
                next_chunk_id = current_chunk_id + 1
                if next_chunk_id >= len(self.chunk_results):
                    break  # No more chunks
                
                mapping_key = f"chunk_{current_chunk_id:03d}->chunk_{next_chunk_id:03d}"
                mapping = prompt_mappings.get(mapping_key, {})
                
                # Mapping format: {i_id: j_id} where i is current chunk, j is next chunk
                # Check if current object maps to something in next chunk
                next_obj_id = mapping.get(current_obj_id)
                
                if next_obj_id is not None:
                    # Continue chain
                    chains[start_obj_id].append((next_chunk_id, next_obj_id))
                    processed.add((next_chunk_id, next_obj_id))
                    current_chunk_id = next_chunk_id
                    current_obj_id = next_obj_id
                else:
                    # Chain ends
                    break
            
            # Log tracked chain
            if len(chains[start_obj_id]) > 1:
                logger.info(f"          Tracked chain object_{start_obj_id}: {len(chains[start_obj_id])} chunks - {chains[start_obj_id]}")
        
        # Handle unmatched objects (not in any chain yet)
        # Give them IDs that don't conflict with first chunk IDs
        max_first_chunk_id = max(first_chunk_objects) if first_chunk_objects else -1
        unmatched_counter = max_first_chunk_id + 1
        
        for chunk_id, local_id in all_objects:
            if (chunk_id, local_id) not in processed:
                # Create a single-object chain with sequential ID
                chains[unmatched_counter] = [(chunk_id, local_id)]
                processed.add((chunk_id, local_id))
                logger.debug(f"          Unmatched object: chunk_{chunk_id} object_{local_id} -> output object_{unmatched_counter}")
                unmatched_counter += 1
        
        return chains
    
    def _get_chunk_result(self, chunk_id: int) -> Optional[Dict]:
        """Get chunk result by chunk ID."""
        for chunk_result in self.chunk_results:
            if chunk_result["chunk_id"] == chunk_id:
                return chunk_result
        return None
    
    def _copy_frames_from_mask_pngs(
        self, 
        mask_dir: Path, 
        writer: cv2.VideoWriter,
        skip_first_n: int = 0
    ) -> int:
        """
        Copy frames from PNG mask images to the output writer.
        
        Args:
            mask_dir: Directory containing PNG mask files (frame_*.png).
            writer: VideoWriter to write frames to.
            skip_first_n: Number of initial frames to skip (for overlap removal).
        
        Returns:
            Number of frames copied.
        """
        if not mask_dir.exists():
            logger.warning(f"          Mask directory not found: {mask_dir}")
            return 0
        
        # Get all PNG files sorted by frame number
        png_files = sorted(mask_dir.glob("frame_*.png"))
        
        if not png_files:
            logger.warning(f"          No PNG files found in: {mask_dir}")
            return 0
        
        frames_copied = 0
        
        # Skip first N frames (overlap removal)
        for png_file in png_files[skip_first_n:]:
            try:
                # Use PIL for reading (matches PIL saving in chunk_processor)
                pil_image = Image.open(png_file)
                mask_image = np.array(pil_image)
                
                if mask_image is not None and mask_image.size > 0:
                    # Validate mask before writing to video
                    if len(mask_image.shape) == 2 and mask_image.dtype == np.uint8:
                        writer.write(mask_image)
                        frames_copied += 1
                    else:
                        logger.warning(f"          Invalid mask shape/dtype: {png_file} - shape={mask_image.shape}, dtype={mask_image.dtype}")
                else:
                    logger.warning(f"          Failed to read PNG (corrupted or empty): {png_file}")
            except Exception as e:
                logger.error(f"          Error reading PNG {png_file}: {e}")
        
        return frames_copied
