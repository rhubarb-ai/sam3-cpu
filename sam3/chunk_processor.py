"""
SAM3 Chunk Processor Module

Handles processing of a single video chunk with multiple prompts.
Manages video driver sessions and mask generation for chunk-level operations.
Supports cross-chunk mask injection for seamless object continuity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from sam3.logger import get_logger
from sam3.__globals import DEFAULT_PROPAGATION_DIRECTION, DEFAULT_MIN_CHUNK_OVERLAP
from sam3.utils import sanitize_filename

logger = get_logger(__name__)


class ChunkProcessor:
    """
    Handles processing of a single video chunk with multiple prompts.
    
    This processor:
    - Uses a shared video driver (model already loaded by VideoProcessor)
    - Starts a new session for the chunk
    - Iterates through prompts sequentially
    - Generates masks for each prompt
    - Saves masks and metadata per prompt
    - Closes session (but doesn't cleanup driver)
    
    Args:
        chunk_id: Unique identifier for this chunk.
        chunk_video_path: Path to the chunk video file.
        chunk_dir: Directory for storing chunk outputs.
        video_name: Name of the parent video.
        video_metadata: Metadata about the video (width, height, fps, etc.).
        driver: Shared Sam3VideoDriver instance (model already loaded).
    
    Attributes:
        chunk_id: Chunk identifier
        chunk_video_path: Path to chunk video
        chunk_dir: Chunk output directory
        video_name: Parent video name
    """
    
    def __init__(
        self,
        chunk_id: int,
        chunk_video_path: Path,
        chunk_dir: Path,
        video_name: str,
        video_metadata: Dict[str, Any],
        driver  # Shared driver passed from VideoProcessor
    ):
        """Initialize the ChunkProcessor.
        
        Args:
            chunk_id: Unique identifier for this chunk.
            chunk_video_path: Path to the chunk video file.
            chunk_dir: Directory for storing chunk outputs.
            video_name: Name of the parent video.
            video_metadata: Metadata about the video.
            driver: Shared Sam3VideoDriver instance (model already loaded).
        """
        self.chunk_id = chunk_id
        self.chunk_video_path = Path(chunk_video_path)
        self.chunk_dir = Path(chunk_dir)
        self.video_name = video_name
        self.video_metadata = video_metadata
        
        # Use shared driver (model already loaded)
        self.driver = driver
        
        # Create subdirectories for this chunk
        self.masks_dir = self.chunk_dir / "masks"
        self.masks_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.chunk_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.debug(f"ChunkProcessor initialized for chunk {chunk_id} with shared driver")
    

    
    def process_with_prompts(
        self,
        prompts: List[str],
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION,
        prev_chunk_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process chunk with all prompts sequentially.
        
        Workflow:
        1. Start session
        2. For each prompt: reset, add prompt, propagate
        3. If prev_chunk_data is provided, IoU-match first-frame masks against
           previous chunk's last-frame masks and remap IDs for consistency
        4. Save masks, extract carry-forward data
        5. Close session
        
        Args:
            prompts: List of text prompts to process.
            propagation_direction: Direction for mask propagation.
            prev_chunk_data: Optional dictionary from previous chunk containing:
                - "masks": Dict[prompt_str -> Dict[obj_id -> np.ndarray]] last-frame masks
                - "object_ids": Dict[prompt_str -> List[int]] object IDs per prompt
                - "global_next_id": int, next available global ID counter
                If provided, detected objects will be matched to previous chunk's
                objects via IoU and their IDs remapped for consistency.
        
        Returns:
            Dictionary containing:
                - "chunk_id": Chunk identifier
                - "chunk_video_path": Path to chunk video
                - "prompts": Dictionary mapping prompts to their results
                - "num_prompts": Number of processed prompts
                - "carry_forward": Dict with last-frame masks for next chunk handoff
        """
        logger.debug(f"Processing chunk {self.chunk_id} with {len(prompts)} prompt(s)")
        if prev_chunk_data:
            logger.info(f"  Chunk {self.chunk_id}: will match IDs against previous chunk")
        
        # Start video session for this chunk (driver already loaded, just start new session)
        logger.debug(f"  Starting video session for chunk {self.chunk_id}")
        session_id = self.driver.start_session(video_path=str(self.chunk_video_path))
        
        try:
            prompt_results = {}
            
            for prompt in prompts:
                logger.info(f"    Processing prompt: '{prompt}'")
                
                # Process single prompt (with post-propagation ID remapping)
                prompt_result = self._process_single_prompt(
                    session_id=session_id,
                    prompt=prompt,
                    propagation_direction=propagation_direction,
                    prev_chunk_data=prev_chunk_data
                )
                
                prompt_results[prompt] = prompt_result
            
            # Extract last-frame masks for carry-forward to next chunk
            carry_forward = self._extract_carry_forward_data(prompt_results)
        
        finally:
            # Close session after processing all prompts (release video memory, keep driver)
            self.driver.close_session(session_id)
            logger.debug(f"  Closed session for chunk {self.chunk_id} (driver still active)")
        
        return {
            "chunk_id": self.chunk_id,
            "chunk_video_path": str(self.chunk_video_path),
            "prompts": prompt_results,
            "num_prompts": len(prompts),
            "carry_forward": carry_forward
        }
    
    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Compute IoU between two binary/uint8 masks."""
        a = mask_a > 127 if mask_a.dtype == np.uint8 else mask_a.astype(bool)
        b = mask_b > 127 if mask_b.dtype == np.uint8 else mask_b.astype(bool)
        intersection = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(intersection / union) if union > 0 else 0.0

    def _match_and_remap_ids(
        self,
        result_prompt: dict,
        object_ids: set,
        prev_masks: Dict[int, np.ndarray],
        global_next_id: int,
        iou_threshold: float = 0.25,
    ):
        """
        Match this chunk's objects to previous chunk objects via IoU on the first frame,
        then remap all frame outputs for consistent IDs across chunks.

        Returns:
            Tuple of (remapped_result, remapped_object_ids, id_mapping, updated_global_next_id)
        """
        if not result_prompt or not prev_masks:
            id_mapping = {}
            for oid in sorted(object_ids):
                id_mapping[oid] = global_next_id
                global_next_id += 1
            remapped_result = self._apply_id_mapping(result_prompt, id_mapping)
            return remapped_result, set(id_mapping.values()), id_mapping, global_next_id

        first_frame_idx = min(result_prompt.keys())
        first_output = result_prompt.get(first_frame_idx)
        if first_output is None:
            id_mapping = {oid: oid for oid in object_ids}
            return result_prompt, object_ids, id_mapping, global_next_id

        out_obj_ids = first_output.get("out_obj_ids", [])
        if isinstance(out_obj_ids, np.ndarray):
            out_obj_ids = out_obj_ids.tolist()

        # Build first-frame masks for matching
        first_frame_masks = {}
        for obj_id in out_obj_ids:
            idx = out_obj_ids.index(obj_id)
            mask_bool = first_output["out_binary_masks"][idx]
            first_frame_masks[obj_id] = (mask_bool.astype(np.uint8) * 255)

        # Greedy IoU matching
        pairs = []
        for new_id, new_mask in first_frame_masks.items():
            for prev_id, prev_mask in prev_masks.items():
                iou = self._compute_iou(new_mask, prev_mask)
                if iou >= iou_threshold:
                    pairs.append((iou, new_id, prev_id))
        pairs.sort(reverse=True)

        id_mapping = {}
        used_prev_ids = set()
        for iou, new_id, prev_id in pairs:
            if new_id in id_mapping or prev_id in used_prev_ids:
                continue
            id_mapping[new_id] = prev_id
            used_prev_ids.add(prev_id)
            logger.debug(f"      Matched: chunk_obj_{new_id} -> global_obj_{prev_id} (IoU={iou:.3f})")

        # Assign fresh global IDs to unmatched objects
        for oid in sorted(object_ids):
            if oid not in id_mapping:
                id_mapping[oid] = global_next_id
                logger.debug(f"      New object: chunk_obj_{oid} -> global_obj_{global_next_id}")
                global_next_id += 1

        remapped_result = self._apply_id_mapping(result_prompt, id_mapping)
        remapped_ids = set(id_mapping.values())
        return remapped_result, remapped_ids, id_mapping, global_next_id

    @staticmethod
    def _apply_id_mapping(result_prompt: dict, id_mapping: Dict[int, int]) -> dict:
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

    def _process_single_prompt(
        self,
        session_id: int,
        prompt: str,
        propagation_direction: str,
        prev_chunk_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a single prompt for this chunk with post-propagation ID remapping.

        Workflow:
        1. Reset session, add prompt, propagate (chunk-local IDs)
        2. If prev_chunk_data provided, match first-frame masks against previous
           chunk's last-frame masks via IoU and remap all IDs for consistency
        3. Save masks with globally consistent IDs

        Args:
            session_id: Video session identifier.
            prompt: Text prompt to process.
            propagation_direction: Direction for mask propagation.
            prev_chunk_data: Optional dict with previous chunk's last-frame masks.
        
        Returns:
            Dictionary with prompt results including mask info.
        """
        # Reset session to clear previous prompt state
        self.driver.reset_session(session_id)
        
        # Add prompt to session (detects objects with chunk-local IDs)
        self.driver.add_prompt(session_id, prompt)
        
        # Propagate masks through video frames
        logger.debug(f"      Propagating masks with direction: {propagation_direction}")
        result_prompt, object_ids, frame_objects = self.driver.propagate_in_video(
            session_id,
            propagation_direction=propagation_direction
        )
        
        num_objects = len(object_ids)
        logger.info(f"      Found {num_objects} object(s) for prompt '{prompt}' (raw IDs: {sorted(object_ids)})")
        
        # Post-propagation ID remapping for cross-chunk consistency
        id_mapping = {}
        if prev_chunk_data and prompt in prev_chunk_data.get("masks", {}):
            prev_masks = prev_chunk_data["masks"][prompt]
            global_next_id = prev_chunk_data.get("global_next_id", {}).get(prompt, 0)
            
            if prev_masks:
                logger.info(f"      Matching {num_objects} objects against {len(prev_masks)} from previous chunk")
                result_prompt, object_ids, id_mapping, global_next_id = (
                    self._match_and_remap_ids(
                        result_prompt, object_ids,
                        prev_masks, global_next_id,
                    )
                )
                # Store updated global_next_id back for the next chunk
                if "global_next_id" not in prev_chunk_data:
                    prev_chunk_data["global_next_id"] = {}
                prev_chunk_data["global_next_id"][prompt] = global_next_id
                logger.info(f"      Remapped IDs: {sorted(object_ids)}")
        elif not prev_chunk_data:
            # First chunk: initialize global IDs (identity mapping for first chunk)
            global_next_id = 0
            for oid in sorted(object_ids):
                id_mapping[oid] = global_next_id
                global_next_id += 1
            result_prompt = self._apply_id_mapping(result_prompt, id_mapping)
            object_ids = set(id_mapping.values())
        
        # Create prompt-specific directory for masks
        safe_prompt = sanitize_filename(prompt)
        prompt_masks_dir = self.masks_dir / safe_prompt
        prompt_masks_dir.mkdir(exist_ok=True)
        
        # Save masks for each object (with globally consistent IDs)
        if num_objects > 0:
            self._save_prompt_masks(
                result_prompt=result_prompt,
                object_ids=object_ids,
                prompt_masks_dir=prompt_masks_dir
            )
        
        # Rebuild frame_objects with remapped IDs
        frame_objects = {}
        for fidx, output in result_prompt.items():
            oids = output.get("out_obj_ids", [])
            if isinstance(oids, np.ndarray):
                oids = oids.tolist()
            frame_objects[fidx] = oids
        
        # Save prompt metadata
        prompt_metadata = {
            "chunk_id": self.chunk_id,
            "prompt": prompt,
            "num_objects": len(object_ids),
            "object_ids": sorted(object_ids),
            "id_mapping": {str(k): v for k, v in id_mapping.items()},
            "frame_objects": frame_objects,
            "masks_dir": str(prompt_masks_dir)
        }
        
        prompt_metadata_path = self.metadata_dir / f"{safe_prompt}_metadata.json"
        with open(prompt_metadata_path, "w") as f:
            json.dump(prompt_metadata, f, indent=2)
        
        logger.debug(f"      Saved prompt metadata to {prompt_metadata_path}")
        
        return {
            "prompt": prompt,
            "num_objects": len(object_ids),
            "object_ids": sorted(object_ids),
            "id_mapping": {str(k): v for k, v in id_mapping.items()},
            "frame_objects": frame_objects,
            "masks_dir": str(prompt_masks_dir),
            "metadata_path": str(prompt_metadata_path),
            "_result_prompt": result_prompt  # Keep raw results for carry-forward extraction
        }
    
    def _save_prompt_masks(
        self,
        result_prompt: Dict,
        object_ids: set,
        prompt_masks_dir: Path
    ):
        """
        Save masks for all detected objects as PNG images (lossless).
        
        Creates one directory per object ID, with PNG files for each frame.
        This ensures pixel-perfect masks for accurate cross-chunk matching.
        
        Args:
            result_prompt: Dictionary mapping frame indices to prediction results.
            object_ids: Set of all object IDs detected in this prompt.
            prompt_masks_dir: Directory to save mask images.
        """
        logger.debug(f"        Saving masks for {len(object_ids)} object(s) as PNG images")
        
        # Get video properties
        width = self.video_metadata.get("width")
        height = self.video_metadata.get("height")
        
        # Get total frame count from chunk video
        cap = cv2.VideoCapture(str(self.chunk_video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if total_frames <= 0:
            logger.error(f"        Failed to get frame count for chunk video: {self.chunk_video_path}")
            return
        
        logger.debug(f"        Chunk has {total_frames} frames, will save PNG for each frame")
        
        # Create a directory for each object ID
        object_dirs = {}
        for object_id in object_ids:
            object_dir = prompt_masks_dir / f"object_{object_id}"
            object_dir.mkdir(parents=True, exist_ok=True)
            object_dirs[object_id] = object_dir
        
        # Iterate through ALL frames in the chunk (0 to total_frames-1)
        for frame_idx in range(total_frames):
            # Check if this frame has tracking results
            if frame_idx in result_prompt:
                frame_output = result_prompt[frame_idx]
            else:
                # Frame not tracked (no objects detected), will use blank masks
                frame_output = None
            
            # For each object ID, save its mask frame as PNG
            for object_id in object_ids:
                object_dir = prompt_masks_dir / f"object_{object_id}"
                
                # Initialize with blank mask (default)
                mask_uint8 = np.zeros((height, width), dtype=np.uint8)
                
                # If frame was tracked, try to get the actual mask
                if frame_output is not None:
                    # Get out_obj_ids and ensure it's a list/array we can work with
                    out_obj_ids = frame_output.get("out_obj_ids", [])
                    
                    # Convert to list if it's a numpy array
                    if isinstance(out_obj_ids, np.ndarray):
                        out_obj_ids = out_obj_ids.tolist()
                    
                    # Check if this object exists in current frame
                    if object_id in out_obj_ids:
                        # Find the position of object_id in out_obj_ids list
                        try:
                            obj_idx = out_obj_ids.index(object_id)
                            mask_bool = frame_output["out_binary_masks"][obj_idx]
                            
                            # Convert to uint8 (0 or 255)
                            if mask_bool.any():
                                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                            # else: keep blank mask (already initialized)
                        except (IndexError, ValueError) as e:
                            # If we can't find the object or index is wrong, use blank
                            logger.warning(f"Could not get mask for object {object_id} in frame {frame_idx}: {e}")
                    # else: object not in this frame, keep blank mask
                
                # Validate mask before saving
                if mask_uint8.shape != (height, width):
                    logger.error(f"Invalid mask shape {mask_uint8.shape}, expected ({height}, {width}). Creating blank mask.")
                    mask_uint8 = np.zeros((height, width), dtype=np.uint8)
                
                if mask_uint8.dtype != np.uint8:
                    logger.warning(f"Converting mask dtype from {mask_uint8.dtype} to uint8")
                    mask_uint8 = mask_uint8.astype(np.uint8)
                
                # Ensure 2D array (grayscale)
                if len(mask_uint8.shape) > 2:
                    logger.warning(f"Mask has {len(mask_uint8.shape)} dimensions, squeezing to 2D")
                    mask_uint8 = mask_uint8.squeeze()
                
                # Save frame as PNG using PIL (more robust than cv2.imwrite)
                png_path = object_dir / f"frame_{frame_idx:06d}.png"
                try:
                    # Convert numpy array to PIL Image and save
                    pil_image = Image.fromarray(mask_uint8, mode='L')  # 'L' mode = grayscale
                    pil_image.save(png_path, format='PNG', compress_level=1)
                except Exception as e:
                    logger.error(f"Failed to write PNG {png_path}: {e}")
        
        logger.debug(f"        Saved PNG masks for {len(object_ids)} object(s)")
    
    def _extract_carry_forward_data(
        self,
        prompt_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract last-frame mask data from prompt results for carry-forward to the next chunk.
        
        For each prompt, reads the last N overlap frames' masks from the saved PNG files
        and packages them as numpy arrays for injection into the next chunk.
        
        Args:
            prompt_results: Dictionary mapping prompt strings to their processing results.
        
        Returns:
            Dictionary containing:
                - "masks": Dict[prompt_str -> Dict[obj_id -> np.ndarray]] 
                    Maps each prompt to a dict of object masks from the last frame.
                    Each mask is a 2D uint8 numpy array (H x W, values 0 or 255).
                - "object_ids": Dict[prompt_str -> List[int]] object IDs per prompt.
        """
        carry_masks = {}
        carry_obj_ids = {}
        
        for prompt, result in prompt_results.items():
            obj_ids = result.get("object_ids", [])
            masks_dir = result.get("masks_dir")
            result_prompt = result.get("_result_prompt", {})
            
            if not obj_ids:
                continue
            
            prompt_masks = {}
            
            # Strategy 1: Extract directly from propagation result (most accurate)
            if result_prompt:
                # Find the last frame index that has results
                frame_indices = sorted(result_prompt.keys())
                if frame_indices:
                    last_frame_idx = frame_indices[-1]
                    last_frame_output = result_prompt[last_frame_idx]
                    
                    out_obj_ids = last_frame_output.get("out_obj_ids", [])
                    if isinstance(out_obj_ids, np.ndarray):
                        out_obj_ids = out_obj_ids.tolist()
                    
                    for obj_id in obj_ids:
                        if obj_id in out_obj_ids:
                            try:
                                obj_idx = out_obj_ids.index(obj_id)
                                mask_bool = last_frame_output["out_binary_masks"][obj_idx]
                                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                                prompt_masks[obj_id] = mask_uint8
                            except (IndexError, ValueError) as e:
                                logger.warning(
                                    f"carry_forward: could not extract mask for obj {obj_id} "
                                    f"from frame {last_frame_idx}: {e}"
                                )
            
            # Strategy 2: Fall back to reading saved PNG masks
            if not prompt_masks and masks_dir:
                masks_path = Path(masks_dir)
                # Get total frame count
                cap = cv2.VideoCapture(str(self.chunk_video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                last_frame_idx = total_frames - 1
                
                for obj_id in obj_ids:
                    png_path = masks_path / f"object_{obj_id}" / f"frame_{last_frame_idx:06d}.png"
                    if png_path.exists():
                        mask_img = np.array(Image.open(png_path).convert("L"))
                        if mask_img.any():  # Only include non-empty masks
                            prompt_masks[obj_id] = mask_img
            
            if prompt_masks:
                carry_masks[prompt] = prompt_masks
                carry_obj_ids[prompt] = list(prompt_masks.keys())
                logger.debug(
                    f"carry_forward: extracted {len(prompt_masks)} mask(s) "
                    f"for prompt '{prompt}'"
                )
        
        return {
            "masks": carry_masks,
            "object_ids": carry_obj_ids,
        }
    
    def cleanup(self):
        """Release resources and free memory.
        
        Note: This method is intentionally left empty because the driver is shared
        across all chunks and will be cleaned up by VideoProcessor at the end.
        """
        # Driver is shared, so we don't clean it up here
        # VideoProcessor will call driver.cleanup() after all chunks are done
        pass
