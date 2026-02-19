"""
SAM3 Chunk Processor Module

Handles processing of a single video chunk with multiple prompts.
Manages video driver sessions and mask generation for chunk-level operations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from sam3.logger import get_logger
from sam3.__globals import DEFAULT_PROPAGATION_DIRECTION
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
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION
    ) -> Dict[str, Any]:
        """
        Process chunk with all prompts sequentially.
        
        Workflow for each prompt:
        1. Reset session (clear previous prompt state)
        2. Add prompt to session
        3. Propagate masks through chunk frames
        4. Save masks for each object ID
        5. Save metadata for the prompt
        
        Args:
            prompts: List of text prompts to process.
            propagation_direction: Direction for mask propagation.
        
        Returns:
            Dictionary containing:
                - "chunk_id": Chunk identifier
                - "chunk_video_path": Path to chunk video
                - "prompts": Dictionary mapping prompts to their results
                - "num_prompts": Number of processed prompts
        """
        logger.debug(f"Processing chunk {self.chunk_id} with {len(prompts)} prompt(s)")
        
        # Start video session for this chunk (driver already loaded, just start new session)
        logger.debug(f"  Starting video session for chunk {self.chunk_id}")
        session_id = self.driver.start_session(video_path=str(self.chunk_video_path))
        
        try:
            prompt_results = {}
            
            for prompt in prompts:
                logger.info(f"    Processing prompt: '{prompt}'")
                
                # Process single prompt
                prompt_result = self._process_single_prompt(
                    session_id=session_id,
                    prompt=prompt,
                    propagation_direction=propagation_direction
                )
                
                prompt_results[prompt] = prompt_result
        
        finally:
            # Close session after processing all prompts (release video memory, keep driver)
            self.driver.close_session(session_id)
            logger.debug(f"  Closed session for chunk {self.chunk_id} (driver still active)")
        
        return {
            "chunk_id": self.chunk_id,
            "chunk_video_path": str(self.chunk_video_path),
            "prompts": prompt_results,
            "num_prompts": len(prompts)
        }
    
    def _process_single_prompt(
        self,
        session_id: int,
        prompt: str,
        propagation_direction: str
    ) -> Dict[str, Any]:
        """
        Process a single prompt for this chunk.
        
        Args:
            session_id: Video session identifier.
            prompt: Text prompt to process.
            propagation_direction: Direction for mask propagation.
        
        Returns:
            Dictionary containing:
                - "prompt": The text prompt
                - "num_objects": Number of detected objects
                - "object_ids": List of object IDs
                - "frame_objects": Mapping of frame indices to object IDs
                - "masks_dir": Directory where masks are saved
                - "metadata_path": Path to prompt metadata file
        """
        # Reset session to clear previous prompt state
        self.driver.reset_session(session_id)
        
        # Add prompt to session
        self.driver.add_prompt(session_id, prompt)
        
        # Propagate masks through video frames
        logger.debug(f"      Propagating masks with direction: {propagation_direction}")
        result_prompt, object_ids, frame_objects = self.driver.propagate_in_video(
            session_id,
            propagation_direction=propagation_direction
        )
        
        num_objects = len(object_ids)
        logger.info(f"      Found {num_objects} object(s) for prompt '{prompt}'")
        
        # Create prompt-specific directory for masks
        safe_prompt = sanitize_filename(prompt)
        prompt_masks_dir = self.masks_dir / safe_prompt
        prompt_masks_dir.mkdir(exist_ok=True)
        
        # Save masks for each object
        if num_objects > 0:
            self._save_prompt_masks(
                result_prompt=result_prompt,
                object_ids=object_ids,
                prompt_masks_dir=prompt_masks_dir
            )
        
        # Save prompt metadata
        prompt_metadata = {
            "chunk_id": self.chunk_id,
            "prompt": prompt,
            "num_objects": num_objects,
            "object_ids": list(object_ids),
            "frame_objects": frame_objects,
            "masks_dir": str(prompt_masks_dir)
        }
        
        prompt_metadata_path = self.metadata_dir / f"{safe_prompt}_metadata.json"
        with open(prompt_metadata_path, "w") as f:
            json.dump(prompt_metadata, f, indent=2)
        
        logger.debug(f"      Saved prompt metadata to {prompt_metadata_path}")
        
        return {
            "prompt": prompt,
            "num_objects": num_objects,
            "object_ids": list(object_ids),
            "frame_objects": frame_objects,
            "masks_dir": str(prompt_masks_dir),
            "metadata_path": str(prompt_metadata_path)
        }
    
    def _save_prompt_masks(
        self,
        result_prompt: Dict,
        object_ids: set,
        prompt_masks_dir: Path
    ):
        """
        Save masks for all detected objects as video files.
        
        Creates one MP4 video file per object ID, where each frame shows the
        binary mask for that object (255 for object pixels, 0 for background).
        
        Args:
            result_prompt: Dictionary mapping frame indices to prediction results.
            object_ids: Set of all object IDs detected in this prompt.
            prompt_masks_dir: Directory to save mask videos.
        """
        logger.debug(f"        Saving masks for {len(object_ids)} object(s)")
        
        # Get video properties
        fps = self.video_metadata.get("fps", 25)
        width = self.video_metadata.get("width")
        height = self.video_metadata.get("height")
        
        # Create a video writer for each object ID
        video_writers = {}
        
        for object_id in object_ids:
            mask_video_path = prompt_masks_dir / f"object_{object_id}.mp4"
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(mask_video_path),
                fourcc,
                fps,
                (width, height),
                isColor=False
            )
            
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to create video writer for {mask_video_path}")
            
            video_writers[object_id] = video_writer
        
        try:
            # Iterate through frames in order
            for frame_idx in sorted(result_prompt.keys()):
                frame_output = result_prompt[frame_idx]
                
                # For each object ID, write its mask frame
                for object_id in object_ids:
                    video_writer = video_writers[object_id]
                    
                    # Get out_obj_ids and ensure it's a list/array we can work with
                    out_obj_ids = frame_output.get("out_obj_ids", [])
                    
                    # Convert to list if it's a numpy array
                    if isinstance(out_obj_ids, np.ndarray):
                        out_obj_ids = out_obj_ids.tolist()
                    
                    # Check if this object exists in current frame
                    if object_id in out_obj_ids:
                        # Find the position of object_id in out_obj_ids list
                        # out_binary_masks is indexed positionally, not by object_id
                        try:
                            obj_idx = out_obj_ids.index(object_id)
                            mask_bool = frame_output["out_binary_masks"][obj_idx]
                            
                            # Convert to uint8 (0 or 255)
                            if mask_bool.any():
                                mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                            else:
                                # Empty mask
                                mask_uint8 = np.zeros((height, width), dtype=np.uint8)
                        except (IndexError, ValueError) as e:
                            # If we can't find the object or index is wrong, write blank
                            logger.warning(f"Could not get mask for object {object_id} in frame {frame_idx}: {e}")
                            mask_uint8 = np.zeros((height, width), dtype=np.uint8)
                    else:
                        # Object not present in this frame, write blank
                        mask_uint8 = np.zeros((height, width), dtype=np.uint8)
                    
                    # Write frame to video
                    video_writer.write(mask_uint8)
        
        finally:
            # Release all video writers
            for object_id, video_writer in video_writers.items():
                video_writer.release()
                logger.debug(f"        Saved mask video for object {object_id}")
    
    def cleanup(self):
        """Release resources and free memory.
        
        Note: This method is intentionally left empty because the driver is shared
        across all chunks and will be cleaned up by VideoProcessor at the end.
        """
        # Driver is shared, so we don't clean it up here
        # VideoProcessor will call driver.cleanup() after all chunks are done
        pass
