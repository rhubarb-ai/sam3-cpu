import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from sam3.drivers import Sam3ImageDriver, Sam3VideoDriver
from sam3.ffmpeglib import ffmpeg_lib
from sam3.logger import get_logger
from sam3.memory_manager import memory_manager
from sam3.__globals import (
    BPE_PATH,
    DEVICE, 
    TEMP_DIR, 
    SUPPORTED_VIDEO_FORMATS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROPAGATION_DIRECTION
)

logger = get_logger(__name__)

class Sam3Entry:
    def __init__(
        self, 
        bpe_path: Optional[str] = None,
        num_workers: Optional[int] = None
    ):
        self.bpe_path = bpe_path or BPE_PATH
        self.num_workers = num_workers or DEFAULT_NUM_WORKERS
        self._video_driver = None  # Will be initialized when processing videos
        self._image_driver = None  # Will be initialized when processing images

    # ========================================================================
    # Video Processing Utility Methods
    # ========================================================================

    def _save_chunk_metadata(self, video_path: str | Path, metadata: dict):
        video_name = str(video_path).split("/")[-1].split(".")[0]
        meta_dir = Path(TEMP_DIR) / video_name / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / "chunk_plan.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, default=str)
        return video_name, meta_path
    
    def _save_prompt_video_metadata(self, metadata: dict, video_name: str, prompt: str, chunk_idx: Optional[int] = None):
        file_name = f"{prompt}.json"
        
        meta_dir = Path(TEMP_DIR) / video_name / "metadata" / str(chunk_idx) if chunk_idx is not None else Path(TEMP_DIR) / video_name / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / file_name

        with open(meta_path, "w") as f:
            json.dump(metadata, f, default=str)
        
        return meta_path
    
    def _update_prompt_video_metadata(self, new_data: dict, video_name: str, prompt: str, chunk_idx: Optional[int] = None):
        file_name = f"{prompt}.json"
        
        meta_dir = Path(TEMP_DIR) / video_name / "metadata" / str(chunk_idx) if chunk_idx is not None else Path(TEMP_DIR) / video_name / "metadata"
        meta_path = meta_dir / file_name

        if not meta_path.is_file():
            logger.warning(f"Metadata file not found for update: {meta_path}")
            logger.warning(f"Saving new metadata instead of updating.")
            self._save_prompt_video_metadata(new_data, video_name, prompt, chunk_idx=chunk_idx)
            return meta_path
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        metadata.update(new_data)

        with open(meta_path, "w") as f:
            json.dump(metadata, f, default=str)
        
        return meta_path
    
    def _save_prompt_video_mask_data(self, result_prompt, object_ids, prompt, video_name, video_meta):
        # Placeholder for actual mask saving logic
        logger.info(f"Saving mask data for prompt '{prompt}' in video '{video_name}'")

        logger.debug(f"Object IDs: {object_ids}")
        
        prompt_masks_dir = Path(TEMP_DIR) / video_name / "masks" / prompt
        prompt_masks_dir.mkdir(parents=True, exist_ok=True)

        for object_id in object_ids:
            mask_path = prompt_masks_dir / f"object_{object_id}.mp4"
            # Placeholder: Save the mask for this object ID to mask_path
            logger.debug(f"Saving mask for object ID {object_id} to {mask_path}")
            # Example: save_mask(mask_data, mask_path)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = video_meta.get("fps") 
            width = video_meta.get("width")  
            height = video_meta.get("height") 
            video_writer = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)
        
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to create video writer for {mask_path}")
            
            try:
                # Placeholder: Write frames to the video writer
                logger.debug(f"Writing mask frames for object ID {object_id} to {mask_path}")
                for _, out in sorted(result_prompt.items()):
                    if object_id in out.get("out_obj_ids", []):
                        if out["out_binary_masks"][object_id].any():
                            mask_bool = out["out_binary_masks"][object_id]
                            mask_uint8 = (mask_bool.astype(np.uint8) * 255)  # 0 for False, 255 for True
                            # cv2.imwrite("temp_mask.png", mask_uint8)  # Debug: Save a sample mask frame
                            video_writer.write(mask_uint8)
                    else:
                        # Write a blank frame if this object ID is not present in the current output
                        blank_frame = np.zeros((height, width), dtype=np.uint8)
                        video_writer.write(blank_frame)
            finally:
                video_writer.release()
        return prompt_masks_dir

    
    def _save_prompt_video_data(
        self, 
        result_prompt, 
        object_ids, 
        frame_objects,
        prompt, 
        video_name, 
        video_meta
    ):
        prompt_masks_dir = self._save_prompt_video_mask_data(result_prompt, object_ids, prompt, video_name, video_meta)
        metadata = {
            "mask_dir": str(prompt_masks_dir),
            "object_ids": list(object_ids),
            "frame_objects": frame_objects
        }
        return self._save_prompt_video_metadata(metadata, video_name, prompt)

    # ========================================================================
    # Input Validation Methods
    # ========================================================================
    
    def _validate_image_input(
        self,
        image_path: Union[str | Path, List[str | Path]],
        prompts: Optional[Union[str, List[str]]] = None,
        boxes: Optional[List[List[float]]] = None
    ) -> Tuple[List[str], Optional[List[str]], Optional[List[List[float]]]]:
        """Validate and normalize image input."""
        # Normalize image paths to list
        if isinstance(image_path, (str, Path)):
            image_paths = [str(image_path)]
        else:
            image_paths = list(image_path)
        
        # Validate all image files exist
        for path in image_paths:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
        
        # Normalize prompts to list
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            else:
                prompts = list(prompts)
        
        # Validate boxes if provided
        if boxes is not None:
            if not isinstance(boxes, list):
                raise ValueError("boxes must be a list")
            
            # Check if single box [x, y, w, h] or multiple [[x, y, w, h], ...]
            if len(boxes) > 0 and isinstance(boxes[0], (int, float)):
                # Single box, wrap in list
                boxes = [boxes]
            
            # Validate box format
            for box in boxes:
                if len(box) != 4:
                    raise ValueError(f"Each box must have 4 values [x, y, w, h], got {len(box)}")
        
        # Ensure either prompts or boxes are provided
        if prompts is None and boxes is None:
            raise ValueError("Must provide either prompts or boxes")
        
        return image_paths, prompts, boxes
    
    def _validate_video_input(
        self,
        video_path: Path,
        prompts: Optional[Union[str, List[str]]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        object_ids: Optional[List[int]] = None,
        segments: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Validate and normalize video input."""
        # Validate video file exists
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        result = {"video_path": video_path}
        
        # Normalize prompts
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]
            result["prompts"] = list(prompts)
        
        # Validate points and labels
        if points is not None:
            if not isinstance(points, list):
                raise ValueError("points must be a list of [x, y] coordinates")
            
            if point_labels is None:
                raise ValueError("point_labels must be provided when points are specified")
            
            if len(points) != len(point_labels):
                raise ValueError(
                    f"Number of points ({len(points)}) must match number of labels ({len(point_labels)})"
                )
            
            # Validate each point has 2 coordinates
            for i, point in enumerate(points):
                if len(point) != 2:
                    raise ValueError(f"Point {i} must have 2 coordinates [x, y], got {len(point)}")
            
            result["points"] = points
            result["point_labels"] = list(point_labels)
        
        # Validate object IDs
        if object_ids is not None:
            if isinstance(object_ids, int):
                object_ids = [object_ids]
            result["object_ids"] = list(object_ids)
        
        # Validate segments (scenario i)
        if segments is not None:
            if not isinstance(segments, dict):
                raise ValueError("segments must be a dictionary")
            
            if "segments" not in segments:
                raise ValueError("segments dict must contain 'segments' key")
            
            # Validate each segment
            for i, segment in enumerate(segments["segments"]):
                # Check for frame-based or time-based
                has_frames = "start_frame" in segment and "end_frame" in segment
                has_time = "start_time_sec" in segment and "end_time_sec" in segment
                
                if not (has_frames or has_time):
                    raise ValueError(
                        f"Segment {i} must have either (start_frame, end_frame) or "
                        f"(start_time_sec, end_time_sec)"
                    )
                
                # Check for prompts or points
                has_prompts = "prompts" in segment
                has_points = "points" in segment and "labels" in segment
                
                if not (has_prompts or has_points):
                    raise ValueError(
                        f"Segment {i} must have either 'prompts' or ('points' and 'labels')"
                    )
                
                # Validate points/labels consistency
                if has_points:
                    if len(segment["points"]) != len(segment["labels"]):
                        raise ValueError(
                            f"Segment {i}: number of points must match number of labels"
                        )
            
            result["segments"] = segments
        
        return result
    

    # ========================================================================
    # Output Management Methods
    # ========================================================================
    
    def _create_output_directory(
        self,
        base_dir: str,
        video_or_image_name: str,
        temp_dir: str = TEMP_DIR,
    ) -> Path:
        """Create output directory structure."""
        output_dir = Path(base_dir) / video_or_image_name
        temp_output_dir = Path(temp_dir) / video_or_image_name
        
        # Create main output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temp output directory
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create chunks subdirectory
        chunks_dir = temp_output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)

        # Create masks subdirectory
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)

        #Create metadata subdirectory
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        metadata_dir = temp_output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        return output_dir, temp_output_dir
    
    # ========================================================================
    # Chunk Processing Methods
    # ========================================================================
    
    def _process_video_prompt(
        self, 
        driver, 
        session_id,
        propagation_direction,
        prompt,
        video_name,
        video_meta: Dict[str, Any]
    ):
        driver.reset_session(session_id)
        driver.add_prompt(session_id, prompt)
        result_prompt, object_ids, frame_objects = driver.propagate_in_video(session_id, propagation_direction=propagation_direction)

        return self._save_prompt_video_data(
            result_prompt,
            object_ids,
            frame_objects,
            prompt,
            video_name,
            video_meta
        )


    def _process_video_chunk(
        self,
        chunk_path: str | Path,
        prompts: List[str],
        video_name: str,
        video_meta: Dict[str, Any],
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION
    ):
        driver = self._get_video_driver()

        # Start session for this chunk
        session_id = driver.start_session(video_path=str(chunk_path)) 

        results = {}

        for prompt in prompts:
            logger.debug(f"Processing chunk {chunk_path} with prompt: {prompt}")
            
            prompt_meta_path = self._process_video_prompt(driver, session_id, propagation_direction, prompt, video_name, video_meta)
            results[prompt] = prompt_meta_path
        return results

    def _process_video_segment(
        self,
        video_path: str | Path,
        prompts: List[str],
        video_meta: Dict[str, Any],
        output_dir: Path,
        temp_output_dir: Path,
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION
    ):
        pass

    def _process_video_in_chunks(
        self,
        video_path: Path,
        prompts: List[str],
        video_meta: Dict[str, Any],
        video_chunks: List[Dict[str, int]],
        output_dir: Path,
        temp_output_dir: Path,
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION
    ):
        output = {}
        for chunk in video_chunks:
            chunk_video_dir = temp_output_dir / "chunks" / "video"
            chunk_video_dir.mkdir(exist_ok=True)
            chunk_path = chunk_video_dir / f"chunk_{chunk['chunk']}.mp4"
            
            # Placeholder: Extract the chunk from the original video and save to chunk_path
            logger.info(f"Extracting chunk {chunk['chunk']} (frames {chunk['start']} to {chunk['end']}) to {chunk_path}")
            
            _ = ffmpeg_lib.create_video_chunk(
                input_video=video_path,
                output_video=chunk_path,
                start_frame=chunk['start'],
                end_frame=chunk['end']
             )

            chunk_result = self._process_video_chunk(
                chunk_path=chunk_path,
                prompts=prompts,
                video_name=video_path.stem,
                video_meta=video_meta,
                propagation_direction=propagation_direction
            )
            output[f"chunk_{chunk['chunk']}"] = {
                "results": chunk_result,
                "start_frame": chunk['start'],
                "end_frame": chunk['end'],
                "chunk_video_path": chunk_path
            }

        return output

    # ========================================================================
    # Video Processing Methods
    # ========================================================================
    

    def process_video_with_prompts(
        self, 
        video_path: str | Path, 
        prompts: Union[str, List[str]],
        output_dir: Optional[str | Path] = DEFAULT_OUTPUT_DIR,
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION,
        # frame_from: Union[int, str, float, None] = None,
        # frame_to: Union[int, str, float, None] = None,
        device: str = DEVICE.type,
        chunk_spread: Literal["even", "default"] = "default"
    ):
        if isinstance(video_path, str):
            video_path = Path(video_path)
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        # Validate input
        validated = self._validate_video_input(video_path, prompts=prompts)
        prompts = validated["prompts"]

        if device not in ['cpu', 'cuda']:
            raise ValueError(
                f"Unsupported device type: {device}. Supported types are 'cpu' and 'cuda'.")
        
        logger.info(f"Processing video: {video_path} with prompt: {prompts}")
        
        video_meta, video_chunks = memory_manager.chunk_plan_video(
            video_path, device=device, chunk_spread=chunk_spread)
        
        # Save this metadata for later use in processing in the temp directory
        video_name, meta_path = self._save_chunk_metadata(video_path, video_meta)

        # Create output directory structure
        output_dir, temp_output_dir = self._create_output_directory(output_dir, video_name)

        if len(video_chunks) == 0:
            logger.warning(f"No valid chunks generated for video: {video_path}. Skipping processing.")
            return None
        elif len(video_chunks) == 1:
            logger.info(f"Processing video in a single chunk.")
            result = self._process_video_segment(
                video_path=video_path,
                prompts=prompts,
                video_meta=video_meta,
                output_dir=output_dir,
                temp_output_dir=temp_output_dir,
                propagation_direction=propagation_direction)
        else:
            logger.info(f"Processing video in {len(video_chunks)} chunks.")
            result = self._process_video_in_chunks(
                video_path=video_path,
                prompts=prompts,
                video_meta=video_meta,
                video_chunks=video_chunks,
                output_dir=output_dir,
                temp_output_dir=temp_output_dir,
                propagation_direction=propagation_direction)
            
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _get_image_driver(self) -> Sam3ImageDriver:
        """Lazy-load image driver."""
        if self._image_driver is None:
            self._image_driver = Sam3ImageDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers or 1
            )
        return self._image_driver
    
    def _get_video_driver(self) -> Sam3VideoDriver:
        """Lazy-load video driver."""
        if self._video_driver is None:
            self._video_driver = Sam3VideoDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers or 1
            )
        return self._video_driver