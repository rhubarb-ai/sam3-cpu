"""
SAM3 Image Processor Module

Handles image segmentation with text prompts and bounding box prompts.
Supports single and batch image processing with sequential prompt iteration.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from sam3.drivers import Sam3ImageDriver
from sam3.logger import get_logger
from sam3.__globals import BPE_PATH, DEFAULT_NUM_WORKERS
from sam3.utils import sanitize_filename

logger = get_logger(__name__)


class ImageProcessor:
    """
    Handles image segmentation with text prompts and bounding boxes.
    
    This processor:
    - Loads images sequentially (image-by-image processing)
    - Processes all prompts for each image before moving to the next
    - Saves masks and metadata for each prompt
    - Manages memory efficiently by cleaning up after each image
    
    Args:
        output_dir: Directory where results will be saved.
        device: Device to use ('cpu' or 'cuda').
        bpe_path: Path to the BPE tokenizer model file.
        num_workers: Number of worker threads for CPU processing.
    
    Attributes:
        driver: Sam3ImageDriver instance for performing inference.
    """
    
    def __init__(
        self,
        output_dir: Path,
        device: str,
        bpe_path: str = BPE_PATH,
        num_workers: int = DEFAULT_NUM_WORKERS
    ):
        """Initialize the ImageProcessor."""
        self.output_dir = Path(output_dir)
        self.device = device
        self.bpe_path = bpe_path
        self.num_workers = num_workers
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-load driver when needed
        self._driver = None
        
        logger.debug(f"ImageProcessor initialized with output_dir: {self.output_dir}")
    
    @property
    def driver(self) -> Sam3ImageDriver:
        """Lazy-load the image driver."""
        if self._driver is None:
            logger.debug("Initializing Sam3ImageDriver...")
            self._driver = Sam3ImageDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers
            )
            logger.debug("Sam3ImageDriver initialized")
        return self._driver
    
    def process_with_prompts(
        self,
        image_paths: List[Path],
        prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Process images with text prompts.
        
        Processing strategy:
        - For each image:
          1. Load image and initialize inference state
          2. Process all prompts sequentially
          3. Save masks and metadata for each prompt
          4. Clean up before moving to next image
        
        Args:
            image_paths: List of paths to image files.
            prompts: List of text prompts to process.
        
        Returns:
            Dictionary containing:
                - "images": List of per-image results
                - "output_dir": Path to output directory
                - "prompts": List of processed prompts
                - "total_images": Number of images processed
                - "total_prompts": Number of prompts per image
        """
        logger.info(f"Processing {len(image_paths)} image(s) with {len(prompts)} prompt(s)")
        
        results = {
            "images": [],
            "output_dir": str(self.output_dir),
            "prompts": prompts,
            "total_images": len(image_paths),
            "total_prompts": len(prompts)
        }
        
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path.name}")
            
            # Process single image with all prompts
            image_result = self._process_single_image_with_prompts(
                image_path=image_path,
                prompts=prompts
            )
            
            results["images"].append(image_result)
            
            # Clean up after each image to free memory
            if self._driver is not None:
                self._driver.cleanup()
                logger.debug(f"Cleaned up memory after processing {image_path.name}")
        
        logger.info(f"Completed processing {len(image_paths)} image(s)")
        
        return results
    
    def _process_single_image_with_prompts(
        self,
        image_path: Path,
        prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Process a single image with all prompts sequentially.
        
        Args:
            image_path: Path to the image file.
            prompts: List of text prompts to process.
        
        Returns:
            Dictionary containing:
                - "image_name": Name of the image
                - "image_path": Path to the image
                - "prompts": Dictionary mapping prompts to their results
                - "metadata_path": Path to metadata file
        """
        image_name = image_path.stem
        
        # Create output directory for this image
        image_output_dir = self.output_dir / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        masks_dir = image_output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        metadata_dir = image_output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        logger.debug(f"Created output directories for image: {image_name}")
        
        # Load image
        logger.debug(f"Loading image: {image_path}")
        image = Image.open(image_path)
        
        # Initialize inference state (encode image once)
        logger.debug("Initializing inference state...")
        processor, inference_state = self.driver.inference(image)
        
        # Process each prompt sequentially
        prompt_results = {}
        
        for prompt in prompts:
            logger.info(f"  Processing prompt: '{prompt}'")
            
            # Process prompt
            prompt_result = self._process_single_prompt(
                processor=processor,
                inference_state=inference_state,
                image=image,
                prompt=prompt,
                image_name=image_name,
                masks_dir=masks_dir,
                metadata_dir=metadata_dir
            )
            
            prompt_results[prompt] = prompt_result
        
        # Save overall metadata for the image
        overall_metadata = {
            "image_name": image_name,
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
            "prompts": prompts,
            "prompt_results": {
                prompt: {
                    "num_objects": len(result["object_ids"]),
                    "object_ids": result["object_ids"],
                    "masks_dir": str(result["masks_dir"])
                }
                for prompt, result in prompt_results.items()
            }
        }
        
        metadata_path = metadata_dir / f"{image_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(overall_metadata, f, indent=2)
        
        logger.debug(f"Saved overall metadata to {metadata_path}")
        
        return {
            "image_name": image_name,
            "image_path": str(image_path),
            "prompts": prompt_results,
            "metadata_path": str(metadata_path),
            "output_dir": str(image_output_dir)
        }
    
    def _process_single_prompt(
        self,
        processor,
        inference_state: Dict,
        image: Image.Image,
        prompt: str,
        image_name: str,
        masks_dir: Path,
        metadata_dir: Path
    ) -> Dict[str, Any]:
        """
        Process a single prompt for an image.
        
        Args:
            processor: Sam3Processor instance.
            inference_state: Current inference state.
            image: PIL Image instance.
            prompt: Text prompt to process.
            image_name: Name of the image.
            masks_dir: Directory to save masks.
            metadata_dir: Directory to save metadata.
        
        Returns:
            Dictionary containing:
                - "prompt": The text prompt
                - "num_objects": Number of detected objects
                - "object_ids": List of object IDs
                - "masks_dir": Directory where masks are saved
                - "metadata_path": Path to prompt metadata file
        """
        # Apply prompt and get predictions
        inference_state = self.driver.prompt_and_predict(
            processor=processor,
            inference_state=inference_state,
            prompt=prompt
        )
        
        # Extract results
        masks = inference_state.get("masks")
        scores = inference_state.get("scores")
        boxes = inference_state.get("boxes")
        
        num_objects = len(scores) if scores is not None else 0
        object_ids = list(range(num_objects))
        
        logger.info(f"    Found {num_objects} object(s) for prompt '{prompt}'")
        
        # Create prompt-specific directory
        safe_prompt = sanitize_filename(prompt)
        prompt_masks_dir = masks_dir / safe_prompt
        prompt_masks_dir.mkdir(exist_ok=True)
        
        # Save masks for each detected object
        if num_objects > 0:
            self._save_masks(
                masks=masks,
                scores=scores,
                boxes=boxes,
                object_ids=object_ids,
                prompt_masks_dir=prompt_masks_dir
            )
        
        # Save prompt metadata
        prompt_metadata = {
            "prompt": prompt,
            "image_name": image_name,
            "num_objects": num_objects,
            "object_ids": object_ids,
            "objects": [
                {
                    "object_id": i,
                    "score": float(scores[i]) if scores is not None else None,
                    "box": boxes[i].tolist() if boxes is not None else None,
                    "mask_file": f"object_{i}_mask.npy"
                }
                for i in range(num_objects)
            ]
        }
        
        prompt_metadata_path = metadata_dir / f"{safe_prompt}_metadata.json"
        with open(prompt_metadata_path, "w") as f:
            json.dump(prompt_metadata, f, indent=2)
        
        logger.debug(f"    Saved prompt metadata to {prompt_metadata_path}")
        
        return {
            "prompt": prompt,
            "num_objects": num_objects,
            "object_ids": object_ids,
            "masks_dir": str(prompt_masks_dir),
            "metadata_path": str(prompt_metadata_path)
        }
    
    def _save_masks(
        self,
        masks: torch.Tensor,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        object_ids: List[int],
        prompt_masks_dir: Path
    ):
        """
        Save masks for all detected objects.
        
        Args:
            masks: Tensor of binary masks.
            scores: Confidence scores.
            boxes: Bounding boxes.
            object_ids: List of object IDs.
            prompt_masks_dir: Directory to save masks.
        """
        for obj_id in object_ids:
            # Extract mask for this object
            mask = masks[obj_id]
            
            # Convert to numpy
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = mask
            
            # Save as numpy array
            mask_path = prompt_masks_dir / f"object_{obj_id}_mask.npy"
            np.save(mask_path, mask_np)
            
            # Also save as PNG for easy visualization
            mask_uint8 = (mask_np.astype(np.uint8) * 255)
            if len(mask_uint8.shape) == 3 and mask_uint8.shape[0] == 1:
                mask_uint8 = mask_uint8[0]  # Remove channel dimension if present
            
            mask_png_path = prompt_masks_dir / f"object_{obj_id}_mask.png"
            Image.fromarray(mask_uint8, mode='L').save(mask_png_path)
            
            logger.debug(f"      Saved mask for object {obj_id} to {mask_path}")
    
    def process_with_boxes(
        self,
        image_path: Path,
        boxes: List[List[float]],
        box_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Process an image with bounding box prompts.
        
        Args:
            image_path: Path to the image file.
            boxes: List of bounding boxes in XYWH format.
            box_labels: List of labels (1=include, 0=exclude) for each box.
        
        Returns:
            Dictionary containing processing results and output paths.
        """
        logger.info(f"Processing image with {len(boxes)} bounding box(es)")
        
        image_name = image_path.stem
        
        # Create output directory for this image
        image_output_dir = self.output_dir / image_name
        image_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        masks_dir = image_output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        metadata_dir = image_output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Load image
        logger.debug(f"Loading image: {image_path}")
        image = Image.open(image_path)
        
        # Initialize inference state
        logger.debug("Initializing inference state...")
        processor, inference_state = self.driver.inference(image)
        
        # Process with multiple boxes and labels
        logger.debug("Processing bounding boxes...")
        inference_state = self.driver.prompt_multi_box_with_labels(
            image=image,
            processor=processor,
            inference_state=inference_state,
            boxes_xywh=boxes,
            box_labels=box_labels
        )
        
        # Extract results
        masks = inference_state.get("masks")
        scores = inference_state.get("scores")
        result_boxes = inference_state.get("boxes")
        
        num_objects = len(scores) if scores is not None else 0
        object_ids = list(range(num_objects))
        
        logger.info(f"  Found {num_objects} object(s) from bounding boxes")
        
        # Save masks
        if num_objects > 0:
            boxes_masks_dir = masks_dir / "boxes"
            boxes_masks_dir.mkdir(exist_ok=True)
            
            self._save_masks(
                masks=masks,
                scores=scores,
                boxes=result_boxes,
                object_ids=object_ids,
                prompt_masks_dir=boxes_masks_dir
            )
        
        # Save metadata
        boxes_metadata = {
            "image_name": image_name,
            "image_path": str(image_path),
            "width": image.width,
            "height": image.height,
            "input_boxes": boxes,
            "box_labels": box_labels,
            "num_objects": num_objects,
            "object_ids": object_ids,
            "objects": [
                {
                    "object_id": i,
                    "score": float(scores[i]) if scores is not None else None,
                    "box": result_boxes[i].tolist() if result_boxes is not None else None,
                    "mask_file": f"object_{i}_mask.npy"
                }
                for i in range(num_objects)
            ]
        }
        
        metadata_path = metadata_dir / "boxes_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(boxes_metadata, f, indent=2)
        
        logger.debug(f"Saved boxes metadata to {metadata_path}")
        
        # Clean up
        if self._driver is not None:
            self._driver.cleanup()
        
        return {
            "image_name": image_name,
            "image_path": str(image_path),
            "num_objects": num_objects,
            "object_ids": object_ids,
            "masks_dir": str(boxes_masks_dir) if num_objects > 0 else None,
            "metadata_path": str(metadata_path),
            "output_dir": str(image_output_dir)
        }
    
    def cleanup(self):
        """Release resources and free memory."""
        if self._driver is not None:
            self._driver.cleanup()
            self._driver = None
            logger.debug("ImageProcessor driver cleaned up")
