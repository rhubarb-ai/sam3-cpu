"""
SAM3 API Module

Main entry point for SAM3 image and video segmentation. This module provides a clean,
modular interface for all SAM3 segmentation tasks with intelligent resource handling.

Supported Input Scenarios:
    - Single/multiple image(s) + text prompts
    - Single/multiple image(s) + bounding boxes
    - Video + text prompts (with automatic chunking and memory management)
    - Video + point prompts (clicks) with labels
    - Video + segments with frame/time ranges

Usage:
    >>> from sam3 import Sam3API
    >>> api = Sam3API()
    >>> # Process video with prompts
    >>> result = api.process_video_with_prompts(
    ...     video_path="video.mp4",
    ...     prompts=["person", "car"],
    ...     output_dir="./results"
    ... )
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from sam3.drivers import Sam3ImageDriver, Sam3VideoDriver
from sam3.utils.logger import get_logger
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


class Sam3API:
    """
    Main API class for SAM3 segmentation operations.
    
    This class provides a unified interface for image and video segmentation with
    automatic device selection (CPU/GPU), memory management, and chunking strategies.
    
    Args:
        bpe_path: Path to the BPE tokenizer model file. Defaults to built-in path.
        num_workers: Number of worker threads for CPU processing. Defaults to 1.
        device: Device to use ('cpu' or 'cuda'). Auto-detected if not specified.
        temp_dir: Temporary directory for intermediate files. Defaults to /tmp/sam3-cpu or /tmp/sam3-gpu.
        output_dir: Default output directory for results. Defaults to ./results.
    
    Example:
        >>> api = Sam3API()
        >>> # Process video
        >>> result = api.process_video_with_prompts(
        ...     video_path="video.mp4",
        ...     prompts=["person", "car"]
        ... )
        >>> # Process image
        >>> result = api.process_image_with_prompts(
        ...     image_path="image.jpg",
        ...     prompts=["person"]
        ... )
    """
    
    def __init__(
        self, 
        bpe_path: Optional[str] = None,
        num_workers: Optional[int] = None,
        device: Optional[str] = None,
        temp_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """Initialize the SAM3 API."""
        self.bpe_path = bpe_path or BPE_PATH
        self.num_workers = num_workers or DEFAULT_NUM_WORKERS
        self.device = device or DEVICE.type
        self.temp_dir = Path(temp_dir or TEMP_DIR)
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        
        # Ensure directories exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Lazy-loaded drivers
        self._video_driver = None
        self._image_driver = None
        
        logger.info(f"Initialized Sam3API with device: {self.device}")
        logger.debug(f"Temp directory: {self.temp_dir}")
        logger.debug(f"Output directory: {self.output_dir}")

    # ========================================================================
    # Video Processing Methods
    # ========================================================================

    def process_video_with_prompts(
        self, 
        video_path: Union[str, Path], 
        prompts: Union[str, List[str]],
        output_dir: Optional[Union[str, Path]] = None,
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION,
        device: Optional[str] = None,
        chunk_spread: Literal["even", "default"] = "default",
        keep_temp_files: bool = False
    ) -> Dict[str, Any]:
        """
        Process a video with text prompts and generate segmentation masks.
        
        This method handles:
        - Automatic memory checking and video chunking
        - Sequential prompt processing per chunk
        - Mask generation and storage
        - Optional visualization overlay (placeholder)
        - Cleanup of temporary files
        
        Args:
            video_path: Path to the input video file.
            prompts: Single prompt string or list of prompts (e.g., ["person", "car"]).
            output_dir: Directory for output files. Defaults to instance output_dir.
            propagation_direction: Direction for mask propagation ("forward", "backward", "both").
            device: Device override ('cpu' or 'cuda'). Uses instance device if not specified.
            chunk_spread: Chunking strategy ("even" or "default").
            keep_temp_files: If True, moves temp files to output_dir. If False, deletes temp files.
        
        Returns:
            Dictionary containing:
                - "video_name": Name of the video
                - "output_dir": Path to output directory
                - "chunks": List of chunk results
                - "prompts": List of processed prompts
                - "metadata_path": Path to metadata file
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If device is not supported or prompts are invalid.
        
        Example:
            >>> api = Sam3API()
            >>> result = api.process_video_with_prompts(
            ...     video_path="video.mp4",
            ...     prompts=["person", "car"],
            ...     keep_temp_files=True
            ... )
            >>> print(f"Processed {len(result['chunks'])} chunks")
        """
        from sam3.video_processor import VideoProcessor
        
        # Validate and normalize inputs
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not video_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
            logger.warning(
                f"Video format {video_path.suffix} may not be supported. "
                f"Supported formats: {SUPPORTED_VIDEO_FORMATS}"
            )
        
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        
        if not prompts:
            raise ValueError("At least one prompt must be provided")
        
        # Use device override or instance device
        device = device or self.device
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Unsupported device: {device}. Must be 'cpu' or 'cuda'.")
        
        # Use output_dir override or instance output_dir
        output_dir = Path(output_dir) if output_dir else self.output_dir
        
        logger.info(f"Processing video: {video_path.name}")
        logger.info(f"Prompts: {prompts}")
        logger.info(f"Device: {device}")
        
        # Initialize VideoProcessor
        processor = VideoProcessor(
            video_path=video_path,
            output_dir=output_dir,
            temp_dir=self.temp_dir,
            device=device,
            bpe_path=self.bpe_path,
            num_workers=self.num_workers
        )
        
        # Process video with prompts
        result = processor.process_with_prompts(
            prompts=prompts,
            propagation_direction=propagation_direction,
            chunk_spread=chunk_spread,
            keep_temp_files=keep_temp_files
        )
        
        logger.info(f"Video processing complete. Output: {result['output_dir']}")
        
        return result

    # ========================================================================
    # Image Processing Methods
    # ========================================================================

    def process_image_with_prompts(
        self,
        image_path: Union[str, Path, List[Union[str, Path]]],
        prompts: Union[str, List[str]],
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process one or more images with text prompts and generate segmentation masks.
        
        This method processes images sequentially (image-by-image), with each image
        being fully processed with all prompts before moving to the next image.
        
        Args:
            image_path: Path to single image or list of image paths.
            prompts: Single prompt string or list of prompts.
            output_dir: Directory for output files. Defaults to instance output_dir.
            device: Device override ('cpu' or 'cuda'). Uses instance device if not specified.
        
        Returns:
            Dictionary containing:
                - "images": List of image results
                - "output_dir": Path to output directory
                - "prompts": List of processed prompts
        
        Raises:
            FileNotFoundError: If any image file doesn't exist.
            ValueError: If device is not supported or prompts are invalid.
        
        Example:
            >>> api = Sam3API()
            >>> # Single image
            >>> result = api.process_image_with_prompts(
            ...     image_path="image.jpg",
            ...     prompts=["person", "car"]
            ... )
            >>> # Multiple images
            >>> result = api.process_image_with_prompts(
            ...     image_path=["img1.jpg", "img2.jpg"],
            ...     prompts=["person"]
            ... )
        """
        from sam3.image_processor import ImageProcessor
        
        # Normalize image paths to list
        if isinstance(image_path, (str, Path)):
            image_paths = [Path(image_path)]
        else:
            image_paths = [Path(p) for p in image_path]
        
        # Validate all image files exist
        for path in image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
        
        # Normalize prompts to list
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        
        if not prompts:
            raise ValueError("At least one prompt must be provided")
        
        # Use device override or instance device
        device = device or self.device
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Unsupported device: {device}. Must be 'cpu' or 'cuda'.")
        
        # Use output_dir override or instance output_dir
        output_dir = Path(output_dir) if output_dir else self.output_dir
        
        logger.info(f"Processing {len(image_paths)} image(s) with {len(prompts)} prompt(s)")
        logger.info(f"Device: {device}")
        
        # Initialize ImageProcessor
        processor = ImageProcessor(
            output_dir=output_dir,
            device=device,
            bpe_path=self.bpe_path,
            num_workers=self.num_workers
        )
        
        # Process images with prompts
        result = processor.process_with_prompts(
            image_paths=image_paths,
            prompts=prompts
        )
        
        logger.info(f"Image processing complete. Output: {result['output_dir']}")
        
        return result

    def process_image_with_boxes(
        self,
        image_path: Union[str, Path],
        boxes: List[List[float]],
        box_labels: Optional[List[int]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an image with bounding box prompts and generate segmentation masks.
        
        Args:
            image_path: Path to the image file.
            boxes: List of bounding boxes in XYWH format [[x, y, w, h], ...].
            box_labels: Optional list of labels (1=include, 0=exclude) for each box.
                       If not provided, all boxes are treated as positive (include).
            output_dir: Directory for output files. Defaults to instance output_dir.
            device: Device override ('cpu' or 'cuda'). Uses instance device if not specified.
        
        Returns:
            Dictionary containing processing results and output paths.
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If boxes are invalid or labels don't match boxes.
        
        Example:
            >>> api = Sam3API()
            >>> boxes = [[100, 150, 200, 300], [400, 200, 180, 320]]
            >>> result = api.process_image_with_boxes(
            ...     image_path="image.jpg",
            ...     boxes=boxes
            ... )
        """
        from sam3.image_processor import ImageProcessor
        
        # Validate image path
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Validate boxes
        if not boxes or not isinstance(boxes, list):
            raise ValueError("boxes must be a non-empty list")
        
        # Check if single box [x, y, w, h] or multiple [[x, y, w, h], ...]
        if isinstance(boxes[0], (int, float)):
            # Single box, wrap in list
            boxes = [boxes]
        
        # Validate box format
        for i, box in enumerate(boxes):
            if len(box) != 4:
                raise ValueError(f"Box {i} must have 4 values [x, y, w, h], got {len(box)}")
        
        # Validate or create box_labels
        if box_labels is not None:
            if len(box_labels) != len(boxes):
                raise ValueError(
                    f"Number of box_labels ({len(box_labels)}) must match "
                    f"number of boxes ({len(boxes)})"
                )
        else:
            # Default: all boxes are positive (include)
            box_labels = [1] * len(boxes)
        
        # Use device override or instance device
        device = device or self.device
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Unsupported device: {device}. Must be 'cpu' or 'cuda'.")
        
        # Use output_dir override or instance output_dir
        output_dir = Path(output_dir) if output_dir else self.output_dir
        
        logger.info(f"Processing image with {len(boxes)} bounding box(es)")
        
        # Initialize ImageProcessor
        processor = ImageProcessor(
            output_dir=output_dir,
            device=device,
            bpe_path=self.bpe_path,
            num_workers=self.num_workers
        )
        
        # Process image with boxes
        result = processor.process_with_boxes(
            image_path=image_path,
            boxes=boxes,
            box_labels=box_labels
        )
        
        logger.info(f"Image processing complete. Output: {result['output_dir']}")
        
        return result

    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def _get_image_driver(self) -> Sam3ImageDriver:
        """Lazy-load image driver."""
        if self._image_driver is None:
            self._image_driver = Sam3ImageDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers
            )
        return self._image_driver
    
    def _get_video_driver(self) -> Sam3VideoDriver:
        """Lazy-load video driver."""
        if self._video_driver is None:
            self._video_driver = Sam3VideoDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers
            )
        return self._video_driver
    
    def cleanup(self):
        """
        Release resources and free memory.
        
        This method cleans up both image and video drivers if they were initialized.
        Essential for preventing memory leaks in long-running applications.
        
        Example:
            >>> api = Sam3API()
            >>> # ... process images/videos ...
            >>> api.cleanup()
        """
        if self._image_driver is not None:
            self._image_driver.cleanup()
            self._image_driver = None
            logger.debug("Image driver cleaned up")
        
        if self._video_driver is not None:
            self._video_driver.cleanup()
            self._video_driver = None
            logger.debug("Video driver cleaned up")
        
        logger.info("Sam3API cleanup complete")
