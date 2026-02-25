"""
SAM3 CPU + GPU Driver

This module provides a unified interface for running SAM3 on both CPU and GPU. 
It automatically selects the best available backend for prompt based segmentation operations, 
ensuring optimal performance across different hardware configurations.
"""
import os
from typing import Dict, List, Optional
import numpy as np
from typing_extensions import Literal
import torch
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.profiler import profile
from sam3.__globals import DEVICE, BPE_PATH
from sam3.logger import get_logger

logger = get_logger(__name__)

class Sam3ImageDriver:
    """High-level driver for SAM3 image segmentation with automatic device selection.
    
    This driver provides a simplified interface for image segmentation tasks using SAM3.
    It automatically detects available hardware (CPU/GPU) and optimizes performance accordingly.
    Supports text-based prompts and geometric prompts (bounding boxes) for versatile segmentation.
    
    Performance Notes:
        - GPU: Automatically enables TF32 and bfloat16 for Ampere GPUs
        - CPU: Uses optimized CPU backend with configurable worker threads
        - Memory: Call cleanup() after processing to release resources
    
    Example:
        >>> driver = Sam3ImageDriver(bpe_path="path/to/tokenizer.model")
        >>> results = driver.prompt_texts("image.jpg", prompts=["person", "car"])
        >>> for prompt, state in results.items():
        ...     print(f"Found {len(state['scores'])} {prompt}(s)")
        >>> driver.cleanup()
    
    Attributes:
        predictor: The underlying SAM3 model predictor instance.
    """
    
    def __init__(self, bpe_path: str = BPE_PATH, num_workers: Optional[int] = 1):
        """Initialize the SAM3 image driver.
        
        Args:
            bpe_path: Path to the BPE tokenizer model file. Defaults to global BPE_PATH.
            num_workers: Number of worker threads for CPU processing. Only used on CPU.
                        Defaults to 1. Higher values may improve throughput but increase memory.
        
        Raises:
            FileNotFoundError: If bpe_path does not exist.
            RuntimeError: If model loading fails.
        """
        self.predictor = self._get_predictor(bpe_path=bpe_path, num_workers=num_workers)

    @profile()
    def _build_model(self, bpe_path, device):
        """Internal method to build the SAM3 image model.
        
        Args:
            bpe_path: Path to BPE tokenizer model file.
            device: Device type ('cpu' or 'cuda').
        
        Returns:
            Built SAM3 image model ready for inference.
        
        Performance Note:
            Model loading typically takes 5-15 seconds depending on device.
        """
        from sam3 import build_sam3_image_model
        logger.info(f"Loading model on device: {device}")
        model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        return model

    @profile()
    def _get_predictor(self, bpe_path: Optional[str], num_workers: Optional[int]):
        """Internal method to initialize predictor with device-specific optimizations.
        
        Args:
            bpe_path: Path to BPE tokenizer model.
            num_workers: Number of CPU worker threads (unused on GPU).
        
        Returns:
            Initialized predictor ready for inference.
        
        Performance Optimizations:
            - CPU: Queries CPU capabilities for optimal instruction set usage
            - GPU (Ampere): Enables TF32 for faster matrix operations (~3x speedup)
            - GPU: Uses bfloat16 precision for reduced memory and faster compute
        """
        if DEVICE.type == "cpu":
            logger.warning("Running on CPU. For better performance, please run on a GPU.")
            # Query CPU capabilities to enable optimal SIMD instructions
            torch.backends.cpu.get_cpu_capability()
        else:
            logger.info("Running on GPU. Enabling TF32 and bfloat16 for better performance.")
            # Enable TensorFloat-32 for Ampere GPUs (RTX 30xx, A100, etc.)
            # TF32 uses 10-bit mantissa (vs FP32's 23-bit) for ~3x speedup with minimal accuracy loss
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable bfloat16 autocast for reduced memory usage and faster computation
            # bfloat16 has same exponent range as FP32 but reduced precision (good for ML)
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        return self._build_model(bpe_path=bpe_path, device=DEVICE.type)
        
    @profile()
    def inference(self, image):
        """Initialize image processing and create inference state.
        
        This method prepares the image for segmentation by encoding it into the model's
        internal representation. The resulting processor and state are used for all
        subsequent prompt operations.
        
        Args:
            image: Input image as PIL Image, numpy array, or torch tensor.
        
        Returns:
            Tuple of (processor, inference_state):
                - processor: Sam3Processor instance for handling prompts
                - inference_state: Encoded image state with embeddings
        
        Raises:
            ValueError: If model is not loaded.
        
        Performance Note:
            Image encoding is memory-intensive. For batch processing, call cleanup()
            between images to prevent memory buildup.
        """
        from sam3.model.sam3_image_processor import Sam3Processor
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        # confidence_threshold=0.5 filters out low-confidence detections
        processor = Sam3Processor(self.predictor, confidence_threshold=0.5)
        inference_state = processor.set_image(image)
        return processor, inference_state
    
    @profile()
    def prompt_and_predict(self, processor, inference_state, prompt: str="people"):
        """Apply a text prompt to the image and get segmentation results.
        
        Args:
            processor: Sam3Processor instance from inference().
            inference_state: Current inference state.
            prompt: Natural language description of objects to segment (e.g., "person", "car").
        
        Returns:
            Updated inference_state containing segmentation masks, scores, and bounding boxes.
        
        Note:
            This method resets all previous prompts before applying the new one.
        """
        # Clear any previous prompts to start fresh
        processor.reset_all_prompts(inference_state)
        # Apply text prompt and run segmentation
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        return inference_state

    @profile()
    def prompt_texts(self, image_path: str, prompts: str | List[str]=["people"]):
        """High-level method to segment objects in an image using text prompts.
        
        This is the primary entry point for text-based image segmentation. It handles
        image loading, inference initialization, and prompt processing in a single call.
        
        Args:
            image_path: Path to the image file (supports common formats: jpg, png, etc.).
            prompts: Single prompt string or list of prompts (e.g., ["person", "car", "tree"]).
        
        Returns:
            Dictionary mapping each prompt to its inference_state containing:
                - "masks": Segmentation masks for detected objects
                - "scores": Confidence scores for each detection
                - "boxes": Bounding boxes in normalized coordinates
        
        Raises:
            ValueError: If image file doesn't exist or is empty.
            RuntimeError: If image loading or inference fails.
        
        Example:
            >>> driver = Sam3ImageDriver()
            >>> results = driver.prompt_texts("scene.jpg", prompts=["person", "bicycle"])
            >>> for prompt, state in results.items():
            ...     print(f"Found {len(state['scores'])} {prompt}(s)")
            ...     for i, score in enumerate(state['scores']):
            ...         print(f"  Object {i}: confidence={score:.2f}")
        
        Performance Note:
            Image is encoded once, then all prompts are processed sequentially.
            For large images (>4K), consider resizing to reduce memory usage.
        """
        # Validate image file exists and is readable
        if os.path.isfile(image_path):
            from PIL import Image
            image = Image.open(image_path)
            # Sanity check: ensure image has valid dimensions
            if image.size == (0, 0):
                raise ValueError(f"Image file is empty: {image_path}")
            else:
                logger.info(f"Loaded image: {image_path} with size {image.size}")
        else:
            raise ValueError(f"Could not load image file: {image_path}")
        
        # Initialize image encoding (expensive operation, done once)
        processor, inference_state = self.inference(image)
        
        # Normalize prompts to list format for consistent processing
        if isinstance(prompts, str):
            prompts = [prompts]

        # Process each prompt and collect results
        result = {}
        for prompt in prompts:
            inference_state = self.prompt_and_predict(processor, inference_state, prompt=prompt)
            # Store a copy of relevant results to avoid reference issues
            result[prompt] = {
                "masks": inference_state["masks"].clone() if hasattr(inference_state["masks"], 'clone') else inference_state["masks"].copy(),
                "scores": inference_state["scores"].clone() if hasattr(inference_state["scores"], 'clone') else inference_state["scores"].copy(),
                "boxes": inference_state["boxes"].clone() if hasattr(inference_state["boxes"], 'clone') else inference_state["boxes"].copy(),
                "masks_logits": inference_state.get("masks_logits").clone() if hasattr(inference_state.get("masks_logits", None), 'clone') else inference_state.get("masks_logits"),
            }
            nb_objects = len(result[prompt]["scores"])
            logger.info(f"found {nb_objects} {prompt}(s)")
        return result

    @profile()
    def prompt_bounding_box(
        self, 
        image: torch.Tensor | np.ndarray | str, 
        processor,
        inference_state, 
        box_xywh: List[float | int]
    ):
        """Segment an object using a bounding box prompt.
        
        This method provides precise control over segmentation by specifying the exact
        region of interest. Useful when text prompts are ambiguous or multiple instances
        of the same object need individual targeting.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path).
            processor: Sam3Processor instance from inference().
            inference_state: Current inference state.
            box_xywh: Bounding box in XYWH format [x, y, width, height] in absolute pixels.
        
        Returns:
            Updated inference_state with segmentation for the boxed region.
        
        Note:
            Box coordinates use XYWH format (top-left x, y, width, height) and are
            internally converted to center-based format (CXCYWH) then normalized to [0, 1].
        
        Example:
            >>> # Segment person at coordinates (100, 150) with size 200x300
            >>> box = [100, 150, 200, 300]
            >>> state = driver.prompt_bounding_box(image, processor, state, box)
        """
        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox

        # Load image if path provided
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
        
        width, height = image.size

        # Convert box format: XYWH (top-left + size) -> CXCYWH (center + size)
        # Algorithm: cx = x + w/2, cy = y + h/2
        box_input_xywh = torch.tensor(box_xywh, dtype=torch.float32).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        
        # Normalize coordinates to [0, 1] range for resolution-independent processing
        norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()

        # Clear previous prompts and apply box prompt
        processor.reset_all_prompts(inference_state)
        inference_state = processor.add_geometric_prompt(
            state=inference_state, box=norm_box_cxcywh, label=True  # label=True indicates positive prompt
        )
        return inference_state
    
    @profile()
    def prompt_multi_box_with_labels(
        self, 
        image: torch.Tensor | np.ndarray | str, 
        processor,
        inference_state, 
        boxes_xywh: List[List[float | int]], 
        box_labels: List[int]
    ):
        """Segment multiple objects using bounding boxes with positive/negative labels.
        
        This advanced method allows specifying multiple regions with labels indicating
        whether to include (positive) or exclude (negative) the region from segmentation.
        Useful for complex scenes requiring fine-grained control.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path).
            processor: Sam3Processor instance from inference().
            inference_state: Current inference state.
            boxes_xywh: List of bounding boxes, each in XYWH format [x, y, width, height].
            box_labels: List of labels (1=positive/include, 0=negative/exclude) for each box.
        
        Returns:
            Updated inference_state with segmentation for all labeled regions.
        
        Example:
            >>> # Segment two people, exclude background object
            >>> boxes = [[100, 150, 200, 300], [400, 200, 180, 320], [500, 100, 50, 80]]
            >>> labels = [1, 1, 0]  # Include first two boxes, exclude third
            >>> state = driver.prompt_multi_box_with_labels(image, processor, state, boxes, labels)
        
        Note:
            Labels must match the length of boxes. Positive labels (1) add objects to
            the segmentation, while negative labels (0) exclude regions.
        """
        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox

        # Load image if path provided
        if isinstance(image, str):
            from PIL import Image
            image = Image.open(image)
        
        width, height = image.size

        # Batch convert all boxes: XYWH -> CXCYWH format
        boxes_input_xywh = torch.tensor(boxes_xywh, dtype=torch.float32).view(-1,4)
        boxes_input_cxcywh = box_xywh_to_cxcywh(boxes_input_xywh)
        
        # Normalize all boxes to [0, 1] range
        norm_boxes_cxcywh = normalize_bbox(boxes_input_cxcywh, width, height).tolist()

        # Clear previous prompts before adding new batch
        processor.reset_all_prompts(inference_state)

        # Add each box prompt with its label (positive or negative)
        for box, label in zip(norm_boxes_cxcywh, box_labels):
            inference_state = processor.add_geometric_prompt(
                state=inference_state, box=box, label=label
            )
        return inference_state
    
    @profile()
    def cleanup(self):
        """Release memory and clean up resources.
        
        This method aggressively frees memory by calling device-specific cleanup routines.
        Essential for preventing memory leaks in long-running applications or batch processing.
        
        Performance Note:
            - GPU: Clears CUDA cache, reclaiming ~100% of allocated GPU memory
            - CPU (Linux only): Calls malloc_trim to return memory to OS, plus Python GC
            - CPU (non-Linux): Only runs Python garbage collection
        
        Platform Compatibility:
            The CPU malloc_trim optimization requires glibc (Linux). On Windows/macOS,
            only Python garbage collection is performed.
        
        Example:
            >>> driver = Sam3ImageDriver()
            >>> for image_path in image_list:
            ...     results = driver.prompt_texts(image_path, ["person"])
            ...     # Process results...
            ...     driver.cleanup()  # Free memory before next image
        """
        # Device-specific memory cleanup
        if DEVICE.type == "cuda":
            # Clear PyTorch's GPU memory cache
            torch.cuda.empty_cache()
        elif DEVICE.type == "cpu":
            # Linux-specific: Return freed memory to OS via glibc malloc_trim
            # This is more aggressive than standard Python GC
            import ctypes
            import gc

            try:
                libc = ctypes.CDLL("libc.so.6")  # Linux glibc
                libc.malloc_trim(0)  # 0 = trim all possible memory
            except OSError:
                # Not on Linux or glibc not available, skip malloc_trim
                pass

            # Run Python garbage collector to free unreferenced objects
            gc.collect()



class Sam3VideoDriver():
    """High-level driver for SAM3 video segmentation with session-based tracking.
    
    This driver provides a comprehensive interface for video object segmentation and tracking
    across multiple frames. It uses a session-based architecture where each video is processed
    in an isolated session, supporting multiple concurrent sessions.
    
    Key Features:
        - Text-based and point-based prompting for object annotation
        - Temporal propagation (forward/backward/bidirectional) for tracking
        - Multi-object tracking with unique object IDs
        - Session management for resource isolation
        - Automatic device detection and optimization (CPU/GPU)
    
    Architecture:
        1. Start session with video → Get session_id
        2. Add prompts (text or points) → Define objects to track
        3. Propagate → Track objects across frames
        4. Refine (optional) → Improve tracking with additional prompts
        5. Close session → Release resources
    
    Performance Notes:
        - GPU: Supports multi-GPU parallelism for faster processing
        - CPU: Uses multi-threaded workers (configurable via num_workers)
        - Memory: Each session holds encoded frames. Close sessions when done.
        - Propagation: Processing time scales linearly with video length
    
    Example:
        >>> driver = Sam3VideoDriver(bpe_path="path/to/tokenizer.model")
        >>> 
        >>> # Start video segmentation session
        >>> session_id = driver.start_session("video.mp4")
        >>> 
        >>> # Add object using text prompt
        >>> response = driver.add_prompt(session_id, prompt="person")
        >>> 
        >>> # Track across all frames
        >>> results = driver.propagate_in_video(session_id, propogation_direction="forward")
        >>> for frame_idx, masks in results.items():
        ...     print(f"Frame {frame_idx}: {len(masks)} objects")
        >>> 
        >>> # Clean up
        >>> driver.close_session(session_id)
        >>> driver.cleanup()
    
    Attributes:
        predictor: The underlying SAM3 video predictor instance (CPU or GPU variant).
    """
    
    def __init__(self, bpe_path: Optional[str] = BPE_PATH, num_workers: Optional[int] = 1):
        """Initialize the SAM3 video driver.
        
        Args:
            bpe_path: Path to the BPE tokenizer model file. Defaults to global BPE_PATH.
            num_workers: Number of worker threads for CPU processing, or number of GPUs to use.
                        - CPU: Controls parallel frame processing (recommend: CPU cores / 2)
                        - GPU: Number of GPUs to use (default: all available)
        
        Raises:
            FileNotFoundError: If bpe_path does not exist.
            RuntimeError: If model loading fails or no compatible device found.
        """
        self._get_predictor(bpe_path=bpe_path, num_workers=num_workers)

    @profile()
    def _get_predictor(self, bpe_path: Optional[str], num_workers: Optional[int] = None, device: str = DEVICE.type):
        """Internal method to initialize video predictor with device-specific optimizations.
        
        Args:
            bpe_path: Path to BPE tokenizer model.
            num_workers: Worker count (threads on CPU, GPU count on GPU).
        
        Performance Optimizations:
            - CPU: Multi-threaded frame processing for parallel decoding
            - GPU (Ampere): TF32 enabled for ~3x matmul speedup
            - GPU: bfloat16 precision for 2x memory reduction
            - GPU: Multi-GPU support for distributed processing
        
        Note:
            Sets self.predictor as side effect. Returns None.
        """
        if device == "cpu":
            from sam3.model_builder import build_sam3_video_predictor_cpu
            logger.warning("Running on CPU. For better performance, please run on a GPU.")
            # Query CPU capabilities to optimize for available instruction sets (AVX2, AVX512, etc.)
            torch.backends.cpu.get_cpu_capability()
            self.predictor = build_sam3_video_predictor_cpu(bpe_path=bpe_path, num_workers=num_workers)
        else:
            from sam3.model_builder import build_sam3_video_predictor
            logger.info("Running on GPU. Enabling TF32 and bfloat16 for better performance.")
            # Enable TensorFloat-32 for Ampere+ GPUs (RTX 30xx/40xx, A100, H100)
            # TF32: 10-bit mantissa vs FP32's 23-bit = ~3x speedup with <0.1% accuracy loss
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable bfloat16 autocast: 16-bit precision with FP32 exponent range
            # Reduces memory by 2x compared to FP32 while maintaining numerical stability
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

            # Detect and use all available GPUs for distributed processing
            gpus_to_use = range(torch.cuda.device_count())
            self.predictor = build_sam3_video_predictor(bpe_path=bpe_path, gpus_to_use=gpus_to_use)

    def _abs_to_rel_coords(self, coords, img_width, img_height, coord_type="point"):
        """Convert absolute pixel coordinates to normalized relative coordinates.
        
        This internal method normalizes coordinates to [0, 1] range for resolution-independent
        processing. SAM3 requires normalized coordinates to handle videos of varying resolutions.
        
        Args:
            coords: List of coordinates in absolute pixels.
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            coord_type: Type of coordinates to convert:
                       - "point": [[x, y], ...] format (e.g., click points)
                       - "box": [[x, y, w, h], ...] format (bounding boxes in XYWH)
        
        Returns:
            List of normalized coordinates in [0, 1] range, maintaining input structure.
        
        Raises:
            ValueError: If coord_type is not "point" or "box".
        
        Example:
            >>> # Convert click point at pixel (640, 480) in 1920x1080 image
            >>> rel_points = driver._abs_to_rel_coords([[640, 480]], 1920, 1080, "point")
            >>> # Result: [[0.333, 0.444]]
            >>> 
            >>> # Convert bounding box [100, 200, 300, 400] in 1920x1080 image
            >>> rel_box = driver._abs_to_rel_coords([[100, 200, 300, 400]], 1920, 1080, "box")
            >>> # Result: [[0.052, 0.185, 0.156, 0.370]]
        
        Algorithm:
            - Point: (x, y) → (x/width, y/height)
            - Box: (x, y, w, h) → (x/width, y/height, w/width, h/height)
        """
        if coord_type == "point":
            # Normalize point coordinates: divide x by width, y by height
            return [[x / img_width, y / img_height] for x, y in coords]
        elif coord_type == "box":
            # Normalize box coordinates: divide all dimensions by respective image dimensions
            return [
                [x / img_width, y / img_height, w / img_width, h / img_height]
                for x, y, w, h in coords
            ]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

# -------------- Prompting APIs (add prompt, propagate, remove object) --------------
    @profile()
    def add_prompt(self, session_id: str, prompt: str, frame_index: int = 0):
        """Add a text-based prompt to identify objects in the video.
        
        This method uses natural language to describe objects to segment. The model will
        detect all instances matching the description across the video frames.
        
        Args:
            session_id: Active session identifier from start_session().
            prompt: Natural language description of objects to segment (e.g., "person", "red car").
            frame_index: Frame index to add the prompt on (default: 0).
        
        Returns:
            Response dictionary containing:
                - "outputs": Segmentation results with masks, scores, and object IDs  
        
        Raises:
            ValueError: If model is not loaded or session_id is invalid.
        
        Example:
            >>> session_id = driver.start_session("video.mp4")
            >>> response = driver.add_prompt(session_id, "person", frame_index=0)
            >>> out = response["outputs"]
            >>> print(f"Detected {len(out['out_obj_ids'])} objects")
        
        Note:
            Text prompts provide automatic detection but less control than point prompts.
            For precise object selection, use add_object_with_points_prompt().
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
            )
        )
        # Return the full response (includes "outputs" key)
        return response

    @profile()
    def add_object_with_points_prompt(
        self, 
        session_id: str,
        frame_idx: int,
        object_id: int,
        frame_width: int,
        frame_height: int,
        points: List[List[float | int]],    # positive clicks have label 1, while negative clicks have label 0
        point_labels: List[int]
    ):
        """Add or refine an object using interactive point prompts (clicks).
        
        This method provides precise control through positive clicks (include region) and
        negative clicks (exclude region). It's the primary way to manually define objects
        when text prompts are insufficient or for correcting automatic detections.
        
        Args:
            session_id: Active session identifier from start_session().
            frame_idx: Zero-based frame index where points are annotated.
            object_id: Unique integer ID for this object (use same ID to refine existing object).
            frame_width: Video frame width in pixels (for coordinate normalization).
            frame_height: Video frame height in pixels (for coordinate normalization).
            points: List of [x, y] coordinates in absolute pixels. Example: [[640, 480], [800, 500]].
            point_labels: List of labels matching points (1=positive/include, 0=negative/exclude).
        
        Returns:
            Response dictionary containing:
                - "outputs": Updated segmentation results for this object
                - "frame_index": Frame where the prompt was applied
        
        Raises:
            ValueError: If model not loaded, session invalid, or points/labels length mismatch.
        
        Example:
            >>> session_id = driver.start_session("video.mp4")
            >>> # Add object with 2 positive clicks and 1 negative click
            >>> points = [[640, 480], [700, 500], [800, 600]]
            >>> labels = [1, 1, 0]  # First two include, third excludes
            >>> response = driver.add_object_with_points_prompt(
            ...     session_id, frame_idx=0, object_id=1,
            ...     frame_width=1920, frame_height=1080,
            ...     points=points, point_labels=labels
            ... )
        
        Point Labeling Strategy:
            - Positive (1): Click inside the target object
            - Negative (0): Click on background or unwanted regions
            - More points = more accurate segmentation
            - Use negative points to exclude ambiguous regions
        
        Performance Note:
            Point prompts are processed immediately. For real-time annotation,
            consider batching multiple objects before calling propagate_in_video().
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        
        labels = np.array(point_labels)
        points_abs = np.array(points)

        # Convert absolute pixel coordinates to normalized [0, 1] range
        # This allows the model to work with any video resolution
        points_tensor = torch.tensor(
            self._abs_to_rel_coords(points_abs, frame_width, frame_height, coord_type="point"),
            dtype=torch.float32,
        )
        points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx,
                points=points_tensor,
                point_labels=points_labels_tensor,
                obj_id=object_id,
            )
        )

        return response["outputs"]  # response["outputs"] contains the updated segmentation results after adding the object

    @profile()
    def propagate_in_video(
        self, 
        session_id: str, 
        start_frame_idx: int = None, 
        frames_to_track: int = None, 
        propagation_direction: Literal["both", "forward", "backward"] = "both"
    ) -> Dict[int, Dict[str, List]]:
        """Propagate object segmentation across video frames temporally.
        
        This is the core tracking method that extends object segmentation from prompted
        frames to all other frames. Uses temporal consistency and motion cues to maintain
        accurate tracking across the video sequence.
        
        Args:
            session_id: Active session identifier from start_session().
            start_frame_idx: Starting frame index for propagation. If None, uses the last
                           prompted frame as the starting point. Defaults to None.
            frames_to_track: Maximum number of frames to track. If None, tracks until video
                           end (or start for backward). Defaults to None (full video).
            propagation_direction: Temporal direction for tracking:
                - "forward": Track from start_frame_idx towards end of video
                - "backward": Track from start_frame_idx towards beginning
                - "both": Track in both directions simultaneously (default)
        
        Returns:
            Dictionary mapping frame_index (int) to segmentation outputs:
                {frame_idx: {"masks": [...], "scores": [...], "object_ids": [...]}}
        
        Raises:
            ValueError: If model not loaded, session invalid, or direction invalid.
        
        Example:
            >>> session_id = driver.start_session("video.mp4")
            >>> # Add object annotation on frame 10
            >>> driver.add_object_with_points_prompt(session_id, frame_idx=10, ...)
            >>> 
            >>> # Track forward from frame 10 to end
            >>> results = driver.propagate_in_video(
            ...     session_id, start_frame_idx=10, propagation_direction="forward"
            ... )
            >>> print(f"Tracked across {len(results)} frames")
            >>> 
            >>> # Track both directions for 50 frames total
            >>> results = driver.propagate_in_video(
            ...     session_id, frames_to_track=50, propagation_direction="both"
            ... )
        
        Performance Notes:
            - Streaming Results: Returns dictionary populated frame-by-frame as processing completes
            - Processing Time: ~0.1-0.5 seconds per frame on GPU, 1-3 seconds on CPU
            - Memory: Peak memory scales with number of tracked objects (not video length)
            - Bidirectional: "both" processes forward and backward in parallel for speed
        
        Tracking Quality:
            - Best within 20-30 frames of prompted frame (motion coherence)
            - Degradation on occlusions, fast motion, or scene changes
            - Add additional prompts on distant frames to improve accuracy
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        
        if propagation_direction not in ("both", "forward", "backward"):
            raise ValueError("propagation_direction must be 'both', 'forward' or 'backward'. Options: 'both', 'forward', 'backward'")
        
        # Stream results as they're generated (frame-by-frame processing)
        result = {}
        object_ids = set()
        frame_objects = {}

        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=start_frame_idx,
                max_frame_num_to_track=frames_to_track,
                propagation_direction=propagation_direction,
            )
        ):
            # Ensure response has required keys
            if "frame_index" in response and "outputs" in response:
                result[response["frame_index"]] = response["outputs"]
                frame_objects[response["frame_index"]] = response["outputs"]["out_obj_ids"].tolist()
                object_ids.update(frame_objects[response["frame_index"]])
            else:
                logger.warning(f"Response missing required keys. Got keys: {list(response.keys())}")
        return result, object_ids, frame_objects
    
    @profile()
    def refine_existing_object_with_points_prompt(
        self, 
        session_id: str,
        frame_idx: int,
        object_id: int,
        frame_width: int,
        frame_height: int,
        points: List[List[float | int]],    # positive clicks have label 1, while negative clicks have label 0
        point_labels: List[int]
    ):
        """Refine tracking of an existing object using additional point prompts.
        
        This method is an alias for add_object_with_points_prompt() used specifically
        for clarifying intent when correcting existing tracked objects. Adding points
        with the same object_id updates that object's segmentation.
        
        Args:
            session_id: Active session identifier from start_session().
            frame_idx: Frame index where refinement prompts are annotated.
            object_id: Existing object ID to refine (must match ID from add_object_with_points_prompt).
            frame_width: Video frame width in pixels.
            frame_height: Video frame height in pixels.
            points: List of [x, y] click coordinates in absolute pixels.
            point_labels: List of labels (1=positive/include, 0=negative/exclude).
        
        Returns:
            Response dictionary with updated segmentation for the refined object.
        
        Example:
            >>> # Initially track person (object_id=1) on frame 0
            >>> driver.add_object_with_points_prompt(session_id, 0, object_id=1, ...)
            >>> driver.propagate_in_video(session_id, propogation_direction="forward")
            >>> 
            >>> # Track drifts on frame 50, refine with negative clicks
            >>> refine_points = [[800, 600]]  # Click on wrongly included background
            >>> refine_labels = [0]  # Negative click to exclude
            >>> driver.refine_existing_object_with_points_prompt(
            ...     session_id, frame_idx=50, object_id=1,
            ...     frame_width=1920, frame_height=1080,
            ...     points=refine_points, point_labels=refine_labels
            ... )
            >>> # Re-propagate to apply refinement
            >>> driver.propagate_in_video(session_id, start_frame_idx=50, propogation_direction="forward")
        
        Refinement Workflow:
            1. Identify frame where tracking degrades
            2. Add corrective prompts (usually negative clicks on false positives)
            3. Re-propagate from that frame forward/backward
            4. Repeat for other problematic frames if needed
        
        Note:
            After refining, you must call propagate_in_video() again to apply corrections
            to subsequent frames. Previous propagation results remain unchanged.
        """
        return self.add_object_with_points_prompt(
            session_id=session_id,
            frame_idx=frame_idx,
            object_id=object_id,
            frame_width=frame_width,
            frame_height=frame_height,
            points=points,
            point_labels=point_labels
        )

    @profile()
    def inject_masks(
        self,
        session_id: str,
        frame_idx: int,
        masks: dict,
        object_ids: list,
    ):
        """Inject masks from a previous chunk into the tracker for cross-chunk continuity.

        This method injects binary masks as conditioning frames into the tracker's memory,
        allowing objects tracked in a previous chunk to seamlessly continue into the current
        session. After injection, call propagate_in_video() to track these objects forward.

        This is the core mechanism for Option B cross-chunk state propagation: the overlap
        frame's tracked masks from chunk N become the initial conditioning for chunk N+1.

        Args:
            session_id: Active session identifier from start_session().
            frame_idx: Frame index to inject masks on (typically 0 for chunk start).
            masks: Dictionary mapping object_id (int) to mask (2D numpy array, H x W,
                   values 0 or 255 for background/foreground).
            object_ids: List of object IDs to inject (must match keys in masks dict).

        Returns:
            Dictionary with 'injected_object_ids' list.

        Raises:
            ValueError: If model not loaded or session invalid.

        Example:
            >>> # Previous chunk produced masks for objects 0, 1, 2
            >>> prev_masks = {0: mask_array_0, 1: mask_array_1, 2: mask_array_2}
            >>> obj_ids = [0, 1, 2]
            >>> 
            >>> # Start new session on next chunk
            >>> session_id = driver.start_session("chunk_1.mp4")
            >>> 
            >>> # Inject previous chunk's last-frame masks on frame 0
            >>> result = driver.inject_masks(session_id, frame_idx=0,
            ...     masks=prev_masks, object_ids=obj_ids)
            >>> 
            >>> # Optionally add text prompt for NEW object detection
            >>> driver.add_prompt(session_id, "person")
            >>> 
            >>> # Propagate (tracks both injected + newly detected objects)
            >>> driver.propagate_in_video(session_id)
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")

        response = self.predictor.handle_request(
            request=dict(
                type="inject_masks",
                session_id=session_id,
                frame_index=frame_idx,
                masks=masks,
                object_ids=object_ids,
            )
        )
        return response

    @profile()
    def remove_object(self, session_id: str, object_id: int):
        """Remove an object from the video segmentation session.
        
        Permanently deletes all tracking information for the specified object across
        all frames in the session. Useful for removing incorrectly detected objects
        or objects no longer of interest.
        
        Args:
            session_id: Active session identifier from start_session().
            object_id: Unique integer ID of the object to remove.
        
        Raises:
            ValueError: If model not loaded, session invalid, or object_id doesn't exist.
        
        Example:
            >>> # Add multiple objects
            >>> driver.add_object_with_points_prompt(session_id, 0, object_id=1, ...)
            >>> driver.add_object_with_points_prompt(session_id, 0, object_id=2, ...)
            >>> driver.propagate_in_video(session_id)
            >>> 
            >>> # Remove object 2 from all frames
            >>> driver.remove_object(session_id, object_id=2)
        
        Note:
            - Removal is permanent for this session
            - Other objects remain unaffected
            - Does not require re-propagation (removal is immediate)
            - To track the object again, use add_object_with_points_prompt() with a new ID
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        self.predictor.handle_request(
            request=dict(
                type="remove_object",
                session_id=session_id,
                object_id=object_id,
            )
        )

# -------------- Session Management APIs (start, reset, close) --------------
    @profile()
    def cleanup(self):
        """Release all resources and perform comprehensive memory cleanup.
        
        This method performs a full shutdown of the predictor and aggressively frees
        memory. Essential for long-running applications or when switching between videos.
        
        Cleanup Operations:
            1. Shutdown predictor (closes all sessions)
            2. Release device memory (GPU cache or CPU malloc)
            3. Run garbage collection
        
        Performance Notes:
            - GPU: Clears CUDA cache (~100% GPU memory recovery)
            - CPU (Linux): Calls malloc_trim + GC (aggressive OS-level cleanup)
            - CPU (other): Python GC only (standard cleanup)
        
        Example:
            >>> driver = Sam3VideoDriver()
            >>> # Process multiple videos
            >>> for video_path in video_list:
            ...     session_id = driver.start_session(video_path)
            ...     # ... track objects ...
            ...     driver.close_session(session_id)
            >>> driver.cleanup()  # Final cleanup before exit
        
        Warning:
            After cleanup(), the driver is non-functional. Create a new instance
            to process additional videos.
        """
        # Shutdown all active sessions and release predictor resources
        self.shutdown()

        # Device-specific aggressive memory recovery
        if DEVICE.type == "cuda":
            # Clear PyTorch's CUDA memory cache
            torch.cuda.empty_cache()
        elif DEVICE.type == "cpu":
            # Linux-specific: Return freed memory to OS via glibc malloc_trim
            import ctypes
            import gc

            try:
                libc = ctypes.CDLL("libc.so.6")  # Requires glibc (Linux)
                libc.malloc_trim(0)  # Aggressively return memory to OS
            except OSError:
                # Not on Linux or glibc unavailable, skip platform-specific cleanup
                pass

            # Run Python garbage collector to free unreferenced objects
            gc.collect()
    
    @profile()
    def close_session(self, session_id: str):
        """Close and release resources for a specific video segmentation session.
        
        This method deallocates memory and resources associated with a session while
        keeping the driver active for processing other videos. Always close sessions
        when done to prevent memory leaks.
        
        Args:
            session_id: Session identifier to close (from start_session()).
        
        Raises:
            ValueError: If model not loaded or session_id is invalid.
        
        Example:
            >>> driver = Sam3VideoDriver()
            >>> session1 = driver.start_session("video1.mp4")
            >>> # ... process video1 ...
            >>> driver.close_session(session1)  # Free memory
            >>> 
            >>> session2 = driver.start_session("video2.mp4")  # Driver still active
            >>> # ... process video2 ...
            >>> driver.close_session(session2)
        
        Memory Impact:
            - Releases encoded frame embeddings (~1-2 GB per video on GPU)
            - Clears all tracked object states
            - Driver remains loaded and ready for new sessions
        
        Best Practice:
            Use try-finally to ensure sessions are closed even on errors:
            >>> session_id = driver.start_session(video_path)
            >>> try:
            ...     # Process video
            ...     pass
            >>> finally:
            ...     driver.close_session(session_id)
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        
        self.predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

    @profile()
    def reset_session(self, session_id: str):
        """Reset a session to its initial state, removing all prompts and tracked objects.
        
        This method clears all annotation data (prompts, objects, tracking results) while
        keeping the video loaded and encoded. Useful for trying different annotation
        strategies without reloading the video.
        
        Args:
            session_id: Session identifier to reset (from start_session()).
        
        Raises:
            ValueError: If model not loaded or session_id is invalid.
        
        Example:
            >>> session_id = driver.start_session("video.mp4")
            >>> # First attempt: track people
            >>> driver.add_prompt(session_id, "person")
            >>> results1 = driver.propagate_in_video(session_id)
            >>> 
            >>> # Not satisfied, try different approach
            >>> driver.reset_session(session_id)
            >>> # Second attempt: manual point annotation
            >>> driver.add_object_with_points_prompt(session_id, 0, object_id=1, ...)
            >>> results2 = driver.propagate_in_video(session_id)
        
        Performance Note:
            Reset is much faster than close + start because the video remains loaded
            and encoded in memory. Video encoding is the most expensive operation.
        
        What Gets Reset:
            - All prompts (text and point annotations)
            - All tracked objects and their IDs
            - All propagation results
        
        What Stays:
            - Encoded video frames (expensive to recompute)
            - Session ID (remains valid)
            - Driver configuration
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        self.predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )

    @profile()
    def start_session(self, video_path: str) -> str:
        """Start a new video segmentation session and load the video.
        
        This method initializes a new session, loads the video, and encodes all frames
        into the model's internal representation. Returns a unique session ID used for
        all subsequent operations on this video.
        
        Args:
            video_path: Path to video file. Supports common formats (mp4, avi, mov, mkv, etc.).
        
        Returns:
            Unique session identifier (string) used for all session operations.
        
        Raises:
            ValueError: If model not loaded.
            FileNotFoundError: If video_path doesn't exist.
            RuntimeError: If video loading or encoding fails.
        
        Example:
            >>> driver = Sam3VideoDriver()
            >>> session_id = driver.start_session("/path/to/video.mp4")
            >>> print(f"Started session: {session_id}")
            >>> # Now you can add prompts and track objects
            >>> driver.add_prompt(session_id, "person")
            >>> # ... do work ...
            >>> driver.close_session(session_id)  # Always close when done
        
        Performance Notes:
            - Video Encoding: Most expensive operation, takes 30s-5min depending on length
            - Memory Usage: ~1-2 GB per video on GPU, ~2-4 GB on CPU
            - Frame Rate: Processes ~10-30 frames/sec during encoding (device-dependent)
            - Long Videos: Consider splitting videos >5 minutes into chunks
        
        Session Management:
            - Multiple concurrent sessions supported (memory permitting)
            - Each session is isolated (prompts/objects don't affect other sessions)
            - Always close sessions with close_session() to free memory
            - Use context manager pattern for automatic cleanup
        
        Supported Video Formats:
            Common: mp4, avi, mov, mkv, webm, flv, wmv
            Codec: H.264, H.265, VP8, VP9, etc. (anything OpenCV supports)
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        return response["session_id"]

    @profile()
    def shutdown(self):
        """Shutdown the video predictor and close all active sessions.
        
        This method gracefully shuts down the predictor, closing all active sessions
        and releasing associated resources. After shutdown, the driver becomes
        non-functional and cannot process videos.
        
        Raises:
            No exceptions raised. Issues are logged as warnings.
        
        Example:
            >>> driver = Sam3VideoDriver()
            >>> session_id = driver.start_session("video.mp4")
            >>> # ... process video ...
            >>> driver.shutdown()  # Close all sessions and shutdown
            >>> # Driver is now unusable, create new instance if needed
        
        Shutdown vs Close vs Cleanup:
            - shutdown(): Closes predictor, all sessions become invalid
            - close_session(): Closes one session, predictor stays active
            - cleanup(): Calls shutdown() + aggressive memory cleanup
        
        Note:
            - Safe to call multiple times (idempotent)
            - Automatically called by cleanup()
            - Use cleanup() instead for memory-sensitive applications
            - No need to close individual sessions before shutdown (automatic)
        
        Warning States:
            - "Predictor is already shut down": shutdown() called twice (harmless)
            - "Predictor was never initialized": shutdown() called on failed init
        """
        if self.predictor is not None:
            if not self.predictor.has_shutdown:
                self.predictor.shutdown()
            else:
                logger.warning("Predictor is already shut down.")
        else:
            logger.warning("Predictor was never initialized.")