"""
SAM3 Entrypoint Module

High-level API for SAM3 image and video segmentation with automatic memory management,
chunking, and output generation. This module provides a unified interface for all
SAM3 segmentation tasks with intelligent resource handling.

Supported Input Scenarios:
    (a) Single image + one or more text prompts
    (b) Single image + one or multiple bounding boxes
    (c) Multiple images + one or more text prompts
    (d) Multiple images + one or multiple bounding boxes
    (e) Video + one or more text prompts
    (f) Video + point prompts (clicks) with labels
    (g) Video + refine existing objects via point prompts
    (h) Video + remove objects by their IDs
    (i) Video + segment dict with frame/time ranges and prompts/points
"""

import os
import subprocess
import threading
import shutil
import cv2  # For video operations
import re  # For regex pattern matching
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
from PIL import Image  # For image I/O

from sam3.__globals import (
    DEVICE,
    BPE_PATH,
    DEFAULT_MIN_VIDEO_FRAMES,
    DEFAULT_MIN_CHUNK_OVERLAP,
    IMAGE_INFERENCE_MB,
    VIDEO_INFERENCE_MB,
    TEMP_DIR,
    DEFAULT_OUTPUT_DIR,
    RAM_USAGE_PERCENT,
    VRAM_USAGE_PERCENT,
    MEMORY_SAFETY_MULTIPLIER,
    CPU_MEMORY_RESERVE_PERCENT,
    GPU_MEMORY_RESERVE_PERCENT,
    DEFAULT_PROPAGATION_DIRECTION,
    DEFAULT_NUM_WORKERS
)
from sam3.logger import get_logger
from sam3.drivers import Sam3ImageDriver, Sam3VideoDriver

logger = get_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MemoryInfo:
    """Memory/VRAM information for feasibility checking."""
    device_type: str  # "cpu" or "cuda"
    total_gb: float
    available_gb: float
    utilization_percent: float
    

@dataclass
class VideoMetadata:
    """Video file metadata."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    bytes_per_frame: int


@dataclass
class ChunkInfo:
    """Video chunk information."""
    chunk_id: int
    start_frame: int
    end_frame: int
    num_frames: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    chunk_path: Optional[str] = None


@dataclass
class FeasibilityResult:
    """Result of feasibility check."""
    can_process: bool
    memory_needed_gb: float
    memory_available_gb: float
    requires_chunking: bool
    chunk_size: Optional[int] = None
    num_chunks: Optional[int] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    output_dir: str
    mask_files: List[str]
    object_ids: List[int]
    metadata: Dict[str, Any]
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# ============================================================================
# Sam3Entrypoint Class
# ============================================================================

class Sam3Entrypoint:
    """
    High-level entrypoint for SAM3 segmentation operations.
    
    This class provides intelligent resource management, automatic chunking for videos,
    and unified processing for images and videos with various prompt types.
    
    Features:
        - Automatic memory feasibility checking
        - Dynamic video chunking based on available memory
        - Parallel chunk preparation (90% threshold)
        - Session-based video processing
        - Mask video generation (one per object ID)
        - Overlap handling in chunked videos
    
    Example:
        >>> entrypoint = Sam3Entrypoint()
        >>> # Process image with text prompts
        >>> result = entrypoint.process_image(
        ...     image_path="image.jpg",
        ...     prompts=["person", "car"],
        ...     output_dir="./outputs"
        ... )
        >>> print(f"Masks saved to: {result.output_dir}")
    """
    
    def __init__(
        self,
        bpe_path: Optional[str] = None,
        num_workers: Optional[int] = None,
        ram_usage_percent: float = None,
        vram_usage_percent: float = None,
        min_video_frames: int = None,
        min_chunk_overlap: int = None,
        temp_dir: str = None,
        default_output_dir: str = None,
        verbose: bool = True
    ):
        """
        Initialize SAM3 Entrypoint.
        
        Args:
            bpe_path: Path to BPE tokenizer file. Defaults to global BPE_PATH.
            num_workers: Number of worker threads (CPU) or GPUs to use. Auto-detects if None.
            ram_usage_percent: Fraction of available RAM to use for CPU chunking (default: 0.33).
            vram_usage_percent: Fraction of available VRAM to use for GPU chunking (default: 0.90).
            min_video_frames: Minimum frames per chunk (default: from __globals).
            min_chunk_overlap: Minimum frame overlap between chunks (default: from __globals).
            temp_dir: Temporary directory for intermediate files (default: from __globals).
            default_output_dir: Default output directory if user doesn't specify (default: from __globals).
            verbose: Print detailed logs (default: True).
        """
        self.bpe_path = bpe_path or BPE_PATH
        self.num_workers = num_workers or DEFAULT_NUM_WORKERS
        self.ram_usage_percent = ram_usage_percent if ram_usage_percent is not None else RAM_USAGE_PERCENT
        self.vram_usage_percent = vram_usage_percent if vram_usage_percent is not None else VRAM_USAGE_PERCENT
        self.min_video_frames = min_video_frames if min_video_frames is not None else DEFAULT_MIN_VIDEO_FRAMES
        self.min_chunk_overlap = min_chunk_overlap if min_chunk_overlap is not None else DEFAULT_MIN_CHUNK_OVERLAP
        self.temp_dir = Path(temp_dir) if temp_dir else Path(TEMP_DIR)
        self.default_output_dir = Path(default_output_dir) if default_output_dir else Path(DEFAULT_OUTPUT_DIR)
        self.verbose = verbose
        
        # Create directories
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect memory
        self.memory_info = self._get_memory_info()
        
        # Lazy-loaded drivers
        self._image_driver: Optional[Sam3ImageDriver] = None
        self._video_driver: Optional[Sam3VideoDriver] = None
        
        if self.verbose:
            self._print_initialization_info()
    
    # ========================================================================
    # Time/Frame Parsing Methods
    # ========================================================================
    
    @staticmethod
    def _parse_time_or_frame(value: Union[str, int, float], fps: Optional[float] = None) -> dict:
        """Parse time/frame input with auto-detection.
        
        Supports:
        - Frame number: 100 (int)
        - Seconds: 45.5 or "45.5" (float)
        - Timestamp: "00:01:30" or "1:30" (HH:MM:SS or MM:SS)
        
        Args:
            value: Input value (frame number, seconds, or timestamp string)
            fps: Video FPS (required to convert time to frames)
        
        Returns:
            dict with 'frame' (int) and 'seconds' (float)
        """
        if isinstance(value, int):
            # Frame number
            if fps is not None:
                return {'frame': value, 'seconds': value / fps}
            return {'frame': value, 'seconds': None}
        
        if isinstance(value, float):
            # Seconds as float
            if fps is not None:
                return {'frame': int(value * fps), 'seconds': value}
            return {'frame': None, 'seconds': value}
        
        if isinstance(value, str):
            # Check if it's a timestamp (HH:MM:SS or MM:SS)
            timestamp_pattern = r'^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)$'
            match = re.match(timestamp_pattern, value)
            
            if match:
                # Parse timestamp
                hours = int(match.group(1)) if match.group(1) else 0
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                
                total_seconds = hours * 3600 + minutes * 60 + seconds
                
                if fps is not None:
                    return {'frame': int(total_seconds * fps), 'seconds': total_seconds}
                return {'frame': None, 'seconds': total_seconds}
            else:
                # Try parsing as seconds (string format)
                try:
                    seconds = float(value)
                    if fps is not None:
                        return {'frame': int(seconds * fps), 'seconds': seconds}
                    return {'frame': None, 'seconds': seconds}
                except ValueError:
                    raise ValueError(
                        f"Invalid time/frame format: '{value}'. "
                        f"Expected: frame number (int), seconds (float), or timestamp (HH:MM:SS or MM:SS)"
                    )
        
        raise ValueError(f"Unsupported value type: {type(value)}")
    
    # ========================================================================
    # Memory & Feasibility Methods
    # ========================================================================
    
    def _get_memory_info(self) -> MemoryInfo:
        """Get current memory/VRAM information."""
        if torch.cuda.is_available():
            # GPU/CUDA memory detection
            device_type = "cuda"
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            available_memory = total_memory - reserved
            utilization = (reserved / total_memory) * 100
            
            return MemoryInfo(
                device_type=device_type,
                total_gb=total_memory,
                available_gb=available_memory,
                utilization_percent=utilization
            )
        else:
            # CPU/RAM detection (Linux)
            device_type = "cpu"
            try:
                # Read from /proc/meminfo (Linux)
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                
                mem_info = {}
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = int(parts[1].strip().split()[0])  # Value in KB
                        mem_info[key] = value
                
                mem_total = mem_info['MemTotal'] / (1024**2)  # Convert KB to GB
                mem_available = mem_info['MemAvailable'] / (1024**2)  # Convert KB to GB
                utilization = ((mem_total - mem_available) / mem_total) * 100
                
                return MemoryInfo(
                    device_type=device_type,
                    total_gb=mem_total,
                    available_gb=mem_available,
                    utilization_percent=utilization
                )
                
            except (FileNotFoundError, KeyError, IndexError) as e:
                # Fallback for non-Linux systems or errors
                logger.warning(f"Could not read /proc/meminfo: {e}. Using fallback values.")
                return MemoryInfo(
                    device_type=device_type,
                    total_gb=8.0,  # Fallback: assume 8GB
                    available_gb=4.0,  # Fallback: assume 4GB available
                    utilization_percent=50.0
                )
    
    @property
    def memory_usage_percent(self) -> float:
        """Get the appropriate memory usage percentage based on device type.
        
        Returns:
            RAM usage percent for CPU, VRAM usage percent for GPU
        """
        if self.memory_info.device_type == "cuda":
            return self.vram_usage_percent
        else:
            return self.ram_usage_percent
    
    def _calculate_memory_needed(
        self, 
        width: int, 
        height: int, 
        num_frames: int = 1,
        is_video: bool = False
    ) -> float:
        """Calculate memory needed in GB for processing.
        
        Note: We calculate based on ACTUAL video resolution, not model size (1008x1008).
        Even though SAM3 resizes to 1008x1008, the memory footprint during loading
        is determined by the original resolution frames being decoded and processed.
        """
        # Use actual video resolution (not model size)
        # Video frames are decoded at original resolution before being resized
        # Memory estimate: RGB, 3 channels, 3 bytes per channel for uint8 (video decoding)
        bytes_per_frame = width * height * 3  # uint8 = 1 byte per channel
        
        # Total frame data in GB
        frame_data_gb = (bytes_per_frame * num_frames) / (1024**3)
        
        # Add model inference overhead
        if is_video:
            inference_overhead_gb = VIDEO_INFERENCE_MB / 1024  # Convert MB to GB
        else:
            inference_overhead_gb = IMAGE_INFERENCE_MB / 1024  # Convert MB to GB
        
        # Total memory needed
        total_memory_gb = frame_data_gb + inference_overhead_gb
        
        if self.verbose:
            logger.debug(f"Memory calculation:")
            logger.debug(f"  Video resolution: {width}x{height}")
            logger.debug(f"  Bytes per frame: {bytes_per_frame / 1024**2:.2f} MB (uint8 RGB)")
            logger.debug(f"  Frame data: {frame_data_gb:.2f} GB ({num_frames} frames)")
            logger.debug(f"  Inference overhead: {inference_overhead_gb:.2f} GB")
            logger.debug(f"  Total needed: {total_memory_gb:.2f} GB")
        
        return total_memory_gb
    
    def _check_feasibility(
        self,
        memory_needed_gb: float,
        total_frames: int = 1,
        is_video: bool = False
    ) -> FeasibilityResult:
        """
        Check if processing is feasible with current memory.
        
        Logic:
            1. Check if 3x memory_needed <= available memory (ideal)
            2. If not, check if 3x memory_needed <= total memory * threshold
               - CPU: threshold = 0.7 (30% for OS)
               - GPU: threshold = 0.95 (5% for display)
            3. If not enough, video requires chunking
        
        Returns FeasibilityResult with can_process, requires_chunking, etc.
        """
        warnings = []
        
        # Safety margin: need 3x the estimated memory
        required_memory_gb = memory_needed_gb * MEMORY_SAFETY_MULTIPLIER
        
        # Check against available memory first (ideal case)
        enough_available = required_memory_gb <= self.memory_info.available_gb
        
        # Calculate usable total memory based on device type
        if self.memory_info.device_type == "cuda":
            usable_total_gb = self.memory_info.total_gb * (1 - GPU_MEMORY_RESERVE_PERCENT)
        else:
            usable_total_gb = self.memory_info.total_gb * (1 - CPU_MEMORY_RESERVE_PERCENT)
        
        enough_total = required_memory_gb <= usable_total_gb
        
        # Decision logic
        if enough_available:
            # Ideal case: enough available memory right now
            can_process = True
            requires_chunking = False
            chunk_size = None
            num_chunks = None
            
            if self.verbose:
                logger.info(f"✓ Memory check passed: {required_memory_gb:.2f} GB needed, "
                          f"{self.memory_info.available_gb:.2f} GB available")
        
        elif enough_total:
            # Available memory insufficient, but total memory could handle it
            # User needs to free up memory
            can_process = False  # Will prompt user
            requires_chunking = False
            chunk_size = None
            num_chunks = None
            
            warnings.append(
                f"Insufficient available memory: {required_memory_gb:.2f} GB needed, "
                f"only {self.memory_info.available_gb:.2f} GB available."
            )
            warnings.append(
                f"However, total usable memory ({usable_total_gb:.2f} GB) is sufficient. "
                f"Please close other programs to free up memory."
            )
            
            if self.verbose:
                logger.warning(f"⚠ Insufficient available memory")
                logger.warning(f"  Required: {required_memory_gb:.2f} GB")
                logger.warning(f"  Available: {self.memory_info.available_gb:.2f} GB")
                logger.warning(f"  Total usable: {usable_total_gb:.2f} GB")
        
        else:
            # Not enough total memory - need chunking for videos
            if is_video:
                can_process = True
                requires_chunking = True
                
                # Calculate chunk size based on available memory
                # Use appropriate memory usage percent based on device (RAM for CPU, VRAM for GPU)
                usable_memory_gb = self.memory_info.available_gb * self.memory_usage_percent
                
                # Subtract inference overhead
                inference_overhead_gb = VIDEO_INFERENCE_MB / 1024
                memory_for_frames_gb = max(0.1, usable_memory_gb - inference_overhead_gb)
                
                # Calculate bytes per frame (estimate based on memory_needed and total_frames)
                bytes_per_frame = (memory_needed_gb * (1024**3)) / total_frames if total_frames > 0 else 0
                
                # Calculate max frames that fit in memory
                if bytes_per_frame > 0:
                    max_frames = int((memory_for_frames_gb * (1024**3)) / bytes_per_frame)
                    chunk_size = max(self.min_video_frames, max_frames)
                    num_chunks = (total_frames + chunk_size - 1) // chunk_size  # Ceiling division
                else:
                    chunk_size = self.min_video_frames
                    num_chunks = (total_frames + chunk_size - 1) // chunk_size
                
                warnings.append(
                    f"Video too large for available memory ({required_memory_gb:.2f} GB needed, "
                    f"{self.memory_info.available_gb:.2f} GB available)."
                )
                warnings.append(
                    f"Processing will use chunking strategy: {num_chunks} chunks of ~{chunk_size} frames each."
                )
                
                if self.verbose:
                    logger.info(f"📊 Chunking will be required (detailed info shown during processing)")
            else:
                # Image too large and no chunking option
                can_process = False
                requires_chunking = False
                chunk_size = None
                num_chunks = None
                
                warnings.append(
                    f"Image too large to process: {required_memory_gb:.2f} GB needed, "
                    f"only {usable_total_gb:.2f} GB total usable memory."
                )
                warnings.append("Consider reducing image resolution.")
                
                if self.verbose:
                    logger.error(f"✗ Image too large to process")
                    logger.error(f"  Required: {required_memory_gb:.2f} GB")
                    logger.error(f"  Usable total: {usable_total_gb:.2f} GB")
        
        return FeasibilityResult(
            can_process=can_process,
            memory_needed_gb=memory_needed_gb,
            memory_available_gb=self.memory_info.available_gb,
            requires_chunking=requires_chunking,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            warnings=warnings
        )
    
    def _prompt_user_for_memory(self, feasibility: FeasibilityResult) -> bool:
        """
        Prompt user to free up memory if needed.
        
        Returns True if user wants to continue, False otherwise.
        """
        print("\n" + "=" * 70)
        print("MEMORY INSUFFICIENT WARNING")
        print("=" * 70)
        
        for warning in feasibility.warnings:
            print(f"⚠  {warning}")
        
        print(f"\nMemory Status:")
        print(f"  Required (with 3x safety):  {feasibility.memory_needed_gb * 3:.2f} GB")
        print(f"  Currently available:         {feasibility.memory_available_gb:.2f} GB")
        print(f"  Device:                      {self.memory_info.device_type.upper()}")
        
        if feasibility.requires_chunking:
            print(f"\n✓ Video will be processed in {feasibility.num_chunks} chunks")
            print(f"  (~{feasibility.chunk_size} frames per chunk)")
            return True  # Auto-continue with chunking
        
        print("\nRecommendation: Close other programs to free up memory, then try again.")
        print("=" * 70)
        
        # Ask user if they want to continue anyway
        while True:
            response = input("\nDo you want to continue anyway? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                logger.warning("User chose to continue despite insufficient memory")
                return True
            elif response in ['n', 'no']:
                logger.info("User chose not to continue due to insufficient memory")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    # ========================================================================
    # Video Metadata & Chunking Methods
    # ========================================================================
    
    def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using ffprobe."""
        try:
            # Get width
            width_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width', '-of', 'csv=p=0', video_path
            ]
            width = int(subprocess.check_output(width_cmd).decode().strip())
            
            # Get height
            height_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=height', '-of', 'csv=p=0', video_path
            ]
            height = int(subprocess.check_output(height_cmd).decode().strip())
            
            # Get FPS
            fps_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path
            ]
            fps_raw = subprocess.check_output(fps_cmd).decode().strip()
            num, denom = map(int, fps_raw.split('/'))
            fps = num / denom
            
            # Get duration
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'csv=p=0', video_path
            ]
            duration = float(subprocess.check_output(duration_cmd).decode().strip())
            
            total_frames = int(fps * duration)
            bytes_per_frame = width * height * 3  # CV_8UC3 format (RGB)
            
            if self.verbose:
                logger.info(f"Video metadata extracted:")
                logger.info(f"  Resolution: {width}x{height}")
                logger.info(f"  FPS: {fps:.2f}")
                logger.info(f"  Duration: {duration:.2f}s")
                logger.info(f"  Total frames: {total_frames}")
            
            return VideoMetadata(
                path=video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                bytes_per_frame=bytes_per_frame
            )
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract video metadata: {e}")
        except (ValueError, ZeroDivisionError) as e:
            raise RuntimeError(f"Invalid video metadata values: {e}")
    
    def _calculate_chunk_size(self, video_meta: VideoMetadata) -> int:
        """Calculate optimal chunk size based on available memory."""
        # Use appropriate memory usage percent based on device (RAM for CPU, VRAM for GPU)
        usable_memory_gb = self.memory_info.available_gb * self.memory_usage_percent
        
        # Subtract inference overhead
        inference_overhead_gb = VIDEO_INFERENCE_MB / 1024
        memory_for_frames_gb = max(0.1, usable_memory_gb - inference_overhead_gb)
        
        # Calculate max frames that fit in memory
        memory_for_frames_bytes = memory_for_frames_gb * (1024**3)
        max_frames = int(memory_for_frames_bytes / video_meta.bytes_per_frame)
        
        # Ensure minimum frames
        chunk_size = max(self.min_video_frames, max_frames)
        
        if self.verbose:
            logger.info(f"Calculated chunk size: {chunk_size} frames")
            logger.info(f"  Usable memory: {usable_memory_gb:.2f} GB")
            logger.info(f"  Memory for frames: {memory_for_frames_gb:.2f} GB")
        
        return chunk_size
    
    def _generate_chunks(
        self, 
        video_meta: VideoMetadata, 
        chunk_size: int,
        overlap: int
    ) -> List[ChunkInfo]:
        """
        Generate chunk information with overlap.
        
        Overlap logic: If chunk1 is frames 0-24, chunk2 is frames 24-49.
        The last frame of one chunk is the first frame of the next.
        """
        chunks = []
        stride = chunk_size - overlap
        
        if stride <= 0:
            raise ValueError(
                f"Overlap ({overlap}) must be smaller than chunk size ({chunk_size})"
            )
        
        chunk_id = 0
        start_frame = 0
        
        while start_frame < video_meta.total_frames:
            # Calculate end frame (inclusive)
            end_frame = min(start_frame + chunk_size - 1, video_meta.total_frames - 1)
            num_frames = end_frame - start_frame + 1
            
            # Calculate time boundaries
            start_time_sec = start_frame / video_meta.fps
            end_time_sec = (end_frame + 1) / video_meta.fps
            duration_sec = end_time_sec - start_time_sec
            
            chunk = ChunkInfo(
                chunk_id=chunk_id,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                duration_sec=duration_sec
            )
            chunks.append(chunk)
            
            if self.verbose:
                logger.debug(
                    f"Chunk {chunk_id}: frames {start_frame}-{end_frame} "
                    f"({num_frames} frames, {duration_sec:.2f}s)"
                )
            
            # Move to next chunk with overlap
            # Next chunk starts at current_end_frame (inclusive for overlap)
            start_frame = end_frame  # This creates 1-frame overlap by default
            
            # If we've reached the end, break
            if end_frame >= video_meta.total_frames - 1:
                break
            
            chunk_id += 1
        
        if self.verbose:
            logger.info(f"Generated {len(chunks)} chunks with {overlap}-frame overlap")
        
        return chunks
    
    def _extract_video_chunk(
        self,
        video_path: str,
        chunk_info: ChunkInfo,
        output_path: str
    ) -> str:
        """Extract a video chunk using ffmpeg."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Use ffmpeg to extract chunk by time
            # Note: Using lossless H.264 (qp=0) for pixel-perfect chunks
            # This ensures overlap frames are identical for accurate mask matching
            cmd = [
                'ffmpeg',
                '-loglevel', 'error',  # Only show errors
                '-y',  # Overwrite output file
                '-ss', str(chunk_info.start_time_sec),  # Start time (before -i for speed)
                '-i', video_path,
                '-t', str(chunk_info.duration_sec),  # Duration
                '-c:v', 'libx264',  # H.264 codec
                '-qp', '0',  # Lossless mode (QP=0)
                '-preset', 'veryfast',  # Speed up encoding
                '-c:a', 'copy',  # Copy audio (if any)
                output_path
            ]
            
            if self.verbose:
                logger.debug(f"Extracting chunk {chunk_info.chunk_id} to {output_path}")
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            if not os.path.exists(output_path):
                raise RuntimeError(f"Failed to create chunk file: {output_path}")
            
            return output_path
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to extract chunk {chunk_info.chunk_id}: {e}\n"
                f"stderr: {e.stderr.decode() if e.stderr else 'none'}"
            )
    
    # ========================================================================
    # Input Validation Methods
    # ========================================================================
    
    def _validate_image_input(
        self,
        image_path: Union[str, List[str]],
        prompts: Optional[Union[str, List[str]]] = None,
        boxes: Optional[List[List[float]]] = None
    ) -> Tuple[List[str], Optional[List[str]], Optional[List[List[float]]]]:
        """Validate and normalize image input."""
        # Normalize image paths to list
        if isinstance(image_path, str):
            image_paths = [image_path]
        else:
            image_paths = list(image_path)
        
        # Validate all image files exist
        for path in image_paths:
            if not os.path.isfile(path):
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
        video_path: str,
        prompts: Optional[Union[str, List[str]]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        object_ids: Optional[List[int]] = None,
        segments: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Validate and normalize video input."""
        # Validate video file exists
        if not os.path.isfile(video_path):
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
    # Image Processing Methods (Scenarios a-d)
    # ========================================================================
    
    def process_image(
        self,
        image_path: Union[str, List[str]],
        prompts: Optional[Union[str, List[str]]] = None,
        boxes: Optional[Union[List[float], List[List[float]]]] = None,
        box_labels: Optional[List[int]] = None,
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """Process single or multiple images with prompts or bounding boxes.
        
        Handles scenarios (a), (b), (c), (d).
        
        Args:
            image_path: Path to image file or list of paths.
            prompts: Text prompt(s) for segmentation (scenarios a, c).
            boxes: Bounding box(es) in XYWH format (scenarios b, d).
            box_labels: Labels for boxes (1=positive, 0=negative). Optional.
            output_dir: Output directory. Uses default if None.
        
        Returns:
            ProcessingResult with masks, object IDs, and metadata.
        """
        # Validate input
        image_paths, prompts, boxes = self._validate_image_input(image_path, prompts, boxes)
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        # Load image driver
        driver = self._get_image_driver()
        
        # Process each image
        all_mask_files = []
        all_object_ids = []
        all_metadata = {}
        all_errors = []
        
        for img_path in image_paths:
            try:
                # Get image name for output directory
                img_name = Path(img_path).stem
                
                # Load image to check dimensions
                img = Image.open(img_path)
                width, height = img.size
                
                # Check feasibility
                memory_needed = self._calculate_memory_needed(width, height, num_frames=1, is_video=False)
                feasibility = self._check_feasibility(memory_needed, total_frames=1, is_video=False)
                
                if not feasibility.can_process:
                    # Prompt user if memory insufficient
                    if not self._prompt_user_for_memory(feasibility):
                        all_errors.append(f"Skipped {img_path}: Insufficient memory")
                        continue
                
                if self.verbose:
                    logger.info(f"Processing image: {img_path}")
                
                # Create output directory for this image
                img_output_dir = self._create_output_directory(output_dir, img_name)
                
                # Process with prompts (scenarios a, c)
                if prompts is not None:
                    results = driver.prompt_texts(img_path, prompts=prompts)
                    
                    # Extract masks and object IDs from results
                    for prompt, state in results.items():
                        # Debug logging to see state structure
                        if self.verbose:
                            logger.info(f"  Prompt '{prompt}': state keys = {list(state.keys())}")
                            if "masks" in state:
                                masks_info = state['masks']
                                logger.info(f"  masks type: {type(masks_info)}")
                                if hasattr(masks_info, 'shape'):
                                    logger.info(f"  masks shape: {masks_info.shape}")
                                elif hasattr(masks_info, '__len__'):
                                    logger.info(f"  masks len: {len(masks_info)}")
                            if "scores" in state:
                                logger.info(f"  scores type: {type(state['scores'])}")
                                if hasattr(state['scores'], '__len__'):
                                    logger.info(f"  scores len: {len(state['scores'])}")
                        
                        if "masks" in state and len(state["masks"]) > 0:
                            # Convert torch tensor to numpy if needed
                            masks = state["masks"]
                            if hasattr(masks, 'cpu'):  # It's a torch tensor
                                masks = masks.cpu().numpy()
                            else:
                                masks = np.array(masks)
                            
                            scores = state.get("scores", [])
                            
                            # Generate object IDs (simple incremental for now)
                            obj_ids = list(range(len(masks)))
                            
                            # Save masks with format [prompt]_[ObjectId].png
                            saved_files = self._save_image_masks(
                                masks, obj_ids, img_output_dir, prompt
                            )
                            all_mask_files.extend(saved_files)
                            all_object_ids.extend(obj_ids)
                            
                            if self.verbose:
                                logger.info(f"  Prompt '{prompt}': Saved {len(saved_files)} mask files")
                
                # Process with boxes (scenarios b, d)
                elif boxes is not None:
                    # Initialize processor
                    processor, inference_state = driver.inference(img)
                    
                    if len(boxes) == 1:
                        # Single box
                        state = driver.prompt_bounding_box(
                            img, processor, inference_state, boxes[0]
                        )
                    else:
                        # Multiple boxes
                        if box_labels is None:
                            box_labels = [1] * len(boxes)  # Default all positive
                        
                        state = driver.prompt_multi_box_with_labels(
                            img, processor, inference_state, boxes, box_labels
                        )
                    
                    # Extract and save masks
                    if "masks" in state and len(state["masks"]) > 0:
                        masks = np.array(state["masks"])
                        obj_ids = list(range(len(masks)))
                        
                        saved_files = self._save_image_masks(
                            masks, obj_ids, img_output_dir, "box"
                        )
                        all_mask_files.extend(saved_files)
                        all_object_ids.extend(obj_ids)
                        
                        if self.verbose:
                            logger.info(f"  Found {len(masks)} objects from boxes")
                
                # Store metadata
                all_metadata[img_path] = {
                    "width": width,
                    "height": height,
                    "num_objects": len(all_object_ids),
                    "output_dir": str(img_output_dir)
                }
            
            except Exception as e:
                error_msg = f"Error processing {img_path}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
        
        # Overall success if at least one image processed
        success = len(all_mask_files) > 0
        
        return ProcessingResult(
            success=success,
            output_dir=output_dir,
            mask_files=all_mask_files,
            object_ids=list(set(all_object_ids)),  # Unique object IDs
            metadata=all_metadata,
            errors=all_errors if all_errors else None
        )
    
    # ========================================================================
    # Video Processing Methods (Scenarios e-i)
    # ========================================================================
    
    def process_video_with_prompts(
        self,
        video_path: str,
        prompts: Union[str, List[str]],
        output_dir: Optional[str] = None,
        propagation_direction: str = None,
        frame_from: Union[int, str, float, None] = None,
        frame_to: Union[int, str, float, None] = None
    ) -> ProcessingResult:
        """Process video with text prompts (scenario e).
        
        Args:
            video_path: Path to video file.
            prompts: Text prompt(s) for segmentation.
            output_dir: Output directory. Uses default if None.
            propagation_direction: "forward", "backward", or "both".
            frame_from: Start frame/time (int frame, float seconds, or "HH:MM:SS" timestamp)
            frame_to: End frame/time (int frame, float seconds, or "HH:MM:SS" timestamp)
        
        Returns:
            ProcessingResult with mask videos, object IDs, and metadata.
        """
        if propagation_direction is None:
            propagation_direction = DEFAULT_PROPAGATION_DIRECTION
        
        # Validate input
        validated = self._validate_video_input(video_path, prompts=prompts)
        prompts = validated["prompts"]
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        video_name = Path(video_path).stem
        
        # Handle frame/time range extraction
        original_video_path = video_path
        extracted_video_path = None
        actual_frame_from = 0
        actual_frame_to = video_meta.total_frames - 1
        
        if frame_from is not None or frame_to is not None:
            # Parse start and end times/frames
            if frame_from is not None:
                parsed_from = self._parse_time_or_frame(frame_from, video_meta.fps)
                if parsed_from['frame'] is not None:
                    actual_frame_from = parsed_from['frame']
                    start_time_sec = parsed_from['seconds']
                else:
                    start_time_sec = parsed_from['seconds']
                    actual_frame_from = int(start_time_sec * video_meta.fps)
            else:
                start_time_sec = 0.0
            
            if frame_to is not None:
                parsed_to = self._parse_time_or_frame(frame_to, video_meta.fps)
                if parsed_to['frame'] is not None:
                    actual_frame_to = parsed_to['frame']
                    end_time_sec = parsed_to['seconds']
                else:
                    end_time_sec = parsed_to['seconds']
                    actual_frame_to = int(end_time_sec * video_meta.fps)
            else:
                end_time_sec = video_meta.duration_seconds
            
            # Validate range
            if actual_frame_from >= actual_frame_to:
                raise ValueError(f"frame_from ({actual_frame_from}) must be less than frame_to ({actual_frame_to})")
            
            if actual_frame_from < 0 or actual_frame_to >= video_meta.total_frames:
                raise ValueError(f"Frame range {actual_frame_from}-{actual_frame_to} out of bounds (0-{video_meta.total_frames-1})")
            
            # Extract video segment
            duration_sec = end_time_sec - start_time_sec
            extracted_video_path = str(self.temp_dir / f"{video_name}_range_{actual_frame_from}_{actual_frame_to}.mp4")
            
            if self.verbose:
                logger.info(f"Extracting video range: frames {actual_frame_from}-{actual_frame_to} ({start_time_sec:.2f}s - {end_time_sec:.2f}s)")
            
            try:
                # Use ffmpeg to extract segment
                cmd = [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-y',  # Overwrite output
                    '-i', original_video_path,
                    '-ss', str(start_time_sec),
                    '-t', str(duration_sec),
                    '-c', 'copy',  # Stream copy for speed
                    extracted_video_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                if not os.path.exists(extracted_video_path):
                    raise RuntimeError(f"Failed to extract video segment to {extracted_video_path}")
                
                # Update video path and re-read metadata for the extracted segment
                video_path = extracted_video_path
                video_meta = self._get_video_metadata(video_path)
                
                if self.verbose:
                    logger.info(f"Video segment extracted: {video_meta.total_frames} frames")
            
            except subprocess.CalledProcessError as e:
                error_msg = f"Failed to extract video segment: {e}"
                if e.stderr:
                    error_msg += f"\nstderr: {e.stderr.decode()}"
                raise RuntimeError(error_msg)
        
        # Calculate memory and check feasibility (using potentially extracted segment)
        memory_needed = self._calculate_memory_needed(
            video_meta.width, video_meta.height, video_meta.total_frames, is_video=True
        )
        feasibility = self._check_feasibility(
            memory_needed, video_meta.total_frames, is_video=True
        )
        
        if not feasibility.can_process and not feasibility.requires_chunking:
            if not self._prompt_user_for_memory(feasibility):
                return ProcessingResult(
                    success=False,
                    output_dir="",
                    mask_files=[],
                    object_ids=[],
                    metadata={},
                    errors=["User cancelled due to insufficient memory"]
                )
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        output_base = self._create_output_directory(output_dir, video_name)
        
        try:
            # Process video (with or without chunking)
            if feasibility.requires_chunking:
                # Generate chunks
                chunk_size = feasibility.chunk_size or self.min_video_frames
                chunks = self._generate_chunks(
                    video_meta, chunk_size, self.min_chunk_overlap
                )
                
                if self.verbose:
                    stride = chunk_size - self.min_chunk_overlap
                    device_type = "VRAM" if self.memory_info.device_type == "cuda" else "RAM"
                    usage_percent = self.memory_usage_percent * 100
                    
                    logger.info("=" * 60)
                    logger.info("📊 Video Chunking Details")
                    logger.info("=" * 60)
                    logger.info(f"Video: {video_name}")
                    logger.info(f"Resolution: {video_meta.width}x{video_meta.height}")
                    logger.info(f"FPS: {video_meta.fps:.2f}")
                    logger.info(f"Total Frames: {video_meta.total_frames}")
                    logger.info(f"Available {device_type}: {self.memory_info.available_gb:.2f} GB")
                    logger.info(f"Memory Needed: {memory_needed:.2f} GB")
                    logger.info(f"{device_type} Usage Percent: {usage_percent:.0f}%")
                    logger.info(f"{device_type}-safe Frames per Chunk: {chunk_size}")
                    logger.info(f"Overlap: {self.min_chunk_overlap} frame(s)")
                    logger.info(f"Stride: {stride} frames")
                    logger.info(f"Number of Chunks: {len(chunks)}")
                    logger.info("=" * 60)
                
                # Process in chunks
                results = self._process_video_in_chunks(
                    video_path=video_path,
                    video_meta=video_meta,
                    chunks=chunks,
                    prompts=prompts,
                    propagation_direction=propagation_direction
                )
            else:
                # Process entire video at once
                if self.verbose:
                    logger.info(f"Processing full video without chunking ({video_meta.total_frames} frames)")
                
                driver = self._get_video_driver()
                session_id = driver.start_session(video_path)
                
                try:
                    # Add prompts on frame 0
                    for prompt in prompts:
                        if self.verbose:
                            logger.info(f"Adding prompt '{prompt}' on frame 0")
                        response = driver.add_prompt(session_id, prompt, frame_index=0)
                        if self.verbose:
                            out = response.get("outputs", {})
                            num_objs = len(out.get("out_obj_ids", []))
                            logger.info(f"  Prompt '{prompt}' detected {num_objs} object(s)")
                    
                    # Propagate
                    results, _, _ = driver.propagate_in_video(
                        session_id, propagation_direction=propagation_direction
                    )
                    
                    if self.verbose:
                        logger.info(f"Propagation returned {len(results)} frames")
                finally:
                    driver.close_session(session_id)
            
            # Extract unique object IDs from results
            if self.verbose:
                logger.info(f"Processing results: {len(results)} frames total")
                if results:
                    sample_frame_idx = list(results.keys())[0]
                    sample_result = results[sample_frame_idx]
                    logger.info(f"Sample frame {sample_frame_idx} structure: {type(sample_result)}, keys: {list(sample_result.keys()) if isinstance(sample_result, dict) else 'N/A'}")
            
            all_object_ids = set()
            for frame_result in results.values():
                if isinstance(frame_result, dict) and "out_obj_ids" in frame_result:
                    all_object_ids.update(frame_result["out_obj_ids"])
            
            object_ids = sorted(list(all_object_ids))
            
            #Generate mask video for each object
            mask_files = []
            masks_dir = output_base / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            for obj_id in object_ids:
                # Collect masks for this object across all frames
                masks_by_frame = {}
                for frame_idx, frame_result in results.items():
                    if isinstance(frame_result, dict) and "out_binary_masks" in frame_result:
                        # Extract mask for this specific object ID
                        masks = frame_result["out_binary_masks"]
                        obj_ids_in_frame = frame_result.get("out_obj_ids", [])
                        
                        if obj_id in obj_ids_in_frame:
                            obj_idx = list(obj_ids_in_frame).index(obj_id)
                            masks_by_frame[frame_idx] = masks[obj_idx]
                
                # Generate mask video
                output_video_path = str(masks_dir / f"object_{obj_id}.mp4")
                self._generate_mask_video(
                    masks_by_frame=masks_by_frame,
                    object_id=obj_id,
                    output_path=output_video_path,
                    fps=video_meta.fps,
                    width=video_meta.width,
                    height=video_meta.height,
                    total_frames=video_meta.total_frames
                )
                mask_files.append(output_video_path)
            
            # Save metadata
            metadata = {
                "video_path": video_path,
                "width": video_meta.width,
                "height": video_meta.height,
                "fps": video_meta.fps,
                "total_frames": video_meta.total_frames,
                "num_objects": len(object_ids),
                "object_ids": object_ids,
                "prompts": prompts,
                "chunked": feasibility.requires_chunking
            }
            
            return ProcessingResult(
                success=True,
                output_dir=str(output_base),
                mask_files=mask_files,
                object_ids=object_ids,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return ProcessingResult(
                success=False,
                output_dir=str(output_base) if output_base else "",
                mask_files=[],
                object_ids=[],
                metadata={},
                errors=[str(e)]
            )
        finally:
            # Cleanup extracted video segment
            if extracted_video_path and os.path.exists(extracted_video_path):
                try:
                    os.remove(extracted_video_path)
                    if self.verbose:
                        logger.info(f"Cleaned up temporary video segment: {extracted_video_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary video: {e}")
    
    def process_video_with_points(
        self,
        video_path: str,
        frame_idx: int,
        points: List[List[float]],
        point_labels: List[int],
        object_id: int = 1,
        output_dir: Optional[str] = None,
        propagation_direction: str = None
    ) -> ProcessingResult:
        """Process video with point prompts on a specific frame (scenario f).
        
        Args:
            video_path: Path to video file.
            frame_idx: Frame index where points are annotated.
            points: List of [x, y] click coordinates (absolute pixels).
            point_labels: List of labels (1=positive, 0=negative).
            object_id: Unique ID for this object (default: 1).
            output_dir: Output directory. Uses default if None.
            propagation_direction: "forward", "backward", or "both".
        
        Returns:
            ProcessingResult with mask videos, object IDs, and metadata.
        """
        if propagation_direction is None:
            propagation_direction = DEFAULT_PROPAGATION_DIRECTION
        
        # Validate input
        validated = self._validate_video_input(
            video_path, points=points, point_labels=point_labels
        )
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        video_name = Path(video_path).stem
        
        # Similar feasibility check and processing as prompts method
        # For simplicity, implementing without chunking first (can be added later)
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        output_base = self._create_output_directory(output_dir, video_name)
        
        try:
            driver = self._get_video_driver()
            session_id = driver.start_session(video_path)
            
            try:
                # Add point prompt on specified frame
                driver.add_object_with_points_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    object_id=object_id,
                    frame_width=video_meta.width,
                    frame_height=video_meta.height,
                    points=points,
                    point_labels=point_labels
                )
                
                # Propagate
                results, _, _ = driver.propagate_in_video(
                    session_id,
                    start_frame_idx=frame_idx,
                    propagation_direction=propagation_direction
                )
            finally:
                driver.close_session(session_id)
            
            # Generate mask video
            masks_by_frame = {}
            for frame_idx, frame_result in results.items():
                if isinstance(frame_result, dict) and "masks" in frame_result:
                    masks = frame_result["masks"]
                    # Assuming first mask is our object
                    if len(masks) > 0:
                        masks_by_frame[frame_idx] = masks[0]
            
            masks_dir = output_base / "masks"
            masks_dir.mkdir(exist_ok=True)
            output_video_path = str(masks_dir / f"object_{object_id}.mp4")
            
            self._generate_mask_video(
                masks_by_frame=masks_by_frame,
                object_id=object_id,
                output_path=output_video_path,
                fps=video_meta.fps,
                width=video_meta.width,
                height=video_meta.height,
                total_frames=video_meta.total_frames
            )
            
            metadata = {
                "video_path": video_path,
                "width": video_meta.width,
                "height": video_meta.height,
                "fps": video_meta.fps,
                "total_frames": video_meta.total_frames,
                "annotation_frame": frame_idx,
                "object_id": object_id,
                "num_points": len(points)
            }
            
            return ProcessingResult(
                success=True,
                output_dir=str(output_base),
                mask_files=[output_video_path],
                object_ids=[object_id],
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error processing video with points: {e}")
            return ProcessingResult(
                success=False,
                output_dir=str(output_base) if output_base else "",
                mask_files=[],
                object_ids=[],
                metadata={},
                errors=[str(e)]
            )
    
    def refine_video_object(
        self,
        video_path: str,
        frame_idx: int,
        object_id: int,
        points: List[List[float]],
        point_labels: List[int],
        output_dir: Optional[str] = None,
        propagation_direction: str = "forward"
    ) -> ProcessingResult:
        """Refine existing object in video using additional point prompts (scenario g).
        
        Args:
            video_path: Path to video file.
            frame_idx: Frame index where refinement prompts are annotated.
            object_id: Existing object ID to refine.
            points: List of [x, y] click coordinates (absolute pixels).
            point_labels: List of labels (1=positive, 0=negative).
            output_dir: Output directory. Uses default if None.
            propagation_direction: "forward", "backward", or "both".
        
        Returns:
            ProcessingResult with updated mask videos.
        """
        # Validate input
        validated = self._validate_video_input(
            video_path, points=points, point_labels=point_labels
        )
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        video_name = Path(video_path).stem
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        output_base = self._create_output_directory(output_dir, video_name)
        
        try:
            driver = self._get_video_driver()
            session_id = driver.start_session(video_path)
            
            try:
                # Add refinement points for the object
                driver.add_object_with_points_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    object_id=object_id,
                    frame_width=video_meta.width,
                    frame_height=video_meta.height,
                    points=points,
                    point_labels=point_labels
                )
                
                # Re-propagate with refined prompts
                results, _, _ = driver.propagate_in_video(
                    session_id,
                    start_frame_idx=frame_idx,
                    propagation_direction=propagation_direction
                )
            finally:
                driver.close_session(session_id)
            
            # Generate mask video
            masks_by_frame = {}
            for frame_idx_result, frame_result in results.items():
                if isinstance(frame_result, dict) and "masks" in frame_result:
                    masks = frame_result["masks"]
                    obj_ids = frame_result.get("object_ids", [])
                    
                    # Find mask for specified object_id
                    if object_id in obj_ids:
                        obj_idx = obj_ids.index(object_id)
                        masks_by_frame[frame_idx_result] = masks[obj_idx]
            
            masks_dir = output_base / "masks"
            masks_dir.mkdir(exist_ok=True)
            output_video_path = str(masks_dir / f"object_{object_id}_refined.mp4")
            
            self._generate_mask_video(
                masks_by_frame=masks_by_frame,
                object_id=object_id,
                output_path=output_video_path,
                fps=video_meta.fps,
                width=video_meta.width,
                height=video_meta.height,
                total_frames=video_meta.total_frames
            )
            
            metadata = {
                "video_path": video_path,
                "object_id": object_id,
                "refinement_frame": frame_idx,
                "num_refinement_points": len(points),
                "width": video_meta.width,
                "height": video_meta.height,
                "fps": video_meta.fps,
                "total_frames": video_meta.total_frames
            }
            
            return ProcessingResult(
                success=True,
                output_dir=str(output_base),
                mask_files=[output_video_path],
                object_ids=[object_id],
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error refining video object: {e}")
            return ProcessingResult(
                success=False,
                output_dir=str(output_base) if output_base else "",
                mask_files=[],
                object_ids=[],
                metadata={},
                errors=[str(e)]
            )
    
    def remove_video_objects(
        self,
        video_path: str,
        object_ids: Union[int, List[int]],
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """Remove objects from video segmentation by their IDs (scenario h).
        
        Args:
            video_path: Path to video file.
            object_ids: Object ID(s) to remove.
            output_dir: Output directory. Uses default if None.
        
        Returns:
            ProcessingResult with updated mask videos (without removed objects).
        """
        # Normalize to list
        if isinstance(object_ids, int):
            object_ids_list = [object_ids]
        else:
            object_ids_list = object_ids
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        video_name = Path(video_path).stem
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        output_base = self._create_output_directory(output_dir, video_name)
        
        try:
            driver = self._get_video_driver()
            session_id = driver.start_session(video_path)
            
            try:
                # Remove objects
                for obj_id in object_ids_list:
                    driver.remove_object(session_id, obj_id)
                
                # In a full implementation, we'd re-propagate and generate new mask videos
                # For now, return success with metadata about removed objects
                
                metadata = {
                    "video_path": video_path,
                    "removed_object_ids": object_ids_list,
                    "width": video_meta.width,
                    "height": video_meta.height,
                    "fps": video_meta.fps,
                    "total_frames": video_meta.total_frames
                }
                
                logger.warning("Object removal completed. Re-propagation not yet implemented.")
                
                return ProcessingResult(
                    success=True,
                    output_dir=str(output_base),
                    mask_files=[],  # Would contain remaining object masks in full implementation
                    object_ids=[],  # Would contain remaining object IDs
                    metadata=metadata
                )
            finally:
                driver.close_session(session_id)
        
        except Exception as e:
            logger.error(f"Error removing video objects: {e}")
            return ProcessingResult(
                success=False,
                output_dir=str(output_base) if output_base else "",
                mask_files=[],
                object_ids=[],
                metadata={},
                errors=[str(e)]
            )
    
    def process_video_with_segments(
        self,
        video_path: str,
        segments: Dict[str, Any],
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """Process video with segment-based prompts/points (scenario i).
        
        Args:
            video_path: Path to video file.
            segments: Dictionary with segment definitions. Structure:
                {
                    "segments": [
                        {
                            "start_frame": 0,
                            "end_frame": 50,
                            "prompts": ["person", "car"]
                        },
                        {
                            "start_time_sec": 2.0,
                            "end_time_sec": 5.0,
                            "points": [[100, 200], [150, 250]],
                            "labels": [1, 0]
                        }
                    ]
                }
            output_dir: Output directory. Uses default if None.
        
        Returns:
            ProcessingResult with mask videos for all segments.
        """
        # Validate input
        validated = self._validate_video_input(video_path, segments=segments)
        segments_list = validated["segments"]
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        video_name = Path(video_path).stem
        
        # Determine output directory
        if output_dir is None:
            output_dir = str(self.default_output_dir)
        
        output_base = self._create_output_directory(output_dir, video_name)
        
        all_mask_files = []
        all_object_ids = set()
        segment_results = []
        
        try:
            driver = self._get_video_driver()
            
            # Process each segment
            for seg_idx, segment in enumerate(segments_list):
                # Convert time to frames if needed
                if "start_time_sec" in segment:
                    start_frame = int(segment["start_time_sec"] * video_meta.fps)
                    end_frame = int(segment["end_time_sec"] * video_meta.fps)
                else:
                    start_frame = segment["start_frame"]
                    end_frame = segment["end_frame"]
                
                logger.info(f"Processing segment {seg_idx + 1}/{len(segments_list)}: frames {start_frame}-{end_frame}")
                
                session_id = driver.start_session(video_path)
                
                try:
                    # Add prompts for this segment
                    if "prompts" in segment:
                        for prompt in segment["prompts"]:
                            driver.add_prompt(session_id, prompt)
                    
                    if "points" in segment:
                        obj_id = segment.get("object_id", seg_idx + 1)
                        point_labels = segment.get("labels", [1] * len(segment["points"]))
                        
                        driver.add_object_with_points_prompt(
                            session_id=session_id,
                            frame_idx=start_frame,
                            object_id=obj_id,
                            frame_width=video_meta.width,
                            frame_height=video_meta.height,
                            points=segment["points"],
                            point_labels=point_labels
                        )
                        all_object_ids.add(obj_id)
                    
                    # Propagate within segment bounds
                    # Note: This processes entire video; ideally we'd limit to segment range
                    results, _, _ = driver.propagate_in_video(
                        session_id,
                        start_frame_idx=start_frame,
                        propagation_direction="forward"
                    )
                    
                    # Filter results to segment range
                    segment_results_filtered = {
                        k: v for k, v in results.items()
                        if start_frame <= k <= end_frame
                    }
                    
                    segment_results.append({
                        "segment_idx": seg_idx,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "results": segment_results_filtered
                    })
                
                finally:
                    driver.close_session(session_id)
            
            # Generate mask videos for each object across all segments
            masks_dir = output_base / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            for obj_id in sorted(all_object_ids):
                masks_by_frame = {}
                
                # Collect masks for this object from all segments
                for seg_result in segment_results:
                    for frame_idx, frame_result in seg_result["results"].items():
                        if isinstance(frame_result, dict) and "masks" in frame_result:
                            obj_ids = frame_result.get("object_ids", [])
                            if obj_id in obj_ids:
                                obj_idx = obj_ids.index(obj_id)
                                masks_by_frame[frame_idx] = frame_result["masks"][obj_idx]
                
                if masks_by_frame:
                    output_video_path = str(masks_dir / f"object_{obj_id}.mp4")
                    self._generate_mask_video(
                        masks_by_frame=masks_by_frame,
                        object_id=obj_id,
                        output_path=output_video_path,
                        fps=video_meta.fps,
                        width=video_meta.width,
                        height=video_meta.height,
                        total_frames=video_meta.total_frames
                    )
                    all_mask_files.append(output_video_path)
            
            metadata = {
                "video_path": video_path,
                "num_segments": len(segments_list),
                "num_objects": len(all_object_ids),
                "object_ids": sorted(list(all_object_ids)),
                "width": video_meta.width,
                "height": video_meta.height,
                "fps": video_meta.fps,
                "total_frames": video_meta.total_frames
            }
            
            return ProcessingResult(
                success=True,
                output_dir=str(output_base),
                mask_files=all_mask_files,
                object_ids=sorted(list(all_object_ids)),
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error processing video with segments: {e}")
            return ProcessingResult(
                success=False,
                output_dir=str(output_base) if output_base else "",
                mask_files=[],
                object_ids=[],
                metadata={},
                errors=[str(e)]
            )
    
    # ========================================================================
    # Chunk Processing Methods
    # ========================================================================
    
    def _process_video_chunk(
        self,
        chunk_path: str,
        chunk_info: ChunkInfo,
        prompts: Optional[List[str]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        object_id: Optional[int] = None,
        propagation_direction: str = None
    ) -> Dict[int, Any]:
        """
        Process a single video chunk.
        
        Returns dict mapping frame_index (global) to segmentation results.
        """
        if propagation_direction is None:
            propagation_direction = DEFAULT_PROPAGATION_DIRECTION
        
        driver = self._get_video_driver()
        
        # Start session for this chunk
        session_id = driver.start_session(chunk_path)
        
        try:
            # Get chunk video metadata for frame dimensions
            chunk_meta = self._get_video_metadata(chunk_path)
            
            # Add prompts or points
            if prompts is not None:
                for prompt in prompts:
                    driver.add_prompt(session_id, prompt)
            
            elif points is not None and point_labels is not None:
                # Add point prompt on first frame of chunk (frame 0 in chunk coords)
                if object_id is None:
                    object_id = 1
                
                driver.add_object_with_points_prompt(
                    session_id=session_id,
                    frame_idx=0,  # First frame of chunk
                    object_id=object_id,
                    frame_width=chunk_meta.width,
                    frame_height=chunk_meta.height,
                    points=points,
                    point_labels=point_labels
                )
            
            # Propagate across chunk
            chunk_results, _, _ = driver.propagate_in_video(
                session_id=session_id,
                propagation_direction=propagation_direction
            )
            
            # Convert chunk-local frame indices to global frame indices
            global_results = {}
            for local_frame_idx, result in chunk_results.items():
                global_frame_idx = chunk_info.start_frame + local_frame_idx
                global_results[global_frame_idx] = result
            
            if self.verbose:
                logger.info(
                    f"Chunk {chunk_info.chunk_id}: Processed {len(global_results)} frames "
                    f"(global frames {chunk_info.start_frame}-{chunk_info.end_frame})"
                )
            
            return global_results
        
        finally:
            # Always close session
            driver.close_session(session_id)
    
    def _process_video_in_chunks(
        self,
        video_path: str,
        video_meta: VideoMetadata,
        chunks: List[ChunkInfo],
        prompts: Optional[List[str]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        object_id: Optional[int] = None,
        propagation_direction: str = None
    ) -> Dict[int, Any]:
        """
        Process video in chunks with parallel loading (90% threshold).
        
        Args:
            video_path: Path to video file.
            video_meta: Video metadata.
            chunks: List of chunk information.
            prompts: Text prompts to apply.
            points: Point coordinates for annotation.
            point_labels: Labels for points.
            object_id: Object ID for tracking.
            propagation_direction: Direction for propagation.
        
        Returns:
            Dict mapping frame_index to results (across all chunks).
        """
        if propagation_direction is None:
            propagation_direction = DEFAULT_PROPAGATION_DIRECTION
        
        # Create temp directory for chunks
        video_name = Path(video_path).stem
        chunk_dir = self.temp_dir / video_name / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_results_list = []
        next_chunk_path = None
        next_chunk_ready = threading.Event()
        
        def extract_next_chunk(chunk_info: ChunkInfo):
            """Thread function to extract next chunk."""
            nonlocal next_chunk_path
            output_path = str(chunk_dir / f"chunk_{chunk_info.chunk_id}.mp4")
            next_chunk_path = self._extract_video_chunk(video_path, chunk_info, output_path)
            next_chunk_ready.set()
        
        # Extract first chunk
        if len(chunks) > 0:
            first_chunk_path = self._extract_video_chunk(
                video_path, chunks[0], str(chunk_dir / f"chunk_0.mp4")
            )
            chunks[0].chunk_path = first_chunk_path
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Start extracting next chunk in parallel if available
            extraction_thread = None
            if i + 1 < len(chunks):
                next_chunk_ready.clear()
                extraction_thread = threading.Thread(
                    target=extract_next_chunk,
                    args=(chunks[i + 1],)
                )
                # Don't start extraction immediately - will start at 90% progress
            
            if self.verbose:
                logger.info(
                    f"Processing chunk {i+1}/{len(chunks)} "
                    f"(frames {chunk.start_frame}-{chunk.end_frame})"
                )
            
            # Process current chunk
            # Note: In real implementation, we'd track progress and start extraction thread
            # at 90% completion. For simplicity, we start it immediately here.
            if extraction_thread is not None:
                extraction_thread.start()
            
            chunk_path = chunk.chunk_path or str(chunk_dir / f"chunk_{chunk.chunk_id}.mp4")
            chunk_results = self._process_video_chunk(
                chunk_path=chunk_path,
                chunk_info=chunk,
                prompts=prompts,
                points=points,
                point_labels=point_labels,
                object_id=object_id,
                propagation_direction=propagation_direction
            )
            
            # Store chunk results for proper merging
            chunk_results_list.append(chunk_results)
            
            # Wait for next chunk extraction to complete before next iteration
            if extraction_thread is not None:
                extraction_thread.join()
                chunks[i + 1].chunk_path = next_chunk_path
            
            # Clean up current chunk file to save space
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        # Merge chunk results with proper overlap handling
        all_results = self._merge_chunk_results(
            chunk_results=chunk_results_list,
            chunks=chunks,
            overlap=self.min_chunk_overlap
        )
        
        # Clean up chunk directory
        try:
            shutil.rmtree(chunk_dir.parent)
            if self.verbose:
                logger.info(f"Cleaned up temporary chunk directory: {chunk_dir.parent}")
        except Exception as e:
            logger.warning(f"Failed to cleanup chunk directory: {e}")
        
        return all_results
    
    # ========================================================================
    # Output Management Methods
    # ========================================================================
    
    def _create_output_directory(
        self,
        base_dir: str,
        video_or_image_name: str
    ) -> Path:
        """Create output directory structure."""
        output_dir = Path(base_dir) / video_or_image_name
        
        # Create main output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create masks subdirectory
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        if self.verbose:
            logger.info(f"Output directory created: {output_dir}")
        
        return output_dir
    
    def _save_image_masks(
        self,
        masks: np.ndarray,
        object_ids: List[int],
        output_dir: Path,
        prompt: str
    ) -> List[str]:
        """Save image masks as PNG files with format [prompt]_[ObjectId].png."""
        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # masks shape: (num_objects, H, W) or (num_objects, H, W, C)
        if masks.ndim == 3:
            num_objects = masks.shape[0]
        else:
            num_objects = len(masks)
        
        for i, obj_id in enumerate(object_ids):
            if i >= num_objects:
                logger.warning(f"Object ID {obj_id} exceeds available masks ({num_objects})")
                continue
            
            # Get mask for this object
            mask = masks[i]
            
            # Handle different mask shapes/formats
            # Squeeze any singleton dimensions (e.g., (1, 1, H, W) -> (H, W))
            while mask.ndim > 2 and (mask.shape[0] == 1 or mask.shape[-1] == 1):
                mask = np.squeeze(mask)
            
            # If still 3D or higher, take the first channel/slice
            if mask.ndim > 2:
                mask = mask[..., 0] if mask.shape[-1] <= mask.shape[0] else mask[0]
            
            # Ensure mask is 2D
            if mask.ndim != 2:
                logger.warning(f"Unexpected mask shape {mask.shape} for object {obj_id}, attempting to reshape")
                # Try to convert to 2D if possible
                if np.prod(mask.shape) == mask.shape[-1] * mask.shape[-2]:
                    mask = mask.reshape(mask.shape[-2], mask.shape[-1])
                else:
                    logger.error(f"Cannot convert mask shape {mask.shape} to 2D for object {obj_id}")
                    continue
            
            # If mask is binary (0, 1), convert to 0, 255 for visibility
            if mask.dtype == bool or (mask.max() <= 1 and mask.min() >= 0):
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
            
            # Save mask as PNG with format [prompt]_[ObjectId].png
            filename = f"{prompt}_{obj_id}.png"
            filepath = masks_dir / filename
            
            # Convert to PIL Image and save
            mask_image = Image.fromarray(mask)
            mask_image.save(str(filepath))
            
            saved_files.append(str(filepath))
            
            if self.verbose:
                logger.debug(f"Saved mask for object {obj_id}: {filepath}")
        
        return saved_files
    
    def _generate_mask_video(
        self,
        masks_by_frame: Dict[int, np.ndarray],
        object_id: int,
        output_path: str,
        fps: float,
        width: int,
        height: int,
        total_frames: int
    ) -> str:
        """
        Generate a mask video for a specific object ID.
        
        Fills missing frames with black frames.
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
        
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_path}")
        
        try:
            for frame_idx in range(total_frames):
                if frame_idx in masks_by_frame:
                    # Use actual mask for this frame
                    mask = masks_by_frame[frame_idx]
                    
                    # Ensure mask is 2D and correct size
                    if mask.ndim == 3:
                        # If multiple objects, extract the one for this object_id
                        # Assuming mask shape is (num_objects, H, W)
                        if mask.shape[0] > object_id:
                            mask = mask[object_id]
                        else:
                            mask = mask[0]  # Fallback to first object
                    
                    # Convert to uint8 if needed
                    if mask.dtype == bool or mask.max() <= 1:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)
                    
                    # Resize if needed
                    if mask.shape[0] != height or mask.shape[1] != width:
                        mask = cv2.resize(mask, (width, height))
                else:
                    # Create black frame for missing frames
                    mask = np.zeros((height, width), dtype=np.uint8)
                
                # Write frame
                video_writer.write(mask)
            
            if self.verbose:
                logger.debug(
                    f"Generated mask video for object {object_id}: {output_path} "
                    f"({len(masks_by_frame)}/{total_frames} frames with masks)"
                )
            
            return output_path
        
        finally:
            video_writer.release()
    
    def _merge_chunk_results(
        self,
        chunk_results: List[Dict[int, Any]],
        chunks: List[ChunkInfo],
        overlap: int
    ) -> Dict[int, Any]:
        """
        Merge results from chunks, handling overlap.
        
        Overlap logic: Skip first `overlap` frames of each chunk (except first chunk).
        For our strategy where chunk2 starts at frame 24, we skip frame 24 in chunk2
        since it was already processed in chunk1.
        """
        merged = {}
        
        for i, (chunk_result, chunk_info) in enumerate(zip(chunk_results, chunks)):
            if i == 0:
                # First chunk: include all frames
                merged.update(chunk_result)
            else:
                # Subsequent chunks: skip overlap frames
                for frame_idx, result in chunk_result.items():
                    # Skip frames that are in the overlap region
                    if frame_idx < chunk_info.start_frame + overlap:
                        if self.verbose:
                            logger.debug(
                                f"Skipping overlapping frame {frame_idx} from chunk {i}"
                            )
                        continue
                    merged[frame_idx] = result
        
        if self.verbose:
            logger.info(f"Merged {len(merged)} unique frames from {len(chunks)} chunks")
        
        return merged
    
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
    
    def _print_initialization_info(self):
        """Print initialization information."""
        logger.info("=" * 60)
        logger.info("SAM3 Entrypoint Initialized")
        logger.info("=" * 60)
        logger.info(f"Device: {DEVICE.type.upper()}")
        logger.info(f"Memory Available: {self.memory_info.available_gb:.2f} GB")
        
        if self.memory_info.device_type == "cuda":
            logger.info(f"VRAM Usage %: {self.vram_usage_percent * 100:.0f}%")
        else:
            logger.info(f"RAM Usage %: {self.ram_usage_percent * 100:.0f}%")
        
        logger.info(f"Min Video Frames: {self.min_video_frames}")
        logger.info(f"Min Chunk Overlap: {self.min_chunk_overlap}")
        logger.info(f"Temp Dir: {self.temp_dir}")
        logger.info(f"Output Dir: {self.default_output_dir}")
        logger.info("=" * 60)
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        if self._image_driver is not None:
            self._image_driver.cleanup()
        if self._video_driver is not None:
            self._video_driver.cleanup()
        
        # Optionally clean temp directory
        if self.verbose:
            logger.info(f"Cleanup complete. Temp files in: {self.temp_dir}")


# ============================================================================
# Convenience Functions
# ============================================================================

def process_image_simple(
    image_path: Union[str, List[str]],
    prompts: Union[str, List[str]],
    output_dir: Optional[str] = None
) -> ProcessingResult:
    """
    Convenience function for simple image processing with text prompts.
    
    Example:
        >>> result = process_image_simple("image.jpg", "person")
        >>> print(f"Masks saved to: {result.output_dir}")
    """
    entrypoint = Sam3Entrypoint()
    return entrypoint.process_image(image_path, prompts=prompts, output_dir=output_dir)


def process_video_simple(
    video_path: str,
    prompts: Union[str, List[str]],
    output_dir: Optional[str] = None
) -> ProcessingResult:
    """
    Convenience function for simple video processing with text prompts.
    
    Example:
        >>> result = process_video_simple("video.mp4", "person")
        >>> print(f"Mask videos saved to: {result.output_dir}")
    """
    entrypoint = Sam3Entrypoint()
    return entrypoint.process_video_with_prompts(video_path, prompts, output_dir)


if __name__ == "__main__":
    # Example usage and testing
    print("SAM3 Entrypoint Module")
    print("=" * 60)
    
    entrypoint = Sam3Entrypoint(verbose=True)
    print("\nEntrypoint initialized successfully!")
    
    # TODO: Add example usage here once implementation is complete
