import functools
import subprocess
import json
import os
import math
from typing import Literal
import psutil
from pathlib import Path
from sam3.ffmpeglib import ffmpeg_lib
from sam3.utils import ram_stat, vram_stat
from sam3.__globals import (
    logger,
    DEVICE, 
    VIDEO_INFERENCE_MB, 
    IMAGE_INFERENCE_MB,
    TENSOR_SIZE_BYTES,
    DEFAULT_MIN_CHUNK_OVERLAP,
    DEFAULT_MIN_VIDEO_FRAMES,
    RAM_USAGE_PERCENT,
    VRAM_USAGE_PERCENT,
    CPU_MEMORY_RESERVE_PERCENT,
    GPU_MEMORY_RESERVE_PERCENT
)

class MemoryError(Exception):
    """Custom exception for memory-related errors."""
    
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"MemoryError: {self.message}"
    
    def __repr__(self):
        return f"MemoryError(message={self.message})"


class MemoryManager:
    def __init__(self):
        pass

    def compute_memory_safe_frames(
            self, 
            width: int, 
            height: int, 
            device: str = DEVICE.type, 
            type: Literal['video', 'image'] = 'video'
        ):
        """Compute maximum frames that fit safely in RAM."""
        bytes_per_frame = width * height * 3  # CV_8UC3
        bytes_per_frame += TENSOR_SIZE_BYTES  # Add estimated tensor size for safety

        logger.debug(f"Frame size in MB: {bytes_per_frame / (1024 ** 2):.2f} MB")

        if device == 'cpu':
            memory_info = ram_stat()
            percent  = RAM_USAGE_PERCENT
            vid_memory_max_usage = RAM_USAGE_PERCENT * memory_info['total'] * (1 - CPU_MEMORY_RESERVE_PERCENT)
        elif device == 'cuda':
            memory_info = vram_stat()
            percent  = VRAM_USAGE_PERCENT
            vid_memory_max_usage = VRAM_USAGE_PERCENT * memory_info['total'] * (1 - GPU_MEMORY_RESERVE_PERCENT)
        else:
            raise ValueError(f"Unsupported device type: {device}")

        if type == 'video':
            available_bytes = memory_info['available'] - VIDEO_INFERENCE_MB * 1024 ** 2
        else:
            available_bytes = memory_info['available'] - IMAGE_INFERENCE_MB * 1024 ** 2
        
        if available_bytes <= 0:
            least_extra_mb = DEFAULT_MIN_VIDEO_FRAMES * (VIDEO_INFERENCE_MB if type == 'video' else IMAGE_INFERENCE_MB) + bytes_per_frame // (1024 ** 2)
            logger.warning(f"Not enough available RAM for inference. Available: {memory_info['available']} bytes, Required: {least_extra_mb} MB. Consider freeing up memory or reducing video resolution.")
            return 0
        
        usable_bytes = int(available_bytes * percent)
        max_frames = usable_bytes // bytes_per_frame
        if max_frames < DEFAULT_MIN_VIDEO_FRAMES:
            logger.warning(f"Estimated max frames ({max_frames}) is below the minimum threshold ({DEFAULT_MIN_VIDEO_FRAMES}). Consider freeing up memory or reducing video resolution.")
            return 0
        logger.debug(f"Estimated max frames that fit in RAM: {max_frames} frames ({max_frames * bytes_per_frame / (1024 ** 2):.2f} MB)")
        return max_frames
    
    def generate_chunks(
            self, 
            total_frames: int, 
            chunk_size: int, 
            chunk_spread: Literal["even", "default"] = "default", 
            overlap: int = None
        ):
        """Generate chunk index ranges."""
        if overlap is None or overlap < DEFAULT_MIN_CHUNK_OVERLAP:
            overlap = DEFAULT_MIN_CHUNK_OVERLAP

        if chunk_spread == "even":
            chunk_size = math.ceil(total_frames / math.ceil(total_frames / chunk_size))
            logger.debug(f"Adjusted chunk size for even spread: {chunk_size} frames")

        stride = chunk_size - overlap
        chunks = []

        start = 0
        idx = 0

        while start < total_frames:
            end = min(start + chunk_size - 1, total_frames - 1)
            if end > start:
                chunks.append({"chunk": idx, "start": start, "end": end})
            start += stride
            idx += 1

        return chunks

    def chunk_plan_video(
            self, 
            video_file: str, 
            device: str = DEVICE.type, 
            chunk_spread: Literal["even", "default"] = "default"
        ):
        # Placeholder for actual chunk planning logic
        logger.info(f"Planning memory chunks for video: {video_file}")
        
        video_info = ffmpeg_lib.get_video_info(video_file)
        if video_info is None:
            logger.warning(f"Could not retrieve video info for: {video_file}")
            return []
        
        logger.info(f"Video info: {video_info}")
        
        frames_per_chunk = self.compute_memory_safe_frames(video_info['width'], video_info['height'], device, type='video')
        
        fps = round(video_info.get('fps', 25))  # Default to 25 if FPS info is missing
        if frames_per_chunk == 0:
            logger.warning(f"Memory constraints are too tight to process any frames for video: {video_file}. Consider freeing up memory or reducing video resolution.")
            raise MemoryError("Insufficient memory to process video frames.")
        
        frames_per_chunk = fps * (frames_per_chunk // fps)  # Adjust to be a multiple of FPS for better chunking

        logger.info(f"Estimated frames per chunk: {frames_per_chunk}")
        
        video_chunks =  self.generate_chunks(video_info['nb_frames'], frames_per_chunk, chunk_spread=chunk_spread)

        metadata = {
            "video": video_file,
            "width": video_info['width'],
            "height": video_info['height'],
            "duration": video_info['duration'],
            "nb_frames": video_info['nb_frames'],
            "fps": video_info['fps'],
            "frames_per_chunk": frames_per_chunk,
            "chunks": video_chunks
        }

        logger.info(f"Generated {len(video_chunks)} chunks for video processing.") 
        logger.info(f"Metadata: {metadata}")

        return metadata, video_chunks


# Singleton instance of MemoryManager
memory_manager = MemoryManager()

# Decorator to check memory before executing a function
def mem_check(device=DEVICE):
    """Profile decorator with global enable/disable control via sam3.__globals.ENABLE_PROFILING"""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pass

        return wrapper

    return decorator