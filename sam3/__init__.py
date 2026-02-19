# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from .model_builder import build_sam3_image_model
from .entrypoint import Sam3Entrypoint
from .wrapper import Sam3Wrapper
from .model.sam3_video_predictor import Sam3VideoPredictor
from .memory_manager import MemoryManager

# New modular API
from .api import Sam3API
from .image_processor import ImageProcessor
from .video_processor import VideoProcessor
from .chunk_processor import ChunkProcessor
from .postprocessor import VideoPostProcessor

# Expose Sam3API as Sam3 for convenience (new default)
Sam3 = Sam3API

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "Sam3Entrypoint",
    "Sam3API",
    "Sam3",
    "Sam3Wrapper",
    "Sam3VideoPredictor",
    "MemoryManager",
    "ImageProcessor",
    "VideoProcessor",
    "ChunkProcessor",
    "VideoPostProcessor",
]

