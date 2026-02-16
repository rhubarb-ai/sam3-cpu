# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from .model_builder import build_sam3_image_model
from .entrypoint import Sam3Entrypoint
from .wrapper import Sam3Wrapper
from .model.sam3_video_predictor import Sam3VideoPredictor

# Expose Sam3Entrypoint as Sam3 for convenience
Sam3 = Sam3Entrypoint

__version__ = "0.1.0"

__all__ = [
    "build_sam3_image_model",
    "Sam3Entrypoint",
    "Sam3",
    "Sam3Wrapper",
    "Sam3VideoPredictor",
]
