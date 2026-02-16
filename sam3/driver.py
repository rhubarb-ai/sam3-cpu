"""
SAM3 CPU + GPU Driver

This module provides a unified interface for running SAM3 on both CPU and GPU. 
It automatically selects the best available backend for prompt based segmentation operations, 
ensuring optimal performance across different hardware configurations.
"""
import os
from typing import List, Optional
import numpy as np
from typing_extensions import Literal
import torch
from sam3.profiler import profile
from sam3.__globals import DEVICE, BPE_PATH
from sam3.logger import get_logger

logger = get_logger(__name__)

class Sam3ImageDriver:
    def __init__(self, bpe_path: str = BPE_PATH, num_workers: Optional[int] = 1):
        self.predictor = self._get_predictor(bpe_path=bpe_path, num_workers=num_workers)

    @profile()
    def _build_model(self, bpe_path, device):
        from sam3 import build_sam3_image_model
        logger.info(f"Loading model on device: {device}")
        model = build_sam3_image_model(bpe_path=bpe_path, device=device)
        return model

    @profile()
    def _get_predictor(self, bpe_path: Optional[str], num_workers: Optional[int]):
        if DEVICE.type == "cpu":
            logger.info("Running on CPU. For better performance, please run on a GPU.")
            torch.backends.cpu.get_cpu_capability()
        else:
            logger.info("Running on GPU. Enabling TF32 and bfloat16 for better performance.")
            # turn on tfloat32 for Ampere GPUs
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

        return self._build_model(bpe_path=bpe_path, device=DEVICE.type)
        
    @profile()
    def inference(self, image):
        from sam3.model.sam3_image_processor import Sam3Processor
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        processor = Sam3Processor(self.predictor, confidence_threshold=0.5)
        inference_state = processor.set_image(image)
        return processor, inference_state
    
    @profile()
    def prompt_and_predict(self, processor, inference_state, prompt: str="people"):
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)
        return inference_state

    @profile()
    def prompt_image(self, image_path: str, prompts: str | List[str]=["people"]):
        if os.path.isfile(image_path):
            from PIL import Image
            image = Image.open(image_path)
            # Check if empty image
            if image.size == (0, 0):
                raise ValueError(f"Image file is empty: {image_path}")
            else:
                logger.info(f"Loaded image: {image_path} with size {image.size}")
        else:
            raise ValueError(f"Could not load image file: {image_path}")
        
        # Run inference and prompting
        processor, inference_state = self.inference(image)
        if isinstance(prompts, str):
            prompts = [prompts]

        result = {}
        for prompt in prompts:
            inference_state = self.prompt_and_predict(processor, inference_state, prompt=prompt)
            result[prompt] = inference_state
            nb_objects = len(inference_state["scores"])
            logger.info(f"found {nb_objects} {prompt}(s)")
        return result





class Sam3VideoDriver():
    def __init__(self, bpe_path: Optional[str] = BPE_PATH, num_workers: Optional[int] = 1):
        self.predictor = self._get_predictor(bpe_path=bpe_path, num_workers=num_workers)

    @profile()
    def _get_predictor(self, bpe_path: Optional[str], num_workers: Optional[int]):
        if DEVICE.type == "cpu":
            from sam3.model_builder import build_sam3_video_predictor_cpu
            logger.info("Running on CPU. For better performance, please run on a GPU.")
            torch.backends.cpu.get_cpu_capability()
            self.predictor = build_sam3_video_predictor_cpu(bpe_path=bpe_path, num_workers=num_workers)
        else:
            from sam3.model_builder import build_sam3_video_predictor
            logger.info("Running on GPU. Enabling TF32 and bfloat16 for better performance.")
            # turn on tfloat32 for Ampere GPUs
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

            # use all available GPUs on the machine
            gpus_to_use = range(torch.cuda.device_count())
            self.predictor = build_sam3_video_predictor(bpe_path=bpe_path, gpus_to_use=gpus_to_use)

    def _abs_to_rel_coords(self, coords, img_width, img_height, coord_type="point"):
        """Convert absolute coordinates to relative coordinates (0-1 range)

        Args:
            coords: List of coordinates
            coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
        """
        if coord_type == "point":
            return [[x / img_width, y / img_height] for x, y in coords]
        elif coord_type == "box":
            return [
                [x / img_width, y / img_height, w / img_width, h / img_height]
                for x, y, w, h in coords
            ]
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")

# -------------- Prompting APIs (add prompt, propagate, remove object) --------------
    @profile()
    def add_prompt(self, session_id: str, prompt: str):
        """Adds a text prompt to an active video segmentation session."""
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                prompt=prompt,
            )
        )
        return response # response["outputs"] contains the segmentation results for the prompt

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
        """Adds an object to the video segmentation results using point prompts."""
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        
        labels = np.array(point_labels)
        points_abs = np.array(points)

        # convert points and labels to tensors; also convert to relative coordinates
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

        return response # response["outputs"] contains the updated segmentation results after adding the object

    @profile()
    def propagate_in_video(
        self, 
        session_id: str, 
        start_frame_idx: int = None, 
        frames_to_track: int = None, 
        propogation_direction: Literal["both", "forward", "backward"] = "both"
    ):
        """Propagates segmentation results to a specific frame in the video.
        
        Args:
            session_id: The session ID for the video segmentation session.
            start_frame_idx: The starting frame index for propagation. If None, it will use the last prompted frame as the starting point.
            frames_to_track: Number of frames to track. If None, it will track until the end of the video.
            propogation_direction: Direction of propagation. Options: "both", "forward" or "backward".
        """
        if self.predictor is None:
            raise ValueError("Model is not loaded.")
        
        if propogation_direction not in ("both", "forward", "backward"):
            raise ValueError("propogation_direction must be 'both', 'forward' or 'backward'. Options: 'both', 'forward', 'backward'")
        
        result = {}
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                start_frame_index=start_frame_idx,
                max_frame_num_to_track=frames_to_track,
                propagation_direction=propogation_direction,
            )
        ):
            result[response["frame_index"]] = response["outputs"]
        return result
    
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
        """Refines an existing object in the video segmentation results using point prompts."""
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
    def remove_object(self, session_id: str, object_id: int):
        """Removes an object from the video segmentation results."""
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
        """Cleans up the predictor and releases resources."""
        self.shutdown()

        # Free Torch memory from CPU and GPU
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "cpu": # Free memory on CPU - Works only on Linux
            import ctypes
            import gc

            if 'predictor' in locals(): 
                del predictor
            if 'response' in locals(): 
                del response
            if 'video_frames_for_vis' in locals(): 
                del video_frames_for_vis
            if 'outputs_per_frame' in locals(): 
                del outputs_per_frame

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)

            gc.collect()
    
    @profile()
    def close_session(self, session_id: str):
        """Closes a video segmentation session."""
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
        """Resets a video segmentation session."""
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
        """Starts a video segmentation session and returns the session ID."""
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
        """Shuts down the video segmentation predictor."""

        if self.predictor is not None:
            if not self.predictor.has_shutdown:
                self.predictor.shutdown()
            else:
                logger.warning("Predictor is already shut down.")
        else:
            logger.warning("Predictor was never initialized.")