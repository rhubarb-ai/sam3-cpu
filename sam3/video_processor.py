"""
SAM3 Video Processor Module

Handles video segmentation with automatic memory management and chunking.
Coordinates chunk processing and manages video-level operations.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from sam3.logger import get_logger
from sam3.memory_manager import memory_manager
from sam3.ffmpeglib import ffmpeg_lib
from sam3.postprocessor import VideoPostProcessor
from sam3.__globals import (
    BPE_PATH,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PROPAGATION_DIRECTION,
    TEMP_DIR
)
from sam3.utils import sanitize_filename

logger = get_logger(__name__)


class VideoProcessor:
    """
    Handles video segmentation with automatic chunking and memory management.
    
    This processor:
    - Analyzes video and creates memory-safe chunks
    - Coordinates chunk processing with prompts
    - Manages temporary and output directories
    - Handles post-processing and visualization (placeholders)
    - Cleans up temporary files
    
    Args:
        video_path: Path to the input video file.
        output_dir: Directory where results will be saved.
        temp_dir: Temporary directory for intermediate files.
        device: Device to use ('cpu' or 'cuda').
        bpe_path: Path to the BPE tokenizer model file.
        num_workers: Number of worker threads for CPU processing.
    
    Attributes:
        video_path: Path to input video
        video_name: Name of the video (without extension)
        output_dir: Output directory for final results
        temp_dir: Temporary directory for intermediate files
    """
    
    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        temp_dir: Path,
        device: str,
        bpe_path: str = BPE_PATH,
        num_workers: int = DEFAULT_NUM_WORKERS
    ):
        """Initialize the VideoProcessor."""
        self.video_path = Path(video_path)
        self.video_name = self.video_path.stem
        self.device = device
        self.bpe_path = bpe_path
        self.num_workers = num_workers
        
        # Directory structure
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        
        # Video-specific directories in temp
        self.video_temp_dir = self.temp_dir / self.video_name
        self.chunks_temp_dir = self.video_temp_dir / "chunks"
        self.meta_temp_dir = self.video_temp_dir / "metadata"
        
        # Video-specific directories in output
        self.video_output_dir = self.output_dir / self.video_name
        self.masks_output_dir = self.video_output_dir / "masks"
        self.meta_output_dir = self.video_output_dir / "metadata"
        self.results_output_dir = self.video_output_dir / "results"
        
        # Shared video driver (loaded once, reused for all chunks)
        self._video_driver = None
        
        # Create directory structure
        self._create_directories()
        
        logger.info(f"VideoProcessor initialized for video: {self.video_name}")
        logger.debug(f"  Temp directory: {self.video_temp_dir}")
        logger.debug(f"  Output directory: {self.video_output_dir}")
    
    def _get_video_driver(self):
        """Get or create the shared video driver (lazy initialization)."""
        if self._video_driver is None:
            from sam3.drivers import Sam3VideoDriver
            logger.info("Initializing shared video driver (model will be loaded once)...")
            self._video_driver = Sam3VideoDriver(
                bpe_path=self.bpe_path,
                num_workers=self.num_workers
            )
            logger.info("Video driver initialized and ready for all chunks")
        return self._video_driver
    
    def _create_directories(self):
        """Create all necessary directories for processing."""
        # Temp directories
        self.video_temp_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_temp_dir.mkdir(exist_ok=True)
        self.meta_temp_dir.mkdir(exist_ok=True)
        
        # Output directories
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.masks_output_dir.mkdir(exist_ok=True)
        self.meta_output_dir.mkdir(exist_ok=True)
        self.results_output_dir.mkdir(exist_ok=True)
        
        logger.debug("Created all necessary directories")
    
    def process_with_prompts(
        self,
        prompts: List[str],
        propagation_direction: str = DEFAULT_PROPAGATION_DIRECTION,
        chunk_spread: str = "default",
        keep_temp_files: bool = False
    ) -> Dict[str, Any]:
        """
        Process video with text prompts.
        
        Workflow:
        1. Analyze video and create chunk plan
        2. Create video chunks
        3. Process each chunk with all prompts
        4. Post-process results (placeholder)
        5. Visualize results (placeholder)
        6. Cleanup temporary files
        
        Args:
            prompts: List of text prompts to process.
            propagation_direction: Direction for mask propagation.
            chunk_spread: Chunking strategy ("even" or "default").
            keep_temp_files: If True, moves temp files to output. If False, deletes them.
        
        Returns:
            Dictionary containing:
                - "video_name": Name of the video
                - "video_path": Path to the video
                - "output_dir": Path to output directory
                - "chunks": List of chunk results
                - "prompts": List of processed prompts
                - "metadata_path": Path to video metadata file
        """
        logger.info(f"Processing video '{self.video_name}' with {len(prompts)} prompt(s)")
        
        # Step 1: Analyze video and create chunk plan
        logger.info("Step 1: Analyzing video and creating chunk plan...")
        video_metadata, video_chunks = self._create_chunk_plan(chunk_spread)
        
        # Save video metadata
        video_meta_path = self._save_video_metadata(video_metadata)
        
        if len(video_chunks) == 0:
            logger.warning("No valid chunks generated. Aborting processing.")
            return {
                "video_name": self.video_name,
                "video_path": str(self.video_path),
                "output_dir": str(self.video_output_dir),
                "chunks": [],
                "prompts": prompts,
                "metadata_path": str(video_meta_path),
                "error": "No valid chunks generated"
            }
        
        logger.info(f"  Generated {len(video_chunks)} chunk(s)")
        
        # Step 2 & 3: Create chunks and process them
        if len(video_chunks) == 1:
            logger.info("Processing video as single chunk...")
            chunk_results = self._process_single_chunk_video(
                prompts=prompts,
                video_metadata=video_metadata,
                propagation_direction=propagation_direction
            )
        else:
            logger.info(f"Processing video in {len(video_chunks)} chunks...")
            chunk_results = self._process_multiple_chunks(
                prompts=prompts,
                video_metadata=video_metadata,
                video_chunks=video_chunks,
                propagation_direction=propagation_direction
            )
        
        # Step 4: Post-processing (placeholder)
        logger.info("Step 4: Post-processing results...")
        self._postprocess_results(chunk_results, prompts)
        
        # Step 5: Visualization (placeholder)
        logger.info("Step 5: Visualizing results...")
        self._visualize_results(chunk_results, prompts)
        
        # Step 6: Cleanup driver
        logger.info("Step 6: Cleaning up driver...")
        if self._video_driver is not None:
            self._video_driver.cleanup()
            self._video_driver = None
            logger.debug("Video driver cleaned up")
        
        # Step 7: Cleanup or move temp files
        logger.info("Step 7: Managing temp files...")
        if keep_temp_files:
            self._move_temp_to_output()
        else:
            self._cleanup_temp_files()
        
        logger.info(f"Video processing complete: {self.video_name}")
        
        return {
            "video_name": self.video_name,
            "video_path": str(self.video_path),
            "output_dir": str(self.video_output_dir),
            "chunks": chunk_results,
            "prompts": prompts,
            "metadata_path": str(video_meta_path),
            "num_chunks": len(video_chunks)
        }
    
    def _create_chunk_plan(self, chunk_spread: str) -> tuple:
        """
        Analyze video and create memory-safe chunk plan.
        
        Args:
            chunk_spread: Chunking strategy ("even" or "default").
        
        Returns:
            Tuple of (video_metadata, video_chunks)
        """
        logger.debug("Creating chunk plan...")
        
        video_metadata, video_chunks = memory_manager.chunk_plan_video(
            video_file=str(self.video_path),
            device=self.device,
            chunk_spread=chunk_spread
        )
        
        logger.debug(f"  Video: {video_metadata['width']}x{video_metadata['height']}, "
                    f"{video_metadata['nb_frames']} frames, "
                    f"{video_metadata['fps']:.2f} fps")
        logger.debug(f"  Chunks: {len(video_chunks)}")
        
        return video_metadata, video_chunks
    
    def _save_video_metadata(self, video_metadata: Dict) -> Path:
        """
        Save video metadata to output directory.
        
        Args:
            video_metadata: Dictionary containing video information.
        
        Returns:
            Path to saved metadata file.
        """
        meta_path = self.meta_output_dir / "video_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(video_metadata, f, indent=2)
        
        logger.debug(f"Saved video metadata to {meta_path}")
        return meta_path
    
    def _process_single_chunk_video(
        self,
        prompts: List[str],
        video_metadata: Dict,
        propagation_direction: str
    ) -> List[Dict[str, Any]]:
        """
        Process video as a single chunk (no splitting needed).
        
        Args:
            prompts: List of text prompts.
            video_metadata: Video metadata dictionary.
            propagation_direction: Direction for mask propagation.
        
        Returns:
            List containing single chunk result.
        """
        from sam3.chunk_processor import ChunkProcessor
        
        logger.info("Processing entire video as single chunk...")
        
        # Create chunk info for the entire video
        chunk_info = {
            "chunk": 0,
            "start": 0,
            "end": video_metadata["nb_frames"] - 1
        }
        
        # Create chunk directory
        chunk_dir = self.chunks_temp_dir / "chunk_0"
        chunk_dir.mkdir(exist_ok=True)
        
        # Use original video directly (no need to extract)
        chunk_video_path = self.video_path
        
        logger.debug(f"  Chunk video path: {chunk_video_path}")
        
        # Initialize ChunkProcessor
        chunk_processor = ChunkProcessor(
            chunk_id=0,
            chunk_video_path=chunk_video_path,
            chunk_dir=chunk_dir,
            video_name=self.video_name,
            video_metadata=video_metadata,
            device=self.device,
            bpe_path=self.bpe_path,
            num_workers=self.num_workers
        )
        
        # Process chunk with all prompts
        chunk_result = chunk_processor.process_with_prompts(
            prompts=prompts,
            propagation_direction=propagation_direction
        )
        
        return [chunk_result]
    
    def _process_multiple_chunks(
        self,
        prompts: List[str],
        video_metadata: Dict,
        video_chunks: List[Dict],
        propagation_direction: str
    ) -> List[Dict[str, Any]]:
        """
        Process video in multiple chunks.
        
        Args:
            prompts: List of text prompts.
            video_metadata: Video metadata dictionary.
            video_chunks: List of chunk specifications.
            propagation_direction: Direction for mask propagation.
        
        Returns:
            List of chunk results.
        """
        from sam3.chunk_processor import ChunkProcessor
        
        chunk_results = []
        
        for chunk_info in video_chunks:
            chunk_id = chunk_info["chunk"]
            start_frame = chunk_info["start"]
            end_frame = chunk_info["end"]
            
            logger.info(f"  Processing chunk {chunk_id + 1}/{len(video_chunks)}: "
                       f"frames {start_frame}-{end_frame}")
            
            # Create chunk directory
            chunk_dir = self.chunks_temp_dir / f"chunk_{chunk_id}"
            chunk_dir.mkdir(exist_ok=True)
            
            # Extract chunk from video
            chunk_video_path = chunk_dir / f"chunk_{chunk_id}.mp4"
            
            logger.debug(f"    Extracting chunk video to {chunk_video_path}")
            ffmpeg_lib.create_video_chunk(
                input_video=str(self.video_path),
                output_video=str(chunk_video_path),
                start_frame=start_frame,
                end_frame=end_frame
            )
            
            # Get shared driver (model loaded once, reused for all chunks)
            driver = self._get_video_driver()
            
            # Initialize ChunkProcessor with shared driver
            chunk_processor = ChunkProcessor(
                chunk_id=chunk_id,
                chunk_video_path=chunk_video_path,
                chunk_dir=chunk_dir,
                video_name=self.video_name,
                video_metadata=video_metadata,
                driver=driver
            )
            
            # Process chunk with all prompts
            chunk_result = chunk_processor.process_with_prompts(
                prompts=prompts,
                propagation_direction=propagation_direction
            )
            
            chunk_results.append(chunk_result)
        
        return chunk_results
    
    def _postprocess_results(
        self,
        chunk_results: List[Dict[str, Any]],
        prompts: List[str]
    ):
        """
        Post-process results from all chunks.
        
        Handles:
        - Matching objects across chunk boundaries using IoU
        - Building ID mappings across all chunks
        - Stitching masks together with continuous frame numbering
        
        Args:
            chunk_results: List of results from each chunk.
            prompts: List of processed prompts.
        """
        logger.info("  Post-processing results...")
        
        if not chunk_results:
            logger.warning("    No chunk results to post-process")
            return
        
        # Load video metadata
        metadata_path = self.meta_output_dir / "video_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                video_metadata = json.load(f)
        else:
            logger.warning("    Video metadata not found, using defaults")
            video_metadata = {}
        
        # Initialize post-processor
        postprocessor = VideoPostProcessor(
            video_name=self.video_name,
            chunk_results=chunk_results,
            video_metadata=video_metadata,
            chunks_temp_dir=self.chunks_temp_dir,
            masks_output_dir=self.video_output_dir / "masks",
            meta_output_dir=self.video_output_dir / "metadata"
        )
        
        # Run post-processing
        postprocessor.process(prompts)
        
        logger.info("  Post-processing complete")
    
    def _visualize_results(
        self,
        chunk_results: List[Dict[str, Any]],
        prompts: List[str]
    ):
        """
        Create visualization overlays for the results.
        
        This is a placeholder for future implementation. Will handle:
        - Overlaying masks on original video
        - Color-coding different objects/prompts
        - Creating comparison videos
        
        Args:
            chunk_results: List of results from each chunk.
            prompts: List of processed prompts.
        """
        logger.info("  [PLACEHOLDER] Visualization not yet implemented")
        logger.debug(f"    Would visualize {len(chunk_results)} chunk(s) for {len(prompts)} prompt(s)")
        
        # TODO: Implement visualization logic
        # - Load original video
        # - Apply masks as overlays
        # - Save visualization video
        
        pass
    
    def _move_temp_to_output(self):
        """
        Move temporary files to output directory for preservation.
        
        Copies entire temp directory structure to output directory,
        then cleans up the temp directory.
        """
        logger.info("  Moving temporary files to output directory...")
        
        # Destination in output
        temp_preserved_dir = self.video_output_dir / "temp_files"
        
        # Copy temp directory to output
        if self.video_temp_dir.exists():
            shutil.copytree(
                self.video_temp_dir,
                temp_preserved_dir,
                dirs_exist_ok=True
            )
            logger.debug(f"    Copied temp files to {temp_preserved_dir}")
        
        # Clean up original temp
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """
        Clean up temporary files.
        
        Removes the entire temp directory for this video.
        """
        logger.info("  Cleaning up temporary files...")
        
        if self.video_temp_dir.exists():
            shutil.rmtree(self.video_temp_dir)
            logger.debug(f"    Removed temp directory: {self.video_temp_dir}")
        else:
            logger.debug("    Temp directory does not exist, nothing to clean")
