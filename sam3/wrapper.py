"""
SAM3 Production Wrapper Module (Refactored)

Provides backward-compatible wrapper API that delegates to Sam3Entrypoint.
This module maintains the original wrapper interface while using the new
driver-based implementation.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from dataclasses import dataclass, asdict

from sam3.entrypoint import Sam3Entrypoint, ProcessingResult
from sam3.logger import get_logger

logger = get_logger(__name__)


# Re-export dataclasses for backward compatibility
@dataclass
class DeviceInfo:
    """Device information with compute and memory stats (deprecated, use entrypoint)"""
    device_type: str  # "cuda" or "cpu"
    device_name: str
    device_count: int
    total_memory_gb: float
    available_memory_gb: float
    memory_utilization_percent: float
    compute_capability: Optional[str] = None


@dataclass
class VideoMetadata:
    """Video file metadata (deprecated, use entrypoint)"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    bytes_per_frame: int


@dataclass
class ChunkInfo:
    """Video chunk information (deprecated, use entrypoint)"""
    chunk_id: int
    start_frame: int
    end_frame: int
    num_frames: int
    chunk_path: Optional[str] = None


class Sam3Wrapper:
    """
    Production wrapper for SAM3 (refactored to use Sam3Entrypoint).
    
    This class maintains backward compatibility with the original wrapper
    while delegating to the new entrypoint implementation.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        ram_usage_percent: float = 0.25,
        min_frames: int = 25,
        chunk_overlap: int = 1,
        prefetch_threshold: float = 0.90,
        tmp_base: str = "/tmp/sam3-cpu",
        verbose: bool = True
    ):
        """Initialize SAM3 wrapper with backward-compatible API.
        
        Args:
            config_path: Path to config.json (overrides other args)
            ram_usage_percent: Fraction of available RAM to use (default: 0.25)
            min_frames: Minimum frames required for processing (default: 25)
            chunk_overlap: Frame overlap between chunks (default: 1)
            prefetch_threshold: Start prefetching next chunk at this progress (default: 0.90)
            tmp_base: Base directory for temporary files (default: /tmp/sam3-cpu)
            verbose: Print detailed logs (default: True)
        """
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            ram_usage_percent = config.get("ram_usage_percent", ram_usage_percent)
            min_frames = config.get("min_frames", min_frames)
            chunk_overlap = config.get("chunk_overlap", chunk_overlap)
            prefetch_threshold = config.get("prefetch_threshold", prefetch_threshold)
            tmp_base = config.get("tmp_base", tmp_base)
            verbose = config.get("verbose", verbose)
        
        # Store params
        self.ram_usage_percent = ram_usage_percent
        self.min_frames = min_frames
        self.chunk_overlap = chunk_overlap
        self.prefetch_threshold = prefetch_threshold
        self.tmp_base = tmp_base
        self.verbose = verbose
        
        # Initialize entrypoint with compatible settings
        self.entrypoint = Sam3Entrypoint(
            default_output_dir=tmp_base,
            min_video_frames=min_frames,
            min_chunk_overlap=chunk_overlap
        )
        
        # Track workspace
        self.current_video_dir = None
        self.current_chunks = None
        self.device_info = None
        
        if verbose:
            logger.info(f"Sam3Wrapper initialized (delegating to Sam3Entrypoint)")
            logger.info(f"  RAM usage: {ram_usage_percent*100:.0f}%")
            logger.info(f"  Min frames: {min_frames}")
            logger.info(f"  Chunk overlap: {chunk_overlap}")
            logger.info(f"  Workspace: {tmp_base}")
    
    def _detect_device(self) -> DeviceInfo:
        """Detect device information (uses entrypoint memory detection)."""
        mem_info = self.entrypoint._get_memory_info()
        
        # Convert to legacy DeviceInfo format
        if mem_info.gpu_available:
            device_type = "cuda"
            total_mem = mem_info.gpu_total
            available_mem = mem_info.gpu_available
        else:
            device_type = "cpu"
            total_mem = mem_info.cpu_total
            available_mem = mem_info.cpu_available
        
        utilization = 100 * (1 - available_mem / total_mem) if total_mem > 0 else 0
        
        return DeviceInfo(
            device_type=device_type,
            device_name=device_type.upper(),
            device_count=1 if mem_info.gpu_available else 0,
            total_memory_gb=total_mem,
            available_memory_gb=available_mem,
            memory_utilization_percent=utilization,
            compute_capability=None
        )
    
    def _print_device_info(self):
        """Print detected device information."""
        if self.device_info is None:
            self.device_info = self._detect_device()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Device Information")
            print(f"{'='*60}")
            print(f"Device Type   : {self.device_info.device_type.upper()}")
            print(f"Total Memory  : {self.device_info.total_memory_gb:.2f} GB")
            print(f"Available     : {self.device_info.available_memory_gb:.2f} GB")
            print(f"Utilization   : {self.device_info.memory_utilization_percent:.1f}%")
            print(f"{'='*60}\n")
    
    def prepare_video(self, video_path: str) -> Dict[str, Any]:
        """Prepare video for processing (delegates to entrypoint).
        
        Args:
            video_path: Path to video file
        
        Returns:
            dict with keys: 'video_meta', 'chunks', 'workspace', 'requirements_check'
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video metadata
        video_meta_new = self.entrypoint._get_video_metadata(video_path)
        
        # Convert to legacy VideoMetadata format
        video_meta = VideoMetadata(
            path=video_path,
            width=video_meta_new.width,
            height=video_meta_new.height,
            fps=video_meta_new.fps,
            total_frames=video_meta_new.total_frames,
            duration_seconds=video_meta_new.duration,
            bytes_per_frame=video_meta_new.width * video_meta_new.height * 3  # RGB
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Video Analysis: {Path(video_path).name}")
            print(f"{'='*60}")
            print(f"Resolution    : {video_meta.width}x{video_meta.height}")
            print(f"FPS           : {video_meta.fps:.2f}")
            print(f"Duration      : {video_meta.duration_seconds:.2f} seconds")
            print(f"Total Frames  : {video_meta.total_frames}")
            print(f"Bytes/Frame   : {video_meta.bytes_per_frame:,}")
        
        # Check feasibility
        memory_needed = self.entrypoint._calculate_memory_needed(
            video_meta.width, video_meta.height, video_meta.total_frames, is_video=True
        )
        feasibility = self.entrypoint._check_feasibility(
            memory_needed, video_meta.total_frames, is_video=True
        )
        
        # Convert to legacy format
        mem_info = self.entrypoint._get_memory_info()
        requirements = {
            'adequate': feasibility.can_process or feasibility.requires_chunking,
            'total_memory_needed_gb': memory_needed,
            'usable_memory_gb': mem_info.cpu_available if not mem_info.gpu_available else mem_info.gpu_available,
            'max_chunk_size': feasibility.chunk_size or video_meta.total_frames,
            'warnings': []
        }
        
        if not feasibility.can_process:
            requirements['warnings'].append(f"{feasibility.message}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"RAM Requirements")
            print(f"{'='*60}")
            print(f"Total Needed  : {requirements['total_memory_needed_gb']:.2f} GB")
            print(f"Usable RAM    : {requirements['usable_memory_gb']:.2f} GB")
            print(f"Max Chunk Size: {requirements['max_chunk_size']} frames")
            
            if requirements['warnings']:
                print(f"\nWarnings:")
                for warning in requirements['warnings']:
                    print(f"  ⚠ {warning}")
        
        # Generate chunks if needed
        if feasibility.requires_chunking:
            chunks_new = self.entrypoint._generate_chunks(
                video_meta_new,
                feasibility.chunk_size,
                self.chunk_overlap
            )
            chunks = [
                ChunkInfo(
                    chunk_id=i,
                    start_frame=c.start_frame,
                    end_frame=c.end_frame,
                    num_frames=c.num_frames,
                    chunk_path=None
                )
                for i, c in enumerate(chunks_new)
            ]
        else:
            chunks = [
                ChunkInfo(
                    chunk_id=0,
                    start_frame=0,
                    end_frame=video_meta.total_frames - 1,
                    num_frames=video_meta.total_frames,
                    chunk_path=None
                )
            ]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Chunking Plan")
            print(f"{'='*60}")
            print(f"Total Chunks  : {len(chunks)}")
            print(f"Chunk Overlap : {self.chunk_overlap} frames")
            if len(chunks) > 0:
                print(f"First Chunk   : frames {chunks[0].start_frame}-{chunks[0].end_frame}")
                print(f"Last Chunk    : frames {chunks[-1].start_frame}-{chunks[-1].end_frame}")
        
        # Setup workspace
        workspace_base = Path(self.tmp_base)
        workspace = workspace_base / Path(video_path).stem
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "metadata").mkdir(exist_ok=True)
        
        self.current_video_dir = workspace
        self.current_chunks = chunks
        
        # Save metadata
        metadata_path = workspace / "metadata" / "video_info.json"
        metadata_dict = {
            'video_meta': asdict(video_meta),
            'chunks': [asdict(c) for c in chunks],
            'requirements': requirements
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Workspace: {workspace}")
            print(f"Metadata : {metadata_path}")
            print(f"{'='*60}\n")
        
        return {
            'video_meta': video_meta,
            'chunks': chunks,
            'workspace': workspace,
            'requirements_check': requirements,
            'metadata_path': metadata_path
        }
    
    def cleanup_video_workspace(self, video_path: Optional[str] = None):
        """Clean up temporary workspace for video."""
        if video_path:
            video_name = Path(video_path).stem
            workspace = Path(self.tmp_base) / video_name
        else:
            workspace = self.current_video_dir
        
        if workspace and workspace.exists():
            import shutil
            shutil.rmtree(workspace)
            if self.verbose:
                logger.info(f"Cleaned up workspace: {workspace}")
    
    def cleanup_all_workspaces(self):
        """Clean up all temporary workspaces."""
        workspace_base = Path(self.tmp_base)
        if workspace_base.exists():
            import shutil
            shutil.rmtree(workspace_base)
            workspace_base.mkdir(parents=True, exist_ok=True)
            if self.verbose:
                logger.info(f"Cleaned up all workspaces in: {workspace_base}")
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information."""
        if self.device_info is None:
            self.device_info = self._detect_device()
        return self.device_info
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace information."""
        return {
            'tmp_base': self.tmp_base,
            'current_video_dir': str(self.current_video_dir) if self.current_video_dir else None,
            'num_chunks': len(self.current_chunks) if self.current_chunks else 0
        }
    
    # ========================================================================
    # New Methods: Delegate to Entrypoint for Full Processing
    # ========================================================================
    
    def process_image(
        self,
        image_path: str,
        prompts: Union[str, List[str]] = None,
        boxes: List[List[float]] = None,
        output_dir: Optional[str] = None
    ) -> ProcessingResult:
        """Process image with prompts or boxes (delegates to entrypoint)."""
        return self.entrypoint.process_image(
            image_path=image_path,
            prompts=prompts,
            boxes=boxes,
            output_dir=output_dir or self.tmp_base
        )
    
    def process_video_with_prompts(
        self,
        video_path: str,
        prompts: Union[str, List[str]],
        output_dir: Optional[str] = None,
        propagation_direction: str = "forward"
    ) -> ProcessingResult:
        """Process video with text prompts (delegates to entrypoint)."""
        return self.entrypoint.process_video_with_prompts(
            video_path=video_path,
            prompts=prompts,
            output_dir=output_dir or self.tmp_base,
            propagation_direction=propagation_direction
        )
    
    def process_video_with_points(
        self,
        video_path: str,
        frame_idx: int,
        points: List[List[float]],
        point_labels: List[int],
        object_id: int = 1,
        output_dir: Optional[str] = None,
        propagation_direction: str = "forward"
    ) -> ProcessingResult:
        """Process video with point prompts (delegates to entrypoint)."""
        return self.entrypoint.process_video_with_points(
            video_path=video_path,
            frame_idx=frame_idx,
            points=points,
            point_labels=point_labels,
            object_id=object_id,
            output_dir=output_dir or self.tmp_base,
            propagation_direction=propagation_direction
        )
