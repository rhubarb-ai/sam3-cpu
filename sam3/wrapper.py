"""
SAM3 Production Wrapper Module

Provides intelligent memory management, device detection, video chunking,
and RAM-aware processing for SAM3 image and video inference.
"""

import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import cv2
import torch
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class DeviceInfo:
    """Device information with compute and memory stats"""
    device_type: str  # "cuda" or "cpu"
    device_name: str
    device_count: int
    total_memory_gb: float
    available_memory_gb: float
    memory_utilization_percent: float
    compute_capability: Optional[str] = None


@dataclass
class VideoMetadata:
    """Video file metadata"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    bytes_per_frame: int


@dataclass
class ChunkInfo:
    """Video chunk information"""
    chunk_id: int
    start_frame: int
    end_frame: int
    num_frames: int
    chunk_path: Optional[str] = None


class Sam3Wrapper:
    """
    Production wrapper for SAM3 with intelligent memory management,
    device detection, and video chunking.
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
        """
        Initialize SAM3 wrapper.
        
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
        
        self.ram_usage_percent = ram_usage_percent
        self.min_frames = min_frames
        self.chunk_overlap = chunk_overlap
        self.prefetch_threshold = prefetch_threshold
        self.tmp_base = tmp_base
        self.verbose = verbose
        
        # Detect device
        self.device_info = self._detect_device()
        if self.verbose:
            self._print_device_info()
        
        # Initialize predictor as None (lazy loading)
        self.predictor = None
        self.current_video_dir: Optional[Path] = None
        self.current_chunks: List[ChunkInfo] = []
    
    def _detect_device(self) -> DeviceInfo:
        """Detect available compute device and gather stats"""
        if torch.cuda.is_available():
            device_type = "cuda"
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            
            # Get VRAM info
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)  # GB
            
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            available_memory = total_memory - reserved
            utilization = (reserved / total_memory) * 100
            
            compute_capability = f"{props.major}.{props.minor}"
            
            return DeviceInfo(
                device_type=device_type,
                device_name=device_name,
                device_count=device_count,
                total_memory_gb=total_memory,
                available_memory_gb=available_memory,
                memory_utilization_percent=utilization,
                compute_capability=compute_capability
            )
        else:
            device_type = "cpu"
            device_count = os.cpu_count() or 1
            
            # Get RAM info from /proc/meminfo
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                
                mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) / (1024**2)  # GB
                mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) / (1024**2)  # GB
                utilization = ((mem_total - mem_available) / mem_total) * 100
                
            except (FileNotFoundError, IndexError):
                # Fallback for non-Linux systems
                import psutil
                mem = psutil.virtual_memory()
                mem_total = mem.total / (1024**3)
                mem_available = mem.available / (1024**3)
                utilization = mem.percent
            
            return DeviceInfo(
                device_type=device_type,
                device_name="CPU",
                device_count=device_count,
                total_memory_gb=mem_total,
                available_memory_gb=mem_available,
                memory_utilization_percent=utilization
            )
    
    def _print_device_info(self):
        """Print formatted device information"""
        info = self.device_info
        print("=" * 60)
        print(f"SAM3 Device Information")
        print("=" * 60)
        print(f"Device Type       : {info.device_type.upper()}")
        print(f"Device Name       : {info.device_name}")
        print(f"Device Count      : {info.device_count}")
        print(f"Total Memory      : {info.total_memory_gb:.2f} GB")
        print(f"Available Memory  : {info.available_memory_gb:.2f} GB")
        print(f"Memory Usage      : {info.memory_utilization_percent:.1f}%")
        if info.compute_capability:
            print(f"Compute Capability: {info.compute_capability}")
        print("=" * 60)
    
    def _get_video_metadata(self, video_path: str) -> VideoMetadata:
        """Extract video metadata using ffprobe"""
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
            bytes_per_frame = width * height * 3  # CV_8UC3 format
            
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
    
    def _calculate_ram_safe_chunk_size(self, video_meta: VideoMetadata) -> int:
        """Calculate maximum frames that fit in RAM based on ram_usage_percent"""
        available_bytes = self.device_info.available_memory_gb * (1024**3)
        usable_bytes = int(available_bytes * self.ram_usage_percent)
        max_frames = usable_bytes // video_meta.bytes_per_frame
        return max(self.min_frames, max_frames)
    
    def _check_video_requirements(self, video_meta: VideoMetadata) -> Dict[str, any]:
        """
        Check if video meets minimum frame requirements and fits in RAM.
        
        Returns:
            dict with keys: 'adequate', 'warnings', 'max_chunk_size'
        """
        max_chunk_size = self._calculate_ram_safe_chunk_size(video_meta)
        warnings = []
        
        # Check minimum frames
        if video_meta.total_frames < self.min_frames:
            warnings.append(
                f"Video has {video_meta.total_frames} frames, "
                f"but minimum is {self.min_frames} frames"
            )
        
        # Check if entire video fits in RAM
        total_memory_needed_gb = (video_meta.total_frames * video_meta.bytes_per_frame) / (1024**3)
        usable_memory_gb = self.device_info.available_memory_gb * self.ram_usage_percent
        
        if total_memory_needed_gb > usable_memory_gb:
            num_chunks = (video_meta.total_frames + max_chunk_size - 1) // max_chunk_size
            warnings.append(
                f"Video requires {total_memory_needed_gb:.2f} GB but only "
                f"{usable_memory_gb:.2f} GB available. "
                f"Will process in {num_chunks} chunks."
            )
        
        adequate = len(warnings) == 0 or video_meta.total_frames >= self.min_frames
        
        return {
            'adequate': adequate,
            'warnings': warnings,
            'max_chunk_size': max_chunk_size,
            'total_memory_needed_gb': total_memory_needed_gb,
            'usable_memory_gb': usable_memory_gb
        }
    
    def _generate_chunks(self, video_meta: VideoMetadata, chunk_size: int) -> List[ChunkInfo]:
        """Generate chunk information with overlap"""
        chunks = []
        stride = chunk_size - self.chunk_overlap
        
        if stride <= 0:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        
        chunk_id = 0
        start_frame = 0
        
        while start_frame < video_meta.total_frames:
            end_frame = min(start_frame + chunk_size - 1, video_meta.total_frames - 1)
            num_frames = end_frame - start_frame + 1
            
            chunks.append(ChunkInfo(
                chunk_id=chunk_id,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames
            ))
            
            chunk_id += 1
            start_frame += stride
            
            # Break if we've reached the end
            if end_frame >= video_meta.total_frames - 1:
                break
        
        return chunks
    
    def _setup_video_workspace(self, video_path: str) -> Path:
        """Create /tmp/sam3-cpu/<video_name>/ structure"""
        video_name = Path(video_path).stem
        video_dir = Path(self.tmp_base) / video_name
        
        # Create directories
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "chunks").mkdir(exist_ok=True)
        (video_dir / "metadata").mkdir(exist_ok=True)
        (video_dir / "results").mkdir(exist_ok=True)
        
        return video_dir
    
    def _save_chunk_metadata(self, chunks: List[ChunkInfo], video_meta: VideoMetadata, output_path: Path):
        """Save chunk metadata to JSON file"""
        metadata = {
            "video": {
                "path": video_meta.path,
                "width": video_meta.width,
                "height": video_meta.height,
                "fps": video_meta.fps,
                "total_frames": video_meta.total_frames,
                "duration_seconds": video_meta.duration_seconds
            },
            "chunking": {
                "ram_usage_percent": self.ram_usage_percent,
                "chunk_overlap": self.chunk_overlap,
                "num_chunks": len(chunks)
            },
            "chunks": [asdict(chunk) for chunk in chunks]
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def prepare_video(self, video_path: str) -> Dict[str, any]:
        """
        Prepare video for processing: analyze, chunk, and setup workspace.
        
        Args:
            video_path: Path to video file
        
        Returns:
            dict with keys: 'video_meta', 'chunks', 'workspace', 'requirements_check'
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get video metadata
        video_meta = self._get_video_metadata(video_path)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Video Analysis: {Path(video_path).name}")
            print(f"{'='*60}")
            print(f"Resolution    : {video_meta.width}x{video_meta.height}")
            print(f"FPS           : {video_meta.fps:.2f}")
            print(f"Duration      : {video_meta.duration_seconds:.2f} seconds")
            print(f"Total Frames  : {video_meta.total_frames}")
            print(f"Bytes/Frame   : {video_meta.bytes_per_frame:,}")
        
        # Check requirements
        requirements = self._check_video_requirements(video_meta)
        
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
        
        if not requirements['adequate']:
            raise ValueError(
                f"Video does not meet minimum requirements. "
                f"Minimum frames: {self.min_frames}"
            )
        
        # Generate chunks
        chunks = self._generate_chunks(video_meta, requirements['max_chunk_size'])
        
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
        workspace = self._setup_video_workspace(video_path)
        self.current_video_dir = workspace
        self.current_chunks = chunks
        
        # Save metadata
        metadata_path = workspace / "metadata" / "chunks.json"
        self._save_chunk_metadata(chunks, video_meta, metadata_path)
        
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
        """
        Clean up temporary workspace for video.
        
        Args:
            video_path: Path to video file (if None, uses current workspace)
        """
        if video_path:
            video_name = Path(video_path).stem
            workspace = Path(self.tmp_base) / video_name
        else:
            workspace = self.current_video_dir
        
        if workspace and workspace.exists():
            shutil.rmtree(workspace)
            if self.verbose:
                print(f"Cleaned up workspace: {workspace}")
    
    def cleanup_all_workspaces(self):
        """Clean up all temporary workspaces in /tmp/sam3-cpu"""
        base_path = Path(self.tmp_base)
        if base_path.exists():
            shutil.rmtree(base_path)
            if self.verbose:
                print(f"Cleaned up all workspaces: {base_path}")
    
    def load_predictor(self, bpe_path: Optional[str] = None, num_workers: Optional[int] = None):
        """
        Load SAM3 predictor (lazy loading).
        
        Args:
            bpe_path: Path to BPE tokenizer file (default: assets/bpe_simple_vocab_16e6.txt.gz)
            num_workers: Number of workers for CPU multi-process (None = 1 for CPU, auto for GPU)
        """
        from sam3.model_builder import (
            build_sam3_video_predictor_cpu,
            build_sam3_video_predictor
        )
        import sam3
        
        # Set default bpe_path if not provided
        if bpe_path is None:
            sam3_root = os.path.dirname(sam3.__file__)
            bpe_path = os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")
        
        if num_workers is None:
            num_workers = 1 if self.device_info.device_type == "cpu" else self.device_info.device_count
        
        if self.verbose:
            print(f"Loading SAM3 predictor...")
            print(f"  Device     : {self.device_info.device_type}")
            print(f"  Workers    : {num_workers}")
            print(f"  BPE Path   : {bpe_path}")
        
        if self.device_info.device_type == "cpu":
            # build_sam3_video_predictor_cpu handles both single and multi-worker
            self.predictor = build_sam3_video_predictor_cpu(
                bpe_path=bpe_path,
                num_workers=num_workers
            )
        else:
            # build_sam3_video_predictor handles GPU
            self.predictor = build_sam3_video_predictor(
                bpe_path=bpe_path,
                gpus_to_use=list(range(num_workers)) if num_workers > 1 else None
            )
        
        if self.verbose:
            print("✓ Predictor loaded successfully\n")
    
    def get_device_info(self) -> DeviceInfo:
        """Get current device information"""
        return self.device_info
    
    def get_workspace_info(self) -> Dict[str, any]:
        """Get current video workspace information"""
        if self.current_video_dir is None:
            return {"workspace": None, "chunks": []}
        
        return {
            "workspace": str(self.current_video_dir),
            "chunks": [asdict(chunk) for chunk in self.current_chunks]
        }


def create_default_config(output_path: str = "config.json"):
    """Create default configuration file"""
    config = {
        "ram_usage_percent": 0.25,
        "min_frames": 25,
        "chunk_overlap": 1,
        "prefetch_threshold": 0.90,
        "tmp_base": "/tmp/sam3-cpu",
        "verbose": True,
        "bpe_path": None
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created default config: {output_path}")


if __name__ == "__main__":
    # Example usage
    wrapper = Sam3Wrapper(verbose=True)
    print("\nDevice detected successfully!")
    print(f"Device: {wrapper.device_info.device_type.upper()}")
    print(f"Available Memory: {wrapper.device_info.available_memory_gb:.2f} GB")
