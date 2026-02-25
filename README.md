# SAM3 CPU - Segment Anything Model 3 (CPU Compatible)

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/rhubarb-ai/sam3-cpu/pulls)

**SAM3 CPU** is a production-ready implementation of Meta's Segment Anything Model 3 (SAM3) with **full CPU compatibility** while preserving existing CUDA support. This project provides a clean, high-level API for both image and video segmentation tasks.

## 🎯 Key Features

- ✅ **Dual Device Support**: Seamlessly runs on CPU or GPU (CUDA)
- ✅ **9 Segmentation Scenarios**: Comprehensive API covering all common use cases
- ✅ **Intelligent Memory Management**: Automatic chunking for large videos
- ✅ **Frame/Time Range Extraction**: Process specific video segments (frames, seconds, or timestamps)
- ✅ **Bidirectional Tracking**: Default propagation in both directions for better accuracy
- ✅ **Production Ready**: Full test coverage, examples, and documentation
- ✅ **Simple API**: Unified interface with `from sam3 import Sam3`
- ✅ **Batch Processing**: Handle multiple images/videos efficiently
- ✅ **Auto-Cleanup**: Temporary files managed automatically

## 📋 Table of Contents

- [Installation](#installation)
- [Platform Support](#platform-support)
- [Quick Start](#quick-start)
- [Advanced Video Features](#advanced-video-features)
  - [Automatic Video Chunking](#automatic-video-chunking)
  - [Cross-Chunk Mask Injection](#cross-chunk-mask-injection)
  - [Frame/Time Range Extraction](#frametime-range-extraction)
  - [Default Propagation Direction](#default-propagation-direction)
- [Usage Examples](#usage-examples)
- [Quick Testing](#quick-testing)
- [Configuration & Globals](#configuration--globals)
- [Memory Management](#memory-management)
- [Performance Profiling](#performance-profiling)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Credits](#credits)
- [License](#license)

## 🚀 Installation

### Platform Support

✅ **Linux** (Ubuntu 20.04+, Debian 11+, RHEL 8+, CentOS 8+)
✅ **macOS** (macOS 11 Big Sur+, both Intel and Apple Silicon)
✅ **CPU** (All platforms)
✅ **GPU** (NVIDIA CUDA on Linux and Windows)

**Note:** The setup script and Makefile are designed to work seamlessly on both Linux and macOS.

### Prerequisites

- **Python 3.12+**
- **ffmpeg/ffprobe** (for video processing)
- **uv** package manager (recommended)

### Quick Setup

We provide an automated setup script that installs all dependencies:

```bash
# Clone the repository
git clone https://github.com/rhubarb-ai/sam3-cpu.git
cd sam3-cpu

# Run automated setup
chmod +x setup.sh
./setup.sh

# Or use Make
make setup
```

### Manual Installation

If you prefer manual installation:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv pip install -e .

# Verify installation
python -c "from sam3 import Sam3; print('✓ SAM3 installed successfully')"
```

### System Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-dev build-essential
```

**Linux (RHEL/CentOS/Fedora):**
```bash
sudo yum install -y ffmpeg python3-devel gcc gcc-c++ make
# or for Fedora
sudo dnf install -y ffmpeg python3-devel gcc gcc-c++ make
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg
```

**Note for macOS Apple Silicon (M1/M2/M3):**
The project works natively on Apple Silicon. No Rosetta 2 required.

## ⚡ Quick Start

### Basic Image Segmentation

```python
from sam3 import Sam3

# Initialize SAM3
sam3 = Sam3(verbose=True)

# Segment objects in an image using text prompts
result = sam3.process_image(
    image_path="assets/images/truck.jpg",
    prompts=["truck", "wheel"],
    output_dir="outputs/demo"
)

print(f"✓ Generated {len(result.mask_files)} masks")
print(f"Object IDs: {result.object_ids}")
```

### Basic Video Segmentation

```python
from sam3 import Sam3

# Initialize SAM3
sam3 = Sam3(verbose=True)

# Segment and track objects in video
result = sam3.process_video_with_prompts(
    video_path="assets/videos/tennis_480p.mp4",
    prompts=["person", "tennis racket"],
    propagation_direction="forward"
)

print(f"✓ Generated {len(result.mask_files)} mask videos")
```

### Using the Makefile (Recommended)

```bash
# Run all examples
make run-all

# Run specific example
make run-example EXAMPLE=a

# Run with different video resolution
make run-all VIDEO_RES=720p

# Run tests
make test
```

## 🎥 Advanced Video Features

### Automatic Video Chunking

SAM3 automatically handles large videos that don't fit in memory by intelligently splitting them into chunks:

```python
from sam3 import Sam3

sam3 = Sam3(verbose=True)

# Process large video - automatically chunked if needed
result = sam3.process_video_with_prompts(
    video_path="large_video.mp4",
    prompts=["person"],
    propagation_direction="both"  # Default: bidirectional tracking
)

# Chunking happens automatically based on available RAM/VRAM
# - CPU: Uses 25% of available RAM by default
# - GPU: Uses 95% of available VRAM
# - Overlap: 1-frame overlap between chunks (configurable)
# - Cleanup: Temporary chunk files are automatically removed
```

**How It Works:**
1. **Memory Calculation**: Estimates memory needed based on video resolution and frames
2. **Auto-Chunking**: If memory insufficient, splits video into RAM-safe chunks
3. **Overlap Handling**: Processes chunks with overlap to ensure continuity
4. **Smart Merging**: Merges results intelligently, avoiding duplicate frames from overlaps

**Example Output:**
```
============================================================
📊 Video Chunking Details
============================================================
Video: Roger-Federer-vs-Rafael-Nadal-Wimbledon-2008_1080p
Resolution: 1920x1080
FPS: 25.00
Total Frames: 6364
Available RAM: 55.48 GB
Memory Needed: 43.61 GB
RAM Usage Percent: 33%
RAM-safe Frames per Chunk: 1684
Overlap: 1 frame(s)
Stride: 1683 frames
Number of Chunks: 4
============================================================
Processing chunk 1/4 (frames 0-1683)...
```

### Cross-Chunk Mask Injection

When a video is split into chunks, objects can be lost at chunk boundaries because each chunk is processed independently. **Cross-chunk mask injection** solves this by carrying tracked object masks from the end of chunk N into the start of chunk N+1:

```python
from sam3 import Sam3API

api = Sam3API()

# Automatic: mask injection happens transparently during multi-chunk processing
result = api.process_video_with_prompts(
    video_path="long_video.mp4",
    prompts=["player", "ball"],
    keep_temp_files=True  # inspect chunk masks
)
api.cleanup()
```

**How It Works:**
1. **Chunk 0**: Detects objects via text prompt, propagates tracker, saves masks
2. **Chunk 1+**: Injects previous chunk's last-frame masks on frame 0 via `tracker.add_new_mask()`, then runs the text prompt to detect any *new* objects, then propagates both injected + new objects
3. **Post-processing**: Injected objects are matched deterministically (same IDs), while newly detected objects fall back to IoU-based matching

**Processing Pre-Split Chunks:**

If you already have separate chunk files, use the driver-level API directly with `test_chunks_injection.py`:

```bash
# Process pre-split chunks with mask injection between them
python3 test_chunks_injection.py \
    --chunks-dir assets/videos/private/my_chunks \
    --prompts player ball \
    --output results/my_injection_test
```

**Example Output:**
```
Chunk 0 (chunk_000.mp4): 3 objs in 25 frames, new=[0, 1, 3]
Chunk 1 (chunk_001.mp4): 4 objs in 25 frames, injected=[0, 1, 3], new=[2]
Chunk 2 (chunk_002.mp4): 4 objs in 25 frames, injected=[0, 1, 2, 3], new=[4, 5]
Chunk 3 (chunk_003.mp4): 6 objs in 25 frames, injected=[0, 3, 4, 5], new=[1, 2]
Chunk 4 (chunk_004.mp4): 6 objs in 25 frames, injected=[0, 1, 2, 3, 4, 5]
```

**Benefits:**
- ✅ **Seamless continuity**: Objects tracked across chunk boundaries without re-detection
- ✅ **Deterministic matching**: Injected objects keep their IDs (no IoU guessing)
- ✅ **Additive**: New objects in later chunks are still detected and tracked
- ✅ **Transparent**: Works automatically in the standard `process_video_with_prompts` pipeline

> For full technical details see [docs/local/CROSS_CHUNK_MASK_INJECTION.md](docs/local/CROSS_CHUNK_MASK_INJECTION.md)

### Frame/Time Range Extraction

Process only specific portions of a video without loading the entire file:

```python
from sam3 import Sam3

sam3 = Sam3(verbose=True)

# ✅ Using frame numbers
result = sam3.process_video_with_prompts(
    video_path="long_video.mp4",
    prompts=["car"],
    frame_from=100,
    frame_to=500,
    propagation_direction="both"
)

# ✅ Using seconds (float)
result = sam3.process_video_with_prompts(
    video_path="long_video.mp4",
    prompts=["person"],
    frame_from=10.5,   # 10.5 seconds
    frame_to=45.0,     # 45 seconds
)

# ✅ Using timestamps (MM:SS or HH:MM:SS)
result = sam3.process_video_with_prompts(
    video_path="long_video.mp4",
    prompts=["ball"],
    frame_from="1:30",  # 1 minute 30 seconds
    frame_to="3:45"     # 3 minutes 45 seconds
)

# ✅ Mix formats (auto-detected)
result = sam3.process_video_with_prompts(
    video_path="sports.mp4",
    prompts=["player"],
    frame_from=0,       # Frame 0
    frame_to="2:00"     # 2 minutes
)
```

**Processing Flow:**
1. **Extract Range**: Uses ffmpeg to extract specified segment (fast, no re-encoding)
2. **Apply Chunking**: If extracted segment is still large, auto-chunking applies
3. **Process**: Runs segmentation only on the extracted frames
4. **Cleanup**: Temporary segment file is automatically removed

**Benefits:**
- ⚡ **Faster**: Process only relevant portions
- 💾 **Memory Efficient**: Don't load entire 2-hour video for 30-second clip
- 🎯 **Precise**: Extract exact moments of interest
- 🔧 **Flexible**: Three input formats (frames, seconds, timestamps)

**Command Line Usage:**
```bash
# Using frame numbers
python examples/example_e_video_with_prompts.py \
    --video long_video.mp4 \
    --prompts "person" \
    --frame-from 0 \
    --frame-to 1000

# Using timestamps
python examples/example_e_video_with_prompts.py \
    --video sports.mp4 \
    --prompts "player" "ball" \
    --frame-from "1:30" \
    --frame-to "5:00" \
    --direction both
```

### Default Propagation Direction

SAM3 now defaults to **bidirectional tracking** (`direction="both"`) for better accuracy:

```python
# Uses bidirectional tracking by default
result = sam3.process_video_with_prompts(
    video_path="video.mp4",
    prompts=["person"]
    # propagation_direction defaults to "both"
)

# Override if needed
result = sam3.process_video_with_prompts(
    video_path="video.mp4",
    prompts=["car"],
    propagation_direction="forward"  # Forward only
)
```

**Why Bidirectional?**
- ✅ Better accuracy: Tracks from prompt frame in both directions
- ✅ Handles occlusions: Objects that disappear and reappear
- ✅ Temporal consistency: Smoother masks across frames

## 📚 Usage Examples

SAM3 supports 9 different segmentation scenarios:

### Scenario A: Single Image + Text Prompts

```bash
# Command line
make run-a

# Python
python examples/example_a_image_with_prompts.py \
    --image assets/images/truck.jpg \
    --prompts truck wheel \
    --output outputs/example_a
```

### Scenario B: Single Image + Bounding Boxes

```bash
# Command line
make run-b

# Python
python examples/example_b_image_with_boxes.py \
    --image assets/images/cafe.png \
    --output outputs/example_b
```

### Scenario C: Batch Images + Text Prompts

```bash
# Command line
make run-c

# Python
python examples/example_c_batch_images_with_prompts.py \
    --images assets/images/truck.jpg assets/images/groceries.jpg \
    --prompts object item \
    --output outputs/example_c
```

### Scenario D: Batch Images + Bounding Boxes

```bash
make run-d
```

### Scenario E: Video + Text Prompts

```bash
# Default 480p
make run-e

# Use 720p video
make run-e VIDEO_RES=720p

# Custom arguments
python examples/example_e_video_with_prompts.py \
    --video assets/videos/tennis_480p.mp4 \
    --prompts person "tennis racket" \
    --direction forward \
    --output outputs/example_e
```

### Scenario F: Video + Point Prompts

```bash
make run-f

# Or with custom points
python examples/example_f_video_with_points.py \
    --video assets/videos/tennis_480p.mp4 \
    --frame 0 \
    --points 320 180 400 200 \
    --labels 1 1 \
    --object-id 1
```

### Scenario G: Refine Video Object

```bash
make run-g

# Refine object with additional points
python examples/example_g_refine_video_object.py \
    --video assets/videos/tennis_480p.mp4 \
    --frame 10 \
    --object-id 1 \
    --points 350 190 \
    --labels 1
```

### Scenario H: Remove Video Objects

```bash
make run-h

# Remove specific objects by ID
python examples/example_h_remove_video_objects.py \
    --video assets/videos/tennis_480p.mp4 \
    --object-ids 2 3
```

### Scenario I: Video + Segment-based Prompts

```bash
make run-i

# Process different video segments with different prompts
# See example_i_video_with_segments.py for segment configuration
```


## ⚡ Quick Testing

For rapid validation without long processing times, use 1-second test clips:

```bash
# Run all quick tests (~2-3 minutes on CPU)
python test_quick.py

# Or use the provided script that creates test clips
# It tests: basic prompts, consistency, multi-prompt, direction override
```

**What It Tests:**
- ✅ Basic video processing with prompts
- ✅ Frame range extraction
- ✅ Bidirectional propagation (default)
- ✅ Multi-prompt handling
- ✅ Direction override

**Example Quick Test Code:**
```python
# Create 1-second test clip
ffmpeg -ss 00:00:00 -i video.mp4 -t 1 -c:v libx264 -c:a copy test_clip.mp4

# Test basic functionality
result = sam3.process_video_with_prompts(
    video_path="test_clip.mp4",
    prompts=["person"],
    propagation_direction="both"  # Default
)
```

This validates all features quickly before running on full videos.

## � Performance Profiling

SAM3 includes built-in performance profiling to help you analyze execution time and identify bottlenecks.

### Enabling Profiling

Profiling is controlled via the `--profile` flag and can be enabled in multiple ways:

#### Method 1: Using Makefile (Recommended)

```bash
# Profile all examples
make profile-all

# Profile all examples with 720p video
make profile-all VIDEO_RES=720p

# Profile specific example
make profile-example EXAMPLE=e

# Profile specific scenario with custom args
make profile-example EXAMPLE=e ARGS="--prompts person car"
```

#### Method 2: Direct Python Execution

```bash
# Profile individual example
uv run python examples/example_a_image_with_prompts.py --profile

# Profile with the grouped runner
uv run python examples/run_all_examples.py --profile --video-resolution 480p
```

#### Method 3: In Python Code

```python
import sys

# Add --profile to sys.argv before importing sam3
sys.argv.append('--profile')

from sam3 import Sam3

# Now all operations will be profiled
sam3 = Sam3(verbose=True)
result = sam3.process_image(
    image_path="assets/images/truck.jpg",
    prompts=["truck"]
)
```

### Understanding Profiling Results

When profiling is enabled, SAM3 generates two output files:

#### 1. `profile_results.json` - Machine-readable results

```json
[
    {
        "function": "process_image",
        "execution_time": 2.456,
        "timestamp": "2026-02-16 20:30:15",
        "args": ["image.jpg"],
        "kwargs": {"prompts": ["truck"]}
    },
    {
        "function": "_get_image_driver",
        "execution_time": 0.123,
        "timestamp": "2026-02-16 20:30:16",
        "args": [],
        "kwargs": {}
    }
]
```

#### 2. `profile_results.txt` - Human-readable results

```
Function: process_image
Time: 2.456s
Timestamp: 2026-02-16 20:30:15
Args: ('image.jpg',)
Kwargs: {'prompts': ['truck']}
---
```

### Profiling Best Practices

**For Development:**
```bash
# Profile fast tests only (skip slow video processing)
make test-fast --profile

# Profile specific scenario
make profile-example EXAMPLE=a
```

**For Benchmarking:**
```bash
# Profile all examples with different resolutions
make profile-all VIDEO_RES=480p
make profile-all VIDEO_RES=720p
make profile-all VIDEO_RES=1080p

# Compare results
cat profile_results.txt
```

**Interpreting Results:**
- Look for functions with high execution times
- Identify repeated calls that could be cached
- Compare times across different resolutions
- Monitor memory allocation patterns

### Platform-Specific Profiling Notes

**Linux:**
- Profiling works out of the box
- Consider using `perf` for system-level profiling
- Use `nvidia-smi` for GPU memory profiling

**macOS:**
- Profiling works natively on both Intel and Apple Silicon
- Use Activity Monitor for system resource monitoring
- Consider using `instruments` for detailed analysis

### Cleaning Profiling Results

```bash
# Remove profiling output files
make clean-profile

# Complete cleanup (includes profiling results)
make distclean
```

## �📖 API Documentation

### Sam3 Class (Main Entrypoint)

```python
from sam3 import Sam3  # Or: from sam3 import Sam3Entrypoint

sam3 = Sam3(
    bpe_path=None,              # Path to BPE tokenizer (auto-detected)
    num_workers=None,            # Number of workers (auto-detected)
    ram_usage_percent=0.25,      # RAM fraction for chunking (0-1)
    min_video_frames=15,         # Minimum frames per chunk
    min_chunk_overlap=1,         # Frame overlap between chunks
    temp_dir="./temp",           # Temporary directory
    default_output_dir="./outputs",  # Default output directory
    verbose=True                 # Enable logging
)
```

### Image Processing

#### `process_image()`

Process single or multiple images with text prompts or bounding boxes.

**Parameters:**
- `image_path` (str | List[str]): Path(s) to image file(s)
- `prompts` (str | List[str], optional): Text prompt(s) for segmentation
- `boxes` (List[float] | List[List[float]], optional): Bounding box(es) in XYWH format
- `box_labels` (List[int], optional): Labels for boxes (1=positive, 0=negative)
- `output_dir` (str, optional): Output directory

**Returns:** `ProcessingResult` with masks, object IDs, and metadata

**Example:**
```python
# Text prompts
result = sam3.process_image(
    image_path="image.jpg",
    prompts=["car", "person"]
)

# Bounding boxes
result = sam3.process_image(
    image_path="image.jpg",
    boxes=[[100, 100, 200, 200]]  # [x, y, width, height]
)

# Multiple images
result = sam3.process_image(
    image_path=["img1.jpg", "img2.jpg"],
    prompts=["object"]
)
```

### Video Processing

#### `process_video_with_prompts()`

Process video with text prompts and automatic propagation.

**Parameters:**
- `video_path` (str): Path to video file
- `prompts` (str | List[str]): Text prompt(s) for segmentation
- `output_dir` (str, optional): Output directory
- `propagation_direction` (str): "forward", "backward", or "both"

**Returns:** `ProcessingResult` with mask videos

**Example:**
```python
result = sam3.process_video_with_prompts(
    video_path="video.mp4",
    prompts=["person", "car"],
    propagation_direction="forward"
)
```

#### `process_video_with_points()`

Process video with point prompts on a specific frame.

**Parameters:**
- `video_path` (str): Path to video file
- `frame_idx` (int): Frame index for point annotation
- `points` (List[List[float]]): List of [x, y] coordinates
- `point_labels` (List[int]): Labels (1=positive, 0=negative)
- `object_id` (int): Unique object ID
- `output_dir` (str, optional): Output directory
- `propagation_direction` (str): "forward", "backward", or "both"

**Returns:** `ProcessingResult` with mask videos

**Example:**
```python
result = sam3.process_video_with_points(
    video_path="video.mp4",
    frame_idx=0,
    points=[[320, 180], [400, 200]],
    point_labels=[1, 1],
    object_id=1
)
```

#### `refine_video_object()`

Refine existing tracked object with additional points.

**Parameters:**
- `video_path` (str): Path to video file
- `frame_idx` (int): Frame index for refinement
- `object_id` (int): Existing object ID to refine
- `points` (List[List[float]]): Refinement point coordinates
- `point_labels` (List[int]): Point labels
- `output_dir` (str, optional): Output directory
- `propagation_direction` (str): Propagation direction

**Example:**
```python
result = sam3.refine_video_object(
    video_path="video.mp4",
    frame_idx=10,
    object_id=1,
    points=[[350, 190]],
    point_labels=[1]
)
```

#### `remove_video_objects()`

Remove objects from video segmentation by their IDs.

**Parameters:**
- `video_path` (str): Path to video file
- `object_ids` (int | List[int]): Object ID(s) to remove
- `output_dir` (str, optional): Output directory

**Example:**
```python
result = sam3.remove_video_objects(
    video_path="video.mp4",
    object_ids=[2, 3]
)
```

#### `process_video_with_segments()`

Process different video segments with different prompts.

**Parameters:**
- `video_path` (str): Path to video file
- `segments` (Dict): Segment definitions with prompts/points
- `output_dir` (str, optional): Output directory

**Example:**
```python
segments = {
    "segments": [
        {
            "start_time_sec": 0.0,
            "end_time_sec": 2.0,
            "prompts": ["person"]
        },
        {
            "start_frame": 50,
            "end_frame": 100,
            "points": [[320, 180]],
            "labels": [1]
        }
    ]
}

result = sam3.process_video_with_segments(
    video_path="video.mp4",
    segments=segments
)
```

### ProcessingResult

All processing methods return a `ProcessingResult` object:

```python
@dataclass
class ProcessingResult:
    success: bool                    # True if processing succeeded
    output_dir: str                  # Output directory path
    mask_files: List[str]            # List of generated mask files
    object_ids: List[int]            # List of detected object IDs
    metadata: Dict[str, Any]         # Additional metadata
    errors: Optional[List[str]]      # Error messages (if any)
```

## 🛠️ Development

### Project Structure

```
sam3-cpu/
├── sam3/                          # Core library
│   ├── __init__.py               # Package exports
│   ├── __globals.py              # Global configuration
│   ├── entrypoint.py             # Main API (2038 lines)
│   ├── wrapper.py                # Backward compatibility
│   ├── model/                    # Model implementations
│   ├── agent/                    # Agent components
│   ├── eval/                     # Evaluation tools
│   └── train/                    # Training utilities
├── examples/                      # Demo scripts
│   ├── example_a_*.py            # Individual scenarios (a-i)
│   └── run_all_examples.py       # Grouped runner
├── tests/                         # Test suite
│   ├── conftest.py               # Pytest configuration
│   └── test_all_scenarios.py     # All scenario tests
├── assets/                        # Sample data
│   ├── images/                   # Test images
│   └── videos/                   # Test videos
├── Makefile                       # Build automation
├── setup.sh                       # Setup script
├── pyproject.toml                # Project configuration
└── README.md                      # This file
```

### Available Make Targets

```bash
# Setup and Installation
make setup                    # Run setup script
make install                  # Install dependencies
make check-uv                 # Check if uv is installed

# Running Examples
make run-all                  # Run all examples
make run-example EXAMPLE=a    # Run specific example (a-i)
make run-a ... run-i          # Run individual examples
make run-all VIDEO_RES=720p   # Change video resolution

# Testing
make test                     # Run all tests
make test-fast                # Run fast tests only
make test-slow                # Run slow tests only
make test-image               # Run image tests only
make test-video               # Run video tests only
make test-scenario SCENARIO=a # Run specific scenario tests
make test-coverage            # Generate coverage report

# Maintenance
make clean                    # Remove cache files
make clean-outputs            # Remove output directories
make clean-cache              # Remove test cache
make distclean                # Complete cleanup
make lint                     # Run linter
make format                   # Format code
make check                    # Run linter + fast tests

# Information
make help                     # Show all commands
make info                     # Display project info
make benchmark                # Run performance benchmarks
```

### Example Usage with Variables

```bash
# Run all examples with 1080p video
make run-all VIDEO_RES=1080p

# Run specific example with custom arguments
make run-example EXAMPLE=e ARGS="--prompts person car --verbose"

# Run tests with verbose output
make test ARGS="-v"

# Run fast tests only
make test-fast
```

## 🧪 Testing

SAM3 includes comprehensive test coverage for all 9 scenarios.

### Running Tests

```bash
# All tests (includes slow video processing)
make test

# Fast tests only (skip slow video tests)
make test-fast

# Specific test categories
make test-image        # Image processing tests
make test-video        # Video processing tests
make test-scenario SCENARIO=a  # Specific scenario

# With coverage
make test-coverage

# Verbose output
make test ARGS="-v"

# Run specific test
uv run pytest tests/test_all_scenarios.py::TestScenarioA::test_single_image_single_prompt -v
```

### Test Categories

Tests are organized with pytest markers:

- `@pytest.mark.image` - Image processing tests
- `@pytest.mark.video` - Video processing tests
- `@pytest.mark.slow` - Slow tests (video processing)
- `@pytest.mark.scenario_a` through `@pytest.mark.scenario_i` - Specific scenarios

### Writing Tests

```python
import pytest
from sam3 import Sam3

@pytest.mark.image
@pytest.mark.scenario_a
def test_my_scenario(sam3_instance, test_image_truck, temp_output_dir):
    result = sam3_instance.process_image(
        image_path=test_image_truck,
        prompts=["truck"],
        output_dir=temp_output_dir
    )
    
    assert result.success
    assert len(result.mask_files) > 0
```

## 🎓 Advanced Usage

### Memory Management

SAM3 automatically manages memory and chunks large videos when needed:

```python
# Configure memory usage
sam3 = Sam3(
    ram_usage_percent=0.25,      # Use 25% of available RAM
    min_video_frames=15,         # Minimum frames per chunk
    min_chunk_overlap=1          # Frame overlap between chunks
)

# Process large video (automatic chunking)
result = sam3.process_video_with_prompts(
    video_path="large_video.mp4",
    prompts=["person"]
)
```

### Custom Configuration

```python
from sam3 import Sam3

sam3 = Sam3(
    bpe_path="/path/to/bpe.json",      # Custom tokenizer
    num_workers=4,                      # Custom worker count
    ram_usage_percent=0.5,              # Use 50% of RAM
    temp_dir="/tmp/sam3",               # Custom temp directory
    default_output_dir="./my_outputs",  # Custom output directory
    verbose=True                        # Enable detailed logging
)
```

### Backward Compatibility

The legacy `Sam3VideoPredictor` API is still supported:

```python
from sam3 import Sam3VideoPredictor

# Old API still works
predictor = Sam3VideoPredictor()
# ... use legacy methods
```

## 🔍 Troubleshooting

### Common Issues

**Issue: "uv: command not found"**
```bash
# Run setup script (works on Linux and macOS)
make setup

# Or install manually
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"

# For permanent solution, add to your shell profile:
# Linux (bash): echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
# macOS (zsh): echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
```

**Issue: "ffmpeg: command not found"**
```bash
# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg

# Linux (RHEL/CentOS/Fedora)
sudo yum install ffmpeg  # or sudo dnf install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
ffprobe -version
```

**Issue: Out of memory during video processing**
```python
# Reduce RAM usage or increase chunk overlap
sam3 = Sam3(ram_usage_percent=0.15)  # Use less RAM
```

**Issue: Tests failing due to missing assets**
```bash
# Ensure assets exist
ls -la assets/images/
ls -la assets/videos/
```

**Issue: Permission denied when running setup.sh**
```bash
# Make script executable
chmod +x setup.sh
./setup.sh
```

### Platform-Specific Issues

#### Linux-Specific

**Issue: "ModuleNotFoundError: No module named 'torch'"**
```bash
# Reinstall with uv
uv pip install -e .

# Or activate virtual environment
source .venv/bin/activate
```

**Issue: CUDA not detected (for GPU users)**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA support if needed
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

#### macOS-Specific

**Issue: "xcrun: error: invalid active developer path"**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Issue: Homebrew not found**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Follow post-install instructions to add to PATH
```

**Issue: Apple Silicon (M1/M2/M3) compatibility**
```bash
# The project works natively on Apple Silicon
# Verify architecture
uname -m  # Should show "arm64" for Apple Silicon

# Python should be ARM64 native
python3 -c "import platform; print(platform.machine())"
```

**Issue: "ImportError: cannot import name 'Sam3'"**
```bash
# Ensure you're using the virtual environment
uv run python -c "from sam3 import Sam3; print('OK')"

# Or explicitly activate venv
source .venv/bin/activate
python -c "from sam3 import Sam3; print('OK')"
```

### Profiling Issues

**Issue: Profile results not generated**
```bash
# Ensure --profile flag is used
make profile-example EXAMPLE=a

# Or check if flag is in command
uv run python examples/example_a_image_with_prompts.py --profile

# Verify files are created
ls -l profile_results.*
```

**Issue: Profiling slows down execution**
```
This is expected! Profiling adds overhead.
For production use, run without --profile flag.
```

### Getting Help

If you encounter issues not listed here:

1. Check the [GitHub Issues](https://github.com/rhubarb-ai/sam3-cpu/issues)
2. Run `make info` to display system information
3. Enable verbose mode: `--verbose` flag
4. Check logs in the output directory

## 📊 Performance

Benchmark results on different hardware:

| Resolution | Frames | CPU (Intel i7) | GPU (NVIDIA RTX 3090) |
|-----------|--------|---------------|----------------------|
| 480p      | 100    | ~45s          | ~8s                  |
| 720p      | 100    | ~90s          | ~12s                 |
| 1080p     | 100    | ~180s         | ~20s                 |

Run your own benchmarks:
```bash
make benchmark
```


## ⚙️ Configuration & Globals

SAM3 uses several global configuration settings that can be customized:

### Default Settings

```python
from sam3.__globals import (
    TEMP_DIR,                          # /tmp/sam3-cpu (or /tmp/sam3-gpu)
    DEFAULT_OUTPUT_DIR,                # ./results
    DEFAULT_PROPAGATION_DIRECTION,     # "both"
    DEFAULT_OVERLAP,                   # 1 frame
    RAM_USAGE_PERCENT,                 # 0.33 (33% for CPU)
    VRAM_USAGE_PERCENT,                # 0.90 (90% for GPU)
)
```

**Key Difference - CPU vs GPU**:
- **GPU (VRAM)**: Can use 90% because GPU memory is dedicated to compute
- **CPU (RAM)**: Uses 33% because RAM is shared with OS and applications
- GPU chunking is more aggressive to maximize throughput

### Customizing Memory Usage

```python
from sam3 import Sam3

# For CPU: Using more or less RAM
sam3_cpu = Sam3(ram_usage_percent=0.50, verbose=True)  # Use 50% of RAM

# For GPU: Adjust VRAM usage (if on GPU system)
sam3_gpu = Sam3(vram_usage_percent=0.80, verbose=True)  # Use 80% of VRAM instead of 90%
```

### Customizing Temporary Directory

```python
# Change temporary directory for chunks
from sam3 import Sam3
sam3 = Sam3(temp_dir="/custom/temp/dir", verbose=True)

# Chunks will be stored in custom location
result = sam3.process_video_with_prompts(
    video_path="video.mp4",
    prompts=["person"]
)
```

**Key Configuration Points:**
- **TEMP_DIR**: Where video chunks are temporarily stored
- **DEFAULT_OUTPUT_DIR**: Where final results (masks, videos) are saved
- **DEFAULT_PROPAGATION_DIRECTION**: Default tracking direction ("both", "forward", "backward")
- **DEFAULT_OVERLAP**: Frame overlap between chunks (default: 1)
- **RAM_USAGE_PERCENT**: Fraction of RAM to use for CPU chunking (default: 0.33 = 33%)
- **VRAM_USAGE_PERCENT**: Fraction of VRAM to use for GPU chunking (default: 0.90 = 90%)

## 💾 Memory Management

SAM3 intelligently manages memory to handle videos of any size on both **CPU (RAM)** and **GPU (VRAM)**.

### How Memory Calculation Works

```python
# Memory needed = frames × video_width × video_height × channels × bytes_per_channel
video_resolution = width × height  # Actual video resolution (not model size!)
channels = 3                       # RGB
bytes_per_channel = 1              # uint8 (video decoding)

memory_needed = num_frames × width × height × 3 × 1
```

**Important**: Memory calculation uses the **actual video resolution**, not the model processing size (1008×1008). Even though SAM3 resizes frames to 1008×1008 during inference, the memory footprint is determined by decoding frames at their original resolution.

**Example 1**: 480p video (854×480, 6,364 frames)
- **Memory per frame**: 854 × 480 × 3 ≈ 1.17 MB
- **Total memory**: 6,364 × 1.17 MB ≈ 7.46 GB
- **With 1.5× safety**: 7.46 × 1.5 ≈ 11.2 GB
- **Available RAM** (33%): 64 GB system → 21 GB usable
- **Result**: ✅ No chunking needed (fits in memory)

**Example 2**: 1080p video (1920×1080, 6,364 frames)
- **Memory per frame**: 1920 × 1080 × 3 ≈ 5.93 MB
- **Total memory**: 6,364 × 5.93 MB ≈ 37.7 GB
- **With 1.5× safety**: 37.7 × 1.5 ≈ 56.6 GB
- **Available RAM** (33%): 64 GB system → 21 GB usable
- **Result**: ⚠️ Chunking required → 4 chunks of ~1,684 frames each

### Chunking Strategy

**CPU (RAM) - 33% usage:**

| Video | Resolution | Frames | Available RAM | Memory Needed | Chunk Size | Chunks |
|-------|-----------|--------|---------------|---------------|------------|--------|
| 480p  | 854×480   | 6,364  | 64 GB (33%)   | 11.2 GB       | N/A        | 1 (no chunking) |
| 720p  | 1280×720  | 6,364  | 64 GB (33%)   | 25.2 GB       | 2,800      | 3      |
| 1080p | 1920×1080 | 6,364  | 64 GB (33%)   | 56.6 GB       | 1,684      | 4      |
| 4K    | 3840×2160 | 6,364  | 64 GB (33%)   | 226 GB        | 420        | 16     |

**GPU (VRAM) - 90% usage:**

| Video | Resolution | Frames | Available VRAM | Memory Needed | Chunk Size | Chunks |
|-------|-----------|--------|----------------|---------------|------------|--------|
| 480p  | 854×480   | 6,364  | 24 GB (90%)    | 11.2 GB       | N/A        | 1 (no chunking) |
| 720p  | 1280×720  | 6,364  | 24 GB (90%)    | 25.2 GB       | N/A        | 1 (no chunking) |
| 1080p | 1920×1080 | 6,364  | 24 GB (90%)    | 56.6 GB       | 4,580      | 2      |
| 4K    | 3840×2160 | 6,364  | 24 GB (90%)    | 226 GB        | 1,142      | 6      |

**Key Difference**: GPU can process larger chunks because:
- Higher memory usage percentage (90% vs 33%)
- Dedicated memory (not shared with OS)
- Better for high-resolution video processing

**Chunking Algorithm:**
1. Calculate memory needed for entire video (based on **actual resolution**)
2. Apply 1.5× safety multiplier for headroom
3. Compare with available memory (RAM or VRAM based on device)
4. Use device-appropriate usage percentage (33% CPU, 90% GPU)
5. If insufficient, calculate optimal chunk size
6. Process chunks with overlap (default: 1 frame)
7. Merge results intelligently (skip overlap frames)
8. Clean up temporary files automatically

**Key Point**: Chunk size depends on **video resolution** AND **device type**!

### Memory Optimization Tips

**For Large Videos:**
```python
# Option 1: Use frame range to process only needed segment
result = sam3.process_video_with_prompts(
    video_path="long_video.mp4",
    prompts=["person"],
    frame_from="2:00",   # Start at 2 minutes
    frame_to="5:00"      # End at 5 minutes
)

# Option 2: Increase RAM usage percentage (if you have spare RAM)
# Edit DEFAULT_RAM_USAGE_PERCENT in sam3/__globals.py
# DEFAULT_RAM_USAGE_PERCENT = 0.40  # Use 40% instead of 25%
```

**For Memory-Constrained Systems:**
```python
# Reduce RAM usage percentage
# Edit sam3/__globals.py:
DEFAULT_RAM_USAGE_PERCENT = 0.15  # Use only 15% of RAM

# This will create more, smaller chunks
# Slower processing, but safer on low-memory systems
```

### Troubleshooting Memory Issues

**"Unable to allocate X GB" Error:**
```bash
# This was fixed! If you see this:
# 1. Update to latest version
# 2. Check that chunking is enabled (look for "Chunking enabled" message)
# 3. Reduce DEFAULT_RAM_USAGE_PERCENT if needed
```

**Out of Memory During Processing:**
```Python
# Even with chunking, individual frames might be too large
# Solution: Use frame range extraction first
result = sam3.process_video_with_prompts(
    video_path="4k_video.mp4",
    prompts=["object"],
    frame_from=0,
    frame_to=100  # Process only first 100 frames
)
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Credits

This project builds on **Meta's Segment Anything Model 3 (SAM3)** and modifies parts of the original SAM3 code to add CPU compatibility while preserving existing CUDA support. The SAM3‑related components remain under the SAM License; see [LICENSE](LICENSE) for details.

## 📧 Contact

**Dr Prashant Aparajeya**  
Email: p.aparajeya@aisimply.uk  
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=rhubarb-ai/sam3-cpu&type=Date)](https://star-history.com/#rhubarb-ai/sam3-cpu&Date)

---

**Made with ❤️ by the Rhubarb AI Team**
