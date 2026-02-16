# SAM3 README Update - New Sections

## Add after "Quick Start" section and before "Usage Examples":

## 🎥 Advanced Video Features

### Automatic Video Chunking

SAM3 automatically handles large videos that don't fit in memory by intelligently splitting them into chunks:

```python
from sam3 import Sam3

sam3 = Sam3(verbose=True)

# Process 10-minute video - automatically chunked if needed
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
2. **Auto-Ch unking**: If memory insufficient, splits video into RAM-safe chunks
3. **Overlap Handling**: Processes chunks with overlap to ensure continuity
4. **Smart Merging**: Merges results intelligently, avoiding duplicate frames from overlaps

**Example Output:**
```
📊 Chunking required for video
  Total frames: 6364
  Chunk size: ~591 frames
  Number of chunks: 11
Chunking enabled: 11 chunks, chunk_size=591, overlap=1
Processing chunk 1/11 (frames 0-590)...
```

### Frame/Time Range Extraction

Process only specific portions of a video without loading the entirefile:

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

## 🧪 Quick Testing

For rapid iteration and testing, use 1-second clips:

```python
# Test all functionality in ~2 minutes (instead of 30+ minutes)
python test_quick.py
```

This runs 4 quick tests:
1. Basic video with prompts (1-sec clip, direction="both")
2. Second run (verifies consistency)
3. Multiple prompts
4. Direction override

**Output:**
```
======================================================================
SUMMARY: Quick Tests Complete
======================================================================
Passed: 4/4
Output directory: results/quick_test
🎉 All quick tests completed!
```

## Add to "Configuration" or create new section:

## ⚙️ Configuration & Globals

SAM3 uses centralized configuration via `sam3/__globals.py`:

```python
# Key settings (all customizable)
DEFAULT_MIN_VIDEO_FRAMES = 15                  # Minimum frames per chunk
DEFAULT_MIN_CHUNK_OVERLAP = 1                  # Frame overlap between chunks
RAM_USAGE_PERCENT = 0.25                       # Use 25% of RAM for chunking (CPU)
GPU_MEMORY_RESERVE_PERCENT = 0.05              # Reserve 5% VRAM (GPU)
DEFAULT_PROPAGATION_DIRECTION = "both"         # Bidirectional tracking
TEMP_DIR = "/tmp/sam3-cpu"  # or "/tmp/sam3-gpu"  # Temporary files location
DEFAULT_OUTPUT_DIR = "./results"               # Default output directory
```

**Customizing at Initialization:**
```python
from sam3 import Sam3

sam3 = Sam3(
    num_workers=2,              # Number of workers (CPU) or GPUs
    ram_usage_percent=0.30,     # Use 30% of RAM instead of 25%
    min_video_frames=20,        # Larger minimum chunk size
    min_chunk_overlap=2,        # 2-frame overlap
    temp_dir="/my/tmp",         # Custom temp directory
    default_output_dir="./my_results",
    verbose=True
)
```

**Temporary File Management:**
- Chunked video segments: `{TEMP_DIR}/{video_name}/chunks/chunk_*.mp4`
- Extracted ranges: `{TEMP_DIR}/{video_name}_range_{from}_{to}.mp4`
- Auto-cleanup: All temporary files removed after processing
- Location: `/tmp/sam3-cpu` (CPU) or `/tmp/sam3-gpu` (GPU)

## Add to "Performance" section or create new:

## 📊 Memory Management

### How Memory Calculation Works

SAM3 accurately predicts memory requirements:

```
Memory Needed = (Model Input Size × Frames × Channels × Bytes) + Inference Overhead

Model Input: 1008×1008 pixels (SAM3 resizes all inputs)
Data Type: float32 (4 bytes per channel)
Channels: 3 (RGB)
Inference Overhead: 6.9 GB (video) or 6.76 GB (image)

Example (1000 frames):
  = (1008 × 1008 × 3 × 4 × 1000) / (1024³) + 6.9
  = 11.34 GB + 6.9 GB
  = 18.24 GB

Safety Multiplier: 3× (requires 54.72 GB total)
```

### Chunking Strategy

Aligned with `scripts/linux/video_chunk_manager.sh`:

1. **Calculate RAM-safe chunk size**: `chunk_size = (available_ram × 0.25) / bytes_per_frame`
2. **Generate chunks with stride**: `stride = chunk_size - overlap`
3. **Process sequentially**: One chunk at a time
4. **Merge results**: Skip overlapping frames from subsequent chunks  
5. **Cleanup**: Remove chunk files immediately after processing

### Chunking vs No Chunking

| Video Length | Frames | Memory Needed | Chunks (56GB RAM) | Processing Time (CPU) |
|--------------|--------|---------------|-------------------|----------------------|
| 10 sec @ 30fps | 300 | 16.8 GB | ✅ No chunking | ~8 min |
| 1 min @ 30fps | 1800 | 88.5 GB | 2 chunks | ~45 min |
| 4 min @ 25fps | 6000 | 290 GB | 11 chunks | ~3 hours |
| 10 min @ 30fps | 18000 | 885 GB | 35 chunks | ~9 hours |

**With Frame Range:**
- Extract 30 sec from 10 min video: Process ~900 frames instead of 18,000 (20× faster!)
- Extract specific event: Skip loading entire video

## 🔧 Troubleshooting

### Chunking Issues

**Problem**: "Memory check passed" but still crashes with "Unable to allocate X GB"

**Solution**: This was fixed in v1.1.0. Update to latest version:
```bash
git pull
uv pip install -e .
```

The fix: Memory calculation now uses model size (1008×1008 float32) instead of original video size.

### Frame Range Extraction Fails

**Problem**: "invalid literal for int() with base 10: ''"

**Solution**: Ensure `ffmpeg` and `ffprobe` are installed:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Verify
ffmpeg -version
ffprobe -version
```

### Mask Videos Not Generated

**Problem**: "Detected 2 object(s)" but "Generated 0 mask video(s)"

**Solution**: Fixed in v1.1.0. The issue was incorrect result structure parsing. Update to latest version.

