# SAM3-CPU

**CPU-compatible wrapper around Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)** for image and video segmentation with intelligent memory management.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/rhubarb-ai/sam3-cpu/pulls)

## Features

- **CPU + GPU** — runs on CPU out of the box; uses CUDA when available
- **Memory-aware chunking** — automatically splits long videos into chunks sized to available RAM / VRAM
- **Cross-chunk continuity** — IoU-based mask remapping keeps object IDs consistent across chunks
- **Text, point, box & mask prompts** — unified API for all prompt types
- **Video segment processing** — process a specific frame range or time range instead of the full video
- **Per-object tracking metadata** — frame ranges, timestamps, and timecodes for every detected object
- **CLI tools** — `image_prompter.py` and `video_prompter.py` for quick experiments

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Tools](#cli-tools)
  - [image\_prompter.py](#image_prompterpy)
  - [video\_prompter.py](#video_prompterpy)
- [Python API](#python-api)
- [Video Chunking](#video-chunking)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.12 + |
| OS | Linux (Ubuntu 20.04+), macOS 11+ (Intel & Apple Silicon) |
| ffmpeg / ffprobe | Required for video processing |
| HuggingFace account | Model checkpoints are hosted on HuggingFace — you must [request access](https://huggingface.co/facebook/sam3) and authenticate (`huggingface-cli login`) before first use |

---

## Installation

### Automated (recommended)

```bash
git clone https://github.com/rhubarb-ai/sam3-cpu.git
cd sam3-cpu
chmod +x setup.sh && ./setup.sh   # or: make setup
```

### Manual

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project
uv pip install -e .
```

### System dependencies

**Linux (Debian / Ubuntu)**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg python3-dev build-essential
```

**macOS**
```bash
brew install ffmpeg
```

### Verify

```bash
python -c "from sam3 import Sam3; print('OK')"
```

---

## Quick Start

### Image segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3(verbose=True)
result = sam3.process_image(
    image_path="assets/images/truck.jpg",
    prompts=["truck", "wheel"],
    output_dir="results/demo",
)
print(result.object_ids)
```

### Video segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3(verbose=True)
result = sam3.process_video_with_prompts(
    video_path="assets/videos/tennis_480p.mp4",
    prompts=["person", "tennis racket"],
    propagation_direction="both",
)
```

### Makefile shortcuts

```bash
make run-all              # run every example
make run-example EXAMPLE=a
make test                 # run pytest suite
```

---

## CLI Tools

Two standalone scripts provide a quick way to run segmentation from the terminal
without writing Python code.

### image\_prompter.py

Segment one or more images with text prompts, click points, or bounding boxes.

#### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--images` | `str+` | *(required)* | One or more image file paths |
| `--prompts` | `str+` | `None` | Text prompts (e.g. `person car`) |
| `--points` | `str+` | `None` | Click points as `x,y` pairs (e.g. `320,240 500,300`) |
| `--point-labels` | `int+` | all `1` | Labels for each point (`1` = positive, `0` = negative) |
| `--bbox` | `float+` | `None` | Bounding box(es) as `x y w h` (multiples of 4 for several boxes) |
| `--output` | `str` | `results` | Output directory |
| `--alpha` | `float` | `0.5` | Overlay alpha for mask visualisation (`0.0`–`1.0`) |
| `--device` | `str` | auto | Force `cpu` or `cuda` (auto-detected if omitted) |

At least one of `--prompts`, `--points`, or `--bbox` is required.

#### Examples

```bash
# Text prompt — segment truck and wheels
uv run python image_prompter.py \
    --images assets/images/truck.jpg \
    --prompts "truck" "wheel" \
    --output results/truck_demo

# Bounding-box prompt — segment inside a rectangle
uv run python image_prompter.py \
    --images assets/images/truck.jpg \
    --bbox 100 50 400 300 \
    --output results/truck_bbox

# Click-point prompt
uv run python image_prompter.py \
    --images assets/images/truck.jpg \
    --points 250,175 \
    --output results/truck_points

# Batch: multiple images with the same text prompt
uv run python image_prompter.py \
    --images img1.jpg img2.jpg img3.jpg \
    --prompts "person" \
    --output results/batch

# Combined: text + points on a single image
uv run python image_prompter.py \
    --images scene.jpg \
    --prompts "dog" \
    --points 120,340 \
    --point-labels 1 \
    --alpha 0.45 --device cpu
```

### video\_prompter.py

Segment a video with text prompts, click points, or binary masks.  Supports
automatic memory-aware chunking, segment processing (frame or time ranges),
and generates per-object tracking metadata.

#### Options

| Flag | Type | Default | Description |
|---|---|---|---|
| `--video` | `str` | *(required)* | Input video file |
| `--prompts` | `str+` | `None` | Text prompts (e.g. `person ball`) |
| `--points` | `str+` | `None` | Click points as `x,y` pairs |
| `--point-labels` | `int+` | all `1` | Labels for each point (`1` = positive, `0` = negative) |
| `--masks` | `str+` | `None` | Binary mask image(s) for initial object prompts |
| `--output` | `str` | `results` | Output directory |
| `--alpha` | `float` | `0.5` | Overlay alpha (`0.0`–`1.0`) |
| `--device` | `str` | auto | Force `cpu` or `cuda` |
| `--chunk-spread` | `str` | `default` | Chunk size strategy: `default` or `even` |
| `--keep-temp` | flag | off | Preserve intermediate chunk files in output |
| `--frame-range` | `int int` | `None` | Process only frames `START..END` (0-based, inclusive) |
| `--time-range` | `str str` | `None` | Process a time segment (seconds, `MM:SS`, or `HH:MM:SS`) |

At least one of `--prompts`, `--points`, or `--masks` is required.
`--frame-range` and `--time-range` are mutually exclusive.

#### Examples

```bash
# Text prompt — segment all people and tennis rackets
uv run python video_prompter.py \
    --video assets/videos/tennis_480p.mp4 \
    --prompts "person" "tennis racket" \
    --output results/tennis_demo

# Point prompt
uv run python video_prompter.py \
    --video assets/videos/tennis_480p.mp4 \
    --points 320,240 \
    --output results/tennis_points

# Mask prompt — provide binary mask images as initial objects
uv run python video_prompter.py \
    --video clip.mp4 \
    --masks player_mask.png ball_mask.png \
    --output results/masks_demo

# Frame-range — process only frames 100 to 500
uv run python video_prompter.py \
    --video match.mp4 \
    --prompts "player" \
    --frame-range 100 500

# Time-range with MM:SS notation — process from 0:05 to 0:30
uv run python video_prompter.py \
    --video match.mp4 \
    --prompts "player" \
    --time-range 0:05 0:30

# Time-range with seconds — process 10s to 45.5s
uv run python video_prompter.py \
    --video match.mp4 \
    --prompts "player" \
    --time-range 10.0 45.5

# Time-range with HH:MM:SS — process a 5-minute segment
uv run python video_prompter.py \
    --video long_match.mp4 \
    --prompts "player" \
    --time-range 0:02:00 0:07:00

# Even chunk spread + preserve temp files
uv run python video_prompter.py \
    --video clip.mp4 \
    --prompts "person" \
    --chunk-spread even --keep-temp

# Force CPU processing
uv run python video_prompter.py \
    --video clip.mp4 \
    --prompts "person" \
    --device cpu
```

---

## Python API

The high-level entry point is `Sam3`:

```python
from sam3 import Sam3
sam3 = Sam3(verbose=True)
```

### Image scenarios

| Method | Description |
|---|---|
| `process_image(image_path, prompts, ...)` | Segment by text prompts |
| `process_image_with_bounding_box(image_path, bbox, ...)` | Segment inside a bounding box |
| `process_image_with_click_points(image_path, click_points, labels, ...)` | Segment at clicked points |

### Video scenarios

| Method | Description |
|---|---|
| `process_video_with_prompts(video_path, prompts, ...)` | Segment & track by text prompts |
| `process_video_with_click_points(video_path, click_points, ...)` | Segment & track from point clicks |
| `process_video_with_masks(video_path, masks, ...)` | Segment & track from initial masks |

All video methods support:

- `propagation_direction` — `"forward"`, `"backward"`, or `"both"` (default)
- `frame_range` / `time_range` / `timestamp_range` — process a sub-section of the video
- Automatic chunking when memory is limited (see below)

Lower-level access is available through `ImageProcessor`, `VideoProcessor`,
`ChunkProcessor`, and the driver classes in `sam3/drivers.py`.
See [docs/local/README_FULL.md](docs/local/README_FULL.md) for the complete API
reference.

---

## Video Chunking

When a video is too large to fit in memory the framework automatically splits it
into overlapping chunks, processes each chunk independently, and stitches the
results back together.

**How it works:**

1. `MemoryManager` computes how many frames fit in available RAM (CPU) or VRAM (GPU).
2. The video is split into chunks with configurable overlap (default 1 frame).
3. Each chunk is segmented and tracked independently.
4. At chunk boundaries, masks from the overlap region are matched using IoU and
   object IDs are remapped so they stay consistent across the full video.

**Key parameters** (set in `config.json` or `sam3/__globals.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `ram_usage_percent` | 0.45 | Fraction of free RAM budget for frames |
| `min_frames` | 25 | Minimum frames per chunk |
| `chunk_overlap` | 1 | Overlap frames between chunks |
| `CHUNK_MASK_MATCHING_IOU_THRESHOLD` | 0.75 | IoU threshold for cross-chunk ID matching |

---

## Output Structure

### image\_prompter.py

```
results/<image_name>/
├── masks/
│   ├── <prompt>/
│   │   ├── object_0_mask.png
│   │   ├── object_1_mask.png
│   │   └── ...
│   └── ...
├── overlays/
│   ├── overlay_<prompt>.png
│   └── ...
└── metadata.json
```

### video\_prompter.py

```
results/<video_name>/
├── masks/
│   ├── <prompt>/
│   │   ├── object_0_mask.mp4     # binary mask video per object
│   │   ├── object_1_mask.mp4
│   │   └── object_tracking.json  # per-object tracking metadata
│   └── ...
├── overlay_<prompt>.mp4          # coloured overlay on original video
├── metadata/
│   ├── video_metadata.json       # fps, resolution, frame count
│   └── memory_info.json          # RAM/VRAM budget analysis
├── temp_files/                   # only when --keep-temp is set
│   ├── chunk_000/
│   │   └── masks/<prompt>/object_<id>/*.png
│   └── ...
└── metadata.json                 # final run metadata
```

### Object tracking metadata

Each `object_tracking.json` contains a list of objects with their frame
presence, timestamps, and timecodes mapped to the **original** video
(accounting for any `--frame-range` / `--time-range` offset):

```json
[
  {
    "object_id": 0,
    "first_frame": 12,
    "last_frame": 487,
    "total_frames_active": 476,
    "total_frames": 500,
    "first_timestamp": 0.48,
    "last_timestamp": 19.48,
    "duration_s": 19.0,
    "first_timecode": "00:00:00.480",
    "last_timecode": "00:00:19.480"
  },
  {
    "object_id": 1,
    "first_frame": 0,
    "last_frame": 500,
    "total_frames_active": 501,
    "total_frames": 500,
    "first_timestamp": 0.0,
    "last_timestamp": 20.0,
    "duration_s": 20.0,
    "first_timecode": "00:00:00.000",
    "last_timecode": "00:00:20.000"
  }
]
```

The top-level `metadata.json` also includes:

- `segment` — the resolved frame range when `--frame-range` or `--time-range` is used (`null` for full-video)
- `object_tracking` — the same per-object data keyed by prompt name

---

## Configuration

Runtime defaults live in `config.json` (loaded at startup) and compile-time
constants in `sam3/__globals.py`.  Key settings:

```jsonc
{
  "ram_usage_percent": 0.45,
  "min_frames": 25,
  "chunk_overlap": 1,
  "tmp_base": "/tmp/sam3-cpu",
  "verbose": true,
  "image_inference_MB": 6755,
  "video_inference_MB": 6895
}
```

---

## Project Structure

```
sam3-cpu/
├── image_prompter.py          # CLI – image segmentation
├── video_prompter.py          # CLI – video segmentation
├── main.py                    # Legacy CLI entry point
├── config.json                # Runtime configuration
├── setup.sh / Makefile        # Build helpers
│
├── sam3/                      # Core package
│   ├── api.py                 # Sam3 high-level API
│   ├── drivers.py             # Sam3ImageDriver / Sam3VideoDriver
│   ├── image_processor.py     # ImageProcessor
│   ├── video_processor.py     # VideoProcessor
│   ├── chunk_processor.py     # ChunkProcessor (cross-chunk logic)
│   ├── postprocessor.py       # VideoPostProcessor
│   ├── memory_manager.py      # MemoryManager
│   ├── model_builder.py       # Model loading
│   ├── __globals.py           # Constants & defaults
│   ├── utils/                 # Utility modules
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   ├── profiler.py
│   │   ├── system_info.py
│   │   ├── ffmpeglib.py
│   │   └── visualization.py
│   ├── model/                 # SAM 3 model definitions
│   ├── sam/                   # SAM core modules
│   └── archive/               # Deprecated & rough scripts
│
├── examples/                  # Runnable example scripts
├── notebook/                  # Jupyter notebooks
├── tests/                     # Pytest test suite
├── scripts/                   # Utility scripts
├── assets/                    # Sample images & videos
└── docs/                      # Extended documentation
```

---

## Testing

```bash
# Run the full suite
uv run python -m pytest tests/ -v

# Run a specific file
uv run python -m pytest tests/test_iou_matching.py -v
```

---

## Citation

If you use this project in your research or applications, please cite **both**
the original SAM 3 paper and this repository.

### SAM 3 (Meta)

```bibtex
@misc{carion2025sam3segmentconcepts,
    title   = {SAM 3: Segment Anything with Concepts},
    author  = {Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and
               Shoubhik Debnath and Ronghang Hu and Didac Suris and
               Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr
               and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo
               and Arpit Kalla and Markus Marks and Joseph Greer and
               Meng Wang and Peize Sun and Roman Rädle and
               Triantafyllos Afouras and Effrosyni Mavroudi and
               Katherine Xu and Tsung-Han Wu and Yu Zhou and
               Liliane Momeni and Rishi Hazra and Shuangrui Ding and
               Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li
               and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár
               and Nikhila Ravi and Kate Saenko and Pengchuan Zhang
               and Christoph Feichtenhofer},
    year    = {2025},
    eprint  = {2511.16719},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url     = {https://arxiv.org/abs/2511.16719},
}
```

### SAM3-CPU

```bibtex
@misc{aparajeya2026sam3cpu,
    title  = {SAM3-CPU: Segment Anything with Concepts — CPU-compatible
              inference with memory-aware chunking},
    author = {Prashant Aparajeya},
    year   = {2026},
    url    = {https://github.com/rhubarb-ai/sam3-cpu},
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The underlying SAM 3 model weights are subject to Meta's [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

---

## Contact

**Dr Prashant Aparajeya**
Email: p.aparajeya@aisimply.uk
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)
