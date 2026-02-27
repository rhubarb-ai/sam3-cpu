# SAM3-CPU

**CPU-compatible wrapper around Meta's [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)** for image and video segmentation with intelligent memory management.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/rhubarb-ai/sam3-cpu/pulls)

## Features

- **CPU + GPU** — runs on CPU out of the box; uses CUDA when available
- **Zero GPU footprint** — `--device cpu` hides GPUs completely (0 MiB VRAM used)
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
- [Device Selection](#device-selection)
- [Quick Start](#quick-start)
- [CLI Tools](#cli-tools)
  - [image\_prompter.py](#image_prompterpy)
  - [video\_prompter.py](#video_prompterpy)
- [Python API](#python-api)
- [Video Chunking](#video-chunking)
- [Output Structure](#output-structure)
- [Profiling](#profiling)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)
- [Contributors](#contributors)

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

# Create virtual environment and install
uv venv
source .venv/bin/activate
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
uv run python -c "from sam3 import Sam3; print('OK')"
```

---

## Device Selection

By default, SAM3-CPU auto-detects CUDA and uses the GPU when available.
Pass `--device cpu` to any CLI tool to force CPU execution:

```bash
# Image — force CPU
uv run python image_prompter.py --image img.jpg --prompts dog --device cpu

# Video — force CPU
uv run python video_prompter.py --video clip.mp4 --prompts person --device cpu
```

When `--device cpu` is specified **no GPU memory is allocated at all** — the
CUDA runtime context is never initialised, so `nvidia-smi` will show 0 MiB
used by the process.

---

## Quick Start

### Image segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3()
result = sam3.process_image_with_prompts(
    image_path="assets/images/truck.jpg",
    prompts=["truck", "wheel"],
    output_dir="results/demo",
)
print(result)
```

### Video segmentation (Python)

```python
from sam3 import Sam3

sam3 = Sam3()
result = sam3.process_video_with_prompts(
    video_path="assets/videos/sample.mp4",
    prompts=["person", "tennis racket"],
    output_dir="results/demo",
)
```

### Makefile shortcuts

```bash
make help                 # show all available targets
make setup                # install uv + dependencies
make test                 # run pytest suite
make run-all              # run every example
make run-example EXAMPLE=a

# CLI tools via make
make image-prompter IMAGES='assets/images/truck.jpg' PROMPTS='truck wheel'
make video-prompter VIDEO='assets/videos/sample.mp4' PROMPTS='person'
make video-prompter VIDEO='clip.mp4' PROMPTS='player' FRAME_RANGE='100 500'
make video-prompter VIDEO='clip.mp4' PROMPTS='player' TIME_RANGE='0:05 0:30'
```

All `make` variables:

| Variable | Used by | Example |
|---|---|---|
| `IMAGES` | `image-prompter` | `'img1.jpg img2.jpg'` |
| `VIDEO` | `video-prompter` | `'clip.mp4'` |
| `PROMPTS` | both | `'person car'` |
| `POINTS` | both | `'320,240 500,300'` |
| `POINT_LABELS` | both | `'1 0'` |
| `BBOX` | `image-prompter` | `'100 50 400 300'` |
| `MASKS` | `video-prompter` | `'mask.png'` |
| `FRAME_RANGE` | `video-prompter` | `'100 500'` |
| `TIME_RANGE` | `video-prompter` | `'0:05 0:30'` |
| `OUTPUT` | both | `'results/demo'` |
| `ALPHA` | both | `0.45` |
| `DEVICE` | both | `cpu` or `cuda` |
| `CHUNK_SPREAD` | `video-prompter` | `even` |
| `KEEP_TEMP` | `video-prompter` | `1` (any non-empty value) |
| `VIDEO_RES` | `run-*` examples | `480p`, `720p`, `1080p` |
| `ARGS` | all | extra flags passed through |

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
    --video assets/videos/sample.mp4 \
    --prompts "person" "tennis racket" \
    --output results/tennis_demo

# Point prompt
uv run python video_prompter.py \
    --video assets/videos/sample.mp4 \
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

The high-level entry point is `Sam3` (alias for `Sam3API`):

```python
from sam3 import Sam3
sam3 = Sam3()
```

### Image scenarios

| Method | Description |
|---|---|
| `process_image_with_prompts(image_path, prompts, ...)` | Segment by text prompts |
| `process_image_with_boxes(image_path, boxes, ...)` | Segment inside bounding boxes |

### Video scenarios

| Method | Description |
|---|---|
| `process_video_with_prompts(video_path, prompts, ...)` | Segment & track by text prompts |

Video processing supports:

- `propagation_direction` — `"forward"`, `"backward"`, or `"both"` (default)
- Automatic chunking when memory is limited (see below)

Lower-level access is available through `ImageProcessor`, `VideoProcessor`,
`ChunkProcessor`, and the driver classes in `sam3/drivers.py`.

---

## Video Chunking

When a video is too large to fit in memory the framework automatically splits it
into overlapping chunks, processes each chunk independently, and stitches the
results back together.

**How it works:**

1. `MemoryManager` computes how many frames fit in available RAM (CPU) or VRAM (GPU).
2. The video is split into chunks with configurable overlap (default 5 frames).
3. Each chunk is segmented and tracked independently.
4. At chunk boundaries, masks from the overlap region are matched using IoU and
   object IDs are remapped so they stay consistent across the full video.

**Key parameters** (set in `config.json` or `sam3/__globals.py`):

| Parameter | Default | Meaning |
|---|---|---|
| `ram_usage_percent` | 0.45 | Fraction of free RAM budget for frames (override in `__globals.py`) |
| `min_frames` | 25 | Minimum frames per chunk |
| `chunk_overlap` | 5 | Overlap frames between chunks (`DEFAULT_MIN_CHUNK_OVERLAP` in `__globals.py`) |
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

## Profiling

SAM3-CPU ships with a built-in **profiler** that measures wall-clock **execution
time** and **memory consumption** (RSS) for any decorated function.  This is
valuable for:

- Identifying bottlenecks (model loading vs. inference vs. post-processing)
- Tracking memory growth across processing stages
- Comparing CPU vs. GPU performance
- Validating that optimisations have the intended effect

### Enabling the profiler

The profiler is **disabled by default** (zero overhead).  Enable it by passing
the `--profile` flag to any script:

```bash
# Profile image segmentation
uv run python image_prompter.py \
    --images assets/images/truck.jpg --prompts truck --profile

# Profile video segmentation
uv run python video_prompter.py \
    --video clip.mp4 --prompts person --profile
```

Or toggle it programmatically:

```python
import sam3.__globals
sam3.__globals.ENABLE_PROFILING = True
```

### Using `@profile()` in your own code

```python
from sam3.utils.profiler import profile

@profile()
def my_heavy_function():
    # ... expensive work ...
    return result
```

When profiling is enabled each decorated call prints a summary:

```
[PROFILED] my_heavy_function | Time: 3.142857s | Memory Used: 128.000000 MB
```

### Output files

Results are appended to two files in the working directory:

| File | Format | Content |
|---|---|---|
| `profile_results.json` | JSON array | One object per call with `function_name`, `timestamp`, `execution_time_seconds`, `memory_used_MB`, `total_process_memory_MB` |
| `profile_results.txt` | Plain text | One line per call — human-readable summary |

**Example `profile_results.json`:**

```json
[
    {
        "function_name": "_build_model",
        "timestamp": "2026-02-16T21:07:27.912279",
        "execution_time_seconds": 6.283576,
        "memory_used_MB": 6750.577,
        "total_process_memory_MB": 7515.546
    },
    {
        "function_name": "inference",
        "timestamp": "2026-02-16T21:07:33.132962",
        "execution_time_seconds": 5.214927,
        "memory_used_MB": 51.323,
        "total_process_memory_MB": 7566.967
    }
]
```

**Example `profile_results.txt`:**

```
_build_model | Time: 6.283576 s | Memory Used: 6750.577 MB | Total Memory: 7515.546 MB
inference    | Time: 5.214927 s | Memory Used: 51.323 MB  | Total Memory: 7566.967 MB
```

A standalone demo is available at `examples/profiler_example.py`:

```bash
uv run python examples/profiler_example.py --profile
```

---

## Configuration

Runtime defaults live in `config.json` (loaded by the wrapper class) and
compile-time constants in `sam3/__globals.py`.  Key settings:

```jsonc
{
  "ram_usage_percent": 0.45,
  "min_frames": 25,
  "chunk_overlap": 1,
  "prefetch_threshold": 0.90,
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
│   └── sam/                   # SAM core modules
│
├── examples/                  # Runnable example scripts
├── notebook/                  # Jupyter notebooks
├── tests/                     # Pytest test suite
├── scripts/                   # Utility scripts
├── assets/                    # Sample images & videos
└── README.md
```

---

## Testing

The test suite lives in `tests/` and uses **pytest**.  Tests are designed to run
**without the SAM3 model** — they exercise helper functions, IoU logic,
stitching, and metadata generation using synthetic data.

### Test files

| File | What it covers |
|---|---|
| `test_iou_matching.py` | IoU computation, mask matching between chunks |
| `test_cross_chunk.py` | Cross-chunk ID remapping and continuity |
| `test_video_prompter.py` | Video prompter helpers — stitching, overlay, timestamp parsing, range resolution, object tracking |
| `test_all_scenarios.py` | End-to-end scenarios (requires model + assets — skipped when unavailable) |
| `conftest.py` | Shared fixtures: asset paths, temp directories, markers |

### Running tests

```bash
# Full suite
uv run python -m pytest tests/ -v

# Single file
uv run python -m pytest tests/test_video_prompter.py -v

# Single test class or method
uv run python -m pytest tests/test_video_prompter.py::TestParseTimestamp -v
uv run python -m pytest tests/test_video_prompter.py::TestBuildObjectTracking::test_frame_offset -v

# Skip slow / model-dependent tests
uv run python -m pytest tests/ -v -m "not slow"
```

### Injecting your own test data

The unit tests create **synthetic videos and masks** on-the-fly (via OpenCV), so
no real assets are needed.  To test with your own data:

1. **Add assets** — place images in `assets/images/` and videos in
   `assets/videos/`.  The fixtures in `conftest.py` will pick them up.
2. **Write a fixture** — add a new `@pytest.fixture` in `conftest.py` that
   points to your file:
    ```python
    @pytest.fixture(scope="session")
    def my_custom_video(assets_dir):
        path = assets_dir / "videos" / "my_clip.mp4"
        if not path.exists():
            pytest.skip(f"Custom video not found: {path}")
        return str(path)
    ```
3. **Use the fixture** in your test function:
    ```python
    def test_my_clip(my_custom_video, temp_output_dir):
        # run processing and assert on results
        ...
    ```

Available markers: `@pytest.mark.slow`, `@pytest.mark.gpu`,
`@pytest.mark.image`, `@pytest.mark.video`.

---

## Known Limitations

- **Cross-chunk object ID reassignment** — Videos are processed in memory-sized
  chunks.  Object IDs are kept consistent across chunk boundaries using
  IoU-based mask matching.  However, if an object **disappears in the middle of
  a chunk** and **reappears in a later chunk** with no overlapping mask in the
  boundary region, it will be assigned a **new ID**.  The same real-world object
  may therefore be counted multiple times in the tracking metadata.  This is an
  inherent trade-off of chunk-based processing without a global re-identification
  step.

- **CPU inference speed** — Running the full SAM 3 model on CPU is
  significantly slower than GPU.  Use `--frame-range` / `--time-range` to
  process only the segment you need, or consider GPU acceleration for
  production workloads.

- **macOS / Windows support** — The project is tested primarily on Linux.
  macOS works for most workflows but may have edge-case differences with
  ffmpeg builds.  Windows support is not yet validated (see
  [Future Work](#future-work)).

---

## Contributing

Contributions are welcome!  Whether it's a bug fix, a new feature, or improved
documentation — we'd love your input.

### How to contribute

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
3. **Make your changes** and add tests where appropriate.
4. **Open a Pull Request** against `main` with a clear description of the
   change and its motivation.

### Becoming a contributor / maintainer

If you'd like to be added as a regular contributor or maintainer, please
[open a GitHub Issue](https://github.com/rhubarb-ai/sam3-cpu/issues/new)
with the title **"Contributor request"** and a brief description of your
background and intended contributions.  GitHub Issues are the preferred
channel for all project-level discussions.

### Guidelines

- Follow existing code style and conventions.
- Keep PRs focused — one logical change per PR.
- Ensure `uv run python -m pytest tests/ -v` passes before submitting.
- Update documentation (especially this README) for user-facing changes.

---

## Future Work

- **Docker support** — Provide a `Dockerfile` + `docker-compose.yml` for
  reproducible, one-command deployment.
- **Full macOS compatibility** — Validate and fix edge cases on Intel and
  Apple Silicon Macs.
- **Windows compatibility** — Test and adapt for native Windows and WSL
  environments.
- **Performance optimisation** — Further speed-up through model quantisation,
  batched inference, and frame-level parallelism to reduce wall-clock time
  on CPU.
- **CI/CD pipeline** — GitHub Actions for automated testing, linting, and
  release packaging.

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
    author = {Prashant Aparajeya, Ankuj Arora},
    year   = {2026},
    url    = {https://github.com/rhubarb-ai/sam3-cpu},
}
```

---

## License

This project is released under the [MIT License](LICENSE).

The underlying SAM 3 model weights are subject to Meta's [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

---

## Contributors

**Dr Prashant Aparajeya**
Email: p.aparajeya@gmail.com
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)
Github Profile: [paparajeya](https://github.com/paparajeya)

**Dr Ankuj Arora**
Email: ankujarora@gmail.com
GitHub: [rhubarb-ai/sam3-cpu](https://github.com/rhubarb-ai/sam3-cpu)
Github Profile: [ankuj](https://github.com/ankuj)