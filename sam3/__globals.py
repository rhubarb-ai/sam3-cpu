import os
import sys

import torch

import sam3
from sam3.utils.logger import LOG_LEVELS, get_logger

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

LOG_LEVEL = LOG_LEVELS["DEBUG"]
logger = get_logger(__name__, level=LOG_LEVEL)

logger.info(f"Python executable: {sys.executable}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for --profile flag
if "--profile" in sys.argv:
    ENABLE_PROFILING = True
    logger.info("🔍 Profiling ENABLED\n")
else:
    ENABLE_PROFILING = False
    logger.info("⚡ Profiling DISABLED (use --profile to enable)\n")

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__))
BPE_PATH = os.path.join(SAM3_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz")

# Video processing defaults
DEFAULT_MIN_VIDEO_FRAMES = 25
DEFAULT_MIN_CHUNK_OVERLAP = 1

SUPPORTED_VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv")

# Memory management
IMAGE_INFERENCE_MB = 6760
VIDEO_INFERENCE_MB = 6900
TENSOR_SIZE_BYTES = 1008 * 1008 * 3 * 4  # Approximate size of a 1008x1008 RGB tensor in bytes

# Model state overhead: the SAM3 tracker stores per-frame feature maps, cached
# masks for N tracked objects, positional encodings, and memory-bank tensors.
# The multiplier is applied on top of the raw pixel cost (width × height × 4)
# to estimate the full per-frame GPU memory footprint.
# Calibrated empirically: 480p → ~40 MB/frame, 1080p → ~70 MB/frame.
MODEL_STATE_MULTIPLIER = 4.5

# Memory usage for chunking (percentage of available memory to use)
RAM_USAGE_PERCENT = 0.975  # Use 97.5% of available RAM for CPU video chunking (conservative)
VRAM_USAGE_PERCENT = 0.975  # Use 97.5% of available VRAM for initial GPU chunk planning
# (the IntraChunkMonitor enforces tighter per-frame
# limits during actual propagation — see below)
CPU_CORES_PERCENT = 0.90  # Legacy — kept for reference; the --cpu-utilisation CLI flag now controls thread count
DEFAULT_CPU_UTILISATION = 100  # Default CPU utilisation percentage (50-100%). 100% = use all logical cores.

MEMORY_SAFETY_MULTIPLIER = 1.5  # Require 1.5x estimated memory for safety (reduced from 3x)
CPU_MEMORY_RESERVE_PERCENT = 0.3  # Reserve 30% for OS
GPU_MEMORY_RESERVE_PERCENT = 0.05  # Reserve 5% for display

# GPU memory offloading: when True, the SAM3 tracker stores per-frame
# inference state (mask-memory features, predicted masks, object pointers)
# on CPU RAM instead of VRAM.  This prevents the monotonic VRAM growth
# observed during long video propagation at the cost of a small speed
# reduction (~10-15% lower tracking FPS).  Has no effect when device is CPU.
OFFLOAD_TRACKER_STATE_TO_CPU = True

# Target memory utilisation for the adaptive chunk manager.  The planner
# aims to size chunks so the bounding resource (VRAM, or RAM when
# offloading) peaks close to this fraction of the effective limit.
# Kept above the VRAM_SOFT_LIMIT_PCT so the IntraChunkMonitor can still
# intervene with a predictive stop on the rare overshoot.
GPU_TARGET_UTILISATION_PCT = 0.90

# Intra-chunk memory guard thresholds (enforced per-frame during propagation)
# These are the live safety limits.  VRAM_USAGE_PERCENT above is only for
# initial chunk *planning*; the guard below is the runtime enforcement.
VRAM_SOFT_LIMIT_PCT = 0.85  # Warn + predict frames-to-limit
VRAM_HARD_LIMIT_PCT = 0.975  # Immediate stop — 2.5% headroom prevents actual OOM

# RAM guard thresholds (same dual-threshold design as VRAM)
RAM_SOFT_LIMIT_PCT = 0.85  # Warn when process RSS reaches 85% of available RAM
RAM_HARD_LIMIT_PCT = 0.975  # Immediate stop — leave 2.5% headroom for OS/other

# Cross-chunk mask injection: when True, inject the previous chunk's last-frame
# masks as conditioning frames at the start of the next chunk.  This gives the
# tracker memory of where objects were, improving continuity without re-detection.
CROSS_CHUNK_MASK_INJECTION = True

# RAM guard for carry data: the carry dict holds numpy masks from every prompt
# across all processed chunks.  On very long videos with many objects this can
# consume significant RAM.  When either limit is hit, the oldest prompt entries
# in carry are dropped to free memory.
CARRY_MAX_RAM_USAGE_PCT = 0.98  # Drop carry entries when RAM usage reaches 98%
CARRY_MIN_FREE_RAM_GB = 1.0  # Drop carry entries when free RAM drops below 1 GB

# Memory bank: stores tracker spatial memory features (maskmem_features,
# maskmem_pos_enc) from the last N frames of each chunk.  After reset, these
# are re-injected at negative frame indices so the tracker starts the new
# chunk with contextual memory from the previous chunk rather than a blank
# state.  num_maskmem=7 means the tracker uses at most 6 non-cond memory
# frames, so 6 is the effective useful maximum.
MEMORY_BANK_MAX_FRAMES = 6

# Maximum exponent step for the background memory predictor's exponential
# backoff schedule.  After this many iterations the polling interval is
# already well past poll_max, so further exponentiation is pointless and
# would overflow float64 on very long runs (>5000 frames).
POLL_BACKOFF_MAX_STEP = 50

# Parallel processing
PARALLEL_CHUNK_THRESHOLD = 0.90  # Start loading next chunk at 90% completion

# Output settings
DEFAULT_PROPAGATION_DIRECTION = "both"
DEFAULT_NUM_WORKERS = 1  # Use all available CPU cores by default
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Post-processing settings
CHUNK_MASK_MATCHING_IOU_THRESHOLD = (
    0.75  # IoU threshold for matching masks across chunks (75% - expecting high values with lossless PNG storage)
)

# Directory settings
TEMP_DIR = "/tmp/sam3-cpu" if DEVICE.type == "cpu" else "/tmp/sam3-gpu"
os.makedirs(TEMP_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = os.path.join("./results")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

PROFILE_OUTPUT_JSON = "profile_results.json"
PROFILE_OUTPUT_TXT = "profile_results.txt"
