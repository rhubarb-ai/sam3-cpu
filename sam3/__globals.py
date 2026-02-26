import torch
import os
import sys
import sam3
from sam3.utils.logger import get_logger, LOG_LEVELS

os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

LOG_LEVEL = LOG_LEVELS["DEBUG"]
logger = get_logger(__name__, level=LOG_LEVEL)

logger.info(f"Python executable: {sys.executable}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for --profile flag
if '--profile' in sys.argv:
    ENABLE_PROFILING = True
    logger.info("🔍 Profiling ENABLED\n")
else:
    ENABLE_PROFILING = False
    logger.info("⚡ Profiling DISABLED (use --profile to enable)\n")

SAM3_ROOT = os.path.join(os.path.dirname(sam3.__file__))
BPE_PATH = os.path.join(SAM3_ROOT, "assets/bpe_simple_vocab_16e6.txt.gz")

# Video processing defaults
DEFAULT_MIN_VIDEO_FRAMES = 25
DEFAULT_MIN_CHUNK_OVERLAP = 5

SUPPORTED_VIDEO_FORMATS = ('.mp4', '.avi', '.mov', '.mkv')

# Memory management
IMAGE_INFERENCE_MB = 6760
VIDEO_INFERENCE_MB = 6900
TENSOR_SIZE_BYTES = 1008*1008*3*4 # Approximate size of a 1008x1008 RGB tensor in bytes

# Memory usage for chunking (percentage of available memory to use)
RAM_USAGE_PERCENT = 0.45   # Use 45% of available RAM for CPU video chunking (conservative)
# RAM_USAGE_PERCENT = 0.65   # Use 65% of available RAM for CPU video chunking (conservative)
VRAM_USAGE_PERCENT = 0.65  # Use 65% of available VRAM for GPU video chunking (aggressive, GPU memory is dedicated)

MEMORY_SAFETY_MULTIPLIER = 1.5  # Require 1.5x estimated memory for safety (reduced from 3x)
CPU_MEMORY_RESERVE_PERCENT = 0.3  # Reserve 30% for OS
GPU_MEMORY_RESERVE_PERCENT = 0.05  # Reserve 5% for display

# Parallel processing
PARALLEL_CHUNK_THRESHOLD = 0.90  # Start loading next chunk at 90% completion

# Output settings
DEFAULT_PROPAGATION_DIRECTION = "both"
DEFAULT_NUM_WORKERS = 1  # Use all available CPU cores by default
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Post-processing settings
CHUNK_MASK_MATCHING_IOU_THRESHOLD = 0.75  # IoU threshold for matching masks across chunks (75% - expecting high values with lossless PNG storage)

# Directory settings
TEMP_DIR = "/tmp/sam3-cpu" if DEVICE.type == "cpu" else "/tmp/sam3-gpu"
os.makedirs(TEMP_DIR, exist_ok=True)

DEFAULT_OUTPUT_DIR = os.path.join("./results")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

PROFILE_OUTPUT_JSON = "profile_results.json"
PROFILE_OUTPUT_TXT = "profile_results.txt"