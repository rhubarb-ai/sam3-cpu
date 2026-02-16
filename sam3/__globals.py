import torch
import os
import sys
import sam3
from sam3.logger import get_logger

logger = get_logger(__name__)

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

