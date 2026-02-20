"""
Quick test to verify lossless chunk and mask implementation.
"""

import subprocess
import sys
from pathlib import Path

def test_lossless_implementation():
    print("="*70)
    print("TESTING LOSSLESS IMPLEMENTATION")
    print("="*70)
    
    # Clean up old results
    results_dir = Path("results/sample_lossless")
    if results_dir.exists():
        print(f"\n🧹 Cleaning up old results: {results_dir}")
        import shutil
        shutil.rmtree(results_dir)
    
    print("\n" + "="*70)
    print("STEP 1: Process Video with Lossless Chunks + PNG Masks")
    print("="*70)
    
    # Run main.py with sample video
    cmd = [
        "uv", "run", "main.py",
        "--video", "assets/videos/sample.mp4",
        "--prompts", "player",  # Just one prompt for quick testing
        "--output", str(results_dir)
    ]
    
    print(f"\nRunning: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Processing failed with exit code {result.returncode}")
        return False
    
    print("\n✅ Processing complete!")
    
    # Check outputs
    print("\n" + "="*70)
    print("STEP 2: Verify PNG Mask Structure")
    print("="*70)
    
    chunks_dir = results_dir / "temp_files" / "chunks"
    
    if not chunks_dir.exists():
        print(f"❌ Chunks directory not found: {chunks_dir}")
        return False
    
    # Check chunk_0
    chunk_0_masks = chunks_dir / "chunk_0" / "masks" / "player"
    
    if not chunk_0_masks.exists():
        print(f"❌ Masks directory not found: {chunk_0_masks}")
        return False
    
    # Check for PNG directories
    object_dirs = list(chunk_0_masks.glob("object_*"))
    print(f"\n✅ Found {len(object_dirs)} object directories in chunk_0")
    
    for obj_dir in object_dirs[:2]:  # Show first 2
        png_files = list(obj_dir.glob("frame_*.png"))
        print(f"   {obj_dir.name}: {len(png_files)} PNG files")
    
    # Check if old MP4 files exist (they shouldn't)
    old_mp4_files = list(chunk_0_masks.glob("object_*.mp4"))
    if old_mp4_files:
        print(f"\n⚠️  Found {len(old_mp4_files)} old MP4 files (expected 0)")
    else:
        print(f"\n✅ No old MP4 files found (correct!)")
    
    # Run post-processing
    print("\n" + "="*70)
    print("STEP 3: Run Post-Processing with PNG Masks")
    print("="*70)
    
    # Create test postprocessing script
    test_script = f"""
import json
from pathlib import Path
from sam3.postprocessor import VideoPostProcessor

results_dir = Path("{results_dir}")
chunks_dir = results_dir / "temp_files" / "chunks"

# Load metadata
with open(results_dir / "metadata" / "video_metadata.json", "r") as f:
    video_metadata = json.load(f)

# Load chunk results (simplified)
chunk_results = []
chunk_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir() and d.name.startswith("chunk_")])

for chunk_dir in chunk_dirs:
    chunk_id = int(chunk_dir.name.split("_")[1])
    masks_dir = chunk_dir / "masks" / "player"
    
    if not masks_dir.exists():
        continue
    
    object_dirs = list(masks_dir.glob("object_*"))
    object_ids = [int(d.name.split("_")[1]) for d in object_dirs]
    
    chunk_results.append({{
        "chunk_id": chunk_id,
        "prompts": {{
            "player": {{
                "prompt": "player",
                "num_objects": len(object_ids),
                "object_ids": object_ids,
                "masks_dir": str(masks_dir)
            }}
        }}
    }})

print(f"Loaded {{len(chunk_results)}} chunks")

# Create postprocessor
postprocessor = VideoPostProcessor(
    video_name="sample_lossless",
    chunk_results=chunk_results,
    video_metadata=video_metadata,
    chunks_temp_dir=chunks_dir,
    masks_output_dir=results_dir / "masks",
    meta_output_dir=results_dir / "metadata"
)

# Process
postprocessor.process(["player"])

print("\\n✅ Post-processing complete!")
"""
    
    test_postproc_path = Path("test_postproc_lossless.py")
    with open(test_postproc_path, "w") as f:
        f.write(test_script)
    
    result = subprocess.run(["uv", "run", str(test_postproc_path)], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n❌ Post-processing failed")
        return False
    
    # Check IoU results
    print("\n" + "="*70)
    print("STEP 4: Check IoU Values")
    print("="*70)
    
    print("\n✅ If you see IoU > 0.9 in the logs above, lossless implementation is working!")
    print("✅ If you see fewer player objects, matching is more accurate!")
    
    return True


if __name__ == "__main__":
    success = test_lossless_implementation()
    sys.exit(0 if success else 1)
