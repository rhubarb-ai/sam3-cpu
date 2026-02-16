#!/usr/bin/env python3
"""
SAM3 CPU Demo Script

Unified demo for SAM3 image and video segmentation on CPU.
Supports single images, videos, and batch processing.

Usage:
    # Image segmentation
    python demo.py --input image.jpg --prompt "cat"
    
    # Video segmentation
    python demo.py --input video.mp4 --prompt "person" --output results/
    
    # Batch processing
    python demo.py --batch images/ --prompt "object" --output results/
    
    # Use custom config
    python demo.py --input video.mp4 --prompt "car" --config config.json
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent))

from sam3.wrapper import Sam3Wrapper


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SAM3 CPU Demo - Unified segmentation for images and videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image with text prompt
  python demo.py --input cat.jpg --prompt "cat"
  
  # Video with text prompt
  python demo.py --input video.mp4 --prompt "person walking"
  
  # Batch process images
  python demo.py --batch images/ --prompt "object" --output results/
  
  # Use custom configuration
  python demo.py --input video.mp4 --prompt "car" --config my_config.json
  
  # Specify custom BPE path
  python demo.py --input image.jpg --prompt "dog" --bpe-path /path/to/bpe.txt.gz
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Path to input file (image or video)"
    )
    input_group.add_argument(
        "--batch", "-b",
        type=str,
        help="Path to directory for batch processing"
    )
    
    # Prompt
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        required=True,
        help="Text prompt for segmentation (e.g., 'cat', 'person walking')"
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    # Model configuration
    parser.add_argument(
        "--bpe-path",
        type=str,
        help="Path to BPE tokenizer file (default: assets/bpe_simple_vocab_16e6.txt.gz)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config.json file"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        help="Number of workers (default: auto-detect)"
    )
    
    # Display options
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display results (only save to disk)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    return parser.parse_args()


def process_image(wrapper: Sam3Wrapper, image_path: str, prompt: str, output_dir: Path):
    """Process a single image"""
    print(f"\n{'='*60}")
    print(f"Processing Image: {Path(image_path).name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load predictor if not already loaded
    if wrapper.predictor is None:
        raise RuntimeError("Predictor not loaded. Call wrapper.load_predictor() first.")
    
    # Perform inference
    print(f"Prompt: '{prompt}'")
    print("Running inference...")
    
    # Note: This is a placeholder - actual SAM3 image inference API may differ
    # You'll need to adapt this based on the actual SAM3 API
    try:
        # Create inference session
        inference_state = wrapper.predictor.init_state(image_rgb)
        
        # Add text prompt
        _, obj_ids, mask_logits = wrapper.predictor.add_new_prompt(
            inference_state=inference_state,
            prompt=prompt,
            obj_id=1
        )
        
        # Get masks
        masks = (mask_logits[0] > 0.0).cpu().numpy()
        
        print(f"✓ Generated {len(masks)} mask(s)")
        
        # Visualize and save
        output_path = output_dir / f"{Path(image_path).stem}_result.png"
        visualize_masks(image_rgb, masks, output_path)
        print(f"✓ Saved result to: {output_path}")
        
        return masks
    
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def process_video(wrapper: Sam3Wrapper, video_path: str, prompt: str, output_dir: Path):
    """Process a video with chunking and memory management"""
    print(f"\n{'='*60}")
    print(f"Processing Video: {Path(video_path).name}")
    print(f"{'='*60}")
    
    # Prepare video (analyze, chunk, setup workspace)
    prep_result = wrapper.prepare_video(video_path)
    video_meta = prep_result['video_meta']
    chunks = prep_result['chunks']
    workspace = prep_result['workspace']
    
    # Load predictor if not already loaded
    if wrapper.predictor is None:
        raise RuntimeError("Predictor not loaded. Call wrapper.load_predictor() first.")
    
    # Open video for reading
    cap = cv2.VideoCapture(video_path)
    
    try:
        # Process each chunk
        all_masks = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n{'='*60}")
            print(f"Processing Chunk {i+1}/{len(chunks)}")
            print(f"Frames: {chunk.start_frame} - {chunk.end_frame}")
            print(f"{'='*60}")
            
            # Read frames for this chunk
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, chunk.start_frame)
            
            for frame_idx in range(chunk.num_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            print(f"Loaded {len(frames)} frames")
            
            # Create inference state for chunk
            inference_state = wrapper.predictor.init_state(
                video=np.array(frames),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )
            
            # Add prompt on first frame of first chunk
            if i == 0:
                print(f"Prompt: '{prompt}'")
                _, obj_ids, mask_logits = wrapper.predictor.add_new_prompt(
                    inference_state=inference_state,
                    frame_idx=0,
                    prompt=prompt,
                    obj_id=1
                )
            
            # Propagate masks through chunk
            print("Propagating masks...")
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in wrapper.predictor.propagate_in_video(
                inference_state
            ):
                video_segments[out_frame_idx] = {
                    obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, obj_id in enumerate(out_obj_ids)
                }
            
            all_masks.append(video_segments)
            print(f"✓ Processed {len(video_segments)} frames")
        
        # Save results
        print(f"\n{'='*60}")
        print(f"Saving Results")
        print(f"{'='*60}")
        
        output_video_path = output_dir / f"{Path(video_path).stem}_result.mp4"
        save_video_with_masks(video_path, all_masks, output_video_path, video_meta.fps)
        print(f"✓ Saved result video to: {output_video_path}")
        
        return all_masks
    
    finally:
        cap.release()
        
        # Cleanup workspace
        if not wrapper.verbose:
            wrapper.cleanup_video_workspace()


def process_batch(wrapper: Sam3Wrapper, batch_dir: str, prompt: str, output_dir: Path):
    """Process a batch of images"""
    batch_path = Path(batch_dir)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in batch_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No images found in {batch_dir}")
    
    print(f"\n{'='*60}")
    print(f"Batch Processing: {len(image_files)} images")
    print(f"{'='*60}")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing {image_file.name}")
        try:
            masks = process_image(wrapper, str(image_file), prompt, output_dir)
            results.append((image_file, masks))
        except Exception as e:
            print(f"✗ Failed to process {image_file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Batch Complete: {len(results)}/{len(image_files)} successful")
    print(f"{'='*60}")
    
    return results


def visualize_masks(image: np.ndarray, masks: np.ndarray, output_path: Path):
    """Visualize masks overlaid on image"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Image with masks
    axes[1].imshow(image)
    if len(masks) > 0:
        # Combine all masks
        combined_mask = masks.sum(axis=0) > 0
        axes[1].imshow(combined_mask, alpha=0.5, cmap='jet')
    axes[1].set_title("With Masks")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_video_with_masks(input_video: str, all_masks: list, output_path: Path, fps: float):
    """Save video with masks overlaid"""
    cap = cv2.VideoCapture(input_video)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        frame_idx = 0
        chunk_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get mask for this frame
            if chunk_idx < len(all_masks):
                chunk_masks = all_masks[chunk_idx]
                if frame_idx in chunk_masks:
                    # Overlay masks
                    for obj_id, mask in chunk_masks[frame_idx].items():
                        # Create colored overlay
                        overlay = np.zeros_like(frame)
                        overlay[mask[0]] = [0, 255, 0]  # Green mask
                        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            out.write(frame)
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize wrapper
        wrapper = Sam3Wrapper(
            config_path=args.config,
            verbose=not args.quiet
        )
        
        # Load predictor
        wrapper.load_predictor(
            bpe_path=args.bpe_path,
            num_workers=args.workers
        )
        
        # Process based on input type
        if args.input:
            input_path = Path(args.input)
            
            # Check if video or image
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            
            if input_path.suffix.lower() in video_extensions:
                process_video(wrapper, str(input_path), args.prompt, output_dir)
            else:
                process_image(wrapper, str(input_path), args.prompt, output_dir)
        
        elif args.batch:
            process_batch(wrapper, args.batch, args.prompt, output_dir)
        
        print(f"\n{'='*60}")
        print("✓ Processing complete!")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
