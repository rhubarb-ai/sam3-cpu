#!/usr/bin/env python3
"""
Simple SAM3 Video Segmentation Demo

Demonstrates the correct SAM3 API usage for video segmentation.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add sam3 to path
sys.path.insert(0, str(Path(__file__).parent))

import sam3
from sam3.model_builder import build_sam3_video_predictor_cpu


def propagate_in_video(predictor, session_id):
    """Collect outputs for all frames"""
    outputs_per_frame = {}
    print("Propagating through video...")
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]
        if frame_idx % 10 == 0:
            print(f"  Processed frame {frame_idx}")
    return outputs_per_frame


def visualize_frame_with_masks(frame, outputs, output_path):
    """Visualize a frame with mask overlays"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original frame
    axes[0].imshow(frame)
    axes[0].set_title("Original Frame")
    axes[0].axis('off')
    
    # Frame with masks
    axes[1].imshow(frame)
    axes[1].set_title(f"With Masks ({len(outputs)} objects)")
    axes[1].axis('off')
    
    # Overlay masks (if any)
    if outputs:
        # Create random colors for objects
        np.random.seed(42)
        colors = np.random.random((len(outputs), 3))
        
        for idx, output in enumerate(outputs):
            # Extract mask (outputs format varies, this is a simplified version)
            # You may need to adjust this based on actual output structure
            if 'mask' in output:
                mask = output['mask']
                axes[1].imshow(mask, alpha=0.5, cmap='jet')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple SAM3 video segmentation demo")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt (e.g., 'person')")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of CPU workers")
    parser.add_argument("--save-all", action="store_true", help="Save all frames (not just first)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SAM3 Video Segmentation Demo")
    print("=" * 60)
    print(f"Input  : {args.input}")
    print(f"Prompt : '{args.prompt}'")
    print(f"Workers: {args.workers}")
    print(f"Output : {output_dir}")
    print("=" * 60)
    
    # Setup SAM3
    print("\nLoading SAM3 model...")
    sam3_root = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_root, "assets/bpe_simple_vocab_16e6.txt.gz")
    
    predictor = build_sam3_video_predictor_cpu(
        bpe_path=bpe_path,
        num_workers=args.workers
    )
    print("✓ Model loaded")
    
    # Load video for visualization
    print("\nLoading video...")
    cap = cv2.VideoCapture(args.input)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"✓ Loaded {len(video_frames)} frames")
    
    # Start session
    print("\nStarting inference session...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=args.input,
        )
    )
    session_id = response["session_id"]
    print(f"✓ Session ID: {session_id}")
    
    try:
        # Add text prompt on frame 0
        print(f"\nAdding prompt '{args.prompt}' on frame 0...")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=args.prompt,
            )
        )
        
        frame_0_outputs = response["outputs"]
        print(f"✓ Detected {len(frame_0_outputs)} object(s) on frame 0")
        
        # Visualize frame 0
        if len(video_frames) > 0:
            output_path = output_dir / "frame_0_result.png"
            visualize_frame_with_masks(video_frames[0], frame_0_outputs, output_path)
        
        # Propagate through video
        print("\n" + "=" * 60)
        all_outputs = propagate_in_video(predictor, session_id)
        print(f"✓ Processed {len(all_outputs)} frames")
        print("=" * 60)
        
        # Save visualizations for selected frames
        if args.save_all:
            frames_to_save = range(len(all_outputs))
        else:
            # Save every 30th frame
            frames_to_save = range(0, len(all_outputs), 30)
        
        print(f"\nSaving {len(list(frames_to_save))} frame visualizations...")
        for frame_idx in frames_to_save:
            if frame_idx in all_outputs and frame_idx < len(video_frames):
                output_path = output_dir / f"frame_{frame_idx:04d}_result.png"
                visualize_frame_with_masks(
                    video_frames[frame_idx],
                    all_outputs[frame_idx],
                    output_path
                )
        
        # Summary
        print("\n" + "=" * 60)
        print("✓ Processing complete!")
        print(f"Total frames  : {len(all_outputs)}")
        print(f"Output dir    : {output_dir}")
        print("=" * 60)
    
    finally:
        # Close session
        print("\nClosing session...")
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        print("✓ Session closed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
