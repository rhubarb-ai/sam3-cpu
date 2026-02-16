#!/usr/bin/env python3
"""
SAM3 Video Processing Example

Simple example demonstrating video processing with the SAM3 wrapper.
Shows memory management, chunking, and mask propagation.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from sam3.wrapper import Sam3Wrapper


def main():
    # Configuration
    video_path = "assets/videos/example_video.mp4"
    prompt = "person"
    
    print("SAM3 Video Processing Example")
    print("="*60)
    print(f"Video : {video_path}")
    print(f"Prompt: '{prompt}'")
    print("="*60)
    
    # Initialize wrapper with custom settings
    wrapper = Sam3Wrapper(
        ram_usage_percent=0.25,    # Use 25% of available RAM
        min_frames=25,              # Minimum frames required
        chunk_overlap=1,            # 1 frame overlap between chunks
        tmp_base="/tmp/sam3-cpu",  # Temporary workspace
        verbose=True                # Print detailed logs
    )
    
    # Load SAM3 predictor
    print("\n" + "="*60)
    print("Loading SAM3 Model")
    print("="*60)
    wrapper.load_predictor(
        num_workers=1        # Number of CPU workers
    )
    
    # Prepare video (analyze, chunk, create workspace)
    print("\n" + "="*60)
    print("Preparing Video")
    print("="*60)
    
    try:
        prep_result = wrapper.prepare_video(video_path)
    except FileNotFoundError:
        print(f"\nError: Video file not found: {video_path}")
        print("Please update video_path in the script to point to a valid video file.")
        return
    except ValueError as e:
        print(f"\nError: {e}")
        return
    
    video_meta = prep_result['video_meta']
    chunks = prep_result['chunks']
    workspace = prep_result['workspace']
    
    # Process video
    print("\n" + "="*60)
    print("Processing Video")
    print("="*60)
    
    # Open video
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    try:
        # Process first chunk as example
        chunk = chunks[0]
        
        print(f"\nProcessing chunk 0/{len(chunks)}")
        print(f"Frames: {chunk.start_frame} - {chunk.end_frame} ({chunk.num_frames} frames)")
        
        # Read frames
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, chunk.start_frame)
        
        for _ in range(chunk.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        print(f"Loaded {len(frames)} frames")
        
        # Initialize inference state
        print("Initializing inference state...")
        inference_state = wrapper.predictor.init_state(
            video=np.array(frames),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )
        
        # Add prompt on first frame
        print(f"Adding prompt: '{prompt}'")
        _, obj_ids, mask_logits = wrapper.predictor.add_new_prompt(
            inference_state=inference_state,
            frame_idx=0,
            prompt=prompt,
            obj_id=1
        )
        
        print(f"Prompt added for object ID: {obj_ids}")
        
        # Propagate masks through video
        print("\nPropagating masks through video...")
        video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in wrapper.predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, obj_id in enumerate(out_obj_ids)
            }
            print(f"  Frame {out_frame_idx}: {len(out_obj_ids)} object(s)")
        
        print(f"\n✓ Successfully processed {len(video_segments)} frames")
        
        # Save results (optional)
        results_dir = workspace / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save first frame with mask overlay as example
        if 0 in video_segments:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original frame
            axes[0].imshow(frames[0])
            axes[0].set_title("Original Frame")
            axes[0].axis('off')
            
            # Frame with mask
            axes[1].imshow(frames[0])
            for obj_id, mask in video_segments[0].items():
                axes[1].imshow(mask[0], alpha=0.5, cmap='jet')
            axes[1].set_title("With Segmentation")
            axes[1].axis('off')
            
            output_path = results_dir / "example_result.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved visualization to: {output_path}")
        
    finally:
        cap.release()
    
    # Show workspace info
    print("\n" + "="*60)
    print("Workspace Information")
    print("="*60)
    workspace_info = wrapper.get_workspace_info()
    print(f"Location: {workspace_info['workspace']}")
    print(f"Chunks  : {len(workspace_info['chunks'])}")
    
    # Cleanup option
    print("\n" + "="*60)
    print("Cleanup")
    print("="*60)
    response = input("Clean up workspace? [y/N]: ")
    
    if response.lower() == 'y':
        wrapper.cleanup_video_workspace()
        print("✓ Workspace cleaned")
    else:
        print(f"Workspace preserved at: {workspace}")
        print(f"To clean up later, run: scripts/linux/cleanup_sam3_tmp.sh")
    
    print("\n" + "="*60)
    print("✓ Example complete!")
    print("="*60)


if __name__ == "__main__":
    main()
