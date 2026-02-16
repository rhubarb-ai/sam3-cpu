#!/usr/bin/env python3
"""
SAM3 Image Processing Example

Simple example demonstrating image segmentation with the SAM3 wrapper.
"""

import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from sam3.wrapper import Sam3Wrapper


def main():
    # Configuration
    image_path = "assets/images/example_image.jpg"
    prompt = "cat"
    
    print("SAM3 Image Segmentation Example")
    print("="*60)
    print(f"Image : {image_path}")
    print(f"Prompt: '{prompt}'")
    print("="*60)
    
    # Initialize wrapper
    wrapper = Sam3Wrapper(verbose=True)
    
    # Load SAM3 predictor
    print("\nLoading SAM3 Model...")
    wrapper.load_predictor()
    
    # Load image
    print(f"\nLoading image: {image_path}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image")
    except:
        print(f"\nError: Image file not found or invalid: {image_path}")
        print("Please update image_path in the script to point to a valid image file.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    
    # Initialize inference state
    print("\nInitializing inference state...")
    inference_state = wrapper.predictor.init_state(image_rgb)
    
    # Add text prompt
    print(f"Adding prompt: '{prompt}'")
    _, obj_ids, mask_logits = wrapper.predictor.add_new_prompt(
        inference_state=inference_state,
        prompt=prompt,
        obj_id=1
    )
    
    # Get masks
    masks = (mask_logits[0] > 0.0).cpu().numpy()
    print(f"✓ Generated {len(masks)} mask(s)")
    
    # Visualize results
    print("\nVisualizing results...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Mask only
    if len(masks) > 0:
        combined_mask = masks.sum(axis=0) > 0
        axes[1].imshow(combined_mask, cmap='gray')
        axes[1].set_title("Segmentation Mask")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image_rgb)
        axes[2].imshow(combined_mask, alpha=0.5, cmap='jet')
        axes[2].set_title("Overlay")
        axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save result
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{Path(image_path).stem}_result.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved result to: {output_path}")
    
    # Show result
    plt.show()
    
    print("\n" + "="*60)
    print("✓ Example complete!")
    print("="*60)


if __name__ == "__main__":
    main()
