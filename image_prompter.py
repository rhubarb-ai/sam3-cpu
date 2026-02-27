#!/usr/bin/env python3
"""
SAM3 Image Prompter

Process one or more images with text prompts, click points, and/or bounding
boxes for segmentation.  Validates available memory before processing and
saves masks, overlays, and metadata to a structured output directory.

Usage:
    # Single image with text prompts
    python image_prompter.py --images photo.jpg --prompts person car

    # Multiple images with a bounding box
    python image_prompter.py --images a.jpg b.jpg --bbox 100 150 200 300

    # Click point on an image
    python image_prompter.py --images scene.jpg --points 320,240 --point-labels 1

    # Combine prompts + bbox, custom output and overlay alpha
    python image_prompter.py --images img.jpg --prompts dog --bbox 50 60 180 220 \\
        --output results/dogs --alpha 0.45

    # Force CPU even if GPU is available
    python image_prompter.py --images img.jpg --prompts cat --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Memory validation helpers
# ---------------------------------------------------------------------------

def _format_bytes(b: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _print_table(rows: List[List[str]], col_widths: Optional[List[int]] = None):
    """Print a simple ASCII table."""
    if not rows:
        return
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    print(sep)
    for idx, row in enumerate(rows):
        line = "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(row, col_widths)) + "|"
        print(line)
        if idx == 0:
            print(sep)
    print(sep)


def _validate_memory(image_path: Path, device: str) -> Dict[str, Any]:
    """Check whether there is enough memory to process this image.

    Returns a dict with memory stats and a boolean ``can_process``.
    """
    from sam3.memory_manager import MemoryManager
    from sam3.utils.helpers import ram_stat, vram_stat
    from sam3.__globals import IMAGE_INFERENCE_MB, RAM_USAGE_PERCENT, VRAM_USAGE_PERCENT

    img = Image.open(image_path)
    width, height = img.size
    img.close()

    mm = MemoryManager()
    max_frames = mm.compute_memory_safe_frames(width, height, device, type="image")

    if device == "cuda":
        mem = vram_stat()
        pct = VRAM_USAGE_PERCENT
        available = mem["free"]
    else:
        mem = ram_stat()
        pct = RAM_USAGE_PERCENT
        available = mem["available"]

    info = {
        "image": str(image_path),
        "resolution": f"{width}x{height}",
        "device": device,
        "total_memory": mem["total"],
        "available_memory": available,
        "inference_overhead_mb": IMAGE_INFERENCE_MB,
        "usage_percent": pct,
        "can_process": max_frames > 0,
    }

    if not info["can_process"]:
        needed = IMAGE_INFERENCE_MB * 1024**2 + width * height * 3
        deficit = needed - available
        info["deficit_bytes"] = max(deficit, 0)

    return info


def _show_memory_table(info: Dict[str, Any]):
    """Print memory validation results as a table."""
    rows = [
        ["Metric", "Value"],
        ["Image", info["image"]],
        ["Resolution", info["resolution"]],
        ["Device", info["device"]],
        ["Total memory", _format_bytes(info["total_memory"])],
        ["Available memory", _format_bytes(info["available_memory"])],
        ["Inference overhead", f"{info['inference_overhead_mb']} MB"],
    ]
    if info["can_process"]:
        rows.append(["Status", "\033[92m✓ Sufficient memory\033[0m"])
    else:
        deficit = info.get("deficit_bytes", 0)
        rows.append(["Status", f"\033[91m✗ Insufficient memory (need {_format_bytes(deficit)} more)\033[0m"])
    _print_table(rows)


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------

def _create_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.5,
                    color: tuple = (30, 144, 255)) -> Image.Image:
    """Blend a coloured mask overlay onto the original image."""
    img_arr = np.array(image.convert("RGB")).copy()
    binary = mask > 0
    if binary.ndim == 3 and binary.shape[0] == 1:
        binary = binary[0]
    overlay = np.zeros_like(img_arr)
    overlay[binary] = color
    img_arr[binary] = (
        (1 - alpha) * img_arr[binary] + alpha * overlay[binary]
    ).astype(np.uint8)
    return Image.fromarray(img_arr)


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def _process_image_with_text_prompts(
    driver,
    image: Image.Image,
    prompts: List[str],
    image_name: str,
    output_dir: Path,
    alpha: float,
):
    """Process an image with one or more text prompts."""
    from sam3.utils.helpers import sanitize_filename

    processor, inference_state = driver.inference(image)

    for prompt in prompts:
        safe = sanitize_filename(prompt)
        prompt_dir = output_dir / safe
        prompt_dir.mkdir(parents=True, exist_ok=True)

        inference_state = driver.prompt_and_predict(processor, inference_state, prompt)

        masks = inference_state.get("masks")
        scores = inference_state.get("scores")
        boxes = inference_state.get("boxes")

        num = len(scores) if scores is not None else 0
        print(f"  Prompt '{prompt}': found {num} object(s)")

        meta_objects = []
        for i in range(num):
            m = masks[i]
            m_np = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
            if m_np.ndim == 3 and m_np.shape[0] == 1:
                m_np = m_np[0]
            m_u8 = (m_np.astype(np.uint8) * 255)

            # Save mask PNG
            mask_path = prompt_dir / f"object_{i}_mask.png"
            Image.fromarray(m_u8, mode="L").save(mask_path)

            # Save overlay
            overlay = _create_overlay(image, m_np, alpha=alpha)
            overlay.save(prompt_dir / f"object_{i}_overlay.png")

            score_val = float(scores[i]) if scores is not None else None
            box_val = boxes[i].tolist() if boxes is not None and len(boxes) > i else None
            meta_objects.append({
                "object_id": i,
                "score": score_val,
                "box": box_val,
                "mask_file": f"object_{i}_mask.png",
                "overlay_file": f"object_{i}_overlay.png",
            })

        # Save prompt metadata
        meta = {
            "image": image_name,
            "prompt": prompt,
            "num_objects": num,
            "objects": meta_objects,
        }
        with open(prompt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)


def _process_image_with_bbox(
    driver,
    image: Image.Image,
    boxes: List[List[float]],
    box_labels: List[int],
    image_name: str,
    output_dir: Path,
    alpha: float,
):
    """Process an image with bounding box prompts."""
    bbox_dir = output_dir / "bbox"
    bbox_dir.mkdir(parents=True, exist_ok=True)

    processor, inference_state = driver.inference(image)

    inference_state = driver.prompt_multi_box_with_labels(
        image=image,
        processor=processor,
        inference_state=inference_state,
        boxes_xywh=boxes,
        box_labels=box_labels,
    )

    masks = inference_state.get("masks")
    scores = inference_state.get("scores")

    num = len(scores) if scores is not None else 0
    print(f"  Bounding box: found {num} object(s)")

    meta_objects = []
    for i in range(num):
        m = masks[i]
        m_np = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
        if m_np.ndim == 3 and m_np.shape[0] == 1:
            m_np = m_np[0]
        m_u8 = (m_np.astype(np.uint8) * 255)

        mask_path = bbox_dir / f"object_{i}_mask.png"
        Image.fromarray(m_u8, mode="L").save(mask_path)

        overlay = _create_overlay(image, m_np, alpha=alpha)
        overlay.save(bbox_dir / f"object_{i}_overlay.png")

        meta_objects.append({
            "object_id": i,
            "score": float(scores[i]) if scores is not None else None,
            "mask_file": f"object_{i}_mask.png",
            "overlay_file": f"object_{i}_overlay.png",
        })

    meta = {
        "image": image_name,
        "boxes": boxes,
        "box_labels": box_labels,
        "num_objects": num,
        "objects": meta_objects,
    }
    with open(bbox_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def _process_image_with_points(
    driver,
    image: Image.Image,
    points: List[List[float]],
    point_labels: List[int],
    image_name: str,
    output_dir: Path,
    alpha: float,
):
    """Process an image with click-point prompts via geometric prompt."""
    import torch
    from sam3.model.box_ops import box_xywh_to_cxcywh

    pts_dir = output_dir / "points"
    pts_dir.mkdir(parents=True, exist_ok=True)

    processor, inference_state = driver.inference(image)

    # Apply point prompt via geometric prompt interface
    processor.reset_all_prompts(inference_state)

    width, height = image.size
    for pt, lbl in zip(points, point_labels):
        # Normalise point to [0, 1]
        norm_pt = [pt[0] / width, pt[1] / height]
        inference_state = processor.add_geometric_prompt(
            state=inference_state, point=norm_pt, label=lbl
        )

    masks = inference_state.get("masks")
    scores = inference_state.get("scores")

    num = len(scores) if scores is not None else 0
    print(f"  Points: found {num} object(s)")

    meta_objects = []
    for i in range(num):
        m = masks[i]
        m_np = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
        if m_np.ndim == 3 and m_np.shape[0] == 1:
            m_np = m_np[0]
        m_u8 = (m_np.astype(np.uint8) * 255)

        mask_path = pts_dir / f"object_{i}_mask.png"
        Image.fromarray(m_u8, mode="L").save(mask_path)

        overlay = _create_overlay(image, m_np, alpha=alpha)
        overlay.save(pts_dir / f"object_{i}_overlay.png")

        meta_objects.append({
            "object_id": i,
            "score": float(scores[i]) if scores is not None else None,
            "mask_file": f"object_{i}_mask.png",
            "overlay_file": f"object_{i}_overlay.png",
        })

    meta = {
        "image": image_name,
        "points": points,
        "point_labels": point_labels,
        "num_objects": num,
        "objects": meta_objects,
    }
    with open(pts_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_bbox(raw: List[str]) -> List[List[float]]:
    """Parse --bbox values into a list of 4-coord bounding boxes.

    Accepts either 4 values (one box) or multiples of 4.
    """
    vals = [float(v) for v in raw]
    if len(vals) % 4 != 0:
        print(f"\033[91m✗ --bbox requires multiples of 4 values (x y w h), got {len(vals)}\033[0m")
        sys.exit(1)
    return [vals[i : i + 4] for i in range(0, len(vals), 4)]


def parse_points(raw: List[str]) -> List[List[float]]:
    """Parse --points values like '320,240' into [[320, 240], ...]."""
    pts = []
    for p in raw:
        parts = p.replace(" ", "").split(",")
        if len(parts) != 2:
            print(f"\033[91m✗ Each point must be x,y — got '{p}'\033[0m")
            sys.exit(1)
        pts.append([float(parts[0]), float(parts[1])])
    return pts


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Image Prompter — segment images with text, click points, or bounding boxes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image_prompter.py --images photo.jpg --prompts person car
  python image_prompter.py --images a.jpg b.jpg --bbox 100 150 200 300
  python image_prompter.py --images scene.jpg --points 320,240 --point-labels 1
  python image_prompter.py --images img.jpg --prompts dog --alpha 0.45 --device cpu
        """,
    )

    parser.add_argument(
        "--images", nargs="+", required=True,
        help="One or more image file paths",
    )
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Text prompts for segmentation (e.g. person car)",
    )
    parser.add_argument(
        "--points", nargs="+", default=None,
        help="Click points as x,y pairs (e.g. 320,240 500,300)",
    )
    parser.add_argument(
        "--point-labels", nargs="+", type=int, default=None,
        help="Labels for each point (1=positive, 0=negative). Must match --points count.",
    )
    parser.add_argument(
        "--bbox", nargs="+", type=float, default=None,
        help="Bounding box(es) as x y w h (multiples of 4 for multiple boxes)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay alpha for mask visualisation (0.0–1.0, default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Force device (auto-detected if omitted)",
    )

    args = parser.parse_args()

    # --- Validation: at least one prompt type --------------------------------
    if not args.prompts and not args.points and not args.bbox:
        print("\033[91m✗ At least one of --prompts, --points, or --bbox must be provided.\033[0m")
        sys.exit(1)

    # --- Validate point-labels match points ----------------------------------
    if args.points and args.point_labels:
        points = parse_points(args.points)
        if len(args.point_labels) != len(points):
            print(f"\033[91m✗ --point-labels count ({len(args.point_labels)}) must match --points count ({len(points)})\033[0m")
            sys.exit(1)
    elif args.points:
        points = parse_points(args.points)
        args.point_labels = [1] * len(points)  # default all positive
    else:
        points = None

    boxes = parse_bbox([str(v) for v in args.bbox]) if args.bbox else None

    # --- Device ---------------------------------------------------------------
    from sam3.__globals import DEVICE as DEFAULT_DEVICE
    device = args.device or DEFAULT_DEVICE.type

    # --- Validate image files -------------------------------------------------
    image_paths = [Path(p) for p in args.images]
    for p in image_paths:
        if not p.is_file():
            print(f"\033[91m✗ Image file not found: {p}\033[0m")
            sys.exit(1)

    # --- Header ---------------------------------------------------------------
    print()
    print("=" * 70)
    print("  SAM3 Image Prompter")
    print("=" * 70)
    print(f"  Images  : {len(image_paths)}")
    if args.prompts:
        print(f"  Prompts : {', '.join(args.prompts)}")
    if points:
        print(f"  Points  : {points}")
    if boxes:
        print(f"  Boxes   : {boxes}")
    print(f"  Device  : {device}")
    print(f"  Output  : {args.output}")
    print(f"  Alpha   : {args.alpha}")
    print("=" * 70)
    print()

    # --- Process each image ---------------------------------------------------
    driver = None
    all_memory_info: List[Dict[str, Any]] = []

    for idx, img_path in enumerate(image_paths):
        image_name = img_path.stem
        print(f"[{idx + 1}/{len(image_paths)}] {img_path.name}")

        # Memory validation
        mem_info = _validate_memory(img_path, device)
        all_memory_info.append(mem_info)
        _show_memory_table(mem_info)

        if not mem_info["can_process"]:
            deficit = mem_info.get("deficit_bytes", 0)
            print(f"\033[91m✗ Cannot process {img_path.name} — need {_format_bytes(deficit)} more memory\033[0m")
            print()
            continue

        # Output directory: [output]/images/[image_name]/
        image_output = Path(args.output) / "images" / image_name
        image_output.mkdir(parents=True, exist_ok=True)

        # Lazy-load driver
        if driver is None:
            print("  Loading model...")
            from sam3.drivers import Sam3ImageDriver
            driver = Sam3ImageDriver()
            print("  Model loaded.\n")

        image = Image.open(img_path)

        # --- Text prompts (loop per prompt, separate folders) -----------------
        if args.prompts:
            _process_image_with_text_prompts(
                driver, image, args.prompts, image_name, image_output, args.alpha,
            )

        # --- Bounding boxes ---------------------------------------------------
        if boxes:
            _process_image_with_bbox(
                driver, image, boxes, [1] * len(boxes),
                image_name, image_output, args.alpha,
            )

        # --- Click points -----------------------------------------------------
        if points:
            _process_image_with_points(
                driver, image, points, args.point_labels,
                image_name, image_output, args.alpha,
            )

        # Save overall image metadata
        img_meta = {
            "image_name": image_name,
            "image_path": str(img_path.resolve()),
            "resolution": f"{image.width}x{image.height}",
            "prompts": args.prompts,
            "points": points,
            "boxes": boxes,
            "memory": mem_info,
        }
        with open(image_output / "metadata.json", "w") as f:
            json.dump(img_meta, f, indent=2, default=str)

        # Cleanup between images to free memory
        if driver is not None:
            driver.cleanup()

        print(f"  ✓ Results saved to {image_output}\n")

    # Cleanup
    if driver is not None:
        driver.cleanup()

    print("=" * 70)
    print("  ✓ Image processing complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
