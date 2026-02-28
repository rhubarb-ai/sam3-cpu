#!/usr/bin/env python3
"""
SAM3 Video Prompter

Process a single video with text prompts, click points, and/or mask images
for segmentation.  Automatically chunks the video based on available memory,
runs cross-chunk IoU-based ID remapping for object continuity, and produces
stitched mask videos and overlay.

Usage:
    # Text prompt
    python video_prompter.py --video clip.mp4 --prompts person ball

    # Click points on first frame
    python video_prompter.py --video clip.mp4 --points 320,240 --point-labels 1

    # Mask image as prompt
    python video_prompter.py --video clip.mp4 --masks mask.png

    # Process a specific segment (frames or time)
    python video_prompter.py --video clip.mp4 --prompts player --frame-range 100 500
    python video_prompter.py --video clip.mp4 --prompts player --time-range 4.0 20.0
    python video_prompter.py --video clip.mp4 --prompts player --time-range 00:01:30 00:03:00

    # Custom options
    python video_prompter.py --video clip.mp4 --prompts player \\
        --output results/match --alpha 0.4 --device cpu --keep-temp
"""

# ---- Force-CPU guard (must run before ANY torch/sam3 import) ----
import os as _os, sys as _sys
if '--device' in _sys.argv:
    _i = _sys.argv.index('--device')
    if _i + 1 < len(_sys.argv) and _sys.argv[_i + 1].lower() == 'cpu':
        _os.environ['CUDA_VISIBLE_DEVICES'] = ''
# -----------------------------------------------------------------

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(b: float) -> str:
    """Human-readable byte size."""
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"


def _table(rows: List[List[str]]):
    """Print a simple ASCII table."""
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    for i, row in enumerate(rows):
        print("|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(row, widths)) + "|")
        if i == 0:
            print(sep)
    print(sep)


# ---------------------------------------------------------------------------
# Time / frame range parsing
# ---------------------------------------------------------------------------

def _parse_timestamp(value: str) -> float:
    """Parse a timestamp string to seconds.

    Accepts:
        - Plain float/int:  "4.5"  -> 4.5
        - MM:SS:            "1:30" -> 90.0
        - HH:MM:SS:        "0:01:30" -> 90.0
    """
    try:
        return float(value)
    except ValueError:
        pass
    parts = value.split(":")
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Cannot parse timestamp: '{value}'")


def _resolve_range(
    video_path: Path,
    frame_range: Optional[Tuple[int, int]],
    time_range: Optional[Tuple[str, str]],
) -> Optional[Tuple[int, int]]:
    """Convert frame_range or time_range to an (start_frame, end_frame) tuple.

    Returns *None* when the full video should be processed.
    """
    if frame_range is not None:
        return (frame_range[0], frame_range[1])
    if time_range is not None:
        from sam3.utils.ffmpeglib import ffmpeg_lib
        info = ffmpeg_lib.get_video_info(str(video_path))
        fps = info["fps"]
        start_sec = _parse_timestamp(time_range[0])
        end_sec = _parse_timestamp(time_range[1])
        return (int(start_sec * fps), min(int(end_sec * fps), info["nb_frames"] - 1))
    return None


def _extract_subclip(
    video_path: Path, start_frame: int, end_frame: int, temp_dir: Path
) -> Path:
    """Extract a sub-clip from *video_path* covering [start_frame, end_frame]."""
    from sam3.utils.ffmpeglib import ffmpeg_lib
    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / f"subclip_{start_frame}_{end_frame}.mp4"
    ffmpeg_lib.create_video_chunk(str(video_path), str(out), start_frame, end_frame)
    return out


# ---------------------------------------------------------------------------
# Memory validation
# ---------------------------------------------------------------------------

def _validate_video_memory(video_path: Path, device: str) -> Dict[str, Any]:
    """Check if there is enough memory to process at least MIN_VIDEO_FRAMES.

    Returns a dict with memory stats, chunk plan, and ``can_process`` flag.
    """
    from sam3.memory_manager import MemoryManager
    from sam3.utils.helpers import ram_stat, vram_stat
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.__globals import (
        VIDEO_INFERENCE_MB,
        RAM_USAGE_PERCENT,
        VRAM_USAGE_PERCENT,
        DEFAULT_MIN_VIDEO_FRAMES,
    )

    video_info = ffmpeg_lib.get_video_info(str(video_path))
    if video_info is None:
        return {"can_process": False, "error": "Could not read video metadata"}

    mm = MemoryManager()
    max_frames = mm.compute_memory_safe_frames(
        video_info["width"], video_info["height"], device, type="video"
    )

    if device == "cuda":
        mem = vram_stat()
        pct = VRAM_USAGE_PERCENT
        available = mem["free"]
    else:
        mem = ram_stat()
        pct = RAM_USAGE_PERCENT
        available = mem["available"]

    info: Dict[str, Any] = {
        "video": str(video_path),
        "resolution": f"{video_info['width']}x{video_info['height']}",
        "total_frames": video_info["nb_frames"],
        "fps": round(video_info.get("fps", 25), 2),
        "duration_s": round(video_info.get("duration", 0), 2),
        "device": device,
        "total_memory": mem["total"],
        "available_memory": available,
        "inference_overhead_mb": VIDEO_INFERENCE_MB,
        "max_frames_per_chunk": max_frames,
        "can_process": max_frames >= DEFAULT_MIN_VIDEO_FRAMES,
    }

    if not info["can_process"]:
        frame_bytes = video_info["width"] * video_info["height"] * 3
        needed = (
            VIDEO_INFERENCE_MB * 1024**2
            + DEFAULT_MIN_VIDEO_FRAMES * frame_bytes
        )
        info["deficit_bytes"] = max(needed - available, 0)
    else:
        info["deficit_bytes"] = 0

    return info


def _show_video_memory_table(info: Dict[str, Any]):
    """Print video memory validation table."""
    rows = [
        ["Metric", "Value"],
        ["Video", info.get("video", "?")],
        ["Resolution", info.get("resolution", "?")],
        ["Frames / FPS", f"{info.get('total_frames', '?')} / {info.get('fps', '?')}"],
        ["Duration", f"{info.get('duration_s', '?')} s"],
        ["Device", info.get("device", "?")],
        ["Total memory", _fmt(info.get("total_memory", 0))],
        ["Available memory", _fmt(info.get("available_memory", 0))],
        ["Inference overhead", f"{info.get('inference_overhead_mb', 0)} MB"],
        ["Max frames/chunk", str(info.get("max_frames_per_chunk", 0))],
    ]
    if info["can_process"]:
        rows.append(["Status", "\033[92m✓ Sufficient memory\033[0m"])
    else:
        deficit = info.get("deficit_bytes", 0)
        rows.append(["Status", f"\033[91m✗ Insufficient memory (need {_fmt(deficit)} more)\033[0m"])
    _table(rows)


# ---------------------------------------------------------------------------
# Chunk plan
# ---------------------------------------------------------------------------

def _make_chunk_plan(
    video_path: Path, device: str, chunk_spread: str = "default"
) -> tuple:
    """Create a memory-safe chunk plan.

    Returns (video_metadata, chunk_list).
    """
    from sam3.memory_manager import memory_manager

    metadata, chunks = memory_manager.chunk_plan_video(
        str(video_path), device=device, chunk_spread=chunk_spread
    )
    return metadata, chunks


# ---------------------------------------------------------------------------
# Mask extraction helpers
# ---------------------------------------------------------------------------

def _extract_last_frame_masks(
    result_prompt: dict,
    object_ids: set,
) -> Dict[int, np.ndarray]:
    """Extract masks from the last frame of propagation results."""
    if not result_prompt:
        return {}
    last_idx = max(result_prompt.keys())
    output = result_prompt[last_idx]
    out_ids = output.get("out_obj_ids", [])
    if isinstance(out_ids, np.ndarray):
        out_ids = out_ids.tolist()
    masks = {}
    for oid in object_ids:
        if oid in out_ids:
            idx = out_ids.index(oid)
            m = output["out_binary_masks"][idx]
            masks[oid] = (m.astype(np.uint8) * 255)
    return masks


# ---------------------------------------------------------------------------
# IoU + ID remapping (same algorithm as ChunkProcessor)
# ---------------------------------------------------------------------------

def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = a > 127 if a.dtype == np.uint8 else a.astype(bool)
    mb = b > 127 if b.dtype == np.uint8 else b.astype(bool)
    inter = np.logical_and(ma, mb).sum()
    union = np.logical_or(ma, mb).sum()
    return float(inter / union) if union > 0 else 0.0


def _match_and_remap(
    result_prompt: dict,
    object_ids: set,
    prev_masks: Dict[int, np.ndarray],
    global_next_id: int,
    iou_threshold: float = 0.25,
):
    """Greedy IoU matching and ID remapping.

    Returns (remapped_result, remapped_ids, id_mapping, updated_next_id).
    """
    if not result_prompt:
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
        return {}, set(mapping.values()), mapping, global_next_id

    first_idx = min(result_prompt.keys())
    first_out = result_prompt.get(first_idx)
    if first_out is None:
        mapping = {o: o for o in object_ids}
        return result_prompt, object_ids, mapping, global_next_id

    out_ids = first_out.get("out_obj_ids", [])
    if isinstance(out_ids, np.ndarray):
        out_ids = out_ids.tolist()

    first_masks = {}
    for oid in out_ids:
        idx = out_ids.index(oid)
        first_masks[oid] = (first_out["out_binary_masks"][idx].astype(np.uint8) * 255)

    if not prev_masks:
        # First chunk — identity mapping
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
    else:
        pairs = []
        for nid, nm in first_masks.items():
            for pid, pm in prev_masks.items():
                iou = _compute_iou(nm, pm)
                if iou >= iou_threshold:
                    pairs.append((iou, nid, pid))
        pairs.sort(reverse=True)

        mapping = {}
        used = set()
        for iou, nid, pid in pairs:
            if nid in mapping or pid in used:
                continue
            mapping[nid] = pid
            used.add(pid)
            print(f"      Matched obj_{nid} → global_{pid} (IoU={iou:.3f})")

        for oid in sorted(object_ids):
            if oid not in mapping:
                mapping[oid] = global_next_id
                print(f"      New obj_{oid} → global_{global_next_id}")
                global_next_id += 1

    # Apply mapping to all frames
    remapped = {}
    for fidx, output in result_prompt.items():
        ids = output.get("out_obj_ids", [])
        ids_list = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)
        new_out = dict(output)
        new_out["out_obj_ids"] = np.array(
            [mapping.get(o, o) for o in ids_list], dtype=np.int64
        )
        remapped[fidx] = new_out

    return remapped, set(mapping.values()), mapping, global_next_id


# ---------------------------------------------------------------------------
# Mask saving helpers
# ---------------------------------------------------------------------------

def _save_chunk_masks(
    result_prompt: dict,
    object_ids: set,
    masks_dir: Path,
    width: int,
    height: int,
    total_frames: int,
):
    """Save per-object per-frame PNG masks."""
    for oid in object_ids:
        obj_dir = masks_dir / f"object_{oid}"
        obj_dir.mkdir(parents=True, exist_ok=True)

    for fidx in range(total_frames):
        output = result_prompt.get(fidx)
        for oid in object_ids:
            mask_u8 = np.zeros((height, width), dtype=np.uint8)
            if output is not None:
                out_ids = output.get("out_obj_ids", [])
                if isinstance(out_ids, np.ndarray):
                    out_ids = out_ids.tolist()
                if oid in out_ids:
                    idx = out_ids.index(oid)
                    m = output["out_binary_masks"][idx]
                    if m.any():
                        mask_u8 = (m.astype(np.uint8) * 255)
            png = masks_dir / f"object_{oid}" / f"frame_{fidx:06d}.png"
            Image.fromarray(mask_u8, mode="L").save(png, compress_level=1)


# ---------------------------------------------------------------------------
# Stitching and overlay
# ---------------------------------------------------------------------------

def _stitch_masks_to_video(
    chunks_dir: Path,
    prompt_name: str,
    object_ids: set,
    chunk_infos: list,
    overlap: int,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
):
    """Stitch per-chunk PNG masks into per-object mask videos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    black = np.zeros((height, width), dtype=np.uint8)

    for oid in sorted(object_ids):
        out_path = output_dir / f"object_{oid}_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)

        for ci, cinfo in enumerate(chunk_infos):
            chunk_id = cinfo["chunk"]
            skip = overlap if ci > 0 else 0
            obj_mask_dir = (
                chunks_dir / f"chunk_{chunk_id}" / "masks" / prompt_name / f"object_{oid}"
            )
            if not obj_mask_dir.exists():
                # Object not present in this chunk — write black frames
                chunk_len = cinfo["end"] - cinfo["start"] + 1
                for _ in range(chunk_len - skip):
                    writer.write(black)
                continue
            pngs = sorted(obj_mask_dir.glob("frame_*.png"))
            for png in pngs[skip:]:
                frame = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if frame is not None:
                    writer.write(frame)

        writer.release()
        print(f"    Saved mask video: {out_path.name}")


def _create_overlay_video(
    video_path: Path,
    mask_videos: List[Path],
    output_path: Path,
    alpha: float = 0.5,
):
    """Overlay coloured masks onto the original video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

    # Open mask video readers
    mask_caps = [cv2.VideoCapture(str(p)) for p in mask_videos if p.exists()]

    # Colour palette
    colours = [
        (30, 144, 255), (255, 50, 50), (50, 205, 50),
        (255, 165, 0), (148, 103, 189), (255, 215, 0),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for i, mc in enumerate(mask_caps):
            ret_m, mask_frame = mc.read()
            if ret_m and mask_frame is not None:
                if mask_frame.ndim == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                binary = mask_frame > 127
                c = colours[i % len(colours)]
                for ch in range(3):
                    overlay[:, :, ch][binary] = (
                        (1 - alpha) * frame[:, :, ch][binary]
                        + alpha * c[ch]
                    ).astype(np.uint8)
        writer.write(overlay)

    for mc in mask_caps:
        mc.release()
    cap.release()
    writer.release()
    print(f"    Saved overlay video: {output_path.name}")


# ---------------------------------------------------------------------------
# Per-object tracking metadata
# ---------------------------------------------------------------------------

def _build_object_tracking(
    mask_dir: Path,
    object_ids: set,
    fps: float,
    frame_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Scan stitched mask videos and compute per-object presence info.

    For each object, determines the first and last frame where the mask is
    non-empty (>= 1 % of pixels active) and translates to timestamps.

    Args:
        mask_dir: Directory containing ``object_{id}_mask.mp4`` files.
        object_ids: Set of global object IDs to scan.
        fps: Video FPS (used for timestamp conversion).
        frame_offset: If a sub-clip was extracted, offset added to frame
            numbers so they refer to the *original* video's timeline.

    Returns:
        List of dicts, one per object, sorted by object ID.
    """
    tracking: List[Dict[str, Any]] = []
    area_threshold = 0.01  # 1 % of frame area

    for oid in sorted(object_ids):
        mp4 = mask_dir / f"object_{oid}_mask.mp4"
        if not mp4.exists():
            tracking.append({
                "object_id": oid,
                "first_frame": None,
                "last_frame": None,
                "total_frames_active": 0,
            })
            continue

        cap = cv2.VideoCapture(str(mp4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixels = max(w * h, 1)

        first_frame = None
        last_frame = None
        active_count = 0

        for fidx in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            active = (frame > 127).sum()
            if active / total_pixels >= area_threshold:
                if first_frame is None:
                    first_frame = fidx
                last_frame = fidx
                active_count += 1

        cap.release()

        # Translate to original-video coordinates
        abs_first = (first_frame + frame_offset) if first_frame is not None else None
        abs_last = (last_frame + frame_offset) if last_frame is not None else None

        entry: Dict[str, Any] = {
            "object_id": oid,
            "first_frame": abs_first,
            "last_frame": abs_last,
            "total_frames_active": active_count,
            "total_frames": total,
        }
        if abs_first is not None:
            entry["first_timestamp"] = round(abs_first / fps, 3)
            entry["last_timestamp"] = round(abs_last / fps, 3)
            entry["duration_s"] = round((abs_last - abs_first + 1) / fps, 3)

            def _ts(sec: float) -> str:
                m, s = divmod(sec, 60)
                h, m = divmod(int(m), 60)
                return f"{h:02d}:{int(m):02d}:{s:06.3f}"

            entry["first_timecode"] = _ts(abs_first / fps)
            entry["last_timecode"] = _ts(abs_last / fps)
        else:
            entry["first_timestamp"] = None
            entry["last_timestamp"] = None
            entry["duration_s"] = 0.0
            entry["first_timecode"] = None
            entry["last_timecode"] = None

        tracking.append(entry)

    return tracking


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def _process_video(
    video_path: Path,
    prompts: Optional[List[str]],
    points: Optional[List[List[float]]],
    point_labels: Optional[List[int]],
    mask_paths: Optional[List[Path]],
    output_dir: Path,
    device: str,
    alpha: float,
    chunk_spread: str,
    keep_temp: bool,
    frame_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[str, str]] = None,
):
    """Full video processing pipeline."""
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.utils.helpers import sanitize_filename
    from sam3.__globals import TEMP_DIR, DEFAULT_MIN_CHUNK_OVERLAP

    video_name = video_path.stem

    # ----- Resolve range & extract sub-clip if needed -----
    resolved = _resolve_range(video_path, frame_range, time_range)
    original_video = video_path
    frame_offset = 0  # offset into the original video for metadata
    if resolved is not None:
        sf, ef = resolved
        frame_offset = sf
        print(f"Extracting segment: frames {sf}–{ef} ...")
        temp_subclip_dir = Path(TEMP_DIR) / video_name / "subclip"
        video_path = _extract_subclip(original_video, sf, ef, temp_subclip_dir)
        print(f"  Sub-clip: {video_path}  ({ef - sf + 1} frames)\n")

    # ----- Memory check -----
    mem_info = _validate_video_memory(video_path, device)
    _show_video_memory_table(mem_info)
    print()

    if not mem_info["can_process"]:
        deficit = mem_info.get("deficit_bytes", 0)
        print(f"\033[91m✗ Cannot process video — need {_fmt(deficit)} more memory.\033[0m")
        sys.exit(1)

    # ----- Chunk plan -----
    print("Creating chunk plan...")
    video_metadata, chunk_list = _make_chunk_plan(video_path, device, chunk_spread)
    n_chunks = len(chunk_list)
    print(f"  {n_chunks} chunk(s), {mem_info.get('max_frames_per_chunk', '?')} frames/chunk")
    print()

    # ----- Directories -----
    temp_base = Path(TEMP_DIR) / video_name
    chunks_dir = temp_base / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    video_output = output_dir / video_name
    video_output.mkdir(parents=True, exist_ok=True)

    # Save video metadata
    meta_dir = video_output / "metadata"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "video_metadata.json", "w") as f:
        json.dump(video_metadata, f, indent=2)
    with open(meta_dir / "memory_info.json", "w") as f:
        json.dump(mem_info, f, indent=2, default=str)

    # ----- Load model once -----
    print("Loading SAM3 video model...")
    from sam3.drivers import Sam3VideoDriver
    driver = Sam3VideoDriver(device=device)
    print("Model loaded.\n")

    # ----- Validate mask dimensions (if masks provided) -----
    if mask_paths:
        vid_info = ffmpeg_lib.get_video_info(str(video_path))
        for mp in mask_paths:
            m_img = Image.open(mp)
            if m_img.size != (vid_info["width"], vid_info["height"]):
                print(
                    f"\033[91m✗ Mask {mp.name} size {m_img.size} does not match "
                    f"video {vid_info['width']}x{vid_info['height']}\033[0m"
                )
                sys.exit(1)

    # ----- Process chunks -----
    overlap = DEFAULT_MIN_CHUNK_OVERLAP
    # Carry-forward state across chunks (per prompt)
    carry: Dict[str, Dict[int, np.ndarray]] = {}
    global_next_ids: Dict[str, int] = {}
    all_object_ids: Dict[str, set] = {}

    for ci, cinfo in enumerate(chunk_list):
        chunk_id = cinfo["chunk"]
        start_frame = cinfo["start"]
        end_frame = cinfo["end"]

        print(f"── Chunk {ci + 1}/{n_chunks} (frames {start_frame}–{end_frame}) ──")

        chunk_dir = chunks_dir / f"chunk_{chunk_id}"
        chunk_dir.mkdir(exist_ok=True)

        # Extract chunk video
        if n_chunks == 1:
            chunk_video = video_path  # use original
        else:
            chunk_video = chunk_dir / f"chunk_{chunk_id}.mp4"
            ffmpeg_lib.create_video_chunk(
                str(video_path), str(chunk_video), start_frame, end_frame
            )

        # Get chunk frame count
        cap = cv2.VideoCapture(str(chunk_video))
        chunk_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Start session
        session_id = driver.start_session(video_path=str(chunk_video))

        try:
            # ----- Text prompts -----
            if prompts:
                for prompt in prompts:
                    safe = sanitize_filename(prompt)
                    print(f"  Prompt: '{prompt}'")

                    driver.reset_session(session_id)
                    driver.add_prompt(session_id, prompt)

                    result, obj_ids, frame_objs = driver.propagate_in_video(
                        session_id, propagation_direction="both"
                    )

                    prev_masks = carry.get(prompt, {})
                    gnid = global_next_ids.get(prompt, 0)

                    result, obj_ids, mapping, gnid = _match_and_remap(
                        result, obj_ids, prev_masks, gnid
                    )
                    global_next_ids[prompt] = gnid

                    # Track all IDs
                    all_object_ids.setdefault(prompt, set()).update(obj_ids)

                    # Save masks
                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _save_chunk_masks(
                        result, obj_ids, masks_dir,
                        video_metadata["width"], video_metadata["height"],
                        chunk_frames,
                    )

                    # Extract carry-forward
                    carry[prompt] = _extract_last_frame_masks(result, obj_ids)
                    print(f"    {len(obj_ids)} object(s)")

            # ----- Point prompts -----
            if points:
                prompt_key = "__points__"
                safe = "points"
                vid_info = video_metadata
                print(f"  Points: {points}")

                driver.reset_session(session_id)

                for pi, (pt, lbl) in enumerate(zip(points, point_labels)):
                    driver.add_object_with_points_prompt(
                        session_id,
                        frame_idx=0,
                        object_id=pi,
                        frame_width=vid_info["width"],
                        frame_height=vid_info["height"],
                        points=[pt],
                        point_labels=[lbl],
                    )

                result, obj_ids, frame_objs = driver.propagate_in_video(
                    session_id, propagation_direction="both"
                )

                prev_masks = carry.get(prompt_key, {})
                gnid = global_next_ids.get(prompt_key, 0)
                result, obj_ids, mapping, gnid = _match_and_remap(
                    result, obj_ids, prev_masks, gnid
                )
                global_next_ids[prompt_key] = gnid
                all_object_ids.setdefault(prompt_key, set()).update(obj_ids)

                masks_dir = chunk_dir / "masks" / safe
                masks_dir.mkdir(parents=True, exist_ok=True)
                _save_chunk_masks(
                    result, obj_ids, masks_dir,
                    vid_info["width"], vid_info["height"],
                    chunk_frames,
                )
                carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)
                print(f"    {len(obj_ids)} object(s)")

            # ----- Mask prompts -----
            if mask_paths:
                prompt_key = "__masks__"
                safe = "masks"
                vid_info = video_metadata
                print(f"  Masks: {[m.name for m in mask_paths]}")

                driver.reset_session(session_id)

                # Inject each mask as a separate object
                mask_dict = {}
                obj_id_list = []
                for mi, mp in enumerate(mask_paths):
                    m = np.array(Image.open(mp).convert("L"))
                    mask_dict[mi] = m
                    obj_id_list.append(mi)

                driver.inject_masks(session_id, frame_idx=0, masks=mask_dict, object_ids=obj_id_list)

                result, obj_ids, frame_objs = driver.propagate_in_video(
                    session_id, propagation_direction="both"
                )

                prev_masks_cf = carry.get(prompt_key, {})
                gnid = global_next_ids.get(prompt_key, 0)
                result, obj_ids, mapping, gnid = _match_and_remap(
                    result, obj_ids, prev_masks_cf, gnid
                )
                global_next_ids[prompt_key] = gnid
                all_object_ids.setdefault(prompt_key, set()).update(obj_ids)

                masks_dir = chunk_dir / "masks" / safe
                masks_dir.mkdir(parents=True, exist_ok=True)
                _save_chunk_masks(
                    result, obj_ids, masks_dir,
                    vid_info["width"], vid_info["height"],
                    chunk_frames,
                )
                carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)
                print(f"    {len(obj_ids)} object(s)")

        finally:
            driver.close_session(session_id)

        print()

    # ----- Stitch masks and create overlay -----
    print("Stitching masks...")
    fps = video_metadata.get("fps", 25)
    w = video_metadata["width"]
    h = video_metadata["height"]

    prompt_keys = []
    if prompts:
        prompt_keys.extend(prompts)
    if points:
        prompt_keys.append("__points__")
    if mask_paths:
        prompt_keys.append("__masks__")

    for pk in prompt_keys:
        safe = sanitize_filename(pk) if not pk.startswith("__") else pk.strip("_")
        oids = all_object_ids.get(pk, set())
        out = video_output / "masks" / safe
        _stitch_masks_to_video(
            chunks_dir, safe, oids, chunk_list, overlap, out, fps, w, h
        )

        # Overlay
        mask_vids = sorted(out.glob("object_*_mask.mp4"))
        if mask_vids:
            overlay_path = video_output / f"overlay_{safe}.mp4"
            _create_overlay_video(video_path, mask_vids, overlay_path, alpha)

    # ----- Cleanup -----
    driver.cleanup()

    if keep_temp:
        dest = video_output / "temp_files"
        if temp_base.exists():
            shutil.copytree(temp_base, dest, dirs_exist_ok=True)
            print(f"  Temp files preserved at: {dest}")

    if temp_base.exists():
        shutil.rmtree(temp_base)

    # Save final metadata
    final_meta = {
        "video": str(video_path),
        "video_name": video_name,
        "output_dir": str(video_output),
        "num_chunks": n_chunks,
        "prompts": prompts,
        "points": points,
        "mask_paths": [str(p) for p in mask_paths] if mask_paths else None,
        "device": device,
        "memory": mem_info,
    }
    with open(video_output / "metadata.json", "w") as f:
        json.dump(final_meta, f, indent=2, default=str)

    print()
    print("=" * 70)
    print(f"  ✓ Video processing complete → {video_output}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 Video Prompter — segment videos with text, click points, or masks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_prompter.py --video clip.mp4 --prompts person ball
  python video_prompter.py --video clip.mp4 --points 320,240 --point-labels 1
  python video_prompter.py --video clip.mp4 --masks mask.png
  python video_prompter.py --video clip.mp4 --prompts player --device cpu --keep-temp
  python video_prompter.py --video clip.mp4 --prompts player --frame-range 100 500
  python video_prompter.py --video clip.mp4 --prompts player --time-range 0:05 0:30
  python video_prompter.py --video clip.mp4 --prompts player --time-range 10.0 45.5
        """,
    )

    parser.add_argument("--video", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--prompts", nargs="+", default=None,
        help="Text prompts (e.g. person ball)",
    )
    parser.add_argument(
        "--points", nargs="+", default=None,
        help="Click points as x,y pairs (e.g. 320,240 500,300)",
    )
    parser.add_argument(
        "--point-labels", nargs="+", type=int, default=None,
        help="Labels for each point (1=positive, 0=negative)",
    )
    parser.add_argument(
        "--masks", nargs="+", default=None,
        help="Mask image file(s) for initial object prompts",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Overlay alpha (0.0–1.0, default: 0.5)",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Force device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--chunk-spread", type=str, default="default", choices=["default", "even"],
        help="Chunk size strategy (default or even)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Preserve intermediate chunk files in output",
    )
    parser.add_argument(
        "--frame-range", nargs=2, type=int, metavar=("START", "END"),
        help="Process only frames START..END (0-based, inclusive)",
    )
    parser.add_argument(
        "--time-range", nargs=2, type=str, metavar=("START", "END"),
        help="Process a time segment (seconds, MM:SS, or HH:MM:SS)",
    )

    args = parser.parse_args()

    # Validate: frame-range and time-range are mutually exclusive
    if args.frame_range and args.time_range:
        print("\033[91m✗ --frame-range and --time-range are mutually exclusive.\033[0m")
        sys.exit(1)

    # Validate: at least one prompt type
    if not args.prompts and not args.points and not args.masks:
        print("\033[91m✗ At least one of --prompts, --points, or --masks must be provided.\033[0m")
        sys.exit(1)

    # Validate video exists
    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"\033[91m✗ Video file not found: {args.video}\033[0m")
        sys.exit(1)

    # Parse points
    points = None
    point_labels = None
    if args.points:
        points = []
        for p in args.points:
            parts = p.replace(" ", "").split(",")
            if len(parts) != 2:
                print(f"\033[91m✗ Each point must be x,y — got '{p}'\033[0m")
                sys.exit(1)
            points.append([float(parts[0]), float(parts[1])])
        point_labels = args.point_labels or [1] * len(points)
        if len(point_labels) != len(points):
            print(f"\033[91m✗ --point-labels count must match --points count\033[0m")
            sys.exit(1)

    # Parse masks
    mask_paths = None
    if args.masks:
        mask_paths = [Path(m) for m in args.masks]
        for mp in mask_paths:
            if not mp.is_file():
                print(f"\033[91m✗ Mask file not found: {mp}\033[0m")
                sys.exit(1)

    # Device
    from sam3.__globals import DEVICE as DEFAULT_DEVICE
    device = args.device or DEFAULT_DEVICE.type

    # Header
    print()
    print("=" * 70)
    print("  SAM3 Video Prompter")
    print("=" * 70)
    print(f"  Video   : {video_path}")
    if args.prompts:
        print(f"  Prompts : {', '.join(args.prompts)}")
    if points:
        print(f"  Points  : {points}")
    if mask_paths:
        print(f"  Masks   : {[m.name for m in mask_paths]}")
    if args.frame_range:
        print(f"  Frames  : {args.frame_range[0]}–{args.frame_range[1]}")
    if args.time_range:
        print(f"  Time    : {args.time_range[0]} → {args.time_range[1]}")
    print(f"  Device  : {device}")
    print(f"  Output  : {args.output}")
    print(f"  Alpha   : {args.alpha}")
    print(f"  Chunking: {args.chunk_spread}")
    print("=" * 70)
    print()

    _process_video(
        video_path=video_path,
        prompts=args.prompts,
        points=points,
        point_labels=point_labels,
        mask_paths=mask_paths,
        output_dir=Path(args.output),
        device=device,
        alpha=args.alpha,
        chunk_spread=args.chunk_spread,
        keep_temp=args.keep_temp,
        frame_range=tuple(args.frame_range) if args.frame_range else None,
        time_range=tuple(args.time_range) if args.time_range else None,
    )


if __name__ == "__main__":
    main()
