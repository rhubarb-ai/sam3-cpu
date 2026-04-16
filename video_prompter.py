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
import os as _os
import sys as _sys

if "--device" in _sys.argv:
    _i = _sys.argv.index("--device")
    if _i + 1 < len(_sys.argv) and _sys.argv[_i + 1].lower() == "cpu":
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""
# -----------------------------------------------------------------

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
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


def _table(rows: list[list[str]]):
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
    frame_range: tuple[int, int] | None,
    time_range: tuple[str, str] | None,
) -> tuple[int, int] | None:
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


def _extract_subclip(video_path: Path, start_frame: int, end_frame: int, temp_dir: Path) -> Path:
    """Extract a sub-clip from *video_path* covering [start_frame, end_frame]."""
    from sam3.utils.ffmpeglib import ffmpeg_lib

    temp_dir.mkdir(parents=True, exist_ok=True)
    out = temp_dir / f"subclip_{start_frame}_{end_frame}.mp4"
    ffmpeg_lib.create_video_chunk(str(video_path), str(out), start_frame, end_frame)
    return out


# ---------------------------------------------------------------------------
# Memory validation
# ---------------------------------------------------------------------------


def _validate_video_memory(
    video_path: Path,
    device: str,
    max_memory_bytes: int = None,
    offload_state_to_cpu: bool = False,
) -> dict[str, Any]:
    """Check if there is enough memory to process at least MIN_VIDEO_FRAMES.

    Returns a dict with memory stats, chunk plan, and ``can_process`` flag.

    Parameters
    ----------
    max_memory_bytes : int, optional
        Simulate a smaller device (for testing).
    offload_state_to_cpu : bool
        Forward to the chunk planner so it sizes by RAM when offloading.
    """
    from sam3.__globals import (
        DEFAULT_MIN_VIDEO_FRAMES,
        VIDEO_INFERENCE_MB,
    )
    from sam3.memory_manager import MemoryManager
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.utils.helpers import ram_stat, vram_stat

    video_info = ffmpeg_lib.get_video_info(str(video_path))
    if video_info is None:
        return {"can_process": False, "error": "Could not read video metadata"}

    mm = MemoryManager()
    max_frames = mm.compute_memory_safe_frames(
        video_info["width"],
        video_info["height"],
        device,
        type="video",
        max_memory_bytes=max_memory_bytes,
        offload_state_to_cpu=offload_state_to_cpu,
    )

    if device == "cuda" and offload_state_to_cpu:
        # With offloading the bounding resource is RAM
        mem = ram_stat()
        available = mem["available"]
    elif device == "cuda":
        mem = vram_stat()
        available = mem["free"]
    else:
        mem = ram_stat()
        available = mem["available"]

    # Apply simulated cap to the reported values
    if max_memory_bytes is not None and max_memory_bytes > 0:
        real_used = mem["total"] - available
        sim_total = max_memory_bytes
        sim_available = max(sim_total - real_used, 0)
        mem = dict(mem)
        mem["total"] = sim_total
        available = sim_available

    info: dict[str, Any] = {
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
        needed = VIDEO_INFERENCE_MB * 1024**2 + DEFAULT_MIN_VIDEO_FRAMES * frame_bytes
        info["deficit_bytes"] = max(needed - available, 0)
    else:
        info["deficit_bytes"] = 0

    return info


def _show_video_memory_table(info: dict[str, Any]):
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
    video_path: Path,
    device: str,
    chunk_spread: str = "default",
    max_memory_bytes: int = None,
    offload_state_to_cpu: bool = False,
) -> tuple:
    """Create a memory-safe chunk plan.

    Returns (video_metadata, chunk_list).
    """
    from sam3.memory_manager import memory_manager

    metadata, chunks = memory_manager.chunk_plan_video(
        str(video_path),
        device=device,
        chunk_spread=chunk_spread,
        max_memory_bytes=max_memory_bytes,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    return metadata, chunks


# ---------------------------------------------------------------------------
# Mask extraction helpers
# ---------------------------------------------------------------------------


def _extract_last_frame_masks(
    result_prompt: dict,
    object_ids: set,
) -> dict[int, np.ndarray]:
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
            masks[oid] = m.astype(np.uint8) * 255
    return masks


def _trim_carry_if_needed(
    carry: dict[str, dict[int, np.ndarray]],
    max_ram_pct: float,
    min_free_gb: float,
) -> int:
    """Drop oldest carry entries when RAM pressure is too high.

    Checks two conditions (whichever hits first triggers trimming):
    1. RAM usage >= *max_ram_pct* (e.g. 0.98 = 98%)
    2. Free RAM < *min_free_gb* (e.g. 1.0 GB)

    Drops entries in insertion order (oldest prompt first) until memory is
    within both limits or carry is empty.

    Returns the number of prompt entries dropped.
    """
    import psutil

    dropped = 0
    while carry:
        mem = psutil.virtual_memory()
        pct_used = mem.percent / 100.0
        free_gb = mem.available / (1024**3)

        if pct_used < max_ram_pct and free_gb > min_free_gb:
            break  # within limits

        # Drop the oldest prompt entry (first key in insertion order)
        oldest_key = next(iter(carry))
        n_masks = len(carry[oldest_key])
        del carry[oldest_key]
        dropped += 1
        print(
            f"\033[93m⚠ RAM guard: dropped carry for '{oldest_key}' "
            f"({n_masks} mask(s)) — RAM {pct_used:.0%} used, "
            f"{free_gb:.1f} GB free\033[0m"
        )

    return dropped


def _trim_memory_bank_if_needed(
    memory_banks: dict[str, list],
    max_ram_pct: float,
    min_free_gb: float,
) -> int:
    """Drop oldest memory-bank frames when RAM pressure is too high.

    Similar to ``_trim_carry_if_needed`` but operates on the per-prompt
    memory bank dicts.  Each iteration drops the oldest entry from the
    longest bank (past-forgetting: recent frames are preserved, oldest
    frames are sacrificed first).

    Returns the total number of individual frame entries dropped.
    """
    import psutil

    dropped = 0
    while memory_banks:
        mem = psutil.virtual_memory()
        pct_used = mem.percent / 100.0
        free_gb = mem.available / (1024**3)

        if pct_used < max_ram_pct and free_gb >= min_free_gb:
            break  # within limits

        # Find the prompt with the longest bank — trim its oldest entry
        longest_key = max(memory_banks, key=lambda k: len(memory_banks[k]))
        bank = memory_banks[longest_key]
        if not bank:
            del memory_banks[longest_key]
            continue

        bank.pop()  # drop the oldest (last) entry — bank is newest-first
        dropped += 1
        if not bank:
            del memory_banks[longest_key]

        print(
            f"\033[93m⚠ RAM guard: dropped memory bank frame for '{longest_key}' "
            f"— RAM {pct_used:.0%} used, {free_gb:.1f} GB free\033[0m"
        )

    return dropped


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
    prev_masks: dict[int, np.ndarray],
    global_next_id: int,
    iou_threshold: float = 0.25,
):
    """Greedy IoU matching and ID remapping.

    Returns (remapped_result, remapped_ids, id_mapping, updated_next_id, iou_matrix).

    ``iou_matrix`` is a nested dict ``{new_id: {prev_id: iou}}`` with ALL
    pairwise comparisons (not just those above the threshold).  It is ``{}``
    for the first chunk where no previous masks exist.
    """
    iou_matrix: dict[int, dict[int, float]] = {}

    if not result_prompt:
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
        return {}, set(mapping.values()), mapping, global_next_id, iou_matrix

    first_idx = min(result_prompt.keys())
    first_out = result_prompt.get(first_idx)
    if first_out is None:
        mapping = {o: o for o in object_ids}
        return result_prompt, object_ids, mapping, global_next_id, iou_matrix

    out_ids = first_out.get("out_obj_ids", [])
    if isinstance(out_ids, np.ndarray):
        out_ids = out_ids.tolist()

    first_masks = {}
    for oid in out_ids:
        idx = out_ids.index(oid)
        first_masks[oid] = first_out["out_binary_masks"][idx].astype(np.uint8) * 255

    if not prev_masks:
        # First chunk — identity mapping
        mapping = {}
        for oid in sorted(object_ids):
            mapping[oid] = global_next_id
            global_next_id += 1
    else:
        pairs = []
        for nid, nm in first_masks.items():
            iou_matrix[nid] = {}
            for pid, pm in prev_masks.items():
                iou = _compute_iou(nm, pm)
                iou_matrix[nid][pid] = round(iou, 6)
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
        new_out["out_obj_ids"] = np.array([mapping.get(o, o) for o in ids_list], dtype=np.int64)
        remapped[fidx] = new_out

    return remapped, set(mapping.values()), mapping, global_next_id, iou_matrix


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
    fps: float = 25.0,
):
    """Save per-object mask videos (lossless MP4) — replaces old PNG pipeline.

    Each object gets a single ``object_{id}_mask.mp4`` in *masks_dir*,
    containing grayscale frames (0=bg, 255=object).  This is ~100× fewer
    I/O operations than the per-frame PNG approach and produces files that
    are directly usable by the cross-chunk stitcher.
    """
    from sam3.streaming_masks import EmptyMaskPool, MaskVideoWriter

    pool = EmptyMaskPool(width, height)
    writers: dict[int, MaskVideoWriter] = {}

    for oid in sorted(object_ids):
        out_path = masks_dir / f"object_{oid}_mask.mp4"
        writers[oid] = MaskVideoWriter(out_path, fps, width, height)

    for fidx in range(total_frames):
        output = result_prompt.get(fidx)
        for oid in sorted(object_ids):
            writer = writers[oid]
            if output is not None:
                out_ids = output.get("out_obj_ids", [])
                if isinstance(out_ids, np.ndarray):
                    out_ids = out_ids.tolist()
                if oid in out_ids:
                    idx = out_ids.index(oid)
                    m = output["out_binary_masks"][idx]
                    if m.any():
                        mask_u8 = m.astype(np.uint8) * 255
                        writer.write_frame(mask_u8)
                        continue
            writer.write_black(1, pool=pool)

    for w in writers.values():
        w.close()


# ---------------------------------------------------------------------------
# Intra-chunk monitoring helpers
# ---------------------------------------------------------------------------


def _ensure_cpu_masks(result: dict) -> None:
    """Convert any GPU tensors in result dict to CPU numpy arrays in-place.

    This must be called before submitting results to the ``AsyncIOWorker``
    to ensure the background thread can access the data safely after CUDA
    memory is released.
    """
    for frame_data in result.values():
        if frame_data is None:
            continue
        masks = frame_data.get("out_binary_masks")
        if masks is not None and hasattr(masks, "cpu"):
            frame_data["out_binary_masks"] = masks.cpu().numpy()
        obj_ids = frame_data.get("out_obj_ids")
        if obj_ids is not None and hasattr(obj_ids, "cpu"):
            frame_data["out_obj_ids"] = obj_ids.cpu().numpy()


def _propagate_with_monitoring(
    driver,
    session_id: str,
    monitor,
    propagation_direction: str = "both",
    max_output_frames: int | None = None,
) -> tuple[dict, set, dict, bool]:
    """Stream-process video propagation with per-frame memory monitoring.

    Iterates through ``driver.propagate_in_video_streaming()`` frame by
    frame, calling ``monitor.check()`` at each step.  If the monitor
    signals a stop (memory pressure), the generator is broken and partial
    results are returned.

    Parameters
    ----------
    driver : Sam3VideoDriver
        The loaded video driver.
    session_id : str
        Active session from ``driver.start_session()``.
    monitor : IntraChunkMonitor
        Pre-initialised monitor (call ``monitor.start()`` before this).
    propagation_direction : str
        ``"both"``, ``"forward"``, or ``"backward"``.
    max_output_frames : int, optional
        When set, stop after collecting this many unique output frames.
        Used to cap subsequent prompts to the same frame range as an
        earlier prompt that triggered a protective stop.  This is NOT
        flagged as ``early_stopped`` — it is a deliberate cap, not a
        memory-pressure event.

    Returns
    -------
    result : dict
        ``{frame_idx: outputs}`` for processed frames.
    object_ids : set
        All unique object IDs found.
    frame_objects : dict
        ``{frame_idx: [obj_ids]}`` per frame.
    early_stopped : bool
        ``True`` if monitor triggered an early stop.
    """
    from sam3.memory_optimizer import clear_memory

    result = {}
    object_ids = set()
    frame_objects = {}
    early_stopped = False

    try:
        for frame_idx, outputs, frame_obj_ids in driver.propagate_in_video_streaming(
            session_id, propagation_direction=propagation_direction
        ):
            # ── Frame cap (from a prior prompt's early stop) ──
            if max_output_frames is not None and len(result) >= max_output_frames:
                break  # deliberate cap — not a memory stop
            if not monitor.check(frame_idx):
                early_stopped = True
                break
            result[frame_idx] = outputs
            frame_objects[frame_idx] = frame_obj_ids
            object_ids.update(frame_obj_ids)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            early_stopped = True
            monitor._stop_reason = "oom_exception"
            clear_memory(monitor.device, full_gc=True)
        else:
            raise

    return result, object_ids, frame_objects, early_stopped


# ---------------------------------------------------------------------------
# Parallel processing helpers
# ---------------------------------------------------------------------------


def _get_mp_context():
    """Return a multiprocessing context safe for any thread state.

    On Linux, ``forkserver`` avoids the deadlock risk of ``fork()`` in a
    multi-threaded parent process (e.g. when PyTorch data-loaders or
    async I/O workers have been used).  On platforms where ``forkserver``
    is unavailable, falls back to ``spawn``.
    """
    import multiprocessing as mp

    available = mp.get_all_start_methods()
    if "forkserver" in available:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Stitching and overlay (LEGACY — PNG-based fallback)
# These functions are deprecated and no longer used by the main pipeline.
# The production pipeline uses sam3.streaming_masks for MP4-based streaming.
# Retained for backward compatibility and testing.
# ---------------------------------------------------------------------------


def _read_mask_png(path: str) -> np.ndarray | None:
    """Read a grayscale PNG mask, robust to zlib conflicts.

    Tries ``cv2.imread`` first (faster), falls back to ``PIL`` if OpenCV
    encounters the zlib-parameter issue that occurs when PIL and OpenCV
    share the same process with different zlib versions.
    """
    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if frame is not None:
        return frame
    # Fallback: PIL is immune to the zlib parameter bug
    try:
        return np.array(Image.open(path))
    except Exception:
        return None


def _stitch_single_object(
    oid: int,
    chunks_dir_str: str,
    prompt_name: str,
    chunk_infos: list,
    overlap: int,
    output_dir_str: str,
    fps: float,
    width: int,
    height: int,
) -> str:
    """Stitch PNG masks for a single object across all chunks into one MP4.

    This is a module-level function so it can be pickled by
    ``ProcessPoolExecutor``.  All path arguments are strings.

    Returns the filename of the created mask video.
    """
    import cv2 as _cv2
    import numpy as _np
    from PIL import Image as _PILImage

    def _read_png(p: str) -> _np.ndarray | None:
        frame = _cv2.imread(p, _cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            return frame
        try:
            return _np.array(_PILImage.open(p))
        except Exception:
            return None

    chunks_dir = Path(chunks_dir_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"object_{oid}_mask.mp4"
    fourcc = _cv2.VideoWriter_fourcc(*"mp4v")
    writer = _cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)
    black = _np.zeros((height, width), dtype=_np.uint8)

    for ci, cinfo in enumerate(chunk_infos):
        chunk_id = cinfo["chunk"]
        skip = overlap if ci > 0 else 0
        obj_mask_dir = chunks_dir / f"chunk_{chunk_id}" / "masks" / prompt_name / f"object_{oid}"
        if not obj_mask_dir.exists():
            chunk_len = cinfo["end"] - cinfo["start"] + 1
            for _ in range(chunk_len - skip):
                writer.write(black)
            continue
        pngs = sorted(obj_mask_dir.glob("frame_*.png"))
        chunk_len = cinfo["end"] - cinfo["start"] + 1
        usable = chunk_len - skip
        for png in pngs[skip : skip + max(0, usable)]:
            frame = _read_png(str(png))
            if frame is not None:
                writer.write(frame)

    writer.release()
    return out_path.name


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
    max_workers: int | None = None,
):
    """Stitch per-chunk PNG masks into per-object mask videos (parallel).

    Each object is stitched in a separate process using
    ``ProcessPoolExecutor`` to utilise all CPU cores.  Falls back to
    sequential processing when only one object exists or in test mode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sorted_ids = sorted(object_ids)

    if not sorted_ids:
        return

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(sorted_ids), 8)

    # Use sequential path for single object (avoids process-spawn overhead)
    if len(sorted_ids) == 1 or max_workers <= 1:
        for oid in sorted_ids:
            name = _stitch_single_object(
                oid,
                str(chunks_dir),
                prompt_name,
                chunk_infos,
                overlap,
                str(output_dir),
                fps,
                width,
                height,
            )
            print(f"    Saved mask video: {name}")
        return

    # Parallel: one process per object
    mp_ctx = _get_mp_context()
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(
                _stitch_single_object,
                oid,
                str(chunks_dir),
                prompt_name,
                chunk_infos,
                overlap,
                str(output_dir),
                fps,
                width,
                height,
            ): oid
            for oid in sorted_ids
        }
        for future in as_completed(futures):
            oid = futures[future]
            try:
                name = future.result()
                print(f"    Saved mask video: {name}")
            except Exception as exc:
                print(f"    \033[91mError stitching object {oid}: {exc}\033[0m")


def _create_overlay_video(
    video_path: Path,
    mask_videos: list[Path],
    output_path: Path,
    alpha: float = 0.5,
):
    """Overlay coloured masks onto the original video.

    Uses vectorised numpy compositing (all 3 channels at once) instead
    of per-channel loops for ~3× speedup on large frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h), True)

    # Open mask video readers
    mask_caps = [cv2.VideoCapture(str(p)) for p in mask_videos if p.exists()]

    # Colour palette (BGR for OpenCV)
    colours = [
        (30, 144, 255),
        (255, 50, 50),
        (50, 205, 50),
        (255, 165, 0),
        (148, 103, 189),
        (255, 215, 0),
    ]
    # Pre-compute colour arrays as (1, 1, 3) float32 for vectorised blending
    colour_arrays = [np.array(c, dtype=np.float32).reshape(1, 1, 3) for c in colours]
    inv_alpha = np.float32(1.0 - alpha)
    f_alpha = np.float32(alpha)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        frame_f32 = frame.astype(np.float32)
        for i, mc in enumerate(mask_caps):
            ret_m, mask_frame = mc.read()
            if ret_m and mask_frame is not None:
                if mask_frame.ndim == 3:
                    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                binary = mask_frame > 127
                if binary.any():
                    c_arr = colour_arrays[i % len(colour_arrays)]
                    # Vectorised: blend all 3 channels at once
                    blended = (inv_alpha * frame_f32[binary] + f_alpha * c_arr).astype(np.uint8)
                    overlay[binary] = blended
        writer.write(overlay)

    for mc in mask_caps:
        mc.release()
    cap.release()
    writer.release()
    print(f"    Saved overlay video: {output_path.name}")


# ---------------------------------------------------------------------------
# Per-object tracking metadata
# ---------------------------------------------------------------------------


def _analyze_single_object(
    oid: int,
    mask_dir_str: str,
    fps: float,
    frame_offset: int = 0,
    min_active_pixels: int = 50,
    gap_tolerance: int = 2,
) -> dict[str, Any]:
    """Analyse a single object's mask video for temporal presence.

    Module-level function so it can be pickled by ``ProcessPoolExecutor``.
    """
    import cv2 as _cv2

    def _ts(sec: float) -> str:
        m, s = divmod(sec, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{int(m):02d}:{s:06.3f}"

    mask_dir = Path(mask_dir_str)
    mp4 = mask_dir / f"object_{oid}_mask.mp4"

    if not mp4.exists():
        return {
            "object_id": oid,
            "intervals": [],
            "num_intervals": 0,
            "total_frames_active": 0,
            "total_frames": 0,
            "total_duration_s": 0.0,
            "first_frame": None,
            "last_frame": None,
            "first_timestamp": None,
            "last_timestamp": None,
            "first_timecode": None,
            "last_timecode": None,
            "mean_mask_area_pct": None,
            "max_mask_area_pct": None,
        }

    cap = _cv2.VideoCapture(str(mp4))
    total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = max(w * h, 1)

    active_frames: list = []
    area_fractions: list = []
    for fidx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        active_px = int((frame > 127).sum())
        if active_px >= min_active_pixels:
            active_frames.append(fidx)
            area_fractions.append(active_px / total_pixels)
    cap.release()

    # Group into intervals with gap tolerance
    raw_intervals: list = []
    if active_frames:
        iv_start = active_frames[0]
        iv_end = active_frames[0]
        for fidx in active_frames[1:]:
            if fidx <= iv_end + 1 + gap_tolerance:
                iv_end = fidx
            else:
                raw_intervals.append((iv_start, iv_end))
                iv_start = fidx
                iv_end = fidx
        raw_intervals.append((iv_start, iv_end))

    intervals: list = []
    for s, e in raw_intervals:
        abs_s = s + frame_offset
        abs_e = e + frame_offset
        intervals.append(
            {
                "start_frame": abs_s,
                "end_frame": abs_e,
                "start_time": round(abs_s / fps, 3),
                "end_time": round(abs_e / fps, 3),
                "start_timecode": _ts(abs_s / fps),
                "end_timecode": _ts(abs_e / fps),
                "duration_frames": abs_e - abs_s + 1,
                "duration_s": round((abs_e - abs_s + 1) / fps, 3),
            }
        )

    active_count = len(active_frames)
    abs_first = (active_frames[0] + frame_offset) if active_frames else None
    abs_last = (active_frames[-1] + frame_offset) if active_frames else None
    total_dur = round(active_count / fps, 3) if active_count else 0.0

    mean_area = round(sum(area_fractions) / len(area_fractions) * 100, 4) if area_fractions else None
    max_area = round(max(area_fractions) * 100, 4) if area_fractions else None

    return {
        "object_id": oid,
        "intervals": intervals,
        "num_intervals": len(intervals),
        "total_frames_active": active_count,
        "total_frames": total,
        "total_duration_s": total_dur,
        "first_frame": abs_first,
        "last_frame": abs_last,
        "first_timestamp": round(abs_first / fps, 3) if abs_first is not None else None,
        "last_timestamp": round(abs_last / fps, 3) if abs_last is not None else None,
        "first_timecode": _ts(abs_first / fps) if abs_first is not None else None,
        "last_timecode": _ts(abs_last / fps) if abs_last is not None else None,
        "mean_mask_area_pct": mean_area,
        "max_mask_area_pct": max_area,
    }


def _build_object_tracking(
    mask_dir: Path,
    object_ids: set,
    fps: float,
    frame_offset: int = 0,
    min_active_pixels: int = 50,
    gap_tolerance: int = 2,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Scan stitched mask videos and compute per-object presence info.

    For each object, determines **all intervals** where the mask is active
    (has at least *min_active_pixels* bright pixels), handling objects that
    appear, disappear, and reappear multiple times.

    Short gaps of up to *gap_tolerance* inactive frames between two active
    regions are bridged so that tiny dips (e.g. from mp4 compression or a
    single missed frame) do not fragment intervals.

    When multiple objects are present, analysis is parallelised across
    CPU cores using ``ProcessPoolExecutor``.

    Args:
        mask_dir: Directory containing ``object_{id}_mask.mp4`` files.
        object_ids: Set of global object IDs to scan.
        fps: Video FPS (used for timestamp conversion).
        frame_offset: If a sub-clip was extracted, offset added to frame
            numbers so they refer to the *original* video's timeline.
        min_active_pixels: Minimum number of pixels > 127 for a frame to
            be considered "active".  Default 50 (≈ noise-only filter).
        gap_tolerance: Maximum gap (in frames) between active frames that
            will be bridged into a single interval.  Default 2.
        max_workers: Maximum number of parallel processes.  Defaults to
            ``min(cpu_count, len(object_ids), 8)``.

    Returns:
        List of dicts, one per object, sorted by object ID.
    """
    sorted_ids = sorted(object_ids)
    if not sorted_ids:
        return []

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(sorted_ids), 8)

    mask_dir_str = str(mask_dir)

    # Single object — avoid process spawn overhead
    if len(sorted_ids) == 1 or max_workers <= 1:
        return [
            _analyze_single_object(
                oid,
                mask_dir_str,
                fps,
                frame_offset,
                min_active_pixels,
                gap_tolerance,
            )
            for oid in sorted_ids
        ]

    # Parallel analysis
    mp_ctx = _get_mp_context()
    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(
                _analyze_single_object,
                oid,
                mask_dir_str,
                fps,
                frame_offset,
                min_active_pixels,
                gap_tolerance,
            ): oid
            for oid in sorted_ids
        }
        for future in as_completed(futures):
            oid = futures[future]
            try:
                results[oid] = future.result()
            except Exception as exc:
                print(f"    \033[91mError analysing object {oid}: {exc}\033[0m")
                results[oid] = {
                    "object_id": oid,
                    "intervals": [],
                    "num_intervals": 0,
                    "total_frames_active": 0,
                    "total_frames": 0,
                    "total_duration_s": 0.0,
                    "first_frame": None,
                    "last_frame": None,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "first_timecode": None,
                    "last_timecode": None,
                    "mean_mask_area_pct": None,
                    "max_mask_area_pct": None,
                }

    # Return sorted by object ID
    return [results[oid] for oid in sorted_ids]


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def _process_video(
    video_path: Path,
    prompts: list[str] | None,
    points: list[list[float]] | None,
    point_labels: list[int] | None,
    mask_paths: list[Path] | None,
    output_dir: Path,
    device: str,
    alpha: float,
    chunk_spread: str,
    keep_temp: bool,
    frame_range: tuple[int, int] | None = None,
    time_range: tuple[str, str] | None = None,
    max_vram_gb: float | None = None,
    max_ram_gb: float | None = None,
    cpu_utilisation: int = 100,
    offload_state_to_cpu: bool | None = None,
):
    """Full video processing pipeline with adaptive dynamic chunking."""
    from datetime import datetime

    import torch

    from sam3.__globals import DEFAULT_MIN_CHUNK_OVERLAP, TEMP_DIR
    from sam3.async_io import AsyncIOWorker
    from sam3.memory_optimizer import AdaptiveChunkManager, IntraChunkMonitor, clear_memory
    from sam3.memory_predictor import MemoryPredictor
    from sam3.utils.ffmpeglib import ffmpeg_lib
    from sam3.utils.helpers import sanitize_filename
    from sam3.utils.memory_sampler import MemorySampler

    # Convert simulated limits to bytes
    max_vram_bytes = int(max_vram_gb * 1024**3) if max_vram_gb else None
    max_ram_bytes = int(max_ram_gb * 1024**3) if max_ram_gb else None

    # ── Clear stale CUDA cache from any previous runs ──
    clear_memory(device, full_gc=True)

    # ── Timing & memory instrumentation ──
    pipeline_start = time.time()
    pipeline_start_iso = datetime.now().isoformat()
    mem_sampler = MemorySampler(interval=1.0, device=device)
    mem_sampler.start()

    # ── OOM predictor (async, non-blocking) ──
    _oom_stop_requested = False

    def _on_soft():
        nonlocal _oom_stop_requested
        print("\033[93m⚠ Memory warning: headroom < 15% — consider reducing chunk size\033[0m")

    def _on_hard():
        nonlocal _oom_stop_requested
        _oom_stop_requested = True
        print("\033[91m✗ Memory critical: headroom < 5% — requesting early stop\033[0m")

    mem_predictor = MemoryPredictor(
        device=device,
        safety_factor=0.85,
        on_soft_stop=_on_soft,
        on_hard_stop=_on_hard,
    )
    mem_predictor.start()
    frames_processed = 0  # running counter for record_frame

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
    # Use simulated VRAM limit if testing with a smaller GPU
    _mem_cap = max_vram_bytes if device == "cuda" else max_ram_bytes

    # Resolve offload flag early — it influences chunk sizing.
    # On CUDA with OFFLOAD_TRACKER_STATE_TO_CPU == True, per-frame state
    # goes to RAM so chunks are sized by RAM, not VRAM.
    # CLI override (True/False) takes precedence; None = auto from globals.
    if offload_state_to_cpu is not None:
        _offload_state = offload_state_to_cpu
    elif device == "cuda":
        try:
            from sam3.__globals import OFFLOAD_TRACKER_STATE_TO_CPU

            _offload_state = OFFLOAD_TRACKER_STATE_TO_CPU
        except ImportError:
            _offload_state = True  # safe default — match drivers.py
    else:
        _offload_state = False

    mem_info = _validate_video_memory(
        video_path,
        device,
        max_memory_bytes=_mem_cap,
        offload_state_to_cpu=_offload_state,
    )
    if _mem_cap:
        mem_info["simulated_limit_gb"] = round(_mem_cap / (1024**3), 1)
    _show_video_memory_table(mem_info)
    print()

    if not mem_info["can_process"]:
        deficit = mem_info.get("deficit_bytes", 0)
        print(f"\033[91m✗ Cannot process video — need {_fmt(deficit)} more memory.\033[0m")
        sys.exit(1)

    # ── Cross-chunk mask injection flag ──
    try:
        from sam3.__globals import CROSS_CHUNK_MASK_INJECTION

        _cross_chunk_inject = CROSS_CHUNK_MASK_INJECTION
    except ImportError:
        _cross_chunk_inject = True

    # ── Memory bank max frames ──
    try:
        from sam3.__globals import MEMORY_BANK_MAX_FRAMES

        _memory_bank_max_frames = MEMORY_BANK_MAX_FRAMES
    except ImportError:
        _memory_bank_max_frames = 6

    # ----- Chunk plan -----
    print("Creating chunk plan...")
    video_metadata, chunk_list = _make_chunk_plan(
        video_path,
        device,
        chunk_spread,
        max_memory_bytes=_mem_cap,
        offload_state_to_cpu=_offload_state,
    )
    initial_chunk_size = mem_info.get("max_frames_per_chunk", 200)
    n_chunks = len(chunk_list)
    print(f"  {n_chunks} chunk(s), {initial_chunk_size} frames/chunk")
    if n_chunks > 1 and _cross_chunk_inject:
        print(
            f"  Cross-chunk mask injection: ON (carry masks → conditioning frames, "
            f"memory bank → {_memory_bank_max_frames} spatial frames)"
        )

    # ----- Adaptive chunk manager with auto-detected memory tier -----
    from sam3.memory_optimizer import get_memory_tier

    _detected_vram = max_vram_bytes or (mem_info.get("vram_total_bytes", 0))
    _detected_ram = max_ram_bytes or (mem_info.get("ram_total_bytes", 0))
    memory_tier = get_memory_tier(_detected_vram, _detected_ram)
    print(
        f"  Memory tier: {memory_tier['tier']} "
        f"(grow={memory_tier['grow_factor']:.2f}×, "
        f"max_growth={memory_tier['max_growth_factor']:.1f}×, "
        f"min_chunk={memory_tier['min_chunk_frames']})"
    )

    adaptive = AdaptiveChunkManager(
        initial_chunk_size=initial_chunk_size,
        device=device,
        vram_limit_bytes=max_vram_bytes,
        ram_limit_bytes=max_ram_bytes,
        tier=memory_tier,
        offload_state_to_cpu=_offload_state,
    )
    # Initialise adaptive per-frame multiplier (learns from actual execution)
    _vid_w = video_metadata.get("width", 1920)
    _vid_h = video_metadata.get("height", 1080)
    adaptive.init_adaptive_multiplier(_vid_w, _vid_h)

    if _mem_cap:
        print(f"  Simulated memory limit: {_mem_cap / (1024**3):.1f} GB")
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
    t_model_start = time.time()
    from sam3.drivers import Sam3VideoDriver

    driver = Sam3VideoDriver(device=device, cpu_utilisation=cpu_utilisation)
    model_load_s = round(time.time() - t_model_start, 3)
    print(f"Model loaded in {model_load_s:.1f}s.\n")
    mem_predictor.record_frame(0)  # baseline after model load

    # ----- Async I/O worker for overlapping writes -----
    async_worker = AsyncIOWorker()
    async_worker.start()

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
    carry: dict[str, dict[int, np.ndarray]] = {}
    global_next_ids: dict[str, int] = {}
    all_object_ids: dict[str, set] = {}

    # ── Memory bank: per-prompt tracker spatial memory across chunks ──
    # Maps prompt key → list of frame state dicts (newest-first).
    memory_banks: dict[str, list] = {}

    # ── RAM guard thresholds for carry data ──
    try:
        from sam3.__globals import CARRY_MAX_RAM_USAGE_PCT, CARRY_MIN_FREE_RAM_GB

        _carry_max_ram_pct = CARRY_MAX_RAM_USAGE_PCT
        _carry_min_free_gb = CARRY_MIN_FREE_RAM_GB
    except ImportError:
        _carry_max_ram_pct = 0.98
        _carry_min_free_gb = 1.0

    # ── Collectors for enriched metadata ──
    chunk_timing: list[dict[str, Any]] = []
    chunk_meta_list: list[dict[str, Any]] = []
    cross_chunk_iou: dict[str, dict[str, Any]] = {}

    # ── Prompt reordering state: heaviest first to minimise overcalculation ──
    _prompt_memory_profile: dict[str, float] = {}  # prompt → peak_vram_mb
    # Create mutable copy of prompts list for reordering across chunks
    if prompts:
        prompts = list(prompts)

    total_frames_in_video = video_metadata.get("nb_frames", 0)

    t_chunks_start = time.time()

    # ── Dynamic chunk loop with OOM recovery ──
    # Instead of iterating a fixed list, we maintain a cursor and can
    # replan remaining chunks after each chunk completes (or on OOM).
    ci = 0  # chunk counter
    chunk_cursor = 0  # index into chunk_list

    while chunk_cursor < len(chunk_list):
        # ── OOM predictor: check if we should bail ──
        if _oom_stop_requested:
            print(f"\033[91m✗ Stopping after chunk {ci} — OOM predictor triggered hard stop\033[0m")
            break

        cinfo = chunk_list[chunk_cursor]
        chunk_id = ci
        start_frame = cinfo["start"]
        end_frame = cinfo["end"]
        current_chunk_frames = end_frame - start_frame + 1
        t_chunk_start = time.time()

        print(f"── Chunk {ci + 1}/{len(chunk_list)} (frames {start_frame}–{end_frame}, size={current_chunk_frames}) ──")

        chunk_dir = chunks_dir / f"chunk_{chunk_id}"
        chunk_dir.mkdir(exist_ok=True)

        # Extract chunk video
        if len(chunk_list) == 1 and chunk_cursor == 0:
            chunk_video = video_path  # use original
        else:
            chunk_video = chunk_dir / f"chunk_{chunk_id}.mp4"
            ffmpeg_lib.create_video_chunk(str(video_path), str(chunk_video), start_frame, end_frame)

        # Get chunk frame count
        cap = cv2.VideoCapture(str(chunk_video))
        chunk_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Reset peak memory stats for this chunk
        if device.startswith("cuda"):
            import torch as _t

            if _t.cuda.is_available():
                _t.cuda.reset_peak_memory_stats()

        # Start session
        session_id = driver.start_session(video_path=str(chunk_video))

        chunk_objects: dict[str, Any] = {}  # per-prompt objects for this chunk
        chunk_n_objects = 0  # total objects across all prompts
        _chunk_oom = False  # flag for OOM during this chunk
        _chunk_early_stop = False  # flag for proactive early stop
        _early_stop_monitor = None  # monitor that triggered early stop
        _chunk_monitor_results: list[dict] = []  # monitor metadata per prompt
        _partial_frames_processed = 0  # frames saved on early stop / OOM
        _chunk_frame_cap: int | None = None  # frame cap from first early-stop

        # ── Prompt reordering: heaviest first (after chunk 0) ──
        if _prompt_memory_profile and prompts and len(prompts) > 1:
            old_order = list(prompts)
            prompts.sort(key=lambda p: _prompt_memory_profile.get(p, 0), reverse=True)
            if prompts != old_order:
                print(f"  Prompt order (heaviest first): {prompts}")

        try:
            # ----- Text prompts -----
            if prompts:
                for pi, prompt in enumerate(prompts):
                    safe = sanitize_filename(prompt)
                    print(f"  Prompt: '{prompt}'")

                    driver.reset_session(session_id)
                    
                    # ── Cross-chunk mask injection ──
                    # Inject carry masks from previous chunk as conditioning frames
                    # BEFORE adding the text prompt.  This gives the tracker memory
                    # of where objects were, improving boundary continuity.
                    _injected_carry = False
                    if _cross_chunk_inject and ci > 0 and prompt in carry and carry[prompt]:
                        _carry_masks = carry[prompt]
                        _carry_ids = sorted(_carry_masks.keys())
                        try:
                            driver.inject_masks(
                                session_id,
                                frame_idx=0,
                                masks=_carry_masks,
                                object_ids=_carry_ids,
                            )
                            _injected_carry = True
                            print(f"    ↳ Injected {len(_carry_ids)} mask(s) from previous chunk")
                        except Exception as _inj_err:
                            print(f"\033[93m    ⚠ Carry injection failed: {_inj_err}\033[0m")

                    # ── Memory bank restoration ──
                    # After inject_masks creates a tracker state, restore past
                    # spatial memory frames at negative indices so propagation
                    # starts with rich context from the previous chunk.
                    if _injected_carry and prompt in memory_banks and memory_banks[prompt]:
                        try:
                            _n_restored = driver.restore_memory_bank(session_id, memory_banks[prompt])
                            if _n_restored:
                                print(f"    ↳ Restored {_n_restored} memory frame(s)")
                        except Exception as _mb_err:
                            print(f"\033[93m    ⚠ Memory bank restore failed: {_mb_err}\033[0m")

                    

                    # ── Monitored streaming propagation ──
                    # If cross-chunk carry was injected successfully, propagate those
                    # carried objects directly. Re-running the text prompt here causes
                    # fresh instance discovery on every chunk boundary, which explodes
                    # the object count for broad prompts like "player".
                    if _injected_carry:
                        print("    ↳ Reusing injected objects; skipping text re-detection")
                    else:
                        driver.add_prompt(session_id, prompt)

                    # Use frame cap from any prior prompt's early stop
                    effective_frames = _chunk_frame_cap if _chunk_frame_cap else chunk_frames
                    expected_iters = effective_frames * 2  # "both" → forward + backward
                    monitor = IntraChunkMonitor(
                        expected_iterations=expected_iters,
                        device=device,
                        vram_limit_bytes=max_vram_bytes,
                    )
                    monitor.start()

                    result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                        driver,
                        session_id,
                        monitor,
                        propagation_direction="both",
                        max_output_frames=_chunk_frame_cap,
                    )
                    _chunk_monitor_results.append(monitor.to_dict())

                    actual_frames = len(result) if result else 0

                    if early_stopped:
                        mon_result = monitor.finalize()
                        if mon_result.stop_reason == "oom_exception":
                            _chunk_oom = True
                            print(f"\033[91m    ✗ CUDA OOM during '{prompt}' propagation\033[0m")
                        else:
                            if not _chunk_early_stop:
                                # First early stop — record monitor for calibration
                                _chunk_early_stop = True
                                _early_stop_monitor = monitor
                            print(
                                f"\033[93m    ⚠ Proactive stop during '{prompt}': "
                                f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m"
                            )

                        # ── Save partial results + set frame cap for remaining prompts ──
                        if result:
                            new_cap = len(result)
                            _chunk_frame_cap = min(_chunk_frame_cap, new_cap) if _chunk_frame_cap else new_cap
                            _partial_frames_processed = _chunk_frame_cap
                            print(f"    Saving partial results: {new_cap} frames (frame cap → {_chunk_frame_cap})")

                            prev_masks = carry.get(prompt, {})
                            gnid = global_next_ids.get(prompt, 0)
                            result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(
                                result, obj_ids, prev_masks, gnid
                            )
                            global_next_ids[prompt] = gnid
                            all_object_ids.setdefault(prompt, set()).update(obj_ids)
                            chunk_n_objects += len(obj_ids)

                            chunk_objects[prompt] = {
                                "object_ids": sorted(obj_ids),
                                "num_objects": len(obj_ids),
                                "id_mapping": {str(k): v for k, v in mapping.items()},
                                "partial": True,
                                "partial_frames": new_cap,
                            }

                            masks_dir = chunk_dir / "masks" / safe
                            masks_dir.mkdir(parents=True, exist_ok=True)
                            _ensure_cpu_masks(result)
                            async_worker.submit(
                                _save_chunk_masks,
                                result,
                                obj_ids,
                                masks_dir,
                                video_metadata["width"],
                                video_metadata["height"],
                                new_cap,
                                video_metadata.get("fps", 25),
                            )
                            carry[prompt] = _extract_last_frame_masks(result, obj_ids)

                            # ── Extract memory bank (partial — still useful for next chunk) ──
                            if _cross_chunk_inject and n_chunks > 1:
                                try:
                                    memory_banks[prompt] = driver.extract_memory_bank(
                                        session_id, max_frames=_memory_bank_max_frames
                                    )
                                except Exception:
                                    pass

                            print(f"    {len(obj_ids)} object(s) (partial)")

                            del result, obj_ids, frame_objs
                            try:
                                del mapping, iou_mat
                            except NameError:
                                pass

                        clear_memory(device, full_gc=True)
                        continue  # ← process remaining prompts (NOT break)

                    # ── Normal or frame-capped completion ──
                    save_frame_count = actual_frames  # respects frame cap
                    frames_processed += save_frame_count
                    mem_predictor.record_frame(frames_processed)

                    prev_masks = carry.get(prompt, {})
                    gnid = global_next_ids.get(prompt, 0)

                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(result, obj_ids, prev_masks, gnid)
                    global_next_ids[prompt] = gnid

                    # Store IoU matrix for cross-chunk metadata
                    if iou_mat and ci > 0:
                        key = f"chunk_{ci - 1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    # Track all IDs
                    all_object_ids.setdefault(prompt, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    chunk_objects[prompt] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    # Save masks (async — overlap I/O with next prompt's GPU work)
                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result,
                        obj_ids,
                        masks_dir,
                        video_metadata["width"],
                        video_metadata["height"],
                        save_frame_count,
                        video_metadata.get("fps", 25),
                    )

                    # Extract carry-forward
                    carry[prompt] = _extract_last_frame_masks(result, obj_ids)

                    # ── Extract memory bank (spatial memory for next chunk) ──
                    if _cross_chunk_inject and n_chunks > 1:
                        try:
                            memory_banks[prompt] = driver.extract_memory_bank(
                                session_id, max_frames=_memory_bank_max_frames
                            )
                        except Exception:
                            pass  # non-critical

                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors so GPU memory is available for next prompt
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

            # ----- Point prompts -----
            if points:
                prompt_key = "__points__"
                safe = "points"
                vid_info = video_metadata
                print(f"  Points: {points}")

                driver.reset_session(session_id)

                # ── Cross-chunk mask injection for point prompts ──
                if _cross_chunk_inject and ci > 0 and prompt_key in carry and carry[prompt_key]:
                    _carry_masks = carry[prompt_key]
                    _carry_ids = sorted(_carry_masks.keys())
                    try:
                        driver.inject_masks(
                            session_id,
                            frame_idx=0,
                            masks=_carry_masks,
                            object_ids=_carry_ids,
                        )
                        print(f"    ↳ Injected {len(_carry_ids)} mask(s) from previous chunk")
                    except Exception as _inj_err:
                        print(f"\033[93m    ⚠ Carry injection failed: {_inj_err}\033[0m")

                # ── Memory bank restoration for point prompts ──
                if _cross_chunk_inject and ci > 0 and prompt_key in memory_banks and memory_banks[prompt_key]:
                    try:
                        _n_restored = driver.restore_memory_bank(session_id, memory_banks[prompt_key])
                        if _n_restored:
                            print(f"    ↳ Restored {_n_restored} memory frame(s)")
                    except Exception as _mb_err:
                        print(f"\033[93m    ⚠ Memory bank restore failed: {_mb_err}\033[0m")

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

                # ── Monitored streaming propagation ──
                effective_frames = _chunk_frame_cap if _chunk_frame_cap else chunk_frames
                expected_iters = effective_frames * 2
                monitor = IntraChunkMonitor(
                    expected_iterations=expected_iters,
                    device=device,
                    vram_limit_bytes=max_vram_bytes,
                )
                monitor.start()

                result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                    driver,
                    session_id,
                    monitor,
                    propagation_direction="both",
                    max_output_frames=_chunk_frame_cap,
                )
                _chunk_monitor_results.append(monitor.to_dict())

                actual_frames = len(result) if result else 0

                if early_stopped:
                    mon_result = monitor.finalize()
                    if mon_result.stop_reason == "oom_exception":
                        _chunk_oom = True
                        print("\033[91m    ✗ CUDA OOM during points propagation\033[0m")
                    else:
                        if not _chunk_early_stop:
                            _chunk_early_stop = True
                            _early_stop_monitor = monitor
                        print(
                            f"\033[93m    ⚠ Proactive stop during points: "
                            f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m"
                        )

                    # ── Save partial results + update frame cap ──
                    if result:
                        new_cap = len(result)
                        _chunk_frame_cap = min(_chunk_frame_cap, new_cap) if _chunk_frame_cap else new_cap
                        _partial_frames_processed = _chunk_frame_cap
                        print(f"    Saving partial results: {new_cap} frames (frame cap → {_chunk_frame_cap})")

                        prev_masks = carry.get(prompt_key, {})
                        gnid = global_next_ids.get(prompt_key, 0)
                        result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(result, obj_ids, prev_masks, gnid)
                        global_next_ids[prompt_key] = gnid
                        all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                        chunk_n_objects += len(obj_ids)

                        chunk_objects[prompt_key] = {
                            "object_ids": sorted(obj_ids),
                            "num_objects": len(obj_ids),
                            "id_mapping": {str(k): v for k, v in mapping.items()},
                            "partial": True,
                            "partial_frames": new_cap,
                        }

                        masks_dir = chunk_dir / "masks" / safe
                        masks_dir.mkdir(parents=True, exist_ok=True)
                        _ensure_cpu_masks(result)
                        async_worker.submit(
                            _save_chunk_masks,
                            result,
                            obj_ids,
                            masks_dir,
                            vid_info["width"],
                            vid_info["height"],
                            new_cap,
                            video_metadata.get("fps", 25),
                        )
                        carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)
                        print(f"    {len(obj_ids)} object(s) (partial)")

                        del result, obj_ids, frame_objs
                        try:
                            del mapping, iou_mat
                        except NameError:
                            pass

                    clear_memory(device, full_gc=True)

                elif actual_frames > 0:
                    save_frame_count = actual_frames
                    frames_processed += save_frame_count
                    mem_predictor.record_frame(frames_processed)

                    prev_masks = carry.get(prompt_key, {})
                    gnid = global_next_ids.get(prompt_key, 0)
                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(result, obj_ids, prev_masks, gnid)
                    global_next_ids[prompt_key] = gnid
                    all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    if iou_mat and ci > 0:
                        key = f"chunk_{ci - 1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt_key] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    chunk_objects[prompt_key] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result,
                        obj_ids,
                        masks_dir,
                        vid_info["width"],
                        vid_info["height"],
                        save_frame_count,
                        video_metadata.get("fps", 25),
                    )
                    carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)

                    # ── Extract memory bank (points normal) ──
                    if _cross_chunk_inject and n_chunks > 1:
                        try:
                            memory_banks[prompt_key] = driver.extract_memory_bank(
                                session_id, max_frames=_memory_bank_max_frames
                            )
                        except Exception:
                            pass

                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors so GPU memory is available for next prompt
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

            # ----- Mask prompts -----
            if mask_paths:
                prompt_key = "__masks__"
                safe = "masks"
                vid_info = video_metadata
                print(f"  Masks: {[m.name for m in mask_paths]}")

                driver.reset_session(session_id)

                # ── Cross-chunk mask injection for mask prompts ──
                if _cross_chunk_inject and ci > 0 and prompt_key in carry and carry[prompt_key]:
                    _carry_masks = carry[prompt_key]
                    _carry_ids = sorted(_carry_masks.keys())
                    try:
                        driver.inject_masks(
                            session_id,
                            frame_idx=0,
                            masks=_carry_masks,
                            object_ids=_carry_ids,
                        )
                        print(f"    ↳ Injected {len(_carry_ids)} mask(s) from previous chunk")
                    except Exception as _inj_err:
                        print(f"\033[93m    ⚠ Carry injection failed: {_inj_err}\033[0m")

                # ── Memory bank restoration for mask prompts ──
                if _cross_chunk_inject and ci > 0 and prompt_key in memory_banks and memory_banks[prompt_key]:
                    try:
                        _n_restored = driver.restore_memory_bank(session_id, memory_banks[prompt_key])
                        if _n_restored:
                            print(f"    ↳ Restored {_n_restored} memory frame(s)")
                    except Exception as _mb_err:
                        print(f"\033[93m    ⚠ Memory bank restore failed: {_mb_err}\033[0m")

                # Inject each user-provided mask as a separate object
                mask_dict = {}
                obj_id_list = []
                for mi, mp in enumerate(mask_paths):
                    m = np.array(Image.open(mp).convert("L"))
                    mask_dict[mi] = m
                    obj_id_list.append(mi)

                driver.inject_masks(session_id, frame_idx=0, masks=mask_dict, object_ids=obj_id_list)

                # ── Monitored streaming propagation ──
                effective_frames = _chunk_frame_cap if _chunk_frame_cap else chunk_frames
                expected_iters = effective_frames * 2
                monitor = IntraChunkMonitor(
                    expected_iterations=expected_iters,
                    device=device,
                    vram_limit_bytes=max_vram_bytes,
                )
                monitor.start()

                result, obj_ids, frame_objs, early_stopped = _propagate_with_monitoring(
                    driver,
                    session_id,
                    monitor,
                    propagation_direction="both",
                    max_output_frames=_chunk_frame_cap,
                )
                _chunk_monitor_results.append(monitor.to_dict())

                actual_frames = len(result) if result else 0

                if early_stopped:
                    mon_result = monitor.finalize()
                    if mon_result.stop_reason == "oom_exception":
                        _chunk_oom = True
                        print("\033[91m    ✗ CUDA OOM during mask propagation\033[0m")
                    else:
                        if not _chunk_early_stop:
                            _chunk_early_stop = True
                            _early_stop_monitor = monitor
                        print(
                            f"\033[93m    ⚠ Proactive stop during masks: "
                            f"{mon_result.stop_reason} at iter {mon_result.iterations_completed}/{expected_iters}\033[0m"
                        )

                    # ── Save partial results + update frame cap ──
                    if result:
                        new_cap = len(result)
                        _chunk_frame_cap = min(_chunk_frame_cap, new_cap) if _chunk_frame_cap else new_cap
                        _partial_frames_processed = _chunk_frame_cap
                        print(f"    Saving partial results: {new_cap} frames (frame cap → {_chunk_frame_cap})")

                        prev_masks_cf = carry.get(prompt_key, {})
                        gnid = global_next_ids.get(prompt_key, 0)
                        result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(result, obj_ids, prev_masks_cf, gnid)
                        global_next_ids[prompt_key] = gnid
                        all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                        chunk_n_objects += len(obj_ids)

                        chunk_objects[prompt_key] = {
                            "object_ids": sorted(obj_ids),
                            "num_objects": len(obj_ids),
                            "id_mapping": {str(k): v for k, v in mapping.items()},
                            "partial": True,
                            "partial_frames": new_cap,
                        }

                        masks_dir = chunk_dir / "masks" / safe
                        masks_dir.mkdir(parents=True, exist_ok=True)
                        _ensure_cpu_masks(result)
                        async_worker.submit(
                            _save_chunk_masks,
                            result,
                            obj_ids,
                            masks_dir,
                            vid_info["width"],
                            vid_info["height"],
                            new_cap,
                            video_metadata.get("fps", 25),
                        )
                        carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)

                        # ── Extract memory bank (masks partial) ──
                        if _cross_chunk_inject and n_chunks > 1:
                            try:
                                memory_banks[prompt_key] = driver.extract_memory_bank(
                                    session_id, max_frames=_memory_bank_max_frames
                                )
                            except Exception:
                                pass

                        print(f"    {len(obj_ids)} object(s) (partial)")

                        del result, obj_ids, frame_objs
                        try:
                            del mapping, iou_mat
                        except NameError:
                            pass

                    clear_memory(device, full_gc=True)

                elif actual_frames > 0:
                    save_frame_count = actual_frames
                    frames_processed += save_frame_count
                    mem_predictor.record_frame(frames_processed)

                    prev_masks_cf = carry.get(prompt_key, {})
                    gnid = global_next_ids.get(prompt_key, 0)
                    result, obj_ids, mapping, gnid, iou_mat = _match_and_remap(result, obj_ids, prev_masks_cf, gnid)
                    global_next_ids[prompt_key] = gnid
                    all_object_ids.setdefault(prompt_key, set()).update(obj_ids)
                    chunk_n_objects += len(obj_ids)

                    if iou_mat and ci > 0:
                        key = f"chunk_{ci - 1}_to_{ci}"
                        cross_chunk_iou.setdefault(key, {})[prompt_key] = {
                            "matrix": {str(k): {str(pk): v for pk, v in row.items()} for k, row in iou_mat.items()},
                            "matched": {str(k): v for k, v in mapping.items()},
                            "threshold": 0.25,
                        }

                    chunk_objects[prompt_key] = {
                        "object_ids": sorted(obj_ids),
                        "num_objects": len(obj_ids),
                        "id_mapping": {str(k): v for k, v in mapping.items()},
                    }

                    masks_dir = chunk_dir / "masks" / safe
                    masks_dir.mkdir(parents=True, exist_ok=True)
                    _ensure_cpu_masks(result)
                    async_worker.submit(
                        _save_chunk_masks,
                        result,
                        obj_ids,
                        masks_dir,
                        vid_info["width"],
                        vid_info["height"],
                        save_frame_count,
                        video_metadata.get("fps", 25),
                    )
                    carry[prompt_key] = _extract_last_frame_masks(result, obj_ids)

                    # ── Extract memory bank (masks normal) ──
                    if _cross_chunk_inject and n_chunks > 1:
                        try:
                            memory_banks[prompt_key] = driver.extract_memory_bank(
                                session_id, max_frames=_memory_bank_max_frames
                            )
                        except Exception:
                            pass

                    print(f"    {len(obj_ids)} object(s)")

                    # Free result tensors
                    del result, obj_ids, frame_objs, mapping, iou_mat
                    clear_memory(device, full_gc=False)

        finally:
            driver.close_session(session_id)

        # ── Measure peak memory for this chunk ──
        # Use the maximum peak across ALL per-prompt monitors — not just
        # torch.cuda.max_memory_allocated() at chunk end.  The latter may
        # underreport because clear_memory() between prompts releases the
        # caching allocator.  A heavy prompt ("person", 26 objects) can
        # hit 80%+ VRAM while a lightweight follow-up ("tennis racket",
        # 3 objects) brings the post-chunk reading back down.  Adaptive
        # sizing must be driven by the WORST prompt, because the video is
        # loaded once and every prompt runs on every chunk.
        peak_vram = 0
        peak_ram = 0
        if device.startswith("cuda"):
            import torch as _t

            if _t.cuda.is_available():
                peak_vram = _t.cuda.max_memory_allocated()
        try:
            peak_ram = psutil.Process().memory_info().rss
        except Exception:
            pass

        # Aggregate worst-case from per-prompt monitors (if available)
        any_soft_warning = False
        if _chunk_monitor_results:
            for mres in _chunk_monitor_results:
                m_peak_vram = int(mres.get("peak_vram_mb", 0) * (1024**2))
                m_peak_ram = int(mres.get("peak_ram_mb", 0) * (1024**2))
                peak_vram = max(peak_vram, m_peak_vram)
                peak_ram = max(peak_ram, m_peak_ram)
                if mres.get("soft_warning_issued") or mres.get("ram_soft_warning_issued"):
                    any_soft_warning = True

        # ── Update prompt memory profile for reordering (heaviest first) ──
        monitor_idx = 0
        if prompts:
            for p in prompts:
                if monitor_idx < len(_chunk_monitor_results):
                    _prompt_memory_profile[p] = _chunk_monitor_results[monitor_idx].get("peak_vram_mb", 0)
                    monitor_idx += 1

        # ── Handle OOM: save partial + rechunk remaining frames ──
        if _chunk_oom:
            print(
                f"\033[93m  ⚠ OOM on chunk {ci + 1} (size={current_chunk_frames}). "
                f"Reducing chunk size and replanning...\033[0m"
            )
            try:
                new_size = adaptive.handle_oom(chunk_id, current_chunk_frames)
            except RuntimeError as exc:
                print(f"\033[91m  ✗ {exc}\033[0m")
                break

            adaptive.current_chunk_size = new_size

            if _partial_frames_processed > 0:
                # ── Partial chunk saved: advance from partial boundary ──
                actual_end = start_frame + _partial_frames_processed - 1
                resume_frame = actual_end + 1 - overlap

                # Record partial chunk metadata
                chunk_dur = round(time.time() - t_chunk_start, 3)
                chunk_timing.append(
                    {
                        "chunk_id": chunk_id,
                        "frames": _partial_frames_processed,
                        "duration_s": chunk_dur,
                        "s_per_frame": round(chunk_dur / max(_partial_frames_processed, 1), 3),
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "peak_ram_mb": round(peak_ram / (1024**2), 1),
                        "n_objects": chunk_n_objects,
                        "pressure": "OOM_PARTIAL",
                        "action": "SHRINK",
                        "intra_chunk_monitors": _chunk_monitor_results,
                        "partial": True,
                        "partial_frames": _partial_frames_processed,
                    }
                )
                chunk_meta_list.append(
                    {
                        "chunk_id": chunk_id,
                        "start_frame": start_frame,
                        "end_frame": actual_end,
                        "start_frame_original": start_frame + frame_offset,
                        "end_frame_original": actual_end + frame_offset,
                        "total_frames": _partial_frames_processed,
                        "processing_time_s": chunk_dur,
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "peak_ram_mb": round(peak_ram / (1024**2), 1),
                        "n_objects": chunk_n_objects,
                        "memory_pressure": "OOM_PARTIAL",
                        "adaptive_action": "SHRINK",
                        "next_chunk_size": new_size,
                        "objects_by_prompt": chunk_objects,
                        "intra_chunk_monitors": _chunk_monitor_results,
                        "partial": True,
                    }
                )

                # Update chunk_list entry for stitching
                cinfo["end"] = actual_end

                frames_processed += _partial_frames_processed

                print(
                    f"  Partial chunk saved ({_partial_frames_processed} frames). "
                    f"Resuming from frame {resume_frame}, "
                    f"new chunk size = {new_size} frames"
                )

                new_chunks = adaptive.replan_remaining(
                    resume_frame,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                if not new_chunks:
                    print("\033[91m  ✗ No viable chunks after OOM reduction\033[0m")
                    break
                chunk_list = chunk_list[: chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)
                chunk_cursor += 1  # advance past partial chunk
            else:
                # No partial frames — retry from start_frame (old behavior)
                new_chunks = adaptive.replan_remaining(
                    start_frame,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                if not new_chunks:
                    print("\033[91m  ✗ No viable chunks after OOM reduction\033[0m")
                    break
                chunk_list = chunk_list[:chunk_cursor] + new_chunks
                n_chunks = len(chunk_list)
                print(f"  Replanned: {len(new_chunks)} chunk(s) remaining, new chunk size = {new_size} frames")

            ci += 1
            continue

        # ── Handle proactive early stop: save partial + smart replan ──
        if _chunk_early_stop and _early_stop_monitor is not None:
            mon_result = _early_stop_monitor.finalize()
            cal = mon_result.calibration

            # Determine whether the stop was triggered by VRAM or RAM.
            _stop_was_vram = mon_result.stop_reason in (
                "hard_limit",
                "predictive_soft_stop",
                "soft_limit_no_calibration",
            )
            _stop_was_ram = mon_result.stop_reason in (
                "ram_hard_limit",
                "ram_soft_limit",
            )

            # ── Special case: offload_state_to_cpu + VRAM-triggered stop ──
            # When offloading is active, per-frame state accumulates in RAM,
            # NOT VRAM.  VRAM usage is approximately frame-independent (model
            # weights + forward-pass peak + caching allocator overhead).  A
            # VRAM soft/hard stop in this mode means the forward-pass peak is
            # intrinsically high — reducing chunk size will NOT reduce VRAM
            # usage.  Instead, apply a modest reduction (SHRINK_WARNING_FACTOR)
            # based on the frames that were actually processed successfully,
            # and let the intra-chunk monitor manage VRAM on the next chunk.
            if _offload_state and _stop_was_vram:
                # The actually-processed frames already worked — the VRAM peak
                # is structural, not frame-driven.  Keep most of them.
                if _partial_frames_processed > 0:
                    safe_frames = max(
                        int(_partial_frames_processed * adaptive.SHRINK_WARNING_FACTOR),
                        adaptive.min_chunk_frames,
                    )
                else:
                    safe_frames = max(
                        int(current_chunk_frames * adaptive.SHRINK_WARNING_FACTOR),
                        adaptive.min_chunk_frames,
                    )
            else:
                # ── Standard path: use calibration to compute frames that target
                # SHRINK_TARGET_PCT of the *bounding* resource.
                # mem = baseline + growth_rate × iterations
                # target: baseline + growth_rate × target_iters = eff_limit × target_pct
                # → target_iters = (eff_limit × target_pct − baseline) / growth_rate
                _eff_limit = adaptive._effective_limit  # RAM when offloading, VRAM otherwise
                target_pct = adaptive.SHRINK_TARGET_PCT
                if cal and cal.growth_rate_per_iter > 0 and cal.baseline_bytes > 0 and _eff_limit > 0:
                    target_mem = _eff_limit * target_pct
                    target_growth = target_mem - cal.baseline_bytes
                    if target_growth > 0:
                        target_iters = target_growth / cal.growth_rate_per_iter
                        safe_frames = max(int(target_iters / 2), adaptive.min_chunk_frames)
                    else:
                        safe_frames = adaptive.min_chunk_frames
                elif cal and cal.safe_iterations > 0:
                    try:
                        from sam3.__globals import VRAM_HARD_LIMIT_PCT

                        hard_pct = VRAM_HARD_LIMIT_PCT
                    except ImportError:
                        hard_pct = 0.975
                    safe_frames = max(
                        int(cal.safe_iterations // 2 * (target_pct / hard_pct)),
                        adaptive.min_chunk_frames,
                    )
                else:
                    safe_frames = current_chunk_frames // 2

            # ── Floor: never shrink below a fraction of actually-processed frames ──
            # If we successfully processed N frames, shrinking to N/30 is absurd.
            # Keep at least 50% of what actually worked (the monitor will catch
            # overshoot on the next chunk if needed).
            if _partial_frames_processed > 0:
                _actual_floor = max(
                    int(_partial_frames_processed * 0.50),
                    adaptive.min_chunk_frames,
                )
                safe_frames = max(safe_frames, _actual_floor)

            safe_frames = max(safe_frames, adaptive.min_chunk_frames)
            adaptive.current_chunk_size = safe_frames

            adaptive.rechunk_events.append(
                {
                    "chunk_id": chunk_id,
                    "from_size": current_chunk_frames,
                    "to_size": safe_frames,
                    "reason": f"proactive_{mon_result.stop_reason}",
                    "calibration": {
                        "growth_rate_mb": round(cal.growth_rate_per_iter / (1024**2), 3) if cal else None,
                        "safe_iters": cal.safe_iterations if cal else None,
                        "confidence": cal.confidence if cal else None,
                    },
                }
            )

            if _partial_frames_processed > 0:
                # ── Partial chunk saved: advance from partial boundary ──
                actual_end = start_frame + _partial_frames_processed - 1
                resume_frame = actual_end + 1 - overlap

                chunk_dur = round(time.time() - t_chunk_start, 3)
                chunk_timing.append(
                    {
                        "chunk_id": chunk_id,
                        "frames": _partial_frames_processed,
                        "duration_s": chunk_dur,
                        "s_per_frame": round(chunk_dur / max(_partial_frames_processed, 1), 3),
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "peak_ram_mb": round(peak_ram / (1024**2), 1),
                        "n_objects": chunk_n_objects,
                        "pressure": f"EARLY_STOP_{mon_result.stop_reason}",
                        "action": "SHRINK",
                        "intra_chunk_monitors": _chunk_monitor_results,
                        "partial": True,
                        "partial_frames": _partial_frames_processed,
                    }
                )
                chunk_meta_list.append(
                    {
                        "chunk_id": chunk_id,
                        "start_frame": start_frame,
                        "end_frame": actual_end,
                        "start_frame_original": start_frame + frame_offset,
                        "end_frame_original": actual_end + frame_offset,
                        "total_frames": _partial_frames_processed,
                        "processing_time_s": chunk_dur,
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "peak_ram_mb": round(peak_ram / (1024**2), 1),
                        "n_objects": chunk_n_objects,
                        "memory_pressure": f"EARLY_STOP_{mon_result.stop_reason}",
                        "adaptive_action": "SHRINK",
                        "next_chunk_size": safe_frames,
                        "objects_by_prompt": chunk_objects,
                        "intra_chunk_monitors": _chunk_monitor_results,
                        "partial": True,
                    }
                )

                # Update chunk_list entry for stitching
                cinfo["end"] = actual_end

                frames_processed += _partial_frames_processed

                _bounding_label = "RAM" if _offload_state else "VRAM"
                _stop_note = " (VRAM-bounded, offloading active)" if _offload_state and _stop_was_vram else ""
                print(
                    f"\033[93m  ⚠ Proactive stop on chunk {ci + 1}: {mon_result.stop_reason}{_stop_note}. "
                    f"Saved {_partial_frames_processed} frames. "
                    f"Next chunk: {current_chunk_frames} → {safe_frames} frames\033[0m"
                )

                new_chunks = adaptive.replan_remaining(
                    resume_frame,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                if not new_chunks:
                    print("\033[91m  ✗ No viable chunks after proactive resizing\033[0m")
                    break
                chunk_list = chunk_list[: chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)
                chunk_cursor += 1  # advance past partial chunk
            else:
                # No partial frames — retry from start_frame (old behavior)
                _bounding_label = "RAM" if _offload_state else "VRAM"
                _stop_note = " (VRAM-bounded, offloading active)" if _offload_state and _stop_was_vram else ""
                print(
                    f"\033[93m  ⚠ Proactive stop on chunk {ci + 1}: {mon_result.stop_reason}{_stop_note}. "
                    f"Next chunk: {current_chunk_frames} → {safe_frames} frames\033[0m"
                )

                new_chunks = adaptive.replan_remaining(
                    start_frame,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                if not new_chunks:
                    print("\033[91m  ✗ No viable chunks after proactive resizing\033[0m")
                    break
                chunk_list = chunk_list[:chunk_cursor] + new_chunks
                n_chunks = len(chunk_list)
                print(f"  Replanned: {len(new_chunks)} chunk(s) remaining, new chunk size = {safe_frames} frames")

            ci += 1
            continue

        # ── Adaptive sizing: evaluate pressure and adjust for next chunk ──
        # Extract calibration from the heaviest prompt (highest peak VRAM)
        # for precision-targeted GROW and SHRINK decisions.
        _baseline_vram = 0
        _growth_rate = 0.0
        _heaviest_peak = 0
        _calibration_confidence = 0.0
        for mres in _chunk_monitor_results:
            cal_data = mres.get("calibration") if isinstance(mres, dict) else None
            mres_peak = mres.get("peak_vram_mb", 0) if isinstance(mres, dict) else 0
            if cal_data and mres_peak >= _heaviest_peak:
                _heaviest_peak = mres_peak
                if cal_data.get("baseline_mb"):
                    _baseline_vram = int(cal_data["baseline_mb"] * (1024**2))
                if cal_data.get("growth_rate_mb_per_iter"):
                    _growth_rate = cal_data["growth_rate_mb_per_iter"] * (1024**2)
                _calibration_confidence = cal_data.get("r_squared", 0.0)

        rec = adaptive.record_chunk(
            chunk_id=chunk_id,
            chunk_size=current_chunk_frames,
            peak_vram_bytes=peak_vram,
            peak_ram_bytes=peak_ram,
            n_objects=chunk_n_objects,
            soft_warning_seen=any_soft_warning,
            baseline_vram_bytes=_baseline_vram,
            growth_rate_per_iter=_growth_rate,
            calibration_confidence=_calibration_confidence,
        )

        _bounding = "RAM" if _offload_state else "VRAM"

        if rec.action == "SHRINK":
            print(
                f"\033[93m  ⚠ Memory pressure {rec.pressure} "
                f"(peak {rec.vram_usage_pct:.0f}% of effective {_bounding}). "
                f"Targeting {rec.target_utilization_pct:.0f}% → "
                f"next chunk: {current_chunk_frames} → {rec.adjusted_chunk_size} frames\033[0m"
            )
            # Replan remaining chunks with smaller size
            next_start = end_frame + 1 - overlap
            if next_start < total_frames_in_video:
                new_chunks = adaptive.replan_remaining(
                    next_start,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                chunk_list = chunk_list[: chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)
                print(f"  Replanned: {len(new_chunks)} chunk(s) remaining")
        elif rec.action == "GROW":
            _grow_method = "calibrated" if _growth_rate > 0 and _baseline_vram > 0 else "heuristic"
            print(
                f"\033[92m  ↑ {_bounding} under-utilised (peak {rec.vram_usage_pct:.0f}%). "
                f"Targeting {rec.target_utilization_pct:.0f}% ({_grow_method}) → "
                f"next chunk: {current_chunk_frames} → {rec.adjusted_chunk_size} frames\033[0m"
            )
            next_start = end_frame + 1 - overlap
            if next_start < total_frames_in_video:
                new_chunks = adaptive.replan_remaining(
                    next_start,
                    total_frames_in_video,
                    overlap,
                    start_chunk_id=ci + 1,
                )
                chunk_list = chunk_list[: chunk_cursor + 1] + new_chunks
                n_chunks = len(chunk_list)

        chunk_dur = round(time.time() - t_chunk_start, 3)
        chunk_timing.append(
            {
                "chunk_id": chunk_id,
                "frames": chunk_frames,
                "duration_s": chunk_dur,
                "s_per_frame": round(chunk_dur / max(chunk_frames, 1), 3),
                "peak_vram_mb": round(peak_vram / (1024**2), 1),
                "peak_ram_mb": round(peak_ram / (1024**2), 1),
                "n_objects": chunk_n_objects,
                "pressure": rec.pressure,
                "action": rec.action,
                "intra_chunk_monitors": _chunk_monitor_results,
            }
        )
        chunk_meta_list.append(
            {
                "chunk_id": chunk_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_frame_original": start_frame + frame_offset,
                "end_frame_original": end_frame + frame_offset,
                "total_frames": chunk_frames,
                "processing_time_s": chunk_dur,
                "peak_vram_mb": round(peak_vram / (1024**2), 1),
                "peak_ram_mb": round(peak_ram / (1024**2), 1),
                "n_objects": chunk_n_objects,
                "memory_pressure": rec.pressure,
                "adaptive_action": rec.action,
                "next_chunk_size": rec.adjusted_chunk_size,
                "objects_by_prompt": chunk_objects,
                "intra_chunk_monitors": _chunk_monitor_results,
            }
        )
        print(
            f"  Chunk {ci + 1} done in {chunk_dur:.1f}s "
            f"({chunk_dur / max(chunk_frames, 1):.2f} s/frame, "
            f"peak VRAM: {peak_vram / (1024**2):.0f} MB, "
            f"peak RAM: {peak_ram / (1024**2):.0f} MB, "
            f"pressure ({_bounding}): {rec.pressure})\n"
        )

        chunk_cursor += 1
        ci += 1

        # ── RAM guard: trim carry and memory banks if memory pressure is high ──
        if carry:
            _trim_carry_if_needed(carry, _carry_max_ram_pct, _carry_min_free_gb)
        if memory_banks:
            _trim_memory_bank_if_needed(memory_banks, _carry_max_ram_pct, _carry_min_free_gb)

    chunk_processing_s = round(time.time() - t_chunks_start, 3)

    # ----- Drain async I/O before stitching -----
    io_errors = async_worker.drain()
    if io_errors:
        print(f"\033[93m  ⚠ {io_errors} async I/O error(s) during mask writing\033[0m")

    # ----- Stitch masks and create overlay -----
    print("Stitching masks...")
    t_stitch_start = time.time()
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

        # Clear previous stitched outputs for this prompt so reruns do not
        # leak stale object videos into the new overlay.
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        # ── Stitch per-chunk mask MP4s into final per-object mask MP4s ──
        from sam3.streaming_masks import create_overlay_from_masks, stitch_chunk_mask_videos

        stitch_chunk_mask_videos(chunks_dir, safe, oids, chunk_list, overlap, out, fps, w, h)

        # Overlay
        mask_vids = sorted(out.glob("object_*_mask.mp4"))
        if mask_vids:
            overlay_path = video_output / f"overlay_{safe}.mp4"
            if overlay_path.exists():
                overlay_path.unlink()
            create_overlay_from_masks(video_path, mask_vids, overlay_path, alpha)

    stitching_s = round(time.time() - t_stitch_start, 3)

    # ----- Per-object temporal analysis -----
    print("Analysing object presence...")
    t_analysis_start = time.time()
    objects_tracking: dict[str, list[dict[str, Any]]] = {}
    for pk in prompt_keys:
        safe = sanitize_filename(pk) if not pk.startswith("__") else pk.strip("_")
        oids = all_object_ids.get(pk, set())
        mask_out = video_output / "masks" / safe
        if oids and mask_out.exists():
            objects_tracking[pk] = _build_object_tracking(mask_out, oids, fps, frame_offset)

    analysis_s = round(time.time() - t_analysis_start, 3)

    # ----- Cleanup -----
    driver.cleanup()
    async_worker.shutdown()

    if keep_temp:
        dest = video_output / "temp_files"
        if temp_base.exists():
            shutil.copytree(temp_base, dest, dirs_exist_ok=True)
            print(f"  Temp files preserved at: {dest}")

    if temp_base.exists():
        shutil.rmtree(temp_base)

    # ── Stop memory sampler & predictor ──
    mem_sampler.stop()
    mem_summary = mem_sampler.summary()
    mem_predictor.stop()
    predictor_summary = mem_predictor.summary()

    # ── Timing summary ──
    pipeline_end = time.time()
    pipeline_end_iso = datetime.now().isoformat()
    total_s = round(pipeline_end - pipeline_start, 3)

    # ── Thread config ──
    thread_config = {
        "intra_op_threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
    }

    # ── Read SAM3 version ──
    sam3_version = "unknown"
    try:
        ver_file = Path(__file__).resolve().parent / "VERSION"
        if ver_file.exists():
            sam3_version = ver_file.read_text().strip()
    except Exception:
        pass

    # ----- Save enriched final metadata -----
    adaptive_summary = adaptive.to_dict()
    async_io_summary = async_worker.to_dict()

    final_meta = {
        "schema_version": "2.2.0",
        "sam3_version": sam3_version,
        "video": str(original_video),
        "video_name": video_name,
        "output_dir": str(video_output),
        "resolution": f"{w}x{h}",
        "total_frames": video_metadata.get("nb_frames"),
        "fps": fps,
        "duration_s": video_metadata.get("duration"),
        "frame_offset": frame_offset,
        "frame_range": list(frame_range) if frame_range else None,
        "time_range": list(time_range) if time_range else None,
        "device": device,
        "thread_config": thread_config,
        "prompts": prompts,
        "points": points,
        "mask_paths": [str(p) for p in mask_paths] if mask_paths else None,
        "num_chunks": n_chunks,
        "num_chunks_processed": ci,
        "overlap_frames": overlap,
        "cross_chunk_mask_injection": _cross_chunk_inject,
        "memory_bank_max_frames": _memory_bank_max_frames,
        "simulated_limits": {
            "max_vram_gb": max_vram_gb,
            "max_ram_gb": max_ram_gb,
        }
        if (max_vram_gb or max_ram_gb)
        else None,
        "timing": {
            "pipeline_start": pipeline_start_iso,
            "pipeline_end": pipeline_end_iso,
            "total_s": total_s,
            "model_load_s": model_load_s,
            "chunk_processing_s": chunk_processing_s,
            "stitching_s": stitching_s,
            "analysis_s": analysis_s,
            "per_chunk": chunk_timing,
        },
        "memory": {
            "pre_run": mem_info,
            **mem_summary,
        },
        "adaptive_chunking": adaptive_summary,
        "async_io": async_io_summary,
        "chunks": chunk_meta_list,
        "cross_chunk_iou": cross_chunk_iou if cross_chunk_iou else None,
        "memory_predictor": predictor_summary,
        "objects": objects_tracking,
    }
    with open(video_output / "metadata.json", "w") as f:
        json.dump(final_meta, f, indent=2, default=str)

    # Defensively re-create metadata/ — the directory is created early in the
    # pipeline (line ~1152) but may vanish during very long runs (filesystem
    # pressure, external cleanup, or OS reclamation).  The top-level
    # metadata.json already contains everything; these files are convenience
    # copies, so we also guard each write so a metadata-dir failure never
    # crashes the pipeline after masks are already saved.
    meta_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Also save the detailed cross-chunk IoU separately for evaluation
        if cross_chunk_iou:
            with open(meta_dir / "cross_chunk_iou.json", "w") as f:
                json.dump(cross_chunk_iou, f, indent=2, default=str)

        # Also save object tracking separately for easy access
        if objects_tracking:
            with open(meta_dir / "object_tracking.json", "w") as f:
                json.dump(objects_tracking, f, indent=2, default=str)

        # Save memory predictor data separately for analysis
        with open(meta_dir / "memory_predictor.json", "w") as f:
            json.dump(predictor_summary, f, indent=2, default=str)

        # Save adaptive chunking data separately for analysis
        with open(meta_dir / "adaptive_chunking.json", "w") as f:
            json.dump(adaptive_summary, f, indent=2, default=str)
    except OSError as exc:
        print(f"  ⚠ Could not write supplementary metadata files: {exc}")
        print("    (All data is still available in metadata.json)")

    print()
    print("=" * 70)
    print(f"  ✓ Video processing complete → {video_output}")
    print(
        f"    Total: {total_s:.1f}s  (model: {model_load_s:.1f}s, chunks: {chunk_processing_s:.1f}s, stitch: {stitching_s:.1f}s)"
    )
    if adaptive.rechunk_events:
        print(
            f"    Adaptive rechunks: {len(adaptive.rechunk_events)}  "
            f"(initial: {adaptive.initial_chunk_size}, final: {adaptive.current_chunk_size} frames/chunk)"
        )
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
        "--prompts",
        nargs="+",
        default=None,
        help="Text prompts (e.g. person ball)",
    )
    parser.add_argument(
        "--points",
        nargs="+",
        default=None,
        help="Click points as x,y pairs (e.g. 320,240 500,300)",
    )
    parser.add_argument(
        "--point-labels",
        nargs="+",
        type=int,
        default=None,
        help="Labels for each point (1=positive, 0=negative)",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        default=None,
        help="Mask image file(s) for initial object prompts",
    )
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay alpha (0.0–1.0, default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Force device (auto-detected if omitted)",
    )
    parser.add_argument(
        "--chunk-spread",
        type=str,
        default="default",
        choices=["default", "even"],
        help="Chunk size strategy (default or even)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve intermediate chunk files in output",
    )
    parser.add_argument(
        "--frame-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Process only frames START..END (0-based, inclusive)",
    )
    parser.add_argument(
        "--time-range",
        nargs=2,
        type=str,
        metavar=("START", "END"),
        help="Process a time segment (seconds, MM:SS, or HH:MM:SS)",
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=None,
        help="Simulate a smaller GPU by capping VRAM (GB). For testing adaptive chunking.",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=None,
        help="Simulate a smaller system by capping RAM (GB). For testing adaptive chunking.",
    )
    parser.add_argument(
        "--cpu-utilisation",
        type=int,
        default=100,
        metavar="PCT",
        help="Percentage of logical CPU cores to use (50-100, default: 100)",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action="store_true",
        default=None,
        help="Offload per-frame tracker state from VRAM to RAM (GPU only). "
        "Keeps VRAM flat at the cost of ~10-15%% slower propagation. "
        "Enabled by default on CUDA; pass this flag to force it on.",
    )
    parser.add_argument(
        "--no-offload-state-to-cpu",
        action="store_true",
        default=False,
        help="Disable tracker state offloading (keep all state on GPU). "
        "Uses more VRAM but avoids the ~10-15%% propagation overhead.",
    )

    args = parser.parse_args()

    # Validate: --offload-state-to-cpu and --no-offload-state-to-cpu are mutually exclusive
    if args.offload_state_to_cpu and args.no_offload_state_to_cpu:
        print("\033[91m✗ --offload-state-to-cpu and --no-offload-state-to-cpu are mutually exclusive.\033[0m")
        sys.exit(1)

    # Resolve offload tri-state: True (forced on), False (forced off), None (auto)
    offload_state: bool | None = None
    if args.offload_state_to_cpu:
        offload_state = True
    elif args.no_offload_state_to_cpu:
        offload_state = False
    # else: None → _process_video will auto-resolve from globals

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
            print("\033[91m✗ --point-labels count must match --points count\033[0m")
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
    if device == "cpu":
        print(f"  CPU util: {args.cpu_utilisation}%")
    if device == "cuda":
        if offload_state is True:
            print("  Offload : ON  (forced via CLI)")
        elif offload_state is False:
            print("  Offload : OFF (forced via CLI)")
        else:
            print("  Offload : auto (from __globals.py)")
    print(f"  Output  : {args.output}")
    print(f"  Alpha   : {args.alpha}")
    print(f"  Chunking: {args.chunk_spread}")
    if args.max_vram_gb:
        print(f"  Max VRAM: {args.max_vram_gb} GB (simulated)")
    if args.max_ram_gb:
        print(f"  Max RAM : {args.max_ram_gb} GB (simulated)")
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
        max_vram_gb=args.max_vram_gb,
        max_ram_gb=args.max_ram_gb,
        cpu_utilisation=args.cpu_utilisation,
        offload_state_to_cpu=offload_state,
    )


if __name__ == "__main__":
    main()
