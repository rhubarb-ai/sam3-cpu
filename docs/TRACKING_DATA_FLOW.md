# Tracking Data Flow In `video_prompter.py`

This note explains how tracking state moves through the video pipeline, especially across chunk boundaries.

## High-Level Flow

```text
input video
  -> chunk plan
  -> per-chunk session
  -> prompt / inject / propagate
  -> remap IDs to global IDs
  -> save chunk-local mask videos
  -> extract carry state for next chunk
  -> stitch chunk outputs into final mask videos
  -> build overlay video
```

The main orchestration lives in `video_prompter.py::_process_video()`.

## The Main State Containers

Inside `_process_video()`, these structures hold the cross-chunk tracking state:

- `chunk_list`
  The current chunk plan. Each item is `{"chunk": id, "start": s, "end": e}`.

- `carry`
  Per-prompt last-frame masks from the previous chunk.
  Shape:
  `dict[prompt_key, dict[global_object_id, mask_u8]]`

- `memory_banks`
  Per-prompt tracker memory frames extracted from the previous chunk.
  Shape:
  `dict[prompt_key, list[memory_frame_dict]]`

- `global_next_ids`
  Per-prompt next free global object ID.

- `all_object_ids`
  Per-prompt set of all global IDs seen across the whole run.

- `cross_chunk_iou`
  Debug/metadata structure storing IoU comparisons between adjacent chunks.

## What Happens On The First Chunk

For the first chunk of a text-prompt run:

1. A chunk video is created from the input video.
2. `driver.start_session()` creates a SAM3 tracking session for that chunk.
3. `driver.reset_session()` clears any prompt state inside that session.
4. No carry data exists yet, so nothing is injected.
5. `driver.add_prompt(session_id, prompt)` runs text-based detection.
6. `_propagate_with_monitoring()` tracks the detected objects through the chunk.
7. `_match_and_remap()` converts chunk-local object IDs into global object IDs.
8. `_save_chunk_masks()` writes per-object mask MP4 files into:
   `.sam3_temp/<video>/chunks/chunk_<id>/masks/<prompt>/object_<global_id>_mask.mp4`
9. `_extract_last_frame_masks()` stores the chunk's final-frame masks into `carry[prompt]`.
10. `driver.extract_memory_bank()` stores tracker memory frames into `memory_banks[prompt]`.

At this point, the next chunk has two ways to continue the tracked objects:

- `carry`: hard mask conditioning on frame 0 of the next chunk
- `memory_banks`: tracker memory features restored at negative frame indices

## What Happens When A New Chunk Starts

For each later chunk, the session starts fresh, but the tracking state is rebuilt from saved cross-chunk data.

### Step 1: Start A Fresh Session

Each chunk gets a new `session_id` from `driver.start_session()`.

This means the tracker does **not** automatically remember the previous chunk. Cross-chunk continuity only happens because we explicitly restore it.

### Step 2: Reset Prompt State

`driver.reset_session(session_id)` clears any prompt-specific state for the current prompt before we rebuild it.

### Step 3: Inject Carry Masks

If `carry[prompt]` exists, we call:

`driver.inject_masks(session_id, frame_idx=0, masks=carry[prompt], object_ids=sorted_ids)`

This creates tracked objects on frame 0 of the new chunk using the previous chunk's final masks.

Effect:

- objects start the new chunk with their previous global IDs
- the tracker gets a hard visual anchor at the boundary

### Step 4: Restore Memory Bank

If a memory bank exists for the same prompt, we call:

`driver.restore_memory_bank(session_id, memory_banks[prompt])`

This restores SAM3's internal memory features so the new chunk starts with short-term tracking context instead of an empty tracker state.

Important ordering:

- `inject_masks()` must happen first
- `restore_memory_bank()` happens second

That order matters because the memory bank is restored into the tracker state created by mask injection.

### Step 5: Decide Whether To Re-Detect

Current behavior for text prompts:

- if carry injection succeeded:
  skip `add_prompt()`
- if no carry exists:
  run `driver.add_prompt(session_id, prompt)`

Why:

- `inject_masks()` continues already-known objects
- `add_prompt()` performs broad text detection and can create fresh objects

If we re-run text detection after successful carry injection for a broad prompt like `"player"`, the model can rediscover extra people or transient detections in every chunk. The current logic avoids that by propagating the injected objects directly.

## Propagation Inside The Chunk

After setup, `_propagate_with_monitoring()` calls:

`driver.propagate_in_video_streaming(session_id, propagation_direction="both")`

During propagation it collects:

- `result`
  Per-frame outputs
- `object_ids`
  Object IDs seen in this chunk
- `frame_objects`
  Which objects appeared on which frames

It also watches RAM/VRAM pressure through `IntraChunkMonitor`.

## ID Continuity Across Chunks

Even with carry injection, the code still runs `_match_and_remap()` after propagation.

What it does:

1. Takes the first available frame from the new chunk result.
2. Compares each new mask against `prev_masks` from `carry[prompt]`.
3. Computes pairwise IoU scores.
4. Greedily matches new objects to previous global IDs.
5. Assigns new global IDs to anything unmatched.
6. Rewrites every frame in the chunk result to use global IDs.

This is the safety net that keeps object numbering stable even if the tracker changes local IDs internally.

## What Data Is Passed To The Next Chunk

At the end of each chunk, the following data is carried forward:

- last-frame masks
  From `_extract_last_frame_masks()`
  Stored in `carry[prompt]`

- tracker memory frames
  From `driver.extract_memory_bank(..., max_frames=MEMORY_BANK_MAX_FRAMES)`
  Stored in `memory_banks[prompt]`

- next global ID counter
  Stored in `global_next_ids[prompt]`

- cumulative object set
  Stored in `all_object_ids[prompt]`

So the next chunk gets:

- object identities
- object shapes at the boundary
- short-term tracker memory

## What Happens If A Chunk Is Replanned

If a chunk hits OOM or an early memory stop:

1. Partial chunk results may be saved.
2. `AdaptiveChunkManager.replan_remaining()` creates a new chunk plan for the remaining frames.
3. The new chunks keep globally increasing chunk IDs.

This chunk ID continuity matters because chunk outputs are saved under `chunk_<id>` directories and later stitched by those IDs.

## Where Chunk Outputs Are Stored

Temporary chunk outputs go under:

`.sam3_temp/<video_name>/chunks/chunk_<id>/`

Typical contents:

- `chunk_<id>.mp4`
  The extracted chunk video

- `masks/<prompt>/object_<global_id>_mask.mp4`
  Per-object mask video for that chunk

Final outputs go under:

`results/<video_name>/`

Typical contents:

- `masks/<prompt>/object_<global_id>_mask.mp4`
  Final stitched per-object mask videos

- `overlay_<prompt>.mp4`
  Final overlay video built from the stitched masks

## How Stitching Works

After all chunks finish:

1. async mask writes are drained
2. `stitch_chunk_mask_videos()` reads each chunk's per-object mask MP4s
3. overlap frames are skipped for later chunks
4. one final stitched mask video is created per global object ID
5. `create_overlay_from_masks()` composites those stitched mask videos over the original video

Before stitching, the existing final mask directory for that prompt is cleared so reruns do not mix stale object videos with the current run.

## The Text-Prompt Chunk Handoff In One Sequence

```text
chunk N ends
  -> extract last-frame masks
  -> extract memory bank
  -> store next global ID

chunk N+1 starts
  -> start fresh session
  -> reset session
  -> inject last-frame masks on frame 0
  -> restore memory bank
  -> skip text re-detection if carry injection succeeded
  -> propagate tracked objects
  -> IoU-remap to global IDs
  -> save masks
  -> extract carry state for chunk N+2
```

## Important Design Idea

Cross-chunk continuity in this pipeline is not one single mechanism. It is the combination of:

- carry-mask injection
- memory-bank restoration
- post-propagation IoU remapping

Each piece solves a different part of the boundary problem:

- carry masks anchor object shape/location
- memory bank restores short-term tracker context
- IoU remapping preserves stable global IDs

## Relevant Code Pointers

- `video_prompter.py::_process_video()`
  Main orchestration
- `video_prompter.py::_extract_last_frame_masks()`
  Builds carry masks from the chunk result
- `video_prompter.py::_match_and_remap()`
  Matches chunk results back to global IDs
- `sam3/drivers.py::inject_masks()`
  Recreates tracked objects at the next chunk boundary
- `sam3/drivers.py::extract_memory_bank()`
  Saves tracker memory frames
- `sam3/drivers.py::restore_memory_bank()`
  Restores tracker memory into the new chunk session
- `sam3/memory_optimizer.py::replan_remaining()`
  Rebuilds the remaining chunk plan after shrink events
- `sam3/streaming_masks.py::stitch_chunk_mask_videos()`
  Stitches chunk mask MP4s
- `sam3/streaming_masks.py::create_overlay_from_masks()`
  Builds the final overlay video
