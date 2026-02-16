#!/usr/bin/env python3
"""Test video processing with fixed API"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sam3.model_builder import build_sam3_video_predictor_cpu
from sam3.__globals import BPE_PATH

# Build predictor
print("Building predictor...")
predictor = build_sam3_video_predictor_cpu(bpe_path=BPE_PATH, num_workers=1)
print("✓ Predictor built")

# Start session
video_path = "assets/videos/bedroom.mp4"
print(f"\nStarting session with {video_path}...")
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]
print(f"✓ Session started: {session_id}")

# Add prompt with frame_index and text parameters (matching working code)
prompt_text = "kids"
frame_idx = 0
print(f"\nAdding prompt '{prompt_text}' on frame {frame_idx}...")
response = predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_idx,
        text=prompt_text,
    )
)
out = response["outputs"]
print(f"✓ Prompt added. Found {len(out['out_obj_ids'])} object(s)")
print(f"  Object IDs: {out['out_obj_ids']}")

# Propagate
print(f"\nPropagating through video...")
outputs_per_frame = {}
for response in predictor.handle_stream_request(
    request=dict(
        type="propagate_in_video",
        session_id=session_id,
    )
):
    outputs_per_frame[response["frame_index"]] = response["outputs"]

print(f"✓ Propagation complete. Processed {len(outputs_per_frame)} frames")

# Close session
print(f"\nClosing session...")
predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
print("✓ Session closed")

print(f"\n✅ SUCCESS! Video processing working correctly.")
print(f"   - Detected objects: {len(out['out_obj_ids'])}")
print(f"   - Tracked across: {len(outputs_per_frame)} frames")
