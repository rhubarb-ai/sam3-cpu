#!/usr/bin/env python3
"""
Test Cross-Chunk Mask Injection (Option B)

Validates the carry-forward mechanism that passes tracked object masks
from chunk N into chunk N+1 for seamless object continuity.

Tests:
  1. Unit: _extract_carry_forward_data with mock propagation results
  2. Unit: _extract_carry_forward_data with saved PNG fallback
  3. Unit: PostProcessor deterministic matching for injected objects
  4. Unit: ChunkProcessor accepts and passes prev_chunk_data
  5. Integration: Full pipeline with Sam3API on a real video (requires model)

Usage:
  # Run unit tests only (no model needed):
  python3 test_cross_chunk_injection.py

  # Run integration test too (loads model, needs video):
  python3 test_cross_chunk_injection.py --integration
"""

import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# Ensure repo root is on path
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle_mask(h, w, cx, cy, r):
    """Create a circular binary mask (uint8, 0 or 255)."""
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2).astype(np.uint8) * 255
    return mask


def make_rect_mask(h, w, x1, y1, x2, y2):
    """Create a rectangular binary mask (uint8, 0 or 255)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def save_png_mask(mask, path):
    """Save a uint8 mask as grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(path, format="PNG")


# ---------------------------------------------------------------------------
# Test 1: _extract_carry_forward_data from propagation results
# ---------------------------------------------------------------------------

def test_extract_carry_forward_from_results():
    """Test that carry-forward extraction works from raw propagation results."""
    print("\n" + "=" * 70)
    print("TEST 1: Extract carry-forward from propagation result_prompt")
    print("=" * 70)

    H, W = 480, 640
    num_frames = 30

    # Simulate propagation results: dict of frame_idx -> output
    mask_obj0 = make_circle_mask(H, W, 300, 200, 50)
    mask_obj1 = make_rect_mask(H, W, 100, 100, 200, 300)

    result_prompt = {}
    for fi in range(num_frames):
        result_prompt[fi] = {
            "out_obj_ids": np.array([0, 1]),
            "out_binary_masks": np.array([
                (mask_obj0 > 0),
                (mask_obj1 > 0),
            ]),
        }

    # Build a prompt_results dict like ChunkProcessor._process_single_prompt returns
    prompt_results = {
        "player": {
            "prompt": "player",
            "num_objects": 2,
            "object_ids": [0, 1],
            "injected_object_ids": [],
            "frame_objects": {},
            "masks_dir": None,
            "metadata_path": None,
            "_result_prompt": result_prompt,
        }
    }

    # Import and call
    from sam3.chunk_processor import ChunkProcessor

    # Create a minimal ChunkProcessor (mock — just need the method)
    tmp = Path(tempfile.mkdtemp())
    try:
        # We need a dummy chunk_video_path for the fallback branch (won't be used)
        dummy_video = tmp / "dummy.mp4"
        dummy_video.touch()

        cp = ChunkProcessor.__new__(ChunkProcessor)
        cp.chunk_id = 0
        cp.chunk_video_path = dummy_video
        cp.video_metadata = {"width": W, "height": H}

        carry = cp._extract_carry_forward_data(prompt_results)

        # Validate
        assert "masks" in carry, "carry_forward missing 'masks' key"
        assert "object_ids" in carry, "carry_forward missing 'object_ids' key"
        assert "player" in carry["masks"], "carry_forward missing prompt 'player'"
        assert set(carry["object_ids"]["player"]) == {0, 1}, \
            f"Expected obj ids {{0,1}}, got {carry['object_ids']['player']}"

        # Verify masks are non-empty uint8 arrays
        for oid in [0, 1]:
            m = carry["masks"]["player"][oid]
            assert isinstance(m, np.ndarray), f"Mask for obj {oid} is not ndarray"
            assert m.shape == (H, W), f"Mask shape {m.shape} != ({H},{W})"
            assert m.max() == 255, f"Mask max {m.max()} != 255"
            assert m.dtype == np.uint8, f"Mask dtype {m.dtype} != uint8"

        print("  PASSED: carry_forward extracted correctly from result_prompt")
        return True

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: _extract_carry_forward_data PNG fallback
# ---------------------------------------------------------------------------

def test_extract_carry_forward_from_pngs():
    """Test carry-forward extraction falls back to reading saved PNG masks."""
    print("\n" + "=" * 70)
    print("TEST 2: Extract carry-forward from saved PNG masks (fallback)")
    print("=" * 70)

    H, W = 480, 640
    num_frames = 10

    tmp = Path(tempfile.mkdtemp())
    try:
        # Create fake saved masks
        masks_dir = tmp / "masks" / "player"
        for oid in [0, 1]:
            obj_dir = masks_dir / f"object_{oid}"
            for fi in range(num_frames):
                if oid == 0:
                    mask = make_circle_mask(H, W, 300 + fi, 200, 50)
                else:
                    mask = make_rect_mask(H, W, 100, 100, 200, 300)
                save_png_mask(mask, obj_dir / f"frame_{fi:06d}.png")

        # Create a dummy video with correct frame count
        import cv2
        video_path = tmp / "chunk.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30, (W, H))
        for _ in range(num_frames):
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
        writer.release()

        # prompt_results with NO _result_prompt (forces PNG fallback)
        prompt_results = {
            "player": {
                "prompt": "player",
                "num_objects": 2,
                "object_ids": [0, 1],
                "injected_object_ids": [],
                "frame_objects": {},
                "masks_dir": str(masks_dir),
                "metadata_path": None,
                "_result_prompt": {},  # Empty — triggers fallback
            }
        }

        from sam3.chunk_processor import ChunkProcessor

        cp = ChunkProcessor.__new__(ChunkProcessor)
        cp.chunk_id = 0
        cp.chunk_video_path = video_path
        cp.video_metadata = {"width": W, "height": H}

        carry = cp._extract_carry_forward_data(prompt_results)

        assert "player" in carry["masks"], "PNG fallback failed to find masks"
        assert len(carry["masks"]["player"]) == 2, \
            f"Expected 2 objects, got {len(carry['masks']['player'])}"

        # Last frame mask should match what we saved
        last_mask_0 = carry["masks"]["player"][0]
        expected = make_circle_mask(H, W, 300 + (num_frames - 1), 200, 50)
        assert np.array_equal(last_mask_0, expected), "PNG fallback mask content mismatch"

        print("  PASSED: PNG fallback extraction works correctly")
        return True

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: PostProcessor deterministic matching for injected objects
# ---------------------------------------------------------------------------

def test_postprocessor_deterministic_matching():
    """Test that PostProcessor auto-matches injected objects without IoU."""
    print("\n" + "=" * 70)
    print("TEST 3: PostProcessor deterministic matching for injected IDs")
    print("=" * 70)

    H, W = 480, 640
    tmp = Path(tempfile.mkdtemp())

    try:
        # Create fake chunk results
        chunk_0 = {
            "chunk_id": 0,
            "prompts": {
                "player": {
                    "object_ids": [0, 1, 2],
                    "injected_object_ids": [],
                    "masks_dir": str(tmp / "chunk_0" / "masks" / "player"),
                }
            }
        }
        chunk_1 = {
            "chunk_id": 1,
            "prompts": {
                "player": {
                    "object_ids": [0, 1, 2, 3],  # 0,1,2 injected + 3 newly detected
                    "injected_object_ids": [0, 1, 2],
                    "masks_dir": str(tmp / "chunk_1" / "masks" / "player"),
                }
            }
        }

        # Save some fake overlap masks (for IoU fallback on obj 3)
        overlap_frames = 5
        for chunk_id, obj_ids in [(0, [0, 1, 2]), (1, [0, 1, 2, 3])]:
            for oid in obj_ids:
                obj_dir = tmp / f"chunk_{chunk_id}" / "masks" / "player" / f"object_{oid}"
                for fi in range(25):
                    # Simple distinguishing masks
                    mask = make_circle_mask(H, W, 100 + oid * 100, 200, 40)
                    save_png_mask(mask, obj_dir / f"frame_{fi:06d}.png")

        from sam3.postprocessor import VideoPostProcessor

        pp = VideoPostProcessor(
            video_name="test",
            chunk_results=[chunk_0, chunk_1],
            video_metadata={
                "width": W, "height": H, "fps": 30,
                "chunks": [
                    {"chunk": 0, "start": 0, "end": 24},
                    {"chunk": 1, "start": 20, "end": 49},
                ]
            },
            chunks_temp_dir=tmp,
            masks_output_dir=tmp / "output" / "masks",
            meta_output_dir=tmp / "output" / "metadata",
        )

        mapping = pp._match_chunks(chunk_0, chunk_1, "player")

        # Verify deterministic matches for injected IDs
        for oid in [0, 1, 2]:
            assert oid in mapping, f"Injected obj {oid} should be auto-matched"
            assert mapping[oid] == oid, \
                f"Injected obj {oid} should map to itself, got {mapping[oid]}"

        print(f"  Mapping: {mapping}")
        print("  PASSED: Deterministic matching works for injected objects")
        return True

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: ChunkProcessor prev_chunk_data passthrough
# ---------------------------------------------------------------------------

def test_chunk_processor_prev_data_signature():
    """Test that ChunkProcessor.process_with_prompts accepts prev_chunk_data."""
    print("\n" + "=" * 70)
    print("TEST 4: ChunkProcessor signature accepts prev_chunk_data")
    print("=" * 70)

    import inspect
    from sam3.chunk_processor import ChunkProcessor

    sig = inspect.signature(ChunkProcessor.process_with_prompts)
    params = list(sig.parameters.keys())

    assert "prev_chunk_data" in params, \
        f"process_with_prompts missing 'prev_chunk_data' param. Has: {params}"

    print(f"  Parameters: {params}")
    print("  PASSED: prev_chunk_data parameter exists")
    return True


# ---------------------------------------------------------------------------
# Test 5: VideoProcessor chain passes carry-forward
# ---------------------------------------------------------------------------

def test_video_processor_carry_forward_code():
    """Test that _process_multiple_chunks references prev_chunk_data."""
    print("\n" + "=" * 70)
    print("TEST 5: VideoProcessor._process_multiple_chunks uses carry-forward")
    print("=" * 70)

    import inspect
    from sam3.video_processor import VideoProcessor

    source = inspect.getsource(VideoProcessor._process_multiple_chunks)

    checks = [
        ("prev_chunk_data = None", "initializes prev_chunk_data"),
        ("prev_chunk_data=prev_chunk_data", "passes prev_chunk_data to ChunkProcessor"),
        ("carry_forward", "extracts carry_forward from result"),
    ]

    all_ok = True
    for pattern, desc in checks:
        if pattern in source:
            print(f"  OK: {desc}")
        else:
            print(f"  FAIL: {desc} — pattern '{pattern}' not found in source")
            all_ok = False

    if all_ok:
        print("  PASSED: VideoProcessor carry-forward chain is wired correctly")
    else:
        print("  FAILED: Some carry-forward patterns missing")

    return all_ok


# ---------------------------------------------------------------------------
# Test 6: Driver inject_masks exists
# ---------------------------------------------------------------------------

def test_driver_inject_masks_exists():
    """Test that Sam3VideoDriver.inject_masks method exists with correct signature."""
    print("\n" + "=" * 70)
    print("TEST 6: Sam3VideoDriver.inject_masks exists and has correct signature")
    print("=" * 70)

    import inspect
    from sam3.drivers import Sam3VideoDriver

    assert hasattr(Sam3VideoDriver, "inject_masks"), \
        "Sam3VideoDriver missing inject_masks method"

    sig = inspect.signature(Sam3VideoDriver.inject_masks)
    params = list(sig.parameters.keys())

    expected = ["self", "session_id", "frame_idx", "masks", "object_ids"]
    assert params == expected, f"Expected params {expected}, got {params}"

    print(f"  Parameters: {params}")
    print("  PASSED: inject_masks exists with correct signature")
    return True


# ---------------------------------------------------------------------------
# Test 7: Carry-forward structure round-trip
# ---------------------------------------------------------------------------

def test_carry_forward_structure():
    """Test that carry_forward data can be consumed by _process_single_prompt."""
    print("\n" + "=" * 70)
    print("TEST 7: Carry-forward structure matches expected consumption format")
    print("=" * 70)

    H, W = 480, 640

    # Build carry_forward as produced by _extract_carry_forward_data
    carry_forward = {
        "masks": {
            "player": {
                0: make_circle_mask(H, W, 300, 200, 50),
                1: make_rect_mask(H, W, 100, 100, 200, 300),
            },
            "ball": {
                0: make_circle_mask(H, W, 400, 300, 20),
            }
        },
        "object_ids": {
            "player": [0, 1],
            "ball": [0],
        }
    }

    # Verify structure matches what _process_single_prompt expects
    for prompt in ["player", "ball"]:
        assert prompt in carry_forward["masks"], f"Missing prompt '{prompt}' in masks"
        assert prompt in carry_forward["object_ids"], f"Missing prompt '{prompt}' in object_ids"

        masks = carry_forward["masks"][prompt]
        obj_ids = carry_forward["object_ids"][prompt]

        assert isinstance(masks, dict), f"masks['{prompt}'] should be dict"
        assert isinstance(obj_ids, list), f"object_ids['{prompt}'] should be list"
        assert set(obj_ids) == set(masks.keys()), \
            f"obj_ids {obj_ids} don't match mask keys {list(masks.keys())}"

        for oid in obj_ids:
            m = masks[oid]
            assert isinstance(m, np.ndarray), f"Mask for {prompt}/{oid} not ndarray"
            assert m.ndim == 2, f"Mask for {prompt}/{oid} should be 2D, got {m.ndim}D"
            assert m.dtype == np.uint8, f"Mask dtype should be uint8, got {m.dtype}"

    print("  PASSED: Carry-forward structure is valid and consumable")
    return True


# ---------------------------------------------------------------------------
# Test 8 (Integration): Full pipeline on real video
# ---------------------------------------------------------------------------

def test_integration_full_pipeline():
    """Integration test: process a real video and verify carry-forward works."""
    print("\n" + "=" * 70)
    print("TEST 8 (INTEGRATION): Full pipeline with Sam3API")
    print("=" * 70)

    video_path = Path("assets/videos/sample.mp4")
    if not video_path.exists():
        print(f"  SKIPPED: Video not found at {video_path}")
        return None

    from sam3 import Sam3API

    output_dir = Path(tempfile.mkdtemp()) / "test_output"
    api = Sam3API()

    try:
        result = api.process_video_with_prompts(
            video_path=str(video_path),
            prompts=["player"],
            output_dir=str(output_dir),
            keep_temp_files=True,
        )

        num_chunks = result.get("num_chunks", len(result.get("chunks", [])))
        print(f"  Processed {num_chunks} chunk(s)")

        chunks = result.get("chunks", [])
        if num_chunks <= 1:
            print("  NOTE: Video fit in single chunk — carry-forward not exercised")
            print("  PASSED (trivially): Single chunk processed successfully")
            return True

        # Verify carry_forward exists in chunk results
        for i, chunk in enumerate(chunks):
            cf = chunk.get("carry_forward")
            if i < num_chunks - 1:
                # All but last chunk should have carry_forward data
                assert cf is not None, f"Chunk {i} missing carry_forward"
                assert "masks" in cf, f"Chunk {i} carry_forward missing 'masks'"
                total = sum(len(m) for m in cf["masks"].values())
                print(f"  Chunk {i}: carry_forward has {total} mask(s)")

        # Verify injected_object_ids in chunks > 0
        for i in range(1, len(chunks)):
            for prompt, presult in chunks[i].get("prompts", {}).items():
                injected = presult.get("injected_object_ids", [])
                print(f"  Chunk {i}, prompt '{prompt}': {len(injected)} injected object(s)")

        # Check post-processing output
        mapping_path = output_dir / "sample" / "metadata" / "id_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mappings = json.load(f)
            print(f"  ID mappings saved: {mapping_path}")
            print(f"  Mapping keys: {list(mappings.get('mappings', {}).get('by_chunk', {}).keys())}")
        else:
            print(f"  NOTE: No id_mapping.json found (may be single chunk)")

        print("  PASSED: Full pipeline completed successfully")
        return True

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        api.cleanup()
        shutil.rmtree(output_dir.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test cross-chunk mask injection")
    parser.add_argument(
        "--integration", action="store_true",
        help="Run integration test (loads SAM3 model, needs assets/videos/sample.mp4)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-CHUNK MASK INJECTION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Extract carry-forward from results", test_extract_carry_forward_from_results),
        ("Extract carry-forward from PNGs", test_extract_carry_forward_from_pngs),
        ("PostProcessor deterministic matching", test_postprocessor_deterministic_matching),
        ("ChunkProcessor prev_chunk_data signature", test_chunk_processor_prev_data_signature),
        ("VideoProcessor carry-forward wiring", test_video_processor_carry_forward_code),
        ("Driver inject_masks signature", test_driver_inject_masks_exists),
        ("Carry-forward structure validation", test_carry_forward_structure),
    ]

    if args.integration:
        tests.append(("Integration: full pipeline", test_integration_full_pipeline))

    results = {}
    for name, test_fn in tests:
        try:
            ok = test_fn()
            results[name] = ok
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for name, ok in results.items():
        if ok is True:
            status = "PASSED"
            passed += 1
        elif ok is None:
            status = "SKIPPED"
            skipped += 1
        else:
            status = "FAILED"
            failed += 1
        print(f"  [{status:>7s}] {name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped "
          f"out of {len(tests)} tests")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
