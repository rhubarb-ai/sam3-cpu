#!/usr/bin/env python3
"""
Consolidated Cross-Chunk Tests

Tests for cross-chunk mask injection, carry-forward extraction,
chunk overlap validation, and post-processor stitching.

Consolidated from:
  - test_cross_chunk_injection.py
  - test_chunks_injection.py
  - test_chain_tracking.py
  - test_chunk_overlap.py
  - test_postprocessor_isolated.py
  - test_lossless.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mask(h: int = 64, w: int = 64, fill: int = 255) -> np.ndarray:
    """Create a simple uint8 mask."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:50, 10:50] = fill
    return m


# ---------------------------------------------------------------------------
# Tests for carry-forward data extraction
# ---------------------------------------------------------------------------

class TestCarryForwardExtraction:
    """Validates that ChunkProcessor._extract_carry_forward_data works correctly."""

    def test_extracts_from_propagation_result(self):
        """Strategy 1: extract directly from in-memory propagation output."""
        from sam3.chunk_processor import ChunkProcessor

        # Build a minimal prompt_results dict with _result_prompt
        mask_a = _make_mask()
        mask_b = _make_mask(fill=128)
        prompt_results = {
            "player": {
                "object_ids": [0, 1],
                "masks_dir": None,
                "_result_prompt": {
                    0: {
                        "out_obj_ids": np.array([0, 1]),
                        "out_binary_masks": [
                            (mask_a > 0).astype(bool),
                            (mask_b > 0).astype(bool),
                        ],
                    },
                    9: {  # last frame
                        "out_obj_ids": np.array([0, 1]),
                        "out_binary_masks": [
                            (mask_a > 0).astype(bool),
                            (mask_b > 0).astype(bool),
                        ],
                    },
                },
            }
        }

        # We need a minimal ChunkProcessor to call the method
        # Just create a bare instance with enough state
        cp = ChunkProcessor.__new__(ChunkProcessor)
        cp.chunk_id = 0
        cp.chunk_video_path = Path("/dev/null")

        carry = cp._extract_carry_forward_data(prompt_results)

        assert "masks" in carry
        assert "player" in carry["masks"]
        assert len(carry["masks"]["player"]) == 2
        assert carry["masks"]["player"][0].shape == (64, 64)

    def test_empty_prompt_results(self):
        """No prompts → empty carry-forward."""
        from sam3.chunk_processor import ChunkProcessor

        cp = ChunkProcessor.__new__(ChunkProcessor)
        cp.chunk_id = 0
        cp.chunk_video_path = Path("/dev/null")

        carry = cp._extract_carry_forward_data({})
        assert carry["masks"] == {}


# ---------------------------------------------------------------------------
# Tests for PostProcessor deterministic matching
# ---------------------------------------------------------------------------

class TestPostProcessorMatching:
    """Test PostProcessor IoU-based matching logic."""

    def test_postprocessor_initializes(self):
        """PostProcessor can be instantiated with minimal data."""
        from sam3.postprocessor import VideoPostProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pp = VideoPostProcessor(
                video_name="test",
                chunk_results=[{"chunk_id": 0, "prompts": {}}],
                video_metadata={"chunks": []},
                chunks_temp_dir=tmpdir / "chunks",
                masks_output_dir=tmpdir / "masks",
                meta_output_dir=tmpdir / "meta",
            )
            assert pp.video_name == "test"

    def test_infer_overlap_single_chunk(self):
        """Single chunk → default overlap."""
        from sam3.postprocessor import VideoPostProcessor
        from sam3.__globals import DEFAULT_MIN_CHUNK_OVERLAP

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pp = VideoPostProcessor(
                video_name="test",
                chunk_results=[{"chunk_id": 0, "prompts": {}}],
                video_metadata={"chunks": [{"chunk": 0, "start": 0, "end": 24}]},
                chunks_temp_dir=tmpdir / "chunks",
                masks_output_dir=tmpdir / "masks",
                meta_output_dir=tmpdir / "meta",
            )
            assert pp.overlap_frames == DEFAULT_MIN_CHUNK_OVERLAP

    def test_infer_overlap_two_chunks(self):
        """Two overlapping chunks → correct overlap computation."""
        from sam3.postprocessor import VideoPostProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pp = VideoPostProcessor(
                video_name="test",
                chunk_results=[
                    {"chunk_id": 0, "prompts": {}},
                    {"chunk_id": 1, "prompts": {}},
                ],
                video_metadata={
                    "chunks": [
                        {"chunk": 0, "start": 0, "end": 24},
                        {"chunk": 1, "start": 20, "end": 49},
                    ]
                },
                chunks_temp_dir=tmpdir / "chunks",
                masks_output_dir=tmpdir / "masks",
                meta_output_dir=tmpdir / "meta",
            )
            # overlap = 24 - 20 + 1 = 5
            assert pp.overlap_frames == 5


# ---------------------------------------------------------------------------
# Tests for ChunkProcessor signature / prev_chunk_data acceptance
# ---------------------------------------------------------------------------

class TestChunkProcessorSignature:
    """Ensure ChunkProcessor accepts prev_chunk_data parameter."""

    def test_process_with_prompts_signature(self):
        """process_with_prompts should accept prev_chunk_data kwarg."""
        import inspect
        from sam3.chunk_processor import ChunkProcessor

        sig = inspect.signature(ChunkProcessor.process_with_prompts)
        assert "prev_chunk_data" in sig.parameters


# ---------------------------------------------------------------------------
# Tests for ID remapping consistency
# ---------------------------------------------------------------------------

class TestIDRemapping:
    """Test that _match_and_remap_ids produces consistent global IDs."""

    def test_first_chunk_identity(self):
        """First chunk (no prev_chunk_data) → identity mapping starting at 0."""
        from sam3.chunk_processor import ChunkProcessor

        cp = ChunkProcessor.__new__(ChunkProcessor)
        result_prompt = {
            0: {
                "out_obj_ids": np.array([0, 1, 2]),
                "out_binary_masks": [_make_mask(), _make_mask(), _make_mask()],
            }
        }
        remapped, new_ids, mapping, next_id = cp._match_and_remap_ids(
            result_prompt,
            object_ids={0, 1, 2},
            prev_masks={},
            global_next_id=0,
        )
        assert mapping == {0: 0, 1: 1, 2: 2}
        assert next_id == 3

    def test_second_chunk_matches(self):
        """Second chunk where objects match previous → same IDs preserved."""
        from sam3.chunk_processor import ChunkProcessor

        cp = ChunkProcessor.__new__(ChunkProcessor)
        # Use two distinct masks so matching is unambiguous
        mask_a = np.zeros((64, 64), dtype=np.uint8)
        mask_a[5:30, 5:30] = 255
        mask_b = np.zeros((64, 64), dtype=np.uint8)
        mask_b[35:60, 35:60] = 255

        prev_masks = {0: mask_a, 1: mask_b}

        # out_binary_masks come from the model as boolean arrays
        result_prompt = {
            0: {
                "out_obj_ids": np.array([0, 1]),
                "out_binary_masks": [(mask_a > 0), (mask_b > 0)],
            }
        }
        _, _, mapping, _ = cp._match_and_remap_ids(
            result_prompt,
            object_ids={0, 1},
            prev_masks=prev_masks,
            global_next_id=2,
        )
        # Both should match previous (IoU = 1.0), mapped to prev IDs
        assert len(mapping) == 2
        assert mapping[0] == 0
        assert mapping[1] == 1

    def test_new_object_gets_new_id(self):
        """Object in new chunk not matching any prev → gets next global ID."""
        from sam3.chunk_processor import ChunkProcessor

        cp = ChunkProcessor.__new__(ChunkProcessor)
        prev_mask = _make_mask()
        new_mask = np.zeros((64, 64), dtype=np.uint8)
        new_mask[55:64, 55:64] = 255  # completely different location

        prev_masks = {0: prev_mask}
        result_prompt = {
            0: {
                "out_obj_ids": np.array([0]),
                "out_binary_masks": [new_mask],
            }
        }
        _, _, mapping, next_id = cp._match_and_remap_ids(
            result_prompt,
            object_ids={0},
            prev_masks=prev_masks,
            global_next_id=5,
        )
        # No IoU match → gets global_next_id = 5
        assert mapping[0] == 5
        assert next_id == 6


# ---------------------------------------------------------------------------
# Tests for memory manager chunk generation
# ---------------------------------------------------------------------------

class TestChunkGeneration:
    """Test MemoryManager.generate_chunks for correctness."""

    def test_single_chunk(self):
        """Video shorter than chunk_size → single chunk."""
        from sam3.memory_manager import MemoryManager

        mm = MemoryManager()
        chunks = mm.generate_chunks(total_frames=20, chunk_size=100, overlap=5)
        assert len(chunks) == 1
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 19

    def test_multiple_chunks_overlap(self):
        """Multiple chunks with correct overlap boundaries."""
        from sam3.memory_manager import MemoryManager

        mm = MemoryManager()
        chunks = mm.generate_chunks(total_frames=50, chunk_size=25, overlap=5)
        assert len(chunks) >= 2
        # Second chunk should start at 25 - 5 = 20
        assert chunks[1]["start"] == 20

    def test_even_spread(self):
        """Even spread mode adjusts chunk sizes."""
        from sam3.memory_manager import MemoryManager

        mm = MemoryManager()
        chunks = mm.generate_chunks(
            total_frames=100, chunk_size=30, overlap=5, chunk_spread="even"
        )
        assert len(chunks) >= 2
        # All chunks should cover the full video
        assert chunks[0]["start"] == 0
        assert chunks[-1]["end"] == 99
