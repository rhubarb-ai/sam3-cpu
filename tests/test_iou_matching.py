#!/usr/bin/env python3
"""
Consolidated IoU Matching Tests

Tests for IoU computation, cross-object matching, and cross-chunk
object continuity via IoU-based ID remapping.

Consolidated from:
  - test_iou.py
  - test_lossless_iou.py
  - test_chunk_iou_matching.py
  - test_cross_object_iou.py
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# IoU computation helper (standalone — no model needed)
# ---------------------------------------------------------------------------

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks."""
    m1 = (mask1 > 0).astype(bool)
    m2 = (mask2 > 0).astype(bool)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestIoUComputation:
    """Test basic IoU computation logic."""

    def test_identical_masks(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        assert compute_iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_a[:50, :] = 1
        mask_b[50:, :] = 1
        assert compute_iou(mask_a, mask_b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_a[:60, :] = 1   # 60 rows
        mask_b[40:, :] = 1   # 60 rows, overlap = rows 40-59 = 20 rows
        # intersection = 20*100 = 2000, union = 100*100 = 10000
        assert compute_iou(mask_a, mask_b) == pytest.approx(2000 / 10000)

    def test_both_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert compute_iou(mask, mask) == pytest.approx(0.0)

    def test_uint8_threshold(self):
        """Values > 0 should be treated as foreground."""
        mask_a = np.full((50, 50), 128, dtype=np.uint8)
        mask_b = np.full((50, 50), 255, dtype=np.uint8)
        assert compute_iou(mask_a, mask_b) == pytest.approx(1.0)


class TestGreedyIoUMatching:
    """Test the greedy IoU-based ID matching algorithm used by ChunkProcessor."""

    @staticmethod
    def _greedy_match(
        current_masks: dict,
        prev_masks: dict,
        iou_threshold: float = 0.25,
    ) -> dict:
        """Standalone version of the greedy matching logic from ChunkProcessor."""
        pairs = []
        for new_id, new_mask in current_masks.items():
            for prev_id, prev_mask in prev_masks.items():
                iou = compute_iou(new_mask, prev_mask)
                if iou >= iou_threshold:
                    pairs.append((iou, new_id, prev_id))
        pairs.sort(reverse=True)

        mapping = {}
        used = set()
        for iou, new_id, prev_id in pairs:
            if new_id in mapping or prev_id in used:
                continue
            mapping[new_id] = prev_id
            used.add(prev_id)
        return mapping

    def test_perfect_match(self):
        """Identical masks → 1:1 match."""
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        prev = {0: mask, 1: mask.copy()}
        curr = {0: mask, 1: mask.copy()}
        mapping = self._greedy_match(curr, prev)
        # Each current ID maps to same prev ID (both identical)
        assert len(mapping) == 2

    def test_no_match_below_threshold(self):
        """Non-overlapping masks → no match."""
        m1 = np.zeros((64, 64), dtype=np.uint8)
        m2 = np.zeros((64, 64), dtype=np.uint8)
        m1[:10, :10] = 255
        m2[50:, 50:] = 255
        prev = {0: m1}
        curr = {0: m2}
        mapping = self._greedy_match(curr, prev)
        assert len(mapping) == 0

    def test_best_pair_wins(self):
        """When multiple candidates exist, highest IoU wins."""
        base = np.zeros((100, 100), dtype=np.uint8)
        m_prev = base.copy(); m_prev[10:60, 10:60] = 255  # 50×50
        m_good = base.copy(); m_good[10:60, 10:60] = 255  # identical
        m_poor = base.copy(); m_poor[40:90, 40:90] = 255  # some overlap
        prev = {0: m_prev}
        curr = {0: m_good, 1: m_poor}
        mapping = self._greedy_match(curr, prev)
        assert mapping.get(0) == 0  # perfect match wins

    def test_one_to_one(self):
        """Each prev ID matched at most once (greedy constraint)."""
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        prev = {0: mask}
        curr = {0: mask.copy(), 1: mask.copy()}
        mapping = self._greedy_match(curr, prev)
        assert len(mapping) == 1  # only one curr can claim prev-0


class TestChunkProcessorIoU:
    """Test ChunkProcessor._compute_iou and _apply_id_mapping."""

    def test_compute_iou_import(self):
        """ChunkProcessor._compute_iou should be importable."""
        from sam3.chunk_processor import ChunkProcessor
        m = np.ones((32, 32), dtype=np.uint8) * 255
        assert ChunkProcessor._compute_iou(m, m) == pytest.approx(1.0)

    def test_apply_id_mapping(self):
        """_apply_id_mapping should remap out_obj_ids in frame outputs."""
        from sam3.chunk_processor import ChunkProcessor

        result_prompt = {
            0: {
                "out_obj_ids": np.array([0, 1]),
                "out_binary_masks": [np.zeros((4, 4)), np.ones((4, 4))],
            }
        }
        mapping = {0: 10, 1: 20}
        remapped = ChunkProcessor._apply_id_mapping(result_prompt, mapping)
        assert list(remapped[0]["out_obj_ids"]) == [10, 20]
