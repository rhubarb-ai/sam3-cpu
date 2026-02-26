"""
Tests for video_prompter.py helper functions.

Exercises the stitching, overlay, IoU remapping, mask saving, and
memory-table helpers WITHOUT loading the SAM3 model.
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Import helpers under test
from video_prompter import (
    _build_object_tracking,
    _compute_iou,
    _create_overlay_video,
    _extract_last_frame_masks,
    _fmt,
    _match_and_remap,
    _parse_timestamp,
    _resolve_range,
    _save_chunk_masks,
    _stitch_masks_to_video,
    _table,
)


# ---------------------------------------------------------------------------
# _fmt
# ---------------------------------------------------------------------------

class TestFmt:
    def test_bytes(self):
        assert _fmt(512) == "512.0 B"

    def test_megabytes(self):
        assert _fmt(10 * 1024**2) == "10.0 MB"

    def test_gigabytes(self):
        assert _fmt(2.5 * 1024**3) == "2.5 GB"


# ---------------------------------------------------------------------------
# _table (just ensure no crash)
# ---------------------------------------------------------------------------

class TestTable:
    def test_empty(self, capsys):
        _table([])
        assert capsys.readouterr().out == ""

    def test_simple(self, capsys):
        _table([["A", "B"], ["1", "2"]])
        out = capsys.readouterr().out
        assert "A" in out and "B" in out


# ---------------------------------------------------------------------------
# _extract_last_frame_masks
# ---------------------------------------------------------------------------

class TestExtractLastFrameMasks:
    def test_empty(self):
        assert _extract_last_frame_masks({}, {0, 1}) == {}

    def test_single_frame(self):
        masks = np.array([
            np.ones((10, 10), dtype=bool),
            np.zeros((10, 10), dtype=bool),
        ])
        result = {
            0: {
                "out_obj_ids": np.array([0, 1]),
                "out_binary_masks": masks,
            }
        }
        out = _extract_last_frame_masks(result, {0, 1})
        assert 0 in out and 1 in out
        assert out[0].max() == 255  # All-ones mask → 255
        assert out[1].max() == 0   # All-zeros mask → 0

    def test_last_frame_picked(self):
        m0 = np.zeros((5, 5), dtype=bool)
        m1 = np.ones((5, 5), dtype=bool)
        result = {
            0: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([m0])},
            5: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([m1])},
        }
        out = _extract_last_frame_masks(result, {0})
        # Should pick frame 5 (last), whose mask is all-ones
        assert out[0].min() == 255


# ---------------------------------------------------------------------------
# _compute_iou  (mirrors test_iou_matching tests for prompter's local copy)
# ---------------------------------------------------------------------------

class TestComputeIoU:
    def test_identical(self):
        m = np.ones((10, 10), dtype=np.uint8) * 255
        assert _compute_iou(m, m) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        a[:5] = 255
        b = np.zeros((10, 10), dtype=np.uint8)
        b[5:] = 255
        assert _compute_iou(a, b) == 0.0

    def test_bool_masks(self):
        a = np.ones((10, 10), dtype=bool)
        b = np.ones((10, 10), dtype=bool)
        assert _compute_iou(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _match_and_remap
# ---------------------------------------------------------------------------

class TestMatchAndRemap:
    def test_first_chunk_increments(self):
        """First chunk with no prev_masks should assign sequential global IDs."""
        masks = np.array([np.ones((5, 5), dtype=bool)])
        result = {0: {"out_obj_ids": np.array([0]), "out_binary_masks": masks}}
        remapped, ids, mapping, gnid = _match_and_remap(result, {0}, {}, 0)
        assert mapping == {0: 0}
        assert gnid == 1

    def test_second_chunk_matches(self):
        """Second chunk should match via IoU to previous global IDs."""
        mask_a = np.zeros((10, 10), dtype=bool)
        mask_a[:5] = True
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_b[5:] = True

        # Prev masks from chunk 0 with global IDs 0 and 1
        prev = {
            0: (mask_a.astype(np.uint8) * 255),
            1: (mask_b.astype(np.uint8) * 255),
        }

        # Current chunk detects same masks but with fresh IDs 0, 1
        cur_masks = np.array([mask_a, mask_b])
        result = {0: {"out_obj_ids": np.array([0, 1]), "out_binary_masks": cur_masks}}

        remapped, ids, mapping, gnid = _match_and_remap(result, {0, 1}, prev, 2)
        # Should match: new 0→global 0, new 1→global 1
        assert mapping[0] == 0
        assert mapping[1] == 1
        assert gnid == 2  # no new IDs allocated

    def test_new_object_assigned(self):
        """Objects not matching prev should get new global IDs."""
        prev_mask = np.ones((10, 10), dtype=np.uint8) * 255
        prev = {0: prev_mask}

        new_mask = np.zeros((10, 10), dtype=bool)  # completely different
        cur = {0: {"out_obj_ids": np.array([0]), "out_binary_masks": np.array([new_mask])}}

        _, _, mapping, gnid = _match_and_remap(cur, {0}, prev, 1)
        # IoU is 0 → no match → new ID
        assert mapping[0] == 1
        assert gnid == 2

    def test_empty_result(self):
        result, ids, mapping, gnid = _match_and_remap({}, {0, 1}, {}, 0)
        assert result == {}
        assert 0 in ids and 1 in ids
        assert gnid == 2


# ---------------------------------------------------------------------------
# _save_chunk_masks
# ---------------------------------------------------------------------------

class TestSaveChunkMasks:
    def test_creates_pngs(self, tmp_path):
        masks = np.array([np.ones((10, 10), dtype=bool)])
        result = {
            0: {"out_obj_ids": np.array([0]), "out_binary_masks": masks},
            1: {"out_obj_ids": np.array([0]), "out_binary_masks": masks},
        }
        masks_dir = tmp_path / "masks" / "prompt"
        _save_chunk_masks(result, {0}, masks_dir, 10, 10, 3)

        obj_dir = masks_dir / "object_0"
        assert obj_dir.exists()
        pngs = sorted(obj_dir.glob("frame_*.png"))
        assert len(pngs) == 3  # 3 frames

        # Frame 0 should be white (mask), frame 2 should be black (no data)
        from PIL import Image

        arr0 = np.array(Image.open(pngs[0]))
        assert arr0.max() == 255

        arr2 = np.array(Image.open(pngs[2]))
        assert arr2.max() == 0  # frame 2 not in result → black


# ---------------------------------------------------------------------------
# _stitch_masks_to_video (multi-chunk)
# ---------------------------------------------------------------------------

class TestStitchMasks:
    def _setup_chunks(self, tmp_path, n_chunks, frames_per_chunk, overlap, obj_ids):
        """Helper: write fake PNGs for each chunk."""
        chunks_dir = tmp_path / "chunks"
        chunk_infos = []

        for ci in range(n_chunks):
            start = ci * (frames_per_chunk - overlap) if ci > 0 else 0
            if ci == 0:
                start = 0
            else:
                start = chunk_infos[-1]["end"] + 1 - overlap
            end = start + frames_per_chunk - 1
            chunk_infos.append({"chunk": ci, "start": start, "end": end})

            for oid in obj_ids:
                obj_dir = chunks_dir / f"chunk_{ci}" / "masks" / "test" / f"object_{oid}"
                obj_dir.mkdir(parents=True)
                for fidx in range(frames_per_chunk):
                    from PIL import Image

                    # Fill with gray (128) so we can detect non-black
                    arr = np.full((10, 10), 128, dtype=np.uint8)
                    Image.fromarray(arr, mode="L").save(obj_dir / f"frame_{fidx:06d}.png")

        return chunks_dir, chunk_infos

    def test_single_chunk_stitching(self, tmp_path):
        chunks_dir, chunk_infos = self._setup_chunks(tmp_path, 1, 10, 5, {0})
        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        assert mp4.exists()
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert n == 10  # no skip for first chunk

    def test_two_chunk_stitching(self, tmp_path):
        """Two chunks of 10 frames with overlap 5 → 15 unique frames total."""
        chunks_dir, chunk_infos = self._setup_chunks(tmp_path, 2, 10, 5, {0})
        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # chunk 0: 10 frames (no skip), chunk 1: 10-5=5 frames → total 15
        assert n == 15

    def test_missing_object_in_chunk_writes_black(self, tmp_path):
        """Object only in chunk 1 → chunk 0 should write black frames."""
        # Create chunk 0 with no objects, chunk 1 with object 0
        chunks_dir = tmp_path / "chunks"
        # Chunk 0: no object dirs
        chunk0_dir = chunks_dir / "chunk_0" / "masks" / "test"
        chunk0_dir.mkdir(parents=True)

        # Chunk 1: has object 0
        from PIL import Image

        obj_dir = chunks_dir / "chunk_1" / "masks" / "test" / "object_0"
        obj_dir.mkdir(parents=True)
        for i in range(10):
            arr = np.full((10, 10), 200, dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(obj_dir / f"frame_{i:06d}.png")

        chunk_infos = [
            {"chunk": 0, "start": 0, "end": 9},
            {"chunk": 1, "start": 5, "end": 14},
        ]

        out = tmp_path / "out"
        _stitch_masks_to_video(chunks_dir, "test", {0}, chunk_infos, 5, out, 25, 10, 10)

        mp4 = out / "object_0_mask.mp4"
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # chunk 0: 10 black frames, chunk 1: 10-5=5 frames → 15 total
        assert n == 15

        # First frame should be black
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        assert ret
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        assert frame.max() == 0, "First frame should be black (object missing in chunk 0)"

        cap.release()


# ---------------------------------------------------------------------------
# _create_overlay_video
# ---------------------------------------------------------------------------

class TestOverlay:
    def test_overlay_produces_video(self, tmp_path):
        """Overlay with a synthetic video and mask."""
        w, h, n = 20, 20, 5

        # Create original video
        vid_path = tmp_path / "source.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_path), fourcc, 25, (w, h), True)
        for _ in range(n):
            frame = np.full((h, w, 3), 128, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        # Create mask video
        mask_path = tmp_path / "mask.mp4"
        mwriter = cv2.VideoWriter(str(mask_path), fourcc, 25, (w, h), False)
        for _ in range(n):
            mask = np.full((h, w), 255, dtype=np.uint8)
            mwriter.write(mask)
        mwriter.release()

        out_path = tmp_path / "overlay.mp4"
        _create_overlay_video(vid_path, [mask_path], out_path, alpha=0.5)

        assert out_path.exists()
        cap = cv2.VideoCapture(str(out_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()


# ---------------------------------------------------------------------------
# _parse_timestamp
# ---------------------------------------------------------------------------

class TestParseTimestamp:
    def test_plain_float(self):
        assert _parse_timestamp("4.5") == 4.5

    def test_integer(self):
        assert _parse_timestamp("10") == 10.0

    def test_mm_ss(self):
        assert _parse_timestamp("1:30") == 90.0

    def test_hh_mm_ss(self):
        assert _parse_timestamp("0:01:30") == 90.0

    def test_hh_mm_ss_large(self):
        assert _parse_timestamp("1:02:03") == 3723.0

    def test_mm_ss_fractional(self):
        assert _parse_timestamp("2:30.5") == 150.5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("not-a-time")

    def test_too_many_colons_raises(self):
        with pytest.raises(ValueError):
            _parse_timestamp("1:2:3:4")


# ---------------------------------------------------------------------------
# _resolve_range
# ---------------------------------------------------------------------------

class TestResolveRange:
    def test_no_range_returns_none(self, tmp_path):
        assert _resolve_range(tmp_path / "v.mp4", None, None) is None

    def test_frame_range_passthrough(self, tmp_path):
        result = _resolve_range(tmp_path / "v.mp4", (10, 50), None)
        assert result == (10, 50)


# ---------------------------------------------------------------------------
# _build_object_tracking
# ---------------------------------------------------------------------------

class TestBuildObjectTracking:
    def test_empty_set(self, tmp_path):
        result = _build_object_tracking(tmp_path, set(), 25.0, 0)
        assert result == []

    def test_missing_mp4(self, tmp_path):
        """Object ID with no corresponding mask video."""
        result = _build_object_tracking(tmp_path, {0}, 25.0, 0)
        assert len(result) == 1
        assert result[0]["object_id"] == 0
        assert result[0]["first_frame"] is None
        assert result[0]["total_frames_active"] == 0

    def test_with_mask_video(self, tmp_path):
        """Create a short mask video with known white frames."""
        w, h, n = 64, 64, 20
        fps = 10.0

        mp4 = tmp_path / "object_0_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
        for i in range(n):
            # Frames 5-14 are white (10 active frames)
            if 5 <= i <= 14:
                frame = np.full((h, w), 255, dtype=np.uint8)
            else:
                frame = np.zeros((h, w), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        result = _build_object_tracking(tmp_path, {0}, fps, frame_offset=0)
        assert len(result) == 1
        r = result[0]
        assert r["object_id"] == 0
        assert r["first_frame"] == 5
        assert r["last_frame"] == 14
        assert r["total_frames_active"] == 10
        assert r["total_frames"] == n
        assert r["first_timestamp"] == 0.5
        assert r["last_timestamp"] == 1.4
        assert "first_timecode" in r
        assert r["first_timecode"].startswith("00:00:")

    def test_frame_offset(self, tmp_path):
        """Frame offset shifts absolute frame numbers."""
        w, h, n = 32, 32, 5
        fps = 10.0
        offset = 100

        mp4 = tmp_path / "object_1_mask.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
        for i in range(n):
            # All frames active
            writer.write(np.full((h, w), 255, dtype=np.uint8))
        writer.release()

        result = _build_object_tracking(tmp_path, {1}, fps, frame_offset=offset)
        r = result[0]
        assert r["first_frame"] == offset
        assert r["last_frame"] == offset + n - 1
        assert r["first_timestamp"] == round(offset / fps, 3)

    def test_multiple_objects_sorted(self, tmp_path):
        """Results are sorted by object ID."""
        w, h, n = 32, 32, 3
        fps = 10.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for oid in [2, 0, 1]:
            mp4 = tmp_path / f"object_{oid}_mask.mp4"
            writer = cv2.VideoWriter(str(mp4), fourcc, fps, (w, h), False)
            for _ in range(n):
                writer.write(np.full((h, w), 255, dtype=np.uint8))
            writer.release()

        result = _build_object_tracking(tmp_path, {2, 0, 1}, fps, 0)
        ids = [r["object_id"] for r in result]
        assert ids == [0, 1, 2]
