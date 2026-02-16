"""
Test Suite for SAM3 - All 9 Scenarios
Tests use actual inference and validate mask generation.
"""

import pytest
from pathlib import Path
import numpy as np
from sam3 import Sam3


class TestScenarioA:
    """Test Scenario A: Single image with text prompts."""
    
    @pytest.mark.image
    @pytest.mark.scenario_a
    def test_single_image_single_prompt(self, sam3_instance, test_image_truck, temp_output_dir):
        """Test single image with one text prompt."""
        result = sam3_instance.process_image(
            image_path=test_image_truck,
            prompts=["truck"],
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"
        assert len(result.object_ids) > 0, "No objects detected"
        
        # Validate mask files exist
        for mask_file in result.mask_files:
            assert Path(mask_file).exists(), f"Mask file not found: {mask_file}"
    
    @pytest.mark.image
    @pytest.mark.scenario_a
    def test_single_image_multiple_prompts(self, sam3_instance, test_image_truck, temp_output_dir):
        """Test single image with multiple text prompts."""
        result = sam3_instance.process_image(
            image_path=test_image_truck,
            prompts=["truck", "wheel", "road"],
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"
        
        # Check output directory structure
        output_path = Path(result.output_dir)
        assert output_path.exists(), "Output directory not created"


class TestScenarioB:
    """Test Scenario B: Single image with bounding boxes."""
    
    @pytest.mark.image
    @pytest.mark.scenario_b
    def test_single_image_single_box(self, sam3_instance, test_image_cafe, temp_output_dir):
        """Test single image with one bounding box."""
        box = [100, 100, 200, 200]  # XYWH format
        
        result = sam3_instance.process_image(
            image_path=test_image_cafe,
            boxes=box,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"
        assert len(result.object_ids) > 0, "No objects detected"
    
    @pytest.mark.image
    @pytest.mark.scenario_b
    def test_single_image_multiple_boxes(self, sam3_instance, test_image_cafe, temp_output_dir):
        """Test single image with multiple bounding boxes."""
        boxes = [[50, 50, 100, 100], [200, 200, 150, 150]]
        box_labels = [1, 1]  # Both positive
        
        result = sam3_instance.process_image(
            image_path=test_image_cafe,
            boxes=boxes,
            box_labels=box_labels,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"


class TestScenarioC:
    """Test Scenario C: Multiple images with text prompts."""
    
    @pytest.mark.image
    @pytest.mark.scenario_c
    def test_batch_images_with_prompts(
        self, 
        sam3_instance, 
        test_image_truck, 
        test_image_groceries,
        temp_output_dir
    ):
        """Test batch processing of multiple images with text prompts."""
        images = [test_image_truck, test_image_groceries]
        prompts = ["object", "item"]
        
        result = sam3_instance.process_image(
            image_path=images,
            prompts=prompts,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"
        
        # Verify metadata has entries for both images
        assert len(result.metadata) == len(images), "Missing metadata for some images"


class TestScenarioD:
    """Test Scenario D: Multiple images with bounding boxes."""
    
    @pytest.mark.image
    @pytest.mark.scenario_d
    def test_batch_images_with_boxes(
        self,
        sam3_instance,
        test_image_cafe,
        test_image_test,
        temp_output_dir
    ):
        """Test batch processing of multiple images with bounding boxes."""
        images = [test_image_cafe, test_image_test]
        boxes = [[50, 50, 150, 150], [100, 100, 200, 200]]
        
        result = sam3_instance.process_image(
            image_path=images,
            boxes=boxes,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No masks generated"


class TestScenarioE:
    """Test Scenario E: Video with text prompts."""
    
    @pytest.mark.video
    @pytest.mark.scenario_e
    @pytest.mark.slow
    def test_video_with_text_prompts_forward(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with text prompts (forward propagation)."""
        result = sam3_instance.process_video_with_prompts(
            video_path=test_video_tennis_480p,
            prompts=["person"],
            output_dir=temp_output_dir,
            propagation_direction="forward"
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"
        assert len(result.object_ids) > 0, "No objects detected"
        
        # Validate mask videos exist
        for mask_file in result.mask_files:
            assert Path(mask_file).exists(), f"Mask video not found: {mask_file}"
            assert Path(mask_file).suffix == ".mp4", "Mask video is not MP4 format"
        
        # Check metadata
        assert "fps" in result.metadata, "Missing FPS in metadata"
        assert "total_frames" in result.metadata, "Missing total_frames in metadata"
    
    @pytest.mark.video
    @pytest.mark.scenario_e
    @pytest.mark.slow
    def test_video_with_multiple_prompts(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with multiple text prompts."""
        result = sam3_instance.process_video_with_prompts(
            video_path=test_video_tennis_480p,
            prompts=["person", "tennis racket"],
            output_dir=temp_output_dir,
            propagation_direction="both"
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"


class TestScenarioF:
    """Test Scenario F: Video with point prompts."""
    
    @pytest.mark.video
    @pytest.mark.scenario_f
    @pytest.mark.slow
    def test_video_with_point_prompts(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with point prompts on specific frame."""
        points = [[320, 180], [400, 200]]  # Center-ish points for 480p
        labels = [1, 1]  # Both positive
        
        result = sam3_instance.process_video_with_points(
            video_path=test_video_tennis_480p,
            frame_idx=0,
            points=points,
            point_labels=labels,
            object_id=1,
            output_dir=temp_output_dir,
            propagation_direction="forward"
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"
        assert 1 in result.object_ids, "Expected object ID not found"
        
        # Check annotation frame in metadata
        assert "annotation_frame" in result.metadata, "Missing annotation_frame in metadata"
    
    @pytest.mark.video
    @pytest.mark.scenario_f
    @pytest.mark.slow
    def test_video_with_mixed_point_labels(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with positive and negative points."""
        points = [[320, 180], [100, 100]]
        labels = [1, 0]  # One positive, one negative
        
        result = sam3_instance.process_video_with_points(
            video_path=test_video_tennis_480p,
            frame_idx=5,
            points=points,
            point_labels=labels,
            object_id=2,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"


class TestScenarioG:
    """Test Scenario G: Refine video object segmentation."""
    
    @pytest.mark.video
    @pytest.mark.scenario_g
    @pytest.mark.slow
    def test_refine_video_object(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test refining existing video object with additional points."""
        refinement_points = [[350, 190]]
        labels = [1]
        
        result = sam3_instance.refine_video_object(
            video_path=test_video_tennis_480p,
            frame_idx=10,
            object_id=1,
            points=refinement_points,
            point_labels=labels,
            output_dir=temp_output_dir,
            propagation_direction="forward"
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No refined mask videos generated"
        assert "refinement_frame" in result.metadata, "Missing refinement_frame in metadata"
        assert "num_refinement_points" in result.metadata, "Missing refinement point count"


class TestScenarioH:
    """Test Scenario H: Remove objects from video segmentation."""
    
    @pytest.mark.video
    @pytest.mark.scenario_h
    def test_remove_single_object(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test removing a single object from video segmentation."""
        result = sam3_instance.remove_video_objects(
            video_path=test_video_tennis_480p,
            object_ids=1,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert "removed_object_ids" in result.metadata, "Missing removed object IDs"
        assert 1 in result.metadata["removed_object_ids"], "Object ID not in removed list"
    
    @pytest.mark.video
    @pytest.mark.scenario_h
    def test_remove_multiple_objects(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test removing multiple objects from video segmentation."""
        object_ids_to_remove = [2, 3]
        
        result = sam3_instance.remove_video_objects(
            video_path=test_video_tennis_480p,
            object_ids=object_ids_to_remove,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        removed_ids = result.metadata.get("removed_object_ids", [])
        for obj_id in object_ids_to_remove:
            assert obj_id in removed_ids, f"Object ID {obj_id} not in removed list"


class TestScenarioI:
    """Test Scenario I: Video with segment-based prompts."""
    
    @pytest.mark.video
    @pytest.mark.scenario_i
    @pytest.mark.slow
    def test_video_with_segments(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with different prompts for different segments."""
        segments = {
            "segments": [
                {
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "prompts": ["person"]
                },
                {
                    "start_time_sec": 2.0,
                    "end_time_sec": 5.0,
                    "points": [[320, 180], [400, 200]],
                    "labels": [1, 1]
                }
            ]
        }
        
        result = sam3_instance.process_video_with_segments(
            video_path=test_video_tennis_480p,
            segments=segments,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"
        assert "num_segments" in result.metadata, "Missing segment count in metadata"
        assert result.metadata["num_segments"] == len(segments["segments"]), "Segment count mismatch"
    
    @pytest.mark.video
    @pytest.mark.scenario_i
    @pytest.mark.slow
    def test_video_with_frame_based_segments(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test video processing with frame-based segment definitions."""
        segments = {
            "segments": [
                {
                    "start_frame": 0,
                    "end_frame": 50,
                    "prompts": ["person", "racket"]
                },
                {
                    "start_frame": 50,
                    "end_frame": 100,
                    "points": [[300, 150]],
                    "labels": [1]
                }
            ]
        }
        
        result = sam3_instance.process_video_with_segments(
            video_path=test_video_tennis_480p,
            segments=segments,
            output_dir=temp_output_dir
        )
        
        assert result.success, f"Processing failed: {result.errors}"
        assert len(result.mask_files) > 0, "No mask videos generated"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.image
    def test_invalid_image_path(self, sam3_instance, temp_output_dir):
        """Test handling of non-existent image file."""
        with pytest.raises(FileNotFoundError):
            sam3_instance.process_image(
                image_path="nonexistent_image.jpg",
                prompts=["test"],
                output_dir=temp_output_dir
            )
    
    @pytest.mark.image
    def test_no_prompts_or_boxes(self, sam3_instance, test_image_truck, temp_output_dir):
        """Test error when neither prompts nor boxes provided."""
        with pytest.raises(ValueError):
            sam3_instance.process_image(
                image_path=test_image_truck,
                output_dir=temp_output_dir
            )
    
    @pytest.mark.video
    def test_invalid_video_path(self, sam3_instance, temp_output_dir):
        """Test handling of non-existent video file."""
        with pytest.raises(FileNotFoundError):
            sam3_instance.process_video_with_prompts(
                video_path="nonexistent_video.mp4",
                prompts=["test"],
                output_dir=temp_output_dir
            )
    
    @pytest.mark.video
    def test_mismatched_points_and_labels(
        self,
        sam3_instance,
        test_video_tennis_480p,
        temp_output_dir
    ):
        """Test error with mismatched number of points and labels."""
        with pytest.raises(ValueError):
            sam3_instance.process_video_with_points(
                video_path=test_video_tennis_480p,
                frame_idx=0,
                points=[[100, 100], [200, 200]],
                point_labels=[1],  # Mismatch: 2 points but 1 label
                output_dir=temp_output_dir
            )


class TestImportAliases:
    """Test that import aliases work correctly."""
    
    def test_sam3_import(self):
        """Test that 'from sam3 import Sam3' works."""
        from sam3 import Sam3
        assert Sam3 is not None
        
        instance = Sam3(verbose=False)
        assert instance is not None
    
    def test_sam3_entrypoint_import(self):
        """Test that 'from sam3 import Sam3Entrypoint' works."""
        from sam3 import Sam3Entrypoint
        assert Sam3Entrypoint is not None
        
        instance = Sam3Entrypoint(verbose=False)
        assert instance is not None
    
    def test_sam3_equals_sam3_entrypoint(self):
        """Test that Sam3 and Sam3Entrypoint are the same class."""
        from sam3 import Sam3, Sam3Entrypoint
        assert Sam3 is Sam3Entrypoint
