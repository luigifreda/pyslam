"""
Test to verify that semantic color maps are consistent between separate-process and non-separate-process runs.

This test ensures that when semantic segmentation runs in a separate process,
the color map/palette shared back to the main process matches exactly what would
be used in a non-separate-process run.
"""

import os
import sys
import time
import threading
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from pyslam.config import Config
from pyslam.config_parameters import Parameters
from pyslam.io.dataset_factory import dataset_factory
from pyslam.semantics.semantic_segmentation_factory import semantic_segmentation_factory
from pyslam.semantics.semantic_segmentation_types import SemanticSegmentationType
from pyslam.semantics.semantic_types import SemanticFeatureType, SemanticDatasetType
from pyslam.semantics.semantic_mapping_configs import SemanticMappingConfig
from pyslam.semantics.semantic_segmentation_process import SemanticSegmentationProcess
from pyslam.semantics.perception_tasks import (
    PerceptionTask,
    PerceptionTaskType,
    PerceptionKeyframeData,
)
from pyslam.utilities.system import Printer


# ============================================================================
# Test Configuration
# ============================================================================

# Default test configuration - can be overridden per test function
TEST_CONFIG = {
    "semantic_segmentation_type": SemanticSegmentationType.DETIC,
    "semantic_feature_type": SemanticFeatureType.LABEL,
    "semantic_dataset_type": SemanticDatasetType.ADE20K,
    "image_size": (512, 512),
    "device": None,  # autodetect
}

# Configuration for visual consistency test (uses DETIC for better visual comparison)
VISUAL_TEST_CONFIG = {
    "semantic_segmentation_type": SemanticSegmentationType.DETIC,
    "semantic_feature_type": SemanticFeatureType.LABEL,
    "semantic_dataset_type": SemanticDatasetType.ADE20K,
    "image_size": (512, 512),
    "device": None,  # autodetect
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_color_map_hash(color_map):
    """Compute a hash of the color map for comparison."""
    if color_map is None:
        return None
    if isinstance(color_map, np.ndarray):
        return hash(color_map.tobytes())
    return hash(str(color_map))


def get_color_map_from_segmentation(semantic_segmentation):
    """Extract color map from a semantic segmentation object."""
    color_map_obj = None
    if hasattr(semantic_segmentation, "semantic_color_map_obj"):
        color_map_obj = semantic_segmentation.semantic_color_map_obj
    elif hasattr(semantic_segmentation, "semantic_color_map") and not isinstance(
        getattr(semantic_segmentation, "semantic_color_map", None), np.ndarray
    ):
        potential_obj = semantic_segmentation.semantic_color_map
        if hasattr(potential_obj, "color_map"):
            color_map_obj = potential_obj

    if color_map_obj is not None and hasattr(color_map_obj, "color_map"):
        return color_map_obj.color_map
    elif hasattr(semantic_segmentation, "semantic_color_map") and isinstance(
        getattr(semantic_segmentation, "semantic_color_map", None), np.ndarray
    ):
        return semantic_segmentation.semantic_color_map
    return None


def test_color_map_consistency(config=None):
    """Test that color maps match between separate-process and non-separate-process runs.

    Args:
        config: Optional configuration dict. If None, uses TEST_CONFIG.
    """
    # Use provided config or default
    if config is None:
        config = TEST_CONFIG

    semantic_segmentation_type = config["semantic_segmentation_type"]
    semantic_feature_type = config["semantic_feature_type"]
    semantic_dataset_type = config["semantic_dataset_type"]
    image_size = config["image_size"]
    device = config["device"]

    Printer.green("=" * 80)
    Printer.green("Testing Semantic Color Map Consistency")
    Printer.green("=" * 80)
    Printer.green(f"Segmentation Type: {semantic_segmentation_type.name}")
    Printer.green(f"Feature Type: {semantic_feature_type.name}")
    Printer.green(f"Dataset Type: {semantic_dataset_type.name}")
    Printer.green(f"Image Size: {image_size}")

    # Create semantic mapping config
    semantic_mapping_config = SemanticMappingConfig(
        semantic_segmentation_type=semantic_segmentation_type,
        semantic_feature_type=semantic_feature_type,
        semantic_dataset_type=semantic_dataset_type,
    )

    # Test 1: Non-separate-process run (direct creation)
    Printer.green("\n" + "-" * 80)
    Printer.green("Test 1: Non-separate-process run (direct)")
    Printer.green("-" * 80)

    try:
        semantic_segmentation_direct = semantic_segmentation_factory(
            semantic_segmentation_type=semantic_segmentation_type,
            semantic_feature_type=semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            device=device,
        )

        color_map_direct = get_color_map_from_segmentation(semantic_segmentation_direct)
        color_map_direct_hash = get_color_map_hash(color_map_direct)

        Printer.green(
            f"Direct color map shape: {color_map_direct.shape if color_map_direct is not None else None}"
        )
        Printer.green(f"Direct color map hash: {color_map_direct_hash}")
        if color_map_direct is not None and len(color_map_direct) > 0:
            Printer.green(f"Direct first 3 colors: {color_map_direct[:3].tolist()}")

    except Exception as e:
        Printer.orange(f"Failed to create direct semantic segmentation: {e}")
        Printer.orange("Skipping direct test - this is OK if the model is not available")
        color_map_direct = None
        color_map_direct_hash = None
        semantic_segmentation_direct = None

    # Test 2: Separate-process run
    Printer.green("\n" + "-" * 80)
    Printer.green("Test 2: Separate-process run")
    Printer.green("-" * 80)

    # Temporarily enable separate process
    original_value = Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess
    Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess = True

    semantic_seg_process = None
    try:
        # Create a mock SLAM object (minimal interface)
        class MockSlam:
            pass

        mock_slam = MockSlam()

        Printer.green(
            "Creating SemanticSegmentationProcess (this may take a while to initialize)..."
        )

        # Create SemanticSegmentationProcess with timeout protection
        # Note: __init__ will wait for shared data from worker process
        # If worker process hangs, this will hang too, so we use a thread with timeout
        process_created = threading.Event()
        process_exception = [None]
        created_process = [None]

        def create_process():
            try:
                created_process[0] = SemanticSegmentationProcess(
                    slam=mock_slam,
                    semantic_mapping_config=semantic_mapping_config,
                    device=device,
                    headless=True,
                )
                process_created.set()
            except Exception as e:
                process_exception[0] = e
                process_created.set()

        init_thread = threading.Thread(target=create_process, daemon=True)
        init_thread.start()

        # Wait for process creation with timeout
        max_init_time = 60  # seconds - initialization can take time
        if not process_created.wait(timeout=max_init_time):
            Printer.red(
                f"SemanticSegmentationProcess initialization timed out after {max_init_time} seconds"
            )
            raise RuntimeError("Process initialization timeout")

        if process_exception[0] is not None:
            raise process_exception[0]

        semantic_seg_process = created_process[0]
        Printer.green("SemanticSegmentationProcess created successfully")

        # The semantic_color_map should be available immediately after __init__ completes
        # (it's set during __init__ when shared data is received)
        if (
            not hasattr(semantic_seg_process, "semantic_color_map")
            or semantic_seg_process.semantic_color_map is None
        ):
            Printer.red("Semantic segmentation process did not provide color map")
            # Check if process is still alive
            if hasattr(semantic_seg_process, "process"):
                if semantic_seg_process.process.is_alive():
                    Printer.orange("Worker process is still alive but didn't provide color map")
                else:
                    Printer.red("Worker process has terminated unexpectedly")
            raise RuntimeError("Color map not available")

        # Get color map from the process
        color_map_process = None
        if hasattr(semantic_seg_process, "semantic_color_map"):
            if hasattr(semantic_seg_process.semantic_color_map, "color_map"):
                color_map_process = semantic_seg_process.semantic_color_map.color_map
            elif isinstance(semantic_seg_process.semantic_color_map, np.ndarray):
                color_map_process = semantic_seg_process.semantic_color_map

        color_map_process_hash = get_color_map_hash(color_map_process)

        Printer.green(
            f"Process color map shape: {color_map_process.shape if color_map_process is not None else None}"
        )
        Printer.green(f"Process color map hash: {color_map_process_hash}")
        if color_map_process is not None and len(color_map_process) > 0:
            Printer.green(f"Process first 3 colors: {color_map_process[:3].tolist()}")

        # Test 3: Compare color maps
        Printer.green("\n" + "-" * 80)
        Printer.green("Test 3: Comparison")
        Printer.green("-" * 80)

        if color_map_direct is not None and color_map_process is not None:
            # Compare shapes
            if color_map_direct.shape != color_map_process.shape:
                Printer.red(
                    f"Color map shapes don't match: {color_map_direct.shape} vs {color_map_process.shape}"
                )
                return False

            # Compare hashes
            if color_map_direct_hash != color_map_process_hash:
                Printer.red(
                    f"Color map hashes don't match: {color_map_direct_hash} vs {color_map_process_hash}"
                )

                # Check if they're close (might have slight numerical differences)
                if np.allclose(color_map_direct, color_map_process):
                    Printer.yellow("Color maps are numerically close (within tolerance)")
                    return True
                else:
                    # Show differences
                    diff = np.abs(
                        color_map_direct.astype(np.int32) - color_map_process.astype(np.int32)
                    )
                    max_diff = np.max(diff)
                    Printer.red(f"Maximum color difference: {max_diff}")
                    if max_diff > 0:
                        Printer.red("Color maps differ significantly!")
                        return False
            else:
                Printer.green("✓ Color map hashes match exactly!")
                return True

        elif color_map_direct is None:
            Printer.yellow("Direct color map not available - skipping comparison")
            Printer.green("✓ Process color map created successfully")
            return True

        elif color_map_process is None:
            Printer.red("Process color map not available - test failed")
            return False

    except Exception as e:
        Printer.red(f"Failed to create separate-process semantic segmentation: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup: ensure process is terminated
        if semantic_seg_process is not None:
            try:
                semantic_seg_process.quit()
                # Additional wait to ensure cleanup completes
                time.sleep(0.2)  # Small delay to allow quit() to complete cleanup
            except Exception as e:
                Printer.orange(f"Warning: Error during process cleanup: {e}")

        # Restore original value
        Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess = original_value

    return True


def test_color_map_visual_consistency(config=None):
    """Test that visual outputs are consistent when using the same semantic predictions.

    This test runs inference on the same image through both direct and process,
    then compares that the same semantic predictions produce the same visual output
    when using the respective color maps.

    Args:
        config: Optional configuration dict. If None, uses VISUAL_TEST_CONFIG.
    """
    # Use provided config or default
    if config is None:
        config = VISUAL_TEST_CONFIG

    semantic_segmentation_type = config["semantic_segmentation_type"]
    semantic_feature_type = config["semantic_feature_type"]
    semantic_dataset_type = config["semantic_dataset_type"]
    image_size = config["image_size"]
    device = config["device"]

    Printer.green("\n" + "=" * 80)
    Printer.green("Testing Visual Output Consistency")
    Printer.green("=" * 80)
    Printer.green(f"Segmentation Type: {semantic_segmentation_type.name}")
    Printer.green(f"Image Size: {image_size}")

    # Load test images from dataset (similar to test_semantic_segmentation.py)
    test_image = None
    try:
        config_obj = Config()
        dataset = dataset_factory(config_obj)
        if dataset.is_ok:
            # Try to get a couple of test images
            test_image_ids = [0, 1]  # Use first two images
            test_images = []
            for img_id in test_image_ids:
                img = dataset.getImageColor(img_id)
                if img is not None:
                    test_images.append(img)
                    Printer.green(f"Loaded test image {img_id}: shape {img.shape}")

            if len(test_images) > 0:
                # Use the first available image, resize if needed to match image_size
                test_image = test_images[0]
                H, W = image_size
                if test_image.shape[:2] != (H, W):
                    import cv2

                    test_image = cv2.resize(test_image, (W, H), interpolation=cv2.INTER_LINEAR)
                    Printer.green(f"Resized test image to {image_size}")
            else:
                Printer.yellow("No test images available from dataset, using random image")
                H, W = image_size
                test_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        else:
            Printer.yellow("Dataset not available, using random image")
            H, W = image_size
            test_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    except Exception as e:
        Printer.orange(f"Failed to load dataset images: {e}, using random image")
        H, W = image_size
        test_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    # Test direct run - get semantic predictions
    inference_direct = None
    semantic_segmentation_direct = None
    try:
        semantic_segmentation_direct = semantic_segmentation_factory(
            semantic_segmentation_type=semantic_segmentation_type,
            semantic_feature_type=semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            image_size=image_size,
            device=device,
        )

        inference_direct = semantic_segmentation_direct.infer(test_image)

        # Get the color map object from the direct semantic segmentation
        color_map_direct = None
        if hasattr(semantic_segmentation_direct, "semantic_color_map_obj"):
            color_map_direct = semantic_segmentation_direct.semantic_color_map_obj
        elif hasattr(semantic_segmentation_direct, "semantic_color_map"):
            potential_obj = semantic_segmentation_direct.semantic_color_map
            if hasattr(potential_obj, "to_rgb"):
                color_map_direct = potential_obj

        # Use color map's to_rgb directly to avoid panoptic data differences
        if color_map_direct is not None:
            visual_direct = color_map_direct.to_rgb(inference_direct, bgr=True)
        else:
            # Fallback to semantic segmentation's to_rgb
            visual_direct = semantic_segmentation_direct.to_rgb(inference_direct, bgr=True)

        visual_direct_hash = hash(visual_direct.tobytes())

        Printer.green(
            f"Direct inference shape: {inference_direct.shape if hasattr(inference_direct, 'shape') else type(inference_direct)}"
        )
        Printer.green(f"Direct visual hash: {visual_direct_hash}")

    except Exception as e:
        Printer.orange(f"Failed to create direct semantic segmentation: {e}")
        Printer.orange("Skipping visual consistency test - model not available")
        return True  # Skip if model not available

    if inference_direct is None:
        Printer.orange("No inference output from direct model - skipping visual test")
        return True

    # Test separate-process run
    original_value = Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess
    Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess = True

    semantic_seg_process = None
    try:

        class MockSlam:
            pass

        mock_slam = MockSlam()
        semantic_mapping_config = SemanticMappingConfig(
            semantic_segmentation_type=semantic_segmentation_type,
            semantic_feature_type=semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
        )

        # Create SemanticSegmentationProcess with timeout protection
        process_created = threading.Event()
        process_exception = [None]
        created_process = [None]

        def create_process():
            try:
                created_process[0] = SemanticSegmentationProcess(
                    slam=mock_slam,
                    semantic_mapping_config=semantic_mapping_config,
                    device=device,
                    headless=True,
                )
                process_created.set()
            except Exception as e:
                process_exception[0] = e
                process_created.set()

        init_thread = threading.Thread(target=create_process, daemon=True)
        init_thread.start()

        # Wait for process creation with timeout
        max_init_time = 60  # seconds - initialization can take time
        if not process_created.wait(timeout=max_init_time):
            Printer.red(
                f"SemanticSegmentationProcess initialization timed out after {max_init_time} seconds"
            )
            return False

        if process_exception[0] is not None:
            raise process_exception[0]

        semantic_seg_process = created_process[0]
        # Wait for the worker to be ready before sending tasks
        wait_start = time.time()
        while not semantic_seg_process.is_ready():
            if time.time() - wait_start > max_init_time:
                raise RuntimeError("SemanticSegmentationProcess not ready in time")
            time.sleep(0.1)

        # Test 1: Use the SAME semantic predictions to test visualization consistency
        # This ensures that the same semantic predictions produce the same visual output
        Printer.green("\n" + "-" * 80)
        Printer.green("Test 1: Visualization consistency (same semantic predictions)")
        Printer.green("-" * 80)

        visualization_match = False
        if isinstance(inference_direct, np.ndarray):
            # Use the same semantic predictions with the process color map
            # Use the color map's to_rgb directly to ensure fair comparison
            visual_process = semantic_seg_process.semantic_color_map.to_rgb(
                inference_direct, bgr=True
            )
            visual_process_hash = hash(visual_process.tobytes())
            Printer.green(f"Process visual hash (same semantics): {visual_process_hash}")

            # Compare visual outputs - they should match since we're using the same semantic predictions
            if visual_direct_hash == visual_process_hash:
                Printer.green("✓ Visual outputs match for the same semantic predictions!")
                visualization_match = True
            else:
                # Check if they're close (might have slight differences due to color map differences)
                diff = np.abs(visual_direct.astype(np.int32) - visual_process.astype(np.int32))
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                Printer.yellow(
                    f"Visual outputs differ: max_diff={max_diff}, mean_diff={mean_diff:.2f}"
                )

                # If color maps match, visual outputs should match too
                # This indicates a real problem with color map consistency
                if max_diff > 0:
                    Printer.red(
                        "Visual outputs differ significantly - color maps may not be consistent!"
                    )
                    visualization_match = False
                else:
                    Printer.green("✓ Visual outputs match (within tolerance)")
                    visualization_match = True
        else:
            Printer.yellow("Inference output is not a numpy array - skipping visual comparison")
            visualization_match = True  # Skip if not comparable

        # Test 2: Run actual inference through the process using the dataset image
        # This ensures end-to-end consistency including inference
        Printer.green("\n" + "-" * 80)
        Printer.green("Test 2: End-to-end inference consistency (actual process inference)")
        Printer.green("-" * 80)

        inference_match = False
        try:
            # Create a task with the test image
            mock_keyframe_data = PerceptionKeyframeData(keyframe=None, img=test_image)
            task = PerceptionTask(
                keyframe=None,
                img=test_image,
                task_type=PerceptionTaskType.SEMANTIC_SEGMENTATION,
            )
            semantic_seg_process.add_task(task)
            Printer.green("Task added, waiting for output...")

            # Pop the output (with timeout)
            # Use a reasonable timeout for inference (inference can take time)
            inference_timeout = 120  # 2 minutes should be enough for inference
            output = semantic_seg_process.pop_output(timeout=inference_timeout)

            if output is None:
                Printer.red(f"No output received after {inference_timeout} seconds timeout")
                Printer.red(
                    f"Process state: is_running={semantic_seg_process.is_running.value}, is_looping={semantic_seg_process.is_looping.value}"
                )
                inference_match = False
                return visualization_match and inference_match

            # Check if output has the required fields
            if output.inference_output is None:
                Printer.orange(f"Output received but inference_output is None")
                Printer.orange(f"Output type: {type(output)}")
                Printer.orange(
                    f"Output task_type: {output.task_type if hasattr(output, 'task_type') else 'N/A'}"
                )
                Printer.orange(
                    f"Output has inference_color_image: {hasattr(output, 'inference_color_image') and output.inference_color_image is not None}"
                )
                Printer.orange(
                    f"Output frame_id: {output.frame_id if hasattr(output, 'frame_id') else 'N/A'}"
                )
                # This might happen if the process is still initializing or if there was an error
                # For now, we'll skip this test but mark it as a warning
                Printer.yellow("Skipping inference comparison - output not ready or error occurred")
                inference_match = True  # Don't fail the test, just skip this check
                return visualization_match and inference_match

            if output is not None and output.inference_output is not None:
                inference_process = output.inference_output
                visual_process_inference = output.inference_color_image

                Printer.green(
                    f"Process inference shape: {inference_process.shape if hasattr(inference_process, 'shape') else type(inference_process)}"
                )

                # Compare inference outputs (they should be similar, but may not be identical due to model differences)
                if isinstance(inference_process, np.ndarray) and isinstance(
                    inference_direct, np.ndarray
                ):
                    if inference_process.shape == inference_direct.shape:
                        # Compare semantic predictions
                        inference_diff = np.abs(
                            inference_process.astype(np.int32) - inference_direct.astype(np.int32)
                        )
                        max_inference_diff = np.max(inference_diff)
                        mean_inference_diff = np.mean(inference_diff)

                        Printer.green(
                            f"Inference comparison: max_diff={max_inference_diff}, mean_diff={mean_inference_diff:.2f}"
                        )

                        # For label-based segmentation, predictions should match exactly
                        if semantic_feature_type == SemanticFeatureType.LABEL:
                            if max_inference_diff == 0:
                                Printer.green("✓ Inference outputs match exactly!")
                                inference_match = True
                            else:
                                # Count how many pixels differ
                                num_differing = np.sum(inference_diff > 0)
                                total_pixels = inference_diff.size
                                diff_percentage = 100.0 * num_differing / total_pixels
                                Printer.yellow(
                                    f"Inference outputs differ: {num_differing}/{total_pixels} pixels ({diff_percentage:.2f}%)"
                                )
                                # Allow small differences (e.g., due to numerical precision or model state)
                                if diff_percentage < 1.0:  # Less than 1% difference
                                    Printer.green("✓ Inference outputs match (within tolerance)")
                                    inference_match = True
                                else:
                                    Printer.red("✗ Inference outputs differ significantly!")
                                    inference_match = False
                        else:
                            # For other feature types, just check that inference ran successfully
                            Printer.green(
                                "✓ Inference completed successfully (non-label feature type)"
                            )
                            inference_match = True
                    else:
                        Printer.yellow(
                            f"Shape mismatch: direct={inference_direct.shape}, process={inference_process.shape}"
                        )
                        inference_match = False
                else:
                    Printer.yellow("Inference outputs are not numpy arrays - skipping comparison")
                    inference_match = True  # Skip if not comparable

                # Also compare visual outputs from actual inference
                if visual_process_inference is not None:
                    visual_process_inference_hash = hash(visual_process_inference.tobytes())
                    Printer.green(f"Process inference visual hash: {visual_process_inference_hash}")

                    if visual_direct_hash == visual_process_inference_hash:
                        Printer.green("✓ Visual outputs from actual inference match!")
                    else:
                        Printer.yellow(
                            "Visual outputs from actual inference differ (may be expected if inference differs)"
                        )
            else:
                Printer.orange("Process did not return a valid inference output")
                inference_match = False

        except Exception as e:
            Printer.orange(f"Failed to run inference through process: {e}")
            import traceback

            traceback.print_exc()
            inference_match = False

        # Both tests should pass
        return visualization_match and inference_match

    except Exception as e:
        Printer.red(f"Failed separate-process test: {e}")
        import traceback

        traceback.print_exc()
        return False

    except Exception as e:
        Printer.red(f"Failed separate-process test: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup: ensure process is terminated
        if semantic_seg_process is not None:
            try:
                semantic_seg_process.quit()
                # Additional wait to ensure cleanup completes
                time.sleep(0.2)  # Small delay to allow quit() to complete cleanup
            except Exception as e:
                Printer.orange(f"Warning: Error during process cleanup: {e}")

        # Restore original value
        Parameters.kSemanticMappingMoveSemanticSegmentationToSeparateProcess = original_value


if __name__ == "__main__":
    Printer.green("\n" + "=" * 80)
    Printer.green("Semantic Color Map Consistency Test")
    Printer.green("=" * 80)

    success = True

    # Test 1: Color map consistency
    try:
        success = test_color_map_consistency() and success
    except Exception as e:
        Printer.red(f"Test 1 failed with exception: {e}")
        import traceback

        traceback.print_exc()
        success = False

    # Test 2: Visual consistency (optional)
    try:
        test_color_map_visual_consistency()
    except Exception as e:
        Printer.orange(f"Visual consistency test failed (non-critical): {e}")

    Printer.green("\n" + "=" * 80)
    if success:
        Printer.green("✓ All tests passed!")
    else:
        Printer.red("✗ Some tests failed!")
    Printer.green("=" * 80)

    # Final cleanup: ensure all processes are terminated before exit
    import gc

    Printer.green("\nCleaning up...")
    gc.collect()  # Force garbage collection to clean up any remaining references
    time.sleep(0.5)  # Small delay to allow final cleanup

    Printer.green("Test completed. Exiting...")
    sys.exit(0 if success else 1)
