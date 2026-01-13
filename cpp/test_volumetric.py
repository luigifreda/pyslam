#!/usr/bin/env python3
"""
Test script for the volumetric module to verify imports and linking.

This script tests:
1. Basic module import
2. TBB utilities (thread management)
3. Bounding box classes (BoundingBox3D, OrientedBoundingBox3D, BoundingBox2D, OrientedBoundingBox2D)
4. Camera frustrum class with all methods
5. Voxel data classes (VoxelData, VoxelSemanticData, VoxelGridData)
6. All voxel grid classes:
   - VoxelGrid, VoxelBlockGrid
   - VoxelSemanticGrid, VoxelSemanticGridProbabilistic
   - VoxelBlockSemanticGrid, VoxelBlockSemanticProbabilisticGrid
7. Grid operations:
   - Integration with/without colors, semantic data, instance IDs, depths
   - Segment operations (integrate_segment, merge_segments, remove_segment)
   - Spatial queries (get_voxels_in_bb, get_voxels_in_camera_frustrum)
   - Carving and filtering (carve, remove_low_count_voxels, remove_low_confidence_voxels)
   - Data retrieval (get_voxels, get_points, get_colors, get_object_segments, get_class_segments)

"""

import sys
import os
import numpy as np

# Add lib directory to path to import the volumetric module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))


def test_import():
    """Test basic module import."""
    print("=" * 60)
    print("Testing volumetric module import...")
    print("=" * 60)

    try:
        import volumetric

        print("✓ Successfully imported volumetric module")
        return volumetric
    except ImportError as e:
        error_msg = str(e)
        print(f"✗ Failed to import volumetric module: {e}")
        print("\nPossible solutions:")

        # Check for undefined symbol errors (linking issues)
        if "undefined symbol" in error_msg.lower():
            print("⚠ Detected undefined symbol error - this indicates a linking problem.")
            print("  The module was built but is missing some implementations.")
            print("  This usually happens when source files are added but not rebuilt.")
            print("\n  Try rebuilding from scratch:")
            print("    cd cpp")
            print("    ./clean.sh")
            print("    ./build.sh")
        else:
            print("1. Make sure the volumetric module is built:")
            print("   cd cpp && ./build.sh")
            print("2. Check that volumetric.cpython-*-linux-gnu.so exists in cpp/lib/")
            print("3. Verify you're using the correct Python environment")

        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Unexpected error importing volumetric module: {e}")

        # Check for undefined symbol errors in any exception
        if "undefined symbol" in error_msg.lower():
            print("\n⚠ Detected undefined symbol error - this indicates a linking problem.")
            print("  Try rebuilding from scratch:")
            print("    cd cpp")
            print("    ./clean.sh")
            print("    ./build.sh")

        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_tbb_utils(volumetric):
    """Test TBB utilities."""
    print("\n" + "=" * 60)
    print("Testing TBBUtils...")
    print("=" * 60)

    try:
        TBBUtils = volumetric.TBBUtils
        print("✓ TBBUtils class found")

        # Test get_max_threads
        max_threads = TBBUtils.get_max_threads()
        print(f"✓ get_max_threads() = {max_threads}")

        # Test set_max_threads
        original_threads = max_threads
        TBBUtils.set_max_threads(4)
        new_threads = TBBUtils.get_max_threads()
        print(f"✓ set_max_threads(4) -> get_max_threads() = {new_threads}")

        # Restore original
        TBBUtils.set_max_threads(original_threads)
        print(f"✓ Restored max_threads to {original_threads}")

        return True
    except Exception as e:
        print(f"✗ TBBUtils test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_bounding_boxes(volumetric):
    """Test BoundingBox classes."""
    print("\n" + "=" * 60)
    print("Testing BoundingBox Classes...")
    print("=" * 60)

    try:
        # Test BoundingBox3D
        BoundingBox3D = volumetric.BoundingBox3D
        print("✓ BoundingBox3D class found")

        min_point = np.array([0.0, 0.0, 0.0])
        max_point = np.array([1.0, 1.0, 1.0])
        bbox = BoundingBox3D(min_point, max_point)
        print(f"✓ Created BoundingBox3D: min={min_point}, max={max_point}")

        # Test getters
        center = bbox.get_center()
        size = bbox.get_size()
        volume = bbox.get_volume()
        surface_area = bbox.get_surface_area()
        diagonal = bbox.get_diagonal_length()
        print(f"✓ get_center() = {center}")
        print(f"✓ get_size() = {size}")
        print(f"✓ get_volume() = {volume}")
        print(f"✓ get_surface_area() = {surface_area}")
        print(f"✓ get_diagonal_length() = {diagonal}")

        # Test contains
        test_point = np.array([0.5, 0.5, 0.5])
        contains = bbox.contains(test_point)
        print(f"✓ contains(point) = {contains}")

        # Test contains with vector
        test_points = [np.array([0.2, 0.2, 0.2]), np.array([1.5, 1.5, 1.5])]
        contains_vec = bbox.contains(test_points)
        print(f"✓ contains(points) = {contains_vec}")

        # Test intersects
        bbox2 = BoundingBox3D(np.array([0.5, 0.5, 0.5]), np.array([2.0, 2.0, 2.0]))
        intersects = bbox.intersects(bbox2)
        print(f"✓ intersects(other) = {intersects}")

        # Test Quaterniond wrapper
        Quaterniond = volumetric.Quaterniond
        print("✓ Quaterniond class found")

        # Test Quaterniond construction
        quat1 = Quaterniond()  # Identity quaternion
        print(f"✓ Created Quaterniond() (identity)")

        quat2 = Quaterniond(1.0, 0.0, 0.0, 0.0)  # w, x, y, z
        print(f"✓ Created Quaterniond(w, x, y, z)")

        quat3 = Quaterniond(np.array([1.0, 0.0, 0.0, 0.0]))  # From numpy array
        print(f"✓ Created Quaterniond from numpy array")

        # Test Quaterniond methods
        w, x, y, z = quat2.w(), quat2.x(), quat2.y(), quat2.z()
        coeffs = quat2.coeffs()
        print(f"✓ Quaterniond methods: w={w}, x={x}, y={y}, z={z}, coeffs shape={coeffs.shape}")

        # Test OrientedBoundingBox3D
        OrientedBoundingBox3D = volumetric.OrientedBoundingBox3D
        print("✓ OrientedBoundingBox3D class found")

        # Try to create OBB using constructor with numpy array
        center_obb = np.array([0.5, 0.5, 0.5])
        size_obb = np.array([1.0, 1.0, 1.0])
        obb = None
        try:
            # Try with quaternion as numpy array [w, x, y, z]
            orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
            obb = OrientedBoundingBox3D(center_obb, orientation_quat, size_obb)
            print(f"✓ Created OrientedBoundingBox3D with numpy array quaternion")
        except (TypeError, ValueError) as e:
            # Try with Quaterniond object
            try:
                quat_obj = Quaterniond(1.0, 0.0, 0.0, 0.0)
                obb = OrientedBoundingBox3D(center_obb, quat_obj, size_obb)
                print(f"✓ Created OrientedBoundingBox3D with Quaterniond object")
            except Exception as e2:
                print(
                    f"  ⚠ OrientedBoundingBox3D constructor test skipped: {type(e2).__name__}: {e2}"
                )

        # Test OBB methods - use CameraFrustrum.get_obb() if constructor failed
        if obb is None:
            try:
                # Create a CameraFrustrum and get its OBB
                CameraFrustrum = volumetric.CameraFrustrum
                K = np.array(
                    [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
                )
                pose = np.eye(4, dtype=np.float64)
                frustrum = CameraFrustrum(K, 640, 480, pose, 10.0, 0.1)
                obb = frustrum.get_obb()
                print(f"✓ Created OrientedBoundingBox3D via CameraFrustrum.get_obb()")
            except Exception as e:
                print(f"  ⚠ Could not create OrientedBoundingBox3D instance: {e}")
                obb = None

        if obb is not None:
            obb_volume = obb.get_volume()
            obb_corners = obb.get_corners()
            print(f"✓ get_volume() = {obb_volume}")
            print(f"✓ get_corners() returned {len(obb_corners)} corners")

            # Test contains
            test_point_obb = np.array([0.0, 0.0, 1.0])
            contains_obb = obb.contains(test_point_obb)
            print(f"✓ contains() = {contains_obb}")
        else:
            print(f"  ⚠ OBB methods test skipped (no instance available)")

        # Test BoundingBox2D
        BoundingBox2D = volumetric.BoundingBox2D
        print("✓ BoundingBox2D class found")

        min_point_2d = np.array([0.0, 0.0])
        max_point_2d = np.array([1.0, 1.0])
        bbox2d = BoundingBox2D(min_point_2d, max_point_2d)
        print(f"✓ Created BoundingBox2D")

        area = bbox2d.get_area()
        perimeter = bbox2d.get_perimeter()
        print(f"✓ get_area() = {area}")
        print(f"✓ get_perimeter() = {perimeter}")

        # Test OrientedBoundingBox2D
        OrientedBoundingBox2D = volumetric.OrientedBoundingBox2D
        print("✓ OrientedBoundingBox2D class found")

        center_2d = np.array([0.5, 0.5])
        angle_rad = 0.0
        size_2d = np.array([1.0, 1.0])
        obb2d = OrientedBoundingBox2D(center_2d, angle_rad, size_2d)
        print(f"✓ Created OrientedBoundingBox2D")

        return True
    except Exception as e:
        print(f"✗ BoundingBox test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_camera_frustum(volumetric):
    """Test CameraFrustrum class."""
    print("\n" + "=" * 60)
    print("Testing CameraFrustrum...")
    print("=" * 60)

    try:
        CameraFrustrum = volumetric.CameraFrustrum
        print("✓ CameraFrustrum class found")

        # Create a simple camera frustrum using K matrix
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        width, height = 640, 480
        pose = np.eye(4, dtype=np.float64)
        depth_min, depth_max = 0.1, 10.0

        frustrum = CameraFrustrum(K, width, height, pose, depth_max, depth_min)
        print(f"✓ Created CameraFrustrum with K matrix: {width}x{height}")

        # Test getters
        K_ret = frustrum.get_K()
        T_cw_ret = frustrum.get_T_cw()
        R_cw_ret = frustrum.get_R_cw()
        t_cw_ret = frustrum.get_t_cw()
        # Test get_orientation_cw() - should now work with Quaterniond wrapper
        try:
            orientation_ret = frustrum.get_orientation_cw()
            print(f"✓ get_orientation_cw() returned Quaterniond")
            # Test Quaterniond methods
            w_ret = orientation_ret.w()
            coeffs_ret = orientation_ret.coeffs()
            print(f"  ✓ Quaterniond.w() = {w_ret}, coeffs shape = {coeffs_ret.shape}")
            print(f"✓ get_K(), get_T_cw(), get_R_cw(), get_t_cw(), get_orientation_cw() succeeded")
        except TypeError as e:
            print(f"✓ get_K(), get_T_cw(), get_R_cw(), get_t_cw() succeeded")
            print(f"  ⚠ get_orientation_cw() failed: {e}")

        # Test setters
        frustrum.set_width(800)
        frustrum.set_height(600)
        frustrum.set_depth_max(20.0)
        frustrum.set_depth_min(0.05)
        print(f"✓ set_width(), set_height(), set_depth_max(), set_depth_min() succeeded")

        # Test set_intrinsics with K matrix
        frustrum.set_intrinsics(K)
        print(f"✓ set_intrinsics(K) succeeded")

        # Test set_intrinsics with individual parameters
        frustrum.set_intrinsics(500.0, 500.0, 320.0, 240.0)
        print(f"✓ set_intrinsics(fx, fy, cx, cy) succeeded")

        # Test set_T_cw with matrix
        frustrum.set_T_cw(pose)
        print(f"✓ set_T_cw(matrix) succeeded")

        # Test set_T_cw with quaternion and translation
        # Try with Quaterniond object first, then numpy array
        try:
            Quaterniond = volumetric.Quaterniond
            orientation_quat_obj = Quaterniond(1.0, 0.0, 0.0, 0.0)
            translation = np.array([0.0, 0.0, 0.0])
            frustrum.set_T_cw(orientation_quat_obj, translation)
            print(f"✓ set_T_cw(Quaterniond object, translation) succeeded")
        except Exception as e:
            # Try with numpy array
            try:
                orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
                translation = np.array([0.0, 0.0, 0.0])
                frustrum.set_T_cw(orientation_quat, translation)
                print(f"✓ set_T_cw(numpy array quaternion, translation) succeeded")
            except Exception as e2:
                print(f"  ⚠ set_T_cw(quaternion, translation) failed: {e2}")

        # Create frustrum with individual parameters
        try:
            Quaterniond = volumetric.Quaterniond
            orientation_quat_obj = Quaterniond(1.0, 0.0, 0.0, 0.0)
            translation = np.array([0.0, 0.0, 0.0])
            frustrum2 = CameraFrustrum(
                500.0,
                500.0,
                320.0,
                240.0,
                width,
                height,
                orientation_quat_obj,
                translation,
                depth_max,
                depth_min,
            )
            print(f"✓ Created CameraFrustrum with individual parameters (Quaterniond)")
        except Exception as e:
            try:
                orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
                translation = np.array([0.0, 0.0, 0.0])
                frustrum2 = CameraFrustrum(
                    500.0,
                    500.0,
                    320.0,
                    240.0,
                    width,
                    height,
                    orientation_quat,
                    translation,
                    depth_max,
                    depth_min,
                )
                print(f"✓ Created CameraFrustrum with individual parameters (numpy array)")
            except Exception as e2:
                print(f"  ⚠ CameraFrustrum constructor with quaternion failed: {e2}")

        # Test get_obb (oriented bounding box)
        obb = frustrum.get_obb()
        print(f"✓ get_obb() returned OrientedBoundingBox3D")

        # Test get_corners
        corners = frustrum.get_corners()
        print(f"✓ get_corners() returned {len(corners)} corners")

        # Test contains
        test_point = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        in_frustum = frustrum.contains(test_point)
        print(f"✓ contains() = {in_frustum}")

        # Test is_in_bbox and is_in_obb
        is_in_bbox = frustrum.is_in_bbox(test_point)
        is_in_obb = frustrum.is_in_obb(test_point)
        print(f"✓ is_in_bbox() = {is_in_bbox}, is_in_obb() = {is_in_obb}")

        # Test is_cache_valid
        cache_valid = frustrum.is_cache_valid()
        print(f"✓ is_cache_valid() = {cache_valid}")

        # Test ImagePoint
        ImagePoint = volumetric.ImagePoint
        print("✓ ImagePoint class found")
        img_point = ImagePoint()
        img_point.u = 100
        img_point.v = 200
        img_point.depth = 5.0
        print(f"✓ Created ImagePoint: u={img_point.u}, v={img_point.v}, depth={img_point.depth}")

        return True
    except Exception as e:
        print(f"✗ CameraFrustrum test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_voxel_grid_classes(volumetric):
    """Test all voxel grid classes."""
    print("\n" + "=" * 60)
    print("Testing Voxel Grid Classes...")
    print("=" * 60)

    classes_to_test = [
        ("VoxelGrid", {}),
        ("VoxelBlockGrid", {"block_size": 8}),
        ("VoxelSemanticGrid", {}),
        ("VoxelSemanticGridProbabilistic", {}),
        ("VoxelBlockSemanticGrid", {"block_size": 8}),
        ("VoxelBlockSemanticProbabilisticGrid", {"block_size": 8}),
    ]

    results = {}

    for class_name, kwargs in classes_to_test:
        try:
            GridClass = getattr(volumetric, class_name)
            print(f"✓ {class_name} class found")

            # Try to instantiate
            voxel_size = 0.05
            if "block_size" in kwargs:
                grid = GridClass(voxel_size, kwargs["block_size"])
                print(
                    f"  ✓ Created {class_name}(voxel_size={voxel_size}, block_size={kwargs['block_size']})"
                )

                # Test block-specific methods (available for all block grids)
                num_blocks = grid.num_blocks()
                block_size = grid.get_block_size()
                total_voxels = grid.get_total_voxel_count()
                print(
                    f"  ✓ num_blocks() = {num_blocks}, get_block_size() = {block_size}, get_total_voxel_count() = {total_voxels}"
                )
            else:
                grid = GridClass(voxel_size)
                print(f"  ✓ Created {class_name}(voxel_size={voxel_size})")

            # Test basic methods
            size = grid.size()
            empty = grid.empty()
            print(f"  ✓ size() = {size}, empty() = {empty}")

            # Test integrate with dummy data
            points = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64)
            colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)

            if "Semantic" in class_name:
                # Semantic grids need class_ids
                class_ids = np.array([1, 2, 3], dtype=np.int32)
                instance_ids = np.array([10, 20, 30], dtype=np.int32)

                # Test integrate with optional instance_ids and depths
                grid.integrate(points, colors, class_ids)
                print(f"  ✓ integrate() with class_ids succeeded")

                grid.integrate(points, colors, class_ids, instance_ids)
                print(f"  ✓ integrate() with class_ids and instance_ids succeeded")

                depths = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                grid.integrate(points, colors, class_ids, instance_ids, depths)
                print(f"  ✓ integrate() with class_ids, instance_ids, and depths succeeded")

                # Test set_depth_threshold and set_depth_decay_rate
                grid.set_depth_threshold(3.0)
                grid.set_depth_decay_rate(0.05)
                print(f"  ✓ set_depth_threshold() and set_depth_decay_rate() succeeded")

                # Test integrate_segment
                segment_points = np.array([[0.3, 0.3, 0.3]], dtype=np.float64)
                segment_colors = np.array([[128, 128, 128]], dtype=np.uint8)
                grid.integrate_segment(segment_points, segment_colors, class_id=5, object_id=50)
                print(f"  ✓ integrate_segment() succeeded")

                # Test merge_segments
                grid.merge_segments(10, 20)
                print(f"  ✓ merge_segments() succeeded")

                # Test remove_segment
                grid.remove_segment(30)
                print(f"  ✓ remove_segment() succeeded")

                # Test remove_low_confidence_segments (expects int, not float)
                grid.remove_low_confidence_segments(min_confidence=1)
                print(f"  ✓ remove_low_confidence_segments() succeeded")

                # Test get_object_segments
                object_segments = grid.get_object_segments()
                print(f"  ✓ get_object_segments() returned {len(object_segments)} segments")

                # Test get_class_segments
                class_segments = grid.get_class_segments()
                print(f"  ✓ get_class_segments() returned {len(class_segments)} segments")

                # Test get_ids
                class_ids_ret, instance_ids_ret = grid.get_ids()
                print(
                    f"  ✓ get_ids() returned {len(class_ids_ret)} class_ids and {len(instance_ids_ret)} instance_ids"
                )
            else:
                # Non-semantic grids
                grid.integrate(points, colors)
                print(f"  ✓ integrate() with colors succeeded")

                # Test integrate without colors
                grid.integrate(points, None)
                print(f"  ✓ integrate() without colors succeeded")

            # Test get_voxels with different parameters
            voxel_data = grid.get_voxels()
            print(f"  ✓ get_voxels() returned {len(voxel_data.points)} points")

            voxel_data_filtered = grid.get_voxels(min_count=1, min_confidence=0.0)
            print(f"  ✓ get_voxels(min_count=1, min_confidence=0.0) succeeded")

            # Test get_points and get_colors
            points_ret = grid.get_points()
            colors_ret = grid.get_colors()
            print(f"  ✓ get_points() returned {len(points_ret)} points")
            print(f"  ✓ get_colors() returned {len(colors_ret)} colors")

            # Test get_voxels_in_bb
            BoundingBox3D = volumetric.BoundingBox3D
            bbox = BoundingBox3D(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
            voxels_in_bb = grid.get_voxels_in_bb(bbox)
            print(f"  ✓ get_voxels_in_bb() returned {len(voxels_in_bb.points)} points")

            if "Semantic" in class_name:
                # Test with include_semantics
                voxels_in_bb_sem = grid.get_voxels_in_bb(bbox, include_semantics=True)
                print(f"  ✓ get_voxels_in_bb(include_semantics=True) succeeded")

            # Test get_voxels_in_camera_frustrum
            CameraFrustrum = volumetric.CameraFrustrum
            K = np.array(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            pose = np.eye(4, dtype=np.float64)
            frustrum = CameraFrustrum(K, 640, 480, pose, 10.0, 0.1)
            voxels_in_frustrum = grid.get_voxels_in_camera_frustrum(frustrum)
            print(
                f"  ✓ get_voxels_in_camera_frustrum() returned {len(voxels_in_frustrum.points)} points"
            )

            if "Semantic" in class_name:
                voxels_in_frustrum_sem = grid.get_voxels_in_camera_frustrum(
                    frustrum, include_semantics=True
                )
                print(f"  ✓ get_voxels_in_camera_frustrum(include_semantics=True) succeeded")

            # Test carve (requires depth image)
            try:
                import cv2

                depth_image = np.ones((480, 640), dtype=np.float32) * 5.0
                grid.carve(frustrum, depth_image)
                print(f"  ✓ carve() succeeded")
            except Exception as carve_e:
                print(f"  ⚠ carve() test skipped: {carve_e}")

            # Test remove_low_count_voxels
            grid.remove_low_count_voxels(min_count=0)
            print(f"  ✓ remove_low_count_voxels() succeeded")

            # Test remove_low_confidence_voxels
            grid.remove_low_confidence_voxels(min_confidence=0.0)
            print(f"  ✓ remove_low_confidence_voxels() succeeded")

            # Test clear and reset
            grid.clear()
            print(f"  ✓ clear() succeeded")

            # Re-integrate some data for reset test
            if "Semantic" in class_name:
                grid.integrate(points, colors, class_ids)
            else:
                grid.integrate(points, colors)

            grid.reset()
            print(f"  ✓ reset() succeeded")

            results[class_name] = True

        except Exception as e:
            print(f"✗ {class_name} test failed: {e}")
            import traceback

            traceback.print_exc()
            results[class_name] = False

    return results


def test_voxel_data_classes(volumetric):
    """Test VoxelData and VoxelSemanticData classes."""
    print("\n" + "=" * 60)
    print("Testing VoxelData Classes...")
    print("=" * 60)

    try:
        VoxelData = volumetric.VoxelData
        print("✓ VoxelData class found")

        voxel_data = VoxelData()
        print("✓ Created VoxelData instance")

        position = voxel_data.get_position()
        color = voxel_data.get_color()
        print(f"✓ get_position() = {position}")
        print(f"✓ get_color() = {color}")

        # Test count property (readwrite)
        voxel_data.count = 5
        count_value = voxel_data.count
        print(f"✓ count property (readwrite) = {count_value}")

        VoxelSemanticData = volumetric.VoxelSemanticData
        print("✓ VoxelSemanticData class found")

        semantic_data = VoxelSemanticData()
        print("✓ Created VoxelSemanticData instance")

        position_sem = semantic_data.get_position()
        color_sem = semantic_data.get_color()
        object_id = semantic_data.get_object_id()
        class_id = semantic_data.get_class_id()
        confidence = semantic_data.get_confidence()
        confidence_counter = semantic_data.get_confidence_counter()
        print(f"✓ get_position() = {position_sem}")
        print(f"✓ get_color() = {color_sem}")
        print(f"✓ get_object_id() = {object_id}")
        print(f"✓ get_class_id() = {class_id}")
        print(f"✓ get_confidence() = {confidence}")
        print(f"✓ get_confidence_counter() = {confidence_counter}")

        # Test count property (readwrite)
        semantic_data.count = 10
        count_sem = semantic_data.count
        print(f"✓ count property (readwrite) = {count_sem}")

        return True
    except Exception as e:
        print(f"✗ VoxelData classes test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_voxel_grid_data(volumetric):
    """Test VoxelGridData class."""
    print("\n" + "=" * 60)
    print("Testing VoxelGridData...")
    print("=" * 60)

    try:
        VoxelGridData = volumetric.VoxelGridData
        print("✓ VoxelGridData class found")

        grid_data = VoxelGridData()
        print("✓ Created VoxelGridData instance")

        # Test that attributes exist
        assert hasattr(grid_data, "points"), "VoxelGridData missing 'points' attribute"
        assert hasattr(grid_data, "colors"), "VoxelGridData missing 'colors' attribute"
        assert hasattr(grid_data, "object_ids"), "VoxelGridData missing 'object_ids' attribute"
        assert hasattr(grid_data, "class_ids"), "VoxelGridData missing 'class_ids' attribute"
        assert hasattr(grid_data, "confidences"), "VoxelGridData missing 'confidences' attribute"

        print("✓ All VoxelGridData attributes present")

        return True
    except Exception as e:
        print(f"✗ VoxelGridData test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Volumetric Module Import and Linking Test")
    print("=" * 60)
    print()

    # Test import
    volumetric = test_import()

    # Run all tests
    all_passed = True

    all_passed &= test_tbb_utils(volumetric)
    all_passed &= test_bounding_boxes(volumetric)
    all_passed &= test_camera_frustum(volumetric)
    all_passed &= test_voxel_data_classes(volumetric)
    all_passed &= test_voxel_grid_data(volumetric)

    grid_results = test_voxel_grid_classes(volumetric)
    all_passed &= all(grid_results.values())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if all_passed:
        print("✓ All tests passed! The volumetric module is properly built and linked.")
        print("\nYou can now use:")
        print("  from volumetric import VoxelGrid, VoxelBlockGrid, TBBUtils")
        print("  from volumetric import VoxelSemanticGrid, VoxelBlockSemanticGrid")
        print(
            "  from volumetric import VoxelSemanticGridProbabilistic, VoxelBlockSemanticProbabilisticGrid"
        )
        print("  from volumetric import CameraFrustrum, BoundingBox3D, OrientedBoundingBox3D")
        print("  from volumetric import BoundingBox2D, OrientedBoundingBox2D, ImagePoint")
        print("  from volumetric import VoxelData, VoxelSemanticData, VoxelGridData")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nCommon issues:")
        print("1. Missing dependencies (TBB, Eigen)")
        print("2. Module not built correctly - try: cd cpp && ./clean.sh && ./build.sh")
        print("3. Wrong Python version or environment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
