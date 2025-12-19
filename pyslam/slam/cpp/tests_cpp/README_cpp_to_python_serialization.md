# C++-to-Python Map Serialization Test

This test verifies that maps saved from C++ can be correctly loaded by Python, ensuring bidirectional interoperability between C++ and Python serialization.

## Overview

The test consists of:
1. **C++ Test** (`test_cpp_to_python_map.cpp`): Generates a test map with keyframes, map points, descriptors, images, and associations, then saves it to JSON.
2. **Python Test** (`test_cpp_to_python_map_serialization.py`): Loads the C++-saved map and verifies all data is correctly deserialized.

## Running the Test

### Automated Script (Recommended)

Simply run the bash script:

```bash
cd pyslam/slam/cpp/tests_cpp
./run_cpp_to_python_serialization_test.sh
```

This script will:
1. Generate a test map from C++
2. Build the C++ test (if needed)
3. Run the Python test to verify the map can be loaded

### Manual Steps

1. **Build the C++ test:**
   ```bash
   cd pyslam/slam/cpp/build
   cmake --build . --target test_cpp_to_python_map
   ```

2. **Generate the test map:**
   ```bash
   cd pyslam/slam/cpp/build/tests_cpp
   ./test_cpp_to_python_map
   ```

3. **Run the Python test:**
   ```bash
   cd pyslam/slam/cpp/tests_py
   python3 test_cpp_to_python_map_serialization.py
   ```

## Test Data

The test map is saved to:
- `pyslam/slam/cpp/tests_py/test_data/cpp_saved_map.json`

## What the Test Verifies

The test checks:

1. **Basic Map Properties:**
   - Correct number of frames, keyframes, and map points
   - Correct max IDs (frame_id, keyframe_id, point_id)
   - Static ID counters (FrameBase._id, MapPointBase._id) are restored

2. **Descriptors:**
   - Keyframes have descriptors loaded
   - Descriptors have correct type (uint8) and dimensions (N x 32 for ORB)
   - Descriptor count matches keypoint count
   - Map points have descriptors loaded

3. **Images:**
   - Keyframes have images loaded (required for `compute_frame_matches`)

4. **Map Point Associations:**
   - Keyframes have `points` arrays populated
   - Map points can be retrieved from keyframes at keypoint indices
   - This is critical for `prepare_input_data_for_pnpsolver` to work

5. **Observations:**
   - Map points have observations linked to keyframes (bidirectional relationships)

6. **Relocalization Prerequisites:**
   - All prerequisites for successful relocalization are met:
     - Valid descriptors
     - Matching descriptor/keypoint counts
     - Keypoints present
     - Images present
     - Good keyframes (not marked as bad)
     - Map point associations working

## Expected Output

On success, you should see:
```
✓ All tests passed!
ℹ C++-to-Python map serialization is working correctly.
```

## Troubleshooting

### Test map file not found
- Make sure you've run the C++ test first to generate the map
- Check that the test data directory exists: `pyslam/slam/cpp/tests_py/test_data/`

### C++ test executable not found
- Build the C++ module first: `cd pyslam/slam/cpp && ./build.sh`
- Or build just the test: `cd pyslam/slam/cpp/build && cmake --build . --target test_cpp_to_python_map`

### Python import errors
- Make sure you're in the correct Python environment
- Install required dependencies: `pip install numpy`

## Related Tests

- **Python-to-C++ Test**: See `README_python_to_cpp_serialization.md` for the reverse direction test

