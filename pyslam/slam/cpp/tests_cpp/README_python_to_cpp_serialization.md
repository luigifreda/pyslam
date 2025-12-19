# Python-to-C++ Map Serialization Tests

This directory contains tests to verify that maps saved from Python can be correctly loaded by C++.

## Test Files

1. **`../tests_py/test_python_to_cpp_map_serialization.py`** - Python script that creates and saves a test map
2. **`test_python_to_cpp_map.cpp`** - C++ test that loads the Python-saved map and verifies correctness

## Running the Tests

### Quick Start (Recommended)

Use the automated test script:

```bash
cd pyslam/slam/cpp/tests_cpp
./run_python_to_cpp_serialization_test.sh
```

This script will:
1. Generate the test map from Python (if not already exists)
2. Build the C++ test (if needed)
3. Run the C++ test
4. Report results

### Manual Steps

If you prefer to run the steps manually:

#### Step 1: Generate the test map (Python)

```bash
cd pyslam/slam/cpp/tests_py
python3 test_python_to_cpp_map_serialization.py
```

This will create a file `test_data/python_saved_map.json` containing a test map with:
- 5 keyframes with descriptors
- 20 map points with descriptors
- Proper ID counters set

#### Step 2: Run the C++ test

```bash
cd pyslam/slam/cpp/tests_cpp
# Build the test (if not already built)
# Then run:
./test_python_to_cpp_map
```

Or use the test runner script:
```bash
./run_all_tests.sh
```

## What the Tests Verify

The C++ test verifies:

1. **Basic loading**: Map can be loaded from JSON without errors
2. **Data integrity**: 
   - Correct number of frames, keyframes, and points
   - Correct max IDs (max_frame_id, max_keyframe_id, max_point_id)
3. **ID counter restoration**:
   - `FrameBase::_id` matches `max_frame_id`
   - `MapPointBase::_id` matches `max_point_id`
4. **Keyframes map**: `keyframes_map` is properly populated
5. **Descriptors**: Keyframes and map points have descriptors loaded
6. **Keypoints**: Keyframes have keypoints loaded
7. **Camera**: Camera data is correctly loaded

## Expected Output

When the test passes, you should see:
```
Testing Python-to-C++ map loading...
Loading map from: ...
Map loaded successfully!
Loaded map statistics:
  Frames: 5
  Keyframes: 5
  Points: 20
  ...
Python-to-C++ map loading tests passed!
```

## Troubleshooting

If the test fails:

1. **File not found**: Make sure you ran the Python script first to generate the test map
2. **Deserialization errors**: Check that the JSON format matches what C++ expects
3. **Missing descriptors**: Verify that descriptors are being saved and loaded correctly
4. **ID counter issues**: Check that `FrameBase::_id` and `MapPointBase::_id` are being restored

## Integration with CI/CD

These tests should be run as part of the build process to ensure Python-C++ interoperability is maintained.

