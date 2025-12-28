"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import sys

import numpy as np
import os
import cv2
from pyslam.utilities.logging import Printer
import traceback

try:
    import ros2_pybindings
    from cv_bridge import CvBridge
except:
    Printer.red(
        "Check ROS2 is (1) installed and (2) sourced, and (3) ros2_pybindings was correctly built"
    )
    Printer.red("Please, double check you have source your ROS2 environment")
    print(traceback.format_exc())
    sys.exit(1)

from collections import defaultdict

from .dataset_types import SensorType, DatasetEnvironmentType, DatasetType
from .dataset import Dataset
from .frame_cache import FrameCache


# class Ros2BagReader:
#     def __init__(self, bag_path, topic_types, sync_slop=0.05):
#         self.bag_path = bag_path
#         self.sync_slop = sync_slop
#         self.topic_types = topic_types
#         self.topics = list(topic_types.keys())
#         self.bridge = CvBridge()

#         self.reader = SequentialReader()
#         storage_options = StorageOptions(uri=self.bag_path, storage_id='sqlite3')
#         converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
#         self.reader.open(storage_options, converter_options)

#         self.reader.set_filter({'topics': self.topics})

#         self.topic_msg_type_map = {
#             topic: self._import_msg_class(msg_type)
#             for topic, msg_type in topic_types.items()
#         }

#         self.cached_msgs = defaultdict(list)
#         self.last_stamp = None

#     def _import_msg_class(self, full_type):
#         module_name, msg_name = full_type.split('/')
#         exec(f"from {module_name}.msg import {msg_name}", globals())
#         return eval(msg_name)

#     def read_step(self):
#         if not self.reader.has_next():
#             return None

#         topic, data, t = self.reader.read_next()
#         msg_type = self.topic_msg_type_map[topic]
#         msg = deserialize_message(data, msg_type)

#         try:
#             stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
#         except AttributeError:
#             stamp = t / 1e9

#         self.cached_msgs[topic].append((stamp, msg))

#         # Try to find a set of messages within sync_slop
#         if all(self.cached_msgs.values()):
#             ref_stamp = min([v[0][0] for v in self.cached_msgs.values()])
#             synced = []
#             for topic in self.topics:
#                 candidates = [m for m in self.cached_msgs[topic] if abs(m[0] - ref_stamp) <= self.sync_slop]
#                 if not candidates:
#                     return None  # No match
#                 synced.append(candidates[0])
#                 self.cached_msgs[topic].remove(candidates[0])
#             return ref_stamp, dict(zip(self.topics, [m[1] for m in synced]))

#         return None


def decode_ros2_depth_image(msg):
    """
    Decode ROS2 depth image message to numpy array.
    Handles different encodings: 32FC1 (float32), 16UC1 (uint16), etc.
    """
    # Validate message is not empty/invalid
    if not msg.encoding or msg.width == 0 or msg.height == 0:
        raise ValueError(
            f"Invalid depth message: encoding='{msg.encoding}', "
            f"width={msg.width}, height={msg.height}, data_size={len(msg.data)}"
        )

    # Check if this is actually a color image (common synchronizer mismatch)
    color_encodings = ["rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8UC3", "8UC4"]
    if (
        msg.encoding.lower() in color_encodings
        or "rgb" in msg.encoding.lower()
        or "bgr" in msg.encoding.lower()
    ):
        raise ValueError(
            f"Received color image encoding '{msg.encoding}' for depth topic. "
            f"This may indicate a synchronizer mismatch. Skipping this frame."
        )

    data_bytes = bytes(msg.data)
    data_size = len(data_bytes)
    expected_pixels = msg.height * msg.width

    if expected_pixels == 0:
        raise ValueError("Invalid depth message: zero dimensions")

    # Determine dtype and bytes per pixel based on encoding
    if msg.encoding == "32FC1":
        dtype = np.float32
        bytes_per_pixel = 4
    elif msg.encoding == "16UC1" or msg.encoding == "16SC1":
        dtype = np.uint16 if msg.encoding == "16UC1" else np.int16
        bytes_per_pixel = 2
    elif msg.encoding == "8UC1":
        dtype = np.uint8
        bytes_per_pixel = 1
    else:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

    expected_bytes = expected_pixels * bytes_per_pixel

    # Validate data size matches expected dimensions
    if data_size != expected_bytes:
        # Try to infer correct dimensions if they don't match
        actual_pixels = data_size // bytes_per_pixel
        if actual_pixels % msg.width == 0:
            # Height might be wrong, recalculate
            inferred_height = actual_pixels // msg.width
            print(
                f"Warning: Depth image size mismatch. Expected {msg.height}x{msg.width} "
                f"({expected_bytes} bytes), got {data_size} bytes. "
                f"Inferring height as {inferred_height}."
            )
            image = np.frombuffer(data_bytes, dtype=dtype).reshape((inferred_height, msg.width))
        elif actual_pixels % msg.height == 0:
            # Width might be wrong, recalculate
            inferred_width = actual_pixels // msg.height
            print(
                f"Warning: Depth image size mismatch. Expected {msg.height}x{msg.width} "
                f"({expected_bytes} bytes), got {data_size} bytes. "
                f"Inferring width as {inferred_width}."
            )
            image = np.frombuffer(data_bytes, dtype=dtype).reshape((msg.height, inferred_width))
        else:
            raise ValueError(
                f"Cannot reshape depth image: data size {data_size} bytes "
                f"({actual_pixels} pixels) doesn't match {msg.height}x{msg.width} "
                f"({expected_pixels} pixels) for encoding {msg.encoding}"
            )
    else:
        # Normal case: dimensions match
        image = np.frombuffer(data_bytes, dtype=dtype).reshape((msg.height, msg.width))

    # Convert to float32 if needed (for consistency)
    if dtype != np.float32:
        image = image.astype(np.float32)

    return image


def validate_image_message(msg):
    """Validate ROS image message dimensions and data size before decoding.
    Returns True if message is valid, False otherwise.
    """
    if not msg or not msg.encoding or msg.width == 0 or msg.height == 0:
        return False

    # Validate image dimensions (max 100000 pixels per dimension)
    MAX_DIMENSION = 100000
    if msg.width > MAX_DIMENSION or msg.height > MAX_DIMENSION:
        print(
            f"[validate_image_message] Invalid dimensions: width={msg.width}, "
            f"height={msg.height}, encoding={msg.encoding}"
        )
        return False

    # Validate step value (max 100MB per row)
    MAX_STEP = 100 * 1024 * 1024  # 100MB
    if msg.step > MAX_STEP:
        print(
            f"[validate_image_message] Invalid step: step={msg.step}, "
            f"width={msg.width}, height={msg.height}"
        )
        return False

    # Validate data size matches expected size
    data_size = len(msg.data)
    expected_size = msg.height * msg.step
    if expected_size == 0:
        print(f"[validate_image_message] Invalid step size: step={msg.step}")
        return False

    # Check total expected memory size (max 1GB)
    MAX_TOTAL_SIZE = 1024 * 1024 * 1024  # 1GB
    if expected_size > MAX_TOTAL_SIZE:
        print(
            f"[validate_image_message] Expected size too large: {expected_size} bytes, "
            f"width={msg.width}, height={msg.height}, step={msg.step}"
        )
        return False

    # Allow some tolerance for data size mismatch, but check for extreme cases
    if data_size < expected_size * 0.9 or data_size > expected_size * 1.1:
        # If the mismatch is extreme (more than 10x), it's likely corrupted
        if data_size > expected_size * 10 or expected_size > data_size * 10:
            print(
                f"[validate_image_message] Data size mismatch: expected={expected_size}, "
                f"actual={data_size}, width={msg.width}, height={msg.height}, "
                f"step={msg.step}, encoding={msg.encoding}"
            )
            return False

    return True


def decode_color_image_sync(color_msg, bridge, convert_color_rgb2bgr=False):
    """Decode a color image message from sync reader (ROS message).

    Args:
        color_msg: ROS sensor_msgs Image message
        bridge: CvBridge instance
        convert_color_rgb2bgr: Whether to apply additional RGB->BGR conversion

    Returns:
        Decoded cv2 image or None on error.
    """
    if not validate_image_message(color_msg):
        print(f"[decode_color_image_sync] Invalid color image message, skipping")
        return None

    try:
        # Try cv_bridge with original encoding first to avoid conversion bug
        # This matches the approach we used in C++ async reader
        if color_msg.encoding == "rgb8" or color_msg.encoding == "RGB8":
            # Get image in original RGB8 encoding, then convert manually
            cv_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
            # Convert RGB to BGR (cv_bridge defaults to BGR output)
            result_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            # Note: convert_color_rgb2bgr flag is not needed here since we already convert RGB->BGR
        elif color_msg.encoding == "bgr8" or color_msg.encoding == "BGR8":
            # Already BGR8, just get it
            result_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            # Note: convert_color_rgb2bgr flag doesn't apply to BGR8
        else:
            # For other encodings, use cv_bridge with desired encoding
            result_img = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            # Apply convert_color_rgb2bgr if needed (for encodings that cv_bridge might not convert properly)
            if convert_color_rgb2bgr:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

        return result_img
    except Exception as e:
        print(f"[decode_color_image_sync] Error decoding color image: {e}")
        print(
            f"  encoding: {color_msg.encoding}, width: {color_msg.width}, "
            f"height: {color_msg.height}, step: {color_msg.step}, "
            f"data_size: {len(color_msg.data)}"
        )
        print(traceback.format_exc())
        return None


def process_color_image_async(
    color_img, convert_color_rgb2bgr=False, debug_read=False, image_name="color"
):
    """Process a color image from async reader (numpy array).

    Args:
        color_img: NumPy array image (already decoded in C++)
        convert_color_rgb2bgr: Whether to apply RGB->BGR conversion
        debug_read: Whether to print debug information
        image_name: Name of the image for debug output

    Returns:
        Processed image or None.
    """
    if color_img is None:
        return None

    if not isinstance(color_img, np.ndarray):
        return None

    result_img = color_img

    # Note: C++ async reader already converts RGB→BGR, so images are already BGR
    # The convert_color_rgb2bgr flag is typically not needed for async reader
    # But we keep it for API consistency (in case C++ behavior changes)
    if convert_color_rgb2bgr and len(color_img.shape) == 3:
        # Additional conversion if requested (though usually redundant)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    if debug_read:
        print(f"[read] {image_name} image shape: {result_img.shape}")

    return result_img


class Ros2bagDataset(Dataset):
    def __init__(
        self,
        path,
        name,
        sensor_type=SensorType.RGBD,
        associations=None,
        start_frame_id=0,
        type=DatasetType.ROS1BAG,
        environment_type=DatasetEnvironmentType.INDOOR,
        fps=30,
        config=None,
    ):
        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        ros_settings = config.ros_settings
        assert ros_settings is not None
        self.ros_settings = ros_settings

        assert "bag_path" in ros_settings
        self.bag_path = ros_settings["bag_path"]
        if not os.path.exists(self.bag_path):
            raise ValueError(f"[Ros2bagDataset] Bag path {self.bag_path} does not exist")

        self.topics_dict = ros_settings["topics"]
        self.topics_list = list(self.topics_dict.values())
        self.convert_color_rgb2bgr = ros_settings.get("convert_color_rgb2bgr", False)

        print(f"[Ros2bagDataset]: Bag: {self.bag_path}, Topics: {self.topics_dict}")

        self.sync_queue_size = int(self.ros_settings["sync_queue_size"])
        self.sync_slop = float(self.ros_settings["sync_slop"])
        self.max_read_ahead = int(self.ros_settings.get("max_read_ahead", 20))
        self.depth_factor = float(self.ros_settings.get("depth_factor", 1.0))

        print(
            f"[Ros2bagDataset] sync_queue_size: {self.sync_queue_size}, sync_slop: {self.sync_slop}, "
            f"max_read_ahead: {self.max_read_ahead}"
        )

        self.color_image_topic = self.topics_dict.get("color_image")
        self.depth_image_topic = self.topics_dict.get("depth_image")
        self.right_color_image_topic = self.topics_dict.get("right_color_image")
        self.imu_topic = self.topics_dict.get("imu")

        assert self.color_image_topic is not None, "Color image topic not found"
        if self.sensor_type == SensorType.STEREO:
            assert self.right_color_image_topic is not None, "Right color image topic not found"
        if self.sensor_type == SensorType.RGBD:
            assert self.depth_image_topic is not None, "Depth image topic not found"

        self.use_async_reader = True  # clearly, async reader is much faster than sync reader
        if self.use_async_reader:
            ros2_bag_reader_type = ros2_pybindings.Ros2BagAsyncReaderATS
        else:
            ros2_bag_reader_type = ros2_pybindings.Ros2BagSyncReaderATS

        self.reader = ros2_bag_reader_type(
            bag_path=self.bag_path,
            topics=self.topics_list,
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
            max_read_ahead=self.max_read_ahead,
        )
        self.topic_timestamps = self.reader.topic_timestamps
        self.topic_counts = self.reader.topic_counts
        self.bridge = CvBridge()

        self.environment_type = environment_type
        self.Ts = 1.0 / fps
        self.scale_viewer_3d = 0.1 if sensor_type != SensorType.MONOCULAR else 0.05

        print("Processing ROS2 bag")
        self.num_frames = self.topic_counts[self.color_image_topic]
        self.max_frame_id = self.num_frames
        print(f"Number of frames: {self.num_frames}")

        num_timestamps = len(self.topic_timestamps[self.color_image_topic])
        print(f"Number of timestamps: {num_timestamps}")
        assert num_timestamps == self.num_frames

        self.cam_stereo_settings = config.cam_stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow("[Ros2bagDataset] automatically rectifying the stereo images")
            if self.cam_stereo_settings is None:
                sys.exit("ERROR: Missing stereo settings in YAML config!")
            width = config.cam_settings["Camera.width"]
            height = config.cam_settings["Camera.height"]

            K_l, D_l, R_l, P_l = self.cam_stereo_settings["left"].values()
            K_r, D_r, R_r, P_r = self.cam_stereo_settings["right"].values()

            self.M1l, self.M2l = cv2.initUndistortRectifyMap(
                K_l, D_l, R_l, P_l[:3, :3], (width, height), cv2.CV_32FC1
            )
            self.M1r, self.M2r = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r[:3, :3], (width, height), cv2.CV_32FC1
            )
        self.debug_rectification = False
        # Debug flag to enable verbose output (disabled by default for performance)
        self.debug_read = False

        self.count = -1
        self.color_img = None
        self.depth_img = None
        self.right_color_img = None
        self._prev_timestamp = (
            None  # Previous timestamp for duration calculation (both async and sync readers)
        )
        self._next_timestamp_frame_id = None  # Track which frame_id _next_timestamp is for

        # Frame cache for decoded images to avoid re-reading (optional, can be enabled for random access)
        # Disabled by default for sequential playback (faster, lower memory)
        # Enable if you need random access or to revisit previous frames
        self.frame_cache = FrameCache(enabled=False)

    def read(self):
        # The synchronizer may need multiple messages before finding a match
        result = None
        max_reads = 100  # Limit to prevent infinite loops
        read_count = 0
        while result is None and not self.reader.is_eof() and read_count < max_reads:
            result = self.reader.read_step()
            read_count += 1

        # Debug: check if we're hitting EOF immediately
        if result is None and self.reader.is_eof():
            if self.debug_read:
                print(f"[read] EOF reached after {read_count} reads, count={self.count}")

        if result is not None:
            ts, synced = result

            # Set _next_timestamp for the previous frame (if it exists)
            # This works for both async and sync readers since both return synchronized timestamps
            # The synchronized timestamp is the correct one to use, not topic_timestamps which
            # may not match if frames were skipped during synchronization
            if self._prev_timestamp is not None:
                # Set next timestamp for the previous frame (which was at _prev_timestamp)
                # The previous frame's frame_id is self.count (before increment)
                self._next_timestamp = ts
                self._next_timestamp_frame_id = self.count  # This is for the previous frame

            # Set the current frame's timestamp
            self._timestamp = ts

            # Check if this is the last frame
            is_last_frame = self.reader.is_eof()

            try:
                if self.use_async_reader:
                    self._read_async(synced)
                else:
                    self._read_sync(synced)
            except Exception as e:
                print(f"Error processing synced messages: {e}")
                print(traceback.format_exc())
                self.color_img = None
                self.depth_img = None
                self.right_color_img = None
                self.is_ok = False
                return

            # Cache the frame if caching is enabled
            if self.frame_cache.enabled:
                frame_id = self.count + 1  # Next frame_id after increment
                self.frame_cache.store(
                    frame_id=frame_id,
                    color_img=self.color_img,
                    depth_img=self.depth_img,
                    right_color_img=self.right_color_img,
                    timestamp=ts,
                )

            self.count += 1

            # Store current synchronized timestamp for next iteration
            # This will be used to set _next_timestamp when we read the next frame
            # This works for both async and sync readers since both return synchronized timestamps
            self._prev_timestamp = ts

            # For the last frame, set _next_timestamp for the current frame using fallback
            if is_last_frame:
                # This is the last frame, so set _next_timestamp for it (current frame)
                # Use current timestamp + Ts as fallback since there's no next frame
                self._next_timestamp = ts + self.Ts
                self._next_timestamp_frame_id = self.count  # This is for the current frame

            self.is_ok = True
        else:
            self._timestamp = None
            self.color_img = None
            self.depth_img = None
            self.right_color_img = None
            self.is_ok = False

    def _read_async(self, synced):
        """Process synced messages from async reader (decoded numpy arrays)."""
        # Async reader returns decoded numpy arrays directly
        # synced is a dict mapping topic -> numpy array
        self.color_img = process_color_image_async(
            synced.get(self.color_image_topic),
            convert_color_rgb2bgr=self.convert_color_rgb2bgr,
            debug_read=self.debug_read,
            image_name="color",
        )

        depth_img = synced.get(self.depth_image_topic)
        if depth_img is not None:
            try:
                if isinstance(depth_img, np.ndarray):
                    self.depth_img = depth_img.astype(np.float32)
                    if self.depth_factor != 1.0:
                        self.depth_img *= self.depth_factor
                    if self.debug_read:
                        print(
                            f"[read] depth image shape: {self.depth_img.shape}, type: {self.depth_img.dtype}"
                        )
                else:
                    self.depth_img = None
            except Exception as e:
                print(f"Error processing depth image: {e}")
                print(traceback.format_exc())
                self.depth_img = None
        else:
            self.depth_img = None

        self.right_color_img = process_color_image_async(
            synced.get(self.right_color_image_topic),
            convert_color_rgb2bgr=self.convert_color_rgb2bgr,
            debug_read=self.debug_read,
            image_name="right color",
        )

    def _read_sync(self, synced):
        """Process synced messages from sync reader (ROS messages that need decoding)."""
        # Sync reader returns ROS messages that need decoding
        # Optimize: use .get() to avoid double dictionary lookup
        color_msg = synced.get(self.color_image_topic)
        if color_msg is not None:
            self.color_img = decode_color_image_sync(
                color_msg, bridge=self.bridge, convert_color_rgb2bgr=self.convert_color_rgb2bgr
            )
            if self.color_img is not None and self.debug_read:
                print(f"[read] color image shape: {self.color_img.shape}")
        else:
            self.color_img = None

        depth_msg = synced.get(self.depth_image_topic)
        if depth_msg is not None:
            try:
                # Debug: print message properties on first error
                if self.debug_read or self.count == 0:
                    print(
                        f"[read] depth encoding: {depth_msg.encoding}, "
                        f"width: {depth_msg.width}, height: {depth_msg.height}, "
                        f"step: {depth_msg.step}, data_size: {len(depth_msg.data)}"
                    )
                self.depth_img = decode_ros2_depth_image(depth_msg)
                if self.depth_factor != 1.0:
                    self.depth_img *= self.depth_factor
                if self.debug_read:
                    print(
                        f"[read] depth image shape: {self.depth_img.shape}, type: {self.depth_img.dtype}"
                    )
            except Exception as e:
                print(f"Error reading depth image: {e}")
                print(
                    f"Depth msg properties: encoding={depth_msg.encoding}, "
                    f"width={depth_msg.width}, height={depth_msg.height}, "
                    f"step={depth_msg.step}, data_size={len(depth_msg.data)}"
                )
                print(traceback.format_exc())
                self.depth_img = None
        else:
            self.depth_img = None

        right_msg = synced.get(self.right_color_image_topic)
        if right_msg is not None:
            self.right_color_img = decode_color_image_sync(
                right_msg, bridge=self.bridge, convert_color_rgb2bgr=self.convert_color_rgb2bgr
            )
            if self.right_color_img is not None and self.debug_read:
                print(f"[read] right color image shape: {self.right_color_img.shape}")
        else:
            self.right_color_img = None

    def getImage(self, frame_id):
        if frame_id < self.max_frame_id:
            # Check cache first if enabled
            cached_result = self.frame_cache.get_color(frame_id)
            if cached_result is not None:
                img, cached_ts = cached_result
                self._timestamp = cached_ts
                if self.sensor_type == SensorType.STEREO:
                    img = cv2.remap(img, self.M1l, self.M2l, cv2.INTER_LINEAR)
                # _next_timestamp should already be set from read() for both async and sync readers
                # Both use synchronized timestamps now
                self.is_ok = True
                return img

            # Sequential read if not cached
            # Read up to frame_id to get the current frame
            while self.count < frame_id and self.is_ok:
                self.read()

            img = self.color_img
            if self.sensor_type == SensorType.STEREO and img is not None:
                img = cv2.remap(img, self.M1l, self.M2l, cv2.INTER_LINEAR)
            self.is_ok = img is not None

            # _next_timestamp is set correctly in read() when we read the next frame
            # For sequential access, when we read frame N+1, _next_timestamp is set for frame N
            # Check if _next_timestamp is for the current frame or a previous frame
            if (
                self._next_timestamp is None
                or self._next_timestamp == self._timestamp
                or self._next_timestamp_frame_id != frame_id
            ):
                # _next_timestamp is not set for the current frame yet
                # This means we haven't read the next frame, so use fallback
                # Use fallback: assume constant frame rate
                self._next_timestamp = (
                    self._timestamp + self.Ts if self._timestamp is not None else None
                )
                self._next_timestamp_frame_id = frame_id  # Mark it as set for this frame
            return img
        self.is_ok = False
        self._timestamp = None
        return None

    def getImageRight(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None
        if frame_id < self.max_frame_id:
            # Check cache first if enabled
            cached_result = self.frame_cache.get_right(frame_id)
            if cached_result is not None:
                img, cached_ts = cached_result
                self._timestamp = cached_ts
                if self.sensor_type == SensorType.STEREO:
                    img = cv2.remap(img, self.M1r, self.M2r, cv2.INTER_LINEAR)
                # _next_timestamp should already be set from read() for both async and sync readers
                # Both use synchronized timestamps now
                self.is_ok = True
                return img

            # Sequential read if not cached
            while self.count < frame_id and self.is_ok:
                self.read()
            img = self.right_color_img
            if self.sensor_type == SensorType.STEREO and img is not None:
                img = cv2.remap(img, self.M1r, self.M2r, cv2.INTER_LINEAR)
            self.is_ok = img is not None
            # _next_timestamp is already set correctly in read() for both async and sync readers
            # Both use synchronized timestamps now
            return img
        self.is_ok = False
        self._timestamp = None
        return None

    def getDepth(self, frame_id):
        if self.sensor_type != SensorType.RGBD:
            return None
        frame_id += self.start_frame_id
        if frame_id < self.max_frame_id:
            # Check cache first if enabled
            cached_result = self.frame_cache.get_depth(frame_id)
            if cached_result is not None:
                img, cached_ts = cached_result
                self._timestamp = cached_ts
                # _next_timestamp should already be set from read() for both async and sync readers
                # Both use synchronized timestamps now
                self.is_ok = True
                return img

            # Sequential read if not cached
            while self.count < frame_id and self.is_ok:
                self.read()
            img = self.depth_img
            self.is_ok = img is not None
            # _next_timestamp is already set correctly in read() for both async and sync readers
            # Both use synchronized timestamps now
            return img
        self.is_ok = False
        self._timestamp = None
        return None
