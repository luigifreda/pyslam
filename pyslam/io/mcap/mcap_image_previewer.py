"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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

import cv2
import numpy as np
import time
import datetime
import traceback
from typing import Optional

from pyslam.io.mcap.mcap_reader import McapReader
from pyslam.utilities.img_management import ImgWriter


class McapImagePreviewer:
    """
    Preview and visualize image messages from MCAP files with synchronized playback.
    It expects the messages to be ROS2 messages.
    """

    def __init__(
        self, path: str, selected_topics: Optional[list[str]] = None, verbose: bool = True
    ):
        """
        Initialize the MCAP image previewer.

        Args:
            path: Path to MCAP file or directory containing MCAP files
            selected_topics: Optional list of topic names to visualize
        """
        self.mcap_reader = McapReader(path, selected_topics=selected_topics)
        self.selected_topics = selected_topics
        self.img_writer = ImgWriter(
            font_scale=0.7, font_color=(0, 255, 0), font_thickness=1, font_line_type=cv2.LINE_AA
        )
        self.topic_to_window_name = {}
        self.initialized_windows = set()  # Track windows that have been sized
        self.verbose = verbose

    def set_selected_topics(self, selected_topics: list[str]):
        """Update the list of topics to visualize."""
        self.selected_topics = selected_topics

    def visualize_images(self):
        """
        Main entry point for visualizing images from MCAP files.
        Orchestrates the entire playback process.
        """
        self._print_summary()
        messages = self._collect_image_messages()
        if not messages:
            return

        unique_sorted_topics = self._setup_windows(messages)
        self._playback_images(messages)

    def _print_summary(self):
        """Print MCAP file summary information."""
        print("Getting MCAP information...")
        summary = self.mcap_reader.get_summary()
        print(summary)

    def _extract_msg_timestamp(self, msg) -> float:
        """
        Extract timestamp from ROS2 message header.

        Args:
            msg: ROS2 message with header.stamp

        Returns:
            Timestamp in seconds as float
        """
        return msg.ros_msg.header.stamp.sec + msg.ros_msg.header.stamp.nanosec * 1e-9

    def _collect_image_messages(self) -> list:
        """
        Collect and sort image messages from MCAP files.

        Returns:
            Sorted list of image messages, or empty list if none found
        """
        print("Collecting image messages from MCAP...")
        messages_iter = self.mcap_reader.iter_ros2(selected_topics=self.selected_topics)
        messages = []
        for path, msg in messages_iter:
            schema_name = getattr(msg.schema, "name", "")
            if "sensor_msgs/msg/Image" in schema_name or schema_name.endswith("/Image"):
                messages.append(msg)

        if not messages:
            print("No image messages found in the selected topics.")
            return []

        messages.sort(key=lambda msg: self._extract_msg_timestamp(msg))
        return messages

    def _setup_windows(self, messages: list) -> list[str]:
        """
        Create OpenCV windows for each unique topic.

        Args:
            messages: List of image messages

        Returns:
            Sorted list of unique topic names
        """
        unique_sorted_topics = sorted(set(msg.channel.topic for msg in messages))
        for topic in unique_sorted_topics:
            window_name = f"McapImagePreviewer - {topic}"
            self.topic_to_window_name[topic] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
        return unique_sorted_topics

    def _process_image(
        self, img: np.ndarray, topic: str, timestamp: float, encoding: str = ""
    ) -> np.ndarray:
        """
        Process an image for display: normalize, apply colormap, and add text overlay.

        Args:
            img: Input image array
            topic: Topic name to display
            timestamp: Timestamp to display
            encoding: Image encoding (e.g., 'rgb8', 'bgr8', '32FC1')

        Returns:
            Processed image ready for display
        """
        img_out = img.copy()
        encoding_lower = encoding.lower()

        # Normalize float32 depth images to 8-bit
        if img_out.dtype == np.float32:
            # Handle depth images (32FC1) - filter out invalid values (0, inf, nan)
            valid_mask = np.isfinite(img_out) & (img_out > 0)
            if np.any(valid_mask):
                # Normalize valid depth values to 0-255 range
                valid_values = img_out[valid_mask]
                min_val, max_val = valid_values.min(), valid_values.max()
                if max_val > min_val:
                    # Normalize to 0-255
                    img_out = np.clip((img_out - min_val) / (max_val - min_val) * 255, 0, 255)
                else:
                    # All values are the same
                    img_out = np.zeros_like(img_out)
                img_out = img_out.astype(np.uint8)
                # Set invalid pixels to black
                img_out[~valid_mask] = 0
            else:
                # No valid depth values
                img_out = np.zeros_like(img_out, dtype=np.uint8)

            # Apply colormap to grayscale images (depth images)
            if len(img_out.shape) == 2:
                img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)

        # Normalize 16-bit images to 8-bit
        elif img_out.dtype == np.uint16:
            norm = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX)
            img_out = norm.astype(np.uint8)

            # Apply colormap to grayscale images
            if len(img_out.shape) == 2:
                img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)

        # Handle RGB/BGR color images: convert based on encoding
        elif img_out.dtype == np.uint8 and len(img_out.shape) == 3 and img_out.shape[2] == 3:
            # RGB images from ROS2 are stored as RGB, but OpenCV expects BGR
            if encoding_lower == "rgb8":
                # Convert RGB to BGR for proper display
                img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            elif encoding_lower == "bgr8":
                # Already BGR, no conversion needed
                pass
            else:
                # Unknown encoding, assume RGB and convert to BGR
                img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

        # Add text overlay with topic and timestamp
        ts_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
        self.img_writer.write(img_out, f"{ts_str} - {topic}", (10, 30))

        return img_out

    def _synchronize_playback(
        self, msg_timestamp: float, start_time: float, wall_time_start: float
    ) -> None:
        """
        Synchronize playback timing to match original message timestamps.

        Args:
            msg_timestamp: Current message timestamp
            start_time: First message timestamp (reference)
            wall_time_start: Wall clock time when playback started
        """
        target_wall_time = wall_time_start + (msg_timestamp - start_time)
        sleep_time = target_wall_time - time.time()

        if sleep_time > 0.001:
            time.sleep(sleep_time)
        elif sleep_time < -0.5:
            print(f"McapImagePreviewer: lagging behind by {-sleep_time:.2f}s")

    def _handle_user_input(self) -> tuple[bool, float]:
        """
        Handle keyboard input from user.

        Returns:
            Tuple of (should_quit, pause_duration)
            pause_duration is the time spent paused in seconds
        """
        key = cv2.waitKey(1)
        should_quit = key & 0xFF in [ord("q"), 27]
        was_paused = key & 0xFF == ord("p")

        pause_duration = 0.0
        if should_quit:
            print("McapImagePreviewer: quitting.")
        elif was_paused:
            print("McapImagePreviewer: paused. Press any key to resume.")
            pause_start = time.time()
            cv2.waitKey(-1)
            pause_duration = time.time() - pause_start

        return should_quit, pause_duration

    def _playback_images(self, messages: list) -> None:
        """
        Play back images with synchronized timing and user interaction.

        Args:
            messages: Sorted list of image messages
        """
        start_msg_timestamp = self._extract_msg_timestamp(messages[0])
        wall_time_start = time.time()

        for msg in messages:
            try:
                msg_timestamp = self._extract_msg_timestamp(msg)
                topic = msg.channel.topic

                # Decode and process image
                img = self.mcap_reader.ros2_decoder.decode_image(msg.ros_msg)
                # Get encoding from message for proper color space handling
                encoding = getattr(msg.ros_msg, "encoding", "")
                img_out = self._process_image(img, topic, msg_timestamp, encoding)

                # Synchronize playback timing
                self._synchronize_playback(msg_timestamp, start_msg_timestamp, wall_time_start)

                if self.verbose:
                    print(
                        f"McapImagePreviewer: displaying image {img.shape}, type {img.dtype} message from topic '{topic}'\n"
                        f"\t msg timestamp {msg_timestamp:.3f}s\n"
                        f"\t wall time: {time.time() - wall_time_start:.3f}s"
                    )

                # Resize window to image size on first display for this topic
                window_name = self.topic_to_window_name[topic]
                if window_name not in self.initialized_windows:
                    height, width = img_out.shape[:2]
                    cv2.resizeWindow(window_name, width, height)
                    self.initialized_windows.add(window_name)

                # Display image
                cv2.imshow(window_name, img_out)

                # Handle user input (check before processing to allow immediate quit)
                should_quit, pause_duration = self._handle_user_input()
                if should_quit:
                    break

                # Adjust wall_time_start to account for pause time to maintain synchronization
                if pause_duration > 0:
                    wall_time_start += pause_duration

            except Exception as e:
                print(f"McapImagePreviewer: failed to display image from {msg.channel.topic}: {e}")
                traceback.print_exc()

        cv2.destroyAllWindows()
        print("McapImagePreviewer: playback finished.")
