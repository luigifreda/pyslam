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

import torch.multiprocessing as mp
from pyslam.utilities.multi_processing import MultiprocessingManager
import time
import traceback
import cv2
import numpy as np
import queue

from pyslam.utilities.data_management import empty_queue


kSleepTime = 0.04  # [s]


class CvImageViewer:
    _instance = None

    def __init__(self):
        self.is_running = mp.Value("i", 0)  # Flag to manage process lifecycle

        # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        args = (
            self.queue,
            self.key_queue,
            self.is_running,
        )
        self.process = mp.Process(target=self.run, args=args)
        self.process.start()

    # # destructor
    # def __del__(self):
    #     self.quit()

    @classmethod
    def is_running(cls):
        return (cls._instance is not None) and (cls._instance.is_running.value != 0)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def quit(self):
        """Stop the viewer process."""
        if self.is_running.value == 0:
            return
        print(f"CvImageViewer closing...")
        self.is_running.value = 0

        # Send a sentinel value (None) to signal shutdown
        try:
            self.queue.put((None, None), block=False)
        except Exception as e:
            print(f"CvImageViewer: Error sending shutdown signal: {e}")

        # Give the process time to finish its current iteration
        time.sleep(0.5)

        # Join the process with a timeout (longer for spawn method)
        timeout = 5 if mp.get_start_method() != "spawn" else 15
        self.process.join(timeout=timeout)

        if self.process.is_alive():
            print(f"Warning: CvImageViewer process did not terminate in time, forcing kill.")
            try:
                self.process.terminate()
            except Exception as e:
                print(f"CvImageViewer: Error during process termination: {e}")

        # Shut down the manager AFTER the process has exited
        # This prevents the manager from closing queues while the process is still using them
        if hasattr(self, "mp_manager") and self.mp_manager.manager is not None:
            try:
                self.mp_manager.manager.shutdown()
            except Exception as e:
                print(f"CvImageViewer: Warning: Error shutting down manager: {e}")

        print(f"CvImageViewer closed")

    def run(self, queue, key_queue, is_running):
        """Run in a separate process to display images using cv2.imshow."""
        image_map = {}  # name -> window_name
        is_running.value = 1

        try:
            while is_running.value == 1:
                try:
                    # Process images from queue (non-blocking check)
                    # Check queue without blocking, but process all available images
                    should_exit = False
                    while True:
                        # Check if queue has items (non-blocking)
                        try:
                            if queue.empty():
                                break
                            image, name = queue.get_nowait()

                            # Check for sentinel value (None) signaling shutdown
                            if image is None:
                                should_exit = True
                                break  # Exit inner loop

                            try:
                                # Store window name (only once per window)
                                if name not in image_map:
                                    image_map[name] = name

                                # Display the image (already in BGR format from draw())
                                cv2.imshow(name, image)

                            except Exception as e:
                                print(f"CvImageViewer: update_image: encountered exception: {e}")
                                traceback.print_exc()
                        except queue.Empty:
                            # Queue is empty, break inner loop
                            break
                        except Exception as e:
                            # Other queue errors - log and break
                            print(f"CvImageViewer: Queue error: {e}")
                            break

                    # Check if we should exit (sentinel received or is_running is 0)
                    if should_exit or is_running.value == 0:
                        break

                    # Process keyboard events (non-blocking, 1ms wait)
                    # Use a shorter timeout to be more responsive to shutdown
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:  # 255 means no key pressed
                        try:
                            key_queue.put(key, block=False)
                        except:
                            pass

                except Exception as e:
                    print(f"CvImageViewer: Unexpected error: {e}")
                    traceback.print_exc()
                    break

                # Small sleep to prevent high CPU usage, but allow frequent is_running checks
                time.sleep(kSleepTime)

            # end while

            # Cleanup: close all windows
            try:
                empty_queue(queue)
            except Exception as e:
                print(f"CvImageViewer: Error emptying queue: {e}")

            # Destroy all windows
            try:
                for name in image_map.values():
                    try:
                        cv2.destroyWindow(name)
                    except:
                        pass
                # Also call destroyAllWindows as a safety measure
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"CvImageViewer: Error destroying windows: {e}")

        except Exception as e:
            print(f"CvImageViewer: run: encountered exception: {e}")
            traceback.print_exc()

        print(f"CvImageViewer: run: closed")

    def draw(self, image, name, format=None):
        """Queue an image to be displayed in the separate process.

        Args:
            image: numpy array in BGR format (OpenCV standard) or grayscale
            name: window name for display
            format: unused (kept for compatibility)
        """
        if image is not None:
            self.queue.put((image, name))

    def get_key(self):
        """Get the next key from the queue."""
        if not self.key_queue.empty():
            return self.key_queue.get()
        else:
            return ""
