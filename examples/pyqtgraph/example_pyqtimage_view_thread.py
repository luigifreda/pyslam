import sys
import platform

sys.path.append("../../")
from pyslam.config import Config

import numpy as np
import cv2
import time
import threading
from queue import Queue
import signal

from pyslam.viz.qimage_thread import QimageViewer
from pyslam.utilities.system import set_rlimit


class ImageGenerator:
    def __init__(self, image_queues: list[Queue], control_queue: Queue):
        self.image_queues: list[Queue] = image_queues
        self.control_queue = control_queue
        self.running = True
        self.border_thickness = 10
        self.i = 0
        self.j = 1

    def generate_images(self):
        """Generate images in a separate thread"""
        while self.running:
            try:

                # Create a random RGB image for testing
                random_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
                random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

                # Create second image with border
                random_image2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
                random_image2 = cv2.cvtColor(random_image2, cv2.COLOR_BGR2RGB)  # Convert to RGB
                # Draw a black rectangle along the border
                height, width = random_image2.shape[:2]
                cv2.rectangle(
                    random_image2,
                    (0, 0),
                    (width - 1, height - 1),
                    (0, 0, 0),
                    thickness=self.border_thickness,
                )
                random_image2 = np.tile(random_image2, (self.j, 1, 1))

                # Put images in queue for display thread
                for image_queue in self.image_queues:
                    image_queue.put(
                        {
                            "image1": random_image,
                            "image2": random_image2,
                            "frame_id": self.i,
                        }
                    )

                self.i += 1
                if self.i % 30 == 0:
                    self.j += 1
                    if self.j % 10 == 0:
                        self.j = 1

                time.sleep(0.04)  # Simulate dynamic updates

            except Exception as e:
                print(f"Error in image generation: {e}")
                break

        print("Image generator thread finished")

    def stop(self):
        self.running = False


class ImageDisplayQt:
    def __init__(self, image_queue, control_queue):
        self.image_queue = image_queue
        self.control_queue = control_queue
        self.running = True
        self.viewer = QimageViewer.get_instance()

    def display_images(self):
        """Display images in a separate thread"""
        while self.running:
            try:
                # Get images from queue
                if not self.image_queue.empty():
                    image_data = self.image_queue.get(timeout=0.1)

                    # Display images
                    self.viewer.draw(image_data["image1"], "random image 1 - PyQt")
                    self.viewer.draw(image_data["image2"], "random image 2 - PyQt")

                    # Check for quit key
                    key = self.viewer.get_key()
                    if key and key != chr(0):
                        print("key: ", key)
                        if key == "q":
                            self.viewer.quit()
                            self.control_queue.put("quit")
                            self.running = False
                            break

                time.sleep(0.01)  # Small delay to prevent busy waiting

            except Exception as e:
                if "Empty" not in str(e):  # Ignore empty queue exceptions
                    print(f"Error in image display 1: {e}")
                time.sleep(0.01)

        print("Image display 1 thread finished")

    def stop(self):
        self.running = False


class ImageDisplayCv2:
    def __init__(self, image_queue, control_queue, name: str = ""):
        self.image_queue = image_queue
        self.control_queue = control_queue
        self.running = True
        self.name = name

    def display_images(self):
        """Display images in a separate thread"""
        # On macOS, OpenCV GUI functions must be called from the main thread
        # Check if we're in a worker thread on macOS
        is_main_thread = threading.current_thread() is threading.main_thread()
        is_macos = platform.system() == "Darwin"

        if is_macos and not is_main_thread:
            print(f"Warning: OpenCV GUI functions cannot be called from worker threads on macOS.")
            print(f"Skipping OpenCV display for '{self.name if self.name else 'worker thread'}'.")
            print(f"Use the main thread display or PyQtGraph viewer instead.")
            # Just wait and exit gracefully
            while self.running:
                time.sleep(0.1)
            print(f"Image display 2 thread finished (skipped on macOS)")
            return

        window_name1 = f'random image 1 - Opencv {self.name if self.name else ""}'
        window_name2 = f'random image 2 - Opencv {self.name if self.name else ""}'
        cv2.namedWindow(window_name1)
        cv2.namedWindow(window_name2)
        # Set window properties to ensure it can receive focus
        cv2.setWindowProperty(window_name1, cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty(window_name2, cv2.WND_PROP_TOPMOST, 1)

        while self.running:
            try:
                # Always check for keyboard input first
                key = cv2.waitKey(1) & 0xFF

                # Check for quit key
                if key == ord("q"):
                    self.control_queue.put("quit")
                    self.running = False
                    break

                # Get images from queue if available
                if not self.image_queue.empty():
                    image_data = self.image_queue.get(timeout=0.001)  # Very short timeout
                    cv2.imshow(window_name1, image_data["image1"])
                    cv2.imshow(window_name2, image_data["image2"])

            except Exception as e:
                if "Empty" not in str(e):  # Ignore empty queue exceptions
                    print(f"Error in image display 2: {e}")
                time.sleep(0.01)

        cv2.destroyWindow(window_name1)
        cv2.destroyWindow(window_name2)
        print("Image display 2 thread finished")

    def stop(self):
        self.running = False


class InputHandler:
    def __init__(self, control_queue, processes: list[any] = []):
        self.processes = processes
        self.control_queue = control_queue
        self.running = True

    def stop_processes(self):
        """Stop all processes"""
        if self.processes:
            for process in self.processes:
                if hasattr(process, "stop"):
                    process.stop()

    def handle_input(self):
        """Handle keyboard input in a separate thread"""
        while self.running:
            try:
                # Check control queue for quit signals
                if not self.control_queue.empty():
                    signal = self.control_queue.get_nowait()
                    if signal == "quit":
                        self.running = False
                        break

                # This could be extended to handle other input sources
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("Keyboard interrupt received")
                self.control_queue.put("quit")
                self.running = False
                break

        self.stop_processes()
        print("Input handler thread finished")

    def stop(self):
        self.running = False
        self.stop_processes()


# Example Usage
if __name__ == "__main__":

    use_multiprocessing_init = False

    if use_multiprocessing_init:
        import torch.multiprocessing as mp

        # NOTE: The following set_start_method() is needed by multiprocessing for using CUDA acceleration (for instance with torch).
        if mp.get_start_method() != "spawn":
            mp.set_start_method(
                "spawn", force=True
            )  # NOTE: This may generate some pickling problems with multiprocessing
            #    in combination with torch and we need to check it in other places.
            #    This set start method can be checked with MultiprocessingManager.is_start_method_spawn()

        set_rlimit()

    # Create queues for communication between threads
    image_queue_qt = Queue(maxsize=10)  # Limit queue size to prevent memory issues
    image_queue_cv2 = Queue(maxsize=10)  # Limit queue size to prevent memory issues
    image_queue_cv2_main = Queue(maxsize=10)  # Limit queue size to prevent memory issues
    control_queue = Queue()

    queues = [image_queue_qt, image_queue_cv2, image_queue_cv2_main]

    # Create thread instances
    image_generator = ImageGenerator(queues, control_queue)
    image_display_qt = ImageDisplayQt(image_queue_qt, control_queue)
    image_display_cv2 = ImageDisplayCv2(image_queue_cv2, control_queue)
    image_display_cv2_main = ImageDisplayCv2(image_queue_cv2_main, control_queue, name="main")

    processes = [image_generator, image_display_qt, image_display_cv2, image_display_cv2_main]
    input_handler = InputHandler(control_queue, processes)

    # Create and start threads
    generator_thread = threading.Thread(target=image_generator.generate_images, daemon=True)
    display_thread_qt = threading.Thread(target=image_display_qt.display_images, daemon=True)
    display_thread_cv2 = threading.Thread(target=image_display_cv2.display_images, daemon=True)
    input_thread = threading.Thread(target=input_handler.handle_input, daemon=True)

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down...")
        control_queue.put("quit")
        image_generator.stop()
        image_display_qt.stop()
        image_display_cv2.stop()
        input_handler.stop()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start all threads
        generator_thread.start()
        display_thread_qt.start()
        display_thread_cv2.start()
        input_thread.start()

        print("All threads started. Press 'q' to quit or Ctrl+C to interrupt.")

        image_display_cv2_main.display_images()

        # Wait for threads to complete
        input_thread.join()
        generator_thread.join()
        display_thread_qt.join()
        display_thread_cv2.join()

    except Exception as e:
        print(f"Error in main: {e}")

    print("Done")
