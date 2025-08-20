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

# import multiprocessing as mp
import torch.multiprocessing as mp
from pyslam.utilities.utils_mp import MultiprocessingManager

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import cv2
import time

from pyslam.utilities.utils_sys import Logging, locally_configure_qt_environment
from pyslam.utilities.utils_data import empty_queue


kVerbose = False


class Resizer:
    def __init__(self, window, name, max_width, max_height, screen_width, screen_height):
        self.name = name
        self.window = window
        self.max_width = max_width
        self.max_height = max_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.aspect_ratio = float(self.window.width() / self.window.height())
        self.in_resize_event = False  # Flag to prevent recursive resize events
        self.is_mouse_resizing = False
        self.set_max_size = False
        self.reset_timer = QTimer()
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self.reset_max_size)

    def reset_max_size(self):
        if kVerbose:
            print(f'Resizer: "{self.name}" reset max size')
        self.window.setMaximumSize(self.screen_width, self.screen_height)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_mouse_resizing = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_mouse_resizing = False
        super().mouseReleaseEvent(event)

    # Override the resizeEvent to maintain aspect ratio. Ensures that the width and height are resized proportionally.
    def resizeEvent(self, event):
        if self.in_resize_event:
            return  # Prevent recursive calls
        self.in_resize_event = True

        width = event.size().width()
        height = event.size().height()

        current_size = self.window.size()

        # Calculate the new height based on the fixed aspect ratio
        new_width = width
        new_height = int(width / self.aspect_ratio)

        if new_width > self.max_width:
            new_width = self.max_width
            new_height = int(self.max_width / self.aspect_ratio)

        if new_height > self.max_height:
            new_height = self.max_height
            new_width = int(self.max_height * self.aspect_ratio)

        self.window.resize(new_width, new_height)
        if self.set_max_size:
            # NOTE: this is a kind of HACK and seems necessary to actually get a resized window
            self.window.setFixedWidth(new_width)
            self.window.setFixedHeight(new_height)
            self.set_max_size = False
        self.reset_timer.start(100)  # Restart the timer to remove the fixed size (100ms delay)

        if kVerbose:
            print(
                f'Resizer: "{self.name}" resize event {width}x{height} -> current size: {current_size.width()}x{current_size.height()}, new size: {new_width}x{new_height}, desired aspect ratio: {self.aspect_ratio:.2f}, actual aspect ratio: {new_width/new_height:.2f}'
            )
        self.in_resize_event = False  # Reset flag
        event.accept()


class QimageViewer:
    _instance = None

    def __init__(self):

        self.is_running = mp.Value("i", 1)  # Flag to manage process lifecycle
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()

        self.key = mp.Value("i", 0)

        self.screen_width = None
        self.screen_height = None

        # Start the GUI process
        self.process = mp.Process(
            target=self.run, args=(self.queue, self.key_queue, self.key, self.is_running)
        )
        self.process.start()

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
        print(f"QimageViewer closing...")
        self.is_running.value = 0
        self.process.join(timeout=5)
        if self.process.is_alive():
            print(f"Warning: QimageViewer process did not terminate in time, forcing kill.")
            self.process.terminate()
        print(f"QimageViewer closed")

    def init(self):
        locally_configure_qt_environment()
        self.app = QApplication([])  # Initialize the application
        self.image_map = {}  # name -> (label,window)
        self.key
        self.offset = 0

        # Fetch desktop dimensions
        desktop = self.app.desktop()
        screen_rect = desktop.screenGeometry()
        self.screen_width = screen_rect.width()
        self.screen_height = screen_rect.height()

    def close(self):
        for name, (label, window, resizer) in self.image_map.items():
            window.close()

    def init_window(self, name, width, height):
        window = QMainWindow()
        window.setWindowTitle(name)
        window.setGeometry(10 + self.offset, 10 + self.offset, 200, 200)  # Set a default size
        self.offset += 30

        current_position = window.frameGeometry().topLeft()
        window.setGeometry(
            current_position.x() + self.offset, current_position.y() + self.offset, width, height
        )  # Set a default size

        label = QLabel(window)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)  # Ensure image fits the label size
        window.setCentralWidget(label)

        # Define maximum size for the window (e.g., half the screen size)
        self.max_width = int(self.screen_width * 4.0 / 5)
        self.max_height = int(self.screen_height * 4.0 / 5)

        # Set maximum size constraints
        # window.setMaximumSize(max_width, max_height)
        # label.setMaximumSize(max_width, max_height)

        window.keyPressEvent = self.on_key_press
        window.keyReleaseEvent = self.on_key_release
        window.closeEvent = self.on_close

        resizer = Resizer(
            window, name, self.max_width, self.max_height, self.screen_width, self.screen_height
        )
        window.resizeEvent = resizer.resizeEvent

        window.show()
        return label, window, resizer

    def update_image(self, image, name, format=QImage.Format_BGR888):
        if name not in self.image_map:
            label, window, resizer = self.init_window(name, image.shape[1], image.shape[0])
            self.image_map[name] = (label, window, resizer)
        else:
            label, window, resizer = self.image_map[name]
        height, width, channels = image.shape
        new_aspect_ratio = float(width) / height
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, format)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        window_size = window.size()
        label_size = label.size()
        is_label_size_different = (
            abs(label_size.width() - window_size.width()) > 1e-3
            or abs(label_size.height() - window_size.height()) > 1e-3
        )
        if abs(resizer.aspect_ratio - new_aspect_ratio) > 1e-3 or is_label_size_different:
            if kVerbose:
                print(
                    f"QimageViewer: window {name}:  image size: {width}x{height}, aspect ratio changed from {resizer.aspect_ratio} to {new_aspect_ratio}"
                )
            # Reset size constraints temporarily to prevent resizing event being blocked
            # window.setMaximumSize(self.screen_width, self.screen_height)
            resizer.aspect_ratio = new_aspect_ratio
            resizer.set_max_size = True
            window.resize(width, height)

    def run(self, queue, key_queue, key, is_running):
        self.key = key
        self.key_queue_thread = key_queue
        self.init()
        while is_running.value == 1:
            if not queue.empty():
                image, name, format = queue.get()
                if image is not None:
                    self.update_image(image, name, format)
            self.app.processEvents()  # Process PyQt events
            time.sleep(0.01)  # Prevent high CPU usage
        empty_queue(queue)
        self.close()

    def draw(self, image, name, format=QImage.Format_BGR888):
        if image is not None:
            self.queue.put((image, name, format))

    def on_key_press(self, event):
        key = event.key()
        try:
            self.key.value = ord(event.text())  # Convert to int
            self.key_queue_thread.put(self.key.value)
        except Exception as e:
            print(f"QimageViewer: on_key_press: encountered exception: {e}")
        # if key == Qt.Key_Up:
        #     print('Up arrow key pressed')
        # elif key == Qt.Key_Down:
        #     print('Down arrow key pressed')
        # elif key == Qt.Key_Left:
        #     print('Left arrow key pressed')
        # elif key == Qt.Key_Right:
        #     print('Right arrow key pressed')
        # else:
        #     print(f'Key pressed: {chr(key)}')  # Print the character of the key

    def on_key_release(self, event):
        # self.key.value = 0  # Reset to no key symbol
        pass

    def on_close(self, event):
        self.is_running.value = 0
        print(f"{mp.current_process().name} - QimageViewer closed figure")

    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())
        else:
            return ""
