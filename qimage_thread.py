#import multiprocessing as mp 
import torch.multiprocessing as mp
from utils_mp import MultiprocessingManager

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2
import time

from utils_sys import Logging, locally_configure_qt_environment

class Resizer:
    def __init__(self, window, name):
        self.name = name
        self.window = window
        self.aspect_ratio = self.window.width() / self.window.height()
        
    def resizeEvent(self, event):
        """
        Override the resizeEvent to maintain aspect ratio.
        Ensures that the width and height are resized proportionally.
        """
        width = event.size().width()
        height = event.size().height()     
        
        # Calculate the new height based on the fixed aspect ratio
        new_height = int(width / self.aspect_ratio)
        
        if new_height != height:
            # Set the new height based on the calculated value while keeping the width fixed
            self.window.setFixedHeight(new_height)
        event.accept()    


class QimageViewer:
    _instance = None    
    def __init__(self):
        
        self.is_running = mp.Value('i', 1)  # Flag to manage process lifecycle
        self.mp_manager = MultiprocessingManager()
        self.queue = self.mp_manager.Queue()
        self.key_queue = self.mp_manager.Queue()
               
        self.key = mp.Value('i', 0)
                        
        # Start the GUI process
        self.process = mp.Process(target=self.run, args=(self.queue, self.key_queue, self.key, self.is_running))
        self.process.start()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def quit(self):
        """Stop the viewer process."""
        print(f'QimageViewer closing...')
        self.is_running.value = 0
        self.process.join(timeout=5)
        if self.process.is_alive():
            print(f"Warning: QimageViewer process did not terminate in time, forcing kill.")
        self.process.terminate()
        print(f'QimageViewer closed')

    def init(self):
        locally_configure_qt_environment()
        self.app = QApplication([])  # Initialize the application
        self.image_map = {}  # name -> (label,window)
        self.key 
        self.offset = 0
        
    def close(self):
        for name, (label, window) in self.image_map.items():
            window.close()

    def init_window(self, name, width, height):
        window = QMainWindow()        
        window.setWindowTitle(name)
        window.setGeometry(10+self.offset, 10+self.offset, 200, 200)  # Set a default size
        self.offset += 30
        
        current_position = window.frameGeometry().topLeft()
        window.setGeometry(current_position.x()+self.offset, current_position.y()+self.offset, width, height)  # Set a default size        
        
        label = QLabel(window)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)  # Ensure image fits the label size
        window.setCentralWidget(label)
        
        window.keyPressEvent = self.on_key_press
        window.keyReleaseEvent = self.on_key_release
        window.closeEvent = self.on_close
        
        resizer = Resizer(window, name)
        window.resizeEvent = resizer.resizeEvent
        
        window.show()
        return label, window
        
    def update_image(self, image, name, format=QImage.Format_BGR888):
        if name not in self.image_map:
            label, window = self.init_window(name, image.shape[1], image.shape[0])
            self.image_map[name] = (label, window)
        else: 
            label, window = self.image_map[name]
        """Convert the numpy image to QPixmap and display it."""
        height, width, channels = image.shape
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, format)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
                
    def run(self, queue, key_queue, key, is_running):
        """Run the PyQt application in a separate process."""
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

        self.close()

    def draw(self, image, name, format=QImage.Format_BGR888):
        """Add an image to the queue."""
        if image is not None:
            self.queue.put((image,name,format))

    def on_key_press(self, event):
        key = event.key()
        try:
            self.key.value = ord(event.text())  # Convert to int 
            self.key_queue_thread.put(self.key.value)                 
        except Exception as e:
            print(f'QimageViewer: on_key_press: encountered exception: {e}')            
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
        #self.key.value = 0  # Reset to no key symbol
        pass
    
    def on_close(self, event):
        self.is_running.value = 0
        print(f"{mp.current_process().name} - QimageViewer closed figure")
     
    def get_key(self):
        if not self.key_queue.empty():
            return chr(self.key_queue.get())                
        else:
            return ''