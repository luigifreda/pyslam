import sys
sys.path.append('../../')

import numpy as np
import cv2  
import time

from qimage_thread import QimageViewer

# Example Usage
if __name__ == "__main__":
    viewer = QimageViewer.get_instance()
    do_loop = True
    
    while do_loop:
        # Create a random RGB image for testing
        random_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        random_image = cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        viewer.draw(random_image, "random image 1")
        
        random_image2 = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        random_image2 = cv2.cvtColor(random_image2, cv2.COLOR_BGR2RGB)  # Convert to RGB
        viewer.draw(random_image2, "random image 2")            
        
        time.sleep(0.04)  # Simulate dynamic updates
        
        key = viewer.get_key()

        if key and key != chr(0):
            print('key: ', key)
            if key == 'q':
                viewer.quit()
                do_loop = False
