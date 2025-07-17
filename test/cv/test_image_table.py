import sys
sys.path.append("../../")
from pyslam.config import Config

import numpy as np
import cv2

from pyslam.utilities.utils_img import ImageTable


# Example usage
if __name__ == "__main__":
    # Create an ImageTable instance
    image_table = ImageTable(num_columns=3, resize_scale=0.5)

    # Add some images
    for i in range(6):
        img = np.random.randint(0, 256, (100+i*2, 100+i*3, 3), dtype=np.uint8)
        image_table.add(img)

    # Render the table
    table_image = image_table.render()

    # Display the table image
    cv2.imshow("Image Table", table_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()