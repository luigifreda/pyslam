import sys 
import cv2
from datetime import datetime
from webcam import Webcam

import sys
sys.path.append("../")

from pyslam.utilities.timer import Timer

# CHESSBOARD SIZE
chessboard_size = (11,7)

# grab an image every 
kSaveImageDeltaTime = 1  # second

if __name__ == "__main__":

    camera_num = 0
    if len(sys.argv) == 2:
            camera_num = int(sys.argv[1])
    print('opening camera: ', camera_num)

    webcam = Webcam(camera_num)
    webcam.start()
    
    timer = Timer()
    lastSaveTime = timer.elapsed()
 
    while True:
        
        # get image from webcam
        image = webcam.get_current_frame()
        if image is not None: 

            # check if pattern found
            ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), chessboard_size, None)
        
            if ret == True:     
                print('found chessboard')
                # save image
                filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.bmp'
                image_path="./calib_images/" + filename
                
                elapsedTimeSinceLastSave = timer.elapsed() - lastSaveTime
                do_save = elapsedTimeSinceLastSave > kSaveImageDeltaTime
                
                if do_save:
                    lastSaveTime = timer.elapsed()
                    print('saving file ', image_path)
                    cv2.imwrite(image_path, image)

                # draw the corners
                image = cv2.drawChessboardCorners(image, chessboard_size, corners, ret)                       

            cv2.imshow('camera', image)                

        else: 
            pass
            #print('empty image')                
                            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        
    #webcam.quit()            
