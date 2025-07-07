#!/usr/bin/env -S python3 -O

import sys 
import time


sys.path.append("../../")
from pyslam.config import Config

from pyslam.utilities.utils_mt import SimpleTaskTimer


class MyCallback:
    def __init__(self):
        self.time0 = None

    def __call__(self):
        if self.time0 is None:
            self.time0 = time.time()
        print(f"Timer fired at {time.time()-self.time0} seconds")


if __name__ == "__main__":

    my_callback = MyCallback()
    
    # Create a repeating timer that fires every 2 seconds
    timer = SimpleTaskTimer(interval=1, callback=my_callback, single_shot=True)

    # Start the timer
    timer.start()

    # Let the timer run for 10 seconds and then stop it
    time.sleep(10)
    timer.stop()
    
    print('Done')