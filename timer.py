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

import cv2
from moving_average import MovingAverage
from utils import Printer

#timer_print = print 
timer_print = Printer.cyan 

class Timer: 
    def __init__(self, name = '', is_verbose = False):
        self._name = name 
        self._is_verbose = is_verbose
        self._is_paused = False 
        self._start_time = None 
        self._accumulated = 0 
        self._elapsed = 0         
        self.start()

    def start(self):
        self._accumulated = 0         
        self._start_time = cv2.getTickCount()

    def pause(self): 
        now_time = cv2.getTickCount()
        self._accumulated += (now_time - self._start_time)/cv2.getTickFrequency() 
        self._is_paused = True   

    def resume(self): 
        if self._is_paused: # considered only if paused 
            self._start_time = cv2.getTickCount()
            self._is_paused = False                      

    def elapsed(self):
        if self._is_paused:
            self._elapsed = self._accumulated
        else:
            now = cv2.getTickCount()
            self._elapsed = self._accumulated + (now - self._start_time)/cv2.getTickFrequency()        
        if self._is_verbose is True:      
            name =  self._name
            if self._is_paused:
                name += ' [paused]'
            message = 'Timer::' + name + ' - elapsed: ' + str(self._elapsed) 
            timer_print(message)
        return self._elapsed                


class TimerFps(Timer):
    def __init__(self, name='', average_width = 10, is_verbose = True): 
        super().__init__(name, is_verbose)   
        self.moving_average = MovingAverage(average_width)

    def refresh(self): 
        elapsed = self.elapsed()
        self.moving_average.getAverage(elapsed)
        self.start()
        if self._is_verbose is True:
            dT = self.moving_average.getAverage()
            name =  self._name
            if self._is_paused:
                name += ' [paused]'            
            message = 'Timer::' + name + ' - fps: ' + str(1./dT) + ', T: ' + str(dT)
            timer_print(message)
        