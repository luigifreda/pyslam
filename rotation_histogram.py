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

import math 
import numpy as np 


class RotationHistogram(object):
    def __init__(self,histogram_lenght=30):
        self.histogram_lenght=histogram_lenght
        self.factor = 1.0/histogram_lenght
        self.histo = [ [] for i in range(self.histogram_lenght) ] 
        
    def push(self, rot, idx): 
        if rot < 0.0:
            rot += 360.0
        bin = int(round(rot*self.factor))
        if bin == self.histogram_lenght:
            bin = 0
        assert(bin>=0 and bin<self.histogram_lenght)
        self.histo[bin].append(idx)

    def compute_3_max(self):
        max1=max2=max3=0.
        ind1=ind2=ind3=-1 
        for i in range(self.histogram_lenght):
            s = len(self.histo[i])
            if(s>max1):
                max3=max2
                max2=max1
                max1=s
                ind3=ind2
                ind2=ind1
                ind1=i
            elif(s>max2):
                max3=max2
                max2=s
                ind3=ind2
                ind2=i
            elif(s>max3):
                max3=s
                ind3=i
        if max2 < 0.1*max1:
            ind2=-1
            ind3=-1
        elif max3 < 0.1*max1:
            ind3=-1
        return ind1, ind2, ind3
    
    def get_invalid_idxs(self): 
        ind1, ind2, ind3 = self.compute_3_max()
        invalid_idxs = []
        for i in range(self.histogram_lenght):        
            if i != ind1 and i != ind2 and i != ind3: 
                invalid_idxs.extend(self.histo[i])
        return invalid_idxs
    
    def get_valid_idxs(self): 
        ind1, ind2, ind3 = self.compute_3_max()
        valid_idxs = []
        if ind1 != -1: 
            valid_idxs.extend(self.histo[ind1])
        if ind2 != -1: 
            valid_idxs.extend(self.histo[ind2])
        if ind3 != -1: 
            valid_idxs.extend(self.histo[ind3])            
        return valid_idxs    
    
    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return 'RotationHistogram ' + str(self.histo)       

