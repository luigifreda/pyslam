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

import sys
import numpy as np 
from enum import Enum


class GroundTruthType(Enum):
    NONE = 1
    KITTI = 2
    TUM = 3
    SIMPLE = 4


kScaleSimple = 1 
kScaleKitti = 1   
kScaleTum = 20    


def groundtruth_factory(settings):

    type=GroundTruthType.NONE
    associations = None 

    type = settings['type']
    path = settings['base_path']
    name = settings['name']
           
    print('using groundtruth: ', type)   
    if type == 'kitti':         
        return KittiGroundTruth(path, name, associations, GroundTruthType.KITTI)
    if type == 'tum':          
        if 'associations' in settings:
            associations = settings['associations']        
        return TumGroundTruth(path, name, associations, GroundTruthType.TUM)
    if type == 'video' or type == 'folder':   
        name = settings['groundtruth_file']
        return SimpleGroundTruth(path, name, associations, GroundTruthType.SIMPLE)     
    else:
        print('not using groundtruth')
        print('if you are using main_vo.py, your estimated trajectory will not make sense!')          
        return GroundTruth(path, name, associations=None, type=GroundTruthType.NONE)


# base class 
class GroundTruth(object):
    def __init__(self, path, name, associations=None, type=GroundTruthType.NONE):
        self.path=path 
        self.name=name 
        self.type=type    
        self.associations=associations 
        self.filename=None
        self.file_associations=None         
        self.data=None 
        self.scale = 1

    def getDataLine(self, frame_id):
        return self.data[frame_id].strip().split()
    
    def getPoseAndAbsoluteScale(self, frame_id):
        return 0,0,0,1

    # convert the dataset into 'Simple' format  [x,y,z,scale]
    def convertToSimpleXYZ(self, filename='groundtruth.txt'):
        out_file = open(filename,"w")
        num_lines = len(self.data)
        print('num_lines:', num_lines)
        for ii in range(num_lines):
            x,y,z,scale = self.getPoseAndAbsoluteScale(ii)
            if ii == 0:
                scale = 1 # first sample: we do not have a relative 
            out_file.write( "%f %f %f %f \n" % (x,y,z,scale) )
        out_file.close()


# read the ground truth from a simple file containining [x,y,z,scale] lines
class SimpleGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, type = GroundTruthType.KITTI): 
        super().__init__(path, name, associations, type)
        self.scale = kScaleSimple
        self.filename=path + '/' + name
        with open(self.filename) as f:
            self.data = f.readlines()
            self.found = True 
        if self.data is None:
            sys.exit('ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!') 

    def getPoseAndAbsoluteScale(self, frame_id):
        ss = self.getDataLine(frame_id-1)
        x_prev = self.scale*float(ss[0])
        y_prev = self.scale*float(ss[1])
        z_prev = self.scale*float(ss[2])     
        ss = self.getDataLine(frame_id) 
        x = self.scale*float(ss[0])
        y = self.scale*float(ss[1])
        z = self.scale*float(ss[2])
        abs_scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        return x,y,z,abs_scale 


class KittiGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, type = GroundTruthType.KITTI): 
        super().__init__(path, name, associations, type)
        self.scale = kScaleKitti
        self.filename=path + '/poses/' + name + '.txt'   # N.B.: this may depend on how you deployed the groundtruth files 
        with open(self.filename) as f:
            self.data = f.readlines()
            self.found = True 
        if self.data is None:
            sys.exit('ERROR while reading groundtruth file: please, check how you deployed the files and if the code is consistent with this!') 

    def getPoseAndAbsoluteScale(self, frame_id):
        ss = self.getDataLine(frame_id-1)
        x_prev = self.scale*float(ss[3])
        y_prev = self.scale*float(ss[7])
        z_prev = self.scale*float(ss[11])     
        ss = self.getDataLine(frame_id) 
        x = self.scale*float(ss[3])
        y = self.scale*float(ss[7])
        z = self.scale*float(ss[11])
        abs_scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        return x,y,z,abs_scale 


class TumGroundTruth(GroundTruth):
    def __init__(self, path, name, associations=None, type = GroundTruthType.TUM): 
        super().__init__(path, name, associations, type)
        self.scale = kScaleTum 
        self.filename=path + '/' + name + '/' + 'groundtruth.txt'     # N.B.: this may depend on how you deployed the groundtruth files 
        self.file_associations=path + '/' + name + '/' + associations # N.B.: this may depend on how you name the associations file
        with open(self.filename) as f:
            self.data = f.readlines()[3:] # skip the first three rows, which are only comments 
            self.data = [line.strip().split() for line in  self.data] 
        if self.data is None:
            sys.exit('ERROR while reading groundtruth file!') 
        if self.file_associations is not None: 
            with open(self.file_associations) as f:
                self.associations = f.readlines()
                self.associations = [line.strip().split() for line in self.associations] 
            if self.associations is None:
                sys.exit('ERROR while reading associations file!')    
        self.association_matches = self.associate(self.associations,self.data)  
        out_file=open('tum_association_matches.txt','w')               
        out_file.write(str(self.association_matches))
        out_file.close()

    def getDataLine(self, frame_id):
        return self.data[self.association_matches[frame_id][1]]

    def getPoseAndAbsoluteScale(self, frame_id):
        ss = self.getDataLine(frame_id-1) 
        x_prev = self.scale*float(ss[1])
        y_prev = self.scale*float(ss[2])
        z_prev = self.scale*float(ss[3])     
        ss = self.getDataLine(frame_id) 
        x = self.scale*float(ss[1])
        y = self.scale*float(ss[2])
        z = self.scale*float(ss[3])
        abs_scale = np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
        return x,y,z,abs_scale 

    @staticmethod
    def associate(first_list, second_list, offset=0, max_difference=0.02):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
        to find the closest match for every input tuple.
        
        Input:
        first_list -- first list of (stamp,data) tuples
        second_list -- second list of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
        
        """
        potential_matches = [(abs(float(a[0]) - (float(b[0]) + offset)), ia, ib) # a[0] and b[0] extract the first element which is a timestamp 
                            for ia,a in enumerate(first_list)      #for counter, value in enumerate(some_list)
                            for ib,b in enumerate(second_list)
                            if abs(float(a[0])  - (float(b[0])  + offset)) < max_difference]
        potential_matches.sort()
        matches = []
        first_flag = [False]*len(first_list)
        second_flag = [False]*len(second_list)
        for diff, ia, ib in potential_matches:
            if first_flag[ia] is False and second_flag[ib] is False:
                #first_list.remove(a)
                first_flag[ia] = True
                #second_list.remove(b)
                second_flag[ib] = True 
                matches.append((ia, ib, diff))    
        matches.sort()
        return matches    

