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
import cv2
import os
import glob

from multiprocessing import Process, Queue


class DatasetType(Enum):
    NONE = 1
    KITTI = 2
    TUM = 3
    VIDEO = 4
    FOLDER = 5  # generic folder of pics 
    LIVE = 6


def dataset_factory(settings):
    type=DatasetType.NONE
    associations = None    
    path = None 

    type = settings['type']
    name = settings['name']    
    path = settings['base_path'] 
    if 'associations' in settings:
        associations = settings['associations']

    if type == 'kitti':
        return KittiDataset(path, name, associations, DatasetType.KITTI)
    if type == 'tum':
        return TumDataset(path, name, associations, DatasetType.TUM)
    if type == 'video':
        return VideoDataset(path, name, associations, DatasetType.VIDEO)   
    if type == 'folder':
        return FolderDataset(path, name, associations, DatasetType.FOLDER)      
    if type == 'live':
        return LiveDataset(path, name, associations, DatasetType.LIVE)              
    return None 


class Dataset(object):
    def __init__(self, path, name, associations=None, type=DatasetType.NONE):
        self.path=path 
        self.name=name 
        self.type=type    
        self.is_ok = True;          

    def isOk(self):
        return self.is_ok

    def getImage(self, frame_id):
        return None 

    def getImage1(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    def getImageColor(self, frame_id):
        img = self.getImage(frame_id)
        if img.ndim == 2:
            return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
        else:
            return img 


class VideoDataset(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, associations, type)    
        self.filename = path + '/' + name 
        #print('video: ', self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            raise IOError('Cannot open movie file: ', self.filename)
        else: 
            print('Processing Video Input')
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   
            print('num frames: ', self.num_frames)  
        self.is_init = False   
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is > 0 
        if self.is_init is False and frame_id > 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.is_init = True
        ret, image = self.cap.read()
        if ret is False:
            print('ERROR in reading from file: ', self.filename)
        return image       


class LiveDataset(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, associations, type)    
        self.camera_num = name # use name for camera number
        self.cap = cv2.VideoCapture(self.camera_num) 
        if not self.cap.isOpened():
            raise IOError('Cannot open camera') 
            
    def getImage(self, frame_id):
        ret, image = self.cap.read()
        if ret is False:
            print('ERROR in reading from camera: ', self.camera_num)
        return image           


class FolderDataset(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, associations, type)    
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.i = 0        
        if self.maxlen == 0:
          raise IOError('No images were found in folder: ', path)            
            
    def getImage(self, frame_id):
        if self.i == self.maxlen:
            return (None, False)
        image_file = self.listing[self.i]
        img = cv2.imread(image_file, 0)
        if img is None: 
            raise IOError('error reading file: ', image_file)               
        # Increment internal counter.
        self.i = self.i + 1
        return img

class FolderDatasetParallelStatus:
    def __init__(self, i, maxlen, listing, skip):
        self.i = i
        self.maxlen = maxlen
        self.listing = listing 
        self.skip = skip  

class FolderDatasetParallel(Dataset): 
    def __init__(self, path, name, associations=None, type=DatasetType.VIDEO): 
        super().__init__(path, name, associations, type)    
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.i = 0        
        if self.maxlen == 0:
          raise IOError('No images were found in folder: ', path)     

        folder_status = FolderDatasetParallelStatus(i,maxlen,listing,skip)

        self.q = Queue(maxsize=2)        
        self.vp = Process(target=self._update_image, args=(self.q,folder_status,))
        self.vp.daemon = True                 
            
    def _update_image(self, q, status):
        while(True):
            self.ret, self.current_frame = self.cap.read()
            if self.ret is True: 
                #self.current_frame= self.cap.read()[1]
                #if q.full():
                #    old_frame = self.q.get()
                self.q.put(self.current_frame)
                print('q.size: ', self.q.qsize())
        time.sleep(0.005)

    def getImage(self, frame_id):
        if self.i == self.maxlen:
            return (None, False)
        image_file = self.listing[self.i]
        img = cv2.imread(image_file, 0)
        if img is None: 
            raise IOError('error reading file: ', image_file)               
        # Increment internal counter.
        self.i = self.i + 1
        return img 

class Webcam(object):
    def __init__(self, camera_num=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.current_frame = None 
        self.ret = None 
        self.q = Queue(maxsize=2)        
        self.vp = Process(target=self._update_frame, args=(self.q,))
        self.vp.daemon = True

    # create thread for capturing images
    def start(self):
        self.vp.start()        
        
    # process function     
    def _update_frame(self, q):
        while(True):
            self.ret, self.current_frame = self.cap.read()
            if self.ret is True: 
                #self.current_frame= self.cap.read()[1]
                #if q.full():
                #    old_frame = self.q.get()
                self.q.put(self.current_frame)
                print('q.size: ', self.q.qsize())
        time.sleep(0.005)
                  
    # get the current frame
    def get_current_frame(self):
        img = None 
        while not self.q.empty():  # get the last one 
            img = self.q.get()         
        return img

class KittiDataset(Dataset):
    def __init__(self, path, name, associations=None, type=DatasetType.KITTI): 
        super().__init__(path, name, associations, type)
        print('Processing KITTI Input')

    def getImage(self, frame_id):
        img = cv2.imread(self.path + '/sequences/' + self.name + '/image_0/' + str(frame_id).zfill(6) + '.png', 0)
        self.is_ok = (img is not None)
        return img 

    def getImage1(self, frame_id):
        img = cv2.imread(self.path + '/sequences/' + self.name + '/image_1/' + str(frame_id).zfill(6) + '.png', 0) 
        self.is_ok = (img is not None)        
        return img 


class TumDataset(Dataset):
    def __init__(self, path, name, associations, type=DatasetType.TUM): 
        super().__init__(path, name, associations, type)
        print('Processing TUM Input')        
        self.base_path=self.path + '/' + self.name + '/'
        associations_file=self.path + '/' + self.name + '/' + associations
        with open(associations_file) as f:
            self.associations = f.readlines()
        if self.associations is None:
            sys.exit('ERROR while reading associations file!')    
        self.max_frame_id = len(self.associations)    

    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[1]
            img = cv2.imread(file, 0)
            self.is_ok = (img is not None)
        else:
            self.is_ok = False                      
        return img 

    def getDepth(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[3]
            img = cv2.imread(file, 0)
            self.is_ok = (img is not None)
        else:
            self.is_ok = False                      
        return img 
