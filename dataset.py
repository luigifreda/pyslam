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
import time 
import csv
import re 
import datetime
from multiprocessing import Process, Queue, Value 
from utils_sys import Printer 

import ujson as json


class DatasetType(Enum):
    NONE = 1
    KITTI = 2
    TUM = 3
    EUROC = 4
    VIDEO = 5
    FOLDER = 6  # generic folder of pics 
    LIVE = 7

class SensorType(Enum):
    MONOCULAR=0,
    STEREO=1,
    RGBD=2

def dataset_factory(config):
    dataset_settings = config.dataset_settings
    type=DatasetType.NONE
    associations = None
    timestamps = None    
    path = None 
    is_color = None  # used for kitti datasets

    type = dataset_settings['type'].lower()
    name = dataset_settings['name']    
    
    sensor_type = SensorType.MONOCULAR
    if 'sensor_type' in dataset_settings:
        if dataset_settings['sensor_type'].lower() == 'mono':
            sensor_type = SensorType.MONOCULAR
        if dataset_settings['sensor_type'].lower() == 'stereo':
            sensor_type = SensorType.STEREO
        if dataset_settings['sensor_type'].lower() == 'rgbd':
            sensor_type = SensorType.RGBD
    Printer.green(f'dataset_factory - sensor_type: {sensor_type.name}')
    
    path = dataset_settings['base_path'] 
    path = os.path.expanduser(path)
    
    start_frame_id = 0
    if 'start_frame_id' in dataset_settings:
        Printer.green(f'dataset_factory - start_frame_id: {dataset_settings["start_frame_id"]}')
        start_frame_id = int(dataset_settings['start_frame_id'])
    
    if 'associations' in dataset_settings:
        associations = dataset_settings['associations']
    if 'timestamps' in dataset_settings:
        timestamps = dataset_settings['timestamps']
    if 'is_color' in dataset_settings:
        is_color = dataset_settings['is_color']

    dataset = None 

    if type == 'kitti':
        dataset = KittiDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.KITTI)
        dataset.set_is_color(is_color)   
    if type == 'tum':
        dataset = TumDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.TUM)
    if type == 'euroc':
        dataset = EurocDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.EUROC, config) 
    if type == 'video':
        dataset = VideoDataset(path, name, sensor_type, associations, timestamps, start_frame_id, DatasetType.VIDEO)   
    if type == 'folder':
        fps = 10 # a default value 
        if 'fps' in dataset_settings:
            fps = int(dataset_settings['fps'])
        dataset = FolderDataset(path, name, sensor_type, fps, associations, timestamps, start_frame_id, DatasetType.FOLDER)      
    if type == 'live':
        dataset = LiveDataset(path, name, sensor_type, associations, start_frame_id, DatasetType.LIVE)   
                
    return dataset 


class Dataset(object):
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, fps=None, associations=None, start_frame_id=0, type=DatasetType.NONE):
        self.path = path 
        self.name = name 
        self.type = type    
        self.sensor_type = sensor_type
        self.scale_viewer_3d = 0.1 
        self.is_ok = True
        self.fps = fps   
        self.num_frames = None
        if fps is not None:       
            self.Ts = 1./fps 
        else: 
            self.Ts = None 
          
        self.start_frame_id = start_frame_id
        self.timestamps = None 
        self._timestamp = None       # current timestamp if available [s]
        self._next_timestamp = None  # next timestamp if available otherwise an estimate [s]
        
    def isOk(self):
        return self.is_ok
    
    def sensorType(self):
        return self.sensor_type

    def getImage(self, frame_id):
        return None 

    def getImageRight(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    # Adjust frame id with start frame id only here
    def getImageColor(self, frame_id):
        frame_id += self.start_frame_id
        try: 
            img = self.getImage(frame_id)
            if img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            Printer.red(f'Cannot open dataset: {self.name}, path: {self.path}')
            return img    
        
    # Adjust frame id with start frame id only here
    def getImageColorRight(self, frame_id):
        frame_id += self.start_frame_id
        try: 
            img = self.getImageRight(frame_id)
            if img is not None and img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None  
            #raise IOError('Cannot open dataset: ', self.name, ', path: ', self.path)        
            Printer.red(f'Cannot open dataset: {self.name}, path: {self.path}, right image')
            return img         
        
    def getTimestamp(self):
        return self._timestamp
    
    def getNextTimestamp(self):
        return self._next_timestamp

    def _read_timestamps(self, timestamps_file):
        timestamps = []
        try:
            with open(timestamps_file, 'r') as file:
                for line in file:
                    timestamp = float(line.strip())
                    timestamps.append(timestamp)
        except FileNotFoundError:
            print('Timestamps file not found:', timestamps_file)
        return timestamps
    
    def to_json(self):
        return {
            'type': self.type.name,            
            'name': self.name,
            'sensor_type': self.sensor_type.name,
            'path': self.path,            
            'start_frame_id': self.start_frame_id,
            'fps': self.fps,
            'num_frames': self.num_frames
        }
        
    def save_info(self, path):
        filename = path + '/dataset_info.json'
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


class VideoDataset(Dataset): 
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, associations=None, timestamps=None, start_frame_id=0, type=DatasetType.VIDEO): 
        super().__init__(path, name, sensor_type, None, associations, start_frame_id, type)    
        if sensor_type != SensorType.MONOCULAR:
            raise ValueError('Video dataset only supports MONOCULAR sensor type')
        self.filename = path + '/' + name 
                
        #print('video: ', self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        self.i = 0        
        self.timestamps = None
        if timestamps is not None:
            self.timestamps = self._read_timestamps(path + '/' + timestamps)
            Printer.green('read timestamps from ' + path + '/' + timestamps)
        if not self.cap.isOpened():
            raise IOError('Cannot open movie file: ', self.filename)
        else: 
            print('Processing Video Input')
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps 
            print('num frames: ', self.num_frames)  
            print('fps: ', self.fps)              
        self.is_init = False   
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is > 0 
        if self.is_init is False and frame_id > 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        self.is_init = True
        ret, image = self.cap.read()
        if self.timestamps is not None:
            # read timestamps from timestamps file
            self._timestamp = float(self.timestamps[self.i])
            self._next_timestamp = float(self.timestamps[self.i + 1])
            self.i += 1
        else:
            #self._timestamp = time.time()  # rough timestamp if nothing else is available 
            self._timestamp = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)*1000)
            self._next_timestamp = self._timestamp + self.Ts 
        if ret is False:
            print('ERROR while reading from file: ', self.filename)
        return image       



class LiveDataset(Dataset): 
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, associations=None, start_frame_id=0, type=DatasetType.VIDEO): 
        super().__init__(path, name, sensor_type, None, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR:
            raise ValueError('Video dataset only supports MONOCULAR sensor type')           
        self.camera_num = name # use name for camera number
        print('opening camera device: ', self.camera_num)
        self.cap = cv2.VideoCapture(self.camera_num)   
        if not self.cap.isOpened():
            raise IOError('Cannot open camera') 
        else:
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.Ts = 1./self.fps             
            print('fps: ', self.fps)    
            
    def getImage(self, frame_id):
        ret, image = self.cap.read()
        self._timestamp = time.time()  # rough timestamp if nothing else is available 
        self._next_timestamp = self._timestamp + self.Ts         
        if ret is False:
            print('ERROR in reading from camera: ', self.camera_num)
        return image           



class FolderDataset(Dataset): 
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, fps=None, associations=None, timestamps=None, start_frame_id=0, type=DatasetType.VIDEO): 
        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR:
            raise ValueError('Video dataset only supports MONOCULAR sensor type')        
        if fps is None: 
            fps = 10 # default value  
        self.fps = fps 
        print('fps: ', self.fps)  
        self.Ts = 1./self.fps 
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.num_frames = self.maxlen
        self.i = 0        
        if self.maxlen == 0:
          raise IOError('No images were found in folder: ', path)   
        self._timestamp = 0.
        self.timestamps = None
        if timestamps is not None:
            self.timestamps = self._read_timestamps(path + '/' + timestamps)
        
    def getImage(self, frame_id):
        if self.i == self.maxlen:
            return None
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        pattern = re.compile(r'\d+')
        if self.timestamps is not None:
            # read timestamps from timestamps file
            self._timestamp = float(self.timestamps[self.i])
            self._next_timestamp = float(self.timestamps[self.i + 1])

        elif pattern.search(image_file.split('/')[-1].split('.')[0]):
            # read timestamps from image filename
            self._timestamp = float(image_file.split('/')[-1].split('.')[0])
            self._next_timestamp = float(self.listing[self.i + 1].split('/')[-1].split('.')[0])
        else:
            self._timestamp += self.Ts
            self._next_timestamp = self._timestamp + self.Ts 
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

# this is experimental 
class FolderDatasetParallel(Dataset): 
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, fps=None, associations=None, start_frame_id=0, type=DatasetType.VIDEO): 
        super().__init__(path, name, sensor_type, fps, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR:
            raise ValueError('Video dataset only supports MONOCULAR sensor type')            
        print('fps: ', self.fps)  
        self.Ts = 1./self.fps    
        self._timestamp = 0     
        self.skip=1
        self.listing = []    
        self.maxlen = 1000000    
        print('Processing Image Directory Input')
        self.listing = glob.glob(path + '/' + self.name)
        self.listing.sort()
        self.listing = self.listing[::self.skip]
        #print('list of files: ', self.listing)
        self.maxlen = len(self.listing)
        self.num_frames = self.maxlen
        self.i = 0        
        if self.maxlen == 0:
          raise IOError('No images were found in folder: ', path)     

        self.is_running = Value('i',1)  
        
        self.folder_status = FolderDatasetParallelStatus(self.i,self.maxlen,self.listing,self.skip)

        self.q = Queue(maxsize=10)    
        self.q.put(self.folder_status)   # pass the folder status with the initialization  
        
        self.vp = Process(target=self._update_image, args=(self.q,))
        self.vp.daemon = True                 
            
    # create thread for reading images
    def start(self):
        self.vp.start()        
        
    def quit(self):
        print('webcam closing...') 
        self.is_running.value = 0
        self.vp.join(timeout=3)     
                    
    def _update_image(self, q):
        folder_status = q.get()  
        while self.is_running.value == 1:
            while not q.full():
                self.current_frame = self._get_image(folder_status)
                self.q.put(self.current_frame)
                #print('q.size: ', self.q.qsize())
        time.sleep(0.005)

    def _get_image(self, folder_status):
        if self.i == folder_status.maxlen:
            return (None, False)
        image_file = folder_status.listing[self.i]
        img = cv2.imread(image_file)         
        if img is None: 
            raise IOError('error reading file: ', image_file)               
        # Increment internal counter.
        self.i = self.i + 1
        return img 
    
    # get the current frame
    def getImage(self):
        img = None 
        while not self.q.empty():  # get the last one
            self._timestamp += self.Ts
            self._next_timestamp = self._timestamp + self.Ts                  
            img = self.q.get()         
        return img    



class Webcam(object):
    def __init__(self, camera_num=0):
        self.cap = cv2.VideoCapture(camera_num)
        self.current_frame = None 
        self.ret = None 
        
        self.is_running = Value('i',1)        
        self.q = Queue(maxsize=2)        
        self.vp = Process(target=self._update_frame, args=(self.q,self.is_running,))
        self.vp.daemon = True

    # create thread for capturing images
    def start(self):
        self.vp.start()        
        
    def quit(self):
        print('webcam closing...') 
        self.is_running.value = 0
        self.vp.join(timeout=3)               
        
    # process function     
    def _update_frame(self, q, is_running):
        while is_running.value == 1:
            self.ret, self.current_frame = self.cap.read()
            if self.ret is True: 
                #self.current_frame= self.cap.read()[1]
                if q.full():
                    old_frame = self.q.get()
                self.q.put(self.current_frame)
                print('q.size: ', self.q.qsize())           
        time.sleep(0.005)
                  
    # get the current frame
    def get_current_frame(self):
        img = None 
        while not self.q.empty():  # get last available image
            img = self.q.get()         
        return img



class KittiDataset(Dataset):
    def __init__(self, path, name, sensor_type=SensorType.STEREO, associations=None, start_frame_id=0, type=DatasetType.KITTI): 
        super().__init__(path, name, sensor_type, 10, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.STEREO:
            raise ValueError('Video dataset only supports MONOCULAR and STEREO sensor types')        
        self.fps = 10
        if sensor_type == SensorType.STEREO:
            self.scale_viewer_3d = 1          
        self.image_left_path = '/image_0/'
        self.image_right_path = '/image_1/'           
        self.timestamps = np.loadtxt(self.path + '/sequences/' + self.name + '/times.txt', dtype=np.float64)
        self.max_frame_id = len(self.timestamps)
        self.num_frames = self.max_frame_id
        print('Processing KITTI Sequence of lenght: ', len(self.timestamps))
        
    def set_is_color(self,val):
        self.is_color = val 
        if self.is_color:
            print('dataset in color!')            
            self.image_left_path = '/image_2/'
            self.image_right_path = '/image_3/'                           
        
    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            try: 
                img = cv2.imread(self.path + '/sequences/' + self.name + self.image_left_path + str(frame_id).zfill(6) + '.png')
                self._timestamp = self.timestamps[frame_id]
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[frame_id+1]
            else:
                self._next_timestamp = self._timestamp + self.Ts             
        self.is_ok = (img is not None)
        return img 

    def getImageRight(self, frame_id):
        print(f'[KittiDataset] getImageRight: {frame_id}')
        img = None
        if frame_id < self.max_frame_id:        
            try: 
                img = cv2.imread(self.path + '/sequences/' + self.name + self.image_right_path + str(frame_id).zfill(6) + '.png') 
                self._timestamp = self.timestamps[frame_id]        
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )   
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[frame_id+1]
            else:
                self._next_timestamp = self._timestamp + self.Ts                   
        self.is_ok = (img is not None)        
        return img 


class TumDataset(Dataset):
    def __init__(self, path, name, sensor_type=SensorType.RGBD, associations=None, start_frame_id=0, type=DatasetType.TUM): 
        super().__init__(path, name, sensor_type, 30, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.RGBD:
            raise ValueError('Video dataset only supports MONOCULAR and RGBD sensor types')          
        self.fps = 30
        self.scale_viewer_3d = 0.1
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05             
        print('Processing TUM Sequence')        
        self.base_path=self.path + '/' + self.name + '/'
        associations_file=self.path + '/' + self.name + '/' + associations
        with open(associations_file) as f:
            self.associations = f.readlines()
            self.max_frame_id = len(self.associations)   
            self.num_frames = self.max_frame_id        
        if self.associations is None:
            sys.exit('ERROR while reading associations file!')    

    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[1]
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations[frame_id+1].strip().split()[0])
            else:
                self._next_timestamp = self._timestamp + self.Ts              
        else:
            self.is_ok = False     
            self._timestamp = None                  
        return img 

    def getDepth(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None # force a monocular camera if required (to get a monocular tracking even if depth is available)
        frame_id += self.start_frame_id
        img = None
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations[frame_id].strip().split()[3]
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations[frame_id+1].strip().split()[0])
            else:
                self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 


class EurocDataset(Dataset):
    def __init__(self, path, name, sensor_type=SensorType.STEREO, associations=None, start_frame_id=0, type=DatasetType.EUROC, config=None): 
        super().__init__(path, name, sensor_type, 20, associations, start_frame_id, type)
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.STEREO:
            raise ValueError('Video dataset only supports MONOCULAR and STEREO sensor types')           
        self.fps = 20
        if sensor_type == SensorType.STEREO:
            self.scale_viewer_3d = 0.1                      
        self.image_left_path = '/mav0/cam0/data/'
        self.image_right_path = '/mav0/cam1/data/'
        self.image_left_csv_path = '/mav0/cam0/data.csv'
        self.image_right_csv_path = '/mav0/cam1/data.csv'
        
        timestamps_and_filenames_left = self.read_data(self.path + '/' + self.name + self.image_left_csv_path)
        timestamps_and_filenames_right = self.read_data(self.path + '/' + self.name + self.image_right_csv_path)

        self.timestamps = np.array([x[0] for x in timestamps_and_filenames_left])
        self.filenames = np.array([x[1] for x in timestamps_and_filenames_left])
        self.timestamps_right = np.array([x[0] for x in timestamps_and_filenames_right])
        self.filenames_right = np.array([x[1] for x in timestamps_and_filenames_right])
        self.max_frame_id = len(self.timestamps)
        self.num_frames = self.max_frame_id
        
        # in case of stereo mode, we rectify the stereo images
        self.stereo_settings = config.stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow('[EuroDataset] automatically rectifying the stereo images')
            if self.stereo_settings is None: 
                sys.exit('ERROR: we are missing stereo settings in Euroc YAML settings!')   
            width = config.width 
            height = config.height         
            
            K_l = self.stereo_settings['left']['K']
            D_l = self.stereo_settings['left']['D']
            R_l = self.stereo_settings['left']['R']
            P_l = self.stereo_settings['left']['P']
            
            K_r = self.stereo_settings['right']['K']
            D_r = self.stereo_settings['right']['D']
            R_r = self.stereo_settings['right']['R']
            P_r = self.stereo_settings['right']['P']
            
            self.M1l,self.M2l = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l[0:3,0:3], (width, height), cv2.CV_32FC1)
            self.M1r,self.M2r = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r[0:3,0:3], (width, height), cv2.CV_32FC1)
        self.debug_rectification = False # DEBUGGING
        print('Processing Euroc Sequence of lenght: ', len(self.timestamps))
        
    def read_data(self, csv_file):
        timestamps_and_filenames = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header row
            for row in reader:
                timestamp_ns = int(row[0])
                filename = row[1]
                timestamp_s = (timestamp_ns / 1000000000)
                timestamps_and_filenames.append((timestamp_s, filename))
        return timestamps_and_filenames
                        
        
    def getImage(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:
            try: 
                img = cv2.imread(self.path + '/' + self.name + self.image_left_path + self.filenames[frame_id])
                if self.sensor_type == SensorType.STEREO:
                    # rectify image
                    if self.debug_rectification:
                        imgs = img 
                    img = cv2.remap(img,self.M1l,self.M2l,cv2.INTER_LINEAR)
                    if self.debug_rectification: 
                        imgs = np.concatenate((imgs,img),axis=1)
                        cv2.imshow('left raw and rectified images',imgs)
                        cv2.waitKey(1)
                
                self._timestamp = self.timestamps[frame_id]
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps[frame_id+1]
            else:
                self._next_timestamp = self._timestamp + self.Ts            
        self.is_ok = (img is not None)
        return img 

    def getImageRight(self, frame_id):
        img = None
        if frame_id < self.max_frame_id:        
            try: 
                img = cv2.imread(self.path + '/' + self.name + self.image_right_path + self.filenames_right[frame_id]) 
                if self.sensor_type == SensorType.STEREO:
                    # rectify image
                    if self.debug_rectification:
                        imgs = img                     
                    img = cv2.remap(img,self.M1r,self.M2r,cv2.INTER_LINEAR)
                    if self.debug_rectification: 
                        imgs = np.concatenate((imgs,img),axis=1)
                        cv2.imshow('right raw and rectified images',imgs)
                        cv2.waitKey(1)                           
                self._timestamp = self.timestamps_right[frame_id]        
            except:
                print('could not retrieve image: ', frame_id, ' in path ', self.path )   
            if frame_id+1 < self.max_frame_id:   
                self._next_timestamp = self.timestamps_right[frame_id+1]
            else:
                self._next_timestamp = self._timestamp + self.Ts                  
        self.is_ok = (img is not None)        
        return img 