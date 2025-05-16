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

from multiprocessing import Process, Queue, Value 
from utils_sys import Printer
from utils_serialization import SerializableEnum, register_class, SerializationJSON
from utils_string import levenshtein_distance

import ujson as json
from dataset_types import DatasetType, SensorType, DatasetEnvironmentType, MinimalDatasetConfig

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import Config # Only imported when type checking, not at runtime


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kSettingsFolder = kRootFolder + '/settings'


# Base class for implementing datasets
class Dataset(object):
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, fps=None, associations=None, start_frame_id=0, 
                 type=DatasetType.NONE, environment_type=DatasetEnvironmentType.OUTDOOR):
        self.path = path 
        self.name = name 
        self.type = type    
        self.sensor_type = sensor_type
        self.environment_type = environment_type
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
        self.associations = associations  # name of the file with the associations
        self.has_gt_semantics = False # false by default and can be changed to true by certain datasets (replica, scannet, etc.)
        self.minimal_config = MinimalDatasetConfig()
        
    def isOk(self):
        return self.is_ok
    
    def sensorType(self):
        return self.sensor_type
    
    def environmentType(self):
        return self.environment_type

    def getImage(self, frame_id):
        return None 

    def getImageRight(self, frame_id):
        return None

    def getDepth(self, frame_id):
        return None        

    def getDepthRight(self, frame_id):
        return None    
    
    # Adjust frame id with start frame id only here
    def getImageColor(self, frame_id):
        frame_id += self.start_frame_id
        if self.num_frames is not None and frame_id >= self.num_frames:
            if self.is_ok:
                Printer.yellow(f'Dataset end: {self.name}, path: {self.path}, frame id: {frame_id}')
                self.is_ok = False
            return None
        try: 
            img = self.getImage(frame_id)
            if img is None:
                return None
            if img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None
            self.is_ok = False
            if self.num_frames is not None and frame_id >= self.num_frames:
                Printer.yellow(f'Dataset end: {self.name}, path: {self.path}, frame id: {frame_id}')
            else:    
                Printer.red(f'Cannot open dataset: {self.name}, path: {self.path}, frame id: {frame_id}')
            return img    
        
    # Adjust frame id with start frame id only here
    def getImageColorRight(self, frame_id):
        frame_id += self.start_frame_id
        if self.num_frames is not None and frame_id >= self.num_frames:
            return None        
        try: 
            img = self.getImageRight(frame_id)
            if img is None:
                return None            
            if img is not None and img.ndim == 2:
                return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)     
            else:
                return img             
        except:
            img = None      
            if self.num_frames is not None and frame_id >= self.num_frames:
                Printer.yellow(f'Dataset end: {self.name}, path: {self.path}, right image, frame id: {frame_id}')
            else:    
                Printer.red(f'Cannot open dataset: {self.name}, path: {self.path}, right image, frame id: {frame_id}')            
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
        return self.minimal_config.to_json()
        
    # @staticmethod
    # def from_json(json_str):
    #     minimal_config = MinimalDatasetConfig.from_json(json_str)
    #     return dataset_factory(minimal_config)
        
    def save_info(self, path):
        filename = path + '/dataset_info.json'
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f, indent=4)



class VideoDataset(Dataset): 
    def __init__(self, path, name, sensor_type=SensorType.MONOCULAR, associations=None, timestamps=None, start_frame_id=0, type=DatasetType.VIDEO): 
        super().__init__(path, name, sensor_type, None, associations, start_frame_id, type)    
        if sensor_type != SensorType.MONOCULAR:
            raise ValueError('VideoDataset only supports MONOCULAR sensor type at present time')
        self.filename = os.path.join(path, name)
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"VideoDataset: File does not exist: {self.filename}")
                
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap.isOpened():
            raise IOError(f"VideoDataset: Cannot open movie file: {self.filename}")
                     
        self.timestamps = None
        if timestamps is not None:
            timestamps_path = os.path.join(path, timestamps)
            if not os.path.exists(timestamps_path):
                raise FileNotFoundError(f"Timestamps file does not exist: {timestamps_path}")            
            self.timestamps = self._read_timestamps(timestamps_path)
            Printer.green('read timestamps from ' + timestamps_path)

        print('Processing Video Input')
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.Ts = 1.0 / self.fps if self.fps > 0 else 0 
        self.i = 0           
        self.is_init = False           
        print(f"VideoDataset: {self.filename}")
        print(f"VideoDataset: Number of frames: {self.num_frames}, FPS: {self.fps}")            
            
    def getImage(self, frame_id):
        # retrieve the first image if its id is >= 0 
        if self.is_init is False and frame_id >= 0:
            self.is_init = True 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            self.i = frame_id

        ret, image = self.cap.read()
        if not ret:
            raise RuntimeError(f"Error reading frame from file: {self.filename}")
                
        if self.timestamps is not None:
            if self.i > len(self.timestamps) - 1:
                raise IndexError("Reached the end of the timestamp list.")            
            # read timestamps from timestamps file
            self._timestamp = float(self.timestamps[self.i])
            if self.i < len(self.timestamps) - 1:
                self._next_timestamp = float(self.timestamps[self.i + 1])
            else:
                self._next_timestamp = self._timestamp + self.Ts
            self.i += 1
        else:
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
        self.listing = glob.glob(path + '/' + self.name) # 'name' is used for specifying a glob pattern
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
            timestamps_path = os.path.join(path, timestamps)
            if not os.path.exists(timestamps_path):
                raise FileNotFoundError(f"Timestamps file does not exist: {timestamps_path}")            
            self.timestamps = self._read_timestamps(timestamps_path)
            Printer.green('read timestamps from ' + timestamps_path)            
        
    def getImage(self, frame_id):
        if self.i == self.maxlen:
            return None
        image_file = self.listing[self.i]
        img = cv2.imread(image_file)
        pattern = re.compile(r'\d+')
        if self.timestamps is not None:
            # read timestamps from timestamps file
            self._timestamp = float(self.timestamps[self.i])
            if self.i < len(self.timestamps) - 1:
                self._next_timestamp = float(self.timestamps[self.i + 1])
            else:
                self._next_timestamp = self._timestamp + self.Ts            

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
        self.environment_type = DatasetEnvironmentType.OUTDOOR
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.STEREO:
            raise ValueError('Video dataset only supports MONOCULAR and STEREO sensor types')        
        self.fps = 10
        if sensor_type == SensorType.STEREO:
            self.scale_viewer_3d = 1          
        self.image_left_path = '/image_0/'
        self.image_right_path = '/image_1/'           
        self.timestamps = np.loadtxt(self.path + '/sequences/' + str(self.name) + '/times.txt', dtype=np.float64)
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
        self.environment_type = DatasetEnvironmentType.INDOOR
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.RGBD:
            raise ValueError('Video dataset only supports MONOCULAR and RGBD sensor types')          
        self.fps = 30
        self.scale_viewer_3d = 0.1
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05             
        print('Processing TUM Sequence')        
        self.base_path = self.path + '/' + self.name + '/'
        self.associations_path = self.path + '/' + self.name + '/' + associations
        with open(self.associations_path) as f:
            self.associations_data = f.readlines()
            self.max_frame_id = len(self.associations_data)   
            self.num_frames = self.max_frame_id        
        if self.associations_data is None:
            sys.exit('ERROR while reading associations file!')    

    def getImage(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            file = self.base_path + self.associations_data[frame_id].strip().split()[1]
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations_data[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations_data[frame_id+1].strip().split()[0])
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
            file = self.base_path + self.associations_data[frame_id].strip().split()[3]
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            self.is_ok = (img is not None)
            self._timestamp = float(self.associations_data[frame_id].strip().split()[0])
            if frame_id +1 < self.max_frame_id: 
                self._next_timestamp = float(self.associations_data[frame_id+1].strip().split()[0])
            else:
                self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 

class ScannetDataset(Dataset):
    def __init__(self, path, name, sensor_type=SensorType.RGBD, associations=None, start_frame_id=0, type=DatasetType.SCANNET, image_size=(640, 480)): 
        super().__init__(path, name, sensor_type, 30, associations, start_frame_id, type)
        self.environment_type = DatasetEnvironmentType.INDOOR
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.RGBD:
            raise ValueError('Video dataset only supports MONOCULAR and RGBD sensor types')          
        self.fps = 30 #TODO(@dvdmc): I couldn't find this anywhere (paper, code, etc.)
        self.scale_viewer_3d = 0.1
        self.depthmap_factor = 1 #TODO(@dvdmc): I don't know why this is 1. It's supposed to be 1000. since the depth is in mm. Did they change it?
        # NOTE: We need the following to resize the RGB images to have the same size as the depth images
        # Other intrinsics are changed in the settings file (see settings/SCANNET.yaml)
        self.image_size = image_size 
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05             
        print('Processing ScanNet Sequence')        
        self.base_path = self.path + '/scans/' + self.name + '/'
        # count the number of frames in the path 
        self.color_paths = sorted(glob.glob(f'{self.base_path}/color/*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.base_path}/depth/*.png'))  
        self.max_frame_id = len(self.color_paths)   
        self.num_frames = self.max_frame_id
        print(f'Number of frames: {self.max_frame_id}')  

    def getImage(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            file = self.base_path + f'/color/{str(frame_id)}.jpg'
            img = cv2.imread(file)
            img = cv2.resize(img, self.image_size)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
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
            file = self.base_path + f'/depth/{str(frame_id)}.png'
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            img = (img/self.depthmap_factor).astype(np.float32)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
            self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 

class EurocDataset(Dataset):
    def __init__(self, path, name, sensor_type=SensorType.STEREO, associations=None, start_frame_id=0, type=DatasetType.EUROC, config=None): 
        super().__init__(path, name, sensor_type, 20, associations, start_frame_id, type)
        self.environment_type = DatasetEnvironmentType.INDOOR 
        # TODO: may be we can better distinguish the dataset type from path name       
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.STEREO:
            raise ValueError('Video dataset only supports MONOCULAR and STEREO sensor types')           
        self.fps = 20
        if sensor_type == SensorType.STEREO:
            self.scale_viewer_3d = 0.1                      
        self.image_left_path = '/mav0/cam0/data/'
        self.image_right_path = '/mav0/cam1/data/'
        self.image_left_csv_path = '/mav0/cam0/data.csv'
        self.image_right_csv_path = '/mav0/cam1/data.csv'
        
        # Here we read and convert the timestamps to seconds
        # NOTE: This is the old way. This has problem with certain Euroc sequences where associating 
        #       the i-th row of the left csv file with the i-th row of the right csv file is not correct. 
        #timestamps_and_filenames_left = self.read_data(self.path + '/' + self.name + self.image_left_csv_path)
        #timestamps_and_filenames_right = self.read_data(self.path + '/' + self.name + self.image_right_csv_path)
        
        # Here we directly read the timestamps (and name of the images) in the image folder kSettingsFolder + '/euroc_timestamps/'
        euroc_timestamps_names = ['MH01', 'MH02', 'MH03', 'MH04', 'MH05', 'V101', 'V102', 'V103', 'V201', 'V202', 'V203']
        name_uc = name.upper()
        name_euroc_timestamps = 'MH04_difficult' # name_uc
        if name_uc in euroc_timestamps_names:
            name_euroc_timestamps = name_uc
        else:
            # Find the closest match based on Levenshtein's distance
            distances = [(candidate, levenshtein_distance(name_uc, candidate)) 
                        for candidate in euroc_timestamps_names]
            closest_match, _ = min(distances, key=lambda x: x[1])
            name_euroc_timestamps = closest_match

        euroc_timestamps_folder_path = kSettingsFolder + '/euroc_timestamps/'
        euroc_timestamps_file_path = euroc_timestamps_folder_path + name_euroc_timestamps + '.txt'
        if not os.path.exists(euroc_timestamps_file_path):
            Printer.red(f'ERROR: [EurocDataset] Cannot find timestamps file: {euroc_timestamps_file_path}!') 
            Printer.red(f'\t There is a problem in associating the input dataset {name} with the timestamps files in {euroc_timestamps_folder_path}.')
            Printer.red(f'\t The Euroc sequence name is sopposed to be one of the following: {euroc_timestamps_names}.')
            raise FileNotFoundError(f"Timestamps file does not exist: {euroc_timestamps_file_path}")
        timestamps_and_filenames_left = self.read_single_data(euroc_timestamps_file_path)
        timestamps_and_filenames_right = self.read_single_data(euroc_timestamps_file_path)

        self.timestamps = np.array([x[0] for x in timestamps_and_filenames_left])
        self.filenames = np.array([x[1] for x in timestamps_and_filenames_left])
        self.timestamps_right = np.array([x[0] for x in timestamps_and_filenames_right])
        self.filenames_right = np.array([x[1] for x in timestamps_and_filenames_right])
        
        self.max_frame_id = len(self.timestamps)
        self.num_frames = self.max_frame_id
        
        # in case of stereo mode, we rectify the stereo images
        self.cam_stereo_settings = config.cam_stereo_settings
        if self.sensor_type == SensorType.STEREO:
            Printer.yellow('[EurocDataset] automatically rectifying the stereo images')
            if self.cam_stereo_settings is None: 
                sys.exit('ERROR: we are missing stereo settings in Euroc YAML settings!')   
            width = config.cam_settings['Camera.width'] 
            height = config.cam_settings['Camera.height']         
            
            K_l = self.cam_stereo_settings['left']['K']
            D_l = self.cam_stereo_settings['left']['D']
            R_l = self.cam_stereo_settings['left']['R']
            P_l = self.cam_stereo_settings['left']['P']
            
            K_r = self.cam_stereo_settings['right']['K']
            D_r = self.cam_stereo_settings['right']['D']
            R_r = self.cam_stereo_settings['right']['R']
            P_r = self.cam_stereo_settings['right']['P']
            
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
                        
    def read_single_data(self, csv_file):
        timestamps_and_filenames = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                timestamp_ns = int(row[0])
                filename = row[0] + '.png'
                timestamp_s = (timestamp_ns / 1000000000)
                timestamps_and_filenames.append((timestamp_s, filename))
        return timestamps_and_filenames                        
        
    def getImage(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            try:
                img_name = self.filenames[frame_id]
                #img_name = self.get_timestamp(frame_id) + '.png'
                img = cv2.imread(self.path + '/' + self.name + self.image_left_path + img_name)
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
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:        
            try:
                img_name = self.filenames_right[frame_id]
                #img_name = self.get_timestamp(frame_id) + '.png'
                print('getImageRight: ', img_name)
                img = cv2.imread(self.path + '/' + self.name + self.image_right_path + img_name) 
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
        

class ReplicaDataset(Dataset):
    fps = 25
    Ts = 1./fps
    def __init__(self, path, name, sensor_type=SensorType.RGBD, associations=None, start_frame_id=0, type=DatasetType.REPLICA): 
        super().__init__(path, name, sensor_type, 30, associations, start_frame_id, type)
        self.environment_type = DatasetEnvironmentType.INDOOR
        if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.RGBD:
            raise ValueError('Video dataset only supports MONOCULAR and RGBD sensor types')          
        self.fps = ReplicaDataset.fps
        self.Ts = ReplicaDataset.Ts
        self.scale_viewer_3d = 0.1
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05             
        print('Processing Replica Sequence')        
        self.base_path = self.path + '/' + self.name + '/'
        # count the number of frames in the path 
        self.color_paths = sorted(glob.glob(f'{self.base_path}/results/frame*.jpg'))
        self.depth_paths = sorted(glob.glob(f'{self.base_path}/results/depth*.png'))        
        self.max_frame_id = len(self.color_paths)   
        self.num_frames = self.max_frame_id
        print(f'Number of frames: {self.max_frame_id}')  

    def getImage(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            file = self.base_path + f'results/frame{str(frame_id).zfill(6)}.jpg'
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
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
            file = self.base_path + f'results/depth{str(frame_id).zfill(6)}.png'
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
            self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 
    
    
    
# Tartanair 
# GT format: tx ty tz qx qy qz qw
# References: 
# https://github.com/castacks/tartanair_tools/blob/master/data_type.md
# https://www.aicrowd.com/challenges/tartanair-visual-slam-stereo-track
class TartanairDataset(Dataset):
    fps = 25
    Ts = 1./fps
    def __init__(self, path, name, sensor_type=SensorType.RGBD, associations=None, start_frame_id=0, type=DatasetType.TARTANAIR, environment_type = DatasetEnvironmentType.INDOOR): 
        super().__init__(path, name, sensor_type, 30, associations, start_frame_id, type)
        self.environment_type = environment_type
        # if sensor_type != SensorType.MONOCULAR and sensor_type != SensorType.RGBD:
        #     raise ValueError('Video dataset only supports MONOCULAR and RGBD sensor types')          
        self.fps = TartanairDataset.fps
        self.Ts = TartanairDataset.Ts
        self.scale_viewer_3d = 0.1
        if sensor_type == SensorType.MONOCULAR:
            self.scale_viewer_3d = 0.05             
        print('Processing Replica Sequence')        
        self.base_path = self.path + '/' + self.name + '/'
        # count the number of frames in the path 
        self.gt_left_file_path = self.base_path + '/pose_left.txt'
        self.max_frame_id = sum(1 for line in open(self.gt_left_file_path))
        self.num_frames = self.max_frame_id
                
        self.left_color_path = self.base_path + '/image_left'
        self.right_color_path = self.base_path + '/image_right'
        self.left_depth_path = self.base_path + '/depth_left'
        self.right_depth_path = self.base_path + '/depth_right'        
        print(f'Number of frames: {self.max_frame_id}')  

    def getImage(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:
            file = self.left_color_path + f'/{str(frame_id).zfill(6)}_left.png'
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
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
            file = self.left_depth_path + f'/{str(frame_id).zfill(6)}_left_depth.npy'
            img = np.load(file)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
            self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img 
    
    def getImageRight(self, frame_id):
        img = None
        # NOTE: frame_id is already shifted by start_frame_id in Dataset.getImageColor()
        if frame_id < self.max_frame_id:        
            file = self.right_color_path + f'/{str(frame_id).zfill(6)}_right.png'
            img = cv2.imread(file)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
            self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None          
        return img     
    
    def getDepthRight(self, frame_id):
        if self.sensor_type == SensorType.MONOCULAR:
            return None # force a monocular camera if required (to get a monocular tracking even if depth is available)
        frame_id += self.start_frame_id
        img = None
        if frame_id < self.max_frame_id:
            file = self.right_depth_path + f'/{str(frame_id).zfill(6)}_right_depth.npy'
            img = np.load(file)
            self.is_ok = (img is not None)
            self._timestamp = frame_id * self.Ts
            self._next_timestamp = self._timestamp + self.Ts                
        else:
            self.is_ok = False      
            self._timestamp = None                       
        return img     