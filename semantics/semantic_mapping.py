"""
* This file is part of PYSLAM 
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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
import os
import time
import cv2
import numpy as np

from collections import defaultdict 

from threading import RLock, Thread, Condition
from queue import Queue 

from config_parameters import Parameters  

from semantic_segmentation_factory import SemanticSegmentationType, semantic_segmentation_factory
from semantic_fusion_methods import average_fusion, bayesian_fusion, count_labels

from semantic_types import SemanticFeatureType
from semantic_utils import SemanticDatasetType, information_weights_factory, labels_color_map_factory, single_label_to_color, similarity_heatmap_point
from timer import TimerFps
from utils_serialization import SerializableEnum, register_class
from utils_sys import Printer, Logging
from utils_mp import MultiprocessingManager
from utils_data import empty_queue

from qimage_thread import QimageViewer
import traceback
import platform

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from slam import Slam # Only imported when type checking, not at runtime
    from keyframe import KeyFrame

kVerbose=True     
kTimerVerbose = False

kSemanticMappingSleepTime = 5e-3  # [s]

kSemanticMappingOnSeparateThread = Parameters.kSemanticMappingOnSeparateThread 
kSemanticMappingDebugAndPrintToFile = Parameters.kSemanticMappingDebugAndPrintToFile

@register_class
class SemanticMappingType(SerializableEnum):
    DENSE = 0 # Pixel-wise segmentation to points maps

def semantic_mapping_factory(slam: 'Slam', headless=False, 
                             image_size=(512, 512),
                             **kwargs):

    semantic_mapping_type = kwargs.get('semantic_mapping_type')
    if semantic_mapping_type is None:
        raise ValueError('semantic_mapping_type is not specified in semantic_mapping_config')
    
    if semantic_mapping_type == SemanticMappingType.DENSE:
        #TODO(dvdmc): fix this with a better approach that checks for existence of configs
        return SemanticMappingDense(slam=slam, headless=headless,  
                                   image_size=image_size,
                                   semantic_segmentation_type=kwargs.get('semantic_segmentation_type'),
                                   semantic_dataset_type=kwargs.get('semantic_dataset_type'),
                                   semantic_feature_type=kwargs.get('semantic_feature_type'))
    else:
        raise ValueError(f'Invalid semantic mapping type: {semantic_mapping_type}')
    
# TODO(dvdmc): some missing features are:
# - Warm start when loading a new local map (not tested)
class SemanticMappingBase:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(self, slam: 'Slam', semantic_mapping_type):
        self.slam = slam
        self.semantic_mapping_type = semantic_mapping_type

        self.queue = Queue()
        self.queue_condition = Condition()
        self.work_thread = None #Thread(target=self.run)
        self.is_running = False

        self._is_idle = True 
        self.idle_codition = Condition()

        self.stop_requested = False
        self.stopped = False
        self.stop_mutex = RLock()

        self.reset_requested = False
        self.reset_mutex = RLock()

        self.last_processed_kf_img_id = None        

        self.init_print()
    
    def init_print(self):
        if kVerbose:
            if kSemanticMappingOnSeparateThread: 
                if kSemanticMappingDebugAndPrintToFile:
                    # Default log to file: logs/semantic_mapping.log
                    logging_file = os.path.join(Parameters.kLogsFolder, 'semantic_mapping.log')
                    SemanticMappingBase.local_logger = Logging.setup_file_logger(
                        'semantic_mapping_logger',
                        logging_file,
                        formatter=Logging.simple_log_formatter
                    )
                    def file_print(*args, **kwargs):
                        message = ' '.join(str(arg) for arg in args)
                        SemanticMappingBase.local_logger.info(message, **kwargs)
                else:
                    def file_print(*args, **kwargs):
                        message = ' '.join(str(arg) for arg in args)
                        return print(message, **kwargs)
                SemanticMappingBase.print = staticmethod(file_print)

    @property
    def map(self):
        return self.slam.map
        
    @property
    def sensor_type(self):
        return self.slam.sensor_type
    
    def request_reset(self):
        SemanticMappingBase.print('SemanticMapping: Requesting reset...')
        if self.reset_requested:
            SemanticMappingBase.print('SemanticMapping: reset already requested...')
            return
        with self.reset_mutex:
            self.reset_requested = True
        while True:
            with self.queue_condition:
                self.queue_condition.notifyAll() # to unblock self.pop_keyframe()              
            with self.reset_mutex:
                if not self.reset_requested:
                    break
            time.sleep(0.1)
            SemanticMappingBase.print('SemanticMapping: waiting for reset...')
        SemanticMappingBase.print('SemanticMapping: ...Reset done.')            
            
    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                SemanticMappingBase.print('SemanticMapping: reset_if_requested() starting...')            
                empty_queue(self.queue)
                self.reset_requested = False
                SemanticMappingBase.print('SemanticMapping: reset_if_requested() ...done')                   
        
    def start(self):
        SemanticMappingBase.print(f'SemanticMapping: starting...')
        self.work_thread = Thread(target=self.run)
        self.work_thread.start()

    def quit(self):
        SemanticMappingBase.print('SemanticMapping: quitting...')
        if self.is_running and self.work_thread is not None:
            self.is_running = False
            self.work_thread.join(timeout=5)
        SemanticMappingBase.print('SemanticMapping: done')   
    
    # push the new keyframe and its image into the queue
    # TODO(dvdmc): currently, we execute semantic mapping in a Thread.
    # VolumetricIntegrator and LoopDetection uses MultiProcessing (mp) and
    # sends tasks. In the future, we should move this to mp.
    # Check volumetric_integrator.py "add_keyframe" and "flush_keyframe_queue"
    def push_keyframe(self, keyframe, img=None, img_right=None, depth=None):
        with self.queue_condition:
            self.queue.put((keyframe,img,img_right,depth))      
            self.queue_condition.notifyAll() 

    # blocking call
    def pop_keyframe(self, timeout=Parameters.kSemanticMappingTimeoutPopKeyframe):
        with self.queue_condition:        
            if self.queue.empty():
                while self.queue.empty() and not self.stop_requested and not self.reset_requested:
                    ok = self.queue_condition.wait(timeout=timeout)
                    if not ok: 
                        break # Timeout occurred
                    #SemanticMappingBase.print('SemanticMapping: waiting for keyframe...')
        if self.queue.empty() or self.stop_requested:
            return None
        try:
            return self.queue.get(timeout=timeout)
        except Exception as e:
            SemanticMappingBase.print(f'SemanticMapping: pop_keyframe: encountered exception: {e}')
            return None
        
    def queue_size(self):
        return self.queue.qsize()

    def is_idle(self):
        with self.idle_codition: 
            return self._is_idle
                    
    def set_idle(self, flag):
        with self.idle_codition: 
            self._is_idle = flag
            self.idle_codition.notifyAll() 
            
    def wait_idle(self, print=print, timeout=None): 
        if self.is_running == False:
            return
        with self.idle_codition:
            while not self._is_idle and self.is_running:
                SemanticMappingBase.print('SemanticMapping: waiting for idle...')
                ok = self.idle_codition.wait(timeout=timeout)
                if not ok:
                    Printer.yellow(f'SemanticMapping: timeout {timeout}s reached, quit waiting for idle')
                    return

    def request_stop(self):
        with self.stop_mutex:
            Printer.yellow('requesting a stop for semantic mapping') 
            self.stop_requested = True
        with self.queue_condition:
            self.queue_condition.notifyAll() # to unblock self.pop_keyframe()  
        
    def is_stop_requested(self):
        with self.stop_mutex:         
            return self.stop_requested    
    
    def stop_if_requested(self):
        with self.stop_mutex:        
            if self.stop_requested and not self.stopped:
                self.stopped = True
                SemanticMappingBase.print('SemanticMapping: stopped...')
                return True
            return False
        
    def is_stopped(self):
        with self.stop_mutex:         
            return self.stopped
    
    def set_not_stop(self, value):
        with self.stop_mutex:              
            if value and self.stopped:
                return False 
            self.stopped = value
            return True
               
    def release(self):   
        if not self.is_running:
            return 
        with self.stop_mutex:              
            self.stopped = False
            self.stop_requested = False
            # emtpy the queue
            while not self.queue.empty():
                self.queue.get()
            self.set_idle(True)
            SemanticMappingBase.print(f'SemanticMapping: released...')
        
    def run(self):
        self.is_running = True
        while self.is_running:
            self.step()
        empty_queue(self.queue) # empty the queue before exiting
        SemanticMappingBase.print('SemanticMapping: loop exit...')

    # Depending on the implementation the step might just add semantics to new frames, keyframes or it might
    # segment objects and track 3D segments
    def step(self):
        if self.map.num_keyframes() > 0:            
            if not self.stop_requested:              
                ret = self.pop_keyframe() # blocking call
                if ret is not None: 
                    self.kf_cur, self.img_cur, self.img_cur_right, self.depth_cur = ret
                    if self.kf_cur is not None:
                        self.last_processed_kf_img_id = self.kf_cur.img_id
                                                                                    
                        self.set_idle(False) 
                        try: 
                            self.do_semantic_mapping()
                        except Exception as e:
                            SemanticMappingBase.print(f'SemanticMapping: encountered exception: {e}')
                            SemanticMappingBase.print(traceback.format_exc())
                        self.set_idle(True)
                        
            elif self.stop_if_requested():
                self.set_idle(True)
                while self.is_stopped():
                    SemanticMappingBase.print(f'SemanticMapping: stopped, idle: {self._is_idle} ...')
                    time.sleep(kSemanticMappingSleepTime)              
        else:
            msg = 'SemanticMapping: waiting for keyframes...'
            #Printer.red(msg)
            #SemanticMappingBase.print(msg)
            time.sleep(kSemanticMappingSleepTime)
        self.reset_if_requested()

    def semantic_mapping_impl(self):
        """
            Semantic mapping implementations assume as possible inputs new KF, RGB image and depth image
            They are provided in self.kf_cur, self.img_cur and self.depth_cur
        """
        raise NotImplementedError
    
    def do_semantic_mapping(self):
        SemanticMappingBase.print('semantic mapping: starting...')                        
                          
        Printer.cyan('@semantic mapping')
        time_start = time.time()
                                        
        if self.kf_cur is None:
            Printer.red('semantic mapping: no keyframe to process')
            return
                        
        if kSemanticMappingOnSeparateThread: 
            SemanticMappingBase.print('..................................')
            SemanticMappingBase.print('processing KF: ', self.kf_cur.id, ', queue size: ', self.queue_size())   
        
        self.semantic_mapping_impl()

        elapsed_time = time.time() - time_start
        self.time_semantic_mapping = elapsed_time
        SemanticMappingBase.print(f'semantic mapping elapsed time: {elapsed_time}')

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        return NotImplementedError
    
    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return NotImplementedError
    
    def get_semantic_weight(self, semantic_des):
        return NotImplementedError
    
    #TODO(dvdmc): Missing save and load functions

class SemanticMappingDense(SemanticMappingBase):
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
    
    # TODO(dvdmc): move to types since this is static. Discuss if we want to support different fusion methods for the same semantic feature type
    feature_type_configs = {
        SemanticFeatureType.LABEL: count_labels, 
        SemanticFeatureType.PROBABILITY_VECTOR: bayesian_fusion,
        SemanticFeatureType.FEATURE_VECTOR: average_fusion
    }
    
    def __init__(self, slam: 'Slam', semantic_segmentation_type=SemanticSegmentationType.SEGFORMER, 
                 semantic_dataset_type=SemanticDatasetType.CITYSCAPES, semantic_feature_type=SemanticFeatureType.LABEL, image_size=(512, 512), headless=False):
        
        if semantic_feature_type not in self.feature_type_configs:
            raise ValueError(f'Invalid semantic feature type: {semantic_feature_type}')
        self.semantic_feature_type = semantic_feature_type
        self.semantic_fusion_method = self.feature_type_configs[semantic_feature_type]

        self.semantic_segmentation_type = semantic_segmentation_type
        self.semantic_segmentation = semantic_segmentation_factory(semantic_segmentation_type=semantic_segmentation_type, 
                                                                   semantic_feature_type=self.semantic_feature_type, 
                                                                   semantic_dataset_type=semantic_dataset_type, 
                                                                   image_size=image_size)
        Printer.green(f'semantic_segmentation_type: {semantic_segmentation_type.name}')
        
        self.semantic_dataset_type = semantic_dataset_type
        if semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY:
            self.semantics_color_map = labels_color_map_factory(semantic_dataset_type)
            self.semantic_sigma2_factor = information_weights_factory(semantic_dataset_type)
        else:
            self.semantics_color_map = None
            self.semantic_sigma2_factor = [1.0]

        self.timer_verbose = kTimerVerbose
        self.timer_inference = TimerFps('Inference', is_verbose=self.timer_verbose)
        self.timer_update_keyframe = TimerFps('Update KeyFrame', is_verbose = self.timer_verbose)     
        self.timer_update_mappoints = TimerFps('Update MapPoints', is_verbose = self.timer_verbose)     

        self.headless = headless
        self.draw_semantic_mapping_init = False

        super().__init__(slam, SemanticMappingType.DENSE)
        
    def semantic_mapping_impl(self):

        # do dense semantic segmentation inference
        self.timer_inference.start()        
        self.curr_semantic_prediction = self.semantic_segmentation.infer(self.img_cur)    
        self.timer_inference.refresh() 
        Printer.green(f'#semantic inference, timing: {self.timer_inference.last_elapsed}')
        # TODO(dvdmc): the prints don't work for some reason. They block the Thread   
        # SemanticMappingDense.print(f'#semantic inference, timing: {self.timer_pts_culling.last_elapsed}')                 

        # update keypoints of current keyframe
        self.timer_update_keyframe.start()
        self.kf_cur.set_semantics(self.curr_semantic_prediction)
        self.timer_update_keyframe.refresh()
        Printer.green(f'#set KF semantics, timing: {self.timer_update_keyframe.last_elapsed}')
        # SemanticMappingDense.print(f'#keypoints: {self.kf_cur.num_keypoints()}, timing: {self.timer_update_keyframe.last_elapsed}') 
        
        # update map points of current keyframe
        self.timer_update_mappoints.start()
        self.update_map_points()
        self.timer_update_mappoints.refresh()
        Printer.green(f'#set MPs semantics, timing: {self.timer_update_mappoints.last_elapsed}')
        # SemanticMappingDense.print(f'#map points: {self.kf_cur.num_points()}, timing: {self.timer_update_mappoints.last_elapsed}')

        self.draw_semantic_prediction()

    def draw_semantic_prediction(self):
        if self.headless:
            return
        draw = False
        use_cv2_for_drawing = platform.system() != 'Darwin' # under mac we can't use cv2 imshow here

        if self.curr_semantic_prediction is not None:
            if not self.draw_semantic_mapping_init:
                if use_cv2_for_drawing:
                    cv2.namedWindow('semantic prediction') # to get a resizable window
                self.draw_semantic_mapping_init = True
            draw = True
            semantic_color_img = self.semantic_segmentation.to_rgb(self.curr_semantic_prediction, bgr=True)
            if use_cv2_for_drawing:
                cv2.imshow("semantic prediction", semantic_color_img)
            else:
                QimageViewer.get_instance().draw(semantic_color_img, 'semantic prediction')

        if draw:
            if use_cv2_for_drawing: 
                cv2.waitKey(1)  

    def update_map_points(self):
        kf_cur_points = self.kf_cur.get_points()
        for idx,p in enumerate(kf_cur_points):
            if p is not None:
                p.update_semantics(self.semantic_fusion_method)

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return single_label_to_color(semantic_des, self.semantics_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return single_label_to_color(np.argmax(semantic_des, axis=-1), self.semantics_color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                sims = self.semantic_segmentation.features_to_sims(semantic_des)
                return similarity_heatmap_point(sims, colormap=cv2.COLORMAP_JET, sim_scale=self.semantic_segmentation.sim_scale, bgr=bgr)
            else:
                label = self.semantic_segmentation.features_to_labels(semantic_des)
                return single_label_to_color(label, self.semantics_color_map, bgr=bgr)
            
    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_segmentation.to_rgb(semantic_img, bgr=bgr)
        
    def get_semantic_weight(self, semantic_des):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return self.semantic_sigma2_factor[semantic_des]
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return self.semantic_sigma2_factor[np.argmax(semantic_des, axis=-1)]
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            return self.semantic_sigma2_factor[self.semantic_segmentation.features_to_labels(semantic_des)]
        
    def set_query_word(self, query_word):
        if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.semantic_segmentation.set_query_word(query_word)
        else:
            raise NotImplementedError