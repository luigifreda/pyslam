import os
import sys 
sys.path.append("../../")

from config import Config
config = Config()

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopClosuresImgs

import math
import cv2 
import numpy as np

from dataset import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

config.set_lib('pydbow3')
import pydbow3 as dbow


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kDataFolder = kRootFolder + '/data'
kVocabFile = kDataFolder + '/ORBvoc.txt'
#kVocabFile = kDataFolder + '/orbvoc.dbow3'


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000

    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    voc = dbow.Vocabulary()
    print(f'loading vocabulary...')
    if not os.path.exists(kVocabFile):
        gdrive_url = 'https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF'
        gdrive_download_lambda(url=gdrive_url, path=kVocabFile)
    voc.load(kVocabFile)
    print(f'...done')
    db = dbow.Database()
    db.setVocabulary(voc)
    
    # to nicely visualize current loop candidates in a single image
    loop_closures = LoopClosuresImgs()
    
    # init the similarity matrix
    S_float = np.empty([dataset.num_frames, dataset.num_frames], 'float32')
    S_color = np.empty([dataset.num_frames, dataset.num_frames, 3], 'uint8')
    #S_color = np.full([dataset.num_frames, dataset.num_frames, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
    
    cv2.namedWindow('S', cv2.WINDOW_NORMAL)
        
    img_count = 0
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.isOk():

        timestamp = dataset.getTimestamp()          # get current timestamp 
        img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
            
            loop_closures.reset()
                       
            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(img)   # with DL matchers this a null operation 
            # add image descriptors to database
            db.add(des)
                       
            if img_count >= 1:
                results = db.query(des, max_results=kMaxResultsForLoopClosure+1) # we need plus one to eliminate the best trivial equal to img_id
                for r in results:
                    float_value = r.Score * 255
                    color_value = float_to_color(r.Score)
                    S_float[img_id, r.Id] = float_value
                    S_float[r.Id, img_id] = float_value
                    S_color[img_id, r.Id] = color_value
                    S_color[r.Id, img_id] = color_value
                    # visualize non-trivial loop closures: we check the query results are not too close to the current image
                    if abs(r.Id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                        print(f'result - best id: {r.Id}, score: {r.Score}')
                        loop_img = dataset.getImageColor(r.Id)
                        loop_closures.add(loop_img, r.Id, r.Score)

            font_pos = (50, 50)                   
            cv2.putText(img, f'id: {img_id}', font_pos, LoopClosuresImgs.kFont, LoopClosuresImgs.kFontScale, \
                        LoopClosuresImgs.kFontColor, LoopClosuresImgs.kFontThickness, cv2.LINE_AA)     
            cv2.imshow('img', img)
            
            cv2.imshow('S', S_color)            
            #cv2.imshow('S', convert_float_to_colored_uint8_image(S_float))
            
            if loop_closures.candidates is not None:
                cv2.imshow('loop_closures', loop_closures.candidates)
            
            cv2.waitKey(1)
        else: 
            getchar()
            
        img_id += 1
        img_count += 1