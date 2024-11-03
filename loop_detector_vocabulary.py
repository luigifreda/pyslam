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


import os
import time
import math 
import numpy as np
import cv2
from enum import Enum

from utils_sys import getchar, Printer 

from parameters import Parameters

from utils_files import gdrive_download_lambda 
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes


kVerbose = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'


if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detector_base import print
    

class VocabularyData:
    def __init__(self, vocab_file_path=None, descriptor_type=None, descriptor_dimension=None, url_vocabulary=None, url_type=None):
        self.vocab_file_path = vocab_file_path
        self.descriptor_type = descriptor_type
        self.url_vocabulary = url_vocabulary
        self.url_type = url_type
        self.descriptor_dimension=descriptor_dimension
        
    def check_download(self):
        if self.url_vocabulary is not None and not os.path.exists(self.vocab_file_path):
            if self.url_type == 'gdrive':
                gdrive_url = self.url_vocabulary
                Printer.blue(f'VocabularyData: downloading vocabulary {self.descriptor_type.name} from: {gdrive_url}')
                gdrive_download_lambda(url=gdrive_url, path=self.vocab_file_path)  
        if self.vocab_file_path is not None and not os.path.exists(self.vocab_file_path):
            Printer.red(f'VocabularyData: cannot find vocabulary file: {self.vocab_file_path}')
            raise FileNotFoundError      
        

class OrbVocabularyData(VocabularyData):
    kOrbVocabFile = kDataFolder + '/ORBvoc.txt'
    def __init__(self, vocab_file_path=kOrbVocabFile,
                       descriptor_type=FeatureDescriptorTypes.ORB2,
                       descriptor_dimension=32,
                       url_vocabulary='https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF',
                       url_type='gdrive'):  # download it from gdrive
        super().__init__(vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type)
        

    
class VladVocabularyData(VocabularyData):
    kVladVocabFile = kDataFolder + '/VLADvoc_orb.txt'
    def __init__(self, vocab_file_path=kVladVocabFile,
                       descriptor_type=FeatureDescriptorTypes.ORB2,
                       descriptor_dimension=32,
                       url_vocabulary='https://drive.google.com/file/d/1u6AJEa2aZg7u5aS6vFX2qXiKKePQ_6t2',
                       url_type='gdrive'):  # download it from gdrive
        super().__init__(vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type)