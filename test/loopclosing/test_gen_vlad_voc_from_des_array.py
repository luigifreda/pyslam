import os
import argparse
import sys 
sys.path.append("../../")

from config import Config
config = Config()

import math
import cv2 
import numpy as np

from vlad import VLAD

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/VLADvoc_test.txt'

kMaxNumDescriptors = int(1e8)

def extract_random(descriptors, num_descriptors):
    if descriptors.shape[0] < num_descriptors:
        return descriptors
    else:
        return descriptors[np.random.randint(0,descriptors.shape[0],num_descriptors)]


def main(input_file, output_file):

    num_clusters = 8
    
    print(f'loading descriptors array from {input_file} ...')
    # load descriptors array from a file
    descriptors = np.load(input_file)
    desc_dim = descriptors.shape[1]
    print(f'loaded descriptors array of shape {descriptors.shape}')

    # generate vlad vocabulary
    voc = VLAD(desc_dim=desc_dim,num_clusters=num_clusters)
    print(f'generating vocabulary...')
    used_descriptors = extract_random(descriptors,kMaxNumDescriptors)
    print(f'used descriptors array of shape {used_descriptors.shape}')
        
    voc.fit(used_descriptors)
    print(f'saving vocabulary to {output_file} ...')
    voc.save(output_file)
    print(f'...done')
    
    # reload the vocabulary 
    print('testing reloading vocabulary')
    voc2 = VLAD(desc_dim=desc_dim,num_clusters=num_clusters)
    voc2.load(output_file)
    print('...done')
    

# load descriptors array from a file and generate vocabulary      
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i","--input_file", required=False, type=str, default=kDataFolder+"/orb_descriptors_kitti.npy", help="Path to your descriptors array")
    argparser.add_argument("-o","--output_file", required=False, type=str, default=kOrbVocabFile, help="Path to save the vocabulary")
    args = argparser.parse_args()

    main(args.input_file, args.output_file)