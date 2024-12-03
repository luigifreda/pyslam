import os
import time
import sys 
sys.path.append("./lib")

import pydbow3 as bow

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../../'
kDataFolder = kRootFolder + '/data'


if __name__ == "__main__":

    start = time.time() 
    voc = bow.Vocabulary()
    print(f'loading vocabulary...')
    vocabulary_file = kDataFolder + "/ORBvoc.txt"
    if not os.path.exists(vocabulary_file):
        print(f'cannot find vocab file, download it from https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary')
        sys.exit(0)
    voc.load(vocabulary_file)
    print(f'...done in {time.time() - start} seconds')
    
    print('saving in boost format')
    start = time.time()
    voc.save(path=kDataFolder + "/ORBvoc.dbow3", use_boost=True)
    print(f'...done in {time.time() - start} seconds')
    
    db = bow.Database()
    db.setVocabulary(voc)
    
    print(f'now trying to reload the vocabulary in boost format')
    start = time.time()
    voc2 = bow.Vocabulary()
    voc2.load(kDataFolder + "/ORBvoc.dbow3", use_boost=True)
    print(f'...done in {time.time() - start} seconds')

    # # extract features using OpenCV
    # ...
    # # add features to database
    # for features in features_list:
    #    db.add(features)

    # # query features
    # feature_to_query = 1
    # results = db.query(features_list[feature_to_query])