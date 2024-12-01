import os
import sys 
sys.path.append("./lib")

import pydbow2 as bow

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../../'
kDataFolder = kRootFolder + '/data'


if __name__ == "__main__":

    voc = bow.BinaryVocabulary()
    print(f'loading vocabulary...')
    vocabulary_file = kDataFolder + "/ORBvoc.txt"
    if not os.path.exists(vocabulary_file):
        print(f'cannot find vocab file, download it from https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary')
        sys.exit(0)
    voc.load(vocabulary_file)
    
    print('saving in boost format')
    voc.save(filename=kDataFolder + "/ORBvoc.dbow2", use_boost=True)
    print(f'...done')
    
    print(f'now trying to reload the vocabulary in boost format')
    voc2 = bow.BinaryVocabulary()
    voc2.load(filename=kDataFolder + "/ORBvoc.dbow2", use_boost=True)
    print(f'...done')

    # # extract features using OpenCV
    # ...
    # # add features to database
    # for features in features_list:
    #    db.add(features)

    # # query features
    # feature_to_query = 1
    # results = db.query(features_list[feature_to_query])