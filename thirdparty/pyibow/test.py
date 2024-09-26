import os
import sys 
sys.path.append("./lib")

import pyibow as bow

voc = bow.Vocabulary()
print(f'loading vocabulary...')
if not os.path.exists("ORBvoc.txt"):
    print(f'cannot find vocab file, download it from https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Vocabulary')
    sys.exit(0)
voc.load("ORBvoc.txt")
print(f'...done')
db = bow.Database()
db.setVocabulary(voc)

# # extract features using OpenCV
# ...
# # add features to database
# for features in features_list:
#    db.add(features)

# # query features
# feature_to_query = 1
# results = db.query(features_list[feature_to_query])