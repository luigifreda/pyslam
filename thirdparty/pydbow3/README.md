# PyDBoW3


This is an updated and customized version of https://github.com/JHMeusener/PyDBoW3.
Updated and customized by Luigi Freda for pySLAM.


## Get started

```python
import pydbow3 as bow
voc = bow.Vocabulary()
voc.load("ORBvoc.txt")
db = bow.Database()
db.setVocabulary(voc)
# extract features using OpenCV
...
# add features to database
for features in features_list:
   db.add(features)

# query features
feature_to_query = 1
results = db.query(features_list[feature_to_query])
```

## Prerequisites:
* OpenCV 
* CMake 
* DBoW3

## Install

+ clone the repo with `--recursive`
+ build and install `modules/DBoW3`
+ install with `pip install .` 
+ currently only `*nix` is supported

## Acknowledgement
Fork from https://github.com/xingruiy/PyDBoW3
This work is based on https://github.com/foxis/pyDBoW3
and https://github.com/edmBernard/ybind11_opencv_numpy.git
