# gtsam_factors 

**Author**: Luigi Freda

This code defines a custom **pybind11** module for **pySLAM**, wrapping additional **GTSAM** helpers and factors that are not included in the original codebase.

## Build 

First build and install gtsam along with its Python bindings:
```
<pyslam_root>/install_gtsam.h
```

Then, open a terminal in this folder and run 
```
. <pyslam_root>/pyenv-activate.sh
./buil.sh
```