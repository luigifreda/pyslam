# Typedef MapPoint as python or C++ implementation

Import — by choosing at import time which implementation is the one exposed as MapPoint.

The idea: in a single module (say mappoint.py), you define which class name is bound to MapPoint depending on a flag, config, or environment variable.

# `mappoint.py` file

```python
import os

# Decide based on environment variable
USE_CPP = os.getenv("USE_CPP_MAPPOINT", "0") == "1"

if USE_CPP:
    from my_cpp_module import MapPointCpp as MapPoint
else:
    from my_python_module import MapPoint as MapPoint
```

Then in the rest of your code:

```python
from mappoint import MapPoint

p = MapPoint(...)
```

✅ This way:

You don’t maintain a wrapper.

MapPoint always refers to the implementation you chose.

The rest of your code stays agnostic to whether you’re using the Python or C++ backend.