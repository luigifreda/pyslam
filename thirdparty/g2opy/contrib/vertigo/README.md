# Vertigo — Versatile Extensions for RobusT Inference using Graph Optimization


Vertigo is an extension library for g2o [1] and gtsam 2.0 [5].

It provides a C++ implementation of the switchable constraints described in
[2, 3]. This extension enables g2o or gtsam to solve pose graph SLAM problems
in 2D and 3D despite a large number of false positive loop closure constraints.
	
In addition, a re-implementation of the max-mixture model described in [4]
is also contained.

Furthermore, Vertigo contains a number of standard pose graph SLAM datasets 
and a script to spoil them with false positive loop closure constraints.
These datasets have been used in the evaluations [2] and [3]. They can serve
as a set of benchmark datasets for future developments in robust pose graph SLAM.
	
Have fun, I hope this is useful.   
Hopefully, more stuff will be added to Vertigo in the future.   
Any comments, thoughts and patches are welcome and largely appreciated.


### Contact Information
Niko Sünderhauf  
niko.suenderhauf@roboticvision.org    
<http://nikosuenderhauf.info/>    


### Licence Information
Vertigo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


### How to compile

Dependencies:
-  Either one or both of g2o and gtsam. 
-  Vertigo has been tested with:
-  gtsam 2.0.0
-  g2o rev.19 (SVN from openslam.org)
 

All other dependencies are shared with g2o or gtsam. If you can compile them,
you should be fine.

Compilation follows the usual steps: Create a build directory and go there. 
Run cmake and then make.

-  `mkdir build`
-  `cd build`
-  `cmake ..`
-  `make`

This should create two libraries in the directory lib/ and the example in
examples/robustISAM2.


### Quick start guide using g2o

First compile the library. Then use datasets/generateDataset.py to generate a
dataset spoiled with false positive loop closure constraints. 
 
  `cd datasets`  
  `./generateDataset.py -i manhattan/originalDataset/Olson/manhattanOlson3500.g2o -s`

This creates a file called new.g2o which contains the original Manhattan dataset
with 100 additional random loop closure constraints. All loop closures are
switchable constraints. 
Now lets try to solve it using g2o_viewer.
  
  `g2o_viewer -typeslib ../lib/libvertigo-g2o.so new.g2o`

You should see a plot of the initial map, with the switchable loop closure
constraints in red. Increase the number of iterations to 30, keep Gauss-Newton
activated, and hit "Optimize".
The switchable constraints turn black as they are "switched off" and g2o
converges towards the correct map.


Let us explore some more options of the generateDataset.py script.
In addition to the call above, add the option `-n 500`. Now we have 500 outliers
instead of 100 as before. 

Add an additional `-l` to get local outliers.

Use `-n 10 -g 50` to get 10 groups of 50 outliers each.

For comparison, remove the option `-s`. Now you have the non-switchable default
constraints, which result in a useless map. 

The option `-m` instead of `-s` gives you the max-mixture model described in [4].
Notice that these constraints are originally drawn in orange and turn black when
the alternative null-hypothesis constraint is selected.


The option `--help` tells you more about the possible command line options. 

Especially notice that you will want to provide a reasonable information matrix
for the false positive loop closure constraints (usually compatible with the
matrix given for the true positive constraints), using the --information option.
You can either provide the full upper triangular form, e.g.
`--information=1,0,0,1,0,1` (for a 2D dataset) or just specify a single Value that
will be used for all diagonal entries of the matrix, e.g. `--information=1`



### Acknowledgements

The datasets were provided by / reproduced from:
- **Manhattan/Olson**   provided by Edwin Olson
- **Manhattan/g2o**     released as part of g2o
- **Intel**             released as part of g2o
- **Sphere2500**        released as part of iSAM1 (Michael Kaess)
- **City10000**         released as part of iSAM1 (Michael Kaess)



### References

[1] Rainer Kümmerle, Giorgio Grisetti, Hauke Strasdat, Kurt Konolige, and
    Wolfram Burgard: g2o: A General Framework for Graph Optimization, IEEE
    International Conference on Robotics and Automation (ICRA), 2011 
    Available online: http://openslam.org/g2o.html  

[2] Sünderhauf, N., Protzel, P. (2012). Switchable Constraints for Robust Pose
    Graph SLAM. Proc. of IEEE International Conference on Intelligent Robots and
    Systems (IROS), Vilamoura, Portugal.    

[3] Sünderhauf, N. (2012). Robust Optimization for Simultaneous Localization and
    Mapping. PhD Thesis, Chemnitz University of Technology.   

[4] Olson, E. and Agarwal, P. (2012). Inference on networks of mixtures for
    robust robot mapping. Proc. of Robotics: Science and Systems (RSS), Sydney,
    Australia.  

[5] GTSAM2.0 https://collab.cc.gatech.edu/borg/gtsam/  



