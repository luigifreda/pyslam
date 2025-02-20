## Pose Graph Optimization / Rotation Averaging

Pose Graph Optimization (PGO) is the problem of estimating the global trajectory from a set of relative pose measurements. PGO is typically performed using nonlinear least-squares algorithms (e.g Levenberg-Marquardt) and requires a good initialization in order to converge.

In this experiment, we implement Riemannian Gradient Descent with a reshaping function (Tron et al. 2012). The algorithm is implemented in the function `gradient_initializer` and runs on the GPU using lietorch.

### Running on a .g2o file

Download a 3D problem from [datasets](https://lucacarlone.mit.edu/datasets/) (our implementation currently only supports uniform information matricies in Sphere-A, Torus, Cube, and Garage).

Then run the `gradient_initializer` on the problem
```python
python main.py --problem=torus3D.g2o --steps=500
```

The output graph, `torus3D_rotavg.g2o`, can then be used as the initialization for non-linear least squares optimizers such as `ceres`, `g2o`, and `gtsam`.
