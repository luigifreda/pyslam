import numpy as np
import pyslam.slam.cpp as cpp_module


CKDTree2d = cpp_module.CKDTree2d
CKDTree3d = cpp_module.CKDTree3d
CKDTreeDyn = cpp_module.CKDTreeDyn

if __name__ == "__main__":

    P2 = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2]], dtype=np.float64)
    k2 = CKDTree2d(P2)

    dists, idx = k2.query(np.array([0.9, 0.9]), k=2)  # -> (array([0.141..., ...]), array([3, 1]))
    idx_r = k2.query_ball_point(np.array([0.9, 0.9]), r=0.25)  # -> indices as np.int64

    P3 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)
    k3 = CKDTree3d(P3)
    print(k3.n, k3.d)  # 3, 3

    PM = np.random.randn(1000, 5).astype(np.float64)
    km = CKDTreeDyn(PM)
    d, i = km.query(np.random.randn(5), k=8)
