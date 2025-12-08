import numpy as np
from pyslam.utilities.geometry import poseRt


class Sim3Pose:
    def __init__(
        self,
        R: np.ndarray = np.eye(3, dtype=float),
        t: np.ndarray = np.zeros(3, dtype=float),
        s: float = 1.0,
    ):
        self.R = R
        self.t = t.reshape(3, 1)
        self.s = s
        assert s > 0
        self._T = None
        self._inv_T = None

    def __repr__(self):
        return f"Sim3Pose({self.R}, {self.t}, {self.s})"

    def from_matrix(self, T):
        if isinstance(T, np.ndarray):
            R = T[:3, :3]
            # Compute scale as the average norm of the rows of the rotation matrix
            # self.s = np.mean([np.linalg.norm(R[i, :]) for i in range(3)])
            row_norms = np.linalg.norm(R, axis=1)
            self.s = row_norms.mean()
            self.R = R / self.s
            self.t = T[:3, 3].reshape(3, 1)
        else:
            raise ValueError(f"Input T is not a numpy array. T={T}")
        return self

    def from_se3_matrix(self, T):
        if isinstance(T, np.ndarray):
            self.s = 1.0
            self.R = T[:3, :3]
            self.t = T[:3, 3].reshape(3, 1)
        else:
            raise ValueError(f"Input T is not a numpy array. T={T}")
        return self

    # corresponding homogeneous transformation matrix (4x4)
    def matrix(self):
        if self._T is None:
            self._T = poseRt(self.R * self.s, self.t.ravel())
        return self._T

    def inverse(self):
        return Sim3Pose(self.R.T, -1.0 / self.s * self.R.T @ self.t, 1.0 / self.s)

    # corresponding homogeneous inverse transformation matrix (4x4)
    def inverse_matrix(self):
        if self._inv_T is None:
            self._inv_T = np.eye(4)
            sR_inv = 1.0 / self.s * self.R.T
            self._inv_T[:3, :3] = sR_inv
            self._inv_T[:3, 3] = -sR_inv @ self.t.ravel()
        return self._inv_T

    def to_se3_matrix(self):
        return poseRt(self.R, self.t.squeeze() / self.s)  # [R t/s;0 1]

    def copy(self):
        return Sim3Pose(self.R.copy(), self.t.copy(), self.s)

    # map a 3D point
    def map(self, p3d):
        return self.s * self.R @ p3d.reshape(3, 1) + self.t

    # map a set of 3D points [Nx3]
    def map_points(self, points):
        return (self.s * self.R @ points.T + self.t).T

    # Define the @ operator
    def __matmul__(self, other):
        result = None
        if isinstance(other, Sim3Pose):
            # Perform matrix multiplication within the class
            s_res = self.s * other.s
            R_res = self.R @ other.R
            t_res = self.s * self.R @ other.t + self.t
            result = Sim3Pose(R_res, t_res, s_res)
        elif isinstance(other, np.ndarray):
            if other.shape == (4, 4):
                # Perform matrix multiplication with numpy (4x4) matrix
                R_other = other[:3, :3]  # before scaling
                s_other = np.mean([np.linalg.norm(R_other[i, :]) for i in range(3)])
                R_other = R_other / s_other
                t_other = other[:3, 3].reshape(3, 1)
                s_res = self.s * s_other
                R_res = self.R @ R_other
                t_res = self.s * self.R @ t_other + self.t
                result = Sim3Pose(R_res, t_res, s_res)
            # elif (other.ndim == 1 and other.shape[0] == 3) or \
            #      (other.ndim == 2 and other.shape in [(3, 1), (1, 3)]):
            #     # Perform matrix multiplication with numpy (3x1) vector
            #     result = self.s * self.R @ other + self.t
            else:
                raise TypeError(
                    f"Unsupported operand type(s) for @: '{type(self)}' and '{type(other)}' with shape {other.shape}"
                )
        else:
            raise TypeError(
                "Unsupported operand type(s) for @: '{}' and '{}'".format(type(self), type(other))
            )
        return result

    def __str__(self):
        return f"Sim3Pose(R={self.R}, t={self.t}, s={self.s})"
