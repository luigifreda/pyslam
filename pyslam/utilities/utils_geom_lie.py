"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

"""
Provides functions for Lie group calculations.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import typing

import numpy as np
import scipy.spatial.transform as sst
from packaging.version import Version
from scipy import __version__ as scipy_version

from evo import EvoException
from evo.core import transformations as tr

# scipy.spatial.transform.Rotation.*_matrix() was introduced in 1.4,
# which is not available for Python 2.7.
# Use the legacy direct cosine matrix naming (*_dcm()) if needed.
# TODO: remove this junk once Python 2.7 is finally dead in ROS.
_USE_DCM_NAME = Version(scipy_version) < Version("1.4")


class LieAlgebraException(EvoException):
    pass


def _sst_rotation_from_matrix(so3_matrices: np.ndarray):
    """
    Helper for creating scipy.spatial.transform.Rotation
    from 1..n SO(3) matrices.
    :return: scipy.spatial.transform.Rotation
    """
    if _USE_DCM_NAME:
        return sst.Rotation.from_dcm(so3_matrices)
    else:
        return sst.Rotation.from_matrix(so3_matrices)


def hat(v: np.ndarray) -> np.ndarray:
    """
    :param v: 3x1 vector
    :return: 3x3 skew symmetric matrix
    """
    # yapf: disable
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])
    # yapf: enable


def vee(m: np.ndarray) -> np.ndarray:
    """
    :param m: 3x3 skew symmetric matrix
    :return: 3x1 vector
    """
    return np.array([-m[1, 2], m[0, 2], -m[0, 1]])


def so3_exp(rotation_vector: np.ndarray):
    """
    Computes an SO(3) matrix from a rotation vector representation.
    :param axis: 3x1 rotation vector (axis * angle)
    :return: SO(3) rotation matrix (matrix exponential of so(3))
    """
    if _USE_DCM_NAME:
        return sst.Rotation.from_rotvec(rotation_vector).as_dcm()
    else:
        return sst.Rotation.from_rotvec(rotation_vector).as_matrix()


def so3_log(r: np.ndarray, return_skew: bool = False) -> np.ndarray:
    """
    :param r: SO(3) rotation matrix
    :param return_skew: return skew symmetric Lie algebra element
    :return:
            rotation vector (axis * angle)
        or if return_skew is True:
             3x3 skew symmetric logarithmic map in so(3) (Ma, Soatto eq. 2.8)
    """
    if not is_so3(r):
        raise LieAlgebraException(
            f"matrix is not a valid SO(3) group element: {r}, det: {np.linalg.det(r)}"
        )
    rotation_vector = _sst_rotation_from_matrix(r).as_rotvec()
    if return_skew:
        return hat(rotation_vector)
    else:
        return rotation_vector


def so3_log_angle(r: np.ndarray, degrees: bool = False) -> float:
    """
    :param r: SO(3) rotation matrix
    :param degrees: whether to return in degrees, default is radians
    :return: the rotation angle of the logarithmic map
    """
    angle = np.linalg.norm(so3_log(r, return_skew=False))
    if degrees:
        angle = np.rad2deg(angle)
    return float(angle)


def is_so3(r: np.ndarray) -> bool:
    """
    :param r: a 3x3 matrix
    :return: True if r is in the SO(3) group
    """
    # Check the determinant.
    det_valid = np.allclose(np.linalg.det(r), [1.0], atol=1e-6)
    # Check if the transpose is the inverse.
    inv_valid = np.allclose(r.transpose().dot(r), np.eye(3), atol=1e-6)
    return det_valid and inv_valid


def random_so3() -> np.ndarray:
    """
    :return: a random SO(3) matrix (for debugging)
    """
    return tr.random_rotation_matrix()[:3, :3]
