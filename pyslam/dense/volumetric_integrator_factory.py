import os

from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.system import Printer, import_from

try:
    from .volumetric_integrator_base import VolumetricIntegratorBase
except ImportError:
    VolumetricIntegratorBase = import_from(
        "pyslam.dense.volumetric_integrator_base",
        "VolumetricIntegratorBase",
    )
try:
    from .volumetric_integrator_tsdf import VolumetricIntegratorTsdf
except ImportError:
    VolumetricIntegratorTsdf = import_from(
        "pyslam.dense.volumetric_integrator_tsdf",
        "VolumetricIntegratorTsdf",
    )
try:
    from .volumetric_integrator_gaussian_splatting import VolumetricIntegratorGaussianSplatting
except ImportError:
    VolumetricIntegratorGaussianSplatting = import_from(
        "pyslam.dense.volumetric_integrator_gaussian_splatting",
        "VolumetricIntegratorGaussianSplatting",
    )

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


@register_class
class VolumetricIntegratorType(SerializableEnum):
    TSDF = 0  # Truncated Signed Distance Function with voxel block grid (parallel spatial hashing)
    # "ASH: A Modern Framework for Parallel Spatial Hashing in 3D Perception"
    GAUSSIAN_SPLATTING = 1  # Incremental Gaussian Splatting by leveraging MonoGS backend: pySLAM keyframes are passed as posed input frames to MonoGS backend.
    # You need CUDA to run Gaussian Splatting.
    # As for MonoGS backend, see the paper: "Gaussian Splatting SLAM".

    @staticmethod
    def from_string(name: str):
        try:
            return VolumetricIntegratorType[name]
        except KeyError:
            raise ValueError(f"Invalid VolumetricIntegratorType: {name}")


def volumetric_integrator_factory(
    volumetric_integrator_type, camera, environment_type, sensor_type
):
    if volumetric_integrator_type == VolumetricIntegratorType.TSDF:
        return VolumetricIntegratorTsdf(
            camera=camera, environment_type=environment_type, sensor_type=sensor_type
        )
    elif volumetric_integrator_type == VolumetricIntegratorType.GAUSSIAN_SPLATTING:
        return VolumetricIntegratorGaussianSplatting(
            camera=camera, environment_type=environment_type, sensor_type=sensor_type
        )
    else:
        raise ValueError(f"Invalid VolumetricIntegratorType: {volumetric_integrator_type}")
