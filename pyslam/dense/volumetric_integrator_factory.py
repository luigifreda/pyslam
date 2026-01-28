import os
import traceback

from pyslam.config_parameters import Parameters
from pyslam.utilities.system import import_from
from pyslam.dense.volumetric_integrator_types import VolumetricIntegratorType

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
try:
    from .volumetric_integrator_voxel_grid import VolumetricIntegratorVoxelGrid
except ImportError:
    VolumetricIntegratorVoxelGrid = import_from(
        "pyslam.dense.volumetric_integrator_voxel_grid",
        "VolumetricIntegratorVoxelGrid",
    )
try:
    from .volumetric_integrator_voxel_semantic_grid import VolumetricIntegratorVoxelSemanticGrid
except ImportError:
    VolumetricIntegratorVoxelSemanticGrid = import_from(
        "pyslam.dense.volumetric_integrator_voxel_semantic_grid",
        "VolumetricIntegratorVoxelSemanticGrid",
    )
kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .volumetric_integrator_tsdf import VolumetricIntegratorTsdf
    from .volumetric_integrator_gaussian_splatting import VolumetricIntegratorGaussianSplatting
    from .volumetric_integrator_voxel_grid import VolumetricIntegratorVoxelGrid
    from .volumetric_integrator_voxel_semantic_grid import VolumetricIntegratorVoxelSemanticGrid


def volumetric_integrator_factory(
    volumetric_integrator_type,
    camera,
    environment_type,
    sensor_type,
    viewer_queue=None,
):
    """
    Factory function to create a VolumetricIntegratorBase instance based on the volumetric_integrator_type.
    Args:
        volumetric_integrator_type: VolumetricIntegratorType enum value
        camera: Camera instance
        environment_type: DatasetEnvironmentType enum value
        sensor_type: SensorType enum value
        viewer_queue: Queue instance for the viewer
    Returns:
        VolumetricIntegratorBase instance

    Associations:
        VolumetricIntegratorType.VOXEL_GRID                        -> VolumetricIntegratorVoxelGrid
        VolumetricIntegratorType.VOXEL_SEMANTIC_GRID               -> VolumetricIntegratorVoxelSemanticGrid
        VolumetricIntegratorType.VOXEL_SEMANTIC_PROBABILISTIC_GRID -> VolumetricIntegratorVoxelSemanticGrid
        VolumetricIntegratorType.TSDF                              -> VolumetricIntegratorTsdf
        VolumetricIntegratorType.GAUSSIAN_SPLATTING                -> VolumetricIntegratorGaussianSplatting
    """
    if volumetric_integrator_type == VolumetricIntegratorType.VOXEL_GRID:
        return VolumetricIntegratorVoxelGrid(
            camera=camera,
            environment_type=environment_type,
            sensor_type=sensor_type,
            volumetric_integrator_type=volumetric_integrator_type,
            use_voxel_blocks=Parameters.kVolumetricIntegrationUseVoxelBlocks,  # use voxel blocks by default since it is more efficient
            viewer_queue=viewer_queue,
        )
    elif volumetric_integrator_type == VolumetricIntegratorType.VOXEL_SEMANTIC_GRID:
        return VolumetricIntegratorVoxelSemanticGrid(
            camera=camera,
            environment_type=environment_type,
            sensor_type=sensor_type,
            volumetric_integrator_type=volumetric_integrator_type,
            use_voxel_blocks=Parameters.kVolumetricIntegrationUseVoxelBlocks,  # use voxel blocks by default since it is more efficient
            viewer_queue=viewer_queue,
        )
    elif volumetric_integrator_type == VolumetricIntegratorType.VOXEL_SEMANTIC_PROBABILISTIC_GRID:
        return VolumetricIntegratorVoxelSemanticGrid(
            camera=camera,
            environment_type=environment_type,
            sensor_type=sensor_type,
            volumetric_integrator_type=volumetric_integrator_type,
            use_semantic_probabilistic=True,
            use_voxel_blocks=Parameters.kVolumetricIntegrationUseVoxelBlocks,  # use voxel blocks by default since it is more efficient
            viewer_queue=viewer_queue,
        )
    elif volumetric_integrator_type == VolumetricIntegratorType.TSDF:
        return VolumetricIntegratorTsdf(
            camera=camera,
            environment_type=environment_type,
            sensor_type=sensor_type,
            volumetric_integrator_type=volumetric_integrator_type,
            viewer_queue=viewer_queue,
        )
    elif volumetric_integrator_type == VolumetricIntegratorType.GAUSSIAN_SPLATTING:
        return VolumetricIntegratorGaussianSplatting(
            camera=camera,
            environment_type=environment_type,
            sensor_type=sensor_type,
            volumetric_integrator_type=volumetric_integrator_type,
            viewer_queue=viewer_queue,
        )
    else:
        raise ValueError(f"Invalid VolumetricIntegratorType: {volumetric_integrator_type}")
