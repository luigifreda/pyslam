import os

from pyslam.utilities.serialization import SerializableEnum, register_class


# fmt: off
@register_class
class VolumetricIntegratorType(SerializableEnum):
    VOXEL_GRID = 0  # VoxelGrid (by default, uses voxel blocks if kVolumetricIntegrationUseVoxelBlocks is True)
    VOXEL_SEMANTIC_GRID = 1  # VoxelSemanticGrid with confidence-counter semantic fusion
                             # It stores the most observed semantic class for each voxel and its confidence counter
                             # (by default, uses voxel blocks if kVolumetricIntegrationUseVoxelBlocks is True)
    VOXEL_SEMANTIC_GRID_PROBABILISTIC = 2 # VoxelSemanticGridProbabilistic with probabilistic semantic fusion (slower)
                                          # It stores and updates the probabilities of each observed semantic class for each voxel
                                          # (by default, uses voxel blocks if kVolumetricIntegrationUseVoxelBlocks is True)
    TSDF = 3  # Truncated Signed Distance Function with voxel block grid (parallel spatial hashing)
              # Uses "ASH: A Modern Framework for Parallel Spatial Hashing in 3D Perception"
    GAUSSIAN_SPLATTING = 4  # Incremental Gaussian Splatting by leveraging MonoGS backend: pySLAM keyframes are passed as posed input frames to MonoGS backend.
                            # You need CUDA to run Gaussian Splatting.
                            # As for MonoGS backend, see the paper: "Gaussian Splatting SLAM".
# fmt: on

    @staticmethod
    def from_string(name: str):
        try:
            return VolumetricIntegratorType[name]
        except KeyError:
            raise ValueError(f"Invalid VolumetricIntegratorType: {name}")
