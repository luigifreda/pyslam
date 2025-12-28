"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
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

import numpy as np
import os
import sys
import traceback
import pickle

from pyslam.utilities.logging import Printer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


from pyslam.config_parameters import Parameters

from .semantic_types import SemanticDatasetType, SemanticFeatureType
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_color_utils import (
    labels_to_image,
    labels_color_map_factory,
    need_large_color_map,
)
from .semantic_color_map_factory import semantic_color_map_factory


from typing import TYPE_CHECKING, Callable, Union

if TYPE_CHECKING:
    from pyslam.semantics.semantic_mapping_base import SemanticMappingType, SemanticMappingBase


# Base class for semantic estimators via inference
class SemanticMappingShared:

    semantic_mapping: Union["SemanticMappingType", None] = None
    semantic_feature_type: Union["SemanticFeatureType", None] = None
    semantic_dataset_type: Union["SemanticDatasetType", None] = None
    semantic_fusion_method: Union[Callable[[np.ndarray], np.ndarray], None] = None
    sem_des_to_rgb: Union[Callable[[np.ndarray, bool], np.ndarray], None] = None
    sem_img_to_rgb: Union[Callable[[np.ndarray, bool], np.ndarray], None] = None
    get_semantic_weight: Union[Callable[[np.ndarray], float], None] = None

    @staticmethod
    def is_semantic_mapping_enabled():
        return Parameters.kDoSparseSemanticMappingAndSegmentation

    @staticmethod
    def is_semantic_mapping_set():
        return SemanticMappingShared.semantic_mapping is not None

    @staticmethod
    def set_semantic_mapping(semantic_mapping, force=False):

        if not force and SemanticMappingShared.semantic_mapping is not None:
            raise Exception("SemanticMappingShared: Semantic Estimator is already set!")

        SemanticMappingShared.semantic_mapping = semantic_mapping
        SemanticMappingShared.semantic_feature_type = semantic_mapping.semantic_feature_type
        SemanticMappingShared.semantic_dataset_type = semantic_mapping.semantic_dataset_type
        SemanticMappingShared.semantic_fusion_method = semantic_mapping.semantic_fusion_method
        SemanticMappingShared.sem_des_to_rgb = semantic_mapping.sem_des_to_rgb
        SemanticMappingShared.sem_img_to_rgb = semantic_mapping.sem_img_to_rgb
        SemanticMappingShared.get_semantic_weight = semantic_mapping.get_semantic_weight
        # Initialize the C++ module with the semantic mapping info
        SemanticMappingShared.init_cpp_module(semantic_mapping)

    @staticmethod
    def set_semantic_feature_type(semantic_feature_type: "SemanticFeatureType"):
        SemanticMappingShared.semantic_feature_type = semantic_feature_type
        SemanticMappingShared.set_cpp_semantic_feature_type(semantic_feature_type)

    @staticmethod
    def _init_cpp_module_from_dict(
        semantic_feature_type,
        semantic_dataset_type,
        semantic_fusion_method,
        semantic_entity_type,
        semantic_segmentation_type,
        num_classes,
    ):
        """
        Helper method to initialize C++ module from individual parameters.
        Used by both init_cpp_module() and import_state().
        """
        try:
            from pyslam.slam.cpp import cpp_module
            from pyslam.slam.cpp import CPP_AVAILABLE

            if not CPP_AVAILABLE:
                return

            cpp_module.FeatureSharedResources.semantic_feature_type = semantic_feature_type

            # Set the semantic mapping info in the C++ module
            cpp_module.SemanticMappingSharedResources.semantic_feature_type = semantic_feature_type
            cpp_module.SemanticMappingSharedResources.semantic_dataset_type = semantic_dataset_type
            if semantic_fusion_method is not None:
                cpp_module.SemanticMappingSharedResources.semantic_fusion_method = (
                    semantic_fusion_method
                )
            if semantic_entity_type is not None:
                cpp_module.SemanticMappingSharedResources.semantic_entity_type = (
                    semantic_entity_type
                )
            if semantic_segmentation_type is not None:
                cpp_module.SemanticMappingSharedResources.semantic_segmentation_type = (
                    semantic_segmentation_type
                )

            # Initialize color map
            if num_classes is not None and semantic_dataset_type is not None:
                cpp_dataset_type = semantic_dataset_type
                # Check if we need CUSTOM_SET for large color maps
                if semantic_segmentation_type is not None:
                    needs_large, _ = need_large_color_map(semantic_segmentation_type)
                    if needs_large and num_classes > 1000:
                        cpp_dataset_type = SemanticDatasetType.CUSTOM_SET

                cpp_module.SemanticMappingSharedResources.init_color_map(
                    cpp_dataset_type,
                    num_classes,
                )
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot set cpp_module: {e}")
            traceback.print_exc()

    @staticmethod
    def init_cpp_module(semantic_mapping: "SemanticMappingBase"):
        """
        Initialize C++ module from a semantic_mapping object.
        Extracts all necessary information and calls the helper method.
        """
        try:
            # When using a separate process, use the actual color map size from the semantic segmentation process
            # This ensures the C++ color map matches the Python color map (important for open-vocabulary models)
            num_classes = semantic_mapping.semantic_segmentation.num_classes()
            cpp_dataset_type = semantic_mapping.semantic_dataset_type

            # Check if we need a large color map (for open-vocabulary models like DETIC/EOV-Seg)
            if hasattr(semantic_mapping, "semantic_segmentation_process"):
                # Using separate process - get the actual color map size
                if hasattr(semantic_mapping.semantic_segmentation_process, "semantic_color_map"):
                    color_map = (
                        semantic_mapping.semantic_segmentation_process.semantic_color_map.color_map
                    )
                    if color_map is not None:
                        num_classes = len(color_map)

            # Also check if semantic_mapping has a semantic_color_map attribute (for SemanticMappingDenseProcess)
            if (
                hasattr(semantic_mapping, "semantic_color_map")
                and semantic_mapping.semantic_color_map is not None
            ):
                # Use the actual color map size
                actual_num_classes = len(semantic_mapping.semantic_color_map)
                if actual_num_classes > num_classes:
                    num_classes = actual_num_classes

            # Call the helper method with extracted values
            SemanticMappingShared._init_cpp_module_from_dict(
                semantic_feature_type=semantic_mapping.semantic_feature_type,
                semantic_dataset_type=cpp_dataset_type,
                semantic_fusion_method=semantic_mapping.semantic_fusion_method,
                semantic_entity_type=semantic_mapping.semantic_entity_type,
                semantic_segmentation_type=semantic_mapping.semantic_segmentation_type,
                num_classes=num_classes,
            )
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot set cpp_module: {e}")
            traceback.print_exc()

    @staticmethod
    def cleanup_cpp_module():
        """
        Clean up C++ semantic mapping resources.
        This should be called when semantic mapping is disabled or when shutting down.
        """
        try:
            from pyslam.slam.cpp import cpp_module
            from pyslam.slam.cpp import CPP_AVAILABLE
            from pyslam.slam import USE_CPP

            if not CPP_AVAILABLE:
                return

            # Reset the semantic color map to release resources
            # Note: semantic_color_map is read-only in Python bindings, so we can't set it directly
            # Instead, we reinitialize it with a minimal color map to release the large one
            # This helps free memory and prevents hanging references
            try:
                # Reinitialize with a minimal color map (1 class) to release the large one
                cpp_module.SemanticMappingSharedResources.reset_color_map()
            except Exception as e:
                Printer.orange(
                    f"WARNING: SemanticMappingShared: cannot reset semantic_color_map: {e}"
                )

            # Reset other static members to default values
            # Note: The C++ bindings expect enum objects, not integers
            # Since Python enums don't have NONE values, we skip resetting these
            # The C++ code will initialize them to NONE (-1) by default on next init
            # The important cleanup is releasing the large color map above
            # If we really need to reset them, we could create enum instances, but it's not critical
            # for cleanup purposes since they'll be reinitialized on next use
            pass
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot cleanup cpp_module: {e}")
            traceback.print_exc()

    @staticmethod
    def set_cpp_semantic_feature_type(semantic_feature_type):
        try:
            from pyslam.slam.cpp import cpp_module
            from pyslam.slam.cpp import CPP_AVAILABLE
            from pyslam.slam import USE_CPP

            if not CPP_AVAILABLE:
                return

            cpp_module.SemanticMappingSharedResources.semantic_feature_type = semantic_feature_type
        except Exception as e:
            Printer.orange(
                f"WARNING: SemanticMappingShared: cannot set cpp_semantic_feature_type: {e}"
            )
            traceback.print_exc()

    @staticmethod
    def export_state():
        """
        Export the SemanticMappingShared state as a dictionary for passing to spawned processes.
        This extracts all the essential information needed to recreate the state.

        Returns:
            dict: Dictionary containing the state information, or None if semantic mapping is not enabled
        """
        if SemanticMappingShared.semantic_mapping is None:
            return None

        # We already checked semantic_mapping is not None at line 230
        state = {
            "semantic_feature_type": SemanticMappingShared.semantic_feature_type,
            "semantic_dataset_type": SemanticMappingShared.semantic_dataset_type,
            # Note: Callables (semantic_fusion_method, sem_des_to_rgb, sem_img_to_rgb, get_semantic_weight)
            # are extracted from semantic_mapping object. We'll try to include them, but if they're not
            # picklable, they will be None and we'll need to handle that in import_state.
            # Also include info needed for C++ module initialization
            "semantic_entity_type": SemanticMappingShared.semantic_mapping.semantic_entity_type,
            "semantic_segmentation_type": SemanticMappingShared.semantic_mapping.semantic_segmentation_type,
            "num_classes": (
                SemanticMappingShared.semantic_mapping.semantic_segmentation.num_classes()
                if hasattr(SemanticMappingShared.semantic_mapping, "semantic_segmentation")
                else None
            ),
        }

        # Try to include callables if they're picklable (they might not be if they're bound methods)
        # If they're not picklable, they'll remain None and the spawned process will need to handle that
        try:
            # Test if callables can be pickled
            if SemanticMappingShared.semantic_fusion_method is not None:
                pickle.dumps(SemanticMappingShared.semantic_fusion_method)
                state["semantic_fusion_method"] = SemanticMappingShared.semantic_fusion_method
        except Exception as e:
            Printer.orange(
                f"WARNING: SemanticMappingShared: cannot pickle semantic_fusion_method: {e}"
            )
            traceback.print_exc()
        try:
            if SemanticMappingShared.sem_des_to_rgb is not None:
                pickle.dumps(SemanticMappingShared.sem_des_to_rgb)
                state["sem_des_to_rgb"] = SemanticMappingShared.sem_des_to_rgb
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot pickle sem_des_to_rgb: {e}")
            traceback.print_exc()
        try:
            if SemanticMappingShared.sem_img_to_rgb is not None:
                pickle.dumps(SemanticMappingShared.sem_img_to_rgb)
                state["sem_img_to_rgb"] = SemanticMappingShared.sem_img_to_rgb
        except Exception as e:
            Printer.orange(f"WARNING: SemanticMappingShared: cannot pickle sem_img_to_rgb: {e}")
            traceback.print_exc()
        try:
            if SemanticMappingShared.get_semantic_weight is not None:
                pickle.dumps(SemanticMappingShared.get_semantic_weight)
                state["get_semantic_weight"] = SemanticMappingShared.get_semantic_weight
        except Exception as e:
            Printer.orange(
                f"WARNING: SemanticMappingShared: cannot pickle get_semantic_weight: {e}"
            )
            traceback.print_exc()

        # Export semantic color map parameters for recreation in spawned processes
        # We extract the color map object and its parameters to recreate it properly
        color_map_obj = None
        if hasattr(SemanticMappingShared.semantic_mapping, "semantic_segmentation_process"):
            # Using separate process - get color map object from the process
            semantic_seg_process = (
                SemanticMappingShared.semantic_mapping.semantic_segmentation_process
            )
            if semantic_seg_process is not None:
                # semantic_segmentation_process uses semantic_color_map (the object)
                if hasattr(semantic_seg_process, "semantic_color_map"):
                    color_map_obj = semantic_seg_process.semantic_color_map
        elif hasattr(SemanticMappingShared.semantic_mapping, "semantic_segmentation"):
            # NOT using separate process - get color map object from semantic_segmentation
            semantic_seg = SemanticMappingShared.semantic_mapping.semantic_segmentation
            if semantic_seg is not None:
                # Some classes use semantic_color_map_obj (Detic, EOV-Seg, ODISE)
                # Others use semantic_color_map (SegFormer, DeepLab, etc.)
                if hasattr(semantic_seg, "semantic_color_map_obj"):
                    color_map_obj = semantic_seg.semantic_color_map_obj
                elif hasattr(semantic_seg, "semantic_color_map") and not isinstance(
                    getattr(semantic_seg, "semantic_color_map", None), np.ndarray
                ):
                    # Only use if it's an object, not a numpy array
                    potential_obj = semantic_seg.semantic_color_map
                    if hasattr(potential_obj, "color_map"):
                        color_map_obj = potential_obj

        if color_map_obj is not None:
            # Export the color map array for exact color matching
            if hasattr(color_map_obj, "color_map") and color_map_obj.color_map is not None:
                color_map = color_map_obj.color_map
                # Ensure it's a numpy array
                if not isinstance(color_map, np.ndarray):
                    color_map = np.array(color_map)
                state["color_map"] = color_map.copy()
                state["num_classes"] = len(color_map)
                Printer.green(
                    f"SemanticMappingShared: Exported color map "
                    f"(size: {len(state['color_map'])} classes, shape: {state['color_map'].shape}, "
                    f"dtype: {state['color_map'].dtype})"
                )
                if len(state["color_map"]) > 0:
                    color_hash = hash(state["color_map"].tobytes())
                    Printer.green(
                        f"SemanticMappingShared: First 3 exported colors: {state['color_map'][:3].tolist()}, "
                        f"exported color_map hash: {color_hash}"
                    )

            # Export parameters needed to recreate the color map object
            # Note: metadata is not exported as it's typically not picklable and will fallback to StandardColorMap
            state["color_map_params"] = {
                "semantic_dataset_type": color_map_obj.semantic_dataset_type,
                "semantic_feature_type": color_map_obj.semantic_feature_type,
                "num_classes": getattr(color_map_obj, "num_classes", None),
                "semantic_segmentation_type": getattr(
                    color_map_obj, "semantic_segmentation_type", None
                ),
                "device": getattr(color_map_obj, "device", "cpu"),
                "sim_scale": getattr(color_map_obj, "sim_scale", 1.0),
            }
            # Try to export text_embs if available (might not be picklable)
            if hasattr(color_map_obj, "text_embs") and color_map_obj.text_embs is not None:
                try:
                    pickle.dumps(color_map_obj.text_embs)
                    state["color_map_params"]["text_embs"] = color_map_obj.text_embs
                except Exception:
                    Printer.orange(
                        "WARNING: SemanticMappingShared: Cannot pickle text_embs, will recreate without it"
                    )

        return state

    @staticmethod
    def import_state(state_dict, force=False):
        """
        Import/restore the SemanticMappingShared state from a dictionary in a spawned process.
        This recreates the static fields from the exported state.

        Args:
            state_dict: Dictionary containing the state information (from export_state())
            force: If True, overwrite existing state
        """
        if state_dict is None:
            return

        if not force and SemanticMappingShared.semantic_mapping is not None:
            Printer.orange(
                "WARNING: SemanticMappingShared: State already set, use force=True to overwrite"
            )
            return

        # Restore the static fields
        SemanticMappingShared.semantic_feature_type = state_dict.get("semantic_feature_type")
        SemanticMappingShared.semantic_dataset_type = state_dict.get("semantic_dataset_type")

        # Restore callables if they were successfully pickled (they might be None if not picklable)
        SemanticMappingShared.semantic_fusion_method = state_dict.get("semantic_fusion_method")
        SemanticMappingShared.sem_des_to_rgb = state_dict.get("sem_des_to_rgb")
        SemanticMappingShared.get_semantic_weight = state_dict.get("get_semantic_weight")
        SemanticMappingShared.sem_img_to_rgb = state_dict.get("sem_img_to_rgb")

        # Recreate semantic color map object and its methods in spawned processes
        # Use factory with exported parameters, and override with exported color map if available for exact matching
        if SemanticMappingShared.sem_img_to_rgb is None or "color_map_params" in state_dict:
            try:
                # Try to recreate the color map object using the factory
                if "color_map_params" in state_dict and state_dict["color_map_params"] is not None:
                    params = state_dict["color_map_params"]
                    # Recreate using factory parameters
                    recreated_color_map_obj = semantic_color_map_factory(
                        semantic_dataset_type=params["semantic_dataset_type"],
                        semantic_feature_type=params["semantic_feature_type"],
                        num_classes=params.get("num_classes"),
                        text_embs=params.get("text_embs"),
                        device=params.get("device", "cpu"),
                        sim_scale=params.get("sim_scale", 1.0),
                        semantic_segmentation_type=params.get("semantic_segmentation_type"),
                        metadata=None,  # Not exported as it's typically not picklable
                    )

                    # If we have an exported color map, override to ensure exact match
                    # This is important when metadata isn't available (detectron2 maps fallback to standard)
                    if "color_map" in state_dict and state_dict["color_map"] is not None:
                        color_map = state_dict["color_map"]
                        if not isinstance(color_map, np.ndarray):
                            color_map = np.array(color_map)
                        color_map = np.ascontiguousarray(color_map)
                        recreated_color_map_obj.color_map = color_map
                        Printer.green(
                            f"SemanticMappingShared: Overrode color map with exported array "
                            f"(size: {len(color_map)} classes) for exact matching"
                        )
                        if len(color_map) > 0:
                            color_hash = hash(color_map.tobytes())
                            Printer.green(
                                f"SemanticMappingShared: First 3 colors: {color_map[:3].tolist()}, "
                                f"color_map hash: {color_hash}"
                            )
                    else:
                        Printer.green(
                            f"SemanticMappingShared: Recreated color map object using factory parameters"
                        )
                else:
                    # Fallback: recreate using basic parameters
                    Printer.yellow(
                        "SemanticMappingShared: Recreating color map object using basic parameters..."
                    )
                    semantic_feature_type = state_dict.get("semantic_feature_type")
                    semantic_dataset_type = state_dict.get("semantic_dataset_type")
                    semantic_segmentation_type = state_dict.get("semantic_segmentation_type")
                    num_classes = state_dict.get("num_classes")

                    recreated_color_map_obj = semantic_color_map_factory(
                        semantic_dataset_type=semantic_dataset_type,
                        semantic_feature_type=semantic_feature_type,
                        num_classes=num_classes,
                        semantic_segmentation_type=semantic_segmentation_type,
                    )

                # Use the recreated object's methods directly
                SemanticMappingShared.sem_img_to_rgb = recreated_color_map_obj.sem_img_to_rgb
                SemanticMappingShared.sem_des_to_rgb = recreated_color_map_obj.sem_des_to_rgb

                Printer.green(
                    f"SemanticMappingShared: Recreated semantic color map object in spawned process "
                    f"(type: {type(recreated_color_map_obj).__name__}, "
                    f"color_map size: {len(recreated_color_map_obj.color_map) if recreated_color_map_obj.color_map is not None else 0} classes)"
                )

                # Also try to log via VolumetricIntegratorBase if available
                try:
                    from pyslam.dense.volumetric_integrator_base import VolumetricIntegratorBase

                    VolumetricIntegratorBase.print(
                        f"SemanticMappingShared: Recreated semantic color map object "
                        f"(type: {type(recreated_color_map_obj).__name__}, "
                        f"size: {len(recreated_color_map_obj.color_map) if recreated_color_map_obj.color_map is not None else 0} classes)"
                    )
                except:
                    pass  # VolumetricIntegratorBase might not be available in all contexts

            except Exception as e:
                Printer.orange(
                    f"WARNING: SemanticMappingShared: Failed to recreate semantic color map object in spawned process: {e}"
                )
                traceback.print_exc()
                # Fallback to manual recreation if factory fails
                if "color_map" in state_dict and state_dict["color_map"] is not None:
                    try:
                        color_map = state_dict["color_map"]
                        if not isinstance(color_map, np.ndarray):
                            color_map = np.array(color_map)
                        _cached_color_map = np.ascontiguousarray(color_map)

                        def _fallback_sem_img_to_rgb(semantic_img, bgr=False):
                            return labels_to_image(semantic_img, _cached_color_map, bgr=bgr)

                        SemanticMappingShared.sem_img_to_rgb = _fallback_sem_img_to_rgb
                        Printer.yellow(
                            f"SemanticMappingShared: Using fallback sem_img_to_rgb "
                            f"(using exported color_map with {len(_cached_color_map)} classes)"
                        )
                    except Exception as e2:
                        Printer.orange(
                            f"WARNING: SemanticMappingShared: Fallback recreation also failed: {e2}"
                        )

        # Note: sem_des_to_rgb and get_semantic_weight are less critical for volumetric integrator
        # If they're None, the code should handle it gracefully
        if SemanticMappingShared.sem_des_to_rgb is None:
            Printer.orange(
                "WARNING: SemanticMappingShared - import_state(): sem_des_to_rgb is None!"
            )
        if SemanticMappingShared.get_semantic_weight is None:
            Printer.orange(
                "WARNING: SemanticMappingShared - import_state(): get_semantic_weight is None!"
            )

        # Initialize C++ module if needed
        if state_dict.get("semantic_feature_type") is not None:
            SemanticMappingShared._init_cpp_module_from_dict(
                semantic_feature_type=state_dict.get("semantic_feature_type"),
                semantic_dataset_type=state_dict.get("semantic_dataset_type"),
                semantic_fusion_method=state_dict.get("semantic_fusion_method"),
                semantic_entity_type=state_dict.get("semantic_entity_type"),
                semantic_segmentation_type=state_dict.get("semantic_segmentation_type"),
                num_classes=state_dict.get("num_classes"),
            )
