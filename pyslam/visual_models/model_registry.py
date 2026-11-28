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
*
* Declarative catalog of existing pyslam integrations.
*
* Only strings and immutable metadata live here. Importing this catalog does not import
* cv2, torch, TensorFlow, kornia, a vendor repository, or a feature manager. Existing
* constructor branches remain authoritative until their equivalence gates pass.
"""

from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    wrapper: str | None
    group: str
    provides_descriptors: bool
    detector: str | None = None
    descriptor: str | None = None
    source_key: str | None = None
    detector_free: bool = False
    isolate: bool = False
    notes: str = ""

    def feature_info(self):
        """Read existing tables at construction/inspection, never on the frame path."""
        module = importlib.import_module("pyslam.local_features.feature_types")
        info = {}
        if self.descriptor:
            key = module.FeatureDescriptorTypes[self.descriptor]
            info.update(norm_type=module.FeatureInfo.norm_type[key],
                        max_descriptor_distance=module.FeatureInfo.max_descriptor_distance[key])
        if self.detector:
            key = module.FeatureDetectorTypes[self.detector]
            info["need_mask_management"] = module.FeatureInfo.need_mask_management.get(key, False)
        return info

    def source(self):
        """Return recorded provenance, including unresolved license/weight review status."""
        return dict(_sources().get(self.source_key, {}))

    def license(self):
        """Reviewed license summary: a per-model override of the source, else the source review."""
        source = _sources().get(self.source_key, {})
        return source.get("model_licenses", {}).get(self.name) or source.get("license_review", "")


@lru_cache(maxsize=1)
def _sources():
    path = Path(__file__).with_name("legacy_sources.json")
    return json.loads(path.read_text()) if path.exists() else {}


# A symbol is resolved only when requested. Arguments retain the upstream/pyslam
# constructor's semantics; get_local_feature does not rename OpenCV parameters.
_CUSTOM = {
    "ROOT_SIFT": ("feature_root_sift:RootSIFTFeature2D", "opencv", "core"),
    "SHI_TOMASI": ("feature_shitomasi:ShiTomasiDetector", "opencv", "core"),
    "ORB2": ("feature_orbslam2:Orbslam2Feature2D", "orbslam2_features", "core"),
    "SUPERPOINT": ("feature_superpoint:SuperPointFeature2D", "superpoint", "core"),
    "XFEAT": ("feature_xfeat:XFeat2D", "accelerated_features", "core"),
    "TFEAT": ("feature_tfeat:TfeatFeature2D", "tfeat", "core"),
    "HARDNET": ("feature_hardnet:HardnetFeature2D", "hardnet", "core"),
    "GEODESC": ("feature_geodesc:GeodescFeature2D", "geodesc", "tf_models"),
    "SOSNET": ("feature_sosnet:SosnetFeature2D", "SOSNet", "core"),
    "L2NET": ("feature_l2net:L2NetFeature2D", "l2net", "core"),
    "LOGPOLAR": ("feature_logpolar:LogpolarFeature2D", "logpolar", "core"),
    "D2NET": ("feature_d2net:D2NetFeature2D", "d2net", "core"),
    "DELF": ("feature_delf:DelfFeature2D", "tensorflow_models", "tf_models"),
    "CONTEXTDESC": ("feature_contextdesc:ContextDescFeature2D", "contextdesc", "tf_models"),
    "LFNET": ("feature_lfnet:LfNetFeature2D", "lfnet", "tf_models"),
    "R2D2": ("feature_r2d2:R2d2Feature2D", "r2d2", "core"),
    "KEYNET": ("feature_keynet:KeyNetDescFeature2D", "keynet", "tf_models"),
    "DISK": ("feature_disk:DiskFeature2D", "disk", "core"),
    "ALIKED": ("feature_aliked:AlikedFeature2D", "LightGlue", "core"),
    "LIGHTGLUESIFT": ("feature_lightglue_sift:LightGlueSIFTFeature2D", "LightGlue", "core"),
    "KEYNETAFFNETHARDNET": ("feature_keynet_affnet_hardnet:KeyNetAffNetHardNetFeature2D", "kornia", "core"),
    "DEDODE": ("feature_dedode:DeDoDeFeature2D", "dedode", "core"),
    "RDD": ("feature_rdd:RddFeature2D", "rdd", "core"),
}
_CV = {
    "FAST": "FastFeatureDetector_create", "SIFT": "SIFT_create", "SURF": "xfeatures2d.SURF_create",
    "ORB": "ORB_create", "BRISK": "BRISK_create", "KAZE": "KAZE_create", "AKAZE": "AKAZE_create",
    "AGAST": "AgastFeatureDetector_create", "GFTT": "GFTTDetector_create", "MSER": "MSER_create",
    "MSD": "xfeatures2d.MSDDetector_create", "STAR": "xfeatures2d.StarDetector_create",
    "HL": "xfeatures2d.HarrisLaplaceFeatureDetector_create", "FREAK": "xfeatures2d.FREAK_create",
    "BOOST_DESC": "xfeatures2d.BoostDesc_create", "DAISY": "xfeatures2d.DAISY_create",
    "LATCH": "xfeatures2d.LATCH_create", "LUCID": "xfeatures2d.LUCID_create",
    "VGG": "xfeatures2d.VGG_create", "BEBLID": "xfeatures2d.BEBLID_create",
}
_DETECTOR_ONLY = {"SHI_TOMASI", "FAST", "AGAST", "GFTT", "MSER", "MSD", "STAR", "HL"}
_DESCRIPTOR_ONLY = {"TFEAT", "HARDNET", "GEODESC", "SOSNET", "L2NET", "LOGPOLAR", "FREAK",
                    "BOOST_DESC", "DAISY", "LATCH", "LUCID", "VGG", "BEBLID"}


_DETECTOR_FREE_MATCHERS = {"LOFTR", "MAST3R", "ROMA", "TINY_ROMA"}


def _build_specs():
    specs = {}
    for name in sorted(_CUSTOM.keys() | _CV.keys() | {"NONE"}):
        if name in _CUSTOM:
            wrapper, source, group = _CUSTOM[name]
            wrapper = "pyslam.local_features." + wrapper
        elif name in _CV:
            wrapper, source, group = "cv2:" + _CV[name], "opencv", "core"
        else:
            wrapper, source, group = None, None, "core"
        specs[("local_feature", name)] = ModelSpec(
            name, "local_feature", wrapper, group, name not in _DETECTOR_ONLY | {"NONE"},
            detector=name if name not in _DESCRIPTOR_ONLY else None,
            descriptor=name if name not in _DETECTOR_ONLY else None, source_key=source)
    for name, cls, group, source in (
        ("NONE", None, "core", None), ("LK", None, "core", "opencv"),
        ("BF", "BfFeatureMatcher", "core", "opencv"),
        ("FLANN", "FlannFeatureMatcher", "core", "opencv"),
        ("XFEAT", "XFeatMatcher", "core", "accelerated_features"),
        ("LIGHTGLUE", "LightGlueMatcher", "core", "LightGlue"),
        ("LOFTR", "LoFTRMatcher", "dense_matchers", "kornia"),
        ("MAST3R", "Mast3RMatcher", "dense_matchers", "mast3r"),
        ("ROMA", "feature_matcher_detector_free:RoMaMatcher", "dense_matchers", "roma"),
        ("TINY_ROMA", "feature_matcher_detector_free:TinyRoMaMatcher", "dense_matchers", "roma"),
    ):
        module_cls = cls if cls is None or ":" in cls else f"feature_matcher:{cls}"
        detector_free = name in _DETECTOR_FREE_MATCHERS
        specs[("pair_matcher", name)] = ModelSpec(
            name, "pair_matcher", f"pyslam.local_features.{module_cls}" if cls else None,
            group, False, source_key=source, isolate=name == "MAST3R",
            detector_free=detector_free,
            notes="Sentinel: use the tracker API" if cls is None else
                  "Detector-free; unsuitable for descriptor-based map projection search" if detector_free else "")
    for name, wrapper, source in (
        ("MEGALOC", "pyslam.loop_closing.global_feature_megaloc:GlobalFeatureMegaloc", "megaloc"),
        ("COSPLACE", "vpr.feature_extraction.feature_extractor_cosplace:CosPlaceFeatureExtractor", "vpr"),
        ("EIGENPLACES", "vpr.feature_extraction.feature_extractor_eigenplaces:EigenPlacesFeatureExtractor", "vpr"),
        ("NETVLAD", "vpr.feature_extraction.feature_extractor_patchnetvlad:PatchNetVLADFeatureExtractor", "patch_netvlad"),
        ("ALEXNET", "vpr.feature_extraction.feature_extractor_holistic:AlexNetConv3Extractor", "vpr"),
        ("SAD", "vpr.feature_extraction.feature_extractor_holistic:SAD", "vpr"),
        ("HDC-DELF", "vpr.feature_extraction.feature_extractor_holistic:HDCDELF", "vpr"),
        ("SALAD", "pyslam.loop_closing.global_feature_torchhub:GlobalFeatureSalad", "salad"),
        ("BOQ", "pyslam.loop_closing.global_feature_torchhub:GlobalFeatureBoQ", "boq"),
    ):
        specs[("global_descriptor", name)] = ModelSpec(name, "global_descriptor", wrapper, "vpr", False, source_key=source)
    for name in ("NONE", "DBOW2", "DBOW3", "OBINDEX2", "IBOW", "VLAD", "HDC"):
        specs[("global_descriptor", name)] = ModelSpec(
            name, "global_descriptor", None, "core", False,
            notes="Local-descriptor aggregation; construct with LoopDetectorConfigs and its vocabulary")
    return specs


_SPECS = _build_specs()


def get_spec(name, kind="local_feature"):
    if isinstance(name, ModelSpec):
        if name.kind != kind:
            raise ValueError(f"Expected {kind}, got {name.kind}")
        return name
    try:
        normalized = str(name).upper()
        if kind == "global_descriptor" and normalized == "HDC_DELF":
            normalized = "HDC-DELF"
        return _SPECS[(kind, normalized)]
    except KeyError:
        raise ValueError(f"Unknown {kind} {name!r}; inspect list_models()") from None


def list_models(kind=None, group=None):
    return tuple(spec for spec in _SPECS.values()
                 if (kind is None or spec.kind == kind) and (group is None or spec.group == group))


def spec_for_enum(value):
    kinds = {"FeatureDetectorTypes": "local_feature", "FeatureDescriptorTypes": "local_feature",
             "FeatureMatcherTypes": "pair_matcher", "FeatureTrackerTypes": "pair_matcher",
             "GlobalDescriptorType": "global_descriptor", "LocalDescriptorAggregationType": "global_descriptor"}
    try:
        kind = kinds[type(value).__name__]
    except KeyError:
        raise ValueError(f"Unsupported visual-model enum: {type(value).__name__}") from None
    name = value.name
    if type(value).__name__ == "FeatureTrackerTypes":
        name = {"DES_BF": "BF", "DES_FLANN": "FLANN"}.get(name, name)
    return get_spec(name, kind)


def validate_frontend(name, kind="local_feature", requires_descriptors=True):
    spec = get_spec(name, kind)
    if requires_descriptors and not spec.provides_descriptors:
        raise ValueError(f"{spec.name} ({spec.kind}) does not provide per-keypoint descriptors; "
                         "use a detector-free tracker or choose a descriptor-producing local feature")
    return spec
