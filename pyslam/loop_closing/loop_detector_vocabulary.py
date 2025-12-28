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

import os
import time
import math
import numpy as np
import platform

from pyslam.utilities.logging import Printer

from pyslam.config_parameters import Parameters

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

from pyslam.utilities.serialization import Serializable, register_class

kVerbose = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


# NOTE: At present, under mac, boost serialization is very slow, we use txt files instead.
def dbow2_orb_vocabulary_factory(*args, **kwargs):
    use_text_vocabulary = platform.system() == "Darwin"
    if use_text_vocabulary:
        return DBowOrbVocabularyDataTxt(*args, **kwargs)
    else:
        return DBow2OrbVocabularyData(*args, **kwargs)


# NOTE: at present, under mac, boost serialization is very slow, we use txt files instead.
def dbow3_orb_vocabulary_factory(*args, **kwargs):
    use_text_vocabulary = platform.system() == "Darwin"
    if use_text_vocabulary:
        return DBowOrbVocabularyDataTxt(*args, **kwargs)
    else:
        return DBow3OrbVocabularyData(*args, **kwargs)


@register_class
class VocabularyData(Serializable):
    def __init__(
        self,
        vocab_file_path=None,
        descriptor_type=None,
        descriptor_dimension=None,
        url_vocabulary=None,
        url_type=None,
    ):
        self.vocab_file_path = vocab_file_path
        self.descriptor_type = descriptor_type
        self.descriptor_dimension = descriptor_dimension
        self.url_vocabulary = url_vocabulary
        self.url_type = url_type

    def check_download(self):
        if self.url_vocabulary is not None and not os.path.exists(self.vocab_file_path):
            if self.url_type == "gdrive":
                gdrive_url = self.url_vocabulary
                Printer.blue(
                    f"VocabularyData: downloading vocabulary {self.descriptor_type.name} from: {gdrive_url}"
                )
                try:
                    gdrive_download_lambda(url=gdrive_url, path=self.vocab_file_path)
                except Exception as e:
                    Printer.red(f"VocabularyData: cannot download vocabulary from {gdrive_url}")
                    raise e
        if self.vocab_file_path is not None and not os.path.exists(self.vocab_file_path):
            Printer.red(f"VocabularyData: cannot find vocabulary file: {self.vocab_file_path}")
            raise FileNotFoundError


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBowOrbVocabularyDataTxt(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.txt"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBow2OrbVocabularyData(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.dbow2"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=1pvBERLLSUV4IcaInNJMURTb8p-r9-5Xf",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


# NOTE: Under mac, loading the DBOW2 vocabulary is very slow (both from text and from boost archive).
@register_class
class DBow3OrbVocabularyData(VocabularyData):
    kOrbVocabFile = kDataFolder + "/ORBvoc.dbow3"

    def __init__(
        self,
        vocab_file_path=kOrbVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/uc?id=13xmRtop_ow3aPtv3qCT5beG19_mlogqI",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )


@register_class
class VladOrbVocabularyData(VocabularyData):
    kVladVocabFile = kDataFolder + "/VLADvoc_orb.txt"

    def __init__(
        self,
        vocab_file_path=kVladVocabFile,
        descriptor_type=FeatureDescriptorTypes.ORB2,
        descriptor_dimension=32,
        url_vocabulary="https://drive.google.com/file/d/1u6AJEa2aZg7u5aS6vFX2qXiKKePQ_6t2",
        url_type="gdrive",
    ):  # download it from gdrive
        super().__init__(
            vocab_file_path, descriptor_type, descriptor_dimension, url_vocabulary, url_type
        )
