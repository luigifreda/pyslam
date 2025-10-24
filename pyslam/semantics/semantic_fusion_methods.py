"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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

from pyslam.utilities.utils_serialization import SerializableEnum, register_class

# TODO(dvdmc): Fusion methods only make sense for certain semantic types


# TODO(dvdmc): Below is not yet used!
@register_class
class SemanticFusionMehtods(SerializableEnum):
    COUNT_LABELS = 0  # The label with the highest count
    BAYESIAN_FUSION = 1  # Integrate measurements iteratively in a bayesian manner TODO(dvdmc): Check if not normalizing in between results in the same.
    AVERAGE_FUSION = 2  # Average all measurements


def count_labels(labels):
    """
    Count the labels and return the label with the highest count.
    """
    unique_labels = np.unique(labels)
    label_count = np.zeros(len(unique_labels))
    for i, unique_label in enumerate(unique_labels):
        label_count[i] = np.sum(labels == unique_label)
    return unique_labels[label_count.argmax()]


def bayesian_fusion(probs):
    """
    Bayesian fusion of probability vectors.
    https://en.wikipedia.org/wiki/Bayesian_inference#Bayesian_inference_for_parameter_estimation
    Uses the following formula:
    P(θ|D) = P(D|θ) * P(θ) / P(D)
    where:
    - P(θ|D) is the posterior probability of the parameter θ given the data D
    - P(D|θ) is the likelihood of the data D given the parameter θ
    - P(θ) is the prior probability of the parameter θ
    - P(D) is the marginal likelihood of the data D
    """
    num_classes = probs[0].shape[-1]
    prior = np.ones(num_classes) / num_classes
    posterior = prior
    for obs in probs:
        posterior *= obs
        # normalize
        posterior /= np.sum(posterior)
    return posterior


def average_fusion(features):
    """
    Average fusion of features.
    """
    return np.mean(features, axis=0)
