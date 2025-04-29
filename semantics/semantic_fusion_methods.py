"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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

def count_labels(labels):
    unique_labels = np.unique(labels)
    label_count = np.zeros(len(unique_labels))
    for i in range(len(unique_labels)):
        label_count[i] = np.sum(labels == unique_labels[i])
    return unique_labels[label_count.argmax()]

def bayesian_fusion(probs):
    prior = np.ones(probs.shape[0]) / probs.shape[0]
    posterior = prior
    for obs in probs:
        posterior *= obs
    return posterior

    