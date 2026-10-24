/*
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
 */
#include "config_parameters.h"

namespace pyslam {

float Parameters::kMaxDescriptorDistance =
    0.0f; // It is initialized by the first created instance of feature_manager.py at runtime

float Parameters::kFeatureMatchDefaultRatioTest =
    0.7; // This is the default ratio test used by all feature matchers. It can be configured
         // per descriptor in feature_tracker_configs.py

} // namespace pyslam