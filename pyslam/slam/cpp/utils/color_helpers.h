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

#pragma once

#include <array>
#include <cstddef>

namespace pyslam {

class Colors {
  public:
    // colors from
    // https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
    static constexpr std::array<std::array<float, 3>, 10> myjet = {{{{0.0, 0.0, 0.5}},
                                                                    {{0.0, 0.0, 0.99910873}},
                                                                    {{0.0, 0.37843137, 1.0}},
                                                                    {{0.0, 0.83333333, 1.0}},
                                                                    {{0.30044276, 1.0, 0.66729918}},
                                                                    {{0.66729918, 1.0, 0.30044276}},
                                                                    {{1.0, 0.90123457, 0.0}},
                                                                    {{1.0, 0.48002905, 0.0}},
                                                                    {{0.99910873, 0.07334786, 0.0}},
                                                                    {{0.5, 0.0, 0.0}}}};

    static std::array<float, 3> myjet_color(int idx) { return myjet[idx % myjet.size()]; }
};

} // namespace pyslam