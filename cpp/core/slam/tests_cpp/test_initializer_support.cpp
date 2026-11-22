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

// Bounded stereo initialization fallback must match the Python implementation.
#include "camera/camera.h"
#include "config_parameters.h"
#include "initializer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace pyslam;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        auto frame = std::make_shared<Frame>(std::make_shared<PinholeCamera>());
        Initializer initializer(SensorType::STEREO);
        initializer.init(frame, cv::Mat());
        const double base = Parameters::kInitializerNumMinTriangulatedPointsStereo;
        const double floor = std::min(base, 2.0 * Parameters::kInitializerNumMinNumPointsForPnPWithDepth);
        for (int failures : {0, 10, 11, 20, 30, 1000}) {
            initializer.num_failures = failures;
            const auto result = initializer.initialize(frame, cv::Mat());
            require(!result.second, "empty frames must not initialize even after fallback");
            const double expected = failures <= 10 ? base :
                std::max(floor, base * std::pow(2.0 / 3.0, failures / 10));
            require(std::abs(initializer.num_min_triangulated_points - expected) < 1e-9,
                    "stereo support fallback differs from the bounded policy");
        }
        initializer.check_cell_coverage = false;
        initializer.reset();
        require(initializer.num_failures == 0 && initializer.frames.empty() && !initializer.f_ref,
                "reset retained the previous initialization attempt");
        require(initializer.num_min_triangulated_points == base &&
                    initializer.check_cell_coverage == Parameters::kInitializerUseCellCoverageCheck,
                "reset retained relaxed initialization checks");
        for (auto sensor : {SensorType::MONOCULAR, SensorType::RGBD}) {
            Initializer other(sensor);
            const double initial = other.num_min_triangulated_points;
            other.num_failures = 1000;
            other.initialize(frame, cv::Mat());
            require(std::abs(other.num_min_triangulated_points - initial * 2.0 / 3.0) < 1e-9,
                    "non-stereo fallback changed");
        }
        std::cout << "test_initializer_support passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
