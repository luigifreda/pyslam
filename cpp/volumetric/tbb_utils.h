/**
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

#include <memory>

#ifdef TBB_FOUND
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#endif

namespace volumetric {

class TBBUtils {
  public:
#ifdef TBB_FOUND
    // Set the maximum number of threads for TBB (global setting)
    // This affects all TBB parallel operations in this process
    // The global_control object persists until set_max_threads is called again
    // Returns the number of threads set
    static int set_max_threads(int num_threads) {
        if (num_threads <= 0) {
            // Use default (all available threads)
            num_threads =
                tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
        }
        // Use a function-local static unique_ptr to ensure the global_control persists
        // Reset it each time to change the thread limit
        static std::unique_ptr<tbb::global_control> gc;
        gc = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism,
                                                   num_threads);
        return num_threads;
    }

    // Get the current maximum number of threads for TBB
    static int get_max_threads() {
        return tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
    }
#else
    static int set_max_threads(int num_threads) { return num_threads; }
    static int get_max_threads() { return 1; }
#endif
};

} // namespace volumetric
