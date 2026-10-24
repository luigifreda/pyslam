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

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace pyslam {

class RotationHistogram {
  public:
    // NOTE: with 12 bins => new factor = 12/360 equals to old factor = 1/30
    explicit RotationHistogram(int histogram_length = 12)
        : histogram_length_(histogram_length),
          factor_(static_cast<double>(histogram_length) / 360.0), histo_(histogram_length),
          counts_(histogram_length, 0) {
        if (histogram_length_ <= 0) {
            throw std::invalid_argument("RotationHistogram: histogram_length must be > 0");
        }
    }

    void push(const double &rot_deg, const int &idx) {
        int bin = angleToBin(rot_deg);
        histo_[bin].push_back(idx);
        ++counts_[bin];
    }

    void push_entries(const std::vector<double> &rots_deg, const std::vector<int> &idxs) {
        const size_t n = rots_deg.size();
        if (idxs.size() != n) {
            throw std::invalid_argument("RotationHistogram: rots and idxs must have same size");
        }
        for (size_t i = 0; i < n; ++i) {
            int bin = angleToBin(rots_deg[i]);
            histo_[bin].push_back(idxs[i]);
            ++counts_[bin];
        }
    }

    std::vector<int> get_invalid_idxs() const {
        auto [i1, i2, i3] = compute_3_max();
        std::vector<int> out;
        for (int i = 0; i < histogram_length_; ++i) {
            if (i != i1 && i != i2 && i != i3) {
                out.insert(out.end(), histo_[i].begin(), histo_[i].end());
            }
        }
        return out;
    }

    std::vector<int> get_valid_idxs() const {
        auto [i1, i2, i3] = compute_3_max();
        std::vector<int> out;
        if (i1 != -1)
            out.insert(out.end(), histo_[i1].begin(), histo_[i1].end());
        if (i2 != -1)
            out.insert(out.end(), histo_[i2].begin(), histo_[i2].end());
        if (i3 != -1)
            out.insert(out.end(), histo_[i3].begin(), histo_[i3].end());
        return out;
    }

    std::string to_string() const {
        std::stringstream out;
        for (const auto &bin : histo_) {
            out << "[";
            for (const auto &e : bin) {
                out << e << ",";
            }
            out << "]";
        }
        return out.str();
    }

    // Find top-3 bins, with 10% thresholds for 2nd and 3rd
    std::tuple<int, int, int> compute_3_max() const {
        const auto better = [&](int candidate, int current) {
            if (candidate == -1)
                return false;
            if (current == -1)
                return true;
            const int cc = counts_[candidate];
            const int cr = counts_[current];
            if (cc > cr)
                return true;
            if (cc == cr && candidate > current)
                return true;
            return false;
        };

        int max1 = -1, max2 = -1, max3 = -1;
        for (int bin = 0; bin < histogram_length_; ++bin) {
            if (better(bin, max1)) {
                max3 = max2;
                max2 = max1;
                max1 = bin;
            } else if (better(bin, max2)) {
                max3 = max2;
                max2 = bin;
            } else if (better(bin, max3)) {
                max3 = bin;
            }
        }

        if (max1 != -1) {
            const double c1 = static_cast<double>(counts_[max1]);
            if (max2 != -1 && counts_[max2] < 0.1 * c1)
                max2 = -1;
            if (max3 != -1 && counts_[max3] < 0.1 * c1)
                max3 = -1;
        }

        return {max1, max2, max3};
    }

    // Equivalent to the Python staticmethod:
    // returns indices [0..num_matches-1] of matches that fall in top-3 orientation bins
    static std::vector<int> filter_matches_with_histogram_orientation(
        const std::vector<int> &idxs1, const std::vector<int> &idxs2,
        const std::vector<double> &angles1_deg, const std::vector<double> &angles2_deg) {

        if (idxs1.empty() || idxs2.empty())
            return {};
        if (idxs1.size() != idxs2.size()) {
            throw std::invalid_argument("idxs1 and idxs2 must have same length");
        }

        const size_t num_matches = idxs1.size();
        RotationHistogram rot_histo;

        // Push entries: rots = angles1[idxs1] - angles2[idxs2]
        for (size_t i = 0; i < num_matches; ++i) {
            const int i1 = idxs1[i];
            const int i2 = idxs2[i];
            if (i1 < 0 || i1 >= static_cast<int>(angles1_deg.size()) || i2 < 0 ||
                i2 >= static_cast<int>(angles2_deg.size())) {
                throw std::out_of_range("Index out of range in angles arrays");
            }
            const double rot = angles1_deg[i1] - angles2_deg[i2];
            rot_histo.push(rot, static_cast<int>(i)); // store match index
        }

        return rot_histo.get_valid_idxs();
    }

  private:
    int histogram_length_;
    double factor_;
    std::vector<std::vector<int>> histo_;
    std::vector<int> counts_;

    static double wrap_deg_0_360(const double &deg) {
        double m = std::fmod(deg, 360.0);
        if (m < 0.0)
            m += 360.0;
        return m;
    }

    // Match Python logic:
    // bin = int(round(rot * factor)); if bin == histogram_length => 0
    int angleToBin(const double &rot_deg) const {
        const double rot = wrap_deg_0_360(rot_deg);
        long bin = std::lround(rot * factor_);
        if (bin == histogram_length_)
            bin = 0;
        if (bin < 0 || bin > histogram_length_) {
            throw std::runtime_error("RotationHistogram: Invalid bin index");
        }
        return static_cast<int>(bin);
    }
};

} // namespace pyslam
