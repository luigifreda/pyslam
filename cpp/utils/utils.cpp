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

#include "utils.h"

namespace utils {

void extractPatch(const cv::Mat &image, const cv::KeyPoint &kp, const int &patch_size,
                  cv::Mat &patch, const bool use_orientation, const float scale_factor,
                  const int warp_flags) {
    cv::Mat M;
    if (use_orientation) {
        const float s = scale_factor * (float)kp.size / (float)patch_size;

        const float cosine = (kp.angle >= 0) ? cos(kp.angle * (float)CV_PI / 180.0f) : 1.f;
        const float sine = (kp.angle >= 0) ? sin(kp.angle * (float)CV_PI / 180.0f) : 0.f;

        float M_[] = {
            s * cosine, -s * sine,  (-s * cosine + s * sine) * patch_size / 2.0f + kp.pt.x,
            s * sine,   s * cosine, (-s * sine - s * cosine) * patch_size / 2.0f + kp.pt.y};
        M = cv::Mat(2, 3, CV_32FC1, M_).clone();
    } else {
        const float s = scale_factor * (float)kp.size / (float)patch_size;
        float M_[] = {s,   0.f, -s * patch_size / 2.0f + kp.pt.x,
                      0.f, s,   -s * patch_size / 2.0f + kp.pt.y};
        M = cv::Mat(2, 3, CV_32FC1, M_).clone();
    }

    cv::warpAffine(image, patch, M, cv::Size(patch_size, patch_size), warp_flags);
}

void extractPatches(const cv::Mat &image, const std::vector<cv::KeyPoint> &kps,
                    const int &patch_size, std::vector<cv::Mat> &patches,
                    const bool use_orientation, const float scale_factor, const int warp_flags) {
    patches.resize(kps.size());
    {
        py::gil_scoped_release release;
        for (size_t ii = 0, iiEnd = kps.size(); ii < iiEnd; ii++) {
            cv::Mat &patchii = patches[ii];
            const cv::KeyPoint &kpii = kps[ii];
            extractPatch(image, kpii, patch_size, patchii, use_orientation, scale_factor,
                         warp_flags);
        }
    }
}

std::pair<py::array_t<int>, py::array_t<int>>
goodMatchesSimple(const std::vector<std::pair<cv::DMatch, cv::DMatch>> &matches, float ratio_test) {
    std::vector<int> idxs1, idxs2;
    {
        py::gil_scoped_release release;
        for (const auto &pair : matches) {
            const cv::DMatch &m = pair.first;
            const cv::DMatch &n = pair.second;
            if (m.distance < ratio_test * n.distance) {
                idxs1.push_back(m.queryIdx);
                idxs2.push_back(m.trainIdx);
            }
        }
    }

    return {py::array_t<int>(idxs1.size(), idxs1.data()),
            py::array_t<int>(idxs2.size(), idxs2.data())};
}

std::pair<py::array_t<int>, py::array_t<int>>
goodMatchesOneToOne(const std::vector<std::vector<cv::DMatch>> &matches, float ratio_test) {
    std::vector<int> idxs1;
    std::vector<int> idxs2;

    const float INF = std::numeric_limits<float>::infinity();
    std::unordered_map<int, float> dist_match;
    std::unordered_map<int, size_t> index_match;

    {
        py::gil_scoped_release release;
        for (const auto &m_pair : matches) {
            if (m_pair.size() < 2)
                continue;
            const cv::DMatch &m = m_pair[0];
            const cv::DMatch &n = m_pair[1];

            if (m.distance > ratio_test * n.distance)
                continue;

            auto it = dist_match.find(m.trainIdx);
            if (it == dist_match.end()) {
                // New trainIdx match
                dist_match[m.trainIdx] = m.distance;
                idxs1.push_back(m.queryIdx);
                idxs2.push_back(m.trainIdx);
                index_match[m.trainIdx] = idxs2.size() - 1;
            } else if (m.distance < it->second) {
                // Replace existing match with better one
                size_t index = index_match[m.trainIdx];
                idxs1[index] = m.queryIdx;
                idxs2[index] = m.trainIdx;
                dist_match[m.trainIdx] = m.distance;
            }
        }
    }

    // Convert to numpy arrays without copy
    py::array_t<int> out_idxs1(idxs1.size(), idxs1.data());
    py::array_t<int> out_idxs2(idxs2.size(), idxs2.data());

    return std::make_pair(out_idxs1, out_idxs2);
}

py::tuple rowMatches(const std::vector<cv::KeyPoint> &kps1, const std::vector<cv::KeyPoint> &kps2,
                     const std::vector<cv::DMatch> &matches, float max_distance,
                     float max_row_distance, float max_disparity) {
    std::vector<int> idxs1, idxs2;

    {
        py::gil_scoped_release release;
        for (const auto &m : matches) {
            if (m.distance >= max_distance)
                continue;

            const auto &pt1 = kps1[m.queryIdx].pt;
            const auto &pt2 = kps2[m.trainIdx].pt;

            if (std::abs(pt1.y - pt2.y) < max_row_distance &&
                std::abs(pt1.x - pt2.x) < max_disparity) {
                idxs1.push_back(m.queryIdx);
                idxs2.push_back(m.trainIdx);
            }
        }
    }

    return py::make_tuple(py::array(idxs1.size(), idxs1.data()),
                          py::array(idxs2.size(), idxs2.data()));
}

std::pair<std::vector<int>, std::vector<int>>
rowMatches_np(py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
              py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
              const std::vector<cv::DMatch> &matches, float max_distance, float max_row_distance,
              float max_disparity) {
    if (kps1_np.ndim() != 2 || kps1_np.shape(1) != 2)
        throw std::runtime_error("kps1 must be of shape (N, 2)");
    if (kps2_np.ndim() != 2 || kps2_np.shape(1) != 2)
        throw std::runtime_error("kps2 must be of shape (N, 2)");

    auto kps1 = kps1_np.unchecked<2>();
    auto kps2 = kps2_np.unchecked<2>();

    std::vector<int> idxs1, idxs2;

    {
        py::gil_scoped_release release;
        for (const auto &m : matches) {
            if (m.distance >= max_distance)
                continue;

            float x1 = kps1(m.queryIdx, 0), y1 = kps1(m.queryIdx, 1);
            float x2 = kps2(m.trainIdx, 0), y2 = kps2(m.trainIdx, 1);

            if (std::abs(y1 - y2) < max_row_distance && std::abs(x1 - x2) < max_disparity) {
                idxs1.push_back(m.queryIdx);
                idxs2.push_back(m.trainIdx);
            }
        }
    }

    return {idxs1, idxs2};
}

py::tuple rowMatchesWithRatioTest(const std::vector<cv::KeyPoint> &kps1,
                                  const std::vector<cv::KeyPoint> &kps2,
                                  const std::vector<std::vector<cv::DMatch>> &matches,
                                  float max_distance, float max_row_distance, float max_disparity,
                                  float ratio_test) {
    std::vector<int> idxs1, idxs2;

    {
        py::gil_scoped_release release;
        for (const auto &pair : matches) {
            if (pair.size() < 2)
                continue;
            const auto &m = pair[0];
            const auto &n = pair[1];

            if (m.distance >= max_distance || m.distance >= ratio_test * n.distance)
                continue;

            const auto &pt1 = kps1[m.queryIdx].pt;
            const auto &pt2 = kps2[m.trainIdx].pt;

            if (std::abs(pt1.y - pt2.y) < max_row_distance &&
                std::abs(pt1.x - pt2.x) < max_disparity) {
                idxs1.push_back(m.queryIdx);
                idxs2.push_back(m.trainIdx);
            }
        }
    }

    return py::make_tuple(py::array(idxs1.size(), idxs1.data()),
                          py::array(idxs2.size(), idxs2.data()));
}

std::pair<std::vector<int>, std::vector<int>>
rowMatchesWithRatioTest_np(py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
                           py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
                           const std::vector<std::vector<cv::DMatch>> &knn_matches,
                           float max_distance, float max_row_distance, float max_disparity,
                           float ratio_test) {
    if (kps1_np.ndim() != 2 || kps1_np.shape(1) != 2)
        throw std::runtime_error("kps1 must be of shape (N, 2)");
    if (kps2_np.ndim() != 2 || kps2_np.shape(1) != 2)
        throw std::runtime_error("kps2 must be of shape (N, 2)");

    auto kps1 = kps1_np.unchecked<2>();
    auto kps2 = kps2_np.unchecked<2>();

    std::vector<int> idxs1, idxs2;
    {
        py::gil_scoped_release release;
        for (const auto &match_pair : knn_matches) {
            if (match_pair.size() < 2)
                continue;
            const auto &m = match_pair[0];
            const auto &n = match_pair[1];

            if (m.distance >= ratio_test * n.distance || m.distance >= max_distance)
                continue;

            int qidx = m.queryIdx;
            int tidx = m.trainIdx;

            float dy = std::abs(kps1(qidx, 1) - kps2(tidx, 1)); // y
            float dx = std::abs(kps1(qidx, 0) - kps2(tidx, 0)); // x

            if (dy < max_row_distance && dx < max_disparity) {
                idxs1.push_back(qidx);
                idxs2.push_back(tidx);
            }
        }
    }

    return {idxs1, idxs2};
}

py::tuple filterNonRowMatches(const std::vector<cv::KeyPoint> &kps1,
                              const std::vector<cv::KeyPoint> &kps2, const std::vector<int> &idxs1,
                              const std::vector<int> &idxs2, float max_row_distance,
                              float max_disparity) {
    std::vector<int> out_idxs1, out_idxs2;
    size_t N = idxs1.size();

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < N; ++i) {
            const auto &pt1 = kps1[idxs1[i]].pt;
            const auto &pt2 = kps2[idxs2[i]].pt;

            if (std::abs(pt1.y - pt2.y) < max_row_distance &&
                std::abs(pt1.x - pt2.x) < max_disparity) {
                out_idxs1.push_back(idxs1[i]);
                out_idxs2.push_back(idxs2[i]);
            }
        }
    }

    return py::make_tuple(py::array(out_idxs1.size(), out_idxs1.data()),
                          py::array(out_idxs2.size(), out_idxs2.data()));
}

std::pair<std::vector<int>, std::vector<int>>
filterNonRowMatches_np(py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
                       py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
                       py::array_t<int, py::array::c_style | py::array::forcecast> idxs1_np,
                       py::array_t<int, py::array::c_style | py::array::forcecast> idxs2_np,
                       float max_row_distance, float max_disparity) {
    if (kps1_np.ndim() != 2 || kps1_np.shape(1) != 2)
        throw std::runtime_error("kps1 must be of shape (N, 2)");
    if (kps2_np.ndim() != 2 || kps2_np.shape(1) != 2)
        throw std::runtime_error("kps2 must be of shape (N, 2)");

    if (idxs1_np.ndim() != 1 || idxs2_np.ndim() != 1)
        throw std::runtime_error("idxs1 and idxs2 must be 1D arrays");

    if (idxs1_np.shape(0) != idxs2_np.shape(0))
        throw std::runtime_error("idxs1 and idxs2 must have the same length");

    auto kps1 = kps1_np.unchecked<2>();
    auto kps2 = kps2_np.unchecked<2>();
    auto idxs1 = idxs1_np.unchecked<1>();
    auto idxs2 = idxs2_np.unchecked<1>();

    std::vector<int> out_idxs1, out_idxs2;
    size_t N = idxs1.shape(0);

    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < N; ++i) {
            int idx1 = idxs1(i);
            int idx2 = idxs2(i);

            float x1 = kps1(idx1, 0), y1 = kps1(idx1, 1);
            float x2 = kps2(idx2, 0), y2 = kps2(idx2, 1);

            if (std::abs(y1 - y2) < max_row_distance && std::abs(x1 - x2) < max_disparity) {
                out_idxs1.push_back(idx1);
                out_idxs2.push_back(idx2);
            }
        }
    }

    return {out_idxs1, out_idxs2};
}

py::array_t<float>
extractMeanColors(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
                  py::array_t<int, py::array::c_style | py::array::forcecast> img_coords, int delta,
                  std::array<float, 3> default_color) {
    auto buf_img = img.unchecked<3>();           // H x W x C
    auto buf_coords = img_coords.unchecked<2>(); // N x 2

    const int H = buf_img.shape(0);
    const int W = buf_img.shape(1);
    const int C = buf_img.shape(2);
    const int N = buf_coords.shape(0);
    const int patch_size = 1 + 2 * delta;
    const int patch_area = patch_size * patch_size;

    py::array_t<float> result({N, 3});
    auto buf_result = result.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
        for (int i = 0; i < N; ++i) {
            int x = buf_coords(i, 0);
            int y = buf_coords(i, 1);

            if (x - delta >= 0 && x + delta < W && y - delta >= 0 && y + delta < H) {
                for (int c = 0; c < C; ++c) {
                    float acc = 0.0f;
                    for (int dy = -delta; dy <= delta; ++dy) {
                        for (int dx = -delta; dx <= delta; ++dx) {
                            acc += static_cast<float>(buf_img(y + dy, x + dx, c));
                        }
                    }
                    buf_result(i, c) = acc / patch_area;
                }
            } else {
                buf_result(i, 0) = default_color[0];
                buf_result(i, 1) = default_color[1];
                buf_result(i, 2) = default_color[2];
            }
        }
    }

    return result;
}

} // namespace utils