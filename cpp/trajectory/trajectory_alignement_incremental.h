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

#include "trajectory_alignment.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

namespace trajectory_tools {

struct AlignmentOptions {
    double max_align_dt = 1e-1; // max allowed |est_t - bracketing gt_t|
    bool find_scale = true;     // estimate a global scale (Sim(3) vs SE(3))
    double svd_eps = 1e-12;     // numerical guard
    bool verbose = false;
};

struct AlignmentResult {
    Eigen::Matrix4d T_gt_est = Eigen::Matrix4d::Identity(); // maps est -> gt
    Eigen::Matrix4d T_est_gt = Eigen::Matrix4d::Identity(); // inverse
    bool valid = false;                                     // true if last compute succeeded
    size_t n_pairs = 0;                                     // number of associated pairs used
    double sigma2_est = 0.0;                                // source variance used for scale
    Eigen::Vector3d mu_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_gt = Eigen::Vector3d::Zero();
    Eigen::Vector3d singvals = Eigen::Vector3d::Zero(); // SVD singular values (diag)
};

// ================================================================
// Incremental trajectory aligner without LBA
// ================================================================

// Accepts estimated trajectory samples one by one, associates them with GT,
// and computes the transform incrementally using online statistics.
// Useful for real-time applications where full trajectory is incrementally estimated but the
// old estimated samples are not changed (for instance no LBA is performed).
class IncrementalTrajectoryAlignerNoLBA {
  public:
    IncrementalTrajectoryAlignerNoLBA() = default;

    IncrementalTrajectoryAlignerNoLBA(const std::vector<double> &gt_timestamps,
                                      const std::vector<Eigen::Vector3d> &gt_t_wi)
        : opts_() {
        set_gt(gt_timestamps, gt_t_wi);
    }

    IncrementalTrajectoryAlignerNoLBA(const std::vector<double> &gt_timestamps,
                                      const std::vector<Eigen::Vector3d> &gt_t_wi,
                                      const AlignmentOptions &opts)
        : opts_(opts) {
        set_gt(gt_timestamps, gt_t_wi);
    }

    void set_options(const AlignmentOptions &opts) { opts_ = opts; }

    // Set / replace ground-truth trajectory. Timestamps must be strictly increasing.
    // (If not, we will sort them together.)
    void set_gt(const std::vector<double> &gt_timestamps,
                const std::vector<Eigen::Vector3d> &gt_t_wi) {
        gt_ts_ = gt_timestamps;
        gt_pos_ = gt_t_wi;
        if (gt_ts_.size() != gt_pos_.size()) {
            throw std::runtime_error("GT timestamps and positions size mismatch.");
        }
        // ensure sorted
        std::vector<std::pair<double, Eigen::Vector3d>> zipped;
        zipped.reserve(gt_ts_.size());
        for (size_t i = 0; i < gt_ts_.size(); ++i)
            zipped.emplace_back(gt_ts_[i], gt_pos_[i]);
        std::sort(zipped.begin(), zipped.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t i = 0; i < zipped.size(); ++i) {
            gt_ts_[i] = zipped[i].first;
            gt_pos_[i] = zipped[i].second;
        }

        reset();
    }

    // Clear all accumulated stats (keeps GT)
    void reset() {
        n_ = 0;
        mu_est_.setZero();
        mu_gt_.setZero();
        // C_ accumulates sum of (est - mu_est) * (gt - mu_gt)^T (3x3)
        C_.setZero();
        Sxx_est_ = 0.0; // sum of squared deviations of est
        Sxx_gt_ = 0.0;  // sum of squared deviations of gt
        last_valid_result_ = AlignmentResult{};
        last_valid_result_.valid = false;
    }

    // Try to add an estimated position sample (world) at 'timestamp'.
    // Returns true if it was associated to GT (within max_align_dt) and incorporated.
    bool add_estimate(double timestamp, const Eigen::Vector3d &est_t_wi) {
        if (gt_ts_.size() < 2)
            return false;

        // find bracketing gt samples using upper_bound
        auto upper = std::upper_bound(gt_ts_.begin(), gt_ts_.end(), timestamp);
        int j = static_cast<int>(std::distance(gt_ts_.begin(), upper)) - 1;
        if (j < 0 || j >= static_cast<int>(gt_ts_.size()) - 1)
            return false;

        const double t0 = gt_ts_[j], t1 = gt_ts_[j + 1];
        const double dt = timestamp - t0;
        const double dgt = t1 - t0;
        const double abs_dt = std::abs(dt);
        if (dgt <= 0.0 || abs_dt > opts_.max_align_dt)
            return false;

        const double ratio = dt / dgt; // dt >=0 by construction; ratio in [0,1]
        const Eigen::Vector3d gt_interp = (1.0 - ratio) * gt_pos_[j] + ratio * gt_pos_[j + 1];

        // Sample was successfully associated - incorporate it into statistics
        // Online paired update (Welford/Chan for cross-cov and var)
        // Keep Σ (est,gt) convention: C_ = Σ (est_i - μ_est)(gt_i - μ_gt)^T
        n_ += 1;

        const Eigen::Vector3d delta_x = est_t_wi - mu_est_;
        const Eigen::Vector3d delta_y = gt_interp - mu_gt_;

        mu_est_ += delta_x / static_cast<double>(n_);
        mu_gt_ += delta_y / static_cast<double>(n_);

        // After mean update, the deviations to use:
        const Eigen::Vector3d dev_x = est_t_wi - mu_est_; // x - mu_x_new
        const Eigen::Vector3d dev_y = gt_interp - mu_gt_; // y - mu_y_new

        // Cross-cov accumulation (sum form, will normalize by n when needed)
        // See: Chan et al. 1979 / Pebay 2008 for online covariance updates
        C_.noalias() += delta_x * dev_y.transpose();

        // Source variance accumulation (sum of squared deviations)
        Sxx_est_ += delta_x.dot(dev_x);
        Sxx_gt_ += delta_y.dot(dev_y);

        // Recompute transform (may return false if n_ < 3 or other checks fail,
        // but sample was successfully incorporated)
        recompute_transform();

        // Return true since sample was successfully associated and incorporated
        return true;
    }

    // Current result (last successfully computed)
    const AlignmentResult &result() const { return last_valid_result_; }

  private:
    bool recompute_transform() {
        AlignmentResult out;
        out.n_pairs = n_;
        out.mu_est = mu_est_;
        out.mu_gt = mu_gt_;

        if (n_ < 3) {
            last_valid_result_ = out; // not valid yet, but return partial info
            return false;
        }

        const double n_d = static_cast<double>(n_);
        // C_ accumulates (est - mu_est)(gt - mu_gt)^T, but we need (gt - mu_gt)(est - mu_est)^T
        // to match the batch version convention, so transpose it
        const Eigen::Matrix3d Sigma = (C_ / n_d).transpose(); // E[(gt-μgt)(est-μest)^T]
        const double sigma2_est = Sxx_est_ / n_d;             // E[||est-μest||^2]
        const double variance_gt = Sxx_gt_ / n_d;             // E[||gt-μgt||^2]
        out.sigma2_est = sigma2_est;

        if (!(Sigma.allFinite()) || !std::isfinite(sigma2_est)) {
            last_valid_result_ = out;
            return false;
        }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Matrix3d U = svd.matrixU();
        const Eigen::Matrix3d V = svd.matrixV();
        Eigen::Vector3d svals = svd.singularValues();
        out.singvals = svals;

        // Reflection handling
        Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
        if ((U * V.transpose()).determinant() < 0.0)
            S(2, 2) = -1.0;

        const Eigen::Matrix3d R = U * S * V.transpose();

        double scale = 1.0;
        if (opts_.find_scale) {
            // Match batch version: scale = variance_gt / trace(diag(singvals) * S)
            const double numer = variance_gt;
            const double denom = (svals.asDiagonal() * S).trace(); // Tr(diag(σ) S)
            if (!(std::isfinite(numer) && std::isfinite(denom)) || denom <= opts_.svd_eps) {
                last_valid_result_ = out;
                return false;
            }
            scale = numer / denom; // Umeyama scale for est->gt
            // Check for degenerate scale (too small, would cause numerical issues in inverse)
            if (scale <= opts_.svd_eps) {
                last_valid_result_ = out;
                return false;
            }
        }

        const Eigen::Vector3d t = mu_gt_ - scale * R * mu_est_;

        // Build transforms
        out.T_gt_est.setIdentity();
        out.T_gt_est.topLeftCorner<3, 3>() = scale * R;
        out.T_gt_est.topRightCorner<3, 1>() = t;

        out.T_est_gt.setIdentity();
        const Eigen::Matrix3d sR_inv = (1.0 / scale) * R.transpose();
        out.T_est_gt.topLeftCorner<3, 3>() = sR_inv;
        out.T_est_gt.topRightCorner<3, 1>() = -sR_inv * t;

        const bool finite_ok = out.T_gt_est.allFinite() && out.T_est_gt.allFinite();
        out.valid = finite_ok;

        if (opts_.verbose) {
            std::cout << "[IncrementalTrajectoryAlignerNoLBA] n=" << n_ << " scale=" << scale
                      << " singvals=" << svals.transpose() << " valid=" << out.valid << std::endl;
        }

        last_valid_result_ = out;
        return out.valid;
    }

  private:
    // GT data
    std::vector<double> gt_ts_;
    std::vector<Eigen::Vector3d> gt_pos_;

    // Options
    AlignmentOptions opts_;

    // Online stats
    size_t n_ = 0;
    Eigen::Vector3d mu_est_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d mu_gt_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d C_ = Eigen::Matrix3d::Zero(); // sum (est-μe)(gt-μg)^T
    double Sxx_est_ = 0.0;                        // sum ||est-μe||^2
    double Sxx_gt_ = 0.0;                         // sum ||gt-μg||^2

    // Last good (or attempted) result
    AlignmentResult last_valid_result_;
};

// ================================================================
// Incremental trajectory aligner
// ================================================================

// Batch-recompute version: stores all data and recomputes from scratch
// Accepts full trajectory each time, updates associations for new samples,
// and updates positions for existing samples (for SLAM readjustments)
class IncrementalTrajectoryAligner {
  public:
    IncrementalTrajectoryAligner() = default;

    IncrementalTrajectoryAligner(const std::vector<double> &gt_timestamps,
                                 const std::vector<Eigen::Vector3d> &gt_t_wi)
        : opts_() {
        set_gt(gt_timestamps, gt_t_wi);
    }

    IncrementalTrajectoryAligner(const std::vector<double> &gt_timestamps,
                                 const std::vector<Eigen::Vector3d> &gt_t_wi,
                                 const AlignmentOptions &opts)
        : opts_(opts) {
        set_gt(gt_timestamps, gt_t_wi);
    }

    void set_options(const AlignmentOptions &opts) { opts_ = opts; }

    // Set / replace ground-truth trajectory. Timestamps must be strictly increasing.
    // (If not, we will sort them together.)
    void set_gt(const std::vector<double> &gt_timestamps,
                const std::vector<Eigen::Vector3d> &gt_t_wi) {
        gt_ts_ = gt_timestamps;
        gt_pos_ = gt_t_wi;
        if (gt_ts_.size() != gt_pos_.size()) {
            throw std::runtime_error("GT timestamps and positions size mismatch.");
        }
        // ensure sorted
        std::vector<std::pair<double, Eigen::Vector3d>> zipped;
        zipped.reserve(gt_ts_.size());
        for (size_t i = 0; i < gt_ts_.size(); ++i)
            zipped.emplace_back(gt_ts_[i], gt_pos_[i]);
        std::sort(zipped.begin(), zipped.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t i = 0; i < zipped.size(); ++i) {
            gt_ts_[i] = zipped[i].first;
            gt_pos_[i] = zipped[i].second;
        }

        reset();
    }

    // Clear all accumulated data (keeps GT)
    void reset() {
        est_ts_.clear();
        est_pos_.clear();
        gt_interp_pos_.clear();
        has_association_.clear();
        last_valid_result_ = AlignmentResult{};
        last_valid_result_.valid = false;
    }

    // Update with full estimated trajectory.
    // Stores the full new trajectory and updates associations only for new samples
    // (those with index >= current size of stored trajectory) or samples that previously
    // didn't have an association. Reuses existing associations for samples that already
    // have them (assumes timestamps don't change).
    // Then recomputes transform from scratch using all associations.
    // Returns the number of new associations created.
    size_t update_trajectory(const std::vector<double> &est_timestamps,
                             const std::vector<Eigen::Vector3d> &est_positions) {
        if (est_timestamps.size() != est_positions.size()) {
            throw std::runtime_error("Estimated timestamps and positions size mismatch.");
        }
        if (gt_ts_.size() < 2) {
            return 0;
        }

        const size_t old_size = est_ts_.size();
        size_t num_new = 0;

        // Save old timestamps for comparison before overwriting
        std::vector<double> old_est_ts_ = est_ts_;

        // Store the full new trajectory (replaces old one)
        est_ts_ = est_timestamps;
        est_pos_ = est_positions;

        // Resize GT interpolated positions and association flags to match new trajectory size
        const size_t new_size = est_ts_.size();
        gt_interp_pos_.resize(new_size);
        has_association_.resize(new_size);

        // Only compute associations for new samples or samples that previously didn't have one
        // Reuse existing associations for samples that already have them (only if timestamps
        // haven't changed)
        for (size_t i = 0; i < est_timestamps.size(); ++i) {
            // Reuse existing association if available (for i < old_size and previously had
            // association) and timestamp hasn't changed
            if (i < old_size && i < old_est_ts_.size() && has_association_[i]) {
                // Verify timestamp hasn't changed (within numerical precision)
                const double timestamp_diff = std::abs(est_timestamps[i] - old_est_ts_[i]);
                if (timestamp_diff < 1e-10) {
                    // Keep existing association - gt_interp_pos_[i] and has_association_[i] are
                    // already set. No need to recompute.
                    continue;
                }
                // Timestamp changed, need to recompute association
                has_association_[i] = false;
            }

            // Compute new association for:
            // - New samples (i >= old_size)
            // - Existing samples that previously didn't have an association
            const double timestamp = est_timestamps[i];

            // Try to associate with GT
            auto upper = std::upper_bound(gt_ts_.begin(), gt_ts_.end(), timestamp);
            int j = static_cast<int>(std::distance(gt_ts_.begin(), upper)) - 1;
            if (j >= 0 && j < static_cast<int>(gt_ts_.size()) - 1) {
                const double t0 = gt_ts_[j], t1 = gt_ts_[j + 1];
                const double dt = timestamp - t0;
                const double dgt = t1 - t0;
                const double abs_dt = std::abs(dt);
                // Match batch alignment logic: check dt >= 0, dgt > 0, and abs_dt <= max_align_dt
                if (dt >= 0.0 && dgt > 0.0 && abs_dt <= opts_.max_align_dt) {
                    // Successfully associated
                    const double ratio = dt / dgt; // ratio in [0,1]
                    const Eigen::Vector3d gt_interp =
                        (1.0 - ratio) * gt_pos_[j] + ratio * gt_pos_[j + 1];

                    gt_interp_pos_[i] = gt_interp;
                    has_association_[i] = true;
                    num_new++; // Count new associations
                } else {
                    // No valid association
                    has_association_[i] = false;
                }
            } else {
                // No valid association (timestamp outside GT range)
                has_association_[i] = false;
            }
        }

        // Recompute transform from scratch using all stored pairs
        recompute_transform();

        return num_new;
    }

    // Get the number of stored associations
    size_t num_associations() const { return est_ts_.size(); }

    // Get stored estimated timestamps
    const std::vector<double> &get_estimated_timestamps() const { return est_ts_; }

    // Get stored estimated positions
    const std::vector<Eigen::Vector3d> &get_estimated_positions() const { return est_pos_; }

    // Get stored GT interpolated positions
    const std::vector<Eigen::Vector3d> &get_gt_interpolated_positions() const {
        return gt_interp_pos_;
    }

    // Get only associated pairs (positions with valid GT associations)
    std::tuple<std::vector<double>, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
    get_associated_pairs() const {
        std::vector<double> valid_ts;
        std::vector<Eigen::Vector3d> valid_est_pos;
        std::vector<Eigen::Vector3d> valid_gt_pos;
        valid_ts.reserve(est_ts_.size());
        valid_est_pos.reserve(est_pos_.size());
        valid_gt_pos.reserve(est_pos_.size());

        for (size_t i = 0; i < est_pos_.size(); ++i) {
            if (has_association_[i]) {
                valid_ts.push_back(est_ts_[i]);
                valid_est_pos.push_back(est_pos_[i]);
                valid_gt_pos.push_back(gt_interp_pos_[i]);
            }
        }

        return std::make_tuple(valid_ts, valid_est_pos, valid_gt_pos);
    }

    // Current result (last successfully computed)
    const AlignmentResult &result() const { return last_valid_result_; }

  private:
    void recompute_transform() {
        AlignmentResult out;

        // Collect only samples with valid GT associations
        std::vector<Eigen::Vector3d> valid_est_pos;
        std::vector<Eigen::Vector3d> valid_gt_pos;
        valid_est_pos.reserve(est_pos_.size());
        valid_gt_pos.reserve(est_pos_.size());

        for (size_t i = 0; i < est_pos_.size(); ++i) {
            if (has_association_[i]) {
                valid_est_pos.push_back(est_pos_[i]);
                valid_gt_pos.push_back(gt_interp_pos_[i]);
            }
        }

        out.n_pairs = valid_est_pos.size();

        if (valid_est_pos.size() < 3) {
            last_valid_result_ = out; // not valid yet, but return partial info
            return;
        }

        // Compute transform from scratch using batch alignment
        auto [T_gt_est, T_est_gt, valid] =
            align_3d_points_with_svd(valid_gt_pos, valid_est_pos, opts_.find_scale);

        out.T_gt_est = T_gt_est;
        out.T_est_gt = T_est_gt;
        out.valid = valid;

        if (valid) {
            // Compute additional statistics for consistency with the other aligner
            Eigen::Vector3d mu_est = Eigen::Vector3d::Zero();
            Eigen::Vector3d mu_gt = Eigen::Vector3d::Zero();
            for (size_t i = 0; i < valid_est_pos.size(); ++i) {
                mu_est += valid_est_pos[i];
                mu_gt += valid_gt_pos[i];
            }
            mu_est /= static_cast<double>(valid_est_pos.size());
            mu_gt /= static_cast<double>(valid_est_pos.size());
            out.mu_est = mu_est;
            out.mu_gt = mu_gt;

            // Compute variance
            double sigma2_est = 0.0;
            for (size_t i = 0; i < valid_est_pos.size(); ++i) {
                const Eigen::Vector3d dev = valid_est_pos[i] - mu_est;
                sigma2_est += dev.squaredNorm();
            }
            sigma2_est /= static_cast<double>(valid_est_pos.size());
            out.sigma2_est = sigma2_est;

            // Compute SVD for singular values (recompute covariance for this)
            Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < valid_est_pos.size(); ++i) {
                const Eigen::Vector3d centered_gt = valid_gt_pos[i] - mu_gt;
                const Eigen::Vector3d centered_est = valid_est_pos[i] - mu_est;
                cov.noalias() += centered_gt * centered_est.transpose();
            }
            cov /= static_cast<double>(valid_est_pos.size());

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
            out.singvals = svd.singularValues();
        }

        if (opts_.verbose) {
            std::cout << "[IncrementalTrajectoryAligner] n=" << valid_est_pos.size()
                      << " valid=" << out.valid;
            if (out.valid) {
                std::cout << " singvals=" << out.singvals.transpose();
            }
            std::cout << std::endl;
        }

        last_valid_result_ = out;
    }

  private:
    // GT data
    std::vector<double> gt_ts_;
    std::vector<Eigen::Vector3d> gt_pos_;

    // Options
    AlignmentOptions opts_;

    // Stored estimated trajectory data
    std::vector<double> est_ts_;                 // estimated timestamps (full trajectory)
    std::vector<Eigen::Vector3d> est_pos_;       // estimated positions (full trajectory)
    std::vector<Eigen::Vector3d> gt_interp_pos_; // associated GT interpolated positions
    std::vector<bool> has_association_; // flag indicating if sample has valid GT association

    // Last good (or attempted) result
    AlignmentResult last_valid_result_;
};

} // namespace trajectory_tools
