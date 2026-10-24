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
#include "eigen_aliases.h"
#include "utils/nanoflann.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <vector>

namespace pyslam {

// ==========================================================
// ---------- Adaptor: fixed D (compile-time) ----------
// ==========================================================
template <typename Scalar, int D> struct EigenRowMajorNxDAdaptor {
    static_assert(D > 0, "D must be > 0 for the fixed-D adaptor");
    const Scalar *data;
    size_t N;
    EigenRowMajorNxDAdaptor(const Scalar *ptr, size_t n) : data(ptr), N(n) {}
    inline size_t kdtree_get_point_count() const { return N; }
    inline Scalar kdtree_get_pt(size_t idx, size_t dim) const { return data[idx * D + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

// ==========================================================
// ---------- Adaptor: dynamic D (runtime) ----------
// ==========================================================
template <typename Scalar> struct EigenRowMajorNxDAdaptorDyn {
    const Scalar *data;
    size_t N;
    int D;
    EigenRowMajorNxDAdaptorDyn(const Scalar *ptr, size_t n, int d) : data(ptr), N(n), D(d) {}
    inline size_t kdtree_get_point_count() const { return N; }
    inline Scalar kdtree_get_pt(size_t idx, size_t dim) const { return data[idx * D + dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};

// ==========================================================
// Fixed-D KD-tree (D known at compile time: e.g., 2 or 3)
// ==========================================================
// CKDTreeEigen: Fixed-D KD-tree. D is known at compile time.
// D: number of dimensions
// IndexT: type of the index
// Note: The results are sorted by index.
template <typename Scalar, int D, typename IndexT = size_t> class CKDTreeEigen {
  public:
    static_assert(D > 0, "Use the dynamic specialization for runtime-D.");
    using index_t = IndexT;

    // Non-owning (zero-copy)
    CKDTreeEigen(const Scalar *data, size_t N, size_t leaf_max_size = 10)
        : N_(N), own_(), ext_(data), adapt_(data, N), index_(D, adapt_, {leaf_max_size}) {
        index_.buildIndex();
    }

    // Copying (own the data)
    template <int Cols = D>
    CKDTreeEigen(const Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Eigen::RowMajor> &M,
                 size_t leaf_max_size = 10)
        : N_(static_cast<size_t>(M.rows())),
          own_(M.data(), M.data() + static_cast<size_t>(M.rows()) * D), ext_(own_.data()),
          adapt_(ext_, N_), index_(D, adapt_, {leaf_max_size}) {
        static_assert(Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Eigen::RowMajor>::IsRowMajor,
                      "Matrix must be row-major");
        index_.buildIndex();
    }

    // Zero-copy from Eigen::Ref
    template <int Cols = D>
    CKDTreeEigen(
        const Eigen::Ref<const Eigen::Matrix<Scalar, Eigen::Dynamic, Cols, Eigen::RowMajor>> &M,
        size_t leaf_max_size = 10)
        : CKDTreeEigen(M.data(), static_cast<size_t>(M.rows()), leaf_max_size) {}

    size_t n() const { return N_; }
    static constexpr int d() { return D; }
    const Scalar *data_ptr() const { return ext_; }

    // query (Euclidean distances)
    std::pair<std::vector<Scalar>, std::vector<index_t>> query(const Scalar *x, size_t k,
                                                               bool return_distance = true) const {
        std::vector<index_t> idx(k);
        std::vector<Scalar> d2(k);
        nanoflann::KNNResultSet<Scalar, index_t> rs(k);
        rs.init(idx.data(), d2.data());
        index_.findNeighbors(rs, x, nanoflann::SearchParameters(0));
        if (return_distance) {
            for (auto &v : d2)
                v = static_cast<Scalar>(std::sqrt(static_cast<double>(v)));
            return {d2, idx};
        } else
            return {{}, idx};
    }

    template <typename Derived>
    std::pair<std::vector<Scalar>, std::vector<index_t>>
    query(const Eigen::MatrixBase<Derived> &x, size_t k, bool return_distance = true) const {
        Scalar q[D];
        for (int i = 0; i < D; ++i)
            q[i] = static_cast<Scalar>(x(i));
        return query(q, k, return_distance);
    }

    std::vector<index_t> query_ball_point(const Scalar *x, Scalar r) const {
        const Scalar r2 = r * r;
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        index_.radiusSearch(x, r2, matches, nanoflann::SearchParameters());
        std::vector<index_t> out;
        out.reserve(matches.size());
        for (auto &m : matches)
            out.push_back(m.first);

        // Sort the results by index
        std::sort(out.begin(), out.end());

        return out;
    }

    template <typename Derived>
    std::vector<index_t> query_ball_point(const Eigen::MatrixBase<Derived> &x, Scalar r) const {
        Scalar q[D];
        for (int i = 0; i < D; ++i)
            q[i] = static_cast<Scalar>(x(i));
        return query_ball_point(q, r);
    }

    // Batch query_ball_point: multiple query points with individual radii
    std::vector<std::vector<index_t>>
    query_ball_point_batch(const Scalar *queries, const Scalar *radii, size_t num_queries) const {
        std::vector<std::vector<index_t>> results;
        results.reserve(num_queries);

        for (size_t i = 0; i < num_queries; ++i) {
            const Scalar *q = queries + i * D;
            Scalar r = radii[i];
            results.push_back(query_ball_point(q, r));
        }

        return results;
    }

    // Batch query_ball_point with Eigen matrices
    template <typename DerivedQ, typename DerivedR>
    std::vector<std::vector<index_t>>
    query_ball_point_batch(const Eigen::MatrixBase<DerivedQ> &queries,
                           const Eigen::MatrixBase<DerivedR> &radii) const {
        const size_t num_queries = static_cast<size_t>(queries.rows());
        std::vector<std::vector<index_t>> results;
        results.reserve(num_queries);

        for (size_t i = 0; i < num_queries; ++i) {
            Scalar q[D];
            for (int j = 0; j < D; ++j)
                q[j] = static_cast<Scalar>(queries(i, j));
            Scalar r = static_cast<Scalar>(radii(i));
            results.push_back(query_ball_point(q, r));
        }

        return results;
    }

    std::vector<std::pair<index_t, index_t>> query_pairs(Scalar r) const {
        const Scalar r2 = r * r;
        std::vector<std::pair<index_t, index_t>> pairs;
        pairs.reserve(N_);
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        for (index_t i = 0; i < static_cast<index_t>(N_); ++i) {
            const Scalar *pi = data_ptr() + static_cast<size_t>(i) * D;
            matches.clear();
            index_.radiusSearch(pi, r2, matches, nanoflann::SearchParameters());
            for (auto &m : matches) {
                index_t j = m.first;
                if (j > i)
                    pairs.emplace_back(i, j);
            }
        }
        return pairs;
    }

    std::vector<std::tuple<index_t, index_t, Scalar>>
    sparse_distance_matrix(const CKDTreeEigen &other, Scalar max_d) const {
        const Scalar r2 = max_d * max_d;
        std::vector<std::tuple<index_t, index_t, Scalar>> out;
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        for (index_t j = 0; j < static_cast<index_t>(other.N_); ++j) {
            const Scalar *q = other.data_ptr() + static_cast<size_t>(j) * D;
            matches.clear();
            index_.radiusSearch(q, r2, matches, nanoflann::SearchParameters());
            for (auto &m : matches)
                out.emplace_back(m.first, j,
                                 static_cast<Scalar>(std::sqrt(static_cast<double>(m.second))));
        }
        return out;
    }

  private:
    size_t N_{0};                              // number of points
    std::vector<Scalar> own_;                  // non-owning (zero-copy)
    const Scalar *ext_{nullptr};               // pointer to the data
    EigenRowMajorNxDAdaptor<Scalar, D> adapt_; // adaptor to the data

    using Metric = nanoflann::L2_Simple_Adaptor<Scalar, EigenRowMajorNxDAdaptor<Scalar, D>>;
    using KDTree_t =
        nanoflann::KDTreeSingleIndexAdaptor<Metric, EigenRowMajorNxDAdaptor<Scalar, D>, D, index_t>;
    KDTree_t index_;
};

// ==========================================================
// Dynamic-D KD-tree (D known at runtime; works with MatNxM)
// ==========================================================
// CKDTreeEigenDyn: Dynamic-D KD-tree. D is known at runtime.
// Scalar: type of the scalar
// IndexT: type of the index
// Note: The results are sorted by index.
// D: number of dimensions (MatNxM.cols())
// IndexT: type of the index
template <typename Scalar, typename IndexT> class CKDTreeEigenDyn {
  public:
    using index_t = IndexT;

    // Non-owning (zero-copy)
    CKDTreeEigenDyn(const Scalar *data, size_t N, int D, size_t leaf_max_size = 10)
        : N_(N), D_(D), own_(), ext_(data), adapt_(data, N, D),
          index_(D, adapt_, {leaf_max_size}) // note: constructor takes runtime D
    {
        index_.buildIndex();
    }

    // Copying
    CKDTreeEigenDyn(const MatNxM<Scalar> &M, size_t leaf_max_size = 10)
        : N_(static_cast<size_t>(M.rows())), D_(static_cast<int>(M.cols())),
          own_(M.data(), M.data() + static_cast<size_t>(M.size())), ext_(own_.data()),
          adapt_(ext_, N_, D_), index_(D_, adapt_, {leaf_max_size}) {
        static_assert(MatNxM<Scalar>::IsRowMajor, "Matrix must be row-major");
        index_.buildIndex();
    }

    // Zero-copy
    CKDTreeEigenDyn(const Eigen::Ref<const MatNxM<Scalar>> &M, size_t leaf_max_size = 10)
        : CKDTreeEigenDyn(M.data(), static_cast<size_t>(M.rows()), static_cast<int>(M.cols()),
                          leaf_max_size) {}

    size_t n() const { return N_; }
    int d() const { return D_; }
    const Scalar *data_ptr() const { return ext_; }

    std::pair<std::vector<Scalar>, std::vector<index_t>> query(const Scalar *x, size_t k,
                                                               bool return_distance = true) const {
        std::vector<index_t> idx(k);
        std::vector<Scalar> d2(k);
        nanoflann::KNNResultSet<Scalar, index_t> rs(k);
        rs.init(idx.data(), d2.data());
        index_.findNeighbors(rs, x, nanoflann::SearchParameters(0));
        if (return_distance) {
            for (auto &v : d2)
                v = static_cast<Scalar>(std::sqrt(static_cast<double>(v)));
            return {d2, idx};
        } else
            return {{}, idx};
    }

    template <typename Derived>
    std::pair<std::vector<Scalar>, std::vector<index_t>>
    query(const Eigen::MatrixBase<Derived> &x, size_t k, bool return_distance = true) const {
        std::vector<Scalar> q(D_);
        for (int i = 0; i < D_; ++i)
            q[i] = static_cast<Scalar>(x(i));
        return query(q.data(), k, return_distance);
    }

    std::vector<index_t> query_ball_point(const Scalar *x, Scalar r) const {
        const Scalar r2 = r * r;
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        index_.radiusSearch(x, r2, matches, nanoflann::SearchParameters());
        std::vector<index_t> out;
        out.reserve(matches.size());
        for (auto &m : matches)
            out.push_back(m.first);

        // Sort the results by index
        std::sort(out.begin(), out.end());

        return out;
    }

    template <typename Derived>
    std::vector<index_t> query_ball_point(const Eigen::MatrixBase<Derived> &x, Scalar r) const {
        std::vector<Scalar> q(D_);
        for (int i = 0; i < D_; ++i)
            q[i] = static_cast<Scalar>(x(i));
        return query_ball_point(q.data(), r);
    }

    // Batch query_ball_point: multiple query points with individual radii
    std::vector<std::vector<index_t>>
    query_ball_point_batch(const Scalar *queries, const Scalar *radii, size_t num_queries) const {
        std::vector<std::vector<index_t>> results;
        results.reserve(num_queries);

        for (size_t i = 0; i < num_queries; ++i) {
            const Scalar *q = queries + i * D_;
            Scalar r = radii[i];
            results.push_back(query_ball_point(q, r));
        }

        return results;
    }

    // Batch query_ball_point with Eigen matrices
    template <typename DerivedQ, typename DerivedR>
    std::vector<std::vector<index_t>>
    query_ball_point_batch(const Eigen::MatrixBase<DerivedQ> &queries,
                           const Eigen::MatrixBase<DerivedR> &radii) const {
        const size_t num_queries = static_cast<size_t>(queries.rows());
        std::vector<std::vector<index_t>> results;
        results.reserve(num_queries);

        for (size_t i = 0; i < num_queries; ++i) {
            std::vector<Scalar> q(D_);
            for (int j = 0; j < D_; ++j)
                q[j] = static_cast<Scalar>(queries(i, j));
            Scalar r = static_cast<Scalar>(radii(i));
            results.push_back(query_ball_point(q.data(), r));
        }

        return results;
    }

    std::vector<std::pair<index_t, index_t>> query_pairs(Scalar r) const {
        const Scalar r2 = r * r;
        std::vector<std::pair<index_t, index_t>> pairs;
        pairs.reserve(N_);
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        for (index_t i = 0; i < static_cast<index_t>(N_); ++i) {
            const Scalar *pi = data_ptr() + static_cast<size_t>(i) * D_;
            matches.clear();
            index_.radiusSearch(pi, r2, matches, nanoflann::SearchParameters());
            for (auto &m : matches) {
                index_t j = m.first;
                if (j > i)
                    pairs.emplace_back(i, j);
            }
        }
        return pairs;
    }

    std::vector<std::tuple<index_t, index_t, Scalar>>
    sparse_distance_matrix(const CKDTreeEigenDyn &other, Scalar max_d) const {
        const Scalar r2 = max_d * max_d;
        std::vector<std::tuple<index_t, index_t, Scalar>> out;
        std::vector<nanoflann::ResultItem<index_t, Scalar>> matches;
        for (index_t j = 0; j < static_cast<index_t>(other.N_); ++j) {
            const Scalar *q = other.data_ptr() + static_cast<size_t>(j) * other.D_;
            matches.clear();
            index_.radiusSearch(q, r2, matches, nanoflann::SearchParameters());
            for (auto &m : matches)
                out.emplace_back(m.first, j,
                                 static_cast<Scalar>(std::sqrt(static_cast<double>(m.second))));
        }
        return out;
    }

  private:
    size_t N_{0};                              // number of points
    int D_{0};                                 // number of dimensions
    std::vector<Scalar> own_;                  // non-owning (zero-copy)
    const Scalar *ext_{nullptr};               // pointer to the data
    EigenRowMajorNxDAdaptorDyn<Scalar> adapt_; // adaptor to the data

    using Metric = nanoflann::L2_Simple_Adaptor<Scalar, EigenRowMajorNxDAdaptorDyn<Scalar>>;
    using KDTree_t = nanoflann::KDTreeSingleIndexAdaptor<Metric, EigenRowMajorNxDAdaptorDyn<Scalar>,
                                                         -1, index_t>;
    KDTree_t index_;
};

// ===============================

using cKDTree2f = CKDTreeEigen<float, 2, size_t>;
using cKDTree2d = CKDTreeEigen<double, 2, size_t>;

using cKDTree3f = CKDTreeEigen<float, 3, size_t>;
using cKDTree3d = CKDTreeEigen<double, 3, size_t>;

using cKDTreeDynf = CKDTreeEigenDyn<float, size_t>;
using cKDTreeDynd = CKDTreeEigenDyn<double, size_t>;

} // namespace pyslam