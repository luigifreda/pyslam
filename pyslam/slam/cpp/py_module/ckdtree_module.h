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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <tuple>
#include <vector>

#include "ckdtree_eigen.h"
#include "utils/messages.h"
#include "utils/numpy_helpers.h"

namespace py = pybind11;

#define USE_ZERO_COPY_KDTREE 1

// Helpers: check array shape/dtype/contiguity
inline void ensure_2d(const pybind11::array &arr, const char *name) {
    if (arr.ndim() != 2)
        throw std::invalid_argument(std::string(name) + " must be 2D (N,D)");
}

inline void ensure_1d_len(const pybind11::array &arr, ssize_t L, const char *name) {
    if (arr.ndim() != 1 || arr.shape(0) != L)
        throw std::invalid_argument(std::string(name) + " must be 1D of length " +
                                    std::to_string(L));
}

// Convert Python 1D float array to std::array<Scalar, D>
template <typename Scalar, int D>
std::array<Scalar, D>
to_fixed_query(const py::array_t<Scalar, py::array::c_style | py::array::forcecast> &x) {
    ensure_1d_len(x, D, "x");
    std::array<Scalar, D> q{};
    const Scalar *p = x.data();
    for (int i = 0; i < D; ++i)
        q[i] = p[i];
    return q;
}

// Bind fixed-size tree (D is template parameter)
template <typename Scalar, int D, typename IndexT = size_t>
void bind_fixed_tree(py::module_ &m, const char *pyname) {
    using Tree = pyslam::CKDTreeEigen<Scalar, D, IndexT>;
    py::class_<Tree, std::shared_ptr<Tree>>(m, pyname)
#if USE_ZERO_COPY_KDTREE
        // Zero-copy constructor: pass data pointer directly
        .def(py::init([](py::array_t<Scalar, py::array::c_style | py::array::forcecast> points) {
                 ensure_2d(points, "points");
                 if (points.shape(1) != D) {
                     MSG_RED_WARN("points.shape[1] must be D=" + std::to_string(D));
                     throw std::invalid_argument("points.shape[1] must be D=" + std::to_string(D));
                 }
                 // Validate row-major layout
                 if (!points.writeable()) {
                     MSG_RED_WARN("points array must be C-contiguous and writeable");
                     throw std::invalid_argument("points array must be C-contiguous and writeable");
                 }

                 // Use zero-copy constructor with data pointer
                 // NOTE: points must stay alive for the lifetime of the tree
                 return std::make_shared<Tree>(points.data(), points.shape(0));
             }),
             py::arg("points"),
             py::keep_alive<0, 1>(), // Keep the input array alive as long as the tree exists
             "Build KD-tree from points (N,D) [zero-copy].")
#else
        // Constructor from numpy array (N,D), copies internally
        .def(py::init([](py::array_t<Scalar, py::array::c_style | py::array::forcecast> points) {
                 ensure_2d(points, "points");
                 if (points.shape(1) != D)
                     throw std::invalid_argument("points.shape[1] must be D=");
                 // Copy into Eigen MatNx<Scalar, D>
                 Eigen::Matrix<Scalar, Eigen::Dynamic, D, Eigen::RowMajor> M(points.shape(0), D);
                 std::memcpy(M.data(), points.data(),
                             sizeof(Scalar) * static_cast<size_t>(points.size()));
                 return std::make_unique<Tree>(M); // copies inside the KD class as well (own_)
             }),
             py::arg("points"), "Build KD-tree from points (N,D) [copied].")
#endif
        .def_property_readonly("n", &Tree::n)
        .def_property_readonly_static("d", [](py::object) { return D; })

#if USE_ZERO_COPY_KDTREE
        // query(x, k, return_distance=True) -> (dists, idxs)
        .def(
            "query",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               size_t k, bool return_distance) {
                auto q = to_fixed_query<Scalar, D>(x);
                std::vector<Scalar> d;
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    auto res = self.query(q.data(), k, return_distance);
                    d = std::move(res.first);
                    idx = std::move(res.second);
                }

                py::array_t<Scalar> dists;
                if (return_distance) {
                    // Allocate on heap and manage lifetime with capsule
                    auto *vec = new std::vector<Scalar>(std::move(d));

                    dists = py::array_t<Scalar>({static_cast<py::ssize_t>(vec->size())},
                                                {static_cast<py::ssize_t>(sizeof(Scalar))},
                                                vec->data(), py::capsule(vec, [](void *p) {
                                                    delete static_cast<std::vector<Scalar> *>(p);
                                                }));
                } else {
                    dists = py::array_t<Scalar>();
                }

                // Indices need type conversion, so still need copy
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return py::make_tuple(dists, indices);
            },
            py::arg("x"), py::arg("k"), py::arg("return_distance") = true, "k-NN query")
#else
        // query(x, k, return_distance=True) -> (dists, idxs)
        .def(
            "query",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               size_t k, bool return_distance) {
                auto q = to_fixed_query<Scalar, D>(x);
                std::vector<Scalar> d;
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    auto res = self.query(q.data(), k, return_distance);
                    d = std::move(res.first);
                    idx = std::move(res.second);
                }
                py::array_t<Scalar> dists;
                if (return_distance) {
                    dists = py::array_t<Scalar>(d.size());
                    std::memcpy(dists.mutable_data(), d.data(), sizeof(Scalar) * d.size());
                } else {
                    dists = py::array_t<Scalar>(); // None in Python
                }
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return py::make_tuple(dists, indices);
            },
            py::arg("x"), py::arg("k"), py::arg("return_distance") = true, "k-NN query")
#endif
        // query_ball_point overloaded: single query or batch queries
        .def(
            "query_ball_point",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               Scalar r) {
                // Single query case: x is 1D, r is scalar
                auto q = to_fixed_query<Scalar, D>(x);
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    idx = self.query_ball_point(q.data(), static_cast<Scalar>(r));
                }
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return indices;
            },
            py::arg("x"), py::arg("r"), "Radius query: indices within distance r")

        .def(
            "query_ball_point",
            [](const Tree &self,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> queries,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> radii) {
                // Batch query case: queries is 2D, radii is 1D
                ensure_2d(queries, "queries");
                ensure_1d_len(radii, queries.shape(0), "radii");
                if (queries.shape(1) != D)
                    throw std::invalid_argument("queries.shape[1] must be D=" + std::to_string(D));

                const size_t num_queries = static_cast<size_t>(queries.shape(0));
                std::vector<std::vector<typename Tree::index_t>> results;
                {
                    py::gil_scoped_release release;
                    results =
                        self.query_ball_point_batch(queries.data(), radii.data(), num_queries);
                }

                py::list out;
                for (const auto &result : results) {
                    py::array_t<long long> indices(result.size());
                    auto *ip = indices.mutable_data();
                    for (size_t i = 0; i < result.size(); ++i)
                        ip[i] = static_cast<long long>(result[i]);
                    out.append(indices);
                }
                return out;
            },
            py::arg("queries"), py::arg("radii"),
            "Batch radius query: list of indices for each query point")

        // query_pairs(r) -> list[tuple(i,j)]
        .def(
            "query_pairs",
            [](const Tree &self, Scalar r) {
                std::vector<std::pair<typename Tree::index_t, typename Tree::index_t>> pairs;
                {
                    py::gil_scoped_release release;
                    pairs = self.query_pairs(static_cast<Scalar>(r));
                }
                py::list out;
                for (auto &pr : pairs)
                    out.append(py::make_tuple(static_cast<long long>(pr.first),
                                              static_cast<long long>(pr.second)));
                return out;
            },
            py::arg("r"), "All pairs (i<j) within distance r")

        // sparse_distance_matrix(other, max_distance) -> (i,j,dist) triplets
        .def(
            "sparse_distance_matrix",
            [](const Tree &self, const Tree &other, Scalar max_distance) {
                std::vector<std::tuple<typename Tree::index_t, typename Tree::index_t, Scalar>>
                    trips;
                {
                    py::gil_scoped_release release;
                    trips = self.sparse_distance_matrix(other, static_cast<Scalar>(max_distance));
                }
                py::list out;
                for (auto &t : trips)
                    out.append(py::make_tuple(static_cast<long long>(std::get<0>(t)),
                                              static_cast<long long>(std::get<1>(t)),
                                              std::get<2>(t)));
                return out;
            },
            py::arg("other"), py::arg("max_distance"),
            "Triplets (i_self, j_other, dist) for pairs within max_distance");
}

// Bind dynamic-size tree (IndexT is template parameter)
template <typename Scalar, typename IndexT = size_t>
void bind_dynamic_tree(py::module_ &m, const char *pyname) {
    using Tree = pyslam::CKDTreeEigenDyn<Scalar, IndexT>;
    py::class_<Tree, std::shared_ptr<Tree>>(m, pyname)
#if USE_ZERO_COPY_KDTREE
        // Zero-copy constructor: pass data pointer directly
        .def(py::init([](py::array_t<Scalar, py::array::c_style | py::array::forcecast> points) {
                 ensure_2d(points, "points");
                 const ssize_t N = points.shape(0);
                 const ssize_t D = points.shape(1);

                 // Validate row-major layout
                 if (!points.writeable()) {
                     throw std::invalid_argument("points array must be C-contiguous and writeable");
                 }

                 // Use zero-copy constructor with data pointer
                 // NOTE: points must stay alive for the lifetime of the tree
                 return std::make_shared<Tree>(points.data(), N, D);
             }),
             py::arg("points"), py::keep_alive<0, 1>(), // Keep input array alive
             "Build KD-tree from points (N,D) [zero-copy].")
#else
        // Constructor from numpy array (N,D), copies internally
        .def(py::init([](py::array_t<Scalar, py::array::c_style | py::array::forcecast> points) {
                 ensure_2d(points, "points");
                 const ssize_t N = points.shape(0);
                 const ssize_t D = points.shape(1);
                 Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(N, D);
                 std::memcpy(M.data(), points.data(),
                             sizeof(Scalar) * static_cast<size_t>(points.size()));
                 return std::make_unique<Tree>(Eigen::Ref<const decltype(M)>(M));
             }),
             py::arg("points"), "Build KD-tree from points (N,D) [copied].")
#endif
        .def_property_readonly("n", &Tree::n)
        .def_property_readonly("d", &Tree::d)

#if USE_ZERO_COPY_KDTREE
        // query(x, k, return_distance=True) -> (dists, idxs)
        .def(
            "query",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               size_t k, bool return_distance) {
                if (x.ndim() != 1)
                    throw std::invalid_argument("x must be 1D");
                if (x.shape(0) != self.d())
                    throw std::invalid_argument("len(x) must equal tree.d()");
                std::vector<Scalar> q(self.d());
                std::memcpy(q.data(), x.data(), sizeof(Scalar) * q.size());
                std::vector<Scalar> d;
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    auto res = self.query(q.data(), k, return_distance);
                    d = std::move(res.first);
                    idx = std::move(res.second);
                }

                py::array_t<Scalar> dists;
                if (return_distance) {
                    // Allocate on heap and manage lifetime with capsule
                    auto *vec = new std::vector<Scalar>(std::move(d));

                    dists = py::array_t<Scalar>({static_cast<py::ssize_t>(vec->size())},
                                                {static_cast<py::ssize_t>(sizeof(Scalar))},
                                                vec->data(), py::capsule(vec, [](void *p) {
                                                    delete static_cast<std::vector<Scalar> *>(p);
                                                }));
                } else {
                    dists = py::array_t<Scalar>();
                }

                // Indices need type conversion, so still need copy
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return py::make_tuple(dists, indices);
            },
            py::arg("x"), py::arg("k"), py::arg("return_distance") = true, "k-NN query")
#else
        .def(
            "query",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               size_t k, bool return_distance) {
                if (x.ndim() != 1)
                    throw std::invalid_argument("x must be 1D");
                if (x.shape(0) != self.d())
                    throw std::invalid_argument("len(x) must equal tree.d()");
                std::vector<Scalar> q(self.d());
                std::memcpy(q.data(), x.data(), sizeof(Scalar) * q.size());
                std::vector<Scalar> d;
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    auto res = self.query(q.data(), k, return_distance);
                    d = std::move(res.first);
                    idx = std::move(res.second);
                }
                py::array_t<Scalar> dists;
                if (return_distance) {
                    dists = py::array_t<Scalar>(d.size());
                    std::memcpy(dists.mutable_data(), d.data(), sizeof(Scalar) * d.size());
                } else {
                    dists = py::array_t<Scalar>();
                }
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return py::make_tuple(dists, indices);
            },
            py::arg("x"), py::arg("k"), py::arg("return_distance") = true)
#endif
        // query_ball_point overloaded: single query or batch queries
        .def(
            "query_ball_point",
            [](const Tree &self, py::array_t<Scalar, py::array::c_style | py::array::forcecast> x,
               Scalar r) {
                // Single query case: x is 1D, r is scalar
                if (x.ndim() != 1 || x.shape(0) != self.d())
                    throw std::invalid_argument("x must be 1D with length d()");
                std::vector<Scalar> q(self.d());
                std::memcpy(q.data(), x.data(), sizeof(Scalar) * q.size());
                std::vector<typename Tree::index_t> idx;
                {
                    py::gil_scoped_release release;
                    idx = self.query_ball_point(q.data(), r);
                }
                py::array_t<long long> indices(idx.size());
                auto *ip = indices.mutable_data();
                for (size_t i = 0; i < idx.size(); ++i)
                    ip[i] = static_cast<long long>(idx[i]);
                return indices;
            },
            py::arg("x"), py::arg("r"))

        .def(
            "query_ball_point",
            [](const Tree &self,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> queries,
               py::array_t<Scalar, py::array::c_style | py::array::forcecast> radii) {
                // Batch query case: queries is 2D, radii is 1D
                ensure_2d(queries, "queries");
                ensure_1d_len(radii, queries.shape(0), "radii");
                if (queries.shape(1) != self.d())
                    throw std::invalid_argument("queries.shape[1] must equal tree.d()");

                const size_t num_queries = static_cast<size_t>(queries.shape(0));
                std::vector<std::vector<typename Tree::index_t>> results;
                {
                    py::gil_scoped_release release;
                    results =
                        self.query_ball_point_batch(queries.data(), radii.data(), num_queries);
                }

                py::list out;
                for (const auto &result : results) {
                    py::array_t<long long> indices(result.size());
                    auto *ip = indices.mutable_data();
                    for (size_t i = 0; i < result.size(); ++i)
                        ip[i] = static_cast<long long>(result[i]);
                    out.append(indices);
                }
                return out;
            },
            py::arg("queries"), py::arg("radii"),
            "Batch radius query: list of indices for each query point")

        .def(
            "query_pairs",
            [](const Tree &self, Scalar r) {
                std::vector<std::pair<typename Tree::index_t, typename Tree::index_t>> pairs;
                {
                    py::gil_scoped_release release;
                    pairs = self.query_pairs(r);
                }
                py::list out;
                for (auto &pr : pairs)
                    out.append(py::make_tuple(static_cast<long long>(pr.first),
                                              static_cast<long long>(pr.second)));
                return out;
            },
            py::arg("r"))

        .def(
            "sparse_distance_matrix",
            [](const Tree &self, const Tree &other, Scalar max_distance) {
                std::vector<std::tuple<typename Tree::index_t, typename Tree::index_t, Scalar>>
                    trips;
                {
                    py::gil_scoped_release release;
                    trips = self.sparse_distance_matrix(other, static_cast<Scalar>(max_distance));
                }
                py::list out;
                for (auto &t : trips)
                    out.append(py::make_tuple(static_cast<long long>(std::get<0>(t)),
                                              static_cast<long long>(std::get<1>(t)),
                                              std::get<2>(t)));
                return out;
            },
            py::arg("other"), py::arg("max_distance"));
}

// Bind both fixed-size and dynamic-size trees
void bind_ckdtree(py::module &m) {

    bind_fixed_tree<float, 2>(m, "CKDTree2d_f");
    bind_fixed_tree<double, 2>(m, "CKDTree2d_d");

    bind_fixed_tree<float, 3>(m, "CKDTree3d_f");
    bind_fixed_tree<double, 3>(m, "CKDTree3d_d");

    bind_dynamic_tree<float>(m, "CKDTreeDyn_f");
    bind_dynamic_tree<double>(m, "CKDTreeDyn_d");
}
