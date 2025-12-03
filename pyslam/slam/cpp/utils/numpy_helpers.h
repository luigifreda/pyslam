// pyslam/slam/cpp/utils/numpy_utils.h
#pragma once

#include "eigen_aliases.h"

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace py = pybind11;

namespace pyslam {

template <typename T>
inline py::array_t<T> vector_to_numpy_zero_copy(std::vector<T> &vec,
                                                py::object owner = py::none()) {
    return py::array_t<T>({static_cast<py::ssize_t>(vec.size())},
                          {static_cast<py::ssize_t>(sizeof(T))}, vec.data(), owner);
}

template <typename T>
inline void numpy_to_vector_copy(py::object obj, std::vector<T> &vec, const std::string &name) {
    if (py::isinstance<py::array>(obj)) {
        py::array_t<T, py::array::c_style | py::array::forcecast> arr = py::array::ensure(obj);
        if (!arr) {
            throw std::runtime_error(name + ": not a valid numpy array");
        }
        vec.assign(arr.data(), arr.data() + arr.size());
    } else {
        vec = py::cast<std::vector<T>>(obj);
    }
}

template <typename EigenType>
inline py::array_t<typename EigenType::Scalar>
eigen_to_numpy_zero_copy(EigenType &matrix, py::object owner = py::none()) {
    using Scalar = typename EigenType::Scalar;
    constexpr bool IsRowMajor = EigenType::IsRowMajor;

    const py::ssize_t rows = static_cast<py::ssize_t>(matrix.rows());
    const py::ssize_t cols = static_cast<py::ssize_t>(matrix.cols());

    // NumPy expects: strides[0] = bytes to move one row, strides[1] = bytes to move one column
    const py::ssize_t row_stride =
        static_cast<py::ssize_t>(sizeof(Scalar) * (IsRowMajor ? cols : 1));
    const py::ssize_t col_stride =
        static_cast<py::ssize_t>(sizeof(Scalar) * (IsRowMajor ? 1 : rows));

    return py::array_t<Scalar>({rows, cols}, {row_stride, col_stride}, matrix.data(), owner);
}

template <typename EigenType>
inline void numpy_to_eigen_copy(py::object obj, EigenType &matrix, const std::string &name) {
    if (py::isinstance<py::array>(obj)) {
        py::array array = py::array::ensure(obj);
        if (!array) {
            throw std::runtime_error(name + ": not a valid numpy array");
        }

        if constexpr (std::is_same_v<EigenType, MatNx2d>) {
            if (array.ndim() != 2 || array.shape(1) != 2) {
                throw std::runtime_error(name + " must have shape (N,2)");
            }
        }

        using Scalar = typename EigenType::Scalar;
        // Force a C-contiguous buffer of the right dtype
        py::array_t<Scalar, py::array::c_style | py::array::forcecast> arr = array;

        const auto rows = static_cast<Eigen::Index>(arr.shape(0));
        const auto cols = static_cast<Eigen::Index>(arr.shape(1));

        // Read from row-major NumPy into a RowMajor Eigen temp, then assign
        using RowMajorMat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        RowMajorMat tmp(rows, cols);
        std::memcpy(tmp.data(), arr.data(), static_cast<size_t>(rows * cols) * sizeof(Scalar));

        matrix = tmp; // assigns with layout conversion if needed
    } else {
        matrix = py::cast<EigenType>(obj);
    }
}

} // namespace pyslam

//=============================================
//              Macros
//=============================================

// Define a vector property that is a numpy array and is zero-copy
#define DEFINE_VECTOR_PROPERTY_ZERO_COPY(class_type, member_name, value_type, property_name)       \
    .def_property(                                                                                 \
        property_name,                                                                             \
        [](class_type &self) {                                                                     \
            return pyslam::vector_to_numpy_zero_copy(self.member_name, py::cast(&self));           \
        },                                                                                         \
        [](class_type &self, py::object obj) {                                                     \
            pyslam::numpy_to_vector_copy(obj, self.member_name, property_name);                    \
        })

// Define an Eigen property that is a numpy array and is zero-copy
#define DEFINE_EIGEN_ZERO_COPY_PROPERTY(class_type, member_name, eigen_type, property_name)        \
    .def_property(                                                                                 \
        property_name,                                                                             \
        [](class_type &self) {                                                                     \
            return pyslam::eigen_to_numpy_zero_copy(self.member_name, py::cast(&self));            \
        },                                                                                         \
        [](class_type &self, py::object obj) {                                                     \
            pyslam::numpy_to_eigen_copy(obj, self.member_name, property_name);                     \
        })

// Define a Read-Only (RO) Eigen property exposed as a zero-copy NumPy view via a const getter
#define DEFINE_EIGEN_ZERO_COPY_RO_PROPERTY_FROM_GETTER(class_type, getter_expr, property_name)     \
    .def_property_readonly(property_name, [](class_type &self) {                                   \
        auto &ref_const = getter_expr;                                                             \
        auto &ref = const_cast<std::remove_reference_t<decltype(ref_const)> &>(ref_const);         \
        auto arr = pyslam::eigen_to_numpy_zero_copy(ref, py::cast(&self));                         \
        arr.attr("setflags")(py::arg("write") = false);                                            \
        return arr;                                                                                \
    })
