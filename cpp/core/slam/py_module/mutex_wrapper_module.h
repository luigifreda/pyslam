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

#include <mutex>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mutex>

#include "py_wrappers.h"

namespace py = pybind11;

// Register the PyMutexWrapper class
void bind_mutex_wrapper(pybind11::module &m) {

    // NOTE: These classes are just used for wrapping the std::mutex and std::recursive_mutex
    // classes of the Map objects in Python.
    py::class_<pyslam::PyMutexWrapper, std::shared_ptr<pyslam::PyMutexWrapper>>(m, "MutexWrapper")
        .def(py::init<std::mutex &>())
        .def("lock", &pyslam::PyMutexWrapper::lock)
        .def("unlock", &pyslam::PyMutexWrapper::unlock)
        .def("acquire", &pyslam::PyMutexWrapper::acquire, py::arg("blocking") = true,
             py::arg("timeout") = -1)
        .def("release", &pyslam::PyMutexWrapper::release)
        .def("locked", &pyslam::PyMutexWrapper::locked)
        .def("__enter__", &pyslam::PyMutexWrapper::__enter__, py::return_value_policy::reference)
        .def("__exit__", &pyslam::PyMutexWrapper::__exit__);

    py::class_<pyslam::PyRMutexWrapper, std::shared_ptr<pyslam::PyRMutexWrapper>>(
        m, "RecursiveMutexWrapper")
        .def(py::init<std::recursive_mutex &>())
        .def("lock", &pyslam::PyRMutexWrapper::lock)
        .def("unlock", &pyslam::PyRMutexWrapper::unlock)
        .def("acquire", &pyslam::PyRMutexWrapper::acquire, py::arg("blocking") = true,
             py::arg("timeout") = -1)
        .def("release", &pyslam::PyRMutexWrapper::release)
        .def("locked", &pyslam::PyRMutexWrapper::locked)
        .def("__enter__", &pyslam::PyRMutexWrapper::__enter__, py::return_value_policy::reference)
        .def("__exit__", &pyslam::PyRMutexWrapper::__exit__);

    py::class_<pyslam::PyTMutexWrapper, std::shared_ptr<pyslam::PyTMutexWrapper>>(
        m, "TimedMutexWrapper")
        .def(py::init<std::timed_mutex &>())
        .def("lock", &pyslam::PyTMutexWrapper::lock)
        .def("unlock", &pyslam::PyTMutexWrapper::unlock)
        .def("acquire", &pyslam::PyTMutexWrapper::acquire, py::arg("blocking") = true,
             py::arg("timeout") = -1)
        .def("release", &pyslam::PyTMutexWrapper::release)
        .def("locked", &pyslam::PyTMutexWrapper::locked)
        .def("__enter__", &pyslam::PyTMutexWrapper::__enter__, py::return_value_policy::reference)
        .def("__exit__", &pyslam::PyTMutexWrapper::__exit__);

    py::class_<pyslam::PyTRMutexWrapper, std::shared_ptr<pyslam::PyTRMutexWrapper>>(
        m, "RecursiveTimedMutexWrapper")
        .def(py::init<std::recursive_timed_mutex &>())
        .def("lock", &pyslam::PyTRMutexWrapper::lock)
        .def("unlock", &pyslam::PyTRMutexWrapper::unlock)
        .def("acquire", &pyslam::PyTRMutexWrapper::acquire, py::arg("blocking") = true,
             py::arg("timeout") = -1)
        .def("release", &pyslam::PyTRMutexWrapper::release)
        .def("locked", &pyslam::PyTRMutexWrapper::locked)
        .def("__enter__", &pyslam::PyTRMutexWrapper::__enter__, py::return_value_policy::reference)
        .def("__exit__", &pyslam::PyTRMutexWrapper::__exit__);
}
