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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "opencv_type_casters.h"
#include "image_prefetcher.h"

namespace py = pybind11;

PYBIND11_MODULE(data_thread_reader, m) {
    m.doc() = "Background dataset image reader with bounded per-stream queues";

    py::class_<pyslam::ImagePrefetcher>(m, "ImagePrefetcher")
        .def(py::init<int>(), py::arg("capacity") = 6)
        .def("add_stream", &pyslam::ImagePrefetcher::add_stream, py::arg("name"), py::arg("paths"),
             py::arg("imread_flag"), py::arg("npy") = false)
        .def("add_clahe", &pyslam::ImagePrefetcher::add_clahe, py::arg("name"), py::arg("clip"),
             py::arg("tile"))
        .def("add_remap", &pyslam::ImagePrefetcher::add_remap, py::arg("name"), py::arg("map1"),
             py::arg("map2"))
        .def("add_resize", &pyslam::ImagePrefetcher::add_resize, py::arg("name"), py::arg("width"),
             py::arg("height"))
        .def("add_gray_to_rgb", &pyslam::ImagePrefetcher::add_gray_to_rgb, py::arg("name"))
        .def("add_scale_float", &pyslam::ImagePrefetcher::add_scale_float, py::arg("name"),
             py::arg("factor"))
        .def("set_expected_size", &pyslam::ImagePrefetcher::set_expected_size, py::arg("name"),
             py::arg("width"), py::arg("height"))
        .def("start", &pyslam::ImagePrefetcher::start)
        .def("stop", &pyslam::ImagePrefetcher::stop)
        .def("has_stream", &pyslam::ImagePrefetcher::has_stream, py::arg("name"))
        .def("stream_length", &pyslam::ImagePrefetcher::stream_length, py::arg("name"))
        .def("unpublished_count", &pyslam::ImagePrefetcher::unpublished_count)
        .def("capacity", &pyslam::ImagePrefetcher::capacity)
        .def(
            "get",
            [](pyslam::ImagePrefetcher &self, const std::string &name, int frame_id) {
                if (!self.has_stream(name))
                    throw py::key_error(name);
                pyslam::PrefetchImage image;
                {
                    py::gil_scoped_release release;
                    image = self.get(name, frame_id);
                }
                if (!image.ok || image.image.empty())
                    return py::object(py::none());
                return py::cast(std::move(image.image));
            },
            py::arg("name"), py::arg("frame_id"));
}
