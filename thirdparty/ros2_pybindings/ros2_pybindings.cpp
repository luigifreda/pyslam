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

#include <opencv2/core.hpp>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#include "ros2_bag_sync_reader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ros2_pybindings, m) {

    // Bind builtin_interfaces::msg::Time
    py::class_<builtin_interfaces::msg::Time, std::shared_ptr<builtin_interfaces::msg::Time>>(
        m, "Time")
        .def(py::init<>())
        .def_readwrite("sec", &builtin_interfaces::msg::Time::sec)
        .def_readwrite("nanosec", &builtin_interfaces::msg::Time::nanosec);

    // Bind std_msgs::msg::Header
    py::class_<std_msgs::msg::Header, std::shared_ptr<std_msgs::msg::Header>>(m, "Header")
        .def(py::init<>())
        .def_readwrite("stamp", &std_msgs::msg::Header::stamp)
        .def_readwrite("frame_id", &std_msgs::msg::Header::frame_id);

    // Bind sensor_msgs::msg::Image
    py::class_<sensor_msgs::msg::Image, std::shared_ptr<sensor_msgs::msg::Image>>(m, "Image")
        .def(py::init<>())
        .def_readwrite("header", &sensor_msgs::msg::Image::header)
        .def_readwrite("height", &sensor_msgs::msg::Image::height)
        .def_readwrite("width", &sensor_msgs::msg::Image::width)
        .def_readwrite("encoding", &sensor_msgs::msg::Image::encoding)
        .def_readwrite("is_bigendian", &sensor_msgs::msg::Image::is_bigendian)
        .def_readwrite("step", &sensor_msgs::msg::Image::step)
        .def_readwrite("data", &sensor_msgs::msg::Image::data);

    // Bind Ros2BagSyncReaderATS
    py::class_<Ros2BagSyncReaderATS, std::shared_ptr<Ros2BagSyncReaderATS>>(m,
                                                                            "Ros2BagSyncReaderATS")
        .def(py::init<const std::string &, const std::vector<std::string> &, int, double,
                      const std::string &, int>(),
             py::arg("bag_path"), py::arg("topics"), py::arg("queue_size") = 100,
             py::arg("slop") = 0.05, py::arg("storage_id") = "auto", py::arg("max_read_ahead") = 20)
        .def("reset", &Ros2BagSyncReaderATS::reset)
        .def("is_eof", &Ros2BagSyncReaderATS::is_eof)
        .def("read_step",
             [](Ros2BagSyncReaderATS &self) -> py::object {
                 auto result = self.read_step();
                 if (!result) {
                     return py::none();
                 }

                 double ts = result->first;
                 py::dict output;

                 for (const auto &[topic, msg_ptr] : result->second) {
                     if (!msg_ptr) {
                         continue;
                     }

                     if (msg_ptr->encoding.empty() || msg_ptr->width == 0 || msg_ptr->height == 0) {
                         continue;
                     }

                     auto msg_copy = std::make_shared<sensor_msgs::msg::Image>(*msg_ptr);
                     output[py::str(topic)] = py::cast(msg_copy);
                 }

                 if (output.empty()) {
                     return py::none();
                 }

                 return py::make_tuple(ts, output);
             })
        .def(
            "read_all_messages_of_topic",
            [](Ros2BagSyncReaderATS &self, const std::string &topic, bool with_timestamps) {
                auto messages = self.read_all_messages_of_topic(topic, with_timestamps);
                py::list result;
                for (const auto &[ts, msg_ptr] : messages) {
                    if (msg_ptr && !msg_ptr->encoding.empty() && msg_ptr->width > 0 &&
                        msg_ptr->height > 0) {
                        auto msg_copy = std::make_shared<sensor_msgs::msg::Image>(*msg_ptr);
                        result.append(py::make_tuple(ts, py::cast(msg_copy)));
                    }
                }
                return result;
            },
            py::arg("topic"), py::arg("with_timestamps") = false)
        .def_readonly("topic_timestamps", &Ros2BagSyncReaderATS::topic_timestamps)
        .def_readonly("topic_counts", &Ros2BagSyncReaderATS::topic_counts);

    // Bind Ros2BagAsyncReaderATS
    py::class_<Ros2BagAsyncReaderATS, std::shared_ptr<Ros2BagAsyncReaderATS>>(
        m, "Ros2BagAsyncReaderATS")
        .def(py::init<const std::string &, const std::vector<std::string> &, int, double,
                      const std::string &, int, size_t>(),
             py::arg("bag_path"), py::arg("topics"), py::arg("queue_size") = 100,
             py::arg("slop") = 0.05, py::arg("storage_id") = "auto", py::arg("max_read_ahead") = 20,
             py::arg("max_queue_size") = 50)
        .def("reset", &Ros2BagAsyncReaderATS::reset)
        .def("is_eof", &Ros2BagAsyncReaderATS::is_eof)
        .def("read_step",
             [](Ros2BagAsyncReaderATS &self) -> py::object {
                 auto result = self.read_step();
                 if (!result) {
                     return py::none();
                 }

                 double ts = result->timestamp;
                 py::dict output;

                 for (const auto &[topic, img] : result->images) {
                     if (img.empty()) {
                         continue;
                     }

                     // Convert cv::Mat to numpy array with proper memory management
                     // We clone the image and create a numpy array that owns the data
                     cv::Mat img_copy = img.clone();
                     py::object numpy_array;

                     if (img_copy.type() == CV_8UC3) {
                         // BGR color image (H, W, 3) - contiguous data
                         auto arr = py::array_t<uint8_t>({img_copy.rows, img_copy.cols, 3});
                         std::memcpy(arr.mutable_data(), img_copy.data,
                                     img_copy.rows * img_copy.cols * 3);
                         numpy_array = arr;
                     } else if (img_copy.type() == CV_8UC1) {
                         // Grayscale image (H, W) - contiguous data
                         auto arr = py::array_t<uint8_t>({img_copy.rows, img_copy.cols});
                         std::memcpy(arr.mutable_data(), img_copy.data,
                                     img_copy.rows * img_copy.cols);
                         numpy_array = arr;
                     } else if (img_copy.type() == CV_32FC1) {
                         // Depth image (H, W) float32 - contiguous data
                         auto arr = py::array_t<float>({img_copy.rows, img_copy.cols});
                         std::memcpy(arr.mutable_data(), img_copy.data,
                                     img_copy.rows * img_copy.cols * sizeof(float));
                         numpy_array = arr;
                     } else {
                         // Fallback: convert to uint8
                         cv::Mat converted;
                         img_copy.convertTo(converted, CV_8UC1);
                         auto arr = py::array_t<uint8_t>({converted.rows, converted.cols});
                         std::memcpy(arr.mutable_data(), converted.data,
                                     converted.rows * converted.cols);
                         numpy_array = arr;
                     }

                     output[py::str(topic)] = numpy_array;
                 }

                 if (output.empty()) {
                     return py::none();
                 }

                 return py::make_tuple(ts, output);
             })
        .def(
            "read_all_messages_of_topic",
            [](Ros2BagAsyncReaderATS &self, const std::string &topic, bool with_timestamps) {
                auto messages = self.read_all_messages_of_topic(topic, with_timestamps);
                py::list result;
                for (const auto &[ts, msg_ptr] : messages) {
                    if (msg_ptr && !msg_ptr->encoding.empty() && msg_ptr->width > 0 &&
                        msg_ptr->height > 0) {
                        auto msg_copy = std::make_shared<sensor_msgs::msg::Image>(*msg_ptr);
                        result.append(py::make_tuple(ts, py::cast(msg_copy)));
                    }
                }
                return result;
            },
            py::arg("topic"), py::arg("with_timestamps") = false)
        .def_readonly("topic_timestamps", &Ros2BagAsyncReaderATS::topic_timestamps)
        .def_readonly("topic_counts", &Ros2BagAsyncReaderATS::topic_counts);
}
