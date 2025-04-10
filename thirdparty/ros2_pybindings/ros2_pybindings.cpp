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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>

#include <memory>

#include "ros2_bag_sync_reader.hpp"


namespace py = pybind11;

PYBIND11_MODULE(ros2_pybindings, m) {

  // Bind builtin_interfaces::msg::Time
  py::class_<builtin_interfaces::msg::Time>(m, "Time")
    .def(py::init<>())
    .def_readwrite("sec", &builtin_interfaces::msg::Time::sec)
    .def_readwrite("nanosec", &builtin_interfaces::msg::Time::nanosec);

  
  // Bind std_msgs::msg::Header
  py::class_<std_msgs::msg::Header>(m, "Header")
    .def(py::init<>())
    .def_readwrite("stamp", &std_msgs::msg::Header::stamp)
    .def_readwrite("frame_id", &std_msgs::msg::Header::frame_id);

  
  // Bind sensor_msgs::msg::Image
  py::class_<sensor_msgs::msg::Image>(m, "Image")
    .def(py::init<>())
    .def_readwrite("header", &sensor_msgs::msg::Image::header)
    .def_readwrite("height", &sensor_msgs::msg::Image::height)
    .def_readwrite("width", &sensor_msgs::msg::Image::width)
    .def_readwrite("encoding", &sensor_msgs::msg::Image::encoding)
    .def_readwrite("is_bigendian", &sensor_msgs::msg::Image::is_bigendian)
    .def_readwrite("step", &sensor_msgs::msg::Image::step)
    .def_readwrite("data", &sensor_msgs::msg::Image::data);


  // Bind Ros2BagSyncReaderATS  
  py::class_<Ros2BagSyncReaderATS>(m, "Ros2BagSyncReaderATS")
    .def(py::init<const std::string&, const std::vector<std::string>&, int, double>(),
         py::arg("bag_path"), py::arg("topics"), py::arg("queue_size") = 100, py::arg("slop") = 0.05)
    .def("reset", &Ros2BagSyncReaderATS::reset)
    .def("is_eof", &Ros2BagSyncReaderATS::is_eof)
    .def("read_step", [](Ros2BagSyncReaderATS& self) -> py::object {
        auto result = self.read_step();
        if (!result) {
            return py::none();
        }
    
        double ts = result->first;
        py::dict output;
    
        for (const auto& [topic, msg_ptr] : result->second) {
            if (!msg_ptr) {
                std::cerr << "[WARN] Null message pointer for topic: " << topic << std::endl;
                continue;
            }
    
            // Create a Python capsule wrapping the message pointer
            py::object msg_py = py::cast(*msg_ptr);
            output[py::str(topic)] = msg_py;
        }
    
        if (output.empty()) {
            std::cerr << "[WARN] All message pointers were null, skipping this step.\n";
            return py::none();
        }
    
        return py::make_tuple(ts, output);
    })
    .def("read_all_messages_of_topic", [](Ros2BagSyncReaderATS& self, const std::string& topic, bool with_timestamps) {
      std::vector<std::pair<double, sensor_msgs::msg::Image>> result;
      for (const auto& [ts, msg_ptr] : self.read_all_messages_of_topic(topic, with_timestamps)) {
        result.emplace_back(ts, *msg_ptr);
      }
      return result;
    }, py::arg("topic"), py::arg("with_timestamps") = false)
    .def_readonly("topic_timestamps", &Ros2BagSyncReaderATS::topic_timestamps)
    .def_readonly("topic_counts", &Ros2BagSyncReaderATS::topic_counts);
}
