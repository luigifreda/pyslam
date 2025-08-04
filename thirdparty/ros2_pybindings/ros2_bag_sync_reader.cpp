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

#include "ros2_bag_sync_reader.hpp"

#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rclcpp/serialization.hpp>
#include <rcpputils/filesystem_helper.hpp>

#include <chrono>
#include <iostream>

#if WITH_OPENCV_SHOW
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#endif


#define VERBOSE 0


#if WITH_OPENCV_SHOW
void visualize_depth_image(const std::string& topic_name, const sensor_msgs::msg::Image::ConstSharedPtr& msg, const std::string& window_name = "Depth Image") {
  if (msg->encoding != "32FC1") {
    std::cerr << "[visualize_depth_image] Unsupported encoding: " << msg->encoding << std::endl;
    return;
  }

  std::cout << "---- Depth Image Debug ----" << std::endl;
  std::cout << "Topic         : " << topic_name << std::endl;
  std::cout << "Encoding      : " << msg->encoding << std::endl;
  std::cout << "Endian        : " << (msg->is_bigendian ? "Big-endian" : "Little-endian") << std::endl;
  std::cout << "Width x Height: " << msg->width << " x " << msg->height << std::endl;
  std::cout << "Step          : " << msg->step << std::endl;
  std::cout << "Data size     : " << msg->data.size() << std::endl;  

  int width = msg->width;
  int height = msg->height;

  // Reinterpret the raw float buffer
  const float* depth_data = reinterpret_cast<const float*>(msg->data.data());
  cv::Mat depth_float(height, width, CV_32FC1, const_cast<float*>(depth_data));

  // Normalize for visualization
  cv::Mat depth_display;
  double min_val, max_val;
  cv::minMaxLoc(depth_float, &min_val, &max_val, nullptr, nullptr);
  depth_float.convertTo(depth_display, CV_8UC1, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

  // Show it
  cv::imshow(window_name, depth_display);
  cv::waitKey(1);
}
#endif


Ros2BagSyncReaderATS::Ros2BagSyncReaderATS(const std::string& bag_path,
                                           const std::vector<std::string>& topics,
                                           int queue_size,
                                           double slop,
                                           const std::string& storage_id)
  : bag_path_(bag_path), topics_(topics), queue_size_(queue_size), slop_(slop), storage_id_(storage_id) {
  load_bag();
  get_topic_timestamps_and_counts();
  setup_synchronizer();
}

void Ros2BagSyncReaderATS::load_bag() {
  reader_ = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
#ifdef WITH_JAZZY
  rosbag2_storage::StorageOptions storage_options;
#else
  rosbag2_cpp::StorageOptions storage_options;
#endif
  storage_options.uri = bag_path_;

  // Use provided storage_id or auto-detect
  if (storage_id_ == "auto") {
    std::string extension = rcpputils::fs::path(bag_path_).extension().string();
    if (extension == ".mcap") {
      storage_options.storage_id = "mcap";
    } else {
      storage_options.storage_id = "sqlite3";
    }
  } else {
    storage_options.storage_id = storage_id_;
  }

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  reader_->open(storage_options, converter_options);
  reader_->set_filter({topics_});
}

void Ros2BagSyncReaderATS::reset() {
  synced_msgs_.clear();
  load_bag();
}

void Ros2BagSyncReaderATS::get_topic_timestamps_and_counts() {
  topic_timestamps.clear();
  topic_counts.clear();
  rosbag2_cpp::readers::SequentialReader temp_reader;
#ifdef WITH_JAZZY
  rosbag2_storage::StorageOptions storage_options;
#else
  rosbag2_cpp::StorageOptions storage_options;
#endif
  storage_options.uri = bag_path_;

  // Use provided storage_id or auto-detect
  if (storage_id_ == "auto") {
    std::string extension = rcpputils::fs::path(bag_path_).extension().string();
    if (extension == ".mcap") {
      storage_options.storage_id = "mcap";
    } else {
      storage_options.storage_id = "sqlite3";
    }
  } else {
    storage_options.storage_id = storage_id_;
  }

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  temp_reader.open(storage_options, converter_options);
  temp_reader.set_filter({topics_});

  rclcpp::Serialization<sensor_msgs::msg::Image> serializer;

  while (temp_reader.has_next()) {
    auto bag_msg = temp_reader.read_next();
    if (bag_msg == nullptr) continue;

    auto topic = bag_msg->topic_name;
    sensor_msgs::msg::Image msg;
    rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
    serializer.deserialize_message(&serialized_msg, &msg);
    double stamp = rclcpp::Time(msg.header.stamp).seconds();
    topic_timestamps[topic].push_back(stamp);
    topic_counts[topic]++;
  }
}

void Ros2BagSyncReaderATS::messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr& msg1) {
  synced_msgs_.emplace_back(std::make_tuple(msg1, nullptr, nullptr));
}

void Ros2BagSyncReaderATS::messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr& msg1,
                                           const sensor_msgs::msg::Image::ConstSharedPtr& msg2) {
  synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, nullptr));
}

void Ros2BagSyncReaderATS::messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr& msg1,
                                           const sensor_msgs::msg::Image::ConstSharedPtr& msg2,
                                           const sensor_msgs::msg::Image::ConstSharedPtr& msg3) {
  synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, msg3));
}

void Ros2BagSyncReaderATS::setup_synchronizer() {
  for (const auto& topic : topics_) {
    filters_.emplace(topic, std::make_shared<ExposedSimpleFilter<sensor_msgs::msg::Image>>());
  }

  if (topics_.size() == 1) {
    // No synchronizer needed
  } else if (topics_.size() == 2) {
    sync2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
      SyncPolicy2(queue_size_),
      *filters_[topics_[0]],
      *filters_[topics_[1]]);
    
    sync2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

    sync2_->registerCallback(
      std::bind(static_cast<void(Ros2BagSyncReaderATS::*)(const sensor_msgs::msg::Image::ConstSharedPtr&,
                                                          const sensor_msgs::msg::Image::ConstSharedPtr&)>(
                  &Ros2BagSyncReaderATS::messageCallback2),
                this,
                std::placeholders::_1,
                std::placeholders::_2));
  } else if (topics_.size() == 3) {
    sync3_ = std::make_shared<message_filters::Synchronizer<SyncPolicy3>>(
      SyncPolicy3(queue_size_),
      *filters_[topics_[0]],
      *filters_[topics_[1]],
      *filters_[topics_[2]]);
    
    sync3_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

    sync3_->registerCallback(
      std::bind(static_cast<void(Ros2BagSyncReaderATS::*)(const sensor_msgs::msg::Image::ConstSharedPtr&,
                                                          const sensor_msgs::msg::Image::ConstSharedPtr&,
                                                          const sensor_msgs::msg::Image::ConstSharedPtr&)>(
                  &Ros2BagSyncReaderATS::messageCallback3),
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3));
  } else {
    throw std::runtime_error("Only 1, 2, or 3 topics are supported");
  }
}

std::optional<std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>>
Ros2BagSyncReaderATS::read_step() {
  if (!reader_->has_next()) {
    return std::nullopt;
  }

  auto bag_msg = reader_->read_next();
  if (!bag_msg) return std::nullopt;

  auto topic = bag_msg->topic_name;
  rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
  serializer.deserialize_message(&serialized_msg, msg.get());

  if (topics_.size() == 1) {
    messageCallback1(msg);
  } else {
    signal_message(topic, msg);
  }

#if VERBOSE
  std::cout << "Feeding message from topic: " << topic << " at time " 
          << rclcpp::Time(msg->header.stamp).seconds() << std::endl;
#endif 

  if (!synced_msgs_.empty()) {
    auto synced = synced_msgs_.front();
    synced_msgs_.pop_front();
  
    std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr> result;
    sensor_msgs::msg::Image::ConstSharedPtr msg1 = std::get<0>(synced);
    sensor_msgs::msg::Image::ConstSharedPtr msg2 = std::get<1>(synced);
    sensor_msgs::msg::Image::ConstSharedPtr msg3 = std::get<2>(synced);
  
    double stamp = 0.0;
    if (msg1) {
      result[topics_[0]] = msg1;
      stamp = rclcpp::Time(msg1->header.stamp).seconds();
#if WITH_OPENCV_SHOW
      if (msg1->encoding == "32FC1")  visualize_depth_image(topics_[0], msg1, "Depth1");
#endif
    }
    if (topics_.size() > 1 && msg2) {
      result[topics_[1]] = msg2;
      if (stamp == 0.0) stamp = rclcpp::Time(msg2->header.stamp).seconds();
#if WITH_OPENCV_SHOW
      if (msg2->encoding == "32FC1")  visualize_depth_image(topics_[1], msg2, "Depth2");
#endif      
    }
    if (topics_.size() > 2 && msg3) {
      result[topics_[2]] = msg3;
      if (stamp == 0.0) stamp = rclcpp::Time(msg3->header.stamp).seconds();
#if WITH_OPENCV_SHOW
      if (msg3->encoding == "32FC1")  visualize_depth_image(topics_[2], msg3, "Depth2");
#endif       
    }
  
    // Return only if we have all the expected messages
    if (result.size() == topics_.size()) {
      return std::make_pair(stamp, result);
    } else {
      // Skip this sync step if incomplete (or optionally return it with warnings)
      return std::nullopt;
    }
  }

  return std::nullopt;
}

void Ros2BagSyncReaderATS::signal_message(const std::string& topic_name,
                                          const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
#if VERBOSE
  std::cout << "Signal message on topic: " << topic_name << std::endl;                                            
#endif

  if (filters_.find(topic_name) != filters_.end()) {
    filters_[topic_name]->feed(msg);
  } else {
    std::cerr << "No filter found for topic: " << topic_name << std::endl;
  }
}

std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
Ros2BagSyncReaderATS::read_all_messages_of_topic(const std::string& topic, bool with_timestamps) {
  std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>> messages;

  rosbag2_cpp::readers::SequentialReader temp_reader;
#ifdef WITH_JAZZY
  rosbag2_storage::StorageOptions storage_options;
#else
  rosbag2_cpp::StorageOptions storage_options;
#endif
  storage_options.uri = bag_path_;

  // Use provided storage_id or auto-detect
  if (storage_id_ == "auto") {
    std::string extension = rcpputils::fs::path(bag_path_).extension().string();
    if (extension == ".mcap") {
      storage_options.storage_id = "mcap";
    } else {
      storage_options.storage_id = "sqlite3";
    }
  } else {
    storage_options.storage_id = storage_id_;
  }

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  temp_reader.open(storage_options, converter_options);
  temp_reader.set_filter({{topic}});

  rclcpp::Serialization<sensor_msgs::msg::Image> serializer;

  while (temp_reader.has_next()) {
    auto bag_msg = temp_reader.read_next();
    if (!bag_msg) continue;

    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    try {
      rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
      serializer.deserialize_message(&serialized_msg, msg.get());
    } catch (const std::exception& e) {
      std::cerr << "[Deserialization Error] Topic: " << topic 
                << ", Error: " << e.what() << std::endl;
      continue;
    }
    double stamp = rclcpp::Time(msg->header.stamp).seconds();
    messages.emplace_back(stamp, msg);
  }

  return messages;
}
