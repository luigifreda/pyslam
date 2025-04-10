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

#include <string>
#include <deque>
#include <unordered_map>
#include <memory>
#include <optional>

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_cpp/storage_options.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/simple_filter.h>

class Ros2BagSyncReaderATS {

  template<typename MsgType>
  class ExposedSimpleFilter : public message_filters::SimpleFilter<MsgType> {
    public:
      void feed(const std::shared_ptr<const MsgType>& msg) {
        this->signalMessage(msg);  // now accessible
      }
  };

public:
  Ros2BagSyncReaderATS(const std::string& bag_path,
                       const std::vector<std::string>& topics,
                       int queue_size = 100,
                       double slop = 0.05);

  void reset();

  bool is_eof() const { return !reader_->has_next(); }

public: 
  std::optional<std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>> read_step();

  std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
  read_all_messages_of_topic(const std::string& topic, bool with_timestamps = false);

  std::unordered_map<std::string, std::vector<double>> topic_timestamps;
  std::unordered_map<std::string, size_t> topic_counts;

private:
  std::unique_ptr<rosbag2_cpp::readers::SequentialReader> reader_;
  std::string bag_path_;
  std::vector<std::string> topics_;
  int queue_size_;
  double slop_;

  std::unordered_map<std::string, std::shared_ptr<ExposedSimpleFilter<sensor_msgs::msg::Image>> > filters_;

  using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, 
                                                                      sensor_msgs::msg::Image>;
  using SyncPolicy3 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, 
                                                                      sensor_msgs::msg::Image, 
                                                                      sensor_msgs::msg::Image>;
  
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy3>> sync3_;

  std::deque<std::tuple<sensor_msgs::msg::Image::ConstSharedPtr,
                        sensor_msgs::msg::Image::ConstSharedPtr,
                        sensor_msgs::msg::Image::ConstSharedPtr>> synced_msgs_;

private: 
  void load_bag();
  void get_topic_timestamps_and_counts();
  void setup_synchronizer();
  void signal_message(const std::string& topic_name, const sensor_msgs::msg::Image::ConstSharedPtr& msg);

  void messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr& msg1);
  
  void messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr& msg1,
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg2);
  
  void messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr& msg1,
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg2,
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg3);
};