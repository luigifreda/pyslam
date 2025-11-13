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

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
// Include serialization header - required for unique_ptr to work with incomplete types
// The header should be available in ROS2 Foxy and later
#include <rclcpp/serialization.hpp>
#ifdef WITH_JAZZY
#include <rosbag2_storage/storage_options.hpp>
#else
// storage_options.hpp moved from rosbag2_cpp -> rosbag2_storage
#if __has_include(<rosbag2_cpp/storage_options.hpp>)
#include <rosbag2_cpp/storage_options.hpp>
namespace rb2_storage_ns = rosbag2_cpp;
#else
#include <rosbag2_storage/storage_options.hpp>
namespace rb2_storage_ns = rosbag2_storage;
#endif
#endif
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>

#include <sensor_msgs/msg/image.hpp>
#if __has_include(<message_filters/sync_policies/approximate_time.h>)
#include <message_filters/sync_policies/approximate_time.h>
#else
#include <message_filters/sync_policies/approximate_time.hpp>
#endif
#if __has_include(<message_filters/synchronizer.h>)
#include <message_filters/synchronizer.h>
#else
#include <message_filters/synchronizer.hpp>
#endif
#if __has_include(<message_filters/simple_filter.h>)
#include <message_filters/simple_filter.h>
#else
#include <message_filters/simple_filter.hpp>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

class Ros2BagSyncReaderATS {

    template <typename MsgType>
    class ExposedSimpleFilter : public message_filters::SimpleFilter<MsgType> {
      public:
        void feed(const std::shared_ptr<const MsgType> &msg) { this->signalMessage(msg); }
    };

  public:
    Ros2BagSyncReaderATS(const std::string &bag_path, const std::vector<std::string> &topics,
                         int queue_size = 100, double slop = 0.05,
                         const std::string &storage_id = "auto", int max_read_ahead = 20);

    ~Ros2BagSyncReaderATS();

    void reset();

    bool is_eof() const { return !reader_->has_next(); }

    std::optional<
        std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>>
    read_step();

    std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
    read_all_messages_of_topic(const std::string &topic, bool with_timestamps = false);

    std::unordered_map<std::string, std::vector<double>> topic_timestamps;
    std::unordered_map<std::string, size_t> topic_counts;

  private:
    std::unique_ptr<rosbag2_cpp::readers::SequentialReader> reader_;
    std::string bag_path_;
    std::vector<std::string> topics_;
    int queue_size_;
    double slop_;
    int max_read_ahead_;

    std::unordered_map<std::string, std::shared_ptr<ExposedSimpleFilter<sensor_msgs::msg::Image>>>
        filters_;

    using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                                        sensor_msgs::msg::Image>;
    using SyncPolicy3 = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy3>> sync3_;

    std::deque<
        std::tuple<sensor_msgs::msg::Image::ConstSharedPtr, sensor_msgs::msg::Image::ConstSharedPtr,
                   sensor_msgs::msg::Image::ConstSharedPtr>>
        synced_msgs_;

    std::unique_ptr<rclcpp::Serialization<sensor_msgs::msg::Image>> deserializer_;

    std::string storage_id_;

  private:
    using SyncResult =
        std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>;

    void load_bag();
    void get_topic_timestamps_and_counts();
    void setup_synchronizer();
    void signal_message(const std::string &topic_name,
                        const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    void messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr &msg1);

    void messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg2);

    void messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg2,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg3);

    std::optional<SyncResult> pop_synced_message();
    sensor_msgs::msg::Image::SharedPtr
    deserialize_bag_message(const std::shared_ptr<rosbag2_storage::SerializedBagMessage> &bag_msg);
};

class Ros2BagAsyncReaderATS {

    template <typename MsgType>
    class ExposedSimpleFilter : public message_filters::SimpleFilter<MsgType> {
      public:
        void feed(const std::shared_ptr<const MsgType> &msg) { this->signalMessage(msg); }
    };

  public:
    // Decoded result structure for decoded images
    struct DecodedResult {
        double timestamp;
        std::unordered_map<std::string, cv::Mat> images; // topic -> decoded image
    };

    Ros2BagAsyncReaderATS(const std::string &bag_path, const std::vector<std::string> &topics,
                          int queue_size = 100, double slop = 0.05,
                          const std::string &storage_id = "auto", int max_read_ahead = 20,
                          size_t max_queue_size = 50);

    ~Ros2BagAsyncReaderATS();

    void reset();

    bool is_eof() const;

    // Returns decoded images (cv::Mat) instead of ROS messages
    std::optional<DecodedResult> read_step();

    std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
    read_all_messages_of_topic(const std::string &topic, bool with_timestamps = false);

    std::unordered_map<std::string, std::vector<double>> topic_timestamps;
    std::unordered_map<std::string, size_t> topic_counts;

  private:
    std::unique_ptr<rosbag2_cpp::readers::SequentialReader> reader_;
    std::string bag_path_;
    std::vector<std::string> topics_;
    int queue_size_;
    double slop_;
    int max_read_ahead_;
    size_t max_queue_size_;

    std::unordered_map<std::string, std::shared_ptr<ExposedSimpleFilter<sensor_msgs::msg::Image>>>
        filters_;

    using SyncPolicy2 = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                                        sensor_msgs::msg::Image>;
    using SyncPolicy3 = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    std::shared_ptr<message_filters::Synchronizer<SyncPolicy2>> sync2_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy3>> sync3_;

    std::deque<
        std::tuple<sensor_msgs::msg::Image::ConstSharedPtr, sensor_msgs::msg::Image::ConstSharedPtr,
                   sensor_msgs::msg::Image::ConstSharedPtr>>
        synced_msgs_;

    std::unique_ptr<rclcpp::Serialization<sensor_msgs::msg::Image>> deserializer_;

    std::string storage_id_;

    // Threading and synchronization
    std::thread reader_thread_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_not_full_cv_;  // Producer waits when queue is full
    std::condition_variable queue_not_empty_cv_; // Consumer waits when queue is empty
    std::atomic<bool> stop_thread_{false};
    std::atomic<bool> eof_reached_{false};

    // Queue for decoded images (protected by mutex)
    // Stores decoded cv::Mat images instead of ROS messages for better performance
    std::deque<DecodedResult> message_queue_;

  private:
    void load_bag();
    void get_topic_timestamps_and_counts();
    void setup_synchronizer();
    void signal_message(const std::string &topic_name,
                        const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    void messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr &msg1);

    void messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg2);

    void messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg2,
                          const sensor_msgs::msg::Image::ConstSharedPtr &msg3);

    // Internal type for synced ROS messages (before decoding)
    using SyncResult =
        std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>;

    std::optional<SyncResult> pop_synced_message();
    sensor_msgs::msg::Image::SharedPtr
    deserialize_bag_message(const std::shared_ptr<rosbag2_storage::SerializedBagMessage> &bag_msg);

    // Decode ROS image message to cv::Mat
    std::optional<cv::Mat> decode_image(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                                        const std::string &desired_encoding = "bgr8");

    // Decode depth image message to cv::Mat (float32)
    std::optional<cv::Mat> decode_depth_image(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

    // Background thread function (producer)
    void reader_thread_function();
};
