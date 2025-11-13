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

#include <rclcpp/serialization.hpp>
#include <rcpputils/filesystem_helper.hpp>
#include <rmw/serialized_message.h>
#include <rosbag2_cpp/converter_interfaces/serialization_format_converter.hpp>
#include <rosbag2_cpp/typesupport_helpers.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs_compat = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs_compat = std::experimental::filesystem;
#elif __has_include(<rcpputils/filesystem_helper.hpp>)
// Fallback for very old ROS 2 where std::filesystem wasn't viable
#include <rcpputils/filesystem_helper.hpp>
namespace fs_compat = rcpputils::fs;
#else
#error "No filesystem implementation found."
#endif

#define VERBOSE 0

void visualize_depth_image(const std::string &topic_name,
                           const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                           const std::string &window_name = "Depth Image") {
    if (msg->encoding != "32FC1") {
        std::cerr << "[visualize_depth_image] Unsupported encoding: " << msg->encoding << std::endl;
        return;
    }

    std::cout << "---- Depth Image Debug ----" << std::endl;
    std::cout << "Topic         : " << topic_name << std::endl;
    std::cout << "Encoding      : " << msg->encoding << std::endl;
    std::cout << "Endian        : " << (msg->is_bigendian ? "Big-endian" : "Little-endian")
              << std::endl;
    std::cout << "Width x Height: " << msg->width << " x " << msg->height << std::endl;
    std::cout << "Step          : " << msg->step << std::endl;
    std::cout << "Data size     : " << msg->data.size() << std::endl;

    int width = msg->width;
    int height = msg->height;

    // Reinterpret the raw float buffer
    const float *depth_data = reinterpret_cast<const float *>(msg->data.data());
    cv::Mat depth_float(height, width, CV_32FC1, const_cast<float *>(depth_data));

    // Normalize for visualization
    cv::Mat depth_display;
    double min_val, max_val;
    cv::minMaxLoc(depth_float, &min_val, &max_val, nullptr, nullptr);
    depth_float.convertTo(depth_display, CV_8UC1, 255.0 / (max_val - min_val),
                          -min_val * 255.0 / (max_val - min_val));

    // Show it
    cv::imshow(window_name, depth_display);
    cv::waitKey(1);
}

// Helper function to validate message content
static bool is_valid_image_message(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    return msg && !msg->encoding.empty() && msg->width > 0 && msg->height > 0;
}

Ros2BagSyncReaderATS::Ros2BagSyncReaderATS(const std::string &bag_path,
                                           const std::vector<std::string> &topics, int queue_size,
                                           double slop, const std::string &storage_id,
                                           int max_read_ahead)
    : bag_path_(bag_path), topics_(topics), queue_size_(queue_size), slop_(slop),
      storage_id_(storage_id), max_read_ahead_(max_read_ahead),
      deserializer_(std::make_unique<rclcpp::Serialization<sensor_msgs::msg::Image>>()) {
    load_bag();
    get_topic_timestamps_and_counts();
    setup_synchronizer();
}

Ros2BagSyncReaderATS::~Ros2BagSyncReaderATS() = default;

void Ros2BagSyncReaderATS::load_bag() {
    reader_ = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
#ifdef WITH_JAZZY
    rosbag2_storage::StorageOptions storage_options;
#else
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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

    while (temp_reader.has_next()) {
        auto bag_msg = temp_reader.read_next();
        if (!bag_msg) {
            continue;
        }
        auto topic = bag_msg->topic_name;
        double stamp = static_cast<double>(bag_msg->time_stamp) * 1e-9;
        topic_timestamps[topic].push_back(stamp);
        topic_counts[topic]++;
    }
}

void Ros2BagSyncReaderATS::messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr &msg1) {
    if (!is_valid_image_message(msg1)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, nullptr, nullptr));
}

void Ros2BagSyncReaderATS::messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                                            const sensor_msgs::msg::Image::ConstSharedPtr &msg2) {
    if (!is_valid_image_message(msg1) || !is_valid_image_message(msg2)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, nullptr));
}

void Ros2BagSyncReaderATS::messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                                            const sensor_msgs::msg::Image::ConstSharedPtr &msg2,
                                            const sensor_msgs::msg::Image::ConstSharedPtr &msg3) {
    if (!is_valid_image_message(msg1) || !is_valid_image_message(msg2) ||
        !is_valid_image_message(msg3)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, msg3));
}

void Ros2BagSyncReaderATS::setup_synchronizer() {
    for (const auto &topic : topics_) {
        filters_.emplace(topic, std::make_shared<ExposedSimpleFilter<sensor_msgs::msg::Image>>());
    }

    if (topics_.size() == 1) {
        return;
    } else if (topics_.size() == 2) {
        sync2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
            SyncPolicy2(queue_size_), *filters_[topics_[0]], *filters_[topics_[1]]);

        sync2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

        sync2_->registerCallback(std::bind(&Ros2BagSyncReaderATS::messageCallback2, this,
                                           std::placeholders::_1, std::placeholders::_2));
    } else if (topics_.size() == 3) {
        sync3_ = std::make_shared<message_filters::Synchronizer<SyncPolicy3>>(
            SyncPolicy3(queue_size_), *filters_[topics_[0]], *filters_[topics_[1]],
            *filters_[topics_[2]]);

        sync3_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

        sync3_->registerCallback(std::bind(&Ros2BagSyncReaderATS::messageCallback3, this,
                                           std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3));
    } else {
        throw std::runtime_error("Only 1, 2, or 3 topics are supported");
    }
}

void Ros2BagSyncReaderATS::signal_message(const std::string &topic_name,
                                          const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
#if VERBOSE
    std::cout << "Signal message on topic: " << topic_name << std::endl;
#endif

    auto it = filters_.find(topic_name);
    if (it != filters_.end()) {
        it->second->feed(msg);
    } else {
        std::cerr << "No filter found for topic: " << topic_name << std::endl;
    }
}

sensor_msgs::msg::Image::SharedPtr Ros2BagSyncReaderATS::deserialize_bag_message(
    const std::shared_ptr<rosbag2_storage::SerializedBagMessage> &bag_msg) {
    if (!bag_msg || !bag_msg->serialized_data) {
        return nullptr;
    }

    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    try {
        rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
        deserializer_->deserialize_message(&serialized_msg, msg.get());
    } catch (const std::exception &e) {
        std::cerr << "[Deserialization Error] Topic: " << bag_msg->topic_name
                  << ", Error: " << e.what() << std::endl;
        return nullptr;
    }
    return msg;
}

std::optional<Ros2BagSyncReaderATS::SyncResult> Ros2BagSyncReaderATS::pop_synced_message() {
    if (synced_msgs_.empty()) {
        return std::nullopt;
    }

    auto synced = synced_msgs_.front();
    synced_msgs_.pop_front();

    std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr> result;
    result.reserve(topics_.size());

    double stamp = 0.0;
    bool stamp_initialized = false;

    auto add_if_valid = [&](size_t index, const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        if (!msg || !is_valid_image_message(msg) || index >= topics_.size()) {
            return;
        }
        result.emplace(topics_[index], msg);
        if (!stamp_initialized) {
            stamp = rclcpp::Time(msg->header.stamp).seconds();
            stamp_initialized = true;
        }
    };

    if (!topics_.empty()) {
        add_if_valid(0, std::get<0>(synced));
        if (topics_.size() > 1) {
            add_if_valid(1, std::get<1>(synced));
        }
        if (topics_.size() > 2) {
            add_if_valid(2, std::get<2>(synced));
        }
    }

    if (result.size() == topics_.size() && stamp_initialized) {
        return SyncResult{stamp, std::move(result)};
    }

    return std::nullopt;
}

std::optional<
    std::pair<double, std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr>>>
Ros2BagSyncReaderATS::read_step() {
    if (auto ready = pop_synced_message()) {
        return ready;
    }

    int read_ahead_count = 0;
    while (reader_->has_next() && read_ahead_count < max_read_ahead_) {
        auto bag_msg = reader_->read_next();
        if (!bag_msg) {
            break;
        }

        auto msg = deserialize_bag_message(bag_msg);
        if (!msg) {
            continue;
        }

        if (topics_.size() == 1) {
            messageCallback1(msg);
        } else {
            signal_message(bag_msg->topic_name, msg);
        }

        ++read_ahead_count;

        if (auto ready = pop_synced_message()) {
            return ready;
        }
    }

    return std::nullopt;
}

std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
Ros2BagSyncReaderATS::read_all_messages_of_topic(const std::string &topic, bool with_timestamps) {
    (void)with_timestamps;
    std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>> messages;

    rosbag2_cpp::readers::SequentialReader temp_reader;
#ifdef WITH_JAZZY
    rosbag2_storage::StorageOptions storage_options;
#else
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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

    rclcpp::Serialization<sensor_msgs::msg::Image> temp_serializer;

    while (temp_reader.has_next()) {
        auto bag_msg = temp_reader.read_next();
        if (!bag_msg || !bag_msg->serialized_data) {
            continue;
        }

        auto msg = std::make_shared<sensor_msgs::msg::Image>();
        try {
            rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
            temp_serializer.deserialize_message(&serialized_msg, msg.get());
        } catch (const std::exception &e) {
            std::cerr << "[Deserialization Error] Topic: " << topic << ", Error: " << e.what()
                      << std::endl;
            continue;
        }

        double stamp = rclcpp::Time(msg->header.stamp).seconds();
        messages.emplace_back(stamp, msg);
    }

    return messages;
}

// ============================================================================
// Ros2BagAsyncReaderATS Implementation
// ============================================================================

Ros2BagAsyncReaderATS::Ros2BagAsyncReaderATS(const std::string &bag_path,
                                             const std::vector<std::string> &topics, int queue_size,
                                             double slop, const std::string &storage_id,
                                             int max_read_ahead, size_t max_queue_size)
    : bag_path_(bag_path), topics_(topics), queue_size_(queue_size), slop_(slop),
      storage_id_(storage_id), max_read_ahead_(max_read_ahead), max_queue_size_(max_queue_size),
      deserializer_(std::make_unique<rclcpp::Serialization<sensor_msgs::msg::Image>>()),
      stop_thread_(false), eof_reached_(false) {
    load_bag();
    get_topic_timestamps_and_counts();
    setup_synchronizer();

    // Start the background reader thread
    reader_thread_ = std::thread(&Ros2BagAsyncReaderATS::reader_thread_function, this);
}

Ros2BagAsyncReaderATS::~Ros2BagAsyncReaderATS() {
    // Signal thread to stop
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_thread_ = true;
    }
    queue_not_full_cv_.notify_all(); // Wake up thread if it's waiting

    // Wait for thread to finish
    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }
}

void Ros2BagAsyncReaderATS::reset() {
    // Stop current thread
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_thread_ = true;
    }
    queue_not_full_cv_.notify_all();

    if (reader_thread_.joinable()) {
        reader_thread_.join();
    }

    // Clear queue
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.clear();
        synced_msgs_.clear();
        stop_thread_ = false;
        eof_reached_ = false;
    }

    // Reload bag and restart thread
    load_bag();
    reader_thread_ = std::thread(&Ros2BagAsyncReaderATS::reader_thread_function, this);
}

bool Ros2BagAsyncReaderATS::is_eof() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return eof_reached_ && message_queue_.empty();
}

std::optional<Ros2BagAsyncReaderATS::DecodedResult> Ros2BagAsyncReaderATS::read_step() {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Wait until queue is not empty or EOF is reached
    queue_not_empty_cv_.wait(
        lock, [this] { return !message_queue_.empty() || (eof_reached_ && stop_thread_); });

    if (message_queue_.empty()) {
        // EOF reached and queue is empty
        return std::nullopt;
    }

    // Pop from queue
    DecodedResult result = std::move(message_queue_.front());
    message_queue_.pop_front();

    // Notify producer that queue is not full
    queue_not_full_cv_.notify_one();

    return result;
}

void Ros2BagAsyncReaderATS::reader_thread_function() {
    while (true) {
        // Check if we should stop
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_thread_) {
                break;
            }

            // Wait if queue is full
            queue_not_full_cv_.wait(
                lock, [this] { return message_queue_.size() < max_queue_size_ || stop_thread_; });

            if (stop_thread_) {
                break;
            }
        }

        // Try to read a synced message
        std::optional<SyncResult> sync_result = std::nullopt;
        int read_ahead_count = 0;

        // Read messages until we get a synced one or hit limits
        while (!sync_result && reader_->has_next() && read_ahead_count < max_read_ahead_) {
            auto bag_msg = reader_->read_next();
            if (!bag_msg) {
                break;
            }

            auto msg = deserialize_bag_message(bag_msg);
            if (!msg) {
                continue;
            }

            if (topics_.size() == 1) {
                messageCallback1(msg);
            } else {
                signal_message(bag_msg->topic_name, msg);
            }

            ++read_ahead_count;

            // Try to pop a synced message
            sync_result = pop_synced_message();
        }

        // Check EOF
        bool is_eof = !reader_->has_next();

        if (sync_result) {
            // Decode images from ROS messages to cv::Mat
            DecodedResult decoded_result;
            decoded_result.timestamp = sync_result->first;

            for (const auto &[topic, msg_ptr] : sync_result->second) {
                if (!msg_ptr) {
                    continue;
                }

                // Check if this is a depth image (common depth encodings)
                bool is_depth = (msg_ptr->encoding == "32FC1" || msg_ptr->encoding == "16UC1" ||
                                 msg_ptr->encoding == "16SC1" || msg_ptr->encoding == "8UC1");

                std::optional<cv::Mat> decoded_img;
                if (is_depth) {
                    decoded_img = decode_depth_image(msg_ptr);
                } else {
                    decoded_img = decode_image(msg_ptr, "bgr8");
                }

                if (decoded_img) {
                    decoded_result.images[topic] = *decoded_img;
                }
            }

            // Push decoded result to queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                message_queue_.push_back(std::move(decoded_result));
            }
            queue_not_empty_cv_.notify_one();
        } else if (is_eof) {
            // No more messages, mark EOF
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                eof_reached_ = true;
            }
            queue_not_empty_cv_.notify_one(); // Wake up any waiting consumers
            break;
        }
    }
}

void Ros2BagAsyncReaderATS::load_bag() {
    reader_ = std::make_unique<rosbag2_cpp::readers::SequentialReader>();
#ifdef WITH_JAZZY
    rosbag2_storage::StorageOptions storage_options;
#else
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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

void Ros2BagAsyncReaderATS::get_topic_timestamps_and_counts() {
    topic_timestamps.clear();
    topic_counts.clear();
    rosbag2_cpp::readers::SequentialReader temp_reader;
#ifdef WITH_JAZZY
    rosbag2_storage::StorageOptions storage_options;
#else
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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

    while (temp_reader.has_next()) {
        auto bag_msg = temp_reader.read_next();
        if (!bag_msg) {
            continue;
        }
        auto topic = bag_msg->topic_name;
        double stamp = static_cast<double>(bag_msg->time_stamp) * 1e-9;
        topic_timestamps[topic].push_back(stamp);
        topic_counts[topic]++;
    }
}

void Ros2BagAsyncReaderATS::messageCallback1(const sensor_msgs::msg::Image::ConstSharedPtr &msg1) {
    if (!is_valid_image_message(msg1)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, nullptr, nullptr));
}

void Ros2BagAsyncReaderATS::messageCallback2(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                                             const sensor_msgs::msg::Image::ConstSharedPtr &msg2) {
    if (!is_valid_image_message(msg1) || !is_valid_image_message(msg2)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, nullptr));
}

void Ros2BagAsyncReaderATS::messageCallback3(const sensor_msgs::msg::Image::ConstSharedPtr &msg1,
                                             const sensor_msgs::msg::Image::ConstSharedPtr &msg2,
                                             const sensor_msgs::msg::Image::ConstSharedPtr &msg3) {
    if (!is_valid_image_message(msg1) || !is_valid_image_message(msg2) ||
        !is_valid_image_message(msg3)) {
        return;
    }
    synced_msgs_.emplace_back(std::make_tuple(msg1, msg2, msg3));
}

void Ros2BagAsyncReaderATS::setup_synchronizer() {
    for (const auto &topic : topics_) {
        filters_.emplace(topic, std::make_shared<ExposedSimpleFilter<sensor_msgs::msg::Image>>());
    }

    if (topics_.size() == 1) {
        return;
    } else if (topics_.size() == 2) {
        sync2_ = std::make_shared<message_filters::Synchronizer<SyncPolicy2>>(
            SyncPolicy2(queue_size_), *filters_[topics_[0]], *filters_[topics_[1]]);

        sync2_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

        sync2_->registerCallback(std::bind(&Ros2BagAsyncReaderATS::messageCallback2, this,
                                           std::placeholders::_1, std::placeholders::_2));
    } else if (topics_.size() == 3) {
        sync3_ = std::make_shared<message_filters::Synchronizer<SyncPolicy3>>(
            SyncPolicy3(queue_size_), *filters_[topics_[0]], *filters_[topics_[1]],
            *filters_[topics_[2]]);

        sync3_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop_));

        sync3_->registerCallback(std::bind(&Ros2BagAsyncReaderATS::messageCallback3, this,
                                           std::placeholders::_1, std::placeholders::_2,
                                           std::placeholders::_3));
    } else {
        throw std::runtime_error("Only 1, 2, or 3 topics are supported");
    }
}

void Ros2BagAsyncReaderATS::signal_message(const std::string &topic_name,
                                           const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
#if VERBOSE
    std::cout << "Signal message on topic: " << topic_name << std::endl;
#endif

    auto it = filters_.find(topic_name);
    if (it != filters_.end()) {
        it->second->feed(msg);
    } else {
        std::cerr << "No filter found for topic: " << topic_name << std::endl;
    }
}

sensor_msgs::msg::Image::SharedPtr Ros2BagAsyncReaderATS::deserialize_bag_message(
    const std::shared_ptr<rosbag2_storage::SerializedBagMessage> &bag_msg) {
    if (!bag_msg || !bag_msg->serialized_data) {
        return nullptr;
    }

    auto msg = std::make_shared<sensor_msgs::msg::Image>();
    try {
        rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
        deserializer_->deserialize_message(&serialized_msg, msg.get());
    } catch (const std::exception &e) {
        std::cerr << "[Deserialization Error] Topic: " << bag_msg->topic_name
                  << ", Error: " << e.what() << std::endl;
        return nullptr;
    }
    return msg;
}

std::optional<Ros2BagAsyncReaderATS::SyncResult> Ros2BagAsyncReaderATS::pop_synced_message() {
    if (synced_msgs_.empty()) {
        return std::nullopt;
    }

    auto synced = synced_msgs_.front();
    synced_msgs_.pop_front();

    std::unordered_map<std::string, sensor_msgs::msg::Image::ConstSharedPtr> result;
    result.reserve(topics_.size());

    double stamp = 0.0;
    bool stamp_initialized = false;

    auto add_if_valid = [&](size_t index, const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        if (!msg || !is_valid_image_message(msg) || index >= topics_.size()) {
            return;
        }
        result.emplace(topics_[index], msg);
        if (!stamp_initialized) {
            stamp = rclcpp::Time(msg->header.stamp).seconds();
            stamp_initialized = true;
        }
    };

    if (!topics_.empty()) {
        add_if_valid(0, std::get<0>(synced));
        if (topics_.size() > 1) {
            add_if_valid(1, std::get<1>(synced));
        }
        if (topics_.size() > 2) {
            add_if_valid(2, std::get<2>(synced));
        }
    }

    if (result.size() == topics_.size() && stamp_initialized) {
        return SyncResult{stamp, std::move(result)};
    }

    return std::nullopt;
}

std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>>
Ros2BagAsyncReaderATS::read_all_messages_of_topic(const std::string &topic, bool with_timestamps) {
    (void)with_timestamps;
    std::vector<std::pair<double, sensor_msgs::msg::Image::ConstSharedPtr>> messages;

    rosbag2_cpp::readers::SequentialReader temp_reader;
#ifdef WITH_JAZZY
    rosbag2_storage::StorageOptions storage_options;
#else
    rb2_storage_ns::StorageOptions storage_options;
#endif
    storage_options.uri = bag_path_;

    if (storage_id_ == "auto") {
        std::string extension = fs_compat::path(bag_path_).extension().string();
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

    rclcpp::Serialization<sensor_msgs::msg::Image> temp_serializer;

    while (temp_reader.has_next()) {
        auto bag_msg = temp_reader.read_next();
        if (!bag_msg || !bag_msg->serialized_data) {
            continue;
        }

        auto msg = std::make_shared<sensor_msgs::msg::Image>();
        try {
            rclcpp::SerializedMessage serialized_msg(*bag_msg->serialized_data);
            temp_serializer.deserialize_message(&serialized_msg, msg.get());
        } catch (const std::exception &e) {
            std::cerr << "[Deserialization Error] Topic: " << topic << ", Error: " << e.what()
                      << std::endl;
            continue;
        }

        double stamp = rclcpp::Time(msg->header.stamp).seconds();
        messages.emplace_back(stamp, msg);
    }

    return messages;
}

std::optional<cv::Mat>
Ros2BagAsyncReaderATS::decode_image(const sensor_msgs::msg::Image::ConstSharedPtr &msg,
                                    const std::string &desired_encoding) {
    if (!msg || msg->encoding.empty() || msg->width == 0 || msg->height == 0) {
        return std::nullopt;
    }

    // Validate image dimensions to prevent memory allocation errors
    // Reasonable limits: max 100000 pixels per dimension (e.g., 100000x100000)
    const uint32_t MAX_DIMENSION = 100000;
    if (msg->width > MAX_DIMENSION || msg->height > MAX_DIMENSION) {
        std::cerr << "[decode_image] Invalid image dimensions: width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding << std::endl;
        return std::nullopt;
    }

    // Validate step value is reasonable (should be at least width * bytes_per_pixel, but not too
    // large) For typical images, step should be reasonable (e.g., < 100MB per row)
    const size_t MAX_STEP = 100 * 1024 * 1024; // 100MB per row is already extremely large
    if (msg->step > MAX_STEP) {
        std::cerr << "[decode_image] Invalid step size: step=" << msg->step
                  << ", width=" << msg->width << ", height=" << msg->height << std::endl;
        return std::nullopt;
    }

    // Validate data size matches expected size
    size_t data_size = msg->data.size();
    size_t expected_size = static_cast<size_t>(msg->height) * static_cast<size_t>(msg->step);
    if (expected_size == 0) {
        std::cerr << "[decode_image] Invalid step size: step=" << msg->step << std::endl;
        return std::nullopt;
    }

    // Check total expected memory size is reasonable (e.g., < 1GB)
    const size_t MAX_TOTAL_SIZE = 1024 * 1024 * 1024; // 1GB
    if (expected_size > MAX_TOTAL_SIZE) {
        std::cerr << "[decode_image] Expected size too large: " << expected_size
                  << " bytes, width=" << msg->width << ", height=" << msg->height
                  << ", step=" << msg->step << std::endl;
        return std::nullopt;
    }

    // Allow some tolerance (10%) for data size mismatch, but check for extreme cases
    if (data_size < expected_size * 0.9 || data_size > expected_size * 1.1) {
        // If the mismatch is extreme (more than 10x), it's likely corrupted
        if (data_size > expected_size * 10 || expected_size > data_size * 10) {
            std::cerr << "[decode_image] Data size mismatch: expected=" << expected_size
                      << ", actual=" << data_size << ", width=" << msg->width
                      << ", height=" << msg->height << ", step=" << msg->step
                      << ", encoding=" << msg->encoding << std::endl;
            return std::nullopt;
        }
    }

    try {
        // For common encodings, manually construct cv::Mat to avoid cv_bridge issues
        // This is more reliable and avoids potential conversion bugs
        cv::Mat img;

        // Try to use cv_bridge with original encoding first, then convert manually
        // This avoids potential bugs in cv_bridge's encoding conversion
        if (msg->encoding == "rgb8" || msg->encoding == "RGB8") {
            // Get image in original RGB8 encoding, then convert to BGR if needed
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
            img = cv_ptr->image.clone();
            // Convert RGB to BGR if needed
            if (desired_encoding == "bgr8" || desired_encoding == "BGR8") {
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
            }
        } else if (msg->encoding == "bgr8" || msg->encoding == "BGR8") {
            // Already BGR8, just get it
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            img = cv_ptr->image.clone();
        } else if (msg->encoding == "mono8" || msg->encoding == "MONO8" ||
                   msg->encoding == "8UC1") {
            // Grayscale
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
            img = cv_ptr->image.clone();
        } else {
            // For other encodings, use cv_bridge with desired encoding
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(msg, desired_encoding);
            return cv_ptr->image.clone(); // Return a copy
        }

        return img;
    } catch (const cv_bridge::Exception &e) {
        std::cerr << "[decode_image] cv_bridge exception: " << e.what() << ", width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding
                  << ", step=" << msg->step << ", data_size=" << msg->data.size() << std::endl;
        return std::nullopt;
    } catch (const cv::Exception &e) {
        std::cerr << "[decode_image] OpenCV exception: " << e.what() << ", width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding
                  << ", step=" << msg->step << ", data_size=" << msg->data.size() << std::endl;
        return std::nullopt;
    } catch (const std::exception &e) {
        std::cerr << "[decode_image] exception: " << e.what() << ", width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding
                  << ", step=" << msg->step << ", data_size=" << msg->data.size() << std::endl;
        return std::nullopt;
    }
}

std::optional<cv::Mat>
Ros2BagAsyncReaderATS::decode_depth_image(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (!msg || msg->encoding.empty() || msg->width == 0 || msg->height == 0) {
        return std::nullopt;
    }

    // Validate image dimensions to prevent memory allocation errors
    const uint32_t MAX_DIMENSION = 100000;
    if (msg->width > MAX_DIMENSION || msg->height > MAX_DIMENSION) {
        std::cerr << "[decode_depth_image] Invalid image dimensions: width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding << std::endl;
        return std::nullopt;
    }

    // Check if this is actually a color image (common synchronizer mismatch)
    std::string encoding_lower = msg->encoding;
    std::transform(encoding_lower.begin(), encoding_lower.end(), encoding_lower.begin(), ::tolower);
    std::vector<std::string> color_encodings = {"rgb8",  "bgr8", "rgba8", "bgra8",
                                                "mono8", "8uc3", "8uc4"};
    bool is_color = false;
    for (const auto &enc : color_encodings) {
        if (encoding_lower == enc || encoding_lower.find("rgb") != std::string::npos ||
            encoding_lower.find("bgr") != std::string::npos) {
            is_color = true;
            break;
        }
    }
    if (is_color) {
        std::cerr << "[decode_depth_image] Received color image encoding '" << msg->encoding
                  << "' for depth topic. Skipping." << std::endl;
        return std::nullopt;
    }

    try {
        size_t data_size = msg->data.size();
        size_t expected_pixels = static_cast<size_t>(msg->height) * static_cast<size_t>(msg->width);

        if (expected_pixels == 0) {
            return std::nullopt;
        }

        // Calculate expected data size based on encoding
        size_t bytes_per_pixel = 0;
        if (msg->encoding == "32FC1") {
            bytes_per_pixel = 4; // float32
        } else if (msg->encoding == "16UC1" || msg->encoding == "16SC1") {
            bytes_per_pixel = 2; // uint16/int16
        } else if (msg->encoding == "8UC1") {
            bytes_per_pixel = 1; // uint8
        } else {
            std::cerr << "[decode_depth_image] Unsupported depth encoding: " << msg->encoding
                      << std::endl;
            return std::nullopt;
        }

        size_t expected_data_size = expected_pixels * bytes_per_pixel;

        // Check total expected memory size is reasonable (e.g., < 1GB)
        const size_t MAX_TOTAL_SIZE = 1024 * 1024 * 1024; // 1GB
        if (expected_data_size > MAX_TOTAL_SIZE) {
            std::cerr << "[decode_depth_image] Expected size too large: " << expected_data_size
                      << " bytes, width=" << msg->width << ", height=" << msg->height
                      << ", encoding=" << msg->encoding << std::endl;
            return std::nullopt;
        }

        // Validate data size matches expected size (with some tolerance)
        if (data_size < expected_data_size * 0.9 || data_size > expected_data_size * 1.1) {
            // If the mismatch is extreme (more than 10x), it's likely corrupted
            if (data_size > expected_data_size * 10 || expected_data_size > data_size * 10) {
                std::cerr << "[decode_depth_image] Data size mismatch: expected="
                          << expected_data_size << ", actual=" << data_size
                          << ", width=" << msg->width << ", height=" << msg->height
                          << ", encoding=" << msg->encoding
                          << ", bytes_per_pixel=" << bytes_per_pixel << std::endl;
                return std::nullopt;
            }
        }

        cv::Mat depth_img;

        // Handle different depth encodings
        if (msg->encoding == "32FC1") {
            depth_img = cv::Mat(msg->height, msg->width, CV_32FC1,
                                const_cast<void *>(static_cast<const void *>(msg->data.data())))
                            .clone();
        } else if (msg->encoding == "16UC1") {
            cv::Mat depth_uint16(msg->height, msg->width, CV_16UC1,
                                 const_cast<void *>(static_cast<const void *>(msg->data.data())));
            depth_uint16.convertTo(depth_img, CV_32FC1, 1.0 / 1000.0); // Convert mm to meters
        } else if (msg->encoding == "16SC1") {
            cv::Mat depth_int16(msg->height, msg->width, CV_16SC1,
                                const_cast<void *>(static_cast<const void *>(msg->data.data())));
            depth_int16.convertTo(depth_img, CV_32FC1, 1.0 / 1000.0);
        } else if (msg->encoding == "8UC1") {
            cv::Mat depth_uint8(msg->height, msg->width, CV_8UC1,
                                const_cast<void *>(static_cast<const void *>(msg->data.data())));
            depth_uint8.convertTo(depth_img, CV_32FC1, 1.0 / 255.0);
        } else {
            std::cerr << "[decode_depth_image] Unsupported depth encoding: " << msg->encoding
                      << std::endl;
            return std::nullopt;
        }

        return depth_img;
    } catch (const std::exception &e) {
        std::cerr << "[decode_depth_image] exception: " << e.what() << ", width=" << msg->width
                  << ", height=" << msg->height << ", encoding=" << msg->encoding
                  << ", step=" << msg->step << ", data_size=" << msg->data.size() << std::endl;
        return std::nullopt;
    }
}
