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

#include "image_prefetcher.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace pyslam {
namespace {

struct NpyInfo {
    int depth = -1;
    int rows = 0;
    int cols = 0;
    int channels = 1;
    size_t payload_bytes = 0;
};

bool parse_npy_header(const std::string &header, NpyInfo &info) {
    const auto descr_key = header.find("descr");
    if (descr_key == std::string::npos)
        return false;
    const auto key_quote = header.find('\'', descr_key);
    const auto value_quote = key_quote == std::string::npos ? std::string::npos
                                                             : header.find('\'', key_quote + 1);
    const auto value_end =
        value_quote == std::string::npos ? std::string::npos : header.find('\'', value_quote + 1);
    if (value_quote == std::string::npos || value_end == std::string::npos)
        return false;
    const std::string descr = header.substr(value_quote + 1, value_end - value_quote - 1);

    const auto fortran_key = header.find("fortran_order");
    if (fortran_key == std::string::npos)
        return false;
    const auto fortran_window = header.substr(fortran_key, std::min<size_t>(32, header.size() - fortran_key));
    if (fortran_window.find("True") != std::string::npos)
        return false;

    int cv_depth = -1;
    size_t elem_size = 0;
    if (descr == "<f4" || descr == "|f4" || descr == "f4") {
        cv_depth = CV_32F;
        elem_size = 4;
    } else if (descr == "<f8" || descr == "|f8" || descr == "f8") {
        cv_depth = CV_64F;
        elem_size = 8;
    } else if (descr == "<u2" || descr == "|u2" || descr == "u2") {
        cv_depth = CV_16U;
        elem_size = 2;
    } else if (descr == "<i2" || descr == "|i2" || descr == "i2") {
        cv_depth = CV_16S;
        elem_size = 2;
    } else if (descr == "<u1" || descr == "|u1" || descr == "u1") {
        cv_depth = CV_8U;
        elem_size = 1;
    } else if (descr == "<i4" || descr == "|i4" || descr == "i4") {
        cv_depth = CV_32S;
        elem_size = 4;
    } else {
        return false;
    }

    const auto shape_key = header.find("shape");
    const auto open = shape_key == std::string::npos ? std::string::npos : header.find('(', shape_key);
    const auto close = open == std::string::npos ? std::string::npos : header.find(')', open);
    if (open == std::string::npos || close == std::string::npos)
        return false;

    std::vector<int> shape;
    int value = 0;
    bool in_number = false;
    for (size_t i = open + 1; i < close; ++i) {
        const char c = header[i];
        if (c >= '0' && c <= '9') {
            // Bound dimensions while parsing, before integer overflow is possible.
            if (value > 20000 / 10)
                return false;
            value = value * 10 + (c - '0');
            if (value > 20000)
                return false;
            in_number = true;
        } else {
            if (c != ',' && c != ' ' && c != '\t')
                return false;
            if (in_number) {
                shape.push_back(value);
                value = 0;
                in_number = false;
            }
        }
    }
    if (in_number)
        shape.push_back(value);
    if (shape.size() != 2 && shape.size() != 3)
        return false;
    if (shape[0] <= 0 || shape[1] <= 0 || shape[0] > 20000 || shape[1] > 20000)
        return false;

    info.depth = cv_depth;
    info.rows = shape[0];
    info.cols = shape[1];
    info.channels = 1;
    if (shape.size() == 3) {
        if (shape[2] <= 0 || shape[2] > CV_CN_MAX)
            return false;
        info.channels = shape[2];
    }
    info.payload_bytes = static_cast<size_t>(info.rows) * static_cast<size_t>(info.cols) *
                          static_cast<size_t>(info.channels) * elem_size;
    return info.payload_bytes > 0;
}

cv::Mat read_npy(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};

    char magic[6] = {};
    input.read(magic, 6);
    if (!input || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        return {};

    unsigned char major = 0;
    unsigned char minor = 0;
    input.read(reinterpret_cast<char *>(&major), 1);
    input.read(reinterpret_cast<char *>(&minor), 1);
    if (!input)
        return {};

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        input.read(reinterpret_cast<char *>(&len16), 2);
        header_len = len16;
    } else if (major == 2 || major == 3) {
        input.read(reinterpret_cast<char *>(&header_len), 4);
    } else {
        return {};
    }
    if (!input || header_len == 0 || header_len > (16u << 20))
        return {};

    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!input)
        return {};

    NpyInfo info;
    if (!parse_npy_header(header, info))
        return {};

    const auto payload_start = input.tellg();
    input.seekg(0, std::ios::end);
    if (!input || input.tellg() - payload_start < static_cast<std::streamoff>(info.payload_bytes))
        return {};
    input.seekg(payload_start);
    cv::Mat mat(info.rows, info.cols, CV_MAKETYPE(info.depth, info.channels));
    input.read(reinterpret_cast<char *>(mat.data), static_cast<std::streamsize>(info.payload_bytes));
    if (!input)
        return {};
    return mat;
}

} // namespace

ImagePrefetcher::ImagePrefetcher(int capacity) : capacity_(std::max(1, capacity)) {}

ImagePrefetcher::~ImagePrefetcher() { stop(); }

void ImagePrefetcher::ensure_configurable() const {
    if (started_)
        throw std::runtime_error("ImagePrefetcher is already running");
}

ImagePrefetcher::Stream &ImagePrefetcher::stream_or_throw(const std::string &name) {
    const auto it = name_to_index_.find(name);
    if (it == name_to_index_.end())
        throw std::out_of_range("Unknown image stream: " + name);
    return streams_[it->second];
}

const ImagePrefetcher::Stream &ImagePrefetcher::stream_or_throw(const std::string &name) const {
    const auto it = name_to_index_.find(name);
    if (it == name_to_index_.end())
        throw std::out_of_range("Unknown image stream: " + name);
    return streams_[it->second];
}

void ImagePrefetcher::add_stream(const std::string &name, const std::vector<std::string> &paths,
                                 int imread_flag, bool npy) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    if (name.empty() || name_to_index_.count(name) != 0)
        throw std::runtime_error("Duplicate or empty image stream: " + name);

    Stream stream;
    stream.name = name;
    stream.paths = paths;
    stream.imread_flag = imread_flag;
    stream.npy = npy;
    name_to_index_.emplace(name, static_cast<int>(streams_.size()));
    streams_.push_back(std::move(stream));
}

void ImagePrefetcher::add_clahe(const std::string &name, double clip, int tile) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    Op op;
    op.kind = OpKind::Clahe;
    op.clip = clip;
    op.tile = std::max(1, tile);
    stream_or_throw(name).ops.push_back(std::move(op));
}

void ImagePrefetcher::add_remap(const std::string &name, const cv::Mat &map1, const cv::Mat &map2) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    Op op;
    op.kind = OpKind::Remap;
    // Own the maps. The incoming headers may alias NumPy memory.
    op.map1 = map1.clone();
    if (!map2.empty())
        op.map2 = map2.clone();
    stream_or_throw(name).ops.push_back(std::move(op));
}

void ImagePrefetcher::add_resize(const std::string &name, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    if (width <= 0 || height <= 0)
        throw std::runtime_error("resize expects a positive width and height");
    Op op;
    op.kind = OpKind::Resize;
    op.width = width;
    op.height = height;
    stream_or_throw(name).ops.push_back(std::move(op));
}

void ImagePrefetcher::add_gray_to_rgb(const std::string &name) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    Op op;
    op.kind = OpKind::GrayToRgb;
    stream_or_throw(name).ops.push_back(std::move(op));
}

void ImagePrefetcher::add_scale_float(const std::string &name, double factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    if (!(factor > 0.0) || !std::isfinite(factor))
        throw std::runtime_error("scale_float factor must be positive");
    Op op;
    op.kind = OpKind::ScaleFloat;
    op.factor = factor;
    stream_or_throw(name).ops.push_back(std::move(op));
}

void ImagePrefetcher::set_expected_size(const std::string &name, int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    ensure_configurable();
    Stream &stream = stream_or_throw(name);
    stream.expected_width = width;
    stream.expected_height = height;
}

void ImagePrefetcher::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (started_)
        return;
    if (stop_)
        throw std::runtime_error("ImagePrefetcher was already stopped");
    started_ = true;
    worker_ = std::thread([this]() { run(); });
}

void ImagePrefetcher::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    std::lock_guard<std::mutex> join_lock(join_mutex_);
    if (worker_.joinable())
        worker_.join();
}

bool ImagePrefetcher::has_stream(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return name_to_index_.count(name) != 0;
}

int ImagePrefetcher::stream_length(const std::string &name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = name_to_index_.find(name);
    if (it == name_to_index_.end())
        return 0;
    return static_cast<int>(streams_[it->second].paths.size());
}

int ImagePrefetcher::unpublished_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int largest = 0;
    for (const Stream &stream : streams_)
        largest = std::max(largest, static_cast<int>(stream.queue.size()));
    return largest;
}

bool ImagePrefetcher::has_work_locked() const {
    for (const Stream &stream : streams_) {
        if (stream.seek_id >= 0)
            return true;
        if (stream.next_id < static_cast<int>(stream.paths.size()) &&
            static_cast<int>(stream.queue.size()) < capacity_)
            return true;
    }
    return false;
}

void ImagePrefetcher::apply_seeks_locked() {
    for (Stream &stream : streams_) {
        if (stream.seek_id < 0)
            continue;
        stream.queue.clear();
        stream.next_id = stream.seek_id;
        stream.seek_id = -1;
    }
}

int ImagePrefetcher::pick_stream_locked() const {
    int best = -1;
    int best_id = 0;
    for (int i = 0; i < static_cast<int>(streams_.size()); ++i) {
        const Stream &stream = streams_[i];
        if (stream.next_id >= static_cast<int>(stream.paths.size()))
            continue;
        if (static_cast<int>(stream.queue.size()) >= capacity_)
            continue;
        if (best < 0 || stream.next_id < best_id) {
            best = i;
            best_id = stream.next_id;
        }
    }
    return best;
}

PrefetchImage ImagePrefetcher::decode(Stream &stream, int frame_id) {
    PrefetchImage result;
    try {
        if (frame_id < 0 || frame_id >= static_cast<int>(stream.paths.size()))
            return result;
        const std::string &path = stream.paths[frame_id];
        cv::Mat image = stream.npy ? read_npy(path) : cv::imread(path, stream.imread_flag);
        if (image.empty())
            return result;
        if (stream.expected_width > 0 && stream.expected_height > 0 &&
            (image.cols != stream.expected_width || image.rows != stream.expected_height))
            return result;

        for (Op &op : stream.ops) {
            cv::Mat converted;
            switch (op.kind) {
            case OpKind::Clahe: {
                if (image.channels() != 1 || image.depth() != CV_8U)
                    return result;
                if (!op.clahe)
                    op.clahe = cv::createCLAHE(op.clip, cv::Size(op.tile, op.tile));
                op.clahe->apply(image, converted);
                image = std::move(converted);
                break;
            }
            case OpKind::Remap:
                cv::remap(image, converted, op.map1, op.map2, cv::INTER_LINEAR);
                image = std::move(converted);
                break;
            case OpKind::Resize:
                cv::resize(image, converted, cv::Size(op.width, op.height), 0, 0, cv::INTER_LINEAR);
                image = std::move(converted);
                break;
            case OpKind::GrayToRgb:
                if (image.channels() == 1) {
                    cv::cvtColor(image, converted, cv::COLOR_GRAY2RGB);
                    image = std::move(converted);
                }
                break;
            case OpKind::ScaleFloat: {
                // Match NumPy's (image / factor).astype(float32): perform the
                // division in double precision, then round once to float32.
                cv::Mat doubles;
                image.convertTo(doubles, CV_64F);
                converted.create(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
                for (int row = 0; row < image.rows; ++row) {
                    const double *src = doubles.ptr<double>(row);
                    float *dst = converted.ptr<float>(row);
                    for (int col = 0; col < image.cols * image.channels(); ++col)
                        dst[col] = static_cast<float>(src[col] / op.factor);
                }
                image = std::move(converted);
                break;
            }
            }
            if (image.empty())
                return result;
        }
        if (!image.isContinuous())
            image = image.clone();
        result.ok = true;
        result.image = std::move(image);
        return result;
    } catch (const cv::Exception &) {
        return PrefetchImage{};
    } catch (const std::exception &) {
        return PrefetchImage{};
    }
}

void ImagePrefetcher::run() {
    while (true) {
        int stream_index = -1;
        int frame_id = -1;
        int generation = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]() { return stop_ || has_work_locked(); });
            if (stop_)
                return;
            apply_seeks_locked();
            stream_index = pick_stream_locked();
            if (stream_index < 0)
                continue;
            frame_id = streams_[stream_index].next_id;
            generation = streams_[stream_index].seek_generation;
        }

        PrefetchImage decoded = decode(streams_[stream_index], frame_id);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_)
                return;
            Stream &stream = streams_[stream_index];
            if (generation != stream.seek_generation)
                continue;
            // A seek applied while we were decoding may have moved next_id.
            if (stream.next_id != frame_id)
                continue;
            Slot slot;
            slot.frame_id = frame_id;
            slot.ok = decoded.ok;
            slot.image = std::move(decoded.image);
            stream.queue.push_back(std::move(slot));
            stream.next_id = frame_id + 1;
        }
        cv_.notify_all();
    }
}

PrefetchImage ImagePrefetcher::get(const std::string &name, int frame_id) {
    std::lock_guard<std::mutex> request_lock(get_mutex_);
    std::unique_lock<std::mutex> lock(mutex_);
    if (stop_ || !started_)
        return {};
    const auto it = name_to_index_.find(name);
    if (it == name_to_index_.end())
        return {};
    const int stream_index = it->second;
    Stream &stream = streams_[stream_index];
    if (frame_id < 0 || frame_id >= static_cast<int>(stream.paths.size()))
        return {};

    while (!stream.queue.empty() && stream.queue.front().frame_id < frame_id)
        stream.queue.pop_front();

    auto take_front = [&]() -> PrefetchImage {
        Slot slot = std::move(stream.queue.front());
        stream.queue.pop_front();
        lock.unlock();
        cv_.notify_all();
        PrefetchImage result;
        result.ok = slot.ok && !slot.image.empty();
        result.image = std::move(slot.image);
        return result;
    };

    if (!stream.queue.empty() && stream.queue.front().frame_id == frame_id)
        return take_front();

    // Dropping unpublished frames, or asking for an id the worker has not
    // reached, has to wake it. Otherwise it can stay asleep on a full queue.
    cv_.notify_all();

    // Backward/repeated reads also go through the worker. All image decoding
    // and conversions must stay off the requesting thread.
    if (frame_id != stream.next_id) {
        stream.seek_id = frame_id;
        ++stream.seek_generation;
        stream.queue.clear();
        cv_.notify_all();
    }

    cv_.wait(lock, [&]() {
        return stop_ || (!stream.queue.empty() && stream.queue.front().frame_id == frame_id);
    });
    if (!stream.queue.empty() && stream.queue.front().frame_id == frame_id)
        return take_front();
    return {};
}

} // namespace pyslam
