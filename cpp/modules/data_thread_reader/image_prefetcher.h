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
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace pyslam {

// One decoded image. `ok == false` is a missing file or a failed conversion,
// which the Python wrapper turns into None.
struct PrefetchImage {
    bool ok = false;
    cv::Mat image;
};

// Background decoder with one bounded queue per stream.
//
// The worker reads the next frame of whichever stream is furthest behind and
// still has room. A stream that nobody consumes stops at `capacity` frames and
// does not block the others. `get` moves the cv::Mat out of the queue, so the
// caller owns the pixels and the handoff does not copy them.
class ImagePrefetcher {
  public:
    explicit ImagePrefetcher(int capacity);
    ~ImagePrefetcher();

    ImagePrefetcher(const ImagePrefetcher &) = delete;
    ImagePrefetcher &operator=(const ImagePrefetcher &) = delete;

    void add_stream(const std::string &name, const std::vector<std::string> &paths, int imread_flag,
                    bool npy);
    void add_clahe(const std::string &name, double clip, int tile);
    void add_remap(const std::string &name, const cv::Mat &map1, const cv::Mat &map2);
    void add_resize(const std::string &name, int width, int height);
    void add_gray_to_rgb(const std::string &name);
    void add_scale_float(const std::string &name, double factor);
    void set_expected_size(const std::string &name, int width, int height);

    void start();
    void stop();

    PrefetchImage get(const std::string &name, int frame_id);
    bool has_stream(const std::string &name) const;
    int stream_length(const std::string &name) const;
    int unpublished_count() const;
    int capacity() const { return capacity_; }

  private:
    enum class OpKind { Clahe, Remap, Resize, GrayToRgb, ScaleFloat };

    struct Op {
        OpKind kind = OpKind::GrayToRgb;
        double clip = 3.0;
        int tile = 8;
        cv::Mat map1;
        cv::Mat map2;
        int width = 0;
        int height = 0;
        double factor = 1.0;
        cv::Ptr<cv::CLAHE> clahe;
    };

    struct Slot {
        int frame_id = -1;
        bool ok = false;
        cv::Mat image;
    };

    struct Stream {
        std::string name;
        std::vector<std::string> paths;
        int imread_flag = cv::IMREAD_COLOR;
        bool npy = false;
        int expected_width = 0;
        int expected_height = 0;
        std::vector<Op> ops;
        std::deque<Slot> queue;
        int next_id = 0;
        int seek_id = -1;
        int seek_generation = 0;
    };

    void ensure_configurable() const;
    Stream &stream_or_throw(const std::string &name);
    const Stream &stream_or_throw(const std::string &name) const;
    void run();
    bool has_work_locked() const;
    void apply_seeks_locked();
    int pick_stream_locked() const;
    PrefetchImage decode(Stream &stream, int frame_id);

    int capacity_ = 6;
    bool started_ = false;
    bool stop_ = false;
    std::vector<Stream> streams_;
    std::unordered_map<std::string, int> name_to_index_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    // Serialize requests so simultaneous seeks cannot strand another caller.
    std::mutex get_mutex_;
    std::mutex join_mutex_;
    std::thread worker_;
};

} // namespace pyslam
