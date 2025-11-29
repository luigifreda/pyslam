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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace volumetric {

// Macro for position-related members
#define VOXEL_POSITION_MEMBERS()                                                                   \
    std::array<double, 3> position_sum = {0.0, 0.0, 0.0}; // sum of positions

// Macro for position-related methods
#define VOXEL_POSITION_METHODS()                                                                   \
    template <typename Tpos> void update_point(const Tpos x, const Tpos y, const Tpos z) {         \
        position_sum[0] += static_cast<double>(x);                                                 \
        position_sum[1] += static_cast<double>(y);                                                 \
        position_sum[2] += static_cast<double>(z);                                                 \
    }                                                                                              \
    std::array<double, 3> get_position() const {                                                   \
        const double count_d = static_cast<double>(count);                                         \
        std::array<double, 3> avg_coord = {position_sum[0] / count_d, position_sum[1] / count_d,   \
                                           position_sum[2] / count_d};                             \
        return avg_coord;                                                                          \
    }

// Macro for color-related members
#define VOXEL_COLOR_MEMBERS() std::array<float, 3> color_sum = {0.0f, 0.0f, 0.0f}; // sum of colors

// Macro for color-related methods
#define VOXEL_COLOR_METHODS()                                                                      \
    template <typename Tcolor>                                                                     \
    void update_color(const Tcolor color_x, const Tcolor color_y, const Tcolor color_z) {          \
        if constexpr (std::is_same_v<Tcolor, uint8_t>) {                                           \
            color_sum[0] += static_cast<float>(color_x) / 255.0f;                                  \
            color_sum[1] += static_cast<float>(color_y) / 255.0f;                                  \
            color_sum[2] += static_cast<float>(color_z) / 255.0f;                                  \
        } else if constexpr (std::is_same_v<Tcolor, float>) {                                      \
            color_sum[0] += static_cast<float>(color_x);                                           \
            color_sum[1] += static_cast<float>(color_y);                                           \
            color_sum[2] += static_cast<float>(color_z);                                           \
        } else if constexpr (std::is_same_v<Tcolor, double>) {                                     \
            color_sum[0] += static_cast<float>(color_x);                                           \
            color_sum[1] += static_cast<float>(color_y);                                           \
            color_sum[2] += static_cast<float>(color_z);                                           \
        } else {                                                                                   \
            static_assert(!std::is_same_v<Tcolor, uint8_t> && !std::is_same_v<Tcolor, float> &&    \
                              !std::is_same_v<Tcolor, double>,                                     \
                          "Unsupported color type");                                               \
        }                                                                                          \
    }                                                                                              \
    std::array<float, 3> get_color() const {                                                       \
        const float count_f = static_cast<float>(count);                                           \
        std::array<float, 3> avg_color = {color_sum[0] / count_f, color_sum[1] / count_f,          \
                                          color_sum[2] / count_f};                                 \
        return avg_color;                                                                          \
    }

//=================================================================================================
// Voxel Data
//=================================================================================================

// Simple voxel data structure for just storing and managing points and colors.
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
struct VoxelData {
    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
    }
};

//=================================================================================================
// Voxel Semantic Data
//=================================================================================================

// Semantic voxel data structure for storing points, colors, instance IDs, and class IDs.
// The semantic update is a simple voting mechanism.
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
// - Instance IDs and class IDs are used to store the most observed instance and class IDs of the
// voxel.
// - Confidence counter is used to keep track of the confidence of the most observed/voted instance
// and class IDs of the voxel. It is incremented when the instance and class IDs are the same and
// decremented when they are different. If the confidence counter is less than or equal to 0, the
// instance and class IDs are set to the new values and the confidence counter is reset to 1.
struct VoxelSemanticData {
    static float kDepthThreshold; // [m] depth threshold for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    int instance_id = -1;       // instance ID
    int class_id = -1;          // class ID
    int confidence_counter = 0; // confidence counter for the instance and class IDs

    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        instance_id = -1;
        class_id = -1;
        confidence_counter = 0;
    }

    void initialize_semantics(const int instance_id, const int class_id) {
        this->instance_id = instance_id;
        this->class_id = class_id;
        this->confidence_counter = 1;
    }

    void initialize_semantics_with_depth(const int instance_id, const int class_id,
                                         const float depth) {

        if (depth < kDepthThreshold) {
            initialize_semantics(instance_id, class_id);
        }
    }

    void update_semantics(const int instance_id, const int class_id) {

        if (this->instance_id == instance_id && this->class_id == class_id) {
            // Update confidence counter if the instance and class IDs are the same
            this->confidence_counter++;
        } else {
            // If the instance and class IDs are different, decrement the confidence counter
            this->confidence_counter--;
            if (this->confidence_counter <= 0) {
                // If the confidence counter is less than or equal to 0, set the instance
                // and class IDs and reset the confidence counter to 1
                this->instance_id = instance_id;
                this->class_id = class_id;
                this->confidence_counter = 1;
            }
        }
    }

    void update_semantics_with_depth(const int instance_id, const int class_id, const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => confidence_counter = 1
        if (depth < kDepthThreshold) {
            update_semantics(instance_id, class_id);
        }
    }
};

float VoxelSemanticData::kDepthThreshold = 10.0f; // [m] depth threshold for updating semantics with
                                                  // depth ~10 indoor , ~20 outdoor

//=================================================================================================
// Voxel Semantic Data Probabilistic
//=================================================================================================

// Semantic voxel data structure with probabilistic fusion
// Uses Bayesian fusion in log-space for efficient and numerically stable updates
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
// - Instance IDs and class IDs are used to store the most likely label pair of the voxel.
//
// Probabilistic update (ideal Bayesian fusion):
//  Let L = (i,c) be a label pair (instance_id, class_id)
//  We have a prior distribution p(L|old) after having observed n points in the voxel.
//  When we get a new observation x_new (one more point measurement in the voxel), Bayes update
//  says: p(L|x_new, old) = p(x_new|L,old) * p(L|old) / p(x_new|old) Assuming conditional
//  idependence of the observations given the label (the usual "naive Bayes" assumption), we have:
//  P(x_new|L,old) = P(x_new|L)
//  Therefore:
//  p(L|x_new, old) = p(x_new|L) * p(L|old) / sum_i {p(x_new|L_i,old) * p(L_i|old)}
//  Simplifying the notation, we get:
//  p(new) = likelihood(new) * prior(old) / normalization
//  where likelihood(new) = p(x_new|L) is the "likelihood contribution" of the new observation to
//  the label L and prior(old) = p(L|old)
//
// Efficiency features:
// - Sparse storage: only stores probabilities for observed (instance_id, class_id) pairs
// - Log-space arithmetic: avoids numerical underflow, multiplication becomes addition
// - Lazy evaluation: most likely label is cached and recomputed only when needed
// - Memory efficient: O(k) storage where k is the number of unique label pairs observed
// - Uses std::map instead of std::unordered_map for better performance with few observations
//   (typically 1-5 unique labels per voxel, where std::map's O(log n) is effectively constant
//   and has lower overhead than hash tables)
//
// Usage:
// - Call initialize_semantics() for the first observation
// - Call update_semantics() for subsequent observations (optionally with confidence scores)
// - Access most likely label via get_instance_id(), get_class_id(), or get_most_likely_label()
// - For direct member access (instance_id, class_id, confidence_counter), call
// ensure_cache_updated() first
//   or use the getter methods which handle this automatically
struct VoxelSemanticDataProbabilistic {

    static float kDepthThreshold; // [m] depth threshold for updating semantics with depth
    static float kDepthDecayRate; // [1/m] depth decay rate for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    // Sparse probability distribution: (instance_id, class_id) -> log_probability
    // Using log probabilities avoids numerical underflow and makes multiplication additive
    // std::map is used instead of std::unordered_map for better performance with few observations
    // (std::pair<int, int> is naturally comparable, so no custom comparator needed)
    std::map<std::pair<int, int>, float> log_probabilities;

    // Cached most likely label pair (updated lazily), (instance_id, class_id) pair
    mutable std::pair<int, int> most_likely_pair = {-1, -1};
    mutable float most_likely_log_prob = -std::numeric_limits<float>::infinity();
    mutable bool cache_valid = false;

    // Direct access members for compatibility (lazy evaluation via getters)
    // Note: These are mutable to allow const access patterns
    mutable int instance_id = -1;       // cached most likely instance ID
    mutable int class_id = -1;          // cached most likely class ID
    mutable int confidence_counter = 0; // cached confidence counter

    static constexpr float MIN_CONFIDENCE = 1e-10f;

    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        log_probabilities.clear();
        most_likely_pair = {-1, -1};
        most_likely_log_prob = -std::numeric_limits<float>::infinity();
        cache_valid = false;
        // Reset cached members for compatibility
        instance_id = -1;
        class_id = -1;
        confidence_counter = 0;
    }

    void initialize_semantics_log_prob(const int instance_id, const int class_id,
                                       const float log_prob) {
        const auto key = std::make_pair(instance_id, class_id);
        // Initialize with log(confidence), assuming uniform prior
        // Using log(confidence) as initial log probability
        // const float log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));

        log_probabilities[key] = log_prob;
        most_likely_pair = key;
        most_likely_log_prob = log_prob;
        cache_valid = true;
        // Update cached members for compatibility
        this->instance_id = most_likely_pair.first;
        this->class_id = most_likely_pair.second;
        this->confidence_counter = get_confidence_counter();
    }

    void initialize_semantics(const int instance_id, const int class_id,
                              const float confidence = 1.0f) {
        // Initialize with log(confidence), assuming uniform prior
        // Using log(confidence) as initial log probability
        const float log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        initialize_semantics_log_prob(instance_id, class_id, log_prob);
    }

    void initialize_semantics_with_depth(const int instance_id, const int class_id,
                                         const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;
        initialize_semantics_log_prob(instance_id, class_id, log_prob);
    }

    void update_semantics_log_prob(const int instance_id, const int class_id,
                                   const float new_log_prob) {

        // const float new_log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));

        const auto key = std::make_pair(instance_id, class_id);
        // Bayesian fusion in log space: log(P(new|old)) = log(P(new)) + log(P(old)) -
        // log(normalization) For efficiency, we accumulate log probabilities and normalize
        // periodically Here we use a simple additive update: log_prob += log(confidence) This
        // approximates Bayesian fusion when we normalize by the total count
        // Use operator[] for cleaner code - inserts 0.0f if key doesn't exist
        float &log_prob = log_probabilities[key];
        log_prob += new_log_prob;

        // Incremental cache maintenance: update most likely label on-the-fly if cache is valid
        // This avoids full recomputation in the common case
        if (cache_valid) {
            if (key == most_likely_pair) {
                // Still the argmax; value increased
                most_likely_log_prob = log_prob;
            } else if (log_prob > most_likely_log_prob) {
                // New argmax found
                most_likely_log_prob = log_prob;
                most_likely_pair = key;
            }
            // If this is a new key (was 0.0f) and it's not the new max, cache remains valid
        } else {
            // Cache invalid; will be recomputed lazily when needed
        }
    }

    void update_semantics(const int instance_id, const int class_id,
                          const float confidence = 1.0f) {
        const float new_log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        update_semantics_log_prob(instance_id, class_id, new_log_prob);
    }

    void update_semantics_with_depth(const int instance_id, const int class_id, const float depth) {

        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float new_log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;

        update_semantics_log_prob(instance_id, class_id, new_log_prob);
    }

    // Get the most likely (instance_id, class_id) pair
    std::pair<int, int> get_most_likely_label() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair;
    }

    // Get the instance_id of the most likely label
    int get_instance_id() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair.first;
    }

    // Get the class_id of the most likely label
    int get_class_id() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair.second;
    }

    // Get the probability of the most likely label (in linear space)
    float get_most_likely_probability() const {
        if (!cache_valid) {
            update_cache();
        }
        if (log_probabilities.empty()) {
            return 0.0f;
        }
        // Normalize probabilities and return the max
        return std::exp(most_likely_log_prob - get_log_normalization());
    }

    // Get confidence counter (compatible interface) - returns the count for the most likely label
    int get_confidence_counter() const {
        if (!cache_valid) {
            update_cache();
        }
        if (most_likely_pair.first == -1 || most_likely_pair.second == -1) {
            return 0;
        }
        // Return a confidence based on the normalized probability
        // Scale by count to maintain similar semantics to the counter-based version
        return static_cast<int>(get_most_likely_probability() * count);
    }

    // For compatibility with existing interface - these are computed on access
    // Accessing these will trigger cache update if needed
    int get_instance_id_member() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair.first;
    }

    int get_class_id_member() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair.second;
    }

    int get_confidence_counter_member() const { return get_confidence_counter(); }

    // Accessor that updates cached values (for compatibility with direct member access)
    void ensure_cache_updated() const {
        if (!cache_valid) {
            update_cache();
            const_cast<VoxelSemanticDataProbabilistic *>(this)->instance_id =
                most_likely_pair.first;
            const_cast<VoxelSemanticDataProbabilistic *>(this)->class_id = most_likely_pair.second;
            const_cast<VoxelSemanticDataProbabilistic *>(this)->confidence_counter =
                get_confidence_counter();
        }
    }

  private:
    // Update the cache of most likely label
    void update_cache() const {
        if (log_probabilities.empty()) {
            most_likely_pair = {-1, -1};
            most_likely_log_prob = -std::numeric_limits<float>::infinity();
            cache_valid = true;
            return;
        }

        // Find the maximum log probability
        most_likely_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[key, log_prob] : log_probabilities) {
            if (log_prob > most_likely_log_prob) {
                most_likely_log_prob = log_prob;
                most_likely_pair = key;
            }
        }
        cache_valid = true;
    }

    // Compute log of normalization constant (log-sum-exp trick for numerical stability)
    float get_log_normalization() const {
        if (log_probabilities.empty()) {
            return 0.0f;
        }

        // Find max log probability
        float max_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[key, log_prob] : log_probabilities) {
            if (log_prob > max_log_prob) {
                max_log_prob = log_prob;
            }
        }

        // Compute log-sum-exp: log(sum(exp(x_i))) = max + log(sum(exp(x_i - max)))
        float sum = 0.0f;
        for (const auto &[key, log_prob] : log_probabilities) {
            sum += std::exp(log_prob - max_log_prob);
        }
        return max_log_prob + std::log(sum);
    }

    // Prune low probability labels from the probability distribution
    void prune_low_prob(float margin = 5.0f /* nats */) {
        if (!cache_valid)
            update_cache();
        const float thresh = most_likely_log_prob - margin;
        for (auto it = log_probabilities.begin(); it != log_probabilities.end();) {
            if (it->second < thresh)
                it = log_probabilities.erase(it);
            else
                ++it;
        }
        // norm_valid = false; // if we cache logZ
    }
};

float VoxelSemanticDataProbabilistic::kDepthThreshold =
    5.0f; // [m] depth threshold for updating semantics with depth
          // ~5 indoor , ~10 outdoor
float VoxelSemanticDataProbabilistic::kDepthDecayRate =
    0.07f; // [1/m] depth decay rate for updating semantics with depth

//=================================================================================================
// Voxel Concepts
//=================================================================================================

// Concept for basic voxel data types
// Checks that a type has all the required members and methods for basic voxel functionality
template <typename T>
concept Voxel = requires(T v, double x, double y, double z, float cx, float cy, float cz) {
    // Basic voxel members (check they exist and are accessible)
    v.count;
    v.position_sum;
    v.color_sum;

    // Basic voxel methods
    v.update_point(x, y, z);
    { v.get_position() };
    v.update_color(cx, cy, cz);
    { v.get_color() };
    v.reset();
};

// Concept for semantic voxel data types
// Checks that a type has all the required members and methods for semantic voxel functionality
// A semantic voxel must satisfy the Voxel concept plus have semantic-specific members and methods
template <typename T>
concept SemanticVoxel = Voxel<T> && requires(T v, int instance_id, int class_id) {
    // Semantic members (check they exist and are accessible)
    v.instance_id;
    v.class_id;
    v.confidence_counter;

    // Semantic methods
    v.initialize_semantics(instance_id, class_id);
    v.update_semantics(instance_id, class_id);
};

// Concept for semantic voxel data types with depth-based integration
template <typename T>
concept SemanticVoxelWithDepth =
    SemanticVoxel<T> && requires(T v, int instance_id, int class_id, float depth) {
        v.initialize_semantics_with_depth(instance_id, class_id, depth);
        v.update_semantics_with_depth(instance_id, class_id, depth);
    };

// Static assertions to verify the concepts work with the actual types
static_assert(Voxel<VoxelData>, "VoxelData must satisfy the Voxel concept");
static_assert(Voxel<VoxelSemanticData>, "VoxelSemanticData must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticData>,
              "VoxelSemanticData must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticData>,
              "VoxelSemanticData must satisfy the SemanticVoxelWithDepth concept");
static_assert(Voxel<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the SemanticVoxelWithDepth concept");

// Helper trait to detect if get_confidence_counter() exists
template <typename T, typename = void> struct has_get_confidence_counter : std::false_type {};
template <typename T>
struct has_get_confidence_counter<
    T, std::void_t<decltype(std::declval<const T &>().get_confidence_counter())>> : std::true_type {
};
template <typename T>
inline constexpr bool has_get_confidence_counter_v = has_get_confidence_counter<T>::value;

// Helper function to get confidence_counter, using getter if available (for probabilistic),
// otherwise using direct member access (for non-probabilistic)
// This uses SFINAE (Substitution Failure Is Not An Error) to prefer get_confidence_counter() when
// available
template <typename T>
auto get_confidence_counter_value(const T &v) -> decltype(v.get_confidence_counter(), int()) {
    return v.get_confidence_counter();
}
// Fallback: only matches when get_confidence_counter() doesn't exist
template <typename T>
auto get_confidence_counter_value(const T &v)
    -> std::enable_if_t<!has_get_confidence_counter_v<T>, int> {
    return v.confidence_counter;
}

#undef VOXEL_POSITION_MEMBERS
#undef VOXEL_POSITION_METHODS
#undef VOXEL_COLOR_MEMBERS
#undef VOXEL_COLOR_METHODS

} // namespace volumetric