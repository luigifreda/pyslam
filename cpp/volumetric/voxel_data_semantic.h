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

#include "voxel_data.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>

#include <atomic>

// NOTE: This file contains the definitions of the voxel data types and the concepts that they must
// satisfy.

namespace volumetric {

class VoxelSemanticSharedData {
  public:
    inline static std::atomic<int32_t> next_object_id{
        1}; // next object ID (starts at 1, since 0 is reserved for "no specific object")

    VoxelSemanticSharedData() = default;

    static int32_t get_next_object_id() { return next_object_id.fetch_add(1); }
};

//=================================================================================================
// Voxel Semantic Data Comparison Notes
//=================================================================================================
// VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2 are two different approaches
// to semantic voxel data storage and update.
// -VoxelSemanticDataProbabilistic uses a joint probability distribution for the object and class
// IDs.
// - VoxelSemanticDataProbabilistic2 uses separate marginal probability distributions for
//   the object and class IDs.
// - VoxelSemanticDataProbabilistic is more efficient and has a better behavior.
//
// The issue is the independence assumption in VoxelSemanticDataProbabilistic2.
// - VoxelSemanticDataProbabilistic: stores the joint distribution P(object_id, class_id) directly.
//  Observing (5, 10) updates P(5, 10) directly, preserving correlations.
// - VoxelSemanticDataProbabilistic2: assumes independence P(obj, cls) = P(obj) * P(cls) and splits
//  updates. Observing (5, 10) with log_prob L adds L/2 to both P(5) and P(10). This dilutes the
//  signal when marginals are shared across pairs.
//  - Example: Observing (5, 10) 10 times should make P(5, 10) high.
//   With independence, P(5) and P(10) both increase, but if P(5) is also boosted by (5, 20), (5,
//   30), etc., and P(10) by (1, 10), (2, 10), etc., then P(5, 10) = P(5) * P(10) after
//   normalization can be diluted and not reflect the 10 observations.
//  P(10) both increase, but if P(5) is also boosted by (5, 20), (5, 30), etc., and P(10) by (1,
//  10), (2, 10), etc., then P(5, 10) = P(5) * P(10) after normalization can be diluted and not
//  reflect the 10 observations.
// - Normalization: the joint normalization sums over all pairs (obj, cls), which can make joint
//   probabilities very small even when marginals are high.
// - Checking how normalization is computed:
//   + The normalization in VoxelSemanticDataProbabilistic2 normalizes over all possible pairs (obj,
//   cls), not just observed ones. This can make joint probabilities artificially small.
//   + Example: You observe (5, 10) 100 times You also observe (6, 11) once and (7, 12) once
//   VoxelSemanticDataProbabilistic2 normalizes over all 9 possible pairs: (5,10), (5,11), (5,12),
//   (6,10), (6,11), (6,12), (7,10), (7,11), (7,12) Even though (5,10) dominates, its normalized
//   probability is diluted by the other 8 pairs In contrast, VoxelSemanticDataProbabilistic only
//   normalizes over the 3 pairs that were actually observed, giving (5,10) the correct high
//   probability.
// Summary:
// The main issues with VoxelSemanticDataProbabilistic2 are:
// - Independence assumption: object_id and class_id are often correlated, so the factorization
// P(obj, cls) = P(obj) * P(cls) is incorrect.
// - Update splitting: splitting log probabilities in half dilutes the signal when marginals are
// shared across pairs. -Over-normalization: normalizing over all possible pairs (not just observed
// ones) makes probabilities smaller than they should be.

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
// NOTE:
//      Semantics:
//          - class_id=0 -> background, class_id>0 -> actual class
//          - object_id=0 -> no specific object, object_id>0 -> specific object
struct VoxelSemanticData {
    inline static float kDepthThreshold =
        10.0f; // [m] depth threshold for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    int get_confidence_counter() const { return confidence_counter_; }
    float get_confidence() const {
        if (count == 0)
            return 0.0f;
        // confidence_counter_ represents the "strength" of the current label after voting.
        // It's incremented for matching observations and decremented for non-matching ones.
        // When counter <= 0, the label switches and counter resets to 1.
        // Since counter >= 0, it represents net votes for current label since last reset.
        //
        // Confidence is computed as the ratio of counter to total observations.
        // - Range: [0, 1] (clamped)
        // - When counter = count: confidence = 1.0 (all observations match current label)
        // - When counter << count: confidence is low (many conflicting observations)
        const float count_f = static_cast<float>(count);
        const float counter_f = static_cast<float>(confidence_counter_);
        return std::min(1.0f, counter_f / count_f);
    }

    int get_object_id() const { return object_id; }
    int get_class_id() const { return class_id; }

    // Setters for semantic data (needed for operations like merge_segments)
    void set_object_id(int id) { object_id = id; }
    void set_class_id(int id) { class_id = id; }
    void set_confidence_counter(int counter) { confidence_counter_ = counter; }

  protected:
    int object_id = -1;          // object ID
    int class_id = -1;           // class ID
    int confidence_counter_ = 0; // confidence counter for the instance and class IDs

  public:
    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        object_id = -1;
        class_id = -1;
        confidence_counter_ = 0;
    }

    void initialize_semantics(const int object_id, const int class_id) {
        this->object_id = object_id;
        this->class_id = class_id;
        this->confidence_counter_ = 1;
    }

    void initialize_semantics_with_depth(const int object_id, const int class_id,
                                         const float depth) {

        if (depth < kDepthThreshold) {
            initialize_semantics(object_id, class_id);
        }
    }

    void update_semantics(const int object_id, const int class_id) {

        if (this->object_id == object_id && this->class_id == class_id) {
            // Update confidence counter if the instance and class IDs are the same
            this->confidence_counter_++;
        } else {
            // If the instance and class IDs are different, decrement the confidence counter
            this->confidence_counter_--;
            if (this->confidence_counter_ <= 0) {
                // If the confidence counter is less than or equal to 0, set the instance
                // and class IDs and reset the confidence counter to 1
                this->object_id = object_id;
                this->class_id = class_id;
                this->confidence_counter_ = 1;
            }
        }
    }

    void update_semantics_with_depth(const int object_id, const int class_id, const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => confidence_counter = 1
        if (depth < kDepthThreshold) {
            update_semantics(object_id, class_id);
        }
    }
};

//=================================================================================================
// Voxel Semantic Data 2
// (Separate confidence counters for object and class IDs, worse behavior than VoxelSemanticData)
// **NOTE**: Read the comparison note above concerning the differences between
// VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2.
//=================================================================================================

// Semantic voxel data structure for storing points, colors, instance IDs, and class IDs.
// The semantic update is a simple voting mechanism.
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
// - Instance IDs and class IDs are used to store the most observed instance and class IDs of the
// voxel.
// - Separate confidence counters are used to independently track the confidence of the most
// observed/voted instance ID and class ID. Each counter is incremented when its respective ID
// matches and decremented when it differs. If a confidence counter is less than or equal to 0, the
// corresponding ID is set to the new value and its confidence counter is reset to 1.
// **NOTE**: Read the comparison note above concerning the differences between
// VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2.
// NOTE:
//      Semantics:
//          - class_id=0 -> background, class_id>0 -> actual class
//          - object_id=0 -> no specific object, object_id>0 -> specific object
struct VoxelSemanticData2 {
    inline static float kDepthThreshold =
        10.0f; // [m] depth threshold for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    // Get combined confidence counter (for backward compatibility)
    int get_confidence_counter() const {
        return std::min(object_confidence_counter_, class_confidence_counter_);
    }

    // Get confidence for object_id
    float get_object_confidence() const {
        if (count == 0)
            return 0.0f;
        const float count_f = static_cast<float>(count);
        const float counter_f = static_cast<float>(object_confidence_counter_);
        return std::min(1.0f, counter_f / count_f);
    }

    // Get confidence for class_id
    float get_class_confidence() const {
        if (count == 0)
            return 0.0f;
        const float count_f = static_cast<float>(count);
        const float counter_f = static_cast<float>(class_confidence_counter_);
        return std::min(1.0f, counter_f / count_f);
    }

    // Get combined confidence (minimum of object and class confidence)
    float get_confidence() const {
        if (count == 0)
            return 0.0f;
        return std::min(get_object_confidence(), get_class_confidence());
    }

    int get_object_id() const { return object_id; }
    int get_class_id() const { return class_id; }

    // Setters for semantic data (needed for operations like merge_segments)
    void set_object_id(int id) { object_id = id; }
    void set_class_id(int id) { class_id = id; }
    // Set both confidence counters (for backward compatibility)
    void set_confidence_counter(int counter) {
        object_confidence_counter_ = counter;
        class_confidence_counter_ = counter;
    }

  protected:
    int object_id = -1;                 // object ID
    int class_id = -1;                  // class ID
    int object_confidence_counter_ = 0; // confidence counter for the object ID
    int class_confidence_counter_ = 0;  // confidence counter for the class ID

  public:
    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        object_id = -1;
        class_id = -1;
        object_confidence_counter_ = 0;
        class_confidence_counter_ = 0;
    }

    void initialize_semantics(const int object_id, const int class_id) {
        this->object_id = object_id;
        this->class_id = class_id;
        this->object_confidence_counter_ = 1;
        this->class_confidence_counter_ = 1;
    }

    void initialize_semantics_with_depth(const int object_id, const int class_id,
                                         const float depth) {

        if (depth < kDepthThreshold) {
            initialize_semantics(object_id, class_id);
        }
    }

    void update_semantics(const int object_id, const int class_id) {
        // Update object_id confidence independently
        if (this->object_id == object_id) {
            // Object ID matches: increment confidence counter
            this->object_confidence_counter_++;
        } else {
            // Object ID differs: decrement confidence counter
            this->object_confidence_counter_--;
            if (this->object_confidence_counter_ <= 0) {
                // Confidence counter <= 0: switch to new object ID and reset counter
                this->object_id = object_id;
                this->object_confidence_counter_ = 1;
            }
        }

        // Update class_id confidence independently
        if (this->class_id == class_id) {
            // Class ID matches: increment confidence counter
            this->class_confidence_counter_++;
        } else {
            // Class ID differs: decrement confidence counter
            this->class_confidence_counter_--;
            if (this->class_confidence_counter_ <= 0) {
                // Confidence counter <= 0: switch to new class ID and reset counter
                this->class_id = class_id;
                this->class_confidence_counter_ = 1;
            }
        }
    }

    void update_semantics_with_depth(const int object_id, const int class_id, const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => confidence_counter = 1
        if (depth < kDepthThreshold) {
            update_semantics(object_id, class_id);
        }
    }
};

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
//  Let L = (i,c) be a label pair (object_id, class_id)
//  We have a prior distribution p(L|old) after having observed n points in the voxel.
//  When we get a new observation x_new (one more point measurement in the voxel), Bayes update
//  says: p(L|x_new, old) = p(x_new|L,old) * p(L|old) / p(x_new|old) Assuming conditional
//  independence of the observations given the label (the usual "naive Bayes" assumption), we have:
//  P(x_new|L,old) = P(x_new|L)
//  Therefore:
//  p(L|x_new, old) = p(x_new|L) * p(L|old) / sum_i {p(x_new|L_i,old) * p(L_i|old)}
//  Simplifying the notation, we get:
//  p(new) = likelihood(new) * prior(old) / normalization
//  where likelihood(new) = p(x_new|L) is the "likelihood contribution" of the new observation to
//  the label L and prior(old) = p(L|old)
//
// Efficiency features:
// - Sparse storage: only stores probabilities for observed (object_id, class_id) pairs
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
// - Access most likely label via get_object_id(), get_class_id(), or get_most_likely_label()
// - For direct member access (object_id, class_id, confidence_counter), call
// ensure_cache_updated() first
//   or use the getter methods which handle this automatically
//
// **NOTE**: Read the comparison note above concerning the differences between
// VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2.
// NOTE:
//      Semantics:
//          - class_id=0 -> background, class_id>0 -> actual class
//          - object_id=0 -> no specific object, object_id>0 -> specific object
struct VoxelSemanticDataProbabilistic {

    inline static float kDepthThreshold =
        5.0f; // [m] depth threshold for updating semantics with depth
    inline static float kDepthDecayRate =
        0.07f; // [1/m] depth decay rate for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    // Sparse probability distribution: (object_id, class_id) -> log_probability
    // Using log probabilities avoids numerical underflow and makes multiplication additive
    // std::map is used instead of std::unordered_map for better performance with few observations
    // (std::pair<int, int> is naturally comparable, so no custom comparator needed)
    std::map<std::pair<int, int>, float> log_probabilities;

    // Cached most likely label pair (updated lazily), (object_id, class_id) pair
    mutable std::pair<int, int> most_likely_pair = {-1, -1};
    mutable float most_likely_log_prob = -std::numeric_limits<float>::infinity();
    mutable bool cache_valid = false;

  protected:
    // Direct access members for compatibility (lazy evaluation via getters)
    // Note: These are mutable to allow const access patterns
    mutable int object_id = -1;       // cached most likely object ID
    mutable int class_id = -1;        // cached most likely class ID
    mutable float confidence_ = 0.0f; // cached confidence

  public:
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
        object_id = -1;
        class_id = -1;
        confidence_ = 0.0f;
    }

    void initialize_semantics_log_prob(const int object_id, const int class_id,
                                       const float log_prob) {
        const auto key = std::make_pair(object_id, class_id);
        // Initialize with log(confidence), assuming uniform prior
        // Using log(confidence) as initial log probability
        // const float log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));

        log_probabilities[key] = log_prob;
        most_likely_pair = key;
        most_likely_log_prob = log_prob;
        cache_valid = true;
        update_cached_members();
    }

    void initialize_semantics(const int object_id, const int class_id,
                              const float confidence = 1.0f) {
        // Initialize with log(confidence), assuming uniform prior
        // Using log(confidence) as initial log probability
        const float log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        initialize_semantics_log_prob(object_id, class_id, log_prob);
    }

    void initialize_semantics_with_depth(const int object_id, const int class_id,
                                         const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;
        initialize_semantics_log_prob(object_id, class_id, log_prob);
    }

    void update_semantics_log_prob(const int object_id, const int class_id,
                                   const float new_log_prob) {

        // const float new_log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));

        const auto key = std::make_pair(object_id, class_id);
        // Bayesian fusion in log space: log(P(new|old)) = log(P(new)) + log(P(old)) -
        // log(normalization) For efficiency, we accumulate log probabilities and normalize
        // periodically Here we use a simple additive update: log_prob += log(confidence) This
        // approximates Bayesian fusion when we normalize by the total count
        // Use operator[] for cleaner code - inserts 0.0f if key doesn't exist (log(1) = 0)
        // For a new key, this gives us: 0.0f + new_log_prob = new_log_prob (correct for first
        // observation)
        auto it = log_probabilities.find(key);
        if (it == log_probabilities.end()) {
            // New label pair: initialize with the observation's log probability
            log_probabilities[key] = new_log_prob;
            // Update cache if this new label might be the most likely
            if (cache_valid && new_log_prob > most_likely_log_prob) {
                most_likely_log_prob = new_log_prob;
                most_likely_pair = key;
                update_cached_members();
            } else if (!cache_valid) {
                update_cache();
            }
        } else {
            // Existing label pair: accumulate log probability
            float &log_prob = it->second;
            log_prob += new_log_prob;

            // Incremental cache maintenance: keep cached values in sync and avoid full recompute
            if (cache_valid) {
                if (key == most_likely_pair) {
                    // Still the argmax; value changed (could increase or decrease)
                    const float old_most_likely_log_prob = most_likely_log_prob;
                    most_likely_log_prob = log_prob;
                    if (log_prob < old_most_likely_log_prob) {
                        // Max decreased; need full recomputation to verify the argmax
                        update_cache();
                    } else {
                        update_cached_members();
                    }
                } else if (log_prob > most_likely_log_prob) {
                    // New argmax found
                    most_likely_log_prob = log_prob;
                    most_likely_pair = key;
                    update_cached_members();
                }
                // If this label is not the new max, cache remains valid
            } else {
                // Cache invalid; recompute now to keep direct member access consistent
                update_cache();
            }
        }
    }

    void update_semantics(const int object_id, const int class_id, const float confidence = 1.0f) {
        const float new_log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        update_semantics_log_prob(object_id, class_id, new_log_prob);
    }

    void update_semantics_with_depth(const int object_id, const int class_id, const float depth) {

        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float new_log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;

        update_semantics_log_prob(object_id, class_id, new_log_prob);
    }

    // Get the most likely (object_id, class_id) pair
    std::pair<int, int> get_most_likely_label() const {
        if (!cache_valid) {
            update_cache();
        }
        return most_likely_pair;
    }

    // Get the object_id of the most likely label
    int get_object_id() const {
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

    // Setters for semantic data (needed for operations like merge_segments)
    // Note: For probabilistic voxels, these update the cached values but don't modify
    // the underlying probability distribution. Use with caution.
    void set_object_id(int id) {
        ensure_cache_updated();
        object_id = id;
        most_likely_pair.first = id;
    }
    void set_class_id(int id) {
        ensure_cache_updated();
        class_id = id;
        most_likely_pair.second = id;
    }
    void set_confidence_counter(int counter) {
        ensure_cache_updated();
        if (count == 0) {
            confidence_ = 0.0f;
        } else {
            confidence_ = counter / static_cast<float>(count);
        }
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
        // Reuse confidence_ to avoid code duplication
        return static_cast<int>(confidence_ * static_cast<float>(count));
    }

    // Get confidence (compatible interface) - returns the confidence for the most likely label
    float get_confidence() const {
        if (!cache_valid) {
            update_cache();
        }
        return confidence_;
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

    // Accessor that updates cached values (for compatibility with direct member access)
    void ensure_cache_updated() const {
        if (!cache_valid) {
            update_cache();
        }
        update_cached_members();
    }

  private:
    // Helper to compute and store cached members based on current argmax
    void update_cached_members() const {
        const_cast<VoxelSemanticDataProbabilistic *>(this)->object_id = most_likely_pair.first;
        const_cast<VoxelSemanticDataProbabilistic *>(this)->class_id = most_likely_pair.second;
        const_cast<VoxelSemanticDataProbabilistic *>(this)->confidence_ = compute_confidence();
    }

    // Compute the confidence for the most likely label
    float compute_confidence() const {
        if (most_likely_pair.first == -1 || most_likely_pair.second == -1) {
            return 0;
        }
        if (log_probabilities.empty()) {
            return 0;
        }
        const float normalized_prob = std::exp(most_likely_log_prob - get_log_normalization());
        return normalized_prob;
    }

    // Update the cache of most likely label
    void update_cache() const {
        if (log_probabilities.empty()) {
            most_likely_pair = {-1, -1};
            most_likely_log_prob = -std::numeric_limits<float>::infinity();
            cache_valid = true;
            update_cached_members();
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
        update_cached_members();
    }

    // Compute log of normalization constant (log-sum-exp trick for numerical stability)
    float get_log_normalization() const {
        if (log_probabilities.empty()) {
            return 0.0f;
        }

        // Optimize: use cached most_likely_log_prob as max when cache is valid
        // This avoids recomputing the max, which is already known
        float max_log_prob;
        if (cache_valid) {
            max_log_prob = most_likely_log_prob;
        } else {
            // Find max log probability (fallback when cache is invalid)
            max_log_prob = -std::numeric_limits<float>::infinity();
            for (const auto &[key, log_prob] : log_probabilities) {
                if (log_prob > max_log_prob) {
                    max_log_prob = log_prob;
                }
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
        bool pruned_most_likely = false;
        for (auto it = log_probabilities.begin(); it != log_probabilities.end();) {
            if (it->second < thresh) {
                if (it->first == most_likely_pair) {
                    pruned_most_likely = true;
                }
                it = log_probabilities.erase(it);
            } else {
                ++it;
            }
        }
        // Invalidate cache if the most likely pair was pruned or if map is empty
        if (pruned_most_likely || log_probabilities.empty()) {
            cache_valid = false;
        }
        // norm_valid = false; // if we cache logZ
    }
};

//=================================================================================================
// Voxel Semantic Data Probabilistic 2
// (Separate marginal probability distributions for object and class IDs, worse behavior than
// VoxelSemanticDataProbabilistic)
// **NOTE**: Read the comparison note above concerning the differences between
//          VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2.
//=================================================================================================

// Semantic voxel data structure with probabilistic fusion
// Uses Bayesian fusion in log-space for efficient and numerically stable updates
// - Points and colors are integrated into the voxel and then the average position and color are
// computed.
// - Instance IDs and class IDs are used to store the most likely label pair of the voxel.
//
// Probabilistic update (ideal Bayesian fusion):
//  Let L = (i,c) be a label pair (object_id, class_id)
//  We have a prior distribution p(L|old) after having observed n points in the voxel.
//  When we get a new observation x_new (one more point measurement in the voxel), Bayes update
//  says: p(L|x_new, old) = p(x_new|L,old) * p(L|old) / p(x_new|old) Assuming conditional
//  independence of the observations given the label (the usual "naive Bayes" assumption), we have:
//  P(x_new|L,old) = P(x_new|L)
//  Therefore:
//  p(L|x_new, old) = p(x_new|L) * p(L|old) / sum_i {p(x_new|L_i,old) * p(L_i|old)}
//  Simplifying the notation, we get:
//  p(new) = likelihood(new) * prior(old) / normalization
//  where likelihood(new) = p(x_new|L) is the "likelihood contribution" of the new observation to
//  the label L and prior(old) = p(L|old)
//
// Efficiency features:
// - Sparse marginal storage: stores separate distributions for object_id and class_id
//   (assumes independence: P(object_id, class_id) = P(object_id) * P(class_id))
// - Log-space arithmetic: avoids numerical underflow, multiplication becomes addition
// - Lazy evaluation: most likely label is cached and recomputed only when needed
// - Memory efficient: O(m+n) storage where m = unique object_ids, n = unique class_ids
//   (typically much less than O(m*n) for joint distribution)
// - O(1) marginal queries: direct lookup instead of O(k) iteration
// - Uses std::map instead of std::unordered_map for better performance with few observations
//   (typically 1-5 unique IDs per voxel, where std::map's O(log n) is effectively constant
//   and has lower overhead than hash tables)
//
// Important semantic note on confidence:
// - get_confidence() returns the normalized joint probability P(argmax_obj, argmax_cls), NOT a
//   traditional "vote counter / count" style confidence like VoxelSemanticData uses.
// - This represents the probability that the most likely (object_id, class_id) pair is correct
//   under the factorized independence model, which can behave differently from simple voting.
// - Use get_object_confidence() or get_class_confidence() for marginal probabilities.
//
// Usage:
// - Call initialize_semantics() for the first observation
// - Call update_semantics() for subsequent observations (optionally with confidence scores)
// - Access most likely label via get_object_id(), get_class_id(), or get_most_likely_label()
// - For direct member access (object_id, class_id, confidence_counter), call
// ensure_cache_updated() first
//   or use the getter methods which handle this automatically
//
// **NOTE**: Read the comparison note above concerning the differences between
// VoxelSemanticDataProbabilistic and VoxelSemanticDataProbabilistic2.
// NOTE:
//      Semantics:
//          - class_id=0 -> background, class_id>0 -> actual class
//          - object_id=0 -> no specific object, object_id>0 -> specific object
struct VoxelSemanticDataProbabilistic2 {

    inline static float kDepthThreshold =
        5.0f; // [m] depth threshold for updating semantics with depth
    inline static float kDepthDecayRate =
        0.07f; // [1/m] depth decay rate for updating semantics with depth

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS()
    VOXEL_COLOR_MEMBERS()

    // Separate marginal probability distributions for object_id and class_id
    // Using log probabilities avoids numerical underflow and makes multiplication additive
    // std::map is used instead of std::unordered_map for better performance with few observations
    // (typically 1-5 unique IDs per voxel, where std::map's O(log n) is effectively constant)
    // Assumes independence: P(object_id, class_id) = P(object_id) * P(class_id)
    // This is more memory efficient (O(m+n) vs O(m*n)) and enables O(1) marginal queries
    std::map<int, float> object_log_probabilities; // object_id -> log_probability
    std::map<int, float> class_log_probabilities;  // class_id -> log_probability

    // Cached most likely values (updated lazily) - stored separately for efficiency
    // Since we assume independence, most likely pair is (most_likely_object_id,
    // most_likely_class_id)
    mutable int most_likely_object_id = -1;
    mutable float most_likely_object_log_prob = -std::numeric_limits<float>::infinity();
    mutable bool object_cache_valid = false;

    mutable int most_likely_class_id = -1;
    mutable float most_likely_class_log_prob = -std::numeric_limits<float>::infinity();
    mutable bool class_cache_valid = false;

  protected:
    // Direct access members for compatibility (lazy evaluation via getters)
    // Note: These are mutable to allow const access patterns
    mutable int object_id = -1;       // cached most likely object ID
    mutable int class_id = -1;        // cached most likely class ID
    mutable float confidence_ = 0.0f; // cached confidence

  public:
    static constexpr float MIN_CONFIDENCE = 1e-10f;

    VOXEL_POSITION_METHODS()
    VOXEL_COLOR_METHODS()

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        object_log_probabilities.clear();
        class_log_probabilities.clear();
        most_likely_object_id = -1;
        most_likely_object_log_prob = -std::numeric_limits<float>::infinity();
        object_cache_valid = false;
        most_likely_class_id = -1;
        most_likely_class_log_prob = -std::numeric_limits<float>::infinity();
        class_cache_valid = false;
        // Reset cached members for compatibility
        object_id = -1;
        class_id = -1;
        confidence_ = 0.0f;
    }

    void initialize_semantics_log_prob(const int object_id, const int class_id,
                                       const float log_prob) {
        // Initialize marginal distributions independently
        // Assuming independence: P(obj, cls) = P(obj) * P(cls)
        // To get joint probability exp(log_prob), we split it: log P(obj) = log_prob/2, log P(cls)
        // = log_prob/2 This ensures: log P(obj, cls) = log P(obj) + log P(cls) = log_prob
        const float half_log_prob = log_prob * 0.5f;
        object_log_probabilities[object_id] = half_log_prob;
        class_log_probabilities[class_id] = half_log_prob;

        // Initialize caches
        most_likely_object_id = object_id;
        most_likely_object_log_prob = half_log_prob;
        object_cache_valid = true;
        most_likely_class_id = class_id;
        most_likely_class_log_prob = half_log_prob;
        class_cache_valid = true;
        update_cached_members();
    }

    void initialize_semantics(const int object_id, const int class_id,
                              const float confidence = 1.0f) {
        // Initialize with log(confidence), assuming uniform prior
        // Using log(confidence) as initial log probability
        const float log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        initialize_semantics_log_prob(object_id, class_id, log_prob);
    }

    void initialize_semantics_with_depth(const int object_id, const int class_id,
                                         const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;
        initialize_semantics_log_prob(object_id, class_id, log_prob);
    }

    void update_semantics_log_prob(const int object_id, const int class_id,
                                   const float new_log_prob) {
        // Update marginal distributions independently
        // Assuming independence: P(obj, cls) = P(obj) * P(cls)
        // To get joint probability exp(new_log_prob), we split it: add new_log_prob/2 to each
        // This ensures: log P(obj, cls) = log P(obj) + log P(cls) = new_log_prob
        const float half_log_prob = new_log_prob * 0.5f;

        // Update object_id marginal: accumulate log probability
        auto obj_it = object_log_probabilities.find(object_id);
        if (obj_it == object_log_probabilities.end()) {
            // New object_id: initialize with half the observation's log probability
            object_log_probabilities[object_id] = half_log_prob;
        } else {
            // Existing object_id: accumulate half the log probability
            obj_it->second += half_log_prob;
        }

        // Update class_id marginal: accumulate log probability
        auto cls_it = class_log_probabilities.find(class_id);
        if (cls_it == class_log_probabilities.end()) {
            // New class_id: initialize with half the observation's log probability
            class_log_probabilities[class_id] = half_log_prob;
        } else {
            // Existing class_id: accumulate half the log probability
            cls_it->second += half_log_prob;
        }

        // Invalidate caches since distributions changed
        // We invalidate both since we don't know which changed more significantly
        // In the future, we could track which distribution changed and only invalidate that one
        object_cache_valid = false;
        class_cache_valid = false;
    }

    void update_semantics(const int object_id, const int class_id, const float confidence = 1.0f) {
        const float new_log_prob = std::log(std::max(confidence, MIN_CONFIDENCE));
        update_semantics_log_prob(object_id, class_id, new_log_prob);
    }

    void update_semantics_with_depth(const int object_id, const int class_id, const float depth) {

        // if depth < kDepthThreshold, the confidence is 1.0 => log_prob = 0.0
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate) => log_prob = -depth_diff *
        // kDepthDecayRate
        const float new_log_prob =
            depth <= kDepthThreshold ? 0.0f : -(depth - kDepthThreshold) * kDepthDecayRate;

        update_semantics_log_prob(object_id, class_id, new_log_prob);
    }

    // Get the most likely (object_id, class_id) pair
    std::pair<int, int> get_most_likely_label() const {
        ensure_cache_updated();
        return std::make_pair(most_likely_object_id, most_likely_class_id);
    }

    // Get the object_id of the most likely label
    int get_object_id() const {
        if (!object_cache_valid) {
            update_object_cache();
        }
        return most_likely_object_id;
    }

    // Get the class_id of the most likely label
    int get_class_id() const {
        if (!class_cache_valid) {
            update_class_cache();
        }
        return most_likely_class_id;
    }

    // Setters for semantic data (needed for operations like merge_segments)
    // Note: For probabilistic voxels, these update the cached values and underlying distributions
    // to ensure consistency. The set_object_id/set_class_id methods update the distributions
    // to make the specified ID the most likely, while set_confidence_counter is a no-op
    // (confidence is computed from the distribution).
    void set_object_id(int id) {
        // Update the cache first to get current state
        if (!object_cache_valid) {
            update_object_cache();
        }
        // Update the distribution: set the specified object_id to have the same log_prob as current
        // max This makes it the most likely (or tied for most likely)
        if (!object_log_probabilities.empty() && most_likely_object_id != -1 &&
            most_likely_object_log_prob != -std::numeric_limits<float>::infinity()) {
            const float current_max_log_prob = most_likely_object_log_prob;
            object_log_probabilities[id] = current_max_log_prob;
            most_likely_object_id = id;
            most_likely_object_log_prob = current_max_log_prob;
            object_cache_valid = true;
        } else {
            // No existing distribution or invalid state, initialize with default
            // Use 0.0f as default log probability (exp(0) = 1.0 in linear space)
            object_log_probabilities[id] = 0.0f;
            most_likely_object_id = id;
            most_likely_object_log_prob = 0.0f;
            object_cache_valid = true;
        }
        object_id = id;
        // Invalidate class cache since joint probability might have changed
        class_cache_valid = false;
        // Ensure both caches are valid before updating cached members to avoid transient
        // inconsistency
        ensure_cache_updated();
    }
    void set_class_id(int id) {
        // Update the cache first to get current state
        if (!class_cache_valid) {
            update_class_cache();
        }
        // Update the distribution: set the specified class_id to have the same log_prob as current
        // max This makes it the most likely (or tied for most likely)
        if (!class_log_probabilities.empty() && most_likely_class_id != -1 &&
            most_likely_class_log_prob != -std::numeric_limits<float>::infinity()) {
            const float current_max_log_prob = most_likely_class_log_prob;
            class_log_probabilities[id] = current_max_log_prob;
            most_likely_class_id = id;
            most_likely_class_log_prob = current_max_log_prob;
            class_cache_valid = true;
        } else {
            // No existing distribution or invalid state, initialize with default
            // Use 0.0f as default log probability (exp(0) = 1.0 in linear space)
            class_log_probabilities[id] = 0.0f;
            most_likely_class_id = id;
            most_likely_class_log_prob = 0.0f;
            class_cache_valid = true;
        }
        class_id = id;
        // Invalidate object cache since joint probability might have changed
        object_cache_valid = false;
        // Ensure both caches are valid before updating cached members to avoid transient
        // inconsistency
        ensure_cache_updated();
    }
    void set_confidence_counter(int counter) {
        // For probabilistic voxels, confidence is computed from the distribution, not set directly
        // This setter is a no-op to maintain interface compatibility
        // The confidence will be recomputed on the next get_confidence() call
        (void)counter; // Suppress unused parameter warning
    }

    // Get the probability of the most likely label (in linear space)
    float get_most_likely_probability() const {
        ensure_cache_updated();
        if (object_log_probabilities.empty() || class_log_probabilities.empty()) {
            return 0.0f;
        }
        // Normalize probabilities and return the max
        // Joint probability: P(obj, cls) = P(obj) * P(cls) in log space: log P = log P(obj) + log
        // P(cls)
        const float joint_log_prob = most_likely_object_log_prob + most_likely_class_log_prob;
        return std::exp(joint_log_prob - get_log_normalization());
    }

    // Get confidence counter (compatible interface) - returns the count for the most likely label
    // NOTE: This is a compatibility method that converts the normalized joint probability
    // (confidence_) back to a "counter" value by multiplying by count. This is NOT the same as
    // the vote counter in VoxelSemanticData - it's a derived value from the probability
    // distribution.
    int get_confidence_counter() const {
        ensure_cache_updated();
        // Reuse confidence_ to avoid code duplication
        // confidence_ is the normalized joint probability, so we convert it to a "counter" value
        return static_cast<int>(confidence_ * static_cast<float>(count));
    }

    // Get confidence (compatible interface) - returns the confidence for the most likely label
    // NOTE: This returns the normalized joint probability P(argmax_obj, argmax_cls) under the
    // factorized model, NOT a traditional "vote counter / count" style confidence.
    // This is the probability that the most likely (object_id, class_id) pair is correct,
    // computed as: P(obj, cls) = P(obj) * P(cls) after normalization.
    // This can behave differently from simple voting-based confidence, especially when marginals
    // have multiple plausible labels. Use get_object_confidence() or get_class_confidence() for
    // marginal probabilities.
    float get_confidence() const {
        ensure_cache_updated();
        return confidence_;
    }

    // Get confidence for object_id (marginal probability of the most likely object_id)
    float get_object_confidence() const {
        if (!object_cache_valid) {
            update_object_cache();
        }
        if (object_log_probabilities.empty() || most_likely_object_id == -1) {
            return 0.0f;
        }
        // Direct lookup: O(1) instead of O(k) iteration
        const auto it = object_log_probabilities.find(most_likely_object_id);
        if (it == object_log_probabilities.end()) {
            return 0.0f;
        }
        // Normalize by marginal sum only (not joint normalization)
        const float log_obj_norm = get_object_log_normalization();
        return std::exp(it->second - log_obj_norm);
    }

    // Get confidence for class_id (marginal probability of the most likely class_id)
    float get_class_confidence() const {
        if (!class_cache_valid) {
            update_class_cache();
        }
        if (class_log_probabilities.empty() || most_likely_class_id == -1) {
            return 0.0f;
        }
        // Direct lookup: O(1) instead of O(k) iteration
        const auto it = class_log_probabilities.find(most_likely_class_id);
        if (it == class_log_probabilities.end()) {
            return 0.0f;
        }
        // Normalize by marginal sum only (not joint normalization)
        const float log_cls_norm = get_class_log_normalization();
        return std::exp(it->second - log_cls_norm);
    }

    // For compatibility with existing interface - these are computed on access
    // Accessing these will trigger cache update if needed
    int get_instance_id_member() const {
        if (!object_cache_valid) {
            update_object_cache();
        }
        return most_likely_object_id;
    }

    int get_class_id_member() const {
        if (!class_cache_valid) {
            update_class_cache();
        }
        return most_likely_class_id;
    }

    // Accessor that updates cached values (for compatibility with direct member access)
    void ensure_cache_updated() const {
        if (!object_cache_valid) {
            update_object_cache();
        }
        if (!class_cache_valid) {
            update_class_cache();
        }
        update_cached_members();
    }

  private:
    // Helper to compute and store cached members based on current argmax
    void update_cached_members() const {
        const_cast<VoxelSemanticDataProbabilistic2 *>(this)->object_id = most_likely_object_id;
        const_cast<VoxelSemanticDataProbabilistic2 *>(this)->class_id = most_likely_class_id;
        const_cast<VoxelSemanticDataProbabilistic2 *>(this)->confidence_ = compute_confidence();
    }

    // Compute the confidence for the most likely label
    // Returns the normalized joint probability P(argmax_obj, argmax_cls) under the factorized
    // independence model. This is NOT a traditional "vote counter / count" style confidence.
    // Semantics: Returns the probability that the most likely (object_id, class_id) pair is
    // correct, computed as: P(obj, cls) = P(obj) * P(cls) after normalization over all pairs.
    // This can be unintuitive if the marginals have multiple plausible labels, as it represents
    // the probability of the specific pair (argmax_obj, argmax_cls), not a general "confidence"
    // in the labeling.
    float compute_confidence() const {
        if (most_likely_object_id == -1 || most_likely_class_id == -1) {
            return 0;
        }
        if (object_log_probabilities.empty() || class_log_probabilities.empty()) {
            return 0;
        }
        // Joint probability: P(obj, cls) = P(obj) * P(cls) in log space
        const float joint_log_prob = most_likely_object_log_prob + most_likely_class_log_prob;
        const float normalized_prob = std::exp(joint_log_prob - get_log_normalization());
        return normalized_prob;
    }

    // Update the cache for object_id (O(m) where m = number of unique object_ids)
    void update_object_cache() const {
        if (object_log_probabilities.empty()) {
            most_likely_object_id = -1;
            most_likely_object_log_prob = -std::numeric_limits<float>::infinity();
            object_cache_valid = true;
            return;
        }

        // Find argmax for object_id: O(m)
        int best_obj_id = -1;
        float best_obj_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[obj_id, obj_log_prob] : object_log_probabilities) {
            if (obj_log_prob > best_obj_log_prob) {
                best_obj_log_prob = obj_log_prob;
                best_obj_id = obj_id;
            }
        }

        most_likely_object_id = best_obj_id;
        most_likely_object_log_prob = best_obj_log_prob;
        object_cache_valid = true;
    }

    // Update the cache for class_id (O(n) where n = number of unique class_ids)
    void update_class_cache() const {
        if (class_log_probabilities.empty()) {
            most_likely_class_id = -1;
            most_likely_class_log_prob = -std::numeric_limits<float>::infinity();
            class_cache_valid = true;
            return;
        }

        // Find argmax for class_id: O(n)
        int best_cls_id = -1;
        float best_cls_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[cls_id, cls_log_prob] : class_log_probabilities) {
            if (cls_log_prob > best_cls_log_prob) {
                best_cls_log_prob = cls_log_prob;
                best_cls_id = cls_id;
            }
        }

        most_likely_class_id = best_cls_id;
        most_likely_class_log_prob = best_cls_log_prob;
        class_cache_valid = true;
    }

    // Compute log of normalization constant for object_id marginal distribution
    // Uses log-sum-exp trick for numerical stability
    float get_object_log_normalization() const {
        if (object_log_probabilities.empty()) {
            return 0.0f;
        }

        // Find max for numerical stability (first pass)
        float max_obj_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[obj_id, obj_log_prob] : object_log_probabilities) {
            if (obj_log_prob > max_obj_log_prob) {
                max_obj_log_prob = obj_log_prob;
            }
        }

        // Compute sum using log-sum-exp trick (second pass)
        // For small distributions (typically 1-5 elements), two passes are negligible
        float obj_sum = 0.0f;
        for (const auto &[obj_id, obj_log_prob] : object_log_probabilities) {
            obj_sum += std::exp(obj_log_prob - max_obj_log_prob);
        }
        return max_obj_log_prob + std::log(obj_sum);
    }

    // Compute log of normalization constant for class_id marginal distribution
    // Uses log-sum-exp trick for numerical stability
    float get_class_log_normalization() const {
        if (class_log_probabilities.empty()) {
            return 0.0f;
        }

        // Find max for numerical stability (first pass)
        float max_cls_log_prob = -std::numeric_limits<float>::infinity();
        for (const auto &[cls_id, cls_log_prob] : class_log_probabilities) {
            if (cls_log_prob > max_cls_log_prob) {
                max_cls_log_prob = cls_log_prob;
            }
        }

        // Compute sum using log-sum-exp trick (second pass)
        // For small distributions (typically 1-5 elements), two passes are negligible
        float cls_sum = 0.0f;
        for (const auto &[cls_id, cls_log_prob] : class_log_probabilities) {
            cls_sum += std::exp(cls_log_prob - max_cls_log_prob);
        }
        return max_cls_log_prob + std::log(cls_sum);
    }

    // Compute log of joint normalization constant (log-sum-exp trick for numerical stability)
    // For independent marginals: sum over all pairs (obj, cls) of exp(log P(obj) + log P(cls))
    // = sum_obj exp(log P(obj)) * sum_cls exp(log P(cls))
    // In log space: log(sum) = log(sum_obj) + log(sum_cls)
    float get_log_normalization() const {
        if (object_log_probabilities.empty() || class_log_probabilities.empty()) {
            return 0.0f;
        }

        // Joint normalization is sum of marginal normalizations
        return get_object_log_normalization() + get_class_log_normalization();
    }

    // Prune low probability labels from the probability distribution
    // Prunes labels where the joint probability P(obj) * P(cls) is below threshold
    void prune_low_prob(float margin = 5.0f /* nats */) {
        ensure_cache_updated();
        // Safety check: if distributions are empty or most likely is invalid, nothing to prune
        if (object_log_probabilities.empty() || class_log_probabilities.empty() ||
            most_likely_object_id == -1 || most_likely_class_id == -1) {
            return;
        }
        const float joint_log_prob = most_likely_object_log_prob + most_likely_class_log_prob;
        const float thresh = joint_log_prob - margin;
        bool pruned_most_likely_object = false;
        bool pruned_most_likely_class = false;

        // Prune object_ids: remove if max joint prob with any class is below threshold
        for (auto it = object_log_probabilities.begin(); it != object_log_probabilities.end();) {
            // Check if this object_id can form a pair with any class_id above threshold
            bool keep = false;
            for (const auto &[cls_id, cls_log_prob] : class_log_probabilities) {
                const float joint_log_prob = it->second + cls_log_prob;
                if (joint_log_prob >= thresh) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                if (it->first == most_likely_object_id) {
                    pruned_most_likely_object = true;
                }
                it = object_log_probabilities.erase(it);
            } else {
                ++it;
            }
        }

        // Prune class_ids: remove if max joint prob with any object is below threshold
        for (auto it = class_log_probabilities.begin(); it != class_log_probabilities.end();) {
            // Check if this class_id can form a pair with any object_id above threshold
            bool keep = false;
            for (const auto &[obj_id, obj_log_prob] : object_log_probabilities) {
                const float joint_log_prob = obj_log_prob + it->second;
                if (joint_log_prob >= thresh) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                if (it->first == most_likely_class_id) {
                    pruned_most_likely_class = true;
                }
                it = class_log_probabilities.erase(it);
            } else {
                ++it;
            }
        }

        // Invalidate caches if the most likely values were pruned or if distributions are empty
        if (pruned_most_likely_object || object_log_probabilities.empty()) {
            object_cache_valid = false;
        }
        if (pruned_most_likely_class || class_log_probabilities.empty()) {
            class_cache_valid = false;
        }
    }
};

//=================================================================================================
// Semantic Voxel Concepts
//=================================================================================================

// Concept for semantic voxel data types
// Checks that a type has all the required members and methods for semantic voxel functionality
// A semantic voxel must satisfy the Voxel concept plus have semantic-specific methods
template <typename T>
concept SemanticVoxel = Voxel<T> && requires(T v, int object_id, int class_id) {
    // Semantic getter methods (check they exist and are callable)
    v.get_object_id();
    v.get_class_id();
    v.get_confidence_counter();
    v.get_confidence();

    // Semantic methods
    v.initialize_semantics(object_id, class_id);
    v.update_semantics(object_id, class_id);

    // Check return types are convertible to int (using nested requires)
    requires std::is_convertible_v<decltype(v.get_object_id()), int>;
    requires std::is_convertible_v<decltype(v.get_class_id()), int>;
    requires std::is_convertible_v<decltype(v.get_confidence_counter()), int>;
    requires std::is_convertible_v<decltype(v.get_confidence()), float>;
};

// Concept for semantic voxel data types with depth-based integration
template <typename T>
concept SemanticVoxelWithDepth =
    SemanticVoxel<T> && requires(T v, int object_id, int class_id, float depth) {
        v.initialize_semantics_with_depth(object_id, class_id, depth);
        v.update_semantics_with_depth(object_id, class_id, depth);
    };

// Static assertions to verify the concepts work with the actual types
static_assert(Voxel<VoxelSemanticData>, "VoxelSemanticData must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticData>,
              "VoxelSemanticData must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticData>,
              "VoxelSemanticData must satisfy the SemanticVoxelWithDepth concept");

static_assert(Voxel<VoxelSemanticData2>, "VoxelSemanticData2 must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticData2>,
              "VoxelSemanticData2 must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticData2>,
              "VoxelSemanticData2 must satisfy the SemanticVoxelWithDepth concept");

static_assert(Voxel<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticDataProbabilistic>,
              "VoxelSemanticDataProbabilistic must satisfy the SemanticVoxelWithDepth concept");

static_assert(Voxel<VoxelSemanticDataProbabilistic2>,
              "VoxelSemanticDataProbabilistic2 must satisfy the Voxel concept");
static_assert(SemanticVoxel<VoxelSemanticDataProbabilistic2>,
              "VoxelSemanticDataProbabilistic2 must satisfy the SemanticVoxel concept");
static_assert(SemanticVoxelWithDepth<VoxelSemanticDataProbabilistic2>,
              "VoxelSemanticDataProbabilistic2 must satisfy the SemanticVoxelWithDepth concept");

#undef VOXEL_POSITION_MEMBERS
#undef VOXEL_POSITION_METHODS
#undef VOXEL_COLOR_MEMBERS
#undef VOXEL_COLOR_METHODS

} // namespace volumetric
