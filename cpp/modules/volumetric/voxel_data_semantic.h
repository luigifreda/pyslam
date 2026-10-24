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
#include "voxel_semantic_shared_data.h"

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

// IMPORTANT NOTE:
//      SEMANTICS:
//          - class_id =0 -> background,
//                     -1 -> invalid class,
//                     >0 -> actual class
//          - object_id =0 -> no specific object,
//                      -1 -> invalid object,
//                      >0 -> specific object
//     The code relies on < 0 checks, so object_id/instance_id should stay signed.

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
template <typename Tpos, typename Tcolor = float> struct VoxelSemanticDataT {
    inline static float kDepthThreshold =
        10.0f; // [m] depth threshold for updating semantics with depth

    VOXEL_DATA_USING_TYPE(Tpos, Tcolor)

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS(Tpos)
    VOXEL_COLOR_MEMBERS(Tcolor)

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
    VOXEL_POSITION_METHODS(Tpos)
    VOXEL_COLOR_METHODS(Tcolor)

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

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
using VoxelSemanticData = VoxelSemanticDataT<double, float>;
using VoxelSemanticDataD = VoxelSemanticDataT<double, float>;
using VoxelSemanticDataF = VoxelSemanticDataT<float, float>;

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
template <typename Tpos, typename Tcolor = float> struct VoxelSemanticDataProbabilisticT {

    inline static float kDepthThreshold =
        5.0f; // [m] depth threshold for updating semantics with depth
    inline static float kDepthDecayRate =
        0.07f; // [1/m] depth decay rate for updating semantics with depth

    VOXEL_DATA_USING_TYPE(Tpos, Tcolor)

    int count = 0; // number of point data integrated into the voxel
    VOXEL_POSITION_MEMBERS(Tpos)
    VOXEL_COLOR_MEMBERS(Tcolor)

    // Sparse probability distribution: (object_id, class_id) -> log_probability
    // Using log probabilities avoids numerical underflow and makes multiplication additive
    // std::map is used instead of std::unordered_map for better performance with few observations
    // (std::pair<int, int> is naturally comparable, so no custom comparator needed)
    std::map<std::pair<int, int>, float> log_probabilities;

    // Cached most likely label pair (updated lazily), (object_id, class_id) pair
    mutable std::pair<int, int> most_likely_pair = {-1, -1};
    mutable float most_likely_log_prob = -std::numeric_limits<float>::infinity();
    mutable bool cache_valid = false;
    mutable bool norm_valid = false;
    mutable float log_normalization = -std::numeric_limits<float>::infinity();

  protected:
    // Direct access members for compatibility (lazy evaluation via getters)
    // Note: These are mutable to allow const access patterns
    mutable int object_id = -1;       // cached most likely object ID
    mutable int class_id = -1;        // cached most likely class ID
    mutable float confidence_ = 0.0f; // cached confidence

  public:
    static constexpr float MIN_CONFIDENCE = 1e-10f;
    // Base log evidence per observation (used when confidence = 1.0)
    // This ensures repeated observations accumulate evidence: -log(0.9) ≈ 0.105
    // Each observation with confidence=1.0 adds this base value to the log score
    static constexpr float BASE_LOG_PROB_PER_OBSERVATION = 0.10536051565782628f; // -log(0.9)

    VOXEL_POSITION_METHODS(Tpos)
    VOXEL_COLOR_METHODS(Tcolor)

    void reset() {
        count = 0;
        position_sum = {0.0, 0.0, 0.0};
        color_sum = {0.0f, 0.0f, 0.0f};
        // reset semantics
        log_probabilities.clear();
        most_likely_pair = {-1, -1};
        most_likely_log_prob = -std::numeric_limits<float>::infinity();
        cache_valid = false;
        norm_valid = false;
        log_normalization = -std::numeric_limits<float>::infinity();
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
        norm_valid = false;
        update_cached_members();
    }

    void initialize_semantics(const int object_id, const int class_id,
                              const float confidence = 1.0f) {
        // Initialize with base log evidence scaled by confidence
        // For confidence = 1.0, use base log evidence to ensure repeated observations accumulate
        float log_prob;
        if (confidence >= 1.0f) {
            log_prob = BASE_LOG_PROB_PER_OBSERVATION;
        } else {
            log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION;
        }
        initialize_semantics_log_prob(object_id, class_id, log_prob);
    }

    void initialize_semantics_with_depth(const int object_id, const int class_id,
                                         const float depth) {
        // if depth < kDepthThreshold, the confidence is 1.0 => use base log evidence
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate)
        // log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION
        float log_prob;
        if (depth <= kDepthThreshold) {
            log_prob = BASE_LOG_PROB_PER_OBSERVATION;
        } else {
            const float confidence = std::exp(-(depth - kDepthThreshold) * kDepthDecayRate);
            log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION;
        }
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
            // Invalidate normalization before updating cache (normalization changes when we add a
            // new label)
            norm_valid = false;
            // Update cache if this new label might be the most likely
            if (cache_valid && new_log_prob > most_likely_log_prob) {
                most_likely_log_prob = new_log_prob;
                most_likely_pair = key;
                update_cached_members();
            } else if (cache_valid) {
                // Argmax unchanged; refresh confidence against new normalization.
                update_cached_members();
            } else {
                update_cache();
            }
        } else {
            // Existing label pair: accumulate log probability
            float &log_prob = it->second;
            log_prob += new_log_prob;
            // Invalidate normalization before updating cache (normalization changes when we update
            // a label)
            norm_valid = false;

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
                } else {
                    // Argmax unchanged; refresh confidence against new normalization.
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
        // Convert confidence to log evidence
        // For confidence = 1.0, use base log evidence to ensure repeated observations accumulate
        // For confidence < 1.0, scale the base log evidence by confidence
        float new_log_prob;
        if (confidence >= 1.0f) {
            new_log_prob = BASE_LOG_PROB_PER_OBSERVATION;
        } else {
            new_log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION;
        }
        update_semantics_log_prob(object_id, class_id, new_log_prob);
    }

    void update_semantics_with_depth(const int object_id, const int class_id, const float depth) {

        // if depth < kDepthThreshold, the confidence is 1.0 => use base log evidence
        // if depth >= kDepthThreshold, the confidence decays exponentially with depth
        // depth_diff = depth - kDepthThreshold
        // confidence = exp(-depth_diff * kDepthDecayRate)
        // log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION
        float new_log_prob;
        if (depth <= kDepthThreshold) {
            new_log_prob = BASE_LOG_PROB_PER_OBSERVATION;
        } else {
            const float confidence = std::exp(-(depth - kDepthThreshold) * kDepthDecayRate);
            new_log_prob = confidence * BASE_LOG_PROB_PER_OBSERVATION;
        }

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
    // Note: For probabilistic voxels, these overwrite the probability distribution by forcing
    // the label to a single (object_id, class_id) pair. Use with caution.
    void set_object_id(int id) {
        ensure_cache_updated();
        object_id = id;
        most_likely_pair.first = id;
        force_label_distribution();
    }
    void set_class_id(int id) {
        ensure_cache_updated();
        class_id = id;
        most_likely_pair.second = id;
        force_label_distribution();
    }
    void set_confidence_counter(int counter) {
        // For probabilistic voxels, confidence is computed from the distribution, not set directly
        // This setter is a no-op to maintain interface compatibility
        // The confidence will be recomputed on the next get_confidence() call
        (void)counter; // Suppress unused parameter warning
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
        object_id = most_likely_pair.first;
        class_id = most_likely_pair.second;
        confidence_ = compute_confidence();
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
            norm_valid = true;
            log_normalization = -std::numeric_limits<float>::infinity();
            update_cached_members();
            return;
        }

        // Find the maximum log probability and normalization in one pass
        most_likely_log_prob = -std::numeric_limits<float>::infinity();
        float log_sum = -std::numeric_limits<float>::infinity();
        for (const auto &[key, log_prob] : log_probabilities) {
            if (log_prob > most_likely_log_prob) {
                most_likely_log_prob = log_prob;
                most_likely_pair = key;
            }
            log_sum = log_add_exp(log_sum, log_prob);
        }
        cache_valid = true;
        norm_valid = true;
        log_normalization = log_sum;
        update_cached_members();
    }

    void force_label_distribution() {
        log_probabilities.clear();
        if (most_likely_pair.first >= 0 && most_likely_pair.second >= 0) {
            log_probabilities[most_likely_pair] = 0.0f;
            most_likely_log_prob = 0.0f;
            cache_valid = true;
            norm_valid = true;
            log_normalization = 0.0f;
        } else {
            most_likely_log_prob = -std::numeric_limits<float>::infinity();
            cache_valid = false;
            norm_valid = false;
            log_normalization = -std::numeric_limits<float>::infinity();
        }
        update_cached_members();
    }

    // Compute log of normalization constant (log-sum-exp trick for numerical stability)
    float get_log_normalization() const {
        if (log_probabilities.empty()) {
            return 0.0f;
        }

        if (norm_valid) {
            return log_normalization;
        }

        // Single-pass log-sum-exp via incremental log-add-exp
        float log_sum = -std::numeric_limits<float>::infinity();
        for (const auto &[key, log_prob] : log_probabilities) {
            log_sum = log_add_exp(log_sum, log_prob);
        }
        log_normalization = log_sum;
        norm_valid = true;
        return log_normalization;
    }

    static float log_add_exp(const float a, const float b) {
        if (a == -std::numeric_limits<float>::infinity()) {
            return b;
        }
        if (b == -std::numeric_limits<float>::infinity()) {
            return a;
        }
        const float max_ab = std::max(a, b);
        return max_ab + std::log(std::exp(a - max_ab) + std::exp(b - max_ab));
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
        // Recompute cache immediately to ensure consistency
        if (pruned_most_likely || log_probabilities.empty()) {
            cache_valid = false;
            update_cache(); // Recompute cache to ensure most_likely_pair is valid
        }
        norm_valid = false;
    }
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using VoxelSemanticDataProbabilistic = VoxelSemanticDataProbabilisticT<double, float>;
using VoxelSemanticDataProbabilisticD = VoxelSemanticDataProbabilisticT<double, float>;
using VoxelSemanticDataProbabilisticF = VoxelSemanticDataProbabilisticT<float, float>;

} // namespace volumetric

#include "voxel_data_semantic2.h"

namespace volumetric {

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

// #undef VOXEL_POSITION_MEMBERS
// #undef VOXEL_POSITION_METHODS
// #undef VOXEL_COLOR_MEMBERS
// #undef VOXEL_COLOR_METHODS

} // namespace volumetric
