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

// #define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam

#include "numerical_derivative.h"

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>

#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#include <boost/shared_ptr.hpp> // Include Boost

#include <iostream>

using namespace gtsam;
using symbol_shorthand::X;

#define SIM3_FACTOR_REVERSE_ERROR_DIRECTION 0

namespace gtsam_factors {

// Similarity3 prior factor with autodifferencing. Goal is to penalize all terms.
class PriorFactorSimilarity3 : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 prior_inverse_;

    PriorFactorSimilarity3(gtsam::Key key, const gtsam::Similarity3 &prior,
                           const gtsam::SharedNoiseModel &noise)
        : Base(noise, key), prior_inverse_(prior.inverse()) {}

    // Define error function: Logmap of transformation error
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
        gtsam::Vector7 error = gtsam::Similarity3::Logmap(prior_inverse_ * sim);

        if (H) {
            auto functor = [this](const gtsam::Similarity3 &sim) {
                return gtsam::Similarity3::Logmap(prior_inverse_ * sim);
            };
            *H = gtsam::numericalDerivative11<gtsam::Vector7, gtsam::Similarity3>(functor, sim,
                                                                                  1e-5);
        }
        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<PriorFactorSimilarity3> shared_ptr;
};

// Similarity3 prior factor with only scale. Goal is to only penalize scale difference.
class PriorFactorSimilarity3ScaleOnly : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const double prior_scale_; // Prior scale value

    PriorFactorSimilarity3ScaleOnly(gtsam::Key key, double prior_scale, double prior_sigma)
        : Base(gtsam::noiseModel::Isotropic::Sigma(1, prior_sigma), key),
          prior_scale_(prior_scale) {}

    // Define error function: Only penalize scale difference
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
#define USE_LOG_SCALE 1
#if USE_LOG_SCALE
        double scale_error = std::log(sim.scale()) - std::log(prior_scale_); // Log-scale difference
#else
        double scale_error = sim.scale() - prior_scale_;
#endif

        if (H) {
            // Compute Jacobian: d(log(s)) / ds = 1 / s
            Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
#if USE_LOG_SCALE
            J(6) = 1.0 / sim.scale(); // Derivative w.r.t. scale
#else
            J(6) = 1.0; // Derivative w.r.t. scale
#endif
            *H = J;
        }
        return gtsam::Vector1(scale_error); // Return 1D error
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3ScaleOnly>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<PriorFactorSimilarity3ScaleOnly> shared_ptr;
};

// =====================================================================================================================

// Custom version of BetweenFactor<Similarity3> with autodifferencing
// Assuming 
// sim3_1 = Swc1 
// sim3_2 = Swc2
// measured = Sc1c2
class BetweenFactorSimilarity3
    : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key1, key2), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured) 
#else
        measured_inverse_(measured.inverse())
#endif
        {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &sim3_2,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
        const gtsam::Similarity3 sim3_1_inverse = sim3_1.inverse();

        // Compute predicted relative transformation
        const gtsam::Similarity3 predicted =
            sim3_1_inverse * sim3_2; // Swc1.inverse() * Swc2 = Sc1c2 = S12

        // Compute the error in Sim3 space
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        gtsam::Similarity3 errorSim3 = measured_ * predicted.inverse();
#else
        gtsam::Similarity3 errorSim3 = measured_inverse_ * predicted;
#endif

        // Compute Jacobians only if needed
        if (H1) {
            *H1 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s1) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap(measured_ * (s1.inverse() * sim3_2).inverse());
#else
                    return Similarity3::Logmap(measured_inverse_ * (s1.inverse() * sim3_2));
#endif
                },
                sim3_1, 1e-5);
        }
        if (H2) {
            *H2 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s2) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap(measured_ * (sim3_1_inverse * s2).inverse());
#else
                    return Similarity3::Logmap(measured_inverse_ * (sim3_1_inverse * s2));
#endif
                },
                sim3_2, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<BetweenFactorSimilarity3>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<BetweenFactorSimilarity3> shared_ptr;
};


// =====================================================================================================================

// Custom version of BetweenFactor<Similarity3> with autodifferencing for inverse error
// Assuming:
// sim3_1 = Sc1w
// sim3_2 = Sc2w
// measured = Sc1c2
class BetweenFactorSimilarity3Inverse
    : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3Inverse(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key1, key2), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured) 
#else
        measured_inverse_(measured.inverse())
#endif
        {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &sim3_2,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
        const gtsam::Similarity3 sim3_2_inverse = sim3_2.inverse();
        // Compute predicted relative transformation
        const gtsam::Similarity3 predicted =
            sim3_1 * sim3_2_inverse; // Sc1w * Sc2w.inverse() = Sc1c2 = S12

        // Compute the error in Sim3 space
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        // For BetweenFactorSimilarity3Inverse, use predicted.inverse() * measured_ to match backward error semantics
        gtsam::Similarity3 errorSim3 = predicted.inverse() * measured_;
#else
        gtsam::Similarity3 errorSim3 = measured_inverse_ * predicted;
#endif

        // Compute Jacobians only if needed
        if (H1) {
            *H1 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s1) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap((s1 * sim3_2_inverse).inverse() * measured_);
#else
                    return Similarity3::Logmap(measured_inverse_ * (s1 * sim3_2_inverse));
#endif
                },
                sim3_1, 1e-5);
        }
        if (H2) {
            *H2 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s2) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap((sim3_1 * s2.inverse()).inverse() * measured_);
#else
                    return Similarity3::Logmap(measured_inverse_ * (sim3_1 * s2.inverse()));
#endif
                },
                sim3_2, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<BetweenFactorSimilarity3Inverse>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<BetweenFactorSimilarity3Inverse> shared_ptr;
};



// Custom version of BetweenFactor<Similarity3> with autodifferencing for inverse error. Only s1 is optimized
// Assuming:
// sim3_1 = Sc1w
// sim3_2 = Sc2w  FIXED
// measured = Sc1c2
class BetweenFactorSimilarity3InverseOnlyS1
    : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 sim3_2_inverse_;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3InverseOnlyS1(gtsam::Key key_sim3_1, const gtsam::Similarity3 &sim3_2, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key_sim3_1), sim3_2_inverse_(sim3_2.inverse()), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured)
#else
        measured_inverse_(measured.inverse())
#endif
    {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_1,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
        // Compute predicted relative transformation
        const gtsam::Similarity3 predicted =
            sim3_1 * sim3_2_inverse_; // Sc1w * Sc2w.inverse() = Sc1c2 = S12

        // Compute the error in Sim3 space
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        // For BetweenFactorSimilarity3Inverse, use predicted.inverse() * measured_ to match backward error semantics
        gtsam::Similarity3 errorSim3 = predicted.inverse() * measured_;
#else
        gtsam::Similarity3 errorSim3 = measured_inverse_ * predicted;
#endif

        // Compute Jacobians only if needed
        if (H) {
            *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s1) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap((s1 * sim3_2_inverse_).inverse() * measured_);
#else
                    return Similarity3::Logmap(measured_inverse_ * (s1 * sim3_2_inverse_));
#endif
                },
                sim3_1, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<BetweenFactorSimilarity3InverseOnlyS1>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<BetweenFactorSimilarity3InverseOnlyS1> shared_ptr;
};




// Custom version of BetweenFactor<Similarity3> with autodifferencing for inverse error. Only s2 is optimized
// Assuming:
// sim3_1 = Sc1w FIXED
// sim3_2 = Sc2w  
// measured = Sc1c2
class BetweenFactorSimilarity3InverseOnlyS2
    : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
  public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    const gtsam::Similarity3 sim3_1_;
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
    const gtsam::Similarity3 measured_; // Relative Sim3 measurement Sc1c2
#else
    const gtsam::Similarity3 measured_inverse_; // Relative Sim3 measurement Sc1c2
#endif

    BetweenFactorSimilarity3InverseOnlyS2(gtsam::Key key_sim3_2, const gtsam::Similarity3 &sim3_1, const gtsam::Similarity3 &measured,
                             const gtsam::SharedNoiseModel &model)
        : Base(model, key_sim3_2), sim3_1_(sim3_1), 
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        measured_(measured)
#else
        measured_inverse_(measured.inverse())
#endif
    {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3 &sim3_2,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
        const gtsam::Similarity3 sim3_2_inverse = sim3_2.inverse();                                    
        // Compute predicted relative transformation
        const gtsam::Similarity3 predicted =
            sim3_1_ * sim3_2_inverse; // Sc1w * Sc2w.inverse() = Sc1c2 = S12

        // Compute the error in Sim3 space
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
        // For BetweenFactorSimilarity3Inverse, use predicted.inverse() * measured_ to match backward error semantics
        gtsam::Similarity3 errorSim3 = predicted.inverse() * measured_;
#else
        gtsam::Similarity3 errorSim3 = measured_inverse_ * predicted;
#endif

        // Compute Jacobians only if needed
        if (H) {
            *H = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3 &s2) {
#if SIM3_FACTOR_REVERSE_ERROR_DIRECTION
                    return Similarity3::Logmap((sim3_1_ * s2.inverse()).inverse() * measured_);
#else
                    return Similarity3::Logmap(measured_inverse_ * (sim3_1_ * s2.inverse()));
#endif
                },
                sim3_2, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<BetweenFactorSimilarity3InverseOnlyS2>(*this);
    }

    // shorthand for a smart pointer to a factor
    typedef boost::shared_ptr<BetweenFactorSimilarity3InverseOnlyS2> shared_ptr;
};

// =====================================================================================================================

// Function to insert Similarity3 into Values
void insertSimilarity3(gtsam::Values &values, const Key &key, const gtsam::Similarity3 &sim3) {
    values.insert(key, sim3);
}

// Function to get Similarity3 from Values
gtsam::Similarity3 getSimilarity3(const gtsam::Values &values, gtsam::Key key) {
    if (!values.exists(key)) {
        throw std::runtime_error("Key not found in Values.");
    }
    return values.at<gtsam::Similarity3>(key); // Return by value (safe)
}

} // namespace gtsam_factors
