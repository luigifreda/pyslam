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


#include "numerical_derivative.h"

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <boost/make_shared.hpp>

using namespace gtsam;
using namespace gtsam::noiseModel;
using symbol_shorthand::X;


/**
 * Monocular Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactor : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    const Cal3_S2& K_;
    Point3 P_;  // world point
    Point2 p_;  // pixel point 
    double weight_ = 1.0;

    ResectioningFactor(const SharedNoiseModel& model, 
                        const Key& key,
                        const Cal3_S2& calib,
                        const Point2& measured_p,
                        const Point3& world_P)
        : Base(model, key), K_(calib), P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        PinholeCamera<Cal3_S2> camera(pose, K_);        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(2,6);
                return Vector::Zero(2);
            } else {
                const Point2 delta = camera.project(P_, H, boost::none, boost::none) - p_;
                const Vector error(delta);
                if (H) *H *= weight_;
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactor]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(2);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 


/**
 * Monocular Resectioning Factor with automatic differentiation.
 * The pose of the camera is assumed to be Tcw.
 */
class ResectioningFactorTcw : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_;            
    Point3 P_;  // world point
    Point2 p_;  // pixel point 
    double weight_ = 1.0;    

    ResectioningFactorTcw(const SharedNoiseModel& model, 
                        const Key& key,
                        const Cal3_S2& calib,
                        const Point2& measured_p,
                        const Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
            P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& Tcw, boost::optional<Matrix&> H = boost::none) const override {
        auto computeError = [&](const Pose3& Tcw) {
            const gtsam::Matrix3 Rcw = Tcw.rotation().matrix();
            const gtsam::Vector3 tcw = Tcw.translation();            
            const gtsam::Vector3 Pc = Rcw * P_ + tcw;
            const gtsam::Vector2 error = camProject(Pc) - p_;
            return error;
        };
        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(2,6);
                return Vector::Zero(2);
            } else {
                const Vector error(computeError(Tcw));
                // Compute Jacobians if required
                if (H) {    
                    *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Pose3>(computeError, Tcw, 1e-5);
                }
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(2);
        }
    }

    Vector2 camProject(const Vector3 &Pc) const {
        const double invz = 1.0 / Pc[2];
        const double u = fx_ * Pc[0] * invz + cx_;
        const double v = fy_ * Pc[1] * invz + cy_;
        return Vector2(u, v);
    }
      

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 


/**
 * Stereo Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactorStereo : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    Cal3_S2Stereo::shared_ptr K_;    
    Point3 P_;
    StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
    double weight_ = 1.0;    

    ResectioningFactorStereo(const SharedNoiseModel& model, 
                                const Key& key,
                                const Cal3_S2Stereo& calib,
                                const StereoPoint2& measured_p_stereo,
                                const Point3& world_P)
        : Base(model, key), /*K_(calib),*/ P_(world_P), p_stereo_(measured_p_stereo) {
            K_ = boost::make_shared<Cal3_S2Stereo>(calib);
        }

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        StereoCamera camera(pose, K_);
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(3,6);
                return Vector::Zero(3);
            } else {
                const StereoPoint2 delta = camera.project(P_, H, boost::none, boost::none) - p_stereo_;
                const Vector error = delta.vector();
                if (H) *H *= weight_;
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorStereo]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(3);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }    
};


/**
 * Stereo Resectioning Factor with automatic differentiation.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactorStereoTcw : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_, bf_;            
    Point3 P_;               // world point
    StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
    double weight_ = 1.0;    

    ResectioningFactorStereoTcw(const SharedNoiseModel& model, 
                        const Key& key,
                        const Cal3_S2Stereo& calib,
                        const StereoPoint2& measured_p_stereo,
                        const Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
          bf_(calib.baseline()*calib.fx()), P_(world_P), p_stereo_(measured_p_stereo) {}

    Vector evaluateError(const Pose3& Tcw, boost::optional<Matrix&> H = boost::none) const override {
        auto computeError = [&](const Pose3& Tcw) {
            const gtsam::Matrix3 Rcw = Tcw.rotation().matrix();
            const gtsam::Vector3 tcw = Tcw.translation();            
            const gtsam::Vector3 Pc = Rcw * P_ + tcw;
            const gtsam::Vector3 error = camProject(Pc) - p_stereo_.vector();
            return error;
        };
        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = Matrix::Zero(3,6);
                return Vector::Zero(3);
            } else {
                const Vector error(computeError(Tcw));
                // Compute Jacobians if required
                if (H) {    
                    *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(computeError, Tcw, 1e-5);
                }
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(3);
        }
    }

    Vector3 camProject(const Vector3 &Pc) const {
        const double invz = 1.0 / Pc[2];
        const double uL = fx_ * Pc[0] * invz + cx_;
        const double vL = fy_ * Pc[1] * invz + cy_;
        const double uR = uL - bf_ * invz;        
        Vector3 res(uL, uR, vL);
        return res;
    }
        

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 
        
    