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
#include <gtsam/geometry/Similarity3.h>

#include <boost/make_shared.hpp>

using namespace gtsam;
using namespace gtsam::noiseModel;
using symbol_shorthand::X;


#define USE_ANALYTICAL_JACOBIAN_FOR_TCW_RESECTION 1

namespace gtsam_factors {

/**
 * Monocular Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactor : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    const gtsam::Cal3_S2& K_;
    gtsam::Point3 P_;  // world point
    gtsam::Point2 p_;  // pixel point 
    double weight_ = 1.0;

    ResectioningFactor(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2& calib,
                        const gtsam::Point2& measured_p,
                        const gtsam::Point3& world_P)
        : Base(model, key), K_(calib), P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        gtsam::PinholeCamera<gtsam::Cal3_S2> camera(pose, K_);        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(2,6);
                return gtsam::Vector::Zero(2);
            } else {
                const gtsam::Point2 delta = camera.project(P_, H, boost::none, boost::none) - p_;
                const gtsam::Vector error(delta);
                if (H) *H *= weight_;
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactor]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(2);
        }
    }

    void setWeight(double weight) { assert(weight > 0.0); weight_ = weight; }
    double getWeight() const { return weight_; }
}; 


/**
 * Monocular Resectioning Factor.
 * The pose of the camera is assumed to be Tcw.
 */
class ResectioningFactorTcw : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_;            
    gtsam::Point3 P_;  // world point
    gtsam::Point2 p_;  // pixel point 
    double weight_ = 1.0;    

    ResectioningFactorTcw(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2& calib,
                        const gtsam::Point2& measured_p,
                        const gtsam::Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
            P_(world_P), p_(measured_p) {}

#if USE_ANALYTICAL_JACOBIAN_FOR_TCW_RESECTION
    Vector evaluateError(const gtsam::Pose3& Tcw, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(2,6);
                return gtsam::Vector::Zero(2);
            }

            const gtsam::Rot3& Rcw = Tcw.rotation();
            const gtsam::Point3& tcw = Tcw.translation();
            const gtsam::Matrix3 R = Rcw.matrix();
            const gtsam::Vector3 Pc = R * P_ + tcw;

            const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
            const double Zinv = 1.0 / Z;
            const double Zinv2 = Zinv * Zinv;

            // Projection
            const double u = fx_ * X * Zinv + cx_;
            const double v = fy_ * Y * Zinv + cy_;
            gtsam::Vector2 error;
            error << u - p_.x(), v - p_.y();
            error *= weight_;

            if (H) {
                // d(projected)/dPc
                gtsam::Matrix23 J_proj;
                J_proj << fx_ * Zinv,         0.0, -fx_ * X * Zinv2,
                            0.0,    fy_ * Zinv, -fy_ * Y * Zinv2;

                // dPc/dTcw
                gtsam::Matrix36 J_pose;
                J_pose.leftCols<3>() = -R * gtsam::skewSymmetric(P_); // wrt rotation
                J_pose.rightCols<3>() = R;                                     // wrt translation

                *H = weight_ * (J_proj * J_pose);  // Chain rule
            }

            return error;

        } catch( std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(2,6);
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(2);
        }
    }
#else 
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
            if (H) *H = gtsam::Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(2);
        }
    }
#endif

    Vector2 camProject(const gtsam::Vector3 &Pc) const {
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
class ResectioningFactorStereo : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    gtsam::Cal3_S2Stereo::shared_ptr K_;    
    gtsam::Point3 P_;
    gtsam::StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
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
 * Stereo Resectioning Factor.
 * The pose of the camera is assumed to be Twc.
 */
class ResectioningFactorStereoTcw : public gtsam::NoiseModelFactorN<Pose3> {
public:
    using Base = gtsam::NoiseModelFactorN<Pose3>;
    double fx_, fy_, cx_, cy_, bf_;            
    gtsam::Point3 P_;               // world point
    gtsam::StereoPoint2 p_stereo_;  // pixel point (uL, uR, vL)
    double weight_ = 1.0;    

    ResectioningFactorStereoTcw(const gtsam::SharedNoiseModel& model, 
                        const gtsam::Key& key,
                        const gtsam::Cal3_S2Stereo& calib,
                        const gtsam::StereoPoint2& measured_p_stereo,
                        const gtsam::Point3& world_P)
        : Base(model, key), fx_(calib.fx()), fy_(calib.fy()), cx_(calib.px()), cy_(calib.py()), 
          bf_(calib.baseline()*calib.fx()), P_(world_P), p_stereo_(measured_p_stereo) {}

#if USE_ANALYTICAL_JACOBIAN_FOR_TCW_RESECTION
    Vector evaluateError(const gtsam::Pose3& Tcw, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(3,6);
                return gtsam::Vector3::Zero();
            }

            const gtsam::Rot3& Rcw = Tcw.rotation();
            const gtsam::Point3& tcw = Tcw.translation();
            const gtsam::Matrix3 R = Rcw.matrix();
            const gtsam::Vector3 Pc = R * P_ + tcw;

            const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
            const double invZ = 1.0 / Z;
            const double invZ2 = invZ * invZ;

            // Projection
            const double uL = fx_ * X * invZ + cx_;
            const double uR = uL - bf_ * invZ;
            const double vL = fy_ * Y * invZ + cy_;
            gtsam::Vector3 error;
            error << uL - p_stereo_.uL(), uR - p_stereo_.uR(), vL - p_stereo_.v();
            error *= weight_;

            if (H) {
                // d(cam projection) / d(Pc)
                gtsam::Matrix33 J_proj;
                J_proj <<
                    fx_ * invZ,       0.0, -fx_ * X * invZ2,
                    fx_ * invZ,       0.0, -fx_ * X * invZ2 - bf_ * invZ2,
                        0.0, fy_ * invZ, -fy_ * Y * invZ2;

                // d(Pc) / d(Tcw)
                gtsam::Matrix36 J_pose;
                J_pose.leftCols<3>()  = -R * gtsam::skewSymmetric(P_);  // Rotation
                J_pose.rightCols<3>() = R;                              // Translation

                *H = weight_ * (J_proj * J_pose);  // Chain rule
            }

            return error;

        } catch (const std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(3,6);
            std::cerr << "\x1b[31m [ResectioningFactorStereoTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector3::Zero();
        }
    }
#else 
    Vector evaluateError(const gtsam::Pose3& Tcw, boost::optional<gtsam::Matrix&> H = boost::none) const override {
        auto computeError = [&](const gtsam::Pose3& Tcw) {
            const gtsam::Matrix3 Rcw = Tcw.rotation().matrix();
            const gtsam::Vector3 tcw = Tcw.translation();            
            const gtsam::Vector3 Pc = Rcw * P_ + tcw;
            const gtsam::Vector3 error = camProject(Pc) - p_stereo_.vector();
            return error;
        };
        
        try {
            if (weight_ <= std::numeric_limits<double>::epsilon()) {
                if (H) *H = gtsam::Matrix::Zero(3,6);
                return gtsam::Vector::Zero(3);
            } else {
                const Vector error(computeError(Tcw));
                // Compute Jacobians if required
                if (H) {    
                    *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(computeError, Tcw, 1e-5);
                }
                return weight_ * error;
            }
        } catch( std::exception& e) {
            if (H) *H = gtsam::Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorTcw]: " << e.what() << "\x1b[0m" << std::endl;
            return gtsam::Vector::Zero(3);
        }
    }
#endif 

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
        
    

// =====================================================================================================================


// Used by optimizer_gtsam.optimize_sim3()
class SimResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
    private:
        const Cal3_S2& calib_;  // Camera intrinsics
        const Point2 uv_;       // Observed 2D point
        const Point3 P_;        // 3D point
        double weight_ = 1.0;   // Weight
    
    public:
        SimResectioningFactor(const gtsam::Key& sim_pose_key,
                                const gtsam::Cal3_S2& calib,
                                const gtsam::Point2& uv,
                                const gtsam::Point3& P,
                                const gtsam::SharedNoiseModel& noiseModel)
            : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
                calib_(calib), uv_(uv), P_(P) {}
    
        gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                    boost::optional<gtsam::Matrix&> H = boost::none) const override {
            auto computeError = [this](const gtsam::Similarity3& sim) {
                const gtsam::Matrix3 R = sim.rotation().matrix();
                const gtsam::Vector3 t = sim.translation();
                const double s = sim.scale();
                const gtsam::Vector3 transformed_P = s * (R * P_) + t;
                const gtsam::Vector3 projected = calib_.K() * transformed_P;
                const gtsam::Point2 uv = projected.head<2>() / projected[2];
                const gtsam::Vector2 error = uv - uv_;
                return error;
            };
    
            const gtsam::Vector2 error = weight_ * computeError(sim3);
    
            // Compute Jacobians if required
            if (H) {    
                *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Similarity3>(computeError, sim3, 1e-5);
            }
            return error;
        }
    
        void setWeight(double weight) {
            weight_ = weight;
        }
    
        double getWeight() const {
            return weight_;
        }
    
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
            return boost::make_shared<SimResectioningFactor>(*this);
        }    
    };
    
    
    // Used by optimizer_gtsam.optimize_sim3()
    class SimInvResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
    private:
        const Cal3_S2& calib_;  // Camera intrinsics
        const Point2 uv_;       // Observed 2D pixel point
        const Point3 P_;        // 3D camera point
        double weight_ = 1.0;   // Weight    
    
    public:
        SimInvResectioningFactor(const gtsam::Key& sim_pose_key,
                                 const gtsam::Cal3_S2& calib,
                                 const gtsam::Point2& p,
                                 const gtsam::Point3& P,
                                 const gtsam::SharedNoiseModel& noiseModel)
            : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
                calib_(calib), uv_(p), P_(P) {}
    
        gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                    boost::optional<gtsam::Matrix&> H = boost::none) const override {
            auto computeError = [this](const gtsam::Similarity3& sim) {
                const gtsam::Matrix3 R = sim.rotation().matrix();
                const gtsam::Vector3 t = sim.translation();
                const double s = sim.scale();
    
                const gtsam::Matrix3 R_inv = R.transpose();
                const double s_inv = 1.0 / s;
                const gtsam::Vector3 t_inv = -s_inv * (R_inv * t);
                const gtsam::Vector3 transformed_P = s_inv * (R_inv * P_) + t_inv;
                const gtsam::Vector3 projected = calib_.K() * transformed_P;
                const gtsam::Point2 uv = projected.head<2>() / projected[2];
                const gtsam::Vector2 error = uv - uv_;
                return error;
            };   
    
            const gtsam::Vector2 error = weight_ * computeError(sim3);
    
            // Compute Jacobians if required
            if (H) {
                *H = weight_ * gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Similarity3>(computeError, sim3, 1e-5);
            }
    
            return error;
        }
    
        void setWeight(double weight) {
            weight_ = weight;
        }
    
        double getWeight() const {
            return weight_;
        }
    
        virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
            return boost::make_shared<SimInvResectioningFactor>(*this);
        }
    };
    
} // namespace gtsam_factors