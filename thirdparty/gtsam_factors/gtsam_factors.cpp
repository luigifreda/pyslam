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


//#define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Similarity3.h>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>

#include <boost/shared_ptr.hpp>  // Include Boost

#include <iostream>

namespace py = pybind11;

using namespace gtsam;
//using namespace gtsam::noiseModel;
using symbol_shorthand::X;


template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const T& model) {
    return boost::make_shared<T>(model); // This makes a copy of model
}

template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const boost::shared_ptr<T>& model) {
    return model; // No copy, just return the same shared_ptr
}

/**
 * Monocular Resectioning Factor
 */
class ResectioningFactor : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    const Cal3_S2& K_;
    Point3 P_;
    Point2 p_;

    ResectioningFactor(const SharedNoiseModel& model, 
                       const Key& key,
                       const Cal3_S2& calib,
                       const Point2& measured_p,
                       const Point3& world_P)
        : Base(model, key), K_(calib), P_(world_P), p_(measured_p) {}

    Vector evaluateError(const Pose3& pose, boost::optional<Matrix&> H = boost::none) const override {
        PinholeCamera<Cal3_S2> camera(pose, K_);        
        try {
            return camera.project(P_, H, boost::none, boost::none) - p_;
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(2,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactor]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(2);
        }
    }
}; 

/**
 * Stereo Resectioning Factor
 */
class ResectioningFactorStereo : public NoiseModelFactorN<Pose3> {
public:
    using Base = NoiseModelFactorN<Pose3>;
    Cal3_S2Stereo::shared_ptr K_;    
    Point3 P_;
    StereoPoint2 p_stereo_;

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
            StereoPoint2 projected = camera.project(P_, H, boost::none, boost::none);
            return (Vector(3) << projected.uL() - p_stereo_.uL(),
                                projected.uR() - p_stereo_.uR(),
                                projected.v() - p_stereo_.v()).finished();
        } catch( std::exception& e) {
            if (H) *H = Matrix::Zero(3,6);
            // print in red
            std::cerr << "\x1b[31m [ResectioningFactorStereo]: " << e.what() << "\x1b[0m" << std::endl;
            return Vector::Zero(3);
        }
    }
};


class PriorFactorSimilarity3 : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    gtsam::Similarity3 prior_;

    PriorFactorSimilarity3(gtsam::Key key, const gtsam::Similarity3& prior, const gtsam::SharedNoiseModel& noise)
        : Base(noise, key), prior_(prior) {}

    // Define error function: Logmap of transformation error
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        gtsam::Vector7 error = gtsam::Similarity3::Logmap(prior_.inverse() * sim);

        if (H) {
            auto functor = [this](const gtsam::Similarity3& sim) {
                return gtsam::Similarity3::Logmap(prior_.inverse() * sim);
            };
            *H = gtsam::numericalDerivative11<gtsam::Vector7, gtsam::Similarity3>(functor, sim, 1e-5);
        }

        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3>(*this);
    }
};

class PriorFactorSimilarity3ScaleOnly : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    double prior_scale_;  // Prior scale value

    PriorFactorSimilarity3ScaleOnly(gtsam::Key key, double prior_scale, double prior_sigma)
        : Base(gtsam::noiseModel::Isotropic::Sigma(1, prior_sigma), key), prior_scale_(prior_scale) {}

    // Define error function: Only penalize scale difference
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        double scale_error = std::log(sim.scale()) - std::log(prior_scale_);  // Log-scale difference

        if (H) {
            // Compute Jacobian: d(log(s)) / ds = 1 / s
            Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
            J(6) = 1.0 / sim.scale();  // Derivative w.r.t. scale
            *H = J;
        }

        return gtsam::Vector1(scale_error);  // Return 1D error
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3ScaleOnly>(*this);
    }
};


class SimResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
private:
    gtsam::Cal3_S2 calib_;  // Camera intrinsics
    gtsam::Point2 uv_;      // Observed 2D point
    gtsam::Point3 P_;      // 3D point

public:
    SimResectioningFactor(gtsam::Key& sim_pose_key,
                            const gtsam::Cal3_S2& calib,
                            const gtsam::Point2& uv,
                            const gtsam::Point3& P,
                            const gtsam::SharedNoiseModel& noiseModel)
        : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
            calib_(calib), uv_(uv), P_(P) {}

    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        auto functor = [this](const gtsam::Similarity3& sim) {
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();
            const gtsam::Vector3 transformed_P = s * (R * P_) + t;
            gtsam::Vector3 projected = calib_.K() * transformed_P;
            return projected.head<2>() / projected(2) - uv_;
        };

        auto functorLie = [this](const gtsam::Vector7& logmap) {
            const gtsam::Similarity3 sim = gtsam::Similarity3::Expmap(logmap);
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();
            const gtsam::Vector3 transformed_P = s * (R * P_) + t;
            gtsam::Vector3 projected = calib_.K() * transformed_P;
            return projected.head<2>() / projected(2) - uv_;
        };            

        // Compute the error
        const gtsam::Vector2 error = functor(sim3);
        const gtsam::Vector7 logmap = gtsam::Similarity3::Logmap(sim3);

        // Compute Jacobians if required
        if (H) {    
            *H = gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Vector7>(functorLie, logmap, 1e-5);
        }

        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<SimResectioningFactor>(*this);
    }
};


class SimInvResectioningFactor : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
private:
    gtsam::Cal3_S2 calib_;  // Camera intrinsics
    gtsam::Point2 uv_;      // Observed 2D pixel point
    gtsam::Point3 P_;      // 3D camera point

public:
    SimInvResectioningFactor(gtsam::Key& sim_pose_key,
                                const gtsam::Cal3_S2& calib,
                                const gtsam::Point2& p,
                                const gtsam::Point3& P,
                                const gtsam::SharedNoiseModel& noiseModel)
        : gtsam::NoiseModelFactor1<gtsam::Similarity3>(noiseModel, sim_pose_key),
            calib_(calib), uv_(p), P_(P) {}

    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
        auto functor = [this](const gtsam::Similarity3& sim) {
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();

            const gtsam::Matrix3 R_inv = R.transpose();
            const double s_inv = 1.0 / s;
            const gtsam::Vector3 t_inv = -s_inv * (R_inv * t);
            const gtsam::Vector3 transformed_P = s_inv * (R_inv * P_) + t_inv;
            const gtsam::Vector3 projected = calib_.K() * transformed_P;
            return projected.head<2>() / projected(2) - uv_;
        };

        auto functorLie = [this](const gtsam::Vector7& logmap) {
            const gtsam::Similarity3 sim = gtsam::Similarity3::Expmap(logmap);
            const gtsam::Matrix3 R = sim.rotation().matrix();
            const gtsam::Vector3 t = sim.translation();
            const double s = sim.scale();

            const gtsam::Matrix3 R_inv = R.transpose();
            const double s_inv = 1.0 / s;
            const gtsam::Vector3 t_inv = -s_inv * (R_inv * t);
            const gtsam::Vector3 transformed_P = s_inv * (R_inv * P_) + t_inv;
            const gtsam::Vector3 projected = calib_.K() * transformed_P;
            return projected.head<2>() / projected(2) - uv_;
        };            

        // Compute the error
        const gtsam::Vector2 error = functor(sim3);
        const gtsam::Vector7 logmap = gtsam::Similarity3::Logmap(sim3);

        // Compute Jacobians if required
        if (H) {
            *H = gtsam::numericalDerivative11<gtsam::Vector2, gtsam::Vector7>(functorLie, logmap, 1e-5);
        }

        return error;
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<SimInvResectioningFactor>(*this);
    }
};
    

// Function to insert Similarity3 into Values
void insertSimilarity3(gtsam::Values &values, const Key& key, const gtsam::Similarity3 &sim3) {
    values.insert(key, sim3);
}

// Function to get Similarity3 from Values
gtsam::Similarity3 getSimilarity3(const gtsam::Values &values, gtsam::Key key) {
    if (!values.exists(key)) {
        throw std::runtime_error("Key not found in Values.");
    }
    return values.at<gtsam::Similarity3>(key);  // Return by value (safe)
}

PYBIND11_MODULE(gtsam_factors, m) {
    py::module_::import("gtsam");  // Ensure GTSAM is loaded

    // Register NoiseModelFactorN classes
    py::class_<NoiseModelFactorN<Pose3>, std::shared_ptr<NoiseModelFactorN<Pose3>>, NonlinearFactor>(m, "NoiseModelFactorN_Pose3");
    py::class_<NoiseModelFactorN<Similarity3>, std::shared_ptr<NoiseModelFactorN<Similarity3>>, NonlinearFactor>(m, "NoiseModelFactorN_Similarity3");


    py::class_<ResectioningFactor, std::shared_ptr<ResectioningFactor>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactor")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(model, key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Isotropic& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactor::evaluateError)
    .def("error", &ResectioningFactor::error);


    py::class_<ResectioningFactorStereo, std::shared_ptr<ResectioningFactorStereo>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereo")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(model, key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Isotropic& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorStereo::evaluateError)
    .def("error", &ResectioningFactorStereo::error);


    py::class_<PriorFactorSimilarity3, gtsam::NoiseModelFactor1<gtsam::Similarity3>, std::shared_ptr<PriorFactorSimilarity3>>(m, "PriorFactorSimilarity3")
    .def(py::init<const gtsam::Key&, const gtsam::Similarity3&, const gtsam::SharedNoiseModel&>(),
        py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Isotropic& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));       
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Diagonal& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key, const gtsam::Similarity3& prior, const gtsam::noiseModel::Robust& model){
        return new PriorFactorSimilarity3(key, prior, create_shared_noise_model(model));       
    }), py::arg("key"), py::arg("prior"), py::arg("model"))
    .def("evaluateError", &PriorFactorSimilarity3::evaluateError);


    py::class_<PriorFactorSimilarity3ScaleOnly, gtsam::NoiseModelFactor1<gtsam::Similarity3>, std::shared_ptr<PriorFactorSimilarity3ScaleOnly>>(m, "PriorFactorSimilarity3ScaleOnly")
    .def(py::init<const gtsam::Key&, double, double>(),
        py::arg("key"), py::arg("prior_scale"), py::arg("prior_sigma"))
    .def("evaluateError", &PriorFactorSimilarity3ScaleOnly::evaluateError);


    py::class_<SimResectioningFactor, gtsam::NoiseModelFactor, std::shared_ptr<SimResectioningFactor>>(m, "SimResectioningFactor")
    .def(py::init<gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Isotropic& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimResectioningFactor::evaluateError);


    py::class_<SimInvResectioningFactor, gtsam::NoiseModelFactor, std::shared_ptr<SimInvResectioningFactor>>(m, "SimInvResectioningFactor")
    .def(py::init<gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Isotropic& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimInvResectioningFactor::evaluateError);

    m.def("insert_similarity3", &insertSimilarity3, "Insert Similarity3 into Values",
        py::arg("values"), py::arg("key"), py::arg("sim3"));    

    m.def("get_similarity3", &getSimilarity3, "Get Similarity3 from Values",
        py::arg("values"), py::arg("key"));
}