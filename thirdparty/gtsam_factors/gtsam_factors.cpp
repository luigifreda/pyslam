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

#include "optimizers.h"
#include "resectioning.h"
#include "weighted_projection_factors.h"
#include "numerical_derivative.h"

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Cal3DS2.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/geometry/Similarity3.h>

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <gtsam/base/Vector.h>
#include <gtsam/base/Matrix.h>

#include <boost/shared_ptr.hpp>  // Include Boost

#include <iostream>

namespace py = pybind11;

using namespace gtsam;
using symbol_shorthand::X;


template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const T& model) {
    return boost::make_shared<T>(model); // This makes a copy of model
}

template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const boost::shared_ptr<T>& model) {
    return model; // No copy, just return the same shared_ptr
}


// =====================================================================================================================


class SwitchableRobustNoiseModel : public noiseModel::Base {
private:
    SharedNoiseModel robustNoiseModel_;
    SharedNoiseModel diagonalNoiseModel_;
    SharedNoiseModel activeNoiseModel_; // Currently active model

public:
    /// Constructor initializes both diagonal/isotropic and robust models
    SwitchableRobustNoiseModel(int dim, double sigma, double huberThreshold):Base(dim) {

        // Create isotropic noise model
        diagonalNoiseModel_ = noiseModel::Isotropic::Sigma(dim, sigma);

        // Create robust noise model (Huber loss with isotropic base)
        auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
        robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);

        // Start with robust model as default
        activeNoiseModel_ = robustNoiseModel_;
    }

    /// Constructor initializes both diagonal and robust models
    SwitchableRobustNoiseModel(const Vector& sigmas, double huberThreshold):Base(sigmas.size()) {

        // Create digonal noise model
        diagonalNoiseModel_ = noiseModel::Diagonal::Sigmas(sigmas);

        // Create robust noise model (Huber loss with diagonal base)
        auto robustKernel = noiseModel::mEstimator::Huber::Create(huberThreshold);
        robustNoiseModel_ = noiseModel::Robust::Create(robustKernel, diagonalNoiseModel_);

        // Start with robust model as default
        activeNoiseModel_ = robustNoiseModel_;
    }

    virtual ~SwitchableRobustNoiseModel() override {}

    /// Switch to robust/diagonal model
    void setRobustModelActive(bool val) { val? activeNoiseModel_ = robustNoiseModel_ : activeNoiseModel_ = diagonalNoiseModel_; }

    /// Get the currently active noise model
    SharedNoiseModel getActiveModel() const { return activeNoiseModel_; }

    /// Get the robust noise model
    SharedNoiseModel getRobustModel() const { return robustNoiseModel_; }

    /// Get the diagonal noise model
    SharedNoiseModel getDiagonalModel() const { return diagonalNoiseModel_; }

public: 

    /// Override `whiten` method to delegate to active model
    virtual Vector whiten(const Vector& v) const override {
        return activeNoiseModel_->whiten(v);
    }

    /// Override `Whiten` method to delegate to active model
    virtual Matrix Whiten(const Matrix& H) const override {
        return activeNoiseModel_->Whiten(H);
    }

    /// Override `unwhiten` method to delegate to active model
    virtual Vector unwhiten(const Vector& v) const override {
        return activeNoiseModel_->unwhiten(v);
    }

    /// Override `whitenInPlace` method to delegate to active model
    virtual void whitenInPlace(Vector& v) const override {
        activeNoiseModel_->whitenInPlace(v);
    }

    /// Implement the print method for debugging/inspection
    virtual void print(const std::string& name = "") const override {
        std::cout << name << "SwitchableRobustNoiseModel (currently using: ";
        if (activeNoiseModel_ == diagonalNoiseModel_) {
            std::cout << "Diagonal model";
        } else {
            std::cout << "Robust model";
        }
        activeNoiseModel_->print();
        std::cout << ")\n";
    }

    /// Implement the `equals` method for comparison
    virtual bool equals(const noiseModel::Base& expected, double tol = 1e-9) const override {
        const SwitchableRobustNoiseModel* expectedModel = dynamic_cast<const SwitchableRobustNoiseModel*>(&expected);
        if (!expectedModel)
            return false;

        return activeNoiseModel_->equals(*expectedModel->getActiveModel(), tol);
    }

    // Implement the `WhitenSystem` methods as required by the base class
    virtual void WhitenSystem(std::vector<Matrix>& A, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A, b);
    }

    virtual void WhitenSystem(Matrix& A, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A, b);
    }

    virtual void WhitenSystem(Matrix& A1, Matrix& A2, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A1, A2, b);
    }

    virtual void WhitenSystem(Matrix& A1, Matrix& A2, Matrix& A3, Vector& b) const override {
        activeNoiseModel_->WhitenSystem(A1, A2, A3, b);
    }
};


// =====================================================================================================================


// Similarity3 prior factor with autodifferencing. Goal is to penalize all terms.
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


// Similarity3 prior factor with only scale. Goal is to only penalize scale difference.
class PriorFactorSimilarity3ScaleOnly : public gtsam::NoiseModelFactor1<gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor1<gtsam::Similarity3>;
    double prior_scale_;  // Prior scale value

    PriorFactorSimilarity3ScaleOnly(gtsam::Key key, double prior_scale, double prior_sigma)
        : Base(gtsam::noiseModel::Isotropic::Sigma(1, prior_sigma), key), prior_scale_(prior_scale) {}

    // Define error function: Only penalize scale difference
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim,
                                boost::optional<gtsam::Matrix&> H = boost::none) const override {
#define USE_LOG_SCALE 1
#if USE_LOG_SCALE
        double scale_error = std::log(sim.scale()) - std::log(prior_scale_);  // Log-scale difference
#else
        double scale_error = sim.scale() - prior_scale_;
#endif

        if (H) {
            // Compute Jacobian: d(log(s)) / ds = 1 / s
            Eigen::Matrix<double, 1, 7> J = Eigen::Matrix<double, 1, 7>::Zero();
#if USE_LOG_SCALE
            J(6) = 1.0 / sim.scale();  // Derivative w.r.t. scale
#else
            J(6) = 1.0;  // Derivative w.r.t. scale
#endif
            *H = J;
        }
        return gtsam::Vector1(scale_error);  // Return 1D error
    }

    virtual gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<PriorFactorSimilarity3ScaleOnly>(*this);
    }
};


// =====================================================================================================================


// Custom version of BetweenFactor<Similarity3> with autodifferencing
class BetweenFactorSimilarity3 : public gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3> {
public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>;
    gtsam::Similarity3 measured_; // Relative Sim3 measurement

    BetweenFactorSimilarity3(gtsam::Key key1, gtsam::Key key2, const gtsam::Similarity3& measured, const gtsam::SharedNoiseModel& model)
        : Base(model, key1, key2), measured_(measured) {}

    // Compute error (7D residual)
    gtsam::Vector evaluateError(const gtsam::Similarity3& sim3_1, const gtsam::Similarity3& sim3_2,
                                boost::optional<gtsam::Matrix&> H1 = boost::none,
                                boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        // Compute predicted relative transformation
        gtsam::Similarity3 predicted = sim3_1.inverse() * sim3_2; // Swc1.inverse() * Swc2 = Sc1c2 = S12
        
        // Compute the error in Sim3 space
        gtsam::Similarity3 errorSim3 = measured_ * predicted.inverse();
        
        // Compute Jacobians only if needed
        if (H1) {
            *H1 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3& s1) { 
                    return Similarity3::Logmap(measured_ * (s1.inverse() * sim3_2).inverse()); 
                }, sim3_1, 1e-5);
        }
        if (H2) {
            *H2 = gtsam::numericalDerivative11<gtsam::Vector, gtsam::Similarity3>(
                [&](const gtsam::Similarity3& s2) { 
                    return Similarity3::Logmap(measured_ * (sim3_1.inverse() * s2).inverse()); 
                }, sim3_2, 1e-5);
        }

        // Log map to get minimal 7D error representation
        return Similarity3::Logmap(errorSim3);
    }
};


// =====================================================================================================================


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

// Function to insert Pose3 into Values (helper to ensure correct type storage)
void insertPose3(gtsam::Values &values, const Key& key, const gtsam::Pose3 &pose) {
    values.insert(key, pose);
}

// Function to get Pose3 from Values
gtsam::Pose3 getPose3(const gtsam::Values &values, gtsam::Key key) {
    if (!values.exists(key)) {
        throw std::runtime_error("Key not found in Values.");
    }
    return values.at<gtsam::Pose3>(key);  // Return by value (safe)
}

// =====================================================================================================================


PYBIND11_MODULE(gtsam_factors, m) {
    py::module_::import("gtsam");  // Ensure GTSAM is loaded


    // Register NoiseModelFactorN classes
    py::class_<NoiseModelFactorN<Pose3>, std::shared_ptr<NoiseModelFactorN<Pose3>>, NonlinearFactor>(m, "NoiseModelFactorN_Pose3");
    py::class_<NoiseModelFactorN<Similarity3>, std::shared_ptr<NoiseModelFactorN<Similarity3>>, NonlinearFactor>(m, "NoiseModelFactorN_Similarity3");

    // Register base class for BetweenFactorSimilarity3
    py::class_<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>,gtsam::NonlinearFactor, std::shared_ptr<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>>>(m, "NoiseModelFactor2Similarity3")
    .def("print", &gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>::print);


    // NOTE: This commented block does not work. We need to specialize the template for each type we are interested 
    //       as it is done for Similarity3 in numerical_derivative11_v2_sim3 and the other following functions.
    // m.def("numerical_derivative11", 
    //     [](py::function py_func, const auto& x, double delta = 1e-5) -> Eigen::MatrixXd {
    //         // Explicitly define the lambda signature
    //         return numericalDerivative11_Any(py_func, x, delta);
    //     },
    //     py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
    //     "Compute numerical derivative of a function mapping any type");

    // Specialization of numericalDerivative11 for Pose3 -> Vector2
    m.def("numerical_derivative11_v2_pose3", 
        [](py::function py_func, const Pose3& x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return numericalDerivative11_V2_Pose3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Pose3 to Vector2");

    m.def("numerical_derivative11_v3_pose3", 
        [](py::function py_func, const Pose3& x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return numericalDerivative11_V3_Pose3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Pose3 to Vector3");

    // Specialization of numericalDerivative11 for Similarity3 -> Vector2
    m.def("numerical_derivative11_v2_sim3", 
        [](py::function py_func, const Similarity3& x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return numericalDerivative11_V2_Sim3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Similarity3 to Vector2");

    // Specialization of numericalDerivative11 for Similarity3 -> Vector3
    m.def("numerical_derivative11_v3_sim3", 
        [](py::function py_func, const Similarity3& x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return numericalDerivative11_V3_Sim3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Similarity3 to Vector3");        


    py::class_<SwitchableRobustNoiseModel, std::shared_ptr<SwitchableRobustNoiseModel>, noiseModel::Base>(m, "SwitchableRobustNoiseModel")
    .def(py::init([](int dim, double sigma, double huberThreshold) {
        return new SwitchableRobustNoiseModel(dim, sigma, huberThreshold);
    }), py::arg("dim"), py::arg("sigma"), py::arg("huberThreshold"))
    .def(py::init([](const Vector& sigmas, double huberThreshold) {
        return new SwitchableRobustNoiseModel(sigmas, huberThreshold);
    }), py::arg("sigmas"), py::arg("huberThreshold"))
    .def("set_robust_model_active", &SwitchableRobustNoiseModel::setRobustModelActive)
    .def("get_robust_model", &SwitchableRobustNoiseModel::getRobustModel)
    .def("get_diagonal_model", &SwitchableRobustNoiseModel::getDiagonalModel);

    
    py::class_<ResectioningFactor, std::shared_ptr<ResectioningFactor>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactor")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(model, key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactor::evaluateError)
    .def("error", &ResectioningFactor::error)
    .def("set_weight", &ResectioningFactor::setWeight)
    .def("get_weight", &ResectioningFactor::getWeight);


    py::class_<ResectioningFactorTcw, std::shared_ptr<ResectioningFactorTcw>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorTcw")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactorTcw(model, key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactorTcw(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactorTcw(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactorTcw(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorTcw::evaluateError)
    .def("error", &ResectioningFactorTcw::error)
    .def("set_weight", &ResectioningFactorTcw::setWeight)
    .def("get_weight", &ResectioningFactorTcw::getWeight);


    py::class_<ResectioningFactorStereo, std::shared_ptr<ResectioningFactorStereo>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereo")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(model, key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorStereo::evaluateError)
    .def("error", &ResectioningFactorStereo::error)
    .def("set_weight", &ResectioningFactorStereo::setWeight)
    .def("get_weight", &ResectioningFactorStereo::getWeight);    


    py::class_<ResectioningFactorStereoTcw, std::shared_ptr<ResectioningFactorStereoTcw>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereoTcw")
    .def(py::init([](const SharedNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereoTcw(model, key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Diagonal& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereoTcw(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereoTcw(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const SwitchableRobustNoiseModel& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereoTcw(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorStereoTcw::evaluateError)
    .def("error", &ResectioningFactorStereoTcw::error)
    .def("set_weight", &ResectioningFactorStereoTcw::setWeight)
    .def("get_weight", &ResectioningFactorStereoTcw::getWeight);


    py::class_<WeightedGenericProjectionFactorCal3_S2, gtsam::NonlinearFactor, std::shared_ptr<WeightedGenericProjectionFactorCal3_S2>>(m, "WeightedGenericProjectionFactorCal3_S2")
    .def(py::init([] (const Point2& measured, const SharedNoiseModel& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
            return new WeightedGenericProjectionFactorCal3_S2(measured, model, poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))
    .def(py::init([] (const Point2& measured, const noiseModel::Diagonal& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))   
    .def(py::init([] (const Point2& measured, const noiseModel::Robust& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))    
    .def(py::init([] (const Point2& measured, const SwitchableRobustNoiseModel& model, const Key& poseKey, const Key& pointKey, const Cal3_S2& K) {
        return new WeightedGenericProjectionFactorCal3_S2(measured, create_shared_noise_model(model), poseKey, pointKey, boost::make_shared<Cal3_S2>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"), py::arg("K"))         
    .def("get_weight", &WeightedGenericProjectionFactorCal3_S2::getWeight)
    .def("set_weight", &WeightedGenericProjectionFactorCal3_S2::setWeight);


    py::class_<WeightedGenericStereoProjectionFactor3D, gtsam::NonlinearFactor, std::shared_ptr<WeightedGenericStereoProjectionFactor3D>>(m, "WeightedGenericStereoProjectionFactor3D")
    .def(py::init([] (const StereoPoint2& measured, const SharedNoiseModel& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
            return new WeightedGenericStereoProjectionFactor3D(measured, model, poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K")) 
    .def(py::init([] (const StereoPoint2& measured, const noiseModel::Diagonal& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoProjectionFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K"))
    .def(py::init([] (const StereoPoint2& measured, const noiseModel::Robust& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoProjectionFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K")) 
    .def(py::init([] (const StereoPoint2& measured, const SwitchableRobustNoiseModel& model, const Key& poseKey, const Key& landmarkKey, const Cal3_S2Stereo& K) {
        return new WeightedGenericStereoProjectionFactor3D(measured, create_shared_noise_model(model), poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K), boost::none);
    }), py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"), py::arg("K"))              
    .def("get_weight", &WeightedGenericStereoProjectionFactor3D::getWeight)
    .def("set_weight", &WeightedGenericStereoProjectionFactor3D::setWeight);


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
    .def(py::init<const gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Diagonal& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimResectioningFactor::evaluateError)
    .def("get_weight", &SimResectioningFactor::getWeight)
    .def("set_weight", &SimResectioningFactor::setWeight);


    py::class_<SimInvResectioningFactor, gtsam::NoiseModelFactor, std::shared_ptr<SimInvResectioningFactor>>(m, "SimInvResectioningFactor")
    .def(py::init<const gtsam::Key&, const gtsam::Cal3_S2&, const gtsam::Point2&, const gtsam::Point3&, const gtsam::SharedNoiseModel&>(),
            py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Diagonal& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def(py::init([](const gtsam::Key& sim_pose_key, const gtsam::Cal3_S2& calib, const gtsam::Point2& p, const gtsam::Point3& P, const gtsam::noiseModel::Robust& model){
        return new SimInvResectioningFactor(sim_pose_key, calib, p, P, create_shared_noise_model(model));       
    }), py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"), py::arg("model"))
    .def("evaluateError", &SimInvResectioningFactor::evaluateError)
    .def("get_weight", &SimInvResectioningFactor::getWeight)
    .def("set_weight", &SimInvResectioningFactor::setWeight);

    
    m.def("insert_similarity3", &insertSimilarity3, "Insert Similarity3 into Values",
        py::arg("values"), py::arg("key"), py::arg("sim3"));    

    m.def("get_similarity3", &getSimilarity3, "Get Similarity3 from Values",
        py::arg("values"), py::arg("key"));
    
    m.def("insert_pose3", &insertPose3, "Insert Pose3 into Values",
        py::arg("values"), py::arg("key"), py::arg("pose"));
    
    m.def("get_pose3", &getPose3, "Get Pose3 from Values",
        py::arg("values"), py::arg("key"));

#if 0
    // Expose BetweenFactor with Similarity3
    py::class_<BetweenFactor<Similarity3>, gtsam::NonlinearFactor, std::shared_ptr<BetweenFactor<Similarity3>>>(m, "BetweenFactorSimilarity3")
    .def(py::init<const Key&, const Key&, const Similarity3&>(), 
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"))
    .def(py::init<const Key&, const Key&, const Similarity3&, const SharedNoiseModel&>(),
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Diagonal& model) {
        return std::make_shared<BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Robust& model) {
        return std::make_shared<BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def("evaluateError", &BetweenFactor<Similarity3>::evaluateError)
    .def("print", &BetweenFactor<Similarity3>::print)
    .def("measured", &BetweenFactor<Similarity3>::measured)
    .def("__repr__", [](const BetweenFactor<Similarity3>& self) {
        std::ostringstream ss;
        self.print(ss.str());
        return ss.str();
    });
#else 
    py::class_<BetweenFactorSimilarity3, gtsam::NoiseModelFactor2<Similarity3, Similarity3>, std::shared_ptr<BetweenFactorSimilarity3>>(m, "BetweenFactorSimilarity3")
    .def(py::init<const gtsam::Key&, const gtsam::Key&, const gtsam::Similarity3&, const gtsam::SharedNoiseModel&>(),
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
    .def(py::init([](const gtsam::Key& key1, const gtsam::Key& key2, const gtsam::Similarity3& similarity3, const gtsam::noiseModel::Diagonal& model) {
        return new BetweenFactorSimilarity3(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def(py::init([](const gtsam::Key& key1, const gtsam::Key& key2, const gtsam::Similarity3& similarity3, const gtsam::noiseModel::Robust& model) {
        return new BetweenFactorSimilarity3(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def("evaluateError", &BetweenFactorSimilarity3::evaluateError)
    .def("print", &BetweenFactorSimilarity3::print)
    .def("__repr__", [](const BetweenFactorSimilarity3& self) {
        std::ostringstream ss;
        self.print(ss.str());
        return ss.str();    
    });
#endif


    py::class_<LevenbergMarquardtOptimizerG2o, gtsam::NonlinearOptimizer, std::shared_ptr<LevenbergMarquardtOptimizerG2o>>(m, "LevenbergMarquardtOptimizerG2o")
    .def(py::init([](const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initialValues, const gtsam::Ordering& ordering, const gtsam::LevenbergMarquardtParams& params) {
        return new LevenbergMarquardtOptimizerG2o(graph, initialValues, ordering, params);
    }), py::arg("graph"), py::arg("initialValues"), py::arg("ordering"), py::arg("params"))
    .def(py::init([](const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& initialValues, const gtsam::LevenbergMarquardtParams& params) {
        return new LevenbergMarquardtOptimizerG2o(graph, initialValues, params);
    }), py::arg("graph"), py::arg("initialValues"), py::arg("params"))
    .def("optimize", &LevenbergMarquardtOptimizerG2o::optimize);  

}