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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/linear/NoiseModel.h>

#include <boost/shared_ptr.hpp>  // Include Boost

#include <iostream>

namespace py = pybind11;

using namespace gtsam;
//using namespace gtsam::noiseModel;
using symbol_shorthand::X;


template <typename T>
boost::shared_ptr<T> create_shared_noise_model(const T& model) {
    return boost::make_shared<T>(model);
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


PYBIND11_MODULE(gtsam_factors, m) {
    py::module_::import("gtsam");  // Ensure GTSAM is loaded

    py::class_<NoiseModelFactorN<Pose3>, std::shared_ptr<NoiseModelFactorN<Pose3>>, NonlinearFactor>(m, "NoiseModelFactorN_Pose3");  

    py::class_<ResectioningFactor, std::shared_ptr<ResectioningFactor>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactor")
    .def(py::init([](const noiseModel::Isotropic& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2& calib, const Point2& measured_p, const Point3& world_P) {
        return new ResectioningFactor(create_shared_noise_model(model), key, calib, measured_p, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactor::evaluateError)
    .def("error", &ResectioningFactor::error);

    py::class_<ResectioningFactorStereo, std::shared_ptr<ResectioningFactorStereo>, NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereo")
    .def(py::init([](const noiseModel::Isotropic& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }),py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def(py::init([](const noiseModel::Robust& model, const Key& key, const Cal3_S2Stereo& calib, const StereoPoint2& measured_p_stereo, const Point3& world_P) {
        return new ResectioningFactorStereo(create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
    }), py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"), py::arg("world_P"))
    .def("evaluateError", &ResectioningFactorStereo::evaluateError)
    .def("error", &ResectioningFactorStereo::error);

}