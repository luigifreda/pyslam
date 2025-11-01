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

// #define GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR  // before including gtsam

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "numerical_derivative.h"
#include "numerical_derivative_py.h"
#include "optimizers.h"
#include "resectioning.h"
#include "similarity.h"
#include "switchable_robust_noise_model.h"
#include "weighted_projection_factors.h"

#include <iostream>

namespace py = pybind11;


/* Please refer to:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
 * These are required to save one copy operation on Python calls.
 *
 * NOTES
 * =================
 *
 * `PYBIND11_MAKE_OPAQUE` will mark the type as "opaque" for the pybind11
 * automatic STL binding, such that the raw objects can be accessed in Python.
 * Without this they will be automatically converted to a Python object, and all
 * mutations on Python side will not be reflected on C++.
 */
// Declare large container types as opaque BEFORE module definition
// This ensures they are passed by reference/pointer rather than copied
// IMPORTANT: Must be declared before PYBIND11_MODULE
PYBIND11_MAKE_OPAQUE(gtsam::Values);
PYBIND11_MAKE_OPAQUE(gtsam::NonlinearFactorGraph);
PYBIND11_MAKE_OPAQUE(gtsam::Ordering);


PYBIND11_MODULE(gtsam_factors, m) {
    py::module_::import("gtsam"); // Ensure GTSAM is loaded

    // Register NoiseModelFactorN classes
    py::class_<NoiseModelFactorN<Pose3>, std::shared_ptr<NoiseModelFactorN<Pose3>>,
               NonlinearFactor>(m, "NoiseModelFactorN_Pose3");
    py::class_<NoiseModelFactorN<Similarity3>, std::shared_ptr<NoiseModelFactorN<Similarity3>>,
               NonlinearFactor>(m, "NoiseModelFactorN_Similarity3");

    // Register base class for BetweenFactorSimilarity3
    py::class_<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>,
               gtsam::NonlinearFactor,
               std::shared_ptr<gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>>>(
        m, "NoiseModelFactor2Similarity3")
        .def("print", &gtsam::NoiseModelFactor2<gtsam::Similarity3, gtsam::Similarity3>::print);

    // NOTE: This commented block does not work. We need to specialize the template for each type we
    // are interested
    //       as it is done for Similarity3 in numerical_derivative11_v2_sim3 and the other following
    //       functions.
    // m.def("numerical_derivative11",
    //     [](py::function py_func, const auto& x, double delta = 1e-5) -> Eigen::MatrixXd {
    //         // Explicitly define the lambda signature
    //         return numericalDerivative11_Any(py_func, x, delta);
    //     },
    //     py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
    //     "Compute numerical derivative of a function mapping any type");

    // Specialization of numericalDerivative11 for Pose3 -> Vector2
    m.def(
        "numerical_derivative11_v2_pose3",
        [](py::function py_func, const Pose3 &x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return gtsam_factors::numericalDerivative11_V2_Pose3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Pose3 to Vector2");

    m.def(
        "numerical_derivative11_v3_pose3",
        [](py::function py_func, const Pose3 &x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return gtsam_factors::numericalDerivative11_V3_Pose3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Pose3 to Vector3");

    // Specialization of numericalDerivative11 for Similarity3 -> Vector2
    m.def(
        "numerical_derivative11_v2_sim3",
        [](py::function py_func, const Similarity3 &x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return gtsam_factors::numericalDerivative11_V2_Sim3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Similarity3 to Vector2");

    // Specialization of numericalDerivative11 for Similarity3 -> Vector3
    m.def(
        "numerical_derivative11_v3_sim3",
        [](py::function py_func, const Similarity3 &x, double delta = 1e-5) -> Eigen::MatrixXd {
            // Explicitly define the lambda signature
            return gtsam_factors::numericalDerivative11_V3_Sim3(py_func, x, delta);
        },
        py::arg("func"), py::arg("x"), py::arg("delta") = 1e-5,
        "Compute numerical derivative of a function mapping Similarity3 to Vector3");

    py::class_<gtsam_factors::SwitchableRobustNoiseModel,
               std::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>, noiseModel::Base>(
        m, "SwitchableRobustNoiseModel")
        .def(py::init([](int dim, double sigma, double huberThreshold) {
                 return new gtsam_factors::SwitchableRobustNoiseModel(dim, sigma, huberThreshold);
             }),
             py::arg("dim"), py::arg("sigma"), py::arg("huberThreshold"))
        .def(py::init([](const Vector &sigmas, double huberThreshold) {
                 return new gtsam_factors::SwitchableRobustNoiseModel(sigmas, huberThreshold);
             }),
             py::arg("sigmas"), py::arg("huberThreshold"))
        .def("set_robust_model_active",
             &gtsam_factors::SwitchableRobustNoiseModel::setRobustModelActive)
        .def("get_robust_model", &gtsam_factors::SwitchableRobustNoiseModel::getRobustModel)
        .def("get_diagonal_model", &gtsam_factors::SwitchableRobustNoiseModel::getDiagonalModel);

    py::class_<gtsam_factors::ResectioningFactor,
               std::shared_ptr<gtsam_factors::ResectioningFactor>, NoiseModelFactorN<Pose3>>(
        m, "ResectioningFactor")
        .def(py::init([](const SharedNoiseModel &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactor(model, key, calib, measured_p,
                                                              world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Diagonal &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactor(create_shared_noise_model(model), key,
                                                              calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Robust &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactor(create_shared_noise_model(model), key,
                                                              calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &key,
                         const Cal3_S2 &calib, const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactor(create_shared_noise_model(model), key,
                                                              calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def("evaluateError", &gtsam_factors::ResectioningFactor::evaluateError)
        .def("error", &gtsam_factors::ResectioningFactor::error)
        .def("set_weight", &gtsam_factors::ResectioningFactor::setWeight)
        .def("get_weight", &gtsam_factors::ResectioningFactor::getWeight);

    py::class_<gtsam_factors::ResectioningFactorTcw,
               std::shared_ptr<gtsam_factors::ResectioningFactorTcw>, NoiseModelFactorN<Pose3>>(
        m, "ResectioningFactorTcw")
        .def(py::init([](const SharedNoiseModel &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorTcw(model, key, calib, measured_p,
                                                                 world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Diagonal &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorTcw(create_shared_noise_model(model),
                                                                 key, calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Robust &model, const Key &key, const Cal3_S2 &calib,
                         const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorTcw(create_shared_noise_model(model),
                                                                 key, calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def(py::init([](const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &key,
                         const Cal3_S2 &calib, const Point2 &measured_p, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorTcw(create_shared_noise_model(model),
                                                                 key, calib, measured_p, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p"),
             py::arg("world_P"))
        .def("evaluateError", &gtsam_factors::ResectioningFactorTcw::evaluateError)
        .def("error", &gtsam_factors::ResectioningFactorTcw::error)
        .def("set_weight", &gtsam_factors::ResectioningFactorTcw::setWeight)
        .def("get_weight", &gtsam_factors::ResectioningFactorTcw::getWeight);

    py::class_<gtsam_factors::ResectioningFactorStereo,
               std::shared_ptr<gtsam_factors::ResectioningFactorStereo>, NoiseModelFactorN<Pose3>>(
        m, "ResectioningFactorStereo")
        .def(py::init([](const SharedNoiseModel &model, const Key &key, const Cal3_S2Stereo &calib,
                         const StereoPoint2 &measured_p_stereo, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereo(model, key, calib,
                                                                    measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Diagonal &model, const Key &key,
                         const Cal3_S2Stereo &calib, const StereoPoint2 &measured_p_stereo,
                         const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereo(
                     create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def(
            py::init([](const noiseModel::Robust &model, const Key &key, const Cal3_S2Stereo &calib,
                        const StereoPoint2 &measured_p_stereo, const Point3 &world_P) {
                return new gtsam_factors::ResectioningFactorStereo(
                    create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
            }),
            py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
            py::arg("world_P"))
        .def(py::init([](const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &key,
                         const Cal3_S2Stereo &calib, const StereoPoint2 &measured_p_stereo,
                         const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereo(
                     create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def("evaluateError", &gtsam_factors::ResectioningFactorStereo::evaluateError)
        .def("error", &gtsam_factors::ResectioningFactorStereo::error)
        .def("set_weight", &gtsam_factors::ResectioningFactorStereo::setWeight)
        .def("get_weight", &gtsam_factors::ResectioningFactorStereo::getWeight);

    py::class_<gtsam_factors::ResectioningFactorStereoTcw,
               std::shared_ptr<gtsam_factors::ResectioningFactorStereoTcw>,
               NoiseModelFactorN<Pose3>>(m, "ResectioningFactorStereoTcw")
        .def(py::init([](const SharedNoiseModel &model, const Key &key, const Cal3_S2Stereo &calib,
                         const StereoPoint2 &measured_p_stereo, const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereoTcw(model, key, calib,
                                                                       measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def(py::init([](const noiseModel::Diagonal &model, const Key &key,
                         const Cal3_S2Stereo &calib, const StereoPoint2 &measured_p_stereo,
                         const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereoTcw(
                     create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def(
            py::init([](const noiseModel::Robust &model, const Key &key, const Cal3_S2Stereo &calib,
                        const StereoPoint2 &measured_p_stereo, const Point3 &world_P) {
                return new gtsam_factors::ResectioningFactorStereoTcw(
                    create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
            }),
            py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
            py::arg("world_P"))
        .def(py::init([](const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &key,
                         const Cal3_S2Stereo &calib, const StereoPoint2 &measured_p_stereo,
                         const Point3 &world_P) {
                 return new gtsam_factors::ResectioningFactorStereoTcw(
                     create_shared_noise_model(model), key, calib, measured_p_stereo, world_P);
             }),
             py::arg("noise_model"), py::arg("key"), py::arg("calib"), py::arg("measured_p_stereo"),
             py::arg("world_P"))
        .def("evaluateError", &gtsam_factors::ResectioningFactorStereoTcw::evaluateError)
        .def("error", &gtsam_factors::ResectioningFactorStereoTcw::error)
        .def("set_weight", &gtsam_factors::ResectioningFactorStereoTcw::setWeight)
        .def("get_weight", &gtsam_factors::ResectioningFactorStereoTcw::getWeight);

    py::class_<gtsam_factors::WeightedGenericProjectionFactorCal3_S2, gtsam::NonlinearFactor,
               std::shared_ptr<gtsam_factors::WeightedGenericProjectionFactorCal3_S2>>(
        m, "WeightedGenericProjectionFactorCal3_S2")
        .def(py::init([](const Point2 &measured, const SharedNoiseModel &model, const Key &poseKey,
                         const Key &pointKey, const Cal3_S2 &K) {
                 return new gtsam_factors::WeightedGenericProjectionFactorCal3_S2(
                     measured, model, poseKey, pointKey, boost::make_shared<Cal3_S2>(K),
                     boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"),
             py::arg("K"))
        .def(py::init([](const Point2 &measured, const noiseModel::Diagonal &model,
                         const Key &poseKey, const Key &pointKey, const Cal3_S2 &K) {
                 return new gtsam_factors::WeightedGenericProjectionFactorCal3_S2(
                     measured, create_shared_noise_model(model), poseKey, pointKey,
                     boost::make_shared<Cal3_S2>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"),
             py::arg("K"))
        .def(py::init([](const Point2 &measured, const noiseModel::Robust &model,
                         const Key &poseKey, const Key &pointKey, const Cal3_S2 &K) {
                 return new gtsam_factors::WeightedGenericProjectionFactorCal3_S2(
                     measured, create_shared_noise_model(model), poseKey, pointKey,
                     boost::make_shared<Cal3_S2>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"),
             py::arg("K"))
        .def(py::init([](const Point2 &measured,
                         const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &poseKey,
                         const Key &pointKey, const Cal3_S2 &K) {
                 return new gtsam_factors::WeightedGenericProjectionFactorCal3_S2(
                     measured, create_shared_noise_model(model), poseKey, pointKey,
                     boost::make_shared<Cal3_S2>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("pointKey"),
             py::arg("K"))
        .def("get_weight", &gtsam_factors::WeightedGenericProjectionFactorCal3_S2::getWeight)
        .def("set_weight", &gtsam_factors::WeightedGenericProjectionFactorCal3_S2::setWeight);

    py::class_<gtsam_factors::WeightedGenericStereoProjectionFactor3D, gtsam::NonlinearFactor,
               std::shared_ptr<gtsam_factors::WeightedGenericStereoProjectionFactor3D>>(
        m, "WeightedGenericStereoProjectionFactor3D")
        .def(py::init([](const StereoPoint2 &measured, const SharedNoiseModel &model,
                         const Key &poseKey, const Key &landmarkKey, const Cal3_S2Stereo &K) {
                 return new gtsam_factors::WeightedGenericStereoProjectionFactor3D(
                     measured, model, poseKey, landmarkKey, boost::make_shared<Cal3_S2Stereo>(K),
                     boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"),
             py::arg("K"))
        .def(py::init([](const StereoPoint2 &measured, const noiseModel::Diagonal &model,
                         const Key &poseKey, const Key &landmarkKey, const Cal3_S2Stereo &K) {
                 return new gtsam_factors::WeightedGenericStereoProjectionFactor3D(
                     measured, create_shared_noise_model(model), poseKey, landmarkKey,
                     boost::make_shared<Cal3_S2Stereo>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"),
             py::arg("K"))
        .def(py::init([](const StereoPoint2 &measured, const noiseModel::Robust &model,
                         const Key &poseKey, const Key &landmarkKey, const Cal3_S2Stereo &K) {
                 return new gtsam_factors::WeightedGenericStereoProjectionFactor3D(
                     measured, create_shared_noise_model(model), poseKey, landmarkKey,
                     boost::make_shared<Cal3_S2Stereo>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"),
             py::arg("K"))
        .def(py::init([](const StereoPoint2 &measured,
                         const gtsam_factors::SwitchableRobustNoiseModel &model, const Key &poseKey,
                         const Key &landmarkKey, const Cal3_S2Stereo &K) {
                 return new gtsam_factors::WeightedGenericStereoProjectionFactor3D(
                     measured, create_shared_noise_model(model), poseKey, landmarkKey,
                     boost::make_shared<Cal3_S2Stereo>(K), boost::none);
             }),
             py::arg("measured"), py::arg("model"), py::arg("poseKey"), py::arg("landmarkKey"),
             py::arg("K"))
        .def("get_weight", &gtsam_factors::WeightedGenericStereoProjectionFactor3D::getWeight)
        .def("set_weight", &gtsam_factors::WeightedGenericStereoProjectionFactor3D::setWeight);

    py::class_<gtsam_factors::PriorFactorSimilarity3, gtsam::NoiseModelFactor1<gtsam::Similarity3>,
               std::shared_ptr<gtsam_factors::PriorFactorSimilarity3>>(m, "PriorFactorSimilarity3")
        .def(py::init<const gtsam::Key &, const gtsam::Similarity3 &,
                      const gtsam::SharedNoiseModel &>(),
             py::arg("key"), py::arg("prior"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key, const gtsam::Similarity3 &prior,
                         const gtsam::noiseModel::Isotropic &model) {
                 return new gtsam_factors::PriorFactorSimilarity3(key, prior,
                                                                  create_shared_noise_model(model));
             }),
             py::arg("key"), py::arg("prior"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key, const gtsam::Similarity3 &prior,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::PriorFactorSimilarity3(key, prior,
                                                                  create_shared_noise_model(model));
             }),
             py::arg("key"), py::arg("prior"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key, const gtsam::Similarity3 &prior,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::PriorFactorSimilarity3(key, prior,
                                                                  create_shared_noise_model(model));
             }),
             py::arg("key"), py::arg("prior"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key, const gtsam::Similarity3 &prior,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::PriorFactorSimilarity3(key, prior,
                                                                  create_shared_noise_model(model));
             }),
             py::arg("key"), py::arg("prior"), py::arg("model"))
        .def("evaluateError", &gtsam_factors::PriorFactorSimilarity3::evaluateError);

    py::class_<gtsam_factors::PriorFactorSimilarity3ScaleOnly,
               gtsam::NoiseModelFactor1<gtsam::Similarity3>,
               std::shared_ptr<gtsam_factors::PriorFactorSimilarity3ScaleOnly>>(
        m, "PriorFactorSimilarity3ScaleOnly")
        .def(py::init<const gtsam::Key &, double, double>(), py::arg("key"), py::arg("prior_scale"),
             py::arg("prior_sigma"))
        .def("evaluateError", &gtsam_factors::PriorFactorSimilarity3ScaleOnly::evaluateError);

    py::class_<gtsam_factors::SimResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<gtsam_factors::SimResectioningFactor>>(m, "SimResectioningFactor")
        .def(py::init<const gtsam::Key &, const gtsam::Cal3_S2 &, const gtsam::Point2 &,
                      const gtsam::Point3 &, const gtsam::SharedNoiseModel &>(),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::SimResectioningFactor(sim_pose_key, calib, p, P,
                                                                 create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::SimResectioningFactor(sim_pose_key, calib, p, P,
                                                                 create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::SimResectioningFactor(sim_pose_key, calib, p, P,
                                                                 create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def("evaluateError", &gtsam_factors::SimResectioningFactor::evaluateError)
        .def("error", &gtsam_factors::SimResectioningFactor::error)
        .def("get_weight", &gtsam_factors::SimResectioningFactor::getWeight)
        .def("set_weight", &gtsam_factors::SimResectioningFactor::setWeight);

    py::class_<gtsam_factors::SimInvResectioningFactor, gtsam::NoiseModelFactor,
               std::shared_ptr<gtsam_factors::SimInvResectioningFactor>>(m,
                                                                         "SimInvResectioningFactor")
        .def(py::init<const gtsam::Key &, const gtsam::Cal3_S2 &, const gtsam::Point2 &,
                      const gtsam::Point3 &, const gtsam::SharedNoiseModel &>(),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::SimInvResectioningFactor(
                     sim_pose_key, calib, p, P, create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::SimInvResectioningFactor(
                     sim_pose_key, calib, p, P, create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def(py::init([](const gtsam::Key &sim_pose_key, const gtsam::Cal3_S2 &calib,
                         const gtsam::Point2 &p, const gtsam::Point3 &P,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::SimInvResectioningFactor(
                     sim_pose_key, calib, p, P, create_shared_noise_model(model));
             }),
             py::arg("sim_pose_key"), py::arg("calib"), py::arg("p"), py::arg("P"),
             py::arg("model"))
        .def("evaluateError", &gtsam_factors::SimInvResectioningFactor::evaluateError)
        .def("error", &gtsam_factors::SimInvResectioningFactor::error)
        .def("get_weight", &gtsam_factors::SimInvResectioningFactor::getWeight)
        .def("set_weight", &gtsam_factors::SimInvResectioningFactor::setWeight);

    m.def("insert_similarity3", &gtsam_factors::insertSimilarity3, "Insert Similarity3 into Values",
          py::arg("values"), py::arg("key"), py::arg("sim3"));

    m.def("get_similarity3", &gtsam_factors::getSimilarity3, "Get Similarity3 from Values",
          py::arg("values"), py::arg("key"));

#if 0
    // Expose BetweenFactor with Similarity3
    py::class_<gtsam_factors::BetweenFactor<Similarity3>, gtsam::NonlinearFactor, std::shared_ptr<gtsam_factors::BetweenFactor<Similarity3>>>(m, "BetweenFactorSimilarity3")
    .def(py::init<const Key&, const Key&, const Similarity3&>(), 
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"))
    .def(py::init<const Key&, const Key&, const Similarity3&, const SharedNoiseModel&>(),
        py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Diagonal& model) {
        return std::make_shared<gtsam_factors::BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def(py::init([](const Key& key1, const Key& key2, const Similarity3& similarity3, const noiseModel::Robust& model) {
        return std::make_shared<gtsam_factors::BetweenFactor<Similarity3>>(key1, key2, similarity3, create_shared_noise_model(model));       
    }), py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
    .def("evaluateError", &gtsam_factors::BetweenFactor<Similarity3>::evaluateError)
    .def("print", &gtsam_factors::BetweenFactor<Similarity3>::print)
    .def("measured", &gtsam_factors::BetweenFactor<Similarity3>::measured)
    .def("__repr__", [](const gtsam_factors::BetweenFactor<Similarity3>& self) {
        std::ostringstream ss;
        ss << "BetweenFactorSimilarity3(key1=" << self.key1() << ", key2=" << self.key2() << ", similarity3=" << self.measured_.matrix() << ")";
        self.print(ss.str());
        return ss.str();
    });
#else
    py::class_<gtsam_factors::BetweenFactorSimilarity3,
               gtsam::NoiseModelFactor2<Similarity3, Similarity3>,
               std::shared_ptr<gtsam_factors::BetweenFactorSimilarity3>>(m,
                                                                         "BetweenFactorSimilarity3")
        .def(py::init<const gtsam::Key &, const gtsam::Key &, const gtsam::Similarity3 &,
                      const gtsam::SharedNoiseModel &>(),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def("evaluateError", &gtsam_factors::BetweenFactorSimilarity3::evaluateError)
        .def("print", &gtsam_factors::BetweenFactorSimilarity3::print)
        .def("__repr__", [](const gtsam_factors::BetweenFactorSimilarity3 &self) {
            std::ostringstream ss;
            ss << "BetweenFactorSimilarity3(key1=" << self.key1() << ", key2=" << self.key2()
               //<< ", similarity3=" << self.measured_.matrix() << ")";
               << ")";
            self.print(ss.str());
            return ss.str();
        });
#endif

    py::class_<gtsam_factors::BetweenFactorSimilarity3Inverse,
               gtsam::NoiseModelFactor2<Similarity3, Similarity3>,
               std::shared_ptr<gtsam_factors::BetweenFactorSimilarity3Inverse>>(
        m, "BetweenFactorSimilarity3Inverse")
        .def(py::init<const gtsam::Key &, const gtsam::Key &, const gtsam::Similarity3 &,
                      const gtsam::SharedNoiseModel &>(),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3Inverse(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3Inverse(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key1, const gtsam::Key &key2,
                         const gtsam::Similarity3 &similarity3,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3Inverse(
                     key1, key2, similarity3, create_shared_noise_model(model));
             }),
             py::arg("key1"), py::arg("key2"), py::arg("similarity3"), py::arg("model"))
        .def("evaluateError", &gtsam_factors::BetweenFactorSimilarity3Inverse::evaluateError)
        .def("print", &gtsam_factors::BetweenFactorSimilarity3Inverse::print)
        .def("__repr__", [](const gtsam_factors::BetweenFactorSimilarity3Inverse &self) {
            std::ostringstream ss;
            ss << "BetweenFactorSimilarity3Inverse(key1=" << self.key1() << ", key2=" << self.key2()
               //<< ", similarity3=" << self.measured_.matrix() << ")";
               << ")";
            self.print(ss.str());
            return ss.str();
        });


    py::class_<gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1,
               gtsam::NoiseModelFactor1<Similarity3>,
               std::shared_ptr<gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1>>(
        m, "BetweenFactorSimilarity3InverseOnlyS1")
        .def(py::init<const gtsam::Key &, const gtsam::Similarity3 &, const gtsam::Similarity3 &,
                      const gtsam::SharedNoiseModel &>(),
             py::arg("key_sim3_1"), py::arg("sim3_2"), py::arg("measured"), py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &key_sim3_1, 
                         const gtsam::Similarity3 &sim3_2,
                         const gtsam::Similarity3 &measured,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1(
                     key_sim3_1, sim3_2, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_1"), py::arg("sim3_2"), py::arg("measured"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key_sim3_1, 
                         const gtsam::Similarity3 &sim3_2,
                         const gtsam::Similarity3 &measured,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1(
                     key_sim3_1, sim3_2, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_1"), py::arg("sim3_2"), py::arg("measured"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key_sim3_1, 
                         const gtsam::Similarity3 &sim3_2,
                         const gtsam::Similarity3 &measured,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1(
                     key_sim3_1, sim3_2, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_1"), py::arg("sim3_2"), py::arg("measured"), py::arg("model"))
        .def("evaluateError", &gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1::evaluateError)
        .def("print", &gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1::print)
        .def("__repr__", [](const gtsam_factors::BetweenFactorSimilarity3InverseOnlyS1 &self) {
            std::ostringstream ss;
            ss << "BetweenFactorSimilarity3InverseOnlyS1(key=" << self.key()
               //<< ", similarity3=" << self.measured_.matrix() << ")";
               << ")";
            self.print(ss.str());
            return ss.str();
        });
        
    py::class_<gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2,
               gtsam::NoiseModelFactor1<Similarity3>,
               std::shared_ptr<gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2>>(
        m, "BetweenFactorSimilarity3InverseOnlyS2")
        .def(py::init<const gtsam::Key &, const gtsam::Similarity3 &, const gtsam::Similarity3 &,
                      const gtsam::SharedNoiseModel &>(),
             py::arg("key_sim3_2"), py::arg("sim3_1"), py::arg("measured"), py::arg("noise_model"))
        .def(py::init([](const gtsam::Key &key_sim3_2, 
                         const gtsam::Similarity3 &sim3_1,
                         const gtsam::Similarity3 &measured,
                         const gtsam::noiseModel::Diagonal &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2(
                     key_sim3_2, sim3_1, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_2"), py::arg("sim3_1"), py::arg("measured"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key_sim3_2, 
                         const gtsam::Similarity3 &sim3_1,
                         const gtsam::Similarity3 &measured,
                         const gtsam::noiseModel::Robust &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2(
                     key_sim3_2, sim3_1, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_2"), py::arg("sim3_1"), py::arg("measured"), py::arg("model"))
        .def(py::init([](const gtsam::Key &key_sim3_2, 
                         const gtsam::Similarity3 &sim3_1,
                         const gtsam::Similarity3 &measured,
                         const gtsam_factors::SwitchableRobustNoiseModel &model) {
                 return new gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2(
                     key_sim3_2, sim3_1, measured, create_shared_noise_model(model));
             }),
             py::arg("key_sim3_2"), py::arg("sim3_1"), py::arg("measured"), py::arg("model"))
        .def("evaluateError", &gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2::evaluateError)
        .def("print", &gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2::print)
        .def("__repr__", [](const gtsam_factors::BetweenFactorSimilarity3InverseOnlyS2 &self) {
            std::ostringstream ss;
            ss << "BetweenFactorSimilarity3InverseOnlyS2(key=" << self.key()
               //<< ", similarity3=" << self.measured_.matrix() << ")";
               << ")";
            self.print(ss.str());
            return ss.str();
        });

    py::class_<gtsam_factors::LevenbergMarquardtOptimizerG2o, gtsam::NonlinearOptimizer,
               std::shared_ptr<gtsam_factors::LevenbergMarquardtOptimizerG2o>>(
        m, "LevenbergMarquardtOptimizerG2o")
        .def(py::init([](const gtsam::NonlinearFactorGraph &graph,
                         const gtsam::Values &initialValues, const gtsam::Ordering &ordering,
                         const gtsam::LevenbergMarquardtParams &params) {
                 return new gtsam_factors::LevenbergMarquardtOptimizerG2o(graph, initialValues,
                                                                          ordering, params);
             }),
             py::arg("graph"), py::arg("initialValues"), py::arg("ordering"), py::arg("params"))
        .def(py::init([](const gtsam::NonlinearFactorGraph &graph,
                         const gtsam::Values &initialValues,
                         const gtsam::LevenbergMarquardtParams &params) {
                 return new gtsam_factors::LevenbergMarquardtOptimizerG2o(graph, initialValues,
                                                                          params);
             }),
             py::arg("graph"), py::arg("initialValues"), py::arg("params"))
        .def("optimize", &gtsam_factors::LevenbergMarquardtOptimizerG2o::optimize);
}