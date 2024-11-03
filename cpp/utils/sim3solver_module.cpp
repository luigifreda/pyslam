#include <pybind11/pybind11.h>

#include <sstream> 

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "pybind11/eigen.h"
#include "opencv_type_casters.h"
#include "Sim3Solver.h"

namespace py = pybind11;
using namespace pybind11::literals;


using namespace utils;

PYBIND11_MODULE(sim3solver, m) 
{
    // optional module docstring
    m.doc() = "pybind11 plugin for sim3solver module";
    

    py::class_<Sim3SolverInput>(m, "Sim3SolverInput")
        .def(py::init<>())
        .def_readwrite("points_3d_w1", &Sim3SolverInput::mvX3Dw1)
        .def_readwrite("points_3d_w2", &Sim3SolverInput::mvX3Dw2)
        .def_readwrite("sigmas2_1", &Sim3SolverInput::mvSigmaSquare1)
        .def_readwrite("sigmas2_2", &Sim3SolverInput::mvSigmaSquare2)
        .def_readwrite("K1", &Sim3SolverInput::K1)
        .def_readwrite("Rcw1", &Sim3SolverInput::Rcw1)
        .def_readwrite("tcw1", &Sim3SolverInput::tcw1)
        .def_readwrite("K2", &Sim3SolverInput::K2)
        .def_readwrite("Rcw2", &Sim3SolverInput::Rcw2)
        .def_readwrite("tcw2", &Sim3SolverInput::tcw2)
        .def_readwrite("fix_scale", &Sim3SolverInput::bFixScale);


    py::class_<Sim3Solver>(m, "Sim3Solver")
        .def(py::init<const Sim3SolverInput&>())
        .def("set_ransac_parameters", &Sim3Solver::SetRansacParameters,
            "probability"_a = 0.99, "minInliers"_a = 6, "maxIterations"_a = 300)
        .def("find", [](Sim3Solver& s){
            std::vector<uint8_t> vbInliers;
            int nInliers;
            bool bConverged;
            const Eigen::Matrix4f transformation = s.find(vbInliers, nInliers, bConverged);
            return std::make_tuple(transformation, vbInliers, nInliers, bConverged);
        })
        .def("iterate", [](Sim3Solver& s, const int nIterations){
            std::vector<uint8_t> vbInliers; 
            int nInliers;
            bool bNoMore;
            bool bConverged;
            const Eigen::Matrix4f transformation = s.iterate(nIterations, bNoMore, vbInliers, nInliers, bConverged);
            return std::make_tuple(transformation, bNoMore, vbInliers, nInliers, bConverged);
            }, "nIterations"_a)
        .def("get_estimated_transformation", &Sim3Solver::GetEstimatedTransformation)
        .def("get_estimated_scale", &Sim3Solver::GetEstimatedScale)
        .def("get_estimated_rotation", &Sim3Solver::GetEstimatedRotation)
        .def("get_estimated_translation", &Sim3Solver::GetEstimatedTranslation)
        .def("compute_3d_registration_error", &Sim3Solver::Compute3dRegistrationError);
}
