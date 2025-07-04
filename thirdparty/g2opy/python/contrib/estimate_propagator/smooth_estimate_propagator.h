#include <pybind11/pybind11.h>

#include "contrib/estimate_propagator/SmoothEstimatePropagator.hpp"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareTypesEstimatePropagator(py::module & m) {

    py::class_<SmoothEstimatePropagator, EstimatePropagator>(m, "SmoothEstimatePropagator")
        .def(py::init<SparseOptimizer*, const double&, const double&>(),
            "g"_a, 
            "maxDistance"_a=std::numeric_limits<double>::max(),
            "maxEdgeCost"_a=std::numeric_limits<double>::max(),
            py::keep_alive<1, 2>())
        .def("propagate", &SmoothEstimatePropagator::propagate,
            "v"_a,
            py::keep_alive<1, 2>())   // (OptimizableGraph::Vertex*) -> void


    ;

}

}