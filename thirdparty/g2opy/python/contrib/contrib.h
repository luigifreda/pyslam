#include <pybind11/pybind11.h>

#include "sba_addons/types_stereo_sba.h"
#include "estimate_propagator/smooth_estimate_propagator.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareContrib(py::module & m) {

    // sba_addons
    declareTypesStereoSBA(m);

    py::module contrib = m.def_submodule("contrib", "Contrib part of the library'");

    // estimate propagator
    declareTypesEstimatePropagator(contrib);

}

}