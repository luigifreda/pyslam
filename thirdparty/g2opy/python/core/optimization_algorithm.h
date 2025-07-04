#include <pybind11/pybind11.h>

#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_property.h>
#include <g2o/core/optimization_algorithm_with_hessian.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_factory.h>

#include "block_solver.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareOptimizationAlgorithm(py::module & m) {

    py::class_<OptimizationAlgorithm>(m, "OptimizationAlgorithm");

    py::class_<OptimizationAlgorithmProperty>(m, "OptimizationAlgorithmProperty");
    
    

    py::class_<OptimizationAlgorithmWithHessian, OptimizationAlgorithm>(m, "OptimizationAlgorithmWithHessian");

    py::class_<OptimizationAlgorithmGaussNewton, OptimizationAlgorithmWithHessian>(m, "OptimizationAlgorithmGaussNewton")
        .def(py::init([](PyBlockSolverBase& blockSolver){
            return new OptimizationAlgorithmGaussNewton(blockSolver.solver());
        }))
    ;

    py::class_<OptimizationAlgorithmLevenberg, OptimizationAlgorithmWithHessian>(m, "OptimizationAlgorithmLevenberg")
        .def(py::init([](PyBlockSolverBase& blockSolver){
            return new OptimizationAlgorithmLevenberg(blockSolver.solver());
        }))
        .def("set_user_lambda_init", &OptimizationAlgorithmLevenberg::setUserLambdaInit)
    ;

    py::class_<OptimizationAlgorithmDogleg, OptimizationAlgorithmWithHessian>(m, "OptimizationAlgorithmDogleg")
        .def(py::init([](PyBlockSolverBase& blockSolver){
            return new OptimizationAlgorithmDogleg(blockSolver.base_solver());
        }))
    ;


    py::class_<AbstractOptimizationAlgorithmCreator>(m, "AbstractOptimizationAlgorithmCreator");

    py::class_<RegisterOptimizationAlgorithmProxy>(m, "RegisterOptimizationAlgorithmProxy");


}

}  // end namespace g2o