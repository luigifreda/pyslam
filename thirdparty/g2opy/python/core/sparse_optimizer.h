#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/estimate_propagator.h>
#include <g2o/core/hyper_graph_action.h>

#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_with_hessian.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_factory.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace g2o {

class Flag
{
public:
	Flag(bool init=false):value(init){}
	bool value;

        // Methods to make the class pickleable
        py::tuple __getstate__() const {
                // Return the object's state as a tuple
                return py::make_tuple(value);
        }

        void __setstate__(py::tuple t) {
                // Restore the object's state from the tuple
                if (t.size() != 1)
                        throw std::runtime_error("Invalid state!");
                value = t[0].cast<bool>();
        }        
};

void declareSparseOptimizer(py::module & m) {
    using CLS = SparseOptimizer;

    py::class_<Flag>(m, "Flag")
        .def(py::init<>())
        .def(py::init<bool>(),"init"_a)
        .def_readwrite("value", &Flag::value)
        .def("__getstate__", &Flag::__getstate__)
        .def("__setstate__", &Flag::__setstate__);


    py::class_<CLS, OptimizableGraph>(m, "SparseOptimizer")
        // ATTENTION: _solver & _statistics is own by SparseOptimizer and will be
        // deleted in its destructor.
        .def(py::init<>())

        .def("initialize_optimization", (bool (CLS::*) (HyperGraph::EdgeSet&)) &CLS::initializeOptimization,
                "eset"_a)                                                                                           // virtual
        .def("initialize_optimization", (bool (CLS::*) (HyperGraph::VertexSet&, int)) &CLS::initializeOptimization,
                "vset"_a, "level"_a=0)                                                                                   // virtual
        .def("initialize_optimization", (bool (CLS::*) (int)) &CLS::initializeOptimization,
                "level"_a=0)                                                                                   // virtual

        .def("update_initialization", &CLS::updateInitialization,
                "vset"_a, "eset"_a)                                                                                   // virtual, ->bool
        .def("compute_initial_guess", (void (CLS::*) ()) &CLS::computeInitialGuess)                                             // virtual
        .def("compute_initial_guess", (void (CLS::*) (EstimatePropagatorCost&)) &CLS::computeInitialGuess,
                "propagator"_a)                                                                                                // virtual
        .def("set_to_origin", &CLS::setToOrigin)                                                                                   // virtual

        .def("optimize", &CLS::optimize,
                "iterations"_a, "online"_a=false)                                                                                   // -> int

        // .def("compute_marginals", (bool (CLS::*) (SparseBlockMatrix<MatrixXD>&, const std::vector<std::pair<int, int> >&)) 
        //         &CLS::computeMarginals, "spinv"_a, "block_indices"_a)   
        // .def("compute_marginals", (bool (CLS::*) (SparseBlockMatrix<MatrixXD>&, const OptimizableGraph::Vertex*)) 
        //         &CLS::computeMarginals, "spinv"_a, "vertex"_a)  
        // .def("compute_marginals", (bool (CLS::*) (SparseBlockMatrix<MatrixXD>&, const OptimizableGraph::VertexContainer&)) 
        //         &CLS::computeMarginals, "spinv"_a, "vertices"_a)   

		// segfault (一︿一)
        // .def("compute_marginals", [](CLS& optimizer, const std::vector<std::pair<int, int> >& block_indices){
        //         SparseBlockMatrix<MatrixXD> spinv;
        //         optimizer.computeMarginals(spinv, block_indices);
		// 		return spinv;},
		// 		py::return_value_policy::copy)
		// .def("compute_marginals", [](CLS& optimizer, const OptimizableGraph::Vertex* vertex){
        //         SparseBlockMatrix<MatrixXD> spinv;
        //         optimizer.computeMarginals(spinv, vertex);
		// 		return spinv;},
		// 		py::return_value_policy::copy)
		// .def("compute_marginals", [](CLS& optimizer, size_t vertex_id){
        //         SparseBlockMatrix<MatrixXD> spinv;
		// 		const g2o::OptimizableGraph::Vertex* vertex = optimizer.vertex(vertex_id);
        //         optimizer.computeMarginals(spinv, vertex);
		// 		return spinv;},
		// 		py::return_value_policy::copy)
		// .def("compute_marginals", [](CLS& optimizer, const OptimizableGraph::VertexContainer& vertices){
        //         SparseBlockMatrix<MatrixXD> spinv;
        //         optimizer.computeMarginals(spinv, vertices);
		// 		return spinv;},
		// 		py::return_value_policy::copy)

        // The gauge should be fixed() and then the optimization can work (if no additional dof are in
        // the system. The default implementation returns a node with maximum dimension.
        .def("find_gauge", &CLS::findGauge)                                                                       // virtual, -> OptimizableGraph::Vertex*
        .def("gauge_freedom", &CLS::gaugeFreedom)                                                                     // -> bool
        .def("active_chi2", &CLS::activeChi2)                                                                             // -> double
        .def("active_robust_chi2", &CLS::activeRobustChi2)                                                             // -> double
        .def("verbose", &CLS::verbose)                                                                              // -> bool
        .def("set_verbose", &CLS::setVerbose,
                "verbose"_a)                                                                                  // -> void

        .def("set_force_stop_flag_old", &CLS::setForceStopFlag,
                "flag"_a)                                                                                   // -> void
	.def("set_force_stop_flag", [](CLS& optimizer, Flag* flag){ optimizer.setForceStopFlag(&(flag->value)); }, 
				      py::keep_alive<1, 2>())
        .def("force_stop_flag", &CLS::forceStopFlag)                                                             // -> bool*
        .def("terminate", &CLS::terminate)                                                                                   // -> bool

        .def("index_mapping", &CLS::indexMapping)                                                                    // -> const VertexContainer&
        .def("active_vertices", &CLS::activeVertices)                                                                 // -> const VertexContainer&
        .def("active_edges", &CLS::activeEdges)                                                                     // -> const EdgeContainer&
        .def("remove_vertex", &CLS::removeVertex,
                "v"_a, "detach"_a=false)                                                                         // virtual, -> bool

        .def("find_active_vertex", &CLS::findActiveVertex,
                "v"_a)                                                                                   // -> VertexContainer::const_iterator, const
        .def("find_active_edge", &CLS::findActiveEdge,
                "e"_a)                                                                                   // -> EdgeContainer::const_iterator, const

        .def("algorithm", &CLS::algorithm)                                                                     // -> const OptimizationAlgorithm*, const
        .def("solver", &CLS::solver)                                                                                // -> OptimizationAlgorithm*
        //.def("set_algorithm", &CLS::setAlgorithm,
        //        "algorithm"_a, py::keep_alive<1, 2>())                                                                                   // -> void

        .def("set_algorithm", [](CLS& optimizer, OptimizationAlgorithm* algorithm) {optimizer.setAlgorithm(algorithm);}, 
                py::keep_alive<1, 2>())
        .def("set_algorithm", [](CLS& optimizer, OptimizationAlgorithmWithHessian* algorithm) {optimizer.setAlgorithm(algorithm);}, 
				py::keep_alive<1, 2>())
        .def("set_algorithm", [](CLS& optimizer, OptimizationAlgorithmGaussNewton* algorithm) {optimizer.setAlgorithm(algorithm);}, 
				py::keep_alive<1, 2>())
        .def("set_algorithm", [](CLS& optimizer, OptimizationAlgorithmLevenberg* algorithm) {optimizer.setAlgorithm(algorithm);}, 
				py::keep_alive<1, 2>())
        .def("set_algorithm", [](CLS& optimizer, OptimizationAlgorithmDogleg* algorithm) {optimizer.setAlgorithm(algorithm);}, 
				py::keep_alive<1, 2>())
        


        .def("push", (void (CLS::*) (SparseOptimizer::VertexContainer&)) &CLS::push,
                "vlist"_a, py::keep_alive<1, 2>())   
        .def("push", (void (CLS::*) (HyperGraph::VertexSet&)) &CLS::push,
                "vlist"_a, py::keep_alive<1, 2>())   
        .def("push", (void (CLS::*) ()) &CLS::push) 

        .def("pop", (void (CLS::*) (SparseOptimizer::VertexContainer&)) &CLS::pop,
                "vlist"_a, py::keep_alive<1, 2>())  
        .def("pop", (void (CLS::*) (HyperGraph::VertexSet&)) &CLS::pop,
                "vlist"_a, py::keep_alive<1, 2>())   
        .def("pop", (void (CLS::*) ()) &CLS::pop)   

        .def("discard_top", (void (CLS::*) (SparseOptimizer::VertexContainer&)) &CLS::discardTop,
                "vlist"_a, py::keep_alive<1, 2>())                                                                                   // -> void
        .def("discard_top", (void (CLS::*) ()) &CLS::discardTop)   

        .def("clear", &CLS::clear)                                                                                   // virtual, -> void
        .def("compute_active_errors", &CLS::computeActiveErrors)                                                     // virtual, -> void
        .def("update", &CLS::update, 
                "update"_a)                                                                                   // -> void

        .def("batch_statistics", (const BatchStatisticsContainer& (CLS::*) () const) &CLS::batchStatistics)
        .def("set_compute_batch_statistics", &CLS::setComputeBatchStatistics)                                     // -> void
        .def("compute_batch_statistics", &CLS::computeBatchStatistics)

        // callbacks
        .def("add_compute_error_action", &CLS::addComputeErrorAction,
                "action"_a)                                                                                   // -> bool
        .def("remove_compute_error_action", &CLS::removeComputeErrorAction,
                "action"_a)                                                                                   // -> bool

    ;
    

}

}  // end namespace g2o
