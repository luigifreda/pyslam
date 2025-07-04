#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <g2o/core/robust_kernel.h>
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/jacobian_workspace.h>
//#include <g2o/core/parameter_container.h>
#include <g2o/core/optimizable_graph.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareOptimizableGraph(py::module & m) {
    using CLS = OptimizableGraph;

    py::class_<OptimizableGraph, HyperGraph> cls(m, "OptimizableGraph");

        py::enum_<CLS::ActionType>(cls, "ActionType")
            .value("AT_PREITERATION", CLS::ActionType::AT_PREITERATION)
            .value("AT_NUM_ELEMENTS", CLS::ActionType::AT_NUM_ELEMENTS)
            .export_values();

        // typedef std::set<HyperGraphAction*>    HyperGraphActionSet;

        py::class_<CLS::VertexIDCompare>(cls, "VertexIDCompare")
            .def("__call__", [](const CLS::Vertex* v1, const CLS::Vertex* v2) -> bool {
                return v1->id() < v2->id();
            });

        py::class_<CLS::EdgeIDCompare>(cls, "EdgeIDCompare")
            .def("__call__", [](const CLS::Edge* e1, const CLS::Edge* e2) -> bool {
                return e1->internalId() < e2->internalId();
            });

        // typedef std::vector<OptimizableGraph::Vertex*>      VertexContainer;
        // typedef std::vector<OptimizableGraph::Edge*>        EdgeContainer;

        py::class_<CLS::Vertex, HyperGraph::Vertex, HyperGraph::DataContainer>(cls, "OptimizableGraph_Vertex")
            //.def(py::init<>())   // invalid new-expression of abstract class
            .def("clone", &CLS::Vertex::clone,
                    py::return_value_policy::reference)                                                              // virtual, -> Vertex*
            .def("set_to_origin", &CLS::Vertex::setToOrigin)                                                              // -> void

            .def("set_estimate_data", (bool (CLS::Vertex::*) (const double*)) &CLS::Vertex::setEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())
            .def("set_estimate_data", (bool (CLS::Vertex::*) (const std::vector<double>&)) &CLS::Vertex::setEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())
            .def("get_estimate_data", (bool (CLS::Vertex::*) (double*) const) &CLS::Vertex::getEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())
            .def("get_estimate_data", (bool (CLS::Vertex::*) (std::vector<double>&) const) &CLS::Vertex::getEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())

            .def("set_minimal_estimate_data", (bool (CLS::Vertex::*) (const double*)) &CLS::Vertex::setMinimalEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())
            .def("set_minimal_estimate_data", (bool (CLS::Vertex::*) (const std::vector<double>&)) &CLS::Vertex::setMinimalEstimateData,
					"estimate"_a,
					py::keep_alive<1, 2>())
            .def("get_minimal_estimate_data", (bool (CLS::Vertex::*) (double*) const) &CLS::Vertex::getMinimalEstimateData,
                    "estimate"_a)
            .def("get_minimal_estimate_data", (bool (CLS::Vertex::*) (std::vector<double>&) const) &CLS::Vertex::getMinimalEstimateData,
                    "estimate"_a)
            
            .def("estimate_dimension", &CLS::Vertex::estimateDimension)                                                              // virtual, -> int
            .def("minimal_estimate_dimension", &CLS::Vertex::minimalEstimateDimension)                         // virtual, -> int

            .def("oplus", &CLS::Vertex::oplus,
                    "v"_a)                                                                                     // const double* -> void
            .def("hessian_index", &CLS::Vertex::hessianIndex)                                                              // -> int
            .def("set_hessian_index", &CLS::Vertex::setHessianIndex,
                    "ti"_a)                                                                                    // int -> void

            .def("fixed", &CLS::Vertex::fixed)                                                              // -> bool
            .def("set_fixed", &CLS::Vertex::setFixed,
                    "fixed"_a)                                                                                    // bool -> void

            .def("marginalized", &CLS::Vertex::marginalized)                                                              // -> bool
            .def("set_marginalized", &CLS::Vertex::setMarginalized,
                    "marginalized"_a)                                                                            // bool -> void

            .def("dimension", &CLS::Vertex::dimension)                                                              // -> int
            .def("set_id", &CLS::Vertex::setId,
                    "id"_a)                                                                                   // int -> void
            .def("col_in_hessian", &CLS::Vertex::colInHessian)                                                              // -> int
            .def("set_col_in_hessian", &CLS::Vertex::setColInHessian,
                    "c"_a)                                                                                   // int -> void

            .def("graph", (OptimizableGraph* (CLS::Vertex::*) ()) &CLS::Vertex::graph,
                    py::return_value_policy::reference)

            .def("lock_quadratic_form", &CLS::Vertex::lockQuadraticForm)
            .def("unlock_quadratic_form", &CLS::Vertex::unlockQuadraticForm)
            .def("update_cache", &CLS::Vertex::updateCache)
        ;

        py::class_<CLS::Edge, HyperGraph::Edge, HyperGraph::DataContainer>(cls, "OptimizableGraph_Edge")
            //.def(py::init<>())
            .def("clone", &CLS::Edge::clone,
                    py::return_value_policy::reference)                                                              // -> Edge*
            .def("set_measurement_data", &CLS::Edge::setMeasurementData,
                    "m"_a)                                                                                  // const double* -> bool
            .def("get_measurement_data", &CLS::Edge::getMeasurementData,
                    "m"_a)                                                                                  // double * -> bool
            .def("measurement_dimension", &CLS::Edge::measurementDimension)                             // -> int
            .def("set_measurement_from_state", &CLS::Edge::setMeasurementFromState)                         // -> bool
            .def("robust_kernel", &CLS::Edge::robustKernel,
                    py::return_value_policy::reference)                                                              // -> RobustKernel*
            .def("set_robust_kernel", &CLS::Edge::setRobustKernel,
					"ptr"_a,
					py::keep_alive<1, 2>())                                                                                  // RobustKernel* -> void

            .def("initial_estimate_possible", &CLS::Edge::initialEstimatePossible,
                    "from"_a, "to"_a)                                                              // (const OptimizableGraph::VertexSet&, OptimizableGraph::VertexSet&) -> double
            .def("level", &CLS::Edge::level)                                                              // -> int
            .def("set_level", &CLS::Edge::setLevel,
                    "l"_a)                                                                                  // int -> void
            .def("dimension", &CLS::Edge::dimension)                                                              // -> int

            .def("create_vertex", &CLS::Edge::createVertex)
            .def("internal_id", &CLS::Edge::internalId)                                                    // -> long long
            .def("graph", (OptimizableGraph* (CLS::Edge::*) ()) &CLS::Edge::graph,
                    py::return_value_policy::reference) 

            .def("set_parameter_id", &CLS::Edge::setParameterId,
                    "arg_num"_a, "param_id"_a)                                                              // (int, int) -> bool
            .def("parameter", &CLS::Edge::parameter)
            .def("num_parameters", &CLS::Edge::numParameters)
            .def("resize_parameters", &CLS::Edge::resizeParameters)
        ;

        cls.def(py::init<>());
        cls.def("vertex", (CLS::Vertex* (CLS::*) (int)) &CLS::vertex,
                    "id"_a,
                    py::return_value_policy::reference);                                                       // int -> Vertex*

        cls.def("add_graph", &CLS::addGraph,
                    "g"_a,
                    py::keep_alive<1, 2>());                                                                        // OptimizableGraph* -> void
        
        cls.def("add_vertex", (bool (CLS::*) (OptimizableGraph::Vertex*, HyperGraph::Data*)) &CLS::addVertex,
					"v"_a, "user_data"_a,
					py::keep_alive<1, 2>(),
					py::keep_alive<1, 3>());
        cls.def("add_vertex", (bool (CLS::*) (OptimizableGraph::Vertex*)) &CLS::addVertex,
					"v"_a,
					py::keep_alive<1, 2>());
        cls.def("add_vertex", (bool (CLS::*) (HyperGraph::Vertex*, HyperGraph::Data*)) &CLS::addVertex,
					"v"_a, "user_data"_a,
					py::keep_alive<1, 2>(),
					py::keep_alive<1, 3>());
        cls.def("add_vertex", (bool (CLS::*) (HyperGraph::Vertex*)) &CLS::addVertex,
					"v"_a,
					py::keep_alive<1, 2>());

        cls.def("add_edge", (bool (CLS::*) (OptimizableGraph::Edge*)) &CLS::addEdge,
					"e"_a,
					py::keep_alive<1, 2>());
        cls.def("add_edge", (bool (CLS::*) (HyperGraph::Edge*)) &CLS::addEdge,
					"e"_a,
					py::keep_alive<1, 2>());

        cls.def("set_edge_vertex", &CLS::setEdgeVertex,
					"e"_a, "pos"_a, "v"_a,
					py::keep_alive<1, 2>(),
					py::keep_alive<1, 4>());                                          // (HyperGraph::Edge*, int, HyperGraph::Vertex*) -> bool
        cls.def("chi2", &CLS::chi2);                                                              // -> double
        cls.def("max_dimension", &CLS::maxDimension);                                                              // -> int
        cls.def("dimensions", &CLS::dimensions);                                                              // -> std::set<int>

        cls.def("optimize", &CLS::optimize,
                    "iterations"_a, "online"_a=false);                                                              // (int, bool) -> int
        cls.def("pre_iteration", &CLS::preIteration);                                                                // int -> void
        cls.def("post_iteration", &CLS::postIteration);                                                              // int -> void

        cls.def("add_pre_iteration_action", &CLS::addPreIterationAction,
					"action"_a,
					py::keep_alive<1, 2>());                                               // HyperGraphAction* -> bool
        cls.def("add_post_iteration_action", &CLS::addPostIterationAction, 
					"action"_a,
					py::keep_alive<1, 2>());                                               // HyperGraphAction* -> bool
        cls.def("remove_pre_iteration_action", &CLS::removePreIterationAction, 
                    "action"_a,
                    py::keep_alive<1, 2>());                                                              // HyperGraphAction* -> bool
        cls.def("remove_post_iteration_action", &CLS::removePostIterationAction, 
                    "action"_a,
                    py::keep_alive<1, 2>());                                                              // HyperGraphAction* -> bool

        cls.def("push", (void (CLS::*) ()) &CLS::push);
        cls.def("push", (void (CLS::*) (HyperGraph::VertexSet&)) &CLS::push);
        cls.def("pop", (void (CLS::*) ()) &CLS::pop);
        cls.def("pop", (void (CLS::*) (HyperGraph::VertexSet&)) &CLS::pop);        
        cls.def("discard_top", (void (CLS::*) ()) &CLS::discardTop);
        cls.def("discard_top", (void (CLS::*) (HyperGraph::VertexSet&)) &CLS::discardTop);        

        cls.def("load", (bool (CLS::*) (const char*, bool)) &CLS::load,
    	            "filename"_a, "create_edges"_a=true);
        cls.def("save", (bool (CLS::*) (const char*, int) const) &CLS::save,
                    "filename"_a, "level"_a=0);

        cls.def("set_fixed", &CLS::setFixed,
					"vset"_a, "fixes"_a,
					py::keep_alive<1, 2>());                                                 // (HyperGraph::VertexSet&, bool) -> void

        cls.def("clear_parameters", &CLS::clearParameters);
        cls.def("add_parameter", &CLS::addParameter,
					"p"_a,
					py::keep_alive<1, 2>());                                                    // Parameter* -> bool
        cls.def("parameter", &CLS::parameter,
                    "id"_a);                                                                        // int -> Parameter*

        cls.def("verify_information_matrices", &CLS::verifyInformationMatrices,
                    "verbose"_a=false);                                                              // bool -> bool
        
        cls.def_static("init_multi_threading", &CLS::initMultiThreading);
        cls.def("jacobian_workspace", (JacobianWorkspace& (CLS::*) ()) &CLS::jacobianWorkspace);
        //cls.def("parameters", (ParameterContainer& (CLS::*) ()) &CLS::parameters);

        // saveSubset
        // setRenamedTypesFromString
        // isSolverSuitable
        // saveVertex
        // saveParameter
        // saveEdge
        // saveUserData
}

}  // namespace g2o end