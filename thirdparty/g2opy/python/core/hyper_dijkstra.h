#include <pybind11/pybind11.h>

#include <g2o/core/hyper_dijkstra.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void delcareHyperDijkstra(py::module& m) {

    py::class_<HyperDijkstra> cls(m, "HyperDijkstra");

    py::class_<HyperDijkstra::CostFunction>(cls, "HyperDijkstraCostFunction");

    py::class_<HyperDijkstra::TreeAction>(cls, "HyperDijkstraTreeAction")
        .def(py::init<>())
        .def("perform", (double (HyperDijkstra::TreeAction::*) 
            (HyperGraph::Vertex*, HyperGraph::Vertex*, HyperGraph::Edge*)) 
            &HyperDijkstra::TreeAction::perform,
            "v"_a, "vParent"_a, "e"_a,
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>())
        .def("perform", (double (HyperDijkstra::TreeAction::*) 
            (HyperGraph::Vertex*, HyperGraph::Vertex*, HyperGraph::Edge*, double)) 
            &HyperDijkstra::TreeAction::perform,
            "v"_a, "vParent"_a, "e"_a, "distance"_a,
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>())
    ;

    py::class_<HyperDijkstra::AdjacencyMapEntry>(cls, "HyperDijkstraAdjacencyMapEntry")
        .def(py::init<HyperGraph::Vertex*, HyperGraph::Vertex*, HyperGraph::Edge*, double>(),
            "_child"_a=nullptr, "_parent"_a=nullptr, "_edge"_a=nullptr, "_distance"_a=std::numeric_limits<double>::max(),
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>())
        .def("child", &HyperDijkstra::AdjacencyMapEntry::child)
        .def("parent", &HyperDijkstra::AdjacencyMapEntry::parent)
        .def("edge", &HyperDijkstra::AdjacencyMapEntry::edge)
        .def("distance", &HyperDijkstra::AdjacencyMapEntry::distance)
        .def("children", (HyperGraph::VertexSet& (HyperDijkstra::AdjacencyMapEntry::*) ()) 
            &HyperDijkstra::AdjacencyMapEntry::children)
    ;

    cls.def(py::init<HyperGraph*>(), 
        "g"_a,
        py::keep_alive<1, 2>());
    cls.def("visited", &HyperDijkstra::visited);   // -> HyperGraph::VertexSet&
    cls.def("adjacency_map", &HyperDijkstra::adjacencyMap);  // -> AdjacencyMap&
    cls.def("graph", &HyperDijkstra::graph);   // -> HyperGraph*

    cls.def("shortest_paths", (void (HyperDijkstra::*) 
        (HyperGraph::Vertex*, HyperDijkstra::CostFunction*, double, double, bool, double)) 
        &HyperDijkstra::shortestPaths,
        "v"_a, "cost"_a, 
        "maxDistance"_a=std::numeric_limits< double >::max(),
        "comparisonConditioner"_a=1e-3,
        "directed"_a=false,
        "maxEdgeCost"_a=std::numeric_limits< double >::max(),
        py::keep_alive<1, 2>(),
        py::keep_alive<1, 3>());

    cls.def("shortest_paths", (void (HyperDijkstra::*) 
        (HyperGraph::VertexSet&, HyperDijkstra::CostFunction*, double, double, bool, double)) 
        &HyperDijkstra::shortestPaths,
        "vset"_a, "cost"_a, 
        "maxDistance"_a=std::numeric_limits< double >::max(),
        "comparisonConditioner"_a=1e-3,
        "directed"_a=false,
        "maxEdgeCost"_a=std::numeric_limits< double >::max(),
        py::keep_alive<1, 2>(),
        py::keep_alive<1, 3>());

    cls.def_static("compute_tree", &HyperDijkstra::computeTree,
        "amap"_a,
        py::keep_alive<1, 2>()); // (AdjacencyMap&) -> void
    cls.def_static("visit_adjacency_map", &HyperDijkstra::visitAdjacencyMap,
        "amap"_a, "action"_a, "useDistance"_a=false,
        py::keep_alive<1, 2>(),
        py::keep_alive<1, 3>()); // (AdjacencyMap& amap, TreeAction* action, bool useDistance=false) -> void
    cls.def_static("connected_subset", &HyperDijkstra::connectedSubset,
        "connected"_a, "visited"_a, "startingSet"_a, "g"_a, "v"_a, "cost"_a,
        "distance"_a, "comparisonConditioner"_a, "maxEdgeCost"_a=std::numeric_limits< double >::max(),
        py::keep_alive<1, 2>(),
        py::keep_alive<1, 3>(),
        py::keep_alive<1, 4>(),
        py::keep_alive<1, 5>(),
        py::keep_alive<1, 6>(),
        py::keep_alive<1, 7>());   // (HyperGraph::VertexSet& connected, HyperGraph::VertexSet& visited, 94 HyperGraph::VertexSet& startingSet, 95 HyperGraph* g, HyperGraph::Vertex* v, 96 HyperDijkstra::CostFunction* cost, double distance, double comparisonConditioner, 97 double maxEdgeCost) -> void



    py::class_<UniformCostFunction, HyperDijkstra::CostFunction>(m, "UniformCostFunction")
        .def("__call__", &UniformCostFunction::operator(),
            "edge"_a, "from"_a, "to"_a,
            py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            py::keep_alive<1, 4>())
    ;

}

}