#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/plot/datalog.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareDataLog(py::module & m) {

    py::class_<DimensionStats>(m, "DimensionStats")
        .def(py::init<>())
        .def("Reset", &DimensionStats::Reset)
        .def("Add", &DimensionStats::Add,
            "v"_a)   // (const float) -> void
        
        .def_readwrite("isMonotonic", &DimensionStats::isMonotonic)    // bool
        .def_readwrite("sum", &DimensionStats::sum)    // float
        .def_readwrite("sum_sq", &DimensionStats::sum_sq)    // float
        .def_readwrite("min", &DimensionStats::min)    // float
        .def_readwrite("max", &DimensionStats::max)    // float
    ;


    py::class_<DataLogBlock>(m, "DataLogBlock")
        .def(py::init<size_t, size_t, size_t>(),
            "dim"_a, "max_samples"_a, "start_id"_a)

        .def("Samples", &DataLogBlock::Samples)   // -> size_t
        .def("MaxSamples", &DataLogBlock::MaxSamples)   // -> size_t
        .def("SampleSpaceLeft", &DataLogBlock::SampleSpaceLeft)   // -> size_t
        .def("IsFull", &DataLogBlock::IsFull)   // -> bool
        .def("AddSamples", &DataLogBlock::AddSamples,
            "num_samples"_a, "dimensions"_a, "data_dim_major"_a)
        .def("ClearLinked", &DataLogBlock::ClearLinked)
        .def("NextBlock", &DataLogBlock::NextBlock)   // ->DataLogBlock*
        .def("StartId", &DataLogBlock::StartId)   // -> size_t
        .def("DimData", &DataLogBlock::DimData,
            "d"_a)                                    // -> float*
        .def("Dimensions", &DataLogBlock::Dimensions)  // -> size_t
        .def("Sample", &DataLogBlock::Sample,
            "n"_a)                                    // -> const float*
    ;


    py::class_<DataLog>(m, "DataLog")
        .def(py::init<unsigned int>(),
            "block_samples_alloc"_a=10000)

        .def("SetLabels", &DataLog::SetLabels,
            "labels"_a)   // (const std::vector<std::string> &) -> void
        .def("Labels", &DataLog::Labels)   // () -> const std::vector<std::string> &

        .def("Log", (void (DataLog::*) (size_t, const float *, unsigned int)) &DataLog::Log,
            "dimension"_a, "vals"_a, "samples"_a=1)
        .def("Log", (void (DataLog::*) (float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (float, float, float, float, float, float, float, float, float, float)) &DataLog::Log)
        .def("Log", (void (DataLog::*) (const std::vector<float> &)) &DataLog::Log)

        .def("Clear", &DataLog::Clear)
        .def("Save", &DataLog::Save,
            "filename"_a)   // std::string ->

        .def("FirstBlock", &DataLog::FirstBlock)   // () -> const DataLogBlock*
        .def("LastBlock", &DataLog::LastBlock)    // () -> const DataLogBlock*
        .def("Samples", &DataLog::Samples)   // () -> size_t
        .def("Sample", &DataLog::Sample,
            "n"_a)     // (int) -> const float*
        .def("Stats", &DataLog::Stats,
            "dim"_a)   // (size_t) -> const DimensionStats&
    ;





}

}