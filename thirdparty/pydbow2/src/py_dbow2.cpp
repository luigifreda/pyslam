#include <opencv2/opencv.hpp>

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FORB.h>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;
using namespace pybind11::literals;


namespace DBoW2
{
	struct TransformResult{
		DBoW2::BowVector bowVector; 
		DBoW2::FeatureVector featureVector; 		
	};
}

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> BinaryVocabulary;

PYBIND11_MODULE(pydbow2, m)
{
	NDArrayConverter::init_numpy();

	py::class_<DBoW2::TransformResult>(m, "TransformResult")
		.def(py::init<>())
		.def_readwrite("bowVector", &DBoW2::TransformResult::bowVector)
		.def_readwrite("featureVector", &DBoW2::TransformResult::featureVector);

	py::class_<DBoW2::FORB>(m, "FORB")
		.def_static("distance", &DBoW2::FORB::distance)
		.def_static("meanValue", &DBoW2::FORB::meanValue)
		.def_static("toString", &DBoW2::FORB::toString)
		.def_static("fromString", &DBoW2::FORB::fromString)
		.def_static("toMat32F", &DBoW2::FORB::toMat32F)
		.def_static("toMat8U", &DBoW2::FORB::toMat8U);

	py::class_<DBoW2::BowVector>(m, "BowVector")
		.def(py::init<>())
		.def("addWeight", &DBoW2::BowVector::addWeight)
		.def("addIfNotExist", &DBoW2::BowVector::addIfNotExist)
		.def("normalize", &DBoW2::BowVector::normalize)
		.def("saveM", &DBoW2::BowVector::saveM);

	py::enum_<DBoW2::WeightingType>(m, "WeightingType")
		.value("TF_IDF", DBoW2::TF_IDF)
		.value("TF", DBoW2::TF)
		.value("IDF", DBoW2::IDF)
		.value("BINARY", DBoW2::BINARY);

	py::enum_<DBoW2::ScoringType>(m, "ScoringType")
		.value("L1_NORM", DBoW2::L1_NORM)
		.value("L2_NORM", DBoW2::L2_NORM)
		.value("CHI_SQUARE", DBoW2::CHI_SQUARE)
		.value("KL", DBoW2::KL)
		.value("Bhattacharyya", DBoW2::BHATTACHARYYA)
		.value("DOT_PRODUCT", DBoW2::DOT_PRODUCT);

	py::class_<BinaryVocabulary>(m, "BinaryVocabulary")
		.def(py::init<int, int, DBoW2::WeightingType, DBoW2::ScoringType>(),
			 py::arg("k") = 10, py::arg("L") = 5,
			 py::arg("weighting") = DBoW2::TF_IDF, py::arg("scoring") = DBoW2::L1_NORM)
		.def("load", &BinaryVocabulary::loadFromTextFile)			 
		.def("size", &BinaryVocabulary::size)
		.def("score", &BinaryVocabulary::score) 
		.def("transform", [](BinaryVocabulary& o,
			const std::vector<DBoW2::FORB::TDescriptor>& features,
			int levelsup) 
			{ 
				DBoW2::TransformResult r; 
				o.transform(features, r.bowVector, r.featureVector, levelsup);
				return r;
			}, 
			"features"_a, "levelsup"_a);		
}
