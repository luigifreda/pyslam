#include <opencv2/opencv.hpp>

#include <DBoW2/TemplatedVocabulary.h>
#include <DBoW2/FORB.h>
#include <DBoW2/KeyFrameOrbDatabase.h>

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
		.def(py::init<const std::vector<std::pair<DBoW2::WordId, DBoW2::WordValue>> &>())		
		.def("addWeight", &DBoW2::BowVector::addWeight)
		.def("addIfNotExist", &DBoW2::BowVector::addIfNotExist)
		.def("normalize", &DBoW2::BowVector::normalize)
		.def("saveM", &DBoW2::BowVector::saveM)
		.def("toVec", &DBoW2::BowVector::toVec)
		.def("fromVec", &DBoW2::BowVector::fromVec)		
		.def("__repr__", [](const DBoW2::BowVector &obj) {
				std::ostringstream os;
				os << obj;
				return os.str();
			})
		.def("__str__", [](const DBoW2::BowVector &obj) {
			std::ostringstream os;
			os << obj;
			return os.str();
		})
		// Adding pickling support
		.def("__getstate__", [](const DBoW2::BowVector &obj) {
			// Extract state into a serializable format, e.g., a dictionary
			return obj.toVec(); // Assuming toVec() returns a serializable structure
		})
		.def("__setstate__", [](DBoW2::BowVector &obj, const std::vector<std::pair<DBoW2::WordId, DBoW2::WordValue>> &state) {
			obj.fromVec(state); // Restore state from the vector format
		});	

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
		.def("load", [](BinaryVocabulary& o, const std::string& filename, const bool use_boost) {
			if (use_boost)
				o.loadWithBoost(filename);
			else
				o.loadFromTextFile(filename);
		}, py::arg("filename"), py::arg("use_boost") = false)
		.def("create",[](BinaryVocabulary& o, std::vector<DBoW2::FORB::TDescriptor>& features) {
		#if 1
			std::vector<std::vector<DBoW2::FORB::TDescriptor>> vtf(features.size());
			for(size_t i=0;i<features.size();i++){
				vtf[i].resize(features[i].rows);
				for(int r=0;r<features[i].rows;r++)
					vtf[i][r]=features[i].rowRange(r,r+1);
			}
		#else 
			std::vector<std::vector<DBoW2::FORB::TDescriptor>> vtf;
			vtf.push_back(features);
		#endif 
			std::cout << "features size: " << features.size() << " vtf size: " << vtf.size() << std::endl;				
			o.create(vtf);			
			})
		.def("save", [](BinaryVocabulary& o, 
			const std::string& filename, const bool compressed, const bool use_boost) {
			if (use_boost){
				o.saveWithBoost(filename);
			}else {
				if (compressed)
					o.saveToBinaryFile(filename);
				else 
					o.saveToTextFile(filename);
			} 
			}, py::arg("filename"), py::arg("compressed") = false, py::arg("use_boost") = false)			 
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

	py::class_<DBoW2::KeyFrameOrbDatabase>(m, "KeyFrameOrbDatabase")
		.def(py::init<DBoW2::ORBVocabulary &>())
		.def("set_vocabulary", &DBoW2::KeyFrameOrbDatabase::setVocabulary)
		.def("add", &DBoW2::KeyFrameOrbDatabase::add)
		.def("erase", &DBoW2::KeyFrameOrbDatabase::erase)
		.def("clear", &DBoW2::KeyFrameOrbDatabase::clear)
		.def("save", &DBoW2::KeyFrameOrbDatabase::save)
		.def("load", &DBoW2::KeyFrameOrbDatabase::load)
		.def("size", &DBoW2::KeyFrameOrbDatabase::size)
		.def("detect_loop_candidates", &DBoW2::KeyFrameOrbDatabase::detectLoopCandidates)
		.def("detect_relocalization_candidates", &DBoW2::KeyFrameOrbDatabase::detectRelocalizationCandidates)
		.def("print_status", &DBoW2::KeyFrameOrbDatabase::printStatus);
}