#include <opencv2/opencv.hpp>
#include <DBoW3/DBoW3.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;

class Vocabulary
{
public:
	Vocabulary(const std::string &path = std::string(), int k = 10, int L = 5,
			   DBoW3::WeightingType weighting = DBoW3::TF_IDF, DBoW3::ScoringType scoring = DBoW3::L1_NORM)
	{
		vocabulary = new DBoW3::Vocabulary(k, L, weighting, scoring);
		if (!path.empty())
			load(path);
	}
	~Vocabulary()
	{
		delete vocabulary;
	}

	void create(const std::vector<cv::Mat> &training_features)
	{
		vocabulary->create(training_features);
	}

	size_t size() const
	{
		return vocabulary->size();
	}

	void clear()
	{
		vocabulary->clear();
	}

	void load(const std::string &path, bool use_boost=false)
	{
		vocabulary->load(path, use_boost);
	}

	void save(const std::string &path, bool binary_compressed = true, bool use_boost=false)
	{
		vocabulary->save(path, binary_compressed, use_boost);
	}

	DBoW3::BowVector transform(const cv::Mat &features)
	{
		DBoW3::BowVector word;
		vocabulary->transform(features, word);
		return word;
	}

	double score(const DBoW3::BowVector &A, const DBoW3::BowVector &B)
	{
		return vocabulary->score(A, B);
	}

	bool empty() const
	{
		return vocabulary->empty();
	}

	DBoW3::Vocabulary *vocabulary;
};

class Database
{
public:
	Database(const bool use_di = true, const int di_levels = 0)
	{
		database = new DBoW3::Database(use_di, di_levels);
	}

	Database(const std::string &filename)
	{
		database = new DBoW3::Database(filename);
	}

	~Database()
	{
		delete database;
	}

	void setVocabulary(const Vocabulary &vocabulary, bool use_di, int di_levels = 0)
	{
		database->setVocabulary(*vocabulary.vocabulary, use_di, di_levels);
	}

	void setScoring(const DBoW3::ScoringType scoring)
	{
		database->setScoring(scoring);
	}

	void setWeighting(const DBoW3::WeightingType weighting)
	{
		database->setWeighting(weighting);
	}

	// add local descriptors (they will be internally transformed into a global descriptor)
	unsigned int addFeatures(const cv::Mat &features)
	{
		unsigned int res = database->add(features, NULL, NULL);
		return res;		
	}

	// add directly global descriptor
	unsigned int addBowVector(const DBoW3::BowVector &vec)
	{
		unsigned int res = database->add(vec);
		return res;
	}

	// query with local descriptors (they will be internally transformed into a global descriptor)
	std::vector<DBoW3::Result> query(const cv::Mat &features, int max_results = 1, int max_id = -1)
	{
		DBoW3::QueryResults results;
		database->query(features, results, max_results, max_id);
		return results;
	}

	// query directly with global descriptor 
	std::vector<DBoW3::Result> query(const DBoW3::BowVector &vec, int max_results = 1, int max_id = -1)
	{
		DBoW3::QueryResults results;
		database->query(vec, results, max_results, max_id);
		return results;
	}

	void clear()
	{
		database->clear();
	}

	void save(const std::string &filename, bool save_voc=true) const
	{
		database->save(filename, save_voc);
	}

	void load(const std::string &filename, bool load_voc = true)
	{
		database->load(filename, load_voc);
	}

	void loadVocabulary(const std::string &filename, bool use_di, int di_levels = 0)
	{
		DBoW3::Vocabulary voc;
		voc.load(filename);
		database->setVocabulary(voc, use_di, di_levels);
	}

	void printStatus() const
	{
		database->print_status();
	}

	size_t size() const
	{
		return database->size();
	}

private:
	DBoW3::Database *database=nullptr;
};

PYBIND11_MODULE(pydbow3, m)
{
	NDArrayConverter::init_numpy();

	py::enum_<DBoW3::WeightingType>(m, "WeightingType")
		.value("TF_IDF", DBoW3::TF_IDF)
		.value("TF", DBoW3::TF)
		.value("IDF", DBoW3::IDF)
		.value("BINARY", DBoW3::BINARY);

	py::enum_<DBoW3::ScoringType>(m, "ScoringType")
		.value("L1_NORM", DBoW3::L1_NORM)
		.value("L2_NORM", DBoW3::L2_NORM)
		.value("CHI_SQUARE", DBoW3::CHI_SQUARE)
		.value("KL", DBoW3::KL)
		.value("BHATTACHARYYA", DBoW3::BHATTACHARYYA)
		.value("DOT_PRODUCT", DBoW3::DOT_PRODUCT);

	py::class_<DBoW3::Result>(m, "Result")
		.def_readonly("id", &DBoW3::Result::Id)
		.def_readonly("score", &DBoW3::Result::Score)
		.def_readonly("nWords", &DBoW3::Result::nWords)
		.def_readonly("bhatScore", &DBoW3::Result::bhatScore)
		.def_readonly("chiScore", &DBoW3::Result::chiScore)
		.def_readonly("sumCommonVi", &DBoW3::Result::sumCommonVi)
		.def_readonly("sumCommonWi", &DBoW3::Result::sumCommonWi)
		.def_readonly("expectedChiScore", &DBoW3::Result::expectedChiScore);

	py::class_<DBoW3::BowVector>(m, "BowVector")
		.def(py::init<>())
		.def(py::init<const std::vector<std::pair<DBoW3::WordId, DBoW3::WordValue>> &>())
		.def(py::init<const std::vector<DBoW3::WordValue> &>())		
		.def("addWeight", &DBoW3::BowVector::addWeight)
		.def("addIfNotExist", &DBoW3::BowVector::addIfNotExist)
		.def("normalize", &DBoW3::BowVector::normalize)
		.def("saveM", &DBoW3::BowVector::saveM)
		.def("toVec", &DBoW3::BowVector::toVec)	
		.def("fromVec", &DBoW3::BowVector::fromVec)
		.def("fromVecValues", &DBoW3::BowVector::fromVecValues)		
		.def("__repr__", [](const DBoW3::BowVector &obj) {
				std::ostringstream os;
				os << obj;
				return os.str();
			})
		.def("__str__", [](const DBoW3::BowVector &obj) {
			std::ostringstream os;
			os << obj;
			return os.str();
		})
		// Adding pickling support
		.def("__getstate__", [](const DBoW3::BowVector &obj) {
			// Extract state in a serializable format
			return obj.toVec(); // Assuming toVec() returns a picklable structure
		})
		.def("__setstate__", [](DBoW3::BowVector &obj, const std::vector<std::pair<DBoW3::WordId, DBoW3::WordValue>> &state) {
			// Restore the object state
			obj.fromVec(state);
		});					

	py::class_<Vocabulary>(m, "Vocabulary")
		.def(py::init<const std::string &, int, int, DBoW3::WeightingType, DBoW3::ScoringType>(),
			 py::arg("path") = std::string(), py::arg("k") = 10, py::arg("L") = 5,
			 py::arg("weighting") = DBoW3::TF_IDF, py::arg("scoring") = DBoW3::L1_NORM)
		.def("load", &Vocabulary::load, py::arg("path"), py::arg("use_boost") = false)
		.def("save", &Vocabulary::save, py::arg("path"), py::arg("binary_compressed") = true, py::arg("use_boost") = false)
		.def("create", &Vocabulary::create)
		.def("transform", &Vocabulary::transform)
		.def("score", &Vocabulary::score)
		.def("clear", &Vocabulary::clear)
		.def("empty", &Vocabulary::empty)
		.def("size", &Vocabulary::size);

	py::class_<Database>(m, "Database")
		.def(py::init<const bool, const int>(), py::arg("use_di") = false, py::arg("di_levels") = 0)
		.def(py::init<const std::string &>(), py::arg("filename"))
		.def("setVocabulary", &Database::setVocabulary, py::arg("vocabulary"), py::arg("use_di") = false, py::arg("di_levels") = 0)
		.def("setScoring", &Database::setScoring, py::arg("scoring"))
		.def("setWeighting", &Database::setWeighting, py::arg("weighting"))
		.def("clear", &Database::clear)
		.def("save", &Database::save, py::arg("filename"), py::arg("save_voc") = true)
		.def("load", &Database::load, py::arg("filename"), py::arg("load_voc") = true)
		.def("loadVocabulary", &Database::loadVocabulary, py::arg("filename"), py::arg("use_di") = false, py::arg("di_levels") = 0)
		.def("addFeatures", 
			(unsigned int (Database::*)(const cv::Mat &)) &Database::addFeatures,
			py::arg("features"))
		.def("addBowVector",
			(unsigned int (Database::*)(const DBoW3::BowVector &)) &Database::addBowVector,
			py::arg("vec"))
		.def("query",
			(std::vector<DBoW3::Result> (Database::*)(const DBoW3::BowVector&,int,int)) &Database::query,
			py::return_value_policy::copy,
			py::arg("vec"), py::arg("max_results") = 1, py::arg("max_id") = -1)
		.def("query_local_des", 
			(std::vector<DBoW3::Result> (Database::*)(const cv::Mat&,int,int)) &Database::query, 
			py::return_value_policy::copy, 
			py::arg("features"), py::arg("max_results") = 1, py::arg("max_id") = -1)
		.def("print_status", &Database::printStatus)
		.def("size", &Database::size);	
}
