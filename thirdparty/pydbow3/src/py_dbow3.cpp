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

	void clear()
	{
		vocabulary->clear();
	}

	void load(const std::string &path)
	{
		vocabulary->load(path);
	}

	void save(const std::string &path, bool binary_compressed = true)
	{
		vocabulary->save(path, binary_compressed);
	}

	DBoW3::BowVector transform(const std::vector<cv::Mat> &features)
	{
		DBoW3::BowVector word;
		vocabulary->transform(features, word);
		return word;
	}

	double score(const DBoW3::BowVector &A, const DBoW3::BowVector &B)
	{
		return vocabulary->score(A, B);
	}

	DBoW3::Vocabulary *vocabulary;
};

class Database
{
public:
	Database(const std::string &path = std::string())
	{
		if (path.empty())
			database = new DBoW3::Database();
		else
			database = new DBoW3::Database(path);
	}
	~Database()
	{
		delete database;
	}

	void setVocabulary(const Vocabulary &vocabulary, bool use_di, int di_levels = 0)
	{
		database->setVocabulary(*vocabulary.vocabulary, use_di, di_levels);
	}

	unsigned int add(const cv::Mat &features)
	{
		return database->add(features, NULL, NULL);
	}

	std::vector<DBoW3::Result> query(const cv::Mat &features, int max_results = 1, int max_id = -1)
	{
		DBoW3::QueryResults results;
		database->query(features, results, max_results, max_id);
		return results;
	}

	void save(const std::string &filename) const
	{
		database->save(filename);
	}

	void load(const std::string &filename)
	{
		database->load(filename);
	}

	void loadVocabulary(const std::string &filename, bool use_di, int di_levels = 0)
	{
		DBoW3::Vocabulary voc;
		voc.load(filename);
		database->setVocabulary(voc, use_di, di_levels);
	}

private:
	DBoW3::Database *database;
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

	py::class_<Vocabulary>(m, "Vocabulary")
		.def(py::init<const std::string &, int, int, DBoW3::WeightingType, DBoW3::ScoringType>(),
			 py::arg("path") = std::string(), py::arg("k") = 10, py::arg("L") = 5,
			 py::arg("weighting") = DBoW3::TF_IDF, py::arg("scoring") = DBoW3::L1_NORM)
		.def("load", &Vocabulary::load)
		.def("save", &Vocabulary::save)
		.def("create", &Vocabulary::create)
		.def("transform", &Vocabulary::transform, py::return_value_policy::copy)
		.def("score", &Vocabulary::score)
		.def("clear", &Vocabulary::clear);

	py::class_<Database>(m, "Database")
		.def(py::init<const std::string &>(), py::arg("path") = std::string())
		.def("setVocabulary", &Database::setVocabulary, py::arg("vocabulary"), py::arg("use_di") = false, py::arg("di_levels") = 0)
		.def("save", &Database::save)
		.def("load", &Database::load)
		.def("loadVocabulary", &Database::loadVocabulary, py::arg("filename"), py::arg("use_di") = false, py::arg("di_levels") = 0)
		.def("add", &Database::add)
		.def("query", &Database::query, py::return_value_policy::copy, py::arg("features"), py::arg("max_results") = 0, py::arg("max_id") = -1);

	py::class_<DBoW3::Result>(m, "Result")
		.def_readonly("Id", &DBoW3::Result::Id)
		.def_readonly("Score", &DBoW3::Result::Score)
		.def_readonly("nWords", &DBoW3::Result::nWords)
		.def_readonly("bhatScore", &DBoW3::Result::bhatScore)
		.def_readonly("chiScore", &DBoW3::Result::chiScore)
		.def_readonly("sumCommonVi", &DBoW3::Result::sumCommonVi)
		.def_readonly("sumCommonWi", &DBoW3::Result::sumCommonWi)
		.def_readonly("expectedChiScore", &DBoW3::Result::expectedChiScore);
}
