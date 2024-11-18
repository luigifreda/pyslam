#include <opencv2/opencv.hpp>

#include "obindex2/binary_index.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace obindex2;

PYBIND11_MODULE(pyobindex2, m)
{
	NDArrayConverter::init_numpy();


	py::enum_<MergePolicy>(m, "MergePolicy")
		.value("MERGE_POLICY_NONE", MERGE_POLICY_NONE)
		.value("MERGE_POLICY_AND", MERGE_POLICY_AND)
		.value("MERGE_POLICY_OR", MERGE_POLICY_OR)
		.export_values();


	py::class_<InvIndexItem>(m, "InvIndexItem")
		.def(py::init<>())
		.def(py::init<const int, const cv::Point2f, const double, const int>())
		.def_readwrite("image_id", &InvIndexItem::image_id)
		.def_readwrite("pt", &InvIndexItem::pt)
		.def_readwrite("dist", &InvIndexItem::dist)
		.def_readwrite("kp_ind", &InvIndexItem::kp_ind);


	py::class_<ImageMatch>(m, "ImageMatch")
		.def(py::init<>())
		.def(py::init<const int, const double>())
		.def_readwrite("image_id", &ImageMatch::image_id)
		.def_readwrite("score", &ImageMatch::score)
		.def("__lt__", &ImageMatch::operator<);


	py::class_<PointMatches>(m, "PointMatches")
		.def(py::init<>())
		.def_readwrite("query", &PointMatches::query)
		.def_readwrite("train", &PointMatches::train);


	py::class_<ImageIndex>(m, "ImageIndex")
        .def(py::init<unsigned, unsigned, unsigned, MergePolicy, bool, unsigned>(),
             py::arg("k") = 16, py::arg("s") = 150, py::arg("t") = 4,
             py::arg("merge_policy") = MERGE_POLICY_NONE, py::arg("purge_descriptors") = true,
             py::arg("min_feat_apps") = 3)
        .def("addImage", 
			(void (ImageIndex::*)(const unsigned, const std::vector<cv::KeyPoint>&, const cv::Mat&)) &ImageIndex::addImage,
             py::arg("image_id"), py::arg("kps"), py::arg("descs"))
        .def("addImage", 
			(void (ImageIndex::*)(const unsigned, const std::vector<cv::KeyPoint>&, const cv::Mat&, const std::vector<cv::DMatch>&)) &ImageIndex::addImage,
             py::arg("image_id"), py::arg("kps"), py::arg("descs"), py::arg("matches"))
        .def("searchImages", [](ImageIndex& o,
			const cv::Mat& descs, 
			const std::vector<cv::DMatch>& gmatches,
			bool sort) {
				std::vector<ImageMatch> img_matches;
				o.searchImages(descs, gmatches, &img_matches, sort);
				return img_matches;
			}, py::arg("descs"), py::arg("gmatches"), py::arg("sort") = true)		
		.def("searchDescriptors", [](ImageIndex& o, 
			const cv::Mat& descs, 
			const unsigned knn = 2,
            const unsigned checks = 32) {
				std::vector<std::vector<cv::DMatch>> matches;
				o.searchDescriptors(descs, &matches, knn, checks);
				return matches;
			}, py::arg("descs"), py::arg("knn") = 2, py::arg("checks") = 32)			 
        .def("deleteDescriptor", (void (ImageIndex::*)(const unsigned)) &ImageIndex::deleteDescriptor, py::arg("desc_id"))
		.def("getMatchings", [](ImageIndex& o, 
			const std::vector<cv::KeyPoint>& query_kps,
            const std::vector<cv::DMatch>& matches) {
				std::unordered_map<unsigned, PointMatches> point_matches;
				o.getMatchings(query_kps, matches, &point_matches);
				return point_matches;
			}, py::arg("query_kps"), py::arg("matches"))
        .def("numImages", &ImageIndex::numImages)
        .def("numDescriptors", &ImageIndex::numDescriptors)
        .def("rebuild", &ImageIndex::rebuild)
		.def("clear", &ImageIndex::clear)
		.def("save", &ImageIndex::save, py::arg("filename"))
		.def("load", &ImageIndex::load, py::arg("filename"))
		.def("print_status", &ImageIndex::printStatus);
}
