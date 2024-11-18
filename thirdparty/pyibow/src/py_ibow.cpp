#include <opencv2/opencv.hpp>

#include "ibow-lcd/lcdetector.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "ndarray_converter.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace ibow_lcd;

PYBIND11_MODULE(pyibow, m)
{
	NDArrayConverter::init_numpy();

	py::class_<LCDetectorParams>(m, "LCDetectorParams")
		.def(py::init<>())
		.def_readwrite("k", &LCDetectorParams::k)
		.def_readwrite("s", &LCDetectorParams::s)
		.def_readwrite("t", &LCDetectorParams::t)
		.def_readwrite("merge_policy", &LCDetectorParams::merge_policy)
		.def_readwrite("purge_descriptors", &LCDetectorParams::purge_descriptors)
		.def_readwrite("min_feat_apps", &LCDetectorParams::min_feat_apps)
		.def_readwrite("p", &LCDetectorParams::p)
		.def_readwrite("nndr", &LCDetectorParams::nndr)
		.def_readwrite("nndr_bf", &LCDetectorParams::nndr_bf)
		.def_readwrite("ep_dist", &LCDetectorParams::ep_dist)
		.def_readwrite("conf_prob", &LCDetectorParams::conf_prob);


	py::enum_<LCDetectorStatus>(m, "LCDetectorStatus")
		.value("LC_DETECTED", LC_DETECTED)
		.value("LC_NOT_DETECTED", LC_NOT_DETECTED)
		.value("LC_NOT_ENOUGH_IMAGES", LC_NOT_ENOUGH_IMAGES)
		.value("LC_NOT_ENOUGH_ISLANDS", LC_NOT_ENOUGH_ISLANDS)
		.value("LC_NOT_ENOUGH_INLIERS", LC_NOT_ENOUGH_INLIERS)
		.value("LC_TRANSITION", LC_TRANSITION);


	py::class_<LCDetectorResult>(m, "LCDetectorResult")
		.def(py::init<>())
		.def("isLoop", &LCDetectorResult::isLoop)
		.def_readwrite("status", &LCDetectorResult::status)
		.def_readwrite("query_id", &LCDetectorResult::query_id)
		.def_readwrite("train_id", &LCDetectorResult::train_id)
		.def_readwrite("inliers", &LCDetectorResult::inliers)
		.def_readwrite("score", &LCDetectorResult::score);


	py::class_<LCDetector>(m, "LCDetector")
		.def(py::init<const LCDetectorParams&>())
		.def("process", [](LCDetector& d, 
			const unsigned image_id,
            const std::vector<cv::KeyPoint>& kps,
            const cv::Mat& descs) 
			{ 
				LCDetectorResult result;
				d.process(image_id, kps, descs, true/*add_to_index*/, &result);
				return result;
			}, "image_id"_a, "kps"_a, "descs"_a)
		.def("process_without_pushing", [](LCDetector& d, 
			// processing image without adding the input to the index database
			const unsigned image_id,
            const std::vector<cv::KeyPoint>& kps,
            const cv::Mat& descs) 
			{ 
				LCDetectorResult result;
				d.process(image_id, kps, descs, false/*add_to_index*/, &result);
				return result;
			}, "image_id"_a, "kps"_a, "descs"_a)		
		.def("clear", &LCDetector::clear)
		.def("save", &LCDetector::save)
		.def("load", &LCDetector::load)
		.def("num_images", &LCDetector::numImages)
		.def("num_descriptors", &LCDetector::numDescriptors)
		.def("num_pushed_images", &LCDetector::numPushedImages)		
		.def("print_status", &LCDetector::printStatus);
}
