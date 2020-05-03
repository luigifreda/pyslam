#include <pybind11/pybind11.h>

#include <sstream> 

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include "opencv_type_casters.h"
#include "utils.h"

namespace py = pybind11;
using namespace pybind11::literals;


PYBIND11_MODULE(cv2_pyslam_module, m) 
{
    // optional module docstring
    m.doc() = "pybind11 plugin for pyslam_utils module";
    
    // bindings to MSDDetector class
    // static Ptr<MSDDetector> create(int m_patch_radius = 3, 
    //                                int m_search_area_radius = 5,
    //                                int m_nms_radius = 5, 
    //                                int m_nms_scale_radius = 0, 
    //                                float m_th_saliency = 250.0f, 
    //                                int m_kNN = 4,
    //                                float m_scale_factor = 1.25f, 
    //                                int m_n_scales = -1, 
    //                                bool m_compute_orientation = false);   
    py::class_<cv::xfeatures2d::MSDDetector>(m, "MSDDetector")
        .def(py::init<int, int, int, int, float, int, float, int, bool>(),
                                      "m_patch_radius"_a=3, 
                                      "m_search_area_radius"_a=5,
                                      "m_nms_radius"_a= 5, 
                                      "m_nms_scale_radius"_a=0, 
                                      "m_th_saliency"_a=250.0f, 
                                      "m_kNN"_a=4,
                                      "m_scale_factor"_a=1.25f, 
                                      "m_n_scales"_a=-1, 
                                      "m_compute_orientation"a_= false);         
}
