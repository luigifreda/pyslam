#include <opencv2/core.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "opencv_type_casters.h"

namespace py = pybind11;

static cv::Mat add_scalar(const cv::Mat &m, double s) {
    cv::Mat out;
    m.convertTo(out, m.type());
    if (out.channels() == 1) {
        out = out + s;
    } else {
        std::vector<cv::Mat> chs;
        cv::split(out, chs);
        for (auto &c : chs)
            c = c + s;
        cv::merge(chs, out);
    }
    return out;
}

PYBIND11_MODULE(cvcasters_test, m) {
    m.def("identity_mat", [](const cv::Mat &m) { return m; });
    m.def("add_scalar", &add_scalar);

    m.def("roundtrip_vec3f", [](const cv::Vec3f &v) { return v; });
    m.def("roundtrip_point", [](const cv::Point &p) { return p; });
    m.def("roundtrip_point2f", [](const cv::Point2f &p) { return p; });

    m.def("roundtrip_recti", [](const cv::Rect &r) { return r; });
    m.def("roundtrip_rectf", [](const cv::Rect2f &r) { return r; });

    m.def("roundtrip_dmatch", [](const cv::DMatch &d) { return d; });

    m.def("roundtrip_keypoint", [](const cv::KeyPoint &k) { return k; });

    m.def("roundtrip_mats", [](const std::vector<cv::Mat> &mats) { return mats; });
}
