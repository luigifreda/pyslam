#pragma once

#include <opencv2/core/core.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pyslam {

static inline py::array cvmat_to_numpy(const cv::Mat &m) {
    if (m.empty())
        return py::array();
    CV_Assert(m.isContinuous());
    int ndim = (m.channels() == 1) ? 2 : 3;
    std::vector<ssize_t> shape, strides;
    if (ndim == 2) {
        shape = {m.rows, m.cols};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1]};
    } else {
        shape = {m.rows, m.cols, m.channels()};
        strides = {(ssize_t)m.step[0], (ssize_t)m.step[1], (ssize_t)m.elemSize1()};
    }
    const char *fmt = (m.depth() == CV_8U) ? py::format_descriptor<uint8_t>::format().c_str()
                      : (m.depth() == CV_32F)
                          ? py::format_descriptor<float>::format().c_str()
                          : throw std::runtime_error("Unsupported cv::Mat depth");
    return py::array(py::buffer_info((void *)m.data, m.elemSize1(), fmt, ndim, shape, strides));
}

static inline cv::Mat numpy_to_cvmat(const py::array &a, int depth_hint /* CV_8U/CV_32F */) {
    if (!a || a.ndim() == 0)
        return {};
    auto info = a.request();
    int channels = (info.ndim == 3) ? (int)info.shape[2] : 1;
    int type = CV_MAKETYPE(depth_hint, channels);
    cv::Mat m((int)info.shape[0], (int)info.shape[1], type, info.ptr, info.strides[0]);
    return m.clone(); // own memory
}

} // namespace pyslam