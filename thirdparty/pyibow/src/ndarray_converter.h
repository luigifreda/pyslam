// borrowed in spirit from https://github.com/yati-sagade/opencv-ndarray-conversion
// MIT License

# ifndef __NDARRAY_CONVERTER_H__
# define __NDARRAY_CONVERTER_H__

#include <Python.h>
#include <opencv2/core/core.hpp>


#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/core/core.hpp>
#include <stdexcept>
#include <iostream>
#include <exception>

namespace py = pybind11;
using namespace pybind11::literals;


class NDArrayConverter {
public:
    // must call this first, or the other routines don't work!
    static bool init_numpy();
    
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};

//
// Define the type converter
//

#include <pybind11/pybind11.h>

namespace pybind11 { namespace detail {
    
template <> struct type_caster<cv::Mat> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};
    
    
}} // namespace pybind11::detail



namespace pybind11 { namespace detail{

//cv::KeyPoint <-> (pt.x, pt.y, size, angle, response, octave)
template<>
struct type_caster<cv::KeyPoint>{
    PYBIND11_TYPE_CASTER(cv::KeyPoint, _("tuple_x_y_size_angle_response_octave"));

    bool load(handle obj, bool){

        if(!py::isinstance<py::tuple>(obj)){
            std::logic_error("KeyPoint should be a tuple!");
            return false;
        }
        py::tuple keypoint = reinterpret_borrow<py::tuple>(obj);
        if(keypoint.size()!=6){
            std::logic_error("Keypoint (pt.x, pt.y, size, angle, response, octave) tuple should be size of 6");
            return false;
        }

        value = cv::KeyPoint(keypoint[0].cast<float>(), keypoint[1].cast<float>(), keypoint[2].cast<float>(), keypoint[3].cast<float>(), keypoint[4].cast<float>(), keypoint[5].cast<int>());        
        return true;
    }

    static handle cast(const cv::KeyPoint& keypoint, return_value_policy, handle){
        return py::make_tuple(keypoint.pt.x, keypoint.pt.y, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave).release();
    }
};

}} //!  end namespace pybind11::detail


namespace pybind11 { namespace detail{

//cv::DMatch <-> (queryIdx, trainIdx, imgIdx, distance)
template<>
struct type_caster<cv::DMatch>{
    PYBIND11_TYPE_CASTER(cv::DMatch, _("tuple_queryIdx_trainIdx_imgIdx_distance"));

    bool load(handle obj, bool){

        if(!py::isinstance<py::tuple>(obj)){
            std::logic_error("DMatch should be a tuple!");
            return false;
        }
        py::tuple dmatch = reinterpret_borrow<py::tuple>(obj);
        if(dmatch.size()!=4){
            std::logic_error("DMatch (queryIdx, trainIdx, imgIdx, distance) tuple should be size of 4");
            return false;
        }

        value = cv::DMatch(dmatch[0].cast<int>(), 
                               dmatch[1].cast<int>(), 
                               dmatch[2].cast<int>(), 
                               dmatch[3].cast<float>());
        return true;
    }

    static handle cast(const cv::DMatch& dmatch, return_value_policy, handle){
        return py::make_tuple(dmatch.queryIdx, dmatch.trainIdx, dmatch.imgIdx, dmatch.distance).release();
    }
};

}} //!  end namespace pybind11::detail

# endif
