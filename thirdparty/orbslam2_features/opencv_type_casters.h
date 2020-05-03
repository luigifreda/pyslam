
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/core/core.hpp>
#include <stdexcept>
#include <iostream>
#include <exception>

namespace py = pybind11;
using namespace pybind11::literals;



// adapted from https://github.com/pybind/pybind11/issues/2004
// another example for cv::Mat <-> numpy conversions is https://github.com/pybind/pybind11/issues/538#issuecomment-263884464



// void declareCvTypes(py::module & m)   // just experiemental 
// {
//     // N.B. this produces orbslam2_features.KeyPoint which are identical to cv2.KeyPoint but cannot be used as cv2.KeyPoint; 
//     //      at the present time, we use the converter in opencv_type_casters.h!
//     py::class_<cv::KeyPoint>(m, "KeyPoint")
//         .def(py::init<cv::Point2f, float, float, float, int, int>(),"_pt"_a,"_size"_a,"_angle"_a=-1,"_response"_a=0,"_octave"_a=0,"_class_id"_a=-1)
//         .def(py::init<float, float,float, float, float, int, int>(),"x"_a,"y"_a,"_size"_a,"_angle"_a=-1,"_response"_a=0,"_octave"_a=0,"_class_id"_a=-1)
//         .def_readwrite("pt", &cv::KeyPoint::pt) 
//         .def_readwrite("size", &cv::KeyPoint::size)   
//         .def_readwrite("angle", &cv::KeyPoint::angle)    
//         .def_readwrite("response", &cv::KeyPoint::response)    
//         .def_readwrite("octave", &cv::KeyPoint::octave)
//         .def_readwrite("class_id", &cv::KeyPoint::class_id);       

// }


namespace pybind11 { namespace detail{

//cv::Point <-> tuple(x,y) 
template<>
struct type_caster<cv::Point>{
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xi_yi"));

    bool load(handle obj, bool){
        if(!py::isinstance<py::tuple>(obj)){
            std::logic_error("Point(x,y) should be a tuple!");
            return false;
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if(pt.size()!=2){
            std::logic_error("Point(x,y) tuple should be size of 2");
            return false;
        }

        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point& pt, return_value_policy, handle){
        return py::make_tuple(pt.x, pt.y).release();
    }
};

}} //!  end namespace pybind11::detail


namespace pybind11 { namespace detail{

//cv::Point2f <-> tuple(x,y)
template<>
struct type_caster<cv::Point2f>{
    PYBIND11_TYPE_CASTER(cv::Point2f, _("tuple_xf_yf"));

    bool load(handle obj, bool){
        if(!py::isinstance<py::tuple>(obj)){
            std::logic_error("Point2f(x,y) should be a tuple!");
            return false;
        }

        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if(pt.size()!=2){
            std::logic_error("Point2f(x,y) tuple should be size of 2");
            return false;
        }

        value = cv::Point2f(pt[0].cast<float>(), pt[1].cast<float>());
        return true;
    }

    static handle cast(const cv::Point2f& pt, return_value_policy, handle){
        return py::make_tuple(pt.x, pt.y).release();
    }
};

}} //!  end namespace pybind11::detail


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
            std::logic_error("Keypoint (pt.x, pt.y, size, angle, response, octave) tuple should be size of 5");
            return false;
        }

        //value = cv::KeyPoint(keypoint[0].cast<cv::Point2f>(), keypoint[1].cast<float>(), keypoint[2].cast<float>(), keypoint[3].cast<float>(), keypoint[4].cast<int>());
        value = cv::KeyPoint(keypoint[0].cast<float>(), keypoint[1].cast<float>(), keypoint[2].cast<float>(), keypoint[3].cast<float>(), keypoint[4].cast<float>(), keypoint[5].cast<int>());        
        return true;
    }

    static handle cast(const cv::KeyPoint& keypoint, return_value_policy, handle){
        return py::make_tuple(keypoint.pt.x, keypoint.pt.y, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave).release();
    }
};

}} //!  end namespace pybind11::detail



namespace pybind11 { namespace detail{

//cv::Rect <-> (x,y,w,h)     
template<>
struct type_caster<cv::Rect>{
    PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_x_y_w_h"));

    bool load(handle obj, bool){
        if(!py::isinstance<py::tuple>(obj)){
            std::logic_error("Rect should be a tuple!");
            return false;
        }
        py::tuple rect = reinterpret_borrow<py::tuple>(obj);
        if(rect.size()!=4){
            std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
            return false;
        }

        value = cv::Rect(rect[0].cast<int>(), rect[1].cast<int>(), rect[2].cast<int>(), rect[3].cast<int>());
        return true;
    }

    static handle cast(const cv::Rect& rect, return_value_policy, handle){
        return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
    }
};

}} //!  end namespace pybind11::detail



// struct buffer_info {
//     void *ptr;                      /* Pointer to buffer */
//     ssize_t itemsize;               /* Size of one scalar */
//     std::string format;             /* Python struct-style format descriptor */
//     ssize_t ndim;                   /* Number of dimensions */
//     std::vector<ssize_t> shape;     /* Buffer dimensions */
//     std::vector<ssize_t> strides;   /* Strides (in bytes) for each index */
// };


namespace pybind11 { namespace detail{
template<>

// cv::Mat <-> numpy array 
struct type_caster<cv::Mat>{
public:
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat
    bool load(handle obj, bool){
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        //const int ndims = (int)info.ndim;
        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if(ndims == 2){
            nh = info.shape[0];
            nw = info.shape[1];
        } else if(ndims == 3){
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }else{
            char msg[64];
            std::sprintf(msg, "Unsupported dim %d, only support 2d, or 3-d", ndims);
            throw std::logic_error(msg);
            return false;
        }

        int dtype;
        if(info.format == format_descriptor<unsigned char>::format()){
            dtype = CV_8UC(nc);
        }else if (info.format == format_descriptor<int>::format()){
            dtype = CV_32SC(nc);
        }else if (info.format == format_descriptor<float>::format()){
            dtype = CV_32FC(nc);
        }else{
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }

        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval){
        //UNUSED(defval);

        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type)? 2 : 3;

        if(depth == CV_8U){
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }else if(depth == CV_32S){
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }else if(depth == CV_32F){
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }else{
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = {(size_t) nh, (size_t) nw};
            strides = {elemsize * (size_t) nw, elemsize};
        } else if (dim == 3) {
            bufferdim = {(size_t) nh, (size_t) nw, (size_t) nc};
            strides = {(size_t) elemsize * nw * nc, (size_t) elemsize * nc, (size_t) elemsize};
        }
        return array(buffer_info( mat.data,  elemsize,  format, dim, bufferdim, strides )).release();
    }
};

}}//! end namespace pybind11::detail