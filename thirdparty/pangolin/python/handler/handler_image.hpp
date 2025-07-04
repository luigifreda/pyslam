#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/handler/handler_image.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareImageViewHandler(py::module & m) {

    py::class_<ImageViewHandler, std::shared_ptr<ImageViewHandler>, Handler>(m, "ImageViewHandler")
    
    ;

}

}  // namespace pangolin::