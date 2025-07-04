#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <pangolin/display/opengl_render_state.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareOpenGlRenderState(py::module & m) {

    py::enum_<OpenGlStack>(m, "OpenGlStack")
        .value("GlModelViewStack", OpenGlStack::GlModelViewStack)
        .value("GlProjectionStack", OpenGlStack::GlProjectionStack)
        .value("GlTextureStack", OpenGlStack::GlTextureStack)
        .export_values();

    py::enum_<AxisDirection>(m, "AxisDirection")
        .value("AxisNone", AxisDirection::AxisNone)
        .value("AxisNegX", AxisDirection::AxisNegX)
        .value("AxisX", AxisDirection::AxisX)
        .value("AxisNegY", AxisDirection::AxisNegY)
        .value("AxisY", AxisDirection::AxisY)
        .value("AxisNegZ", AxisDirection::AxisNegZ)
        .value("AxisZ", AxisDirection::AxisZ)
        .export_values();

    py::class_<CameraSpec, std::shared_ptr<CameraSpec>>(m, "CameraSpec");
        //.def_readwrite("forward", &CameraSpec::forward)   // "error: invalid array assignment"
        //.def_readwrite("up", &CameraSpec::up)
        //.def_readwrite("right", &CameraSpec::right)
        //.def_readwrite("img_up", &CameraSpec::img_up)
        //.def_readwrite("img_right", &CameraSpec::img_right);

    
    py::class_<OpenGlMatrix, std::shared_ptr<OpenGlMatrix>>(m, "OpenGlMatrix", py::buffer_protocol())
        .def_static("Translate", &OpenGlMatrix::Translate, "x"_a, "y"_a, "z"_a)
        .def_static("Scale", &OpenGlMatrix::Scale, "x"_a, "y"_a, "z"_a)
        .def_static("RotateX", &OpenGlMatrix::RotateX, "theta_rad"_a)
        .def_static("RotateY", &OpenGlMatrix::RotateY, "theta_rad"_a)
        .def_static("RotateZ", &OpenGlMatrix::RotateZ, "theta_rad"_a)

        .def_static("ColMajor4x4", &OpenGlMatrix::ColMajor4x4<float>, "col_major_4x4"_a)
        .def_static("ColMajor4x4", &OpenGlMatrix::ColMajor4x4<double>, "col_major_4x4"_a)

        //.def(py::init<>())
        .def(py::init([](){
                OpenGlMatrix glmat;
                glmat.SetIdentity();
                return glmat;
            }))
        // EIGEN, TOON, OCULUS
#ifdef USE_EIGEN
        .def(py::init<const Eigen::Matrix<float,4,4>&>(), "mat"_a)
        .def(py::init<const Eigen::Matrix<double,4,4>&>(), "mat"_a)
#endif

        .def("Load", &OpenGlMatrix::Load)
        .def("Multiply", &OpenGlMatrix::Multiply)
        .def("SetIdentity", &OpenGlMatrix::SetIdentity)
        .def("Transpose", &OpenGlMatrix::Transpose)
        .def("Inverse", &OpenGlMatrix::Inverse)

        .def("__getitem__", [](OpenGlMatrix &glmat, std::pair<int, int> i) {
                if (i.first >= 4 || i.second >= 4)
                    throw py::index_error();
                return glmat(i.first, i.second);
            })
        .def("__setitem__", [](OpenGlMatrix &glmat, std::pair<int, int> i, GLprecision v) {
                if (i.first >= 4 || i.second >= 4)
                    throw py::index_error();
                glmat(i.first, i.second) = v;
            })

        .def_buffer([](OpenGlMatrix &glmat) -> py::buffer_info {
                return py::buffer_info(
                    glmat.m,
                    {4, 4},
                    {sizeof(GLprecision), sizeof(GLprecision) * 4}
                );
            })

        .def("numpy_view", [](py::object &obj) {   // doesn't copy data
                OpenGlMatrix &glmat = obj.cast<OpenGlMatrix&>();
                return py::array_t<GLprecision>({4, 4}, {sizeof(GLprecision), sizeof(GLprecision) * 4}, glmat.m, obj);
            })

        .def_property("m", 
                [](OpenGlMatrix &glmat) -> const py::array {                                                     // getter
                    return py::array_t<GLprecision>({4, 4}, {sizeof(GLprecision), sizeof(GLprecision) * 4}, glmat.m);
                },
                [](OpenGlMatrix &glmat, Eigen::Matrix<GLprecision, 4, 4>& v) {                                   // setter
                    for(int r=0; r<4; ++r ) {
                        for(int c=0; c<4; ++c ) {
                            glmat.m[c*4+r] = v(r,c);
                        }
                    }
                }
            )

        .def("__mul__", [](const OpenGlMatrix& lhs, const OpenGlMatrix& rhs) {
            return lhs * rhs;})        
    ;


    // something wrong with inhertance
    py::class_<OpenGlMatrixSpec, std::shared_ptr<OpenGlMatrixSpec>, OpenGlMatrix>(m, "OpenGlMatrixSpec")
        .def_readwrite("type", &OpenGlMatrixSpec::type);


    py::class_<OpenGlRenderState, std::shared_ptr<OpenGlRenderState>>(m, "OpenGlRenderState")
        .def(py::init<>())
        .def(py::init<const OpenGlMatrix&>(), 
            "projection_matrix"_a)
        .def(py::init<const OpenGlMatrix&, const OpenGlMatrix&>(),
            "projection_matrix"_a, "modelview_matrx"_a)

        .def("Apply", &OpenGlRenderState::Apply)
        .def("ApplyNView", &OpenGlRenderState::ApplyNView, "view"_a)
        .def("ApplyIdentity", &OpenGlRenderState::ApplyIdentity)
        .def("SetProjectionMatrix", &OpenGlRenderState::SetProjectionMatrix, "m"_a,
            py::return_value_policy::reference_internal)
        .def("SetModelViewMatrix", &OpenGlRenderState::SetModelViewMatrix, "m"_a, 
            py::return_value_policy::reference_internal)
        //.def("Set", &OpenGlRenderState::Set, "m"_a,   // deprecated
        //    py::return_value_policy::reference_internal)

        .def("GetViewOffset", (OpenGlMatrix& (OpenGlRenderState::*) (unsigned int)) 
            &OpenGlRenderState::GetViewOffset, "view"_a, py::return_value_policy::reference_internal)
        .def("GetViewOffset", (OpenGlMatrix (OpenGlRenderState::*) (unsigned int) const)
            &OpenGlRenderState::GetViewOffset, "view"_a)

        .def("GetProjectionMatrix", (OpenGlMatrix& (OpenGlRenderState::*) ()) 
            &OpenGlRenderState::GetProjectionMatrix, py::return_value_policy::reference_internal)
        .def("GetProjectionMatrix", (OpenGlMatrix (OpenGlRenderState::*) () const)
            &OpenGlRenderState::GetProjectionMatrix)
        .def("GetProjectionMatrix", (OpenGlMatrix& (OpenGlRenderState::*) (unsigned int)) 
            &OpenGlRenderState::GetProjectionMatrix, "view"_a, py::return_value_policy::reference_internal)
        .def("GetProjectionMatrix", (OpenGlMatrix (OpenGlRenderState::*) (unsigned int) const)
            &OpenGlRenderState::GetProjectionMatrix, "view"_a)

        .def("GetModelViewMatrix", (OpenGlMatrix& (OpenGlRenderState::*) ()) 
            &OpenGlRenderState::GetModelViewMatrix, py::return_value_policy::reference_internal)
        .def("GetModelViewMatrix", (OpenGlMatrix (OpenGlRenderState::*) () const) 
            &OpenGlRenderState::GetModelViewMatrix)
        .def("GetModelViewMatrix", (OpenGlMatrix (OpenGlRenderState::*) (int) const) 
            &OpenGlRenderState::GetModelViewMatrix, "i"_a)

        .def("GetProjectionModelViewMatrix", &OpenGlRenderState::GetProjectionModelViewMatrix)
        .def("GetProjectiveTextureMatrix", &OpenGlRenderState::GetProjectiveTextureMatrix)
        .def("EnableProjectiveTexturing", &OpenGlRenderState::EnableProjectiveTexturing)
        .def("DisableProjectiveTexturing", &OpenGlRenderState::DisableProjectiveTexturing)
        .def("Follow", &OpenGlRenderState::Follow, "T_wc"_a, "follow"_a)
        .def("Unfollow", &OpenGlRenderState::Unfollow)
    ;
        
        
    m.def("ProjectionMatrix", &ProjectionMatrix, 
        "w"_a, "h"_a, "fu"_a, "fv"_a, "u0"_a, "v0"_a, "zNear"_a, "zFar"_a);

    m.def("ProjectionMatrixOrthographic", &ProjectionMatrixOrthographic,
        "l"_a, "r"_a, "b"_a, "t"_a, "n"_a, "f"_a);

    m.def("ProjectionMatrixRUB_BottomLeft", &ProjectionMatrixRUB_BottomLeft,
        "w"_a, "h"_a, "fu"_a, "fv"_a, "u0"_a, "v0"_a, "zNear"_a, "zFar"_a);

    m.def("ProjectionMatrixRDF_TopLeft", &ProjectionMatrixRDF_TopLeft,
        "w"_a, "h"_a, "fu"_a, "fv"_a, "u0"_a, "v0"_a, "zNear"_a, "zFar"_a);

    m.def("ModelViewLookAtRUB", &ModelViewLookAtRUB,
        "ex"_a, "ey"_a, "ez"_a, "lx"_a, "ly"_a, "lz"_a, "ux"_a, "uy"_a, "uz"_a);
         
    m.def("ModelViewLookAtRDF", &ModelViewLookAtRDF,
        "ex"_a, "ey"_a, "ez"_a, "lx"_a, "ly"_a, "lz"_a, "ux"_a, "uy"_a, "uz"_a);

    m.def("ModelViewLookAt", (OpenGlMatrix (*) (GLprecision, GLprecision, GLprecision, 
        GLprecision, GLprecision, GLprecision, GLprecision, GLprecision, GLprecision))
        &ModelViewLookAt,
        "ex"_a, "ey"_a, "ez"_a, "lx"_a, "ly"_a, "lz"_a, "ux"_a, "uy"_a, "uz"_a);

    m.def("ModelViewLookAt", (OpenGlMatrix (*) (GLprecision, GLprecision,
        GLprecision, GLprecision, GLprecision, GLprecision, AxisDirection)) 
        &ModelViewLookAt,
        "ex"_a, "ey"_a, "ez"_a, "lx"_a, "ly"_a, "lz"_a, "up"_a);

    m.def("IdentityMatrix", (OpenGlMatrix (*) ()) &IdentityMatrix);
    m.def("IdentityMatrix", (OpenGlMatrixSpec (*) (OpenGlStack)) &IdentityMatrix);
    m.def("IdentityMatrix", &negIdentityMatrix);

    // operator<<
}

}  // namespace pangolin::