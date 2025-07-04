#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/scene/interactive.h>
#include <pangolin/scene/renderable.h>
#include <pangolin/scene/interactive_index.h>
#include <pangolin/scene/scenehandler.h>
#include <pangolin/scene/axis.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareScene(py::module & m) {

    // abstract class
    py::class_<Interactive, std::shared_ptr<Interactive>>(m, "Interactive")
        .def("mouse", &Interactive::Mouse,
            "button"_a, "win"_a, "obj"_a, "normal"_a, "pressed"_a, 
            "button_state"_a, "pickId"_a)     // (int button, const GLprecision win[3], const GLprecision obj[3], const GLprecision normal[3], bool pressed, int button_state, int pickId) -> bool
        .def("MouseMotion", &Interactive::MouseMotion,
            "win"_a, "obj"_a, "normal"_a, 
            "button_state"_a, "pickId"_a)     // (const GLprecision win[3], const GLprecision obj[3], const GLprecision normal[3], int button_state, int pickId) -> bool
    ;

    py::class_<RenderParams>(m, "RenderParams")
        .def(py::init<>())
        .def_readwrite("render_mode", &RenderParams::render_mode)
    ;

    // abstract class
    py::class_<Manipulator, Interactive, std::shared_ptr<Manipulator>>(m, "Manipulator")
        .def("Render", &Manipulator::Render,
            "params"_a)     // (const RenderParams&) -> void
    ;



    py::class_<Renderable, std::shared_ptr<Renderable>>(m, "Renderable")
        // .def(py::init<const std::weak_ptr<Renderable>&>(),
        //     "parent"_a=std::weak_ptr<Renderable>(),
        //     py::keep_alive<1, 2>())
        .def(py::init([](const std::shared_ptr<Renderable>& parent){
                std::weak_ptr<Renderable> weak = parent;
                return Renderable(weak);
            }))
        .def(py::init([](){
                std::shared_ptr<Renderable> parent;
                std::weak_ptr<Renderable> weak = parent;
                return Renderable(weak);
            }))

        .def("UniqueGuid", &Renderable::UniqueGuid)   // () -> guid_t
        .def("Render", &Renderable::Render,
            "params"_a=RenderParams(),
            py::keep_alive<1, 2>())     // (const RenderParams&) -> void
        .def("RenderChildren", &Renderable::RenderChildren,
            "params"_a,
            py::keep_alive<1, 2>())     // (const RenderParams&) -> void
        .def("FindChild", &Renderable::FindChild,
            "guid"_a)   // (guid_t) -> std::shared_ptr<Renderable>
        .def("Add", &Renderable::Add,
            "child"_a,
            py::keep_alive<1, 2>())   // (const std::shared_ptr<Renderable>&) -> Renderable&

        .def_readonly("guid", &Renderable::guid)
        // .def_readwrite("parent", &Renderable::parent)
        .def_readwrite("T_pc", &Renderable::T_pc)
        .def_readwrite("child", &Renderable::child)
        .def_readwrite("should_show", &Renderable::should_show)
        .def_readwrite("children", &Renderable::children)
        .def_readwrite("manipulator", &Renderable::manipulator)
    ;


    py::class_<InteractiveIndex, std::shared_ptr<InteractiveIndex>> cls_index(m, "InteractiveIndex");
        py::class_<InteractiveIndex::Token>(cls_index, "Token")
            .def(py::init<>())
            .def("Id", &InteractiveIndex::Token::Id)
        ;
        cls_index.def_static("I", &InteractiveIndex::I);
        cls_index.def("Find", &InteractiveIndex::Find,
            "id"_a);   // (GLuint) -> Interactive*
        cls_index.def("Store", &InteractiveIndex::Store,
            "r"_a,
            py::keep_alive<1, 2>());   // (Interactive*) -> Token
        cls_index.def("Unstore", &InteractiveIndex::Unstore,
            "t"_a,
            py::keep_alive<1, 2>());   // (Token&) -> void


    py::class_<SceneHandler, std::shared_ptr<SceneHandler>, Handler3D>(m, "SceneHandler")
        .def(py::init<Renderable&, OpenGlRenderState&>(),
            "scene"_a, "cam_state"_a)

        // .def("ProcessHitBuffer", &SceneHandler::ProcessHitBuffer,
        //     "hits"_a, "buf"_a, "hit_map"_a)   // (GLint, GLuint*, std::map<GLuint, Interactive*>&) -> void
        .def("ComputeHits", &SceneHandler::ComputeHits,
            "view"_a, "cam_state"_a, "x"_a, "y"_a, "grad_width"_a, 
            "hit_objects"_a,
            py::keep_alive<1, 2>())    // (pangolin::View&, const pangolin::OpenGlRenderState&, int, int, int, std::map<GLuint, Interactive*>&) -> void

        .def("Mouse", &SceneHandler::Mouse,
            "view"_a, "button"_a, "x"_a, "y"_a, "pressed"_a, 
            "button_state"_a,
            py::keep_alive<1, 2>())   // (pangolin::View&, pangolin::MouseButton, int, int, bool, int) -> void

        .def("MouseMotion", &SceneHandler::MouseMotion,
            "view"_a, "x"_a, "y"_a, "button_state"_a,
            py::keep_alive<1, 2>())   // (pangolin::View&, int, int, int) -> void

        .def("Special", &SceneHandler::Special,
            "view"_a, "inType"_a, "x"_a, "y"_a, "p1"_a, "p2"_a, "p3"_a, "p4"_a,
            "button_state"_a,
            py::keep_alive<1, 2>())    // (pangolin::View&, pangolin::InputSpecial, float, float, float, float, float, float, int) -> void

        .def_readwrite("m_selected_objects", &SceneHandler::m_selected_objects)
        // .def_readwrite("scene", &SceneHandler::scene)
        .def_readwrite("grab_width", &SceneHandler::grab_width)
    ;


    py::class_<Axis, Renderable, Interactive, std::shared_ptr<Axis>>(m, "Axis")
        .def(py::init<>())

        .def("Render", &Axis::Render,
            "params"_a,
            py::keep_alive<1, 2>())   // (const RenderParams&) -> void
        .def("Mouse", (bool (Axis::*) (int, const GLprecision[3], const GLprecision[3], const GLprecision[3], bool, int, int)) 
            &Axis::Mouse,
            "button"_a, "win"_a, "obj"_a, "normal"_a, "pressed"_a, 
            "button_state"_a, "pickId"_a)
        .def("Mouse", [](Axis& axis, int button, int button_state, int pickId){
                GLprecision win[3];
                GLprecision obj[3];
                GLprecision normal[3];
                bool pressed=false;
                return axis.Mouse(button, win, obj, normal, pressed, button_state, pickId);
            },
            "button"_a, "button_state"_a, "pickId"_a)

        .def_readwrite("axis_length", &Axis::axis_length)
        .def_readonly("label_x", &Axis::label_x)
        .def_readonly("label_y", &Axis::label_y)
        .def_readonly("label_z", &Axis::label_z)

    ;


}

}