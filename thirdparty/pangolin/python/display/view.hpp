#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <pangolin/display/view.h>
#include <pangolin/display/opengl_render_state.h>
#include <pangolin/handler/handler.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareView(py::module & m) {

    py::enum_<Layout>(m, "Layout")
        .value("LayoutOverlay", Layout::LayoutOverlay)
        .value("LayoutVertical", Layout::LayoutVertical)
        .value("LayoutHorizontal", Layout::LayoutHorizontal)
        .value("LayoutEqual", Layout::LayoutEqual)
        .value("LayoutEqualVertical", Layout::LayoutEqualVertical)
        .value("LayoutEqualHorizontal", Layout::LayoutEqualHorizontal)
        .export_values();

    py::enum_<Lock>(m, "Lock")
        .value("LockLeft", Lock::LockLeft)
        .value("LockBottom", Lock::LockBottom)
        .value("LockCenter", Lock::LockCenter)
        .value("LockRight", Lock::LockRight)
        .value("LockTop", Lock::LockTop)
        .export_values();


    py::class_<View, std::shared_ptr<View>>(m, "View")

        .def(py::init<double>(), "aspect"_a)

        .def("Activate", (void (View::*)() const) &View::Activate,
            "Activate Displays viewport for drawing within this area")
        .def("Activate", (void (View::*)(const OpenGlRenderState&) const) &View::Activate, "state"_a,
            "Activate Displays and set State Matrices")

        .def("ActivateAndScissor", (void (View::*) () const) &View::ActivateAndScissor,
            "Activate Displays viewport and Scissor for drawing within this area")
        .def("ActivateAndScissor", (void (View::*) (const OpenGlRenderState&) const) &View::ActivateAndScissor,
            "Activate Display and set State Matrices")

        .def("ActivateScissorAndClear", (void (View::*) () const) &View::ActivateScissorAndClear,
            "Activate Displays viewport and Scissor for drawing within this area")
        .def("ActivateScissorAndClear", (void (View::*) (const OpenGlRenderState&) const) 
            &View::ActivateScissorAndClear,
            " Activate Display and set State Matrices")

        .def("ActivatePixelOrthographic", &View::ActivatePixelOrthographic, 
            "Activate Display and setup coordinate system for 2d pixel View coordinates")

        .def("ActivateIdentity", &View::ActivateIdentity,
            "Activate Display and reset coordinate system to OpenGL default")

        .def("GetClosestDepth", &View::GetClosestDepth, "winx"_a, "winy"_a, "radius"_a,
            "Return closest depth buffer value within radius of window (winx,winy)")

        .def("GetCamCoordinates", &View::GetCamCoordinates,
            "cam_state"_a, "winx"_a, "winy"_a, "winzdepth"_a, "x"_a, "y"_a, "z"_a,
            "Obtain camera space coordinates of scene at pixel (winx, winy, winzdepth)")

        .def("GetObjectCoordinates", &View::GetObjectCoordinates,
            "cam_state"_a, "winx"_a, "winy"_a, "winzdepth"_a, "x"_a, "y"_a, "z"_a,
            "Obtain object space coordinates of scene at pixel (winx, winy, winzdepth)")

        .def("Resize", &View::Resize, "parent"_a,
            "Given the specification of Display, compute viewport")

        .def("ResizeChildren", &View::ResizeChildren, 
            "Instruct all children to resize")

        .def("Render", &View::Render,
            "Perform any automatic rendering for this View.")

        .def("RenderChildren", &View::RenderChildren,
            "Instruct all children to render themselves if appropriate")

        .def("SetFocus", &View::SetFocus, 
             py::return_value_policy::reference_internal,
            "Set this view as the active View to receive input")

        .def("HasFocus", &View::HasFocus,
            "Returns true iff this view currently has focus and will receive user input")

        .def("SetBounds", (View& (View::*)(Attach, Attach, Attach, Attach)) 
            &View::SetBounds, "bottom"_a, "top"_a, "left"_a, "right"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")
        .def("SetBounds", (View& (View::*)(Attach, Attach, Attach, Attach, bool)) 
            &View::SetBounds, "bottom"_a, "top"_a, "left"_a, "right"_a, "keep_aspect"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")
        .def("SetBounds", (View& (View::*)(Attach, Attach, Attach, Attach, double)) 
            &View::SetBounds, "bottom"_a, "top"_a, "left"_a, "right"_a, "aspect"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")

        .def("SetBounds", 
            [](View &view, float bottom, float top, float left, float right) {
                return std::ref(view.SetBounds(bottom, top, left, right));},
            "bottom"_a, "top"_a, "left"_a, "right"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")
        .def("SetBounds", 
            [](View &view, float bottom, float top, float left, float right, bool keep_aspect) {
                return std::ref(view.SetBounds(bottom, top, left, right, keep_aspect));},
            "bottom"_a, "top"_a, "left"_a, "right"_a, "keep_aspect"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")
        .def("SetBounds", 
            [](View &view, float bottom, float top, float left, float right, double aspect) {
                return std::ref(view.SetBounds(bottom, top, left, right, aspect));},
            "bottom"_a, "top"_a, "left"_a, "right"_a, "aspect"_a,
             py::return_value_policy::reference_internal,
            "Set bounds for the View using mixed fractional / pixel coordinates (OpenGl view coordinates)")
        
        .def("SetHandler", &View::SetHandler, 
             py::return_value_policy::reference_internal,
            "Designate handler for accepting mouse / keyboard input.",
            py::keep_alive<1, 2>())

        .def("SetDrawFunction", &View::SetDrawFunction, "drawFunc"_a,
             py::return_value_policy::reference_internal,
            "Set drawFunc as the drawing function for this view")

        .def("SetAspect", &View::SetAspect, "aspect"_a,
             py::return_value_policy::reference_internal,
            "Force this view to have the given aspect, whilst fitting snuggly")

        .def("SetLock", &View::SetLock, "horizontal"_a, "vertical"_a,
             py::return_value_policy::reference_internal,
            "Set how this view should be positioned relative to its parent")

        .def("SetLayout", &View::SetLayout, "layout"_a,
             py::return_value_policy::reference_internal,
            "Set layout policy for this view")

        .def("AddDisplay", &View::AddDisplay, "view"_a,
             py::return_value_policy::reference_internal,
            "Add view as child", py::keep_alive<1, 2>())

        .def("Show", &View::Show, "show"_a = true,
            py::return_value_policy::reference_internal,
            "Show / hide this view")

        .def("ToggleShow", &View::ToggleShow,
            "Toggle this views visibility")

        .def("IsShown", &View::IsShown,
            "Return whether this view should be shown.")

        .def("GetBounds", &View::GetBounds, 
            "Returns viewport reflecting space that will actually get drawn")

        .def("SaveOnRender", &View::SaveOnRender, "filename_prefix"_a,
            "Specify that this views region in the framebuffer should be saved to\n"
            "a file just before the buffer is flipped.")

        .def("RecordOnRender", &View::RecordOnRender, "record_uri"_a,
            "Specify that this views region in the framebuffer should be saved to\n"
            "a file just before the buffer is flipped.")

        .def("SaveRenderNow", &View::SaveRenderNow, "filename_prefix"_a, "scale"_a = 1,
            "Uses the views default render method to draw into an FBO 'scale' times\n"
            "the size of the view and save to a file.")

        .def("NumChildren", &View::NumChildren,
            "Return number of child views attached to this view")

        .def("__getitem__", &View::operator[],
            py::return_value_policy::reference_internal, py::is_operator(), 
            "Return (i)th child of this view")

        .def("NumVisibleChildren", &View::NumVisibleChildren,
            "Return number of visible child views attached to this view.")

        .def("VisibleChild", &View::VisibleChild,
            py::return_value_policy::reference_internal,
            "Return visible child by index.")

        .def("FindChild", &View::FindChild, "x"_a, "y"_a,
            py::return_value_policy::reference_internal,
            "Return visible child at window coords x,y")

        .def_readwrite("aspect", &View::aspect, "Desired width / height aspect (0 if dynamic)")   // double
        .def_readwrite("top", &View::top)   // Attach
        .def_readwrite("left", &View::left)   // Attach
        .def_readwrite("right", &View::right)   // Attach
        .def_readwrite("bottom", &View::bottom)   // Attach
        .def_readwrite("hlock", &View::hlock)   // Lock
        .def_readwrite("vlock", &View::vlock)   // Lock
        .def_readwrite("layout", &View::layout)   // Layout
        .def_readwrite("scroll_offset", &View::scroll_offset)   // int
        .def_readwrite("vp", &View::vp, "Cached client area (space allocated from parent)")   // Viewport
        .def_readwrite("v", &View::v, "Cached absolute viewport (recomputed on resize - respects aspect)")   // Viewport
        .def_readwrite("show", &View::show, "Should this view be displayed?")   // bool
        .def_readwrite("zorder", &View::zorder, "Child views are rendered in order of low to high z-order")   // int
        .def_readwrite("handler", &View::handler, "Input event handler (if any)")   // Handler*
        .def_readwrite("views", &View::views, "Map for sub-displays (if any)")   // std::vector<View*>

        // std::function<void(View&)>
        .def_readwrite("extern_draw_function", &View::extern_draw_function, "External draw function")
    
        // Private copy constructor
    ;   

}

}  // namespace pangolin::