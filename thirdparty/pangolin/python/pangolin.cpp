#ifdef __linux__
#include <cinttypes>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "utils/params.hpp"
#include "display/attach.hpp"
#include "display/window.hpp"
#include "display/user_app.hpp"
#include "display/viewport.hpp"
#include "display/view.hpp"
#include "display/opengl_render_state.hpp"
#include "handler/handler_enums.hpp"
#include "handler/handler.hpp"
#include "handler/handler_image.hpp"
#include "var/varvaluegeneric.hpp"
#include "var/varvaluet.hpp"
#include "var/varvalue.hpp"
#include "var/var.hpp"
#include "var/varextra.hpp"
#include "display/image_view.hpp"
#include "display/display.hpp"
#include "display/widgets/widgets.hpp"
#include "scene/scene.hpp"
#include "plot/plotter.hpp"
#include "gl/gl.hpp"
#include "gl/gldraw.hpp"
#include "contrib.hpp"



namespace py = pybind11;


namespace pangolin {

PYBIND11_MODULE(pypangolin, m) {
    

    // declaration order matters

    // utils/params
    declareParams(m);

    // display/attach
    declareAttach(m);

    // display/window
    declareWindow(m);

    // display/user_app
    declareUserApp(m);

    // display/viewport
    declareViewport(m);
    // display/view
    declareView(m);

    // display/opengl_render_state
    declareOpenGlRenderState(m);

    // handler/handler_enums
    declareHandlerEnums(m);
    // handler/handler
    declareHandler(m);
    // handler/handler_image
    declareImageViewHandler(m);

    // var/varvaluegeneric
    declareVarValueGeneric(m);
    // var/varvaluet
    declareVarValueT(m);
    // var/varvalue
    declareVarValue(m);
    // var/var
    declareVar(m);
    // var/varextra
    declareVarExtra(m);


    // display/image_view
    declareImageView(m);

    // display/display
    declareDisplay(m);

    // display/widgets/widgets
    declareWidgets(m);

    // scene
    declareScene(m);

    // plot
    declarePlotter(m);

    // gl/gl
    declareGL(m);
    // gl/gldraw
    declareGLDraw(m);

    // contrib
    declareContrib(m);

    }

}