#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/handler/handler.h>
#include <pangolin/plot/plotter.h>

#include "datalog.hpp"
#include "range.hpp"



namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

struct Colour;

void declarePlotter(py::module & m) {

    declareDataLog(m);
    declareRange(m);


    // py::class_<Colour>(m, "Colour")
    //     .def(py::init<>())
    // ;

    py::class_<Colour>(m, "Colour")
        .def_static("White", &Colour::White)
        .def_static("Black", &Colour::Black)
        .def_static("Red", &Colour::Red)
        .def_static("Green", &Colour::Green)
        .def_static("Blue", &Colour::Blue)
        .def_static("Unspecified", &Colour::Unspecified)

        .def(py::init<>())
        .def(py::init<float, float, float, float>(),
            "red"_a, "green"_a, "blue"_a, "alpha"_a=1.0)
        // .def(py::init<float[4]>(), "rgba")

        .def("Get", &Colour::Get)
        .def("WithAlpha", &Colour::WithAlpha)
        .def_static("Hsv", &Colour::Hsv,
            "hue"_a, "sat"_a=1.0, "val"_a=1.0, "alpha"_a=1.0)
    ;


    py::enum_<DrawingMode>(m, "DrawingMode")
        .value("DrawingModePoints", DrawingMode::DrawingModePoints)
        .value("DrawingModeDashed", DrawingMode::DrawingModeDashed)
        .value("DrawingModeLine", DrawingMode::DrawingModeLine)
        .value("DrawingModeNone", DrawingMode::DrawingModeNone)
        .export_values()
    ;

    
    py::class_<Marker> cls_mk(m, "Marker");
        py::enum_<Marker::Direction>(cls_mk, "Direction")
            .value("Horizontal", Marker::Direction::Horizontal)
            .value("Vertical", Marker::Direction::Vertical)
            .export_values()
        ;
        py::enum_<Marker::Equality>(cls_mk, "Equality")
            .value("LessThan", Marker::Equality::LessThan)
            .value("Equal", Marker::Equality::Equal)
            .value("GreaterThan", Marker::Equality::GreaterThan)
            .export_values()
        ;

        cls_mk.def(py::init<Marker::Direction, float, Marker::Equality, Colour>(),
            "d"_a, "value"_a, "leg"_a=Marker::Equality::Equal, "c"_a=Colour());
        cls_mk.def(py::init<const XYRangef&, const Colour&>(),
            "range"_a, "c"_a=Colour());

        cls_mk.def_readwrite("range", &Marker::range);
        cls_mk.def_readwrite("colour", &Marker::colour);

    
    py::class_<Plotter, View/*, Handler*/, std::shared_ptr<Plotter>>(m, "Plotter")
        .def(py::init<DataLog*, float, float, float, float, float, float, Plotter*, Plotter*>(),
            "default_log"_a, "left"_a, "right"_a=600, "bottom"_a=-1, "top"_a=1,
            "tickx"_a=30, "ticky"_a=0.5, 
            "linked_plotter_x"_a=nullptr,
            "linked_plotter_y"_a=nullptr)

        .def("Render", &Plotter::Render)
        .def("GetSelection", &Plotter::GetSelection)
        .def("GetDefaultView", &Plotter::GetDefaultView)
        .def("SetDefaultView", &Plotter::SetDefaultView)

        .def("GetView", &Plotter::GetView)
        .def("SetView", &Plotter::SetView)
        .def("SetViewSmooth", &Plotter::SetViewSmooth)
        .def("ScrollView", &Plotter::ScrollView)
        .def("ScrollViewSmooth", &Plotter::ScrollViewSmooth)
        .def("ScaleView", &Plotter::ScaleView)
        .def("ScaleViewSmooth", &Plotter::ScaleViewSmooth)
        .def("ResetView", &Plotter::ResetView)
        .def("SetTicks", &Plotter::SetTicks)
        .def("Track", &Plotter::Track, "x"_a="$i", "y"_a="")
        .def("ToggleTracking", &Plotter::ToggleTracking)
        .def("Trigger", &Plotter::Trigger, "x"_a="$0", "edge"_a=-1, "value"_a=0.0)
        .def("ToggleTrigger", &Plotter::ToggleTrigger)
        .def("SetBackgroundColour", &Plotter::SetBackgroundColour)
        .def("SetAxisColour", &Plotter::SetAxisColour)
        .def("SetTickColour", &Plotter::SetTickColour)
        .def("ScreenToPlot", &Plotter::ScreenToPlot)
        .def("Keyboard", &Plotter::Keyboard)
        .def("Mouse", &Plotter::Mouse)
        .def("MouseMotion", &Plotter::MouseMotion)
        .def("PassiveMouseMotion", &Plotter::PassiveMouseMotion)
        .def("Special", &Plotter::Special)
        .def("ClearSeries", &Plotter::ClearSeries)
        .def("AddSeries", &Plotter::AddSeries, 
            "x_expr"_a, "y_expr"_a, "mode"_a=DrawingModeLine, "colour"_a=Colour::Unspecified(),
            "title"_a="$y", "log"_a=nullptr)
        .def("PlotTitleFromExpr", &Plotter::PlotTitleFromExpr)
        .def("ClearMarkers", &Plotter::ClearMarkers)
        .def("AddMarker", (Marker& (Plotter::*) 
            (Marker::Direction, float, Marker::Equality, Colour)) &Plotter::AddMarker,
            "d"_a, "value"_a, "leg"_a=Marker::Equal, "c"_a=Colour())
        .def("AddMarker", (Marker& (Plotter::*) (const Marker&)) &Plotter::AddMarker)
        // .def("ClearImplicitPlots", &Plotter::ClearImplicitPlots)
        // .def("AddImplicitPlot", &Plotter::AddImplicitPlot)
    ;

    



}

}