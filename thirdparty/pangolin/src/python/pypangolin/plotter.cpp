/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) Andrey Mnatsakanov, Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "view.hpp"
#include "plotter.hpp"
#include "handler.hpp"
#include <pangolin/plot/plotter.h>


namespace py_pangolin {

  //extern pybind11::class_<pangolin::Handler, PyHandler> handler;
  
  void bind_plotter(pybind11::module& m){
  //   pybind11::class_<pangolin::Handler, PyHandler> handler1(m, "Handler1");
  //   handler1
  //     .def(pybind11::init<>())
  //     .def("Keyboard", &pangolin::Handler::Keyboard)
  //     .def("Mouse", &pangolin::Handler::Mouse)
  //     .def("MouseMotion", &pangolin::Handler::MouseMotion)
  //     .def("PassiveMouseMotion", &pangolin::Handler::PassiveMouseMotion)
  //     .def("Special", &pangolin::Handler::Special);

    pybind11::enum_<pangolin::DrawingMode>(m, "DrawingMode")
      .value("DrawingModePoints", pangolin::DrawingMode::DrawingModePoints)
      .value("DrawingModeDashed", pangolin::DrawingMode::DrawingModeDashed)
      .value("DrawingModeLine", pangolin::DrawingMode::DrawingModeLine)
      .value("DrawingModeNone", pangolin::DrawingMode::DrawingModeNone)
      .export_values();

    pybind11::class_<pangolin::Marker> marker(m, "Marker");

    pybind11::enum_<pangolin::Marker::Direction>(marker, "Direction")
      .value("Horizontal", pangolin::Marker::Direction::Horizontal)
      .value("Vertical", pangolin::Marker::Direction::Vertical)
      .export_values();

    pybind11::enum_<pangolin::Marker::Equality>(marker, "Equality")
      .value("LessThan", pangolin::Marker::Equality::LessThan)
      .value("Equal", pangolin::Marker::Equality::Equal)
      .value("GreaterThan", pangolin::Marker::Equality::GreaterThan)
      .export_values();

    marker.def(pybind11::init<pangolin::Marker::Direction, float, pangolin::Marker::Equality, pangolin::Colour>(), pybind11::arg("d"), pybind11::arg("value"), pybind11::arg("leg") = pangolin::Marker::Equality::Equal, pybind11::arg("c")=pangolin::Colour())
      .def(pybind11::init<const pangolin::XYRangef&, const pangolin::Colour&>(), pybind11::arg("range"), pybind11::arg("c")=pangolin::Colour())
      .def_readwrite("range", &pangolin::Marker::range)
      .def_readwrite("colour", &pangolin::Marker::colour);    

    pybind11::class_<pangolin::Plotter,pangolin::View>(m, "Plotter")
      .def(pybind11::init<pangolin::DataLog*, float, float, float, float, float, float, pangolin::Plotter*, pangolin::Plotter*>(),
           pybind11::arg("default_log"),
           pybind11::arg("left")=0,
           pybind11::arg("right")=600,
           pybind11::arg("bottom")=-1,
           pybind11::arg("top")=1,
           pybind11::arg("tickx")=30,
           pybind11::arg("ticky")=0.5,
           pybind11::arg("linked_plotter_x")=NULL,
           pybind11::arg("linked_plotter_y")=NULL)
      .def(pybind11::init([](pangolin::DataLog* log, float left, float right, float bottom, float top, float tickx, float ticky){return new pangolin::Plotter(log, left, right, bottom, top,tickx, ticky);}),
           pybind11::arg("default_log"),
           pybind11::arg("left")=0,
           pybind11::arg("right")=600,
           pybind11::arg("bottom")=-1,
           pybind11::arg("top")=1,
           pybind11::arg("// TODO: ickx")=30,
           pybind11::arg("ticky")=0.5)

      .def("Render", &pangolin::Plotter::Render)
      .def("GetSelection", &pangolin::Plotter::GetSelection)
      .def("GetDefaultView", &pangolin::Plotter::GetDefaultView)
      .def("SetDefaultView", &pangolin::Plotter::SetDefaultView)
      .def("GetView", &pangolin::Plotter::GetView)
      .def("SetView", &pangolin::Plotter::SetView)
      .def("SetViewSmooth", &pangolin::Plotter::SetViewSmooth)
      .def("ScrollView", &pangolin::Plotter::ScrollView)
      .def("ScrollViewSmooth", &pangolin::Plotter::ScrollViewSmooth)
      .def("ScaleView", &pangolin::Plotter::ScaleView)
      .def("ScaleViewSmooth", &pangolin::Plotter::ScaleViewSmooth)
      .def("SetTicks", &pangolin::Plotter::SetTicks)
      .def("Track", &pangolin::Plotter::Track, pybind11::arg("x")="$i", pybind11::arg("y")="")
      .def("ToggleTracking", &pangolin::Plotter::ToggleTracking)
      .def("Trigger", &pangolin::Plotter::Trigger, pybind11::arg("x")="$0", pybind11::arg("edge")=-1, pybind11::arg("value")=0.0f)
      .def("ToggleTrigger", &pangolin::Plotter::ToggleTrigger)
      .def("SetBackgroundColour", &pangolin::Plotter::SetBackgroundColour)
      .def("SetAxisColour", &pangolin::Plotter::SetAxisColour)
      .def("SetTickColour", &pangolin::Plotter::SetTickColour)
      .def("Keyboard", &pangolin::Plotter::Keyboard)
      .def("Mouse", &pangolin::Plotter::Mouse)
      .def("MouseMotion", &pangolin::Plotter::MouseMotion)
      .def("PassiveMouseMotion", &pangolin::Plotter::PassiveMouseMotion)
      .def("Special", &pangolin::Plotter::Special)
      .def("ClearSeries", &pangolin::Plotter::ClearSeries)
      .def("AddSeries", &pangolin::Plotter::AddSeries)
      .def("PlotTitleFromExpr", &pangolin::Plotter::PlotTitleFromExpr)
      .def("ClearMarkers", &pangolin::Plotter::ClearMarkers)
      .def("AddMarker", (pangolin::Marker& (pangolin::Plotter::*)(pangolin::Marker::Direction, float, pangolin::Marker::Equality, pangolin::Colour))&pangolin::Plotter::AddMarker, pybind11::arg("d"), pybind11::arg("value"), pybind11::arg("leg") = pangolin::Marker::Equality::Equal, pybind11::arg("c")=pangolin::Colour())
      .def("AddMarker", (pangolin::Marker& (pangolin::Plotter::*)(const pangolin::Marker&))&pangolin::Plotter::AddMarker)
    //      .def("ClearImplicitPlots", &pangolin::Plotter::ClearImplicitPlots);
      //   .def("AddImplicitPlot", &pangolin::Plotter::AddImplicitPlot);
      .def("SetBounds", (pangolin::View& (pangolin::Plotter::*)(pangolin::Attach, pangolin::Attach, pangolin::Attach, pangolin::Attach))&pangolin::Plotter::SetBounds)
      .def("SetBounds", (pangolin::View& (pangolin::Plotter::*)(pangolin::Attach, pangolin::Attach, pangolin::Attach, pangolin::Attach, bool))&pangolin::Plotter::SetBounds)
      .def("SetBounds", (pangolin::View& (pangolin::Plotter::*)(pangolin::Attach, pangolin::Attach, pangolin::Attach, pangolin::Attach, double))&pangolin::Plotter::SetBounds);

  }

}  // py_pangolin
