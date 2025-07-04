/*
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/pangolin.h>


namespace py = pybind11;
using namespace pybind11::literals;



namespace pangolin {

namespace {

#if 0

void DrawPoints(py::array_t<double> points) {
    auto r = points.unchecked<2>();
    glBegin(GL_POINTS);
    for (ssize_t i = 0; i < r.shape(0); ++i) {
        glVertex3d(r(i, 0), r(i, 1), r(i, 2));
    }
    glEnd();
}


void DrawPoints(py::array_t<double> points, py::array_t<double> colors) {
    auto r = points.unchecked<2>();
    auto rc = colors.unchecked<2>();

    glBegin(GL_POINTS);
    for (ssize_t i = 0; i < r.shape(0); ++i) {
        glColor3f(rc(i, 0), rc(i, 1), rc(i, 2));
        glVertex3d(r(i, 0), r(i, 1), r(i, 2));
    }
    glEnd();
}

#else 

void DrawPoints(py::array_t<double> points) {
    // Access numpy array
    auto r = points.unchecked<2>(); // Shape: (num_points, 3)

    // Enable client state for vertex arrays
    glEnableClientState(GL_VERTEX_ARRAY);

    // Provide vertex data
    glVertexPointer(3, GL_DOUBLE, 0, r.data(0, 0));

    // Draw points
    glDrawArrays(GL_POINTS, 0, r.shape(0));

    // Disable client state
    glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawPoints(py::array_t<double> points, py::array_t<double> colors) {
    // Access numpy arrays
    auto r = points.unchecked<2>();  // Shape: (num_points, 3)
    auto rc = colors.unchecked<2>(); // Shape: (num_points, 3)

    // Enable client states for vertex arrays and color arrays
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Provide vertex and color data
    glVertexPointer(3, GL_DOUBLE, 0, r.data(0, 0));
    glColorPointer(3, GL_DOUBLE, 0, rc.data(0, 0));

    // Draw points
    glDrawArrays(GL_POINTS, 0, r.shape(0));

    // Disable client states
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

#endif 


#if 0

void DrawMesh(py::array_t<double> vertices, py::array_t<int> triangles, py::array_t<double> colors) {
    // Access numpy arrays
    auto v = vertices.unchecked<2>();   // Shape: (num_vertices, 3)
    auto c = colors.unchecked<2>();     // Shape: (num_vertices, 3)
    auto t = triangles.unchecked<2>();  // Shape: (num_triangles, 3)

    glBegin(GL_TRIANGLES);
    for (ssize_t i = 0; i < t.shape(0); ++i) {
        for (int j = 0; j < 3; ++j) {  // Each triangle has 3 vertices
            int idx = t(i, j);         // Get vertex index
            glColor3f(c(idx, 0), c(idx, 1), c(idx, 2));  // Set vertex color
            glVertex3d(v(idx, 0), v(idx, 1), v(idx, 2)); // Set vertex position
        }
    }
    glEnd();
}

void DrawMonochromeMesh(py::array_t<double> vertices, py::array_t<int> triangles, float r = 0.5f, float g = 0.5f, float b = 0.5f) {
    // Access numpy arrays
    auto v = vertices.unchecked<2>();   // Shape: (num_vertices, 3)
    auto t = triangles.unchecked<2>();  // Shape: (num_triangles, 3)

    glColor3f(r, g, b);  // Set the constant color for the entire mesh

    glBegin(GL_TRIANGLES);
    for (ssize_t i = 0; i < t.shape(0); ++i) {
        for (int j = 0; j < 3; ++j) {  // Each triangle has 3 vertices
            int idx = t(i, j);         // Get vertex index
            glVertex3d(v(idx, 0), v(idx, 1), v(idx, 2)); // Set vertex position
        }
    }
    glEnd();
}

#else

void DrawMesh(py::array_t<double> vertices, py::array_t<int> triangles, py::array_t<double> colors, bool wireframe) {
    // Access numpy arrays
    auto v = vertices.unchecked<2>();   // Shape: (num_vertices, 3)
    auto t = triangles.unchecked<2>();  // Shape: (num_triangles, 3)
    auto c = colors.unchecked<2>();     // Shape: (num_vertices, 3)

    // Enable wireframe mode if requested
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    // Enable client states
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    // Use vertex and color data directly
    glVertexPointer(3, GL_DOUBLE, 0, v.data(0, 0));
    glColorPointer(3, GL_DOUBLE, 0, c.data(0, 0));

    // Use the triangles array directly
    glDrawElements(GL_TRIANGLES, t.size(), GL_UNSIGNED_INT, t.data(0, 0));

    // Disable client states
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    // Reset polygon mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void DrawMonochromeMesh(py::array_t<double> vertices, py::array_t<int> triangles, std::array<float, 3> color, bool wireframe) {
    // Access numpy arrays
    auto v = vertices.unchecked<2>();   // Shape: (num_vertices, 3)
    auto t = triangles.unchecked<2>();  // Shape: (num_triangles, 3)

    // Enable wireframe mode if requested
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    // Set the monochrome color once
    glColor3f(color[0], color[1], color[2]);

    // Enable client state for vertex arrays
    glEnableClientState(GL_VERTEX_ARRAY);

    // Use vertex data directly
    glVertexPointer(3, GL_DOUBLE, 0, v.data(0, 0));

    // Use the triangles array directly
    glDrawElements(GL_TRIANGLES, t.size(), GL_UNSIGNED_INT, t.data(0, 0));

    // Disable client state
    glDisableClientState(GL_VERTEX_ARRAY);

    // Reset polygon mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

#endif

void DrawCameras(py::array_t<double> cameras, float w=1.0, float h_ratio=0.75, float z_ratio=0.6) {
    auto r = cameras.unchecked<3>();

    float h = w * h_ratio;
    float z = w * z_ratio;

    for (ssize_t i = 0; i < r.shape(0); ++i) {
        glPushMatrix();
        // glMultMatrixd(r.data(i, 0, 0));
        glMultTransposeMatrixd(r.data(i, 0, 0));

        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }
}

void DrawCamera(py::array_t<double> camera, float w=1.0, float h_ratio=0.75, float z_ratio=0.6) {
    auto r = camera.unchecked<2>();

    float h = w * h_ratio;
    float z = w * z_ratio;

    glPushMatrix();
    // glMultMatrixd(r.data(0, 0));
    glMultTransposeMatrixd(r.data(0, 0));

    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void DrawLine(py::array_t<double> points, float point_size=0) {
    auto r = points.unchecked<2>();
    glBegin(GL_LINES);
    for (ssize_t i = 0; i < r.shape(0)-1; ++i) {
        glVertex3d(r(i, 0), r(i, 1), r(i, 2));
        glVertex3d(r(i+1, 0), r(i+1, 1), r(i+1, 2));
    }
    glEnd();

    if(point_size > 0) {
        glPointSize(point_size);
        glBegin(GL_POINTS);
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            glVertex3d(r(i, 0), r(i, 1), r(i, 2));
        }
        glEnd();
    }
}



void DrawLines(py::array_t<double> points, float point_size=0) {
    auto r = points.unchecked<2>();

    glBegin(GL_LINES);
    for (ssize_t i = 0; i < r.shape(0); ++i) {
        glVertex3d(r(i, 0), r(i, 1), r(i, 2));
        glVertex3d(r(i, 3), r(i, 4), r(i, 5));
    }
    glEnd();

    if(point_size > 0) {
        glPointSize(point_size);
        glBegin(GL_POINTS);
        for (ssize_t i = 0; i < r.shape(0); ++i) {
            glVertex3d(r(i, 0), r(i, 1), r(i, 2));
            glVertex3d(r(i, 3), r(i, 4), r(i, 5));
        }
        glEnd();
    }
}


void DrawLines(py::array_t<double> points, py::array_t<double> points2, float point_size=0) {
    auto r = points.unchecked<2>();
    auto r2 = points2.unchecked<2>();
    glBegin(GL_LINES);
    for (ssize_t i = 0; i < std::min(r.shape(0), r2.shape(0)); ++i) {
        glVertex3d(r(i, 0), r(i, 1), r(i, 2));
        glVertex3d(r2(i, 0), r2(i, 1), r2(i, 2));
    }
    glEnd();

    if(point_size > 0) {
        glPointSize(point_size);
        glBegin(GL_POINTS);
        for (ssize_t i = 0; i < std::min(r.shape(0), r2.shape(0)); ++i) {
            glVertex3d(r(i, 0), r(i, 1), r(i, 2));
            glVertex3d(r2(i, 0), r2(i, 1), r2(i, 2));
        }
        glEnd();
    }
}


void DrawBoxes(py::array_t<double> cameras, py::array_t<double> sizes) {
    auto r = cameras.unchecked<3>();
    auto rs = sizes.unchecked<2>();

    for (ssize_t i = 0; i < r.shape(0); ++i) {
        glPushMatrix();
        // glMultMatrixd(r.data(i, 0, 0));
        glMultTransposeMatrixd(r.data(i, 0, 0));

        float w = *rs.data(i, 0) / 2.0;  // w/2
        float h = *rs.data(i, 1) / 2.0;
        float z = *rs.data(i, 2) / 2.0;

        glBegin(GL_LINES);
        glVertex3f(-w, -h, -z);
        glVertex3f(w, -h, -z);
        glVertex3f(-w, -h, -z);
        glVertex3f(-w, h, -z);
        glVertex3f(-w, -h, -z);
        glVertex3f(-w, -h, z);

        glVertex3f(w, h, -z);
        glVertex3f(-w, h, -z);
        glVertex3f(w, h, -z);
        glVertex3f(w, -h, -z);
        glVertex3f(w, h, -z);
        glVertex3f(w, h, z);

        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);
        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(-w, h, z);
        glVertex3f(-w, h, -z);

        glVertex3f(w, -h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);
        glVertex3f(w, -h, -z);
        glEnd();

        glPopMatrix();
    }
}


}


void declareContrib(py::module & m) {

    m.def("DrawPoints", (void (*) (py::array_t<double>)) &DrawPoints, 
        "points"_a);
    m.def("DrawPoints", (void (*) (py::array_t<double>, py::array_t<double>)) &DrawPoints,
        "points"_a, "colors"_a);

    m.def("DrawMesh", (void (*) (py::array_t<double>, py::array_t<int>, py::array_t<double>, bool)) &DrawMesh,
        "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a=false);
    m.def("DrawMonochromeMesh", (void (*) (py::array_t<double>, py::array_t<int>, std::array<float, 3>, bool)) &DrawMonochromeMesh,
          py::arg("vertices"), py::arg("triangles"), py::arg("color"), py::arg("wireframe") = false);

    m.def("DrawLine", (void (*) (py::array_t<double>, float)) &DrawLine,
        "points"_a, "point_size"_a=0);
    m.def("DrawLines", (void (*) (py::array_t<double>, float)) &DrawLines,
        "points"_a, "point_size"_a=0);
    m.def("DrawLines", (void (*) (py::array_t<double>, py::array_t<double>, float)) &DrawLines,
        "points"_a, "points2"_a, "point_size"_a=0);

    m.def("DrawCameras", &DrawCameras,
        "poses"_a, "w"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6);
    m.def("DrawCamera", &DrawCamera,
    "poses"_a, "w"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6);

    m.def("DrawBoxes", &DrawBoxes,
        "poses"_a, "sizes"_a);

}

}