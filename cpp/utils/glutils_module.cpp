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
#include <pybind11/numpy.h>

#include <GL/gl.h>


namespace py = pybind11;
using namespace pybind11::literals;


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


class CameraImage {
public:
    // from numpy image 
    template <typename T>
    CameraImage(py::array_t<unsigned char> image, py::array_t<T> pose, size_t id, float w=1.0, float h_ratio=0.75, float z_ratio=0.6, std::array<float, 3> color={0.0, 1.0, 0.0})
        : imageWidth(image.shape(1)), imageHeight(image.shape(0)), id(id), w(w), h_ratio(h_ratio), z_ratio(z_ratio), color(color) {
        // Check if image is color or grayscale
        bool is_color = true;
        if (image.ndim() == 3) {
            if (image.shape(2) != 3) {
                throw std::invalid_argument("Image must have 3 channels");
            }
        } else if (image.ndim() == 2) {
            is_color = false;
        } else {
            throw std::invalid_argument("Image must have 2 or 3 dimensions");
        }
        // Generate texture
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        if (is_color) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
        } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, imageWidth, imageHeight, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, image.data());
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

        setPose(pose);
    }

    ~CameraImage() {
        glDeleteTextures(1, &texture);
    }

    void draw_() const {

        const float h = w * h_ratio;
        const float z = w * z_ratio;

        glColor3f(color[0], color[1], color[2]);

        // Draw the rectangle representing the image with texture
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

        // Reset color to white before drawing the texture
        glColor3f(1.0f, 1.0f, 1.0f);
 
        // Draw the rectangle representing the image with texture
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);

        // Check if face culling is enabled
        GLboolean isCullFaceEnabled = glIsEnabled(GL_CULL_FACE);
        if (isCullFaceEnabled) {
            glDisable(GL_CULL_FACE);
        }

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 0.0f); glVertex3f(-w, -h, z);
        glTexCoord2f(1.0f, 0.0f); glVertex3f(w, -h, z);
        glTexCoord2f(1.0f, 1.0f); glVertex3f(w, h, z);
        glTexCoord2f(0.0f, 1.0f); glVertex3f(-w, h, z);
        glEnd();

        if (isCullFaceEnabled) {
            glEnable(GL_CULL_FACE);
        }        

        glDisable(GL_TEXTURE_2D);     
    }

    void draw() const {

        glPushMatrix();
        glMultTransposeMatrixd(matrix_);

        draw_();

        glPopMatrix();        
    }

    void drawPose(py::array_t<double> pose) const {
        auto r = pose.unchecked<2>();

        glPushMatrix();
        glMultTransposeMatrixd(r.data(0, 0));

        draw_();

        glPopMatrix();
    }

    void drawMatrix(const double* poseMatrix) const {

        glPushMatrix();
        glMultTransposeMatrixd(poseMatrix);

        draw_();

        glPopMatrix();
    }

    template <typename T>
    void setPose(py::array_t<T> pose) {
        if (pose.ndim() != 2 || pose.size() != 16) {
            throw std::runtime_error("Pose must be a 4x4 array.");
        }   
        auto r = pose.template unchecked<2>();
        const auto* data = r.data(0, 0);
        for (int i = 0; i < 16; ++i) {
            matrix_[i] = static_cast<double>(data[i]);
        }
    }

    void setColor(std::array<float, 3> color) {
        this->color = color;
    }

private:
    GLuint texture;
    int imageWidth;
    int imageHeight;
    float w;
    float h_ratio;
    float z_ratio;

    std::array<float, 3> color={0.0, 1.0, 0.0};

    GLdouble matrix_[16] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };    
public:
    size_t id=0;
};

// // Explicitly instantiate the setPose function for float and double
// template CameraImage::CameraImage(py::array_t<unsigned char> image, py::array_t<float> pose, size_t id, float w, float h_ratio, float z_ratio, std::array<float, 3> color);
// template CameraImage::CameraImage(py::array_t<unsigned char> image, py::array_t<double> pose, size_t id, float w, float h_ratio, float z_ratio, std::array<float, 3> color);
// template void CameraImage::setPose<float>(py::array_t<float> pose);
// template void CameraImage::setPose<double>(py::array_t<double> pose);


class CameraImages {
public:
    CameraImages() = default;

    // Add a camera image from numpy array
    template <typename T>
    void add(py::array_t<unsigned char> image, py::array_t<T> poseMatrix, size_t id, float w=1.0, float h_ratio=0.75, float z_ratio=0.6, std::array<float, 3> color={0.0, 1.0, 0.0}) {
        //cams.emplace_back(image, poseMatrix, id, w, h_ratio, z_ratio, color);
        if (poseMatrix.dtype() == py::dtype::of<float>()) {
            // Ensure float32
            cams.emplace_back(image, poseMatrix.template cast<py::array_t<float>>(), id, w, h_ratio, z_ratio, color);
            
        } else {
            // Ensure float64
            cams.emplace_back(image, poseMatrix.template cast<py::array_t<double>>(), id, w, h_ratio, z_ratio, color);
        }
    }

    void drawPoses(py::array_t<double> cameras) const {
        auto r = cameras.unchecked<3>();

        for (ssize_t i = 0; i < r.shape(0); ++i) {
            cams[i].drawMatrix(r.data(i, 0, 0));
        }
    }

    void draw() const {
        for (const auto& cam : cams) {
            cam.draw();
        }
    }

    CameraImage& operator[](size_t i) {
        return cams[i];
    }

    void clear() {
        cams.clear();
    }

    size_t size() const {
        return cams.size();
    }

    void erase(size_t id) {
        auto it = std::find_if(cams.begin(), cams.end(), [id](const CameraImage& cam) { return cam.id == id; });
        if (it != cams.end()) {
            cams.erase(it);
        }
    }

private: 
    std::vector<CameraImage> cams;
};


PYBIND11_MODULE(glutils, m) 
{
    // optional module docstring
    m.doc() = "pybind11 plugin for glutils module";


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

    py::class_<CameraImage>(m, "CameraImage")
        .def(py::init([](py::array_t<unsigned char> image, py::array_t<double> pose, size_t id, float scale, float h_ratio, float z_ratio, std::array<float, 3> color) {
            return new CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
        }), "image"_a, "pose"_a, "id"_a, "scale"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6, "color"_a=std::array<float, 3>{0.0, 1.0, 0.0})
        .def(py::init([](py::array_t<unsigned char> image, py::array_t<float> pose, size_t id, float scale, float h_ratio, float z_ratio, std::array<float, 3> color) {
            return new CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
        }))
        .def("draw", &CameraImage::draw)
        .def("drawPose", &CameraImage::drawPose)
        .def("setPose", [](CameraImage& self, py::array_t<float> pose) {
            self.setPose<float>(pose);
        })
        .def("setPose", [](CameraImage& self, py::array_t<double> pose) {
            self.setPose<double>(pose);
        });

    py::class_<CameraImages>(m, "CameraImages")
        .def(py::init<>())
        //.def("add", &CameraImages::add, "image"_a, "pose"_a, "id"_a, "scale"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6, "color"_a=std::array<float, 3>{0.0, 1.0, 0.0})
        .def("add", [](CameraImages& self, py::array_t<unsigned char> image, py::array_t<float> pose, size_t id, float scale, float h_ratio, float z_ratio, std::array<float, 3> color) {
            self.add(image, pose, id, scale, h_ratio, z_ratio, color);
        }, "image"_a, "pose"_a, "id"_a, "scale"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6, "color"_a=std::array<float, 3>{0.0, 1.0, 0.0})
        .def("add", [](CameraImages& self, py::array_t<unsigned char> image, py::array_t<double> pose, size_t id, float scale, float h_ratio, float z_ratio, std::array<float, 3> color) {
            self.add(image, pose, id, scale, h_ratio, z_ratio, color);
        }, "image"_a, "pose"_a, "id"_a, "scale"_a=1.0, "h_ratio"_a=0.75, "z_ratio"_a=0.6, "color"_a=std::array<float, 3>{0.0, 1.0, 0.0})
        .def("drawPoses", &CameraImages::drawPoses)
        .def("draw", &CameraImages::draw)
        .def("clear", &CameraImages::clear)
        .def("erase", &CameraImages::erase)
        .def("size", &CameraImages::size)
        .def("__getitem__", &CameraImages::operator[], py::return_value_policy::reference)
        .def("__len__", &CameraImages::size);
}
