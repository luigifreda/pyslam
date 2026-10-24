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

#pragma once

#include "glutils_gl_includes.h"

#include "glutils_drawing.h"
#include "glutils_utils.h"
#include <memory>
#include <vector>

namespace glutils {

// Camera image class for texture-based camera visualization
class CameraImage {
  public:
    using Ptr = std::shared_ptr<CameraImage>;

    template <typename T, int Flags>
    CameraImage(const UByteArray &image, const py::array_t<T, Flags> &pose, const size_t id,
                const float w = 1.0f, const float h_ratio = 0.75f, const float z_ratio = 0.6f,
                const std::array<float, 3> &color = {0.0f, 1.0f, 0.0f})
        : texture(0), imageWidth(0), imageHeight(0), w(w), h_ratio(h_ratio), z_ratio(z_ratio),
          color(color), id(id) {
        auto image_info = image.request();
        bool is_color = true;
        if (image_info.ndim == 3) {
            if (image_info.shape[2] != 3) {
                throw std::invalid_argument("Image must have 3 channels");
            }
        } else if (image_info.ndim == 2) {
            is_color = false;
        } else {
            throw std::invalid_argument("Image must have 2 or 3 dimensions");
        }

        imageWidth = static_cast<int>(image_info.shape[1]);
        imageHeight = static_cast<int>(image_info.shape[0]);

        const auto unpack_alignment = glutils::ComputeAlignment(image_info);
        const auto *image_data = static_cast<const unsigned char *>(image_info.ptr);

        const auto pose_matrix = glutils::ExtractPoseMatrix(pose);
        for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
            matrix_[i] = static_cast<GLdouble>(pose_matrix[i]);
        }

        py::gil_scoped_release release;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, static_cast<GLint>(unpack_alignment));

        const GLenum format = is_color ? GL_RGB : GL_LUMINANCE;
        glTexImage2D(GL_TEXTURE_2D, 0, format, imageWidth, imageHeight, 0, format, GL_UNSIGNED_BYTE,
                     image_data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

    ~CameraImage() {
        std::cout << "CameraImage " << this->id << " deleted" << std::endl;
        if (texture != 0) {
            py::gil_scoped_release release;
            glDeleteTextures(1, &texture);
        }
    }

    void draw() const { drawMatrix(matrix_.data()); }

    void drawPose(DoubleArray pose) const {
        const auto pose_matrix = glutils::ExtractPoseMatrix(pose);
        drawMatrix(pose_matrix.data());
    }

    void drawMatrix(const GLdouble *poseMatrix) const {
        py::gil_scoped_release release;
        drawMatrixNoGIL(poseMatrix);
    }

    template <typename T, int Flags> void setPose(const py::array_t<T, Flags> &pose) {
        const auto pose_matrix = glutils::ExtractPoseMatrix(pose);
        for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
            matrix_[i] = static_cast<GLdouble>(pose_matrix[i]);
        }
    }

    void setColor(const std::array<float, 3> &new_color) { this->color = new_color; }

    void setTransparent(bool transparent) { this->isTransparent = transparent; }

  private:
    void drawMatrixNoGIL(const GLdouble *poseMatrix) const {
        glPushMatrix();
        glMultTransposeMatrixd(poseMatrix);
        draw_();
        glPopMatrix();
    }

    void draw_() const {
        const float h = w * h_ratio;
        const float z = w * z_ratio;

        glColor3f(color[0], color[1], color[2]);
        glutils_detail::DrawCameraFrustum(w, h, z);

        if (!isTransparent) {
            glColor3f(1.0f, 1.0f, 1.0f);

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, texture);

            const GLboolean isCullFaceEnabled = glIsEnabled(GL_CULL_FACE);
            if (isCullFaceEnabled) {
                glDisable(GL_CULL_FACE);
            }

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(-w, -h, z);
            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(w, -h, z);
            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(w, h, z);
            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(-w, h, z);
            glEnd();

            if (isCullFaceEnabled) {
                glEnable(GL_CULL_FACE);
            }

            glDisable(GL_TEXTURE_2D);
        }
    }

  private:
    GLuint texture;
    int imageWidth;
    int imageHeight;
    float w;
    float h_ratio;
    float z_ratio;
    bool isTransparent = false;
    std::array<float, 3> color = {0.0f, 1.0f, 0.0f};
    std::array<GLdouble, kMatrixElementCount> matrix_{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  public:
    size_t id = 0;
};

// Container for managing multiple CameraImage objects
class CameraImages {
  public:
    CameraImages() = default;

    // Add a camera image from numpy array
    template <typename T, int Flags>
    void add(const UByteArray &image, const py::array_t<T, Flags> &poseMatrix, const size_t id,
             const float w = 1.0f, const float h_ratio = 0.75f, const float z_ratio = 0.6f,
             const std::array<float, 3> &color = {0.0f, 1.0f, 0.0f}) {
        auto dtype = poseMatrix.dtype();
        const std::string dtype_str = py::str(dtype);
        if (dtype.is(py::dtype::of<float>()) || dtype_str == "float32") {
            const auto cam =
                std::make_shared<CameraImage>(image, poseMatrix, id, w, h_ratio, z_ratio, color);
            cams.push_back(cam);
        } else if (dtype.is(py::dtype::of<double>()) || dtype_str == "float64") {
            const auto cam =
                std::make_shared<CameraImage>(image, poseMatrix, id, w, h_ratio, z_ratio, color);
            cams.push_back(cam);
        } else {
            std::cout << "unmanaged dtype: " << dtype << std::endl;
            throw std::runtime_error("Pose matrix must be float32 or float64.");
        }
    }

    void drawPoses(DoubleArray cameras) const {
        auto info = cameras.request();
        if (info.ndim != 3 || info.shape[1] != 4 || info.shape[2] != 4) {
            throw std::runtime_error("poses must be an Nx4x4 array");
        }

        const auto *pose_data = static_cast<const double *>(info.ptr);
        const std::size_t matrix_count = static_cast<std::size_t>(info.shape[0]);
        if (matrix_count != cams.size()) {
            throw std::runtime_error("poses length must match stored camera images");
        }

        for (std::size_t i = 0; i < matrix_count; ++i) {
            cams[i]->drawMatrix(pose_data + i * kMatrixElementCount);
        }
    }

    void draw() const {
        for (const auto &cam : cams) {
            cam->draw();
        }
    }

    CameraImage::Ptr &operator[](size_t i) { return cams[i]; }

    void clear() { cams.clear(); }

    size_t size() const { return cams.size(); }

    void erase(size_t id) {
        auto it = std::find_if(cams.begin(), cams.end(),
                               [id](const CameraImage::Ptr &cam) { return cam->id == id; });
        if (it != cams.end()) {
            cams.erase(it);
        }
    }

    void setTransparent(size_t id, bool isTransparent) {
        auto it = std::find_if(cams.begin(), cams.end(),
                               [id](const CameraImage::Ptr &cam) { return cam->id == id; });
        if (it != cams.end()) {
            (*it)->setTransparent(isTransparent);
        }
    }

    void setAllTransparent(bool isTransparent) {
        for (auto &cam : cams) {
            cam->setTransparent(isTransparent);
        }
    }

  private:
    std::vector<CameraImage::Ptr> cams;
};

} // namespace glutils
