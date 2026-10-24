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

#include <memory>

namespace pyslam {
namespace camera_model {

// Camera reference held by optimizer edges and factors. Non-owning when built from a
// raw pointer (the caller keeps the camera alive), owning when built from a shared_ptr.
template <typename CameraT> class CameraHandle {
  public:
    explicit CameraHandle(const CameraT *camera) : camera_(camera) {}
    explicit CameraHandle(std::shared_ptr<const CameraT> camera)
        : camera_(camera.get()), owner_(std::move(camera)) {}

    const CameraT *operator->() const { return camera_; }
    const CameraT &operator*() const { return *camera_; }
    const CameraT *get() const { return camera_; }

  private:
    const CameraT *camera_ = nullptr;
    std::shared_ptr<const CameraT> owner_;
};

} // namespace camera_model
} // namespace pyslam
