/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) Andrey Mnatsakanov
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

#pragma once

#include <pybind11/pybind11.h>
#include <pangolin/image/image.h>

namespace py_pangolin {

  template<typename T>
  void bind_image(pybind11::module& m, const std::string& pyname){
    pybind11::class_<pangolin::Image<T> >(m, pyname.c_str())
      .def(pybind11::init<>())
      .def(pybind11::init<T*, size_t, size_t, size_t>())
      .def("SizeBytes", &pangolin::Image<T>::SizeBytes)
      .def("Area", &pangolin::Image<T>::Area)
      .def("IsValid", &pangolin::Image<T>::IsValid)
      .def("IsContiguous", &pangolin::Image<T>::IsContiguous)
      .def("begin", (unsigned char* (pangolin::Image<T>::*)())&pangolin::Image<T>::begin)
      .def("end", (unsigned char* (pangolin::Image<T>::*)())&pangolin::Image<T>::end)
      .def("begin", (const unsigned char* (pangolin::Image<T>::*)() const )&pangolin::Image<T>::begin)
      .def("end", (const unsigned char* (pangolin::Image<T>::*)() const )&pangolin::Image<T>::end)
      .def("size", &pangolin::Image<T>::size)
      .def("Fill", &pangolin::Image<T>::Fill)
      .def("Replace", &pangolin::Image<T>::Replace)
      .def("Memset", &pangolin::Image<T>::Memset)
      .def("CopyFrom", &pangolin::Image<T>::CopyFrom)
      .def("MinMax", &pangolin::Image<T>::MinMax)
      // .def("Sum", [](pangolin::Image<T>& v){ return v.Sum();})
      // .def("Mean", [](pangolin::Image<T>& v){ return v.Mean();})
      .def("RowPtr", (T* (pangolin::Image<T>::*)(size_t))&pangolin::Image<T>::RowPtr)
      .def("RowPtr", (const T* (pangolin::Image<T>::*)(size_t) const )&pangolin::Image<T>::RowPtr)
      .def("InImage", &pangolin::Image<T>::InImage)
      .def("InBounds", (bool (pangolin::Image<T>::*)(int, int) const)&pangolin::Image<T>::InBounds)
      .def("InBounds", (bool (pangolin::Image<T>::*)(float, float, float) const)&pangolin::Image<T>::InBounds)
      // .def("SubImage", (pangolin::Image<T> (pangolin::Image<T>::*)(size_t, size_t, size_t, size_t))&pangolin::Image<T>::SubImage)
      // .def("Row", &pangolin::Image<T>::Row)
      // .def("Col", &pangolin::Image<T>::Col)
      .def("InImage", &pangolin::Image<T>::InImage);
  }

}  // py_pangolin
