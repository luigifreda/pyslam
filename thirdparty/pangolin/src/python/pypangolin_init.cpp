/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
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

#include <pangolin/python/pypangolin_init.h>
#include <pangolin/python/pyvar.h>
#include "pypangolin/pypangolin.h"

namespace pangolin
{

PyMODINIT_FUNC
InitPyPangolinModule()
{
    auto pypango = pybind11::module("pypangolin");
    pypangolin::PopulateModule(pypango);

    if(pypango.ptr()) {
        if (PyType_Ready(&PyVar::Py_type) >= 0) {
            Py_INCREF(&PyVar::Py_type);
            PyModule_AddObject(pypango.ptr(), "Var", (PyObject *)&PyVar::Py_type);
        }else{
            pango_print_error("Unable to create pangolin Python objects.\n");
        }
    }else{
        pango_print_error("Unable to initialise pangolin Python module.\n");
    }

#if PY_MAJOR_VERSION >= 3
    return pypango.ptr();
#endif
}

}
