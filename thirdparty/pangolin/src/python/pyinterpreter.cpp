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
#include <Python.h>

#include <pangolin/python/pypangolin_init.h>
#include <pangolin/python/pyinterpreter.h>
#include <pangolin/python/pyuniqueobj.h>
#include <pangolin/python/pypangoio.h>
#include <pangolin/utils/file_utils.h>

namespace pangolin
{

void PyInterpreter::AttachPrefix(void* data, const std::string& name, VarValueGeneric& /*var*/, bool /*brand_new*/ )
{
    PyInterpreter* self = (PyInterpreter*)data;

    const size_t dot = name.find_first_of('.');
    if(dot != std::string::npos) {
        const std::string base_prefix = name.substr(0,dot);
        if( self->base_prefixes.find(base_prefix) == self->base_prefixes.end() ) {
            self->base_prefixes.insert(base_prefix);
            std::string cmd =
                base_prefix + std::string(" = pypangolin.Var('") +
                base_prefix + std::string("')\n");
            PyRun_SimpleString(cmd.c_str());
        }
    }
}

PyInterpreter::PyInterpreter()
    : pycompleter(0), pycomplete(0)
{
#if PY_MAJOR_VERSION >= 3
    PyImport_AppendInittab("pypangolin", InitPyPangolinModule);
    Py_Initialize();
#else
    Py_Initialize();
    InitPyPangolinModule();
#endif

    // Hook stdout, stderr to this interpreter
    PyObject* mod_sys = PyImport_ImportModule("sys");
    if (mod_sys) {
        PyModule_AddObject(mod_sys, "stdout", (PyObject*)new PyPangoIO(
            &PyPangoIO::Py_type, line_queue, ConsoleLineTypeStdout
            ));
        PyModule_AddObject(mod_sys, "stderr", (PyObject*)new PyPangoIO(
            &PyPangoIO::Py_type, line_queue, ConsoleLineTypeStderr
            ));
    } else {
        pango_print_error("Couldn't import module sys.\n");
    }

    // Attempt to setup readline completion
    PyRun_SimpleString(
        "import pypangolin\n"
        "try:\n"
        "   import readline\n"
        "except ImportError:\n"
        "   import pyreadline as readline\n"
        "\n"
        "import rlcompleter\n"
        "pypangolin.completer = rlcompleter.Completer()\n"
    );
    CheckPrintClearError();

    // Get reference to rlcompleter.Completer() for tab-completion
    PyObject* mod_pangolin = PyImport_ImportModule("pypangolin");
    if(mod_pangolin) {
        pycompleter = PyObject_GetAttrString(mod_pangolin,"completer");
        if(pycompleter) {
            pycomplete  = PyObject_GetAttrString(pycompleter,"complete");
        }
    } else {
        pango_print_error("PyInterpreter: Unable to load module pangolin.\n");
    }

    // Hook namespace prefixes into Python
    RegisterNewVarCallback(&PyInterpreter::AttachPrefix, (void*)this, "");
    ProcessHistoricCallbacks(&PyInterpreter::AttachPrefix, (void*)this, "");

    CheckPrintClearError();
}

PyInterpreter::~PyInterpreter()
{
    Py_Finalize();
}

std::string PyInterpreter::ToString(PyObject* py)
{
    PyUniqueObj pystr = PyObject_Repr(py);
#if PY_MAJOR_VERSION >= 3
    return std::string(PyUnicode_AsUTF8(pystr));
#else
    return std::string(PyString_AsString(pystr));
#endif
}

void PyInterpreter::CheckPrintClearError()
{
    if(PyErr_Occurred()) {
        PyErr_Print();
        PyErr_Clear();
    }
}

PyUniqueObj PyInterpreter::EvalExec(const std::string& cmd)
{
    PyObject* globals = PyModule_GetDict(PyImport_AddModule("__main__"));
#if PY_MAJOR_VERSION >= 3
    PyObject* builtin = PyImport_AddModule("builtins");
#else
    PyObject* builtin = PyImport_AddModule("__builtin__");
#endif

    if(globals && builtin) {
        PyUniqueObj compile = PyObject_GetAttrString(builtin, "compile");
        PyUniqueObj eval = PyObject_GetAttrString(builtin, "eval");

        if(compile && eval)
        {
            PyErr_Clear();
            PyUniqueObj compile_eval_args = Py_BuildValue("(sss)", cmd.c_str(), "<string>", "eval" );
            if(compile_eval_args)
            {
                PyUniqueObj code = PyObject_Call(compile, compile_eval_args, 0);
                if(code) {
                    PyUniqueObj eval_args = Py_BuildValue("(OOO)", *code, globals, globals );
                    if(eval_args) {
                        PyUniqueObj ret = PyObject_Call(eval, eval_args, 0);
                        CheckPrintClearError();
                        return ret;
                    }
                }
            }

            PyErr_Clear();
            PyUniqueObj compile_exec_args = Py_BuildValue("(sss)", cmd.c_str(), "<string>", "exec" );
            if(compile_exec_args)
            {
                PyUniqueObj code = PyObject_Call(compile, compile_exec_args, 0);
                if(code) {
                    PyUniqueObj eval_args = Py_BuildValue("(OOO)", *code, globals, globals );
                    if(eval_args) {
                        PyUniqueObj ret = PyObject_Call(eval, eval_args, 0);
                        CheckPrintClearError();
                        return ret;
                    }
                }
            }
        }
    }

    CheckPrintClearError();
    return PyUniqueObj();
}

std::vector<std::string> PyInterpreter::Complete(const std::string& cmd, int max_options)
{
    std::vector<std::string> ret;
    PyErr_Clear();

    if(pycomplete) {
        for(int i=0; i < max_options; ++i) {
#if PY_MAJOR_VERSION >= 3
            PyUniqueObj args = PyTuple_Pack( 2, PyUnicode_FromString(cmd.c_str()), PyLong_FromSize_t(i) );
            PyUniqueObj result = PyObject_CallObject(pycomplete, args);
            if (result && PyUnicode_Check(result)) {
                std::string res_str(PyUnicode_AsUTF8(result));
#else
            PyUniqueObj args = PyTuple_Pack(2, PyString_FromString(cmd.c_str()), PyInt_FromSize_t(i));
            PyUniqueObj result = PyObject_CallObject(pycomplete, args);
            if (result && PyString_Check(result)) {
                std::string res_str(PyString_AsString(result));
#endif
                if( res_str.find("__")==std::string::npos ||
                    cmd.find("__")!=std::string::npos ||
                    (cmd.size() > 0 && cmd[cmd.size()-1] == '_')
                ) {
                    ret.push_back( res_str );
                }
            }else{
                break;
            }
        }
    }

    return ret;
}

void PyInterpreter::PushCommand(const std::string& cmd)
{
    PyUniqueObj obj = EvalExec(cmd);
    if(obj && obj != Py_None) {
        const std::string output = ToString(obj);
        line_queue.push(
            ConsoleLine(output, ConsoleLineTypeOutput)
        );
    }
}

bool PyInterpreter::PullLine(ConsoleLine& line)
{
    if(line_queue.size()) {
        line = line_queue.front();
        line_queue.pop();
        return true;
    }else{
        return false;
    }
}

}
