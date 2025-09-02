#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dictionary.h"

namespace py = pybind11;

// Tell pybind11 we will bind these containers explicitly
PYBIND11_MAKE_OPAQUE(pyslam::List);
PYBIND11_MAKE_OPAQUE(pyslam::Dict);

namespace pybind11::detail {

// type_caster for Value: Python -> C++ and C++ -> Python
template <> struct type_caster<pyslam::Value> {
  public:
    PYBIND11_TYPE_CASTER(pyslam::Value, _("Value"));

    // Python -> C++
    bool load(handle src, bool) {
        if (!src)
            return false;

        if (src.is_none()) {
            value = pyslam::Value{}; // std::monostate
            return true;
        }

        // Order matters: bool before int
        if (py::isinstance<py::bool_>(src)) {
            value = pyslam::Value(py::cast<bool>(src));
            return true;
        }

        if (py::isinstance<py::int_>(src)) {
            // Will throw if out of int64 range; adjust if you need big ints
            value = pyslam::Value(py::cast<int64_t>(src));
            return true;
        }

        if (py::isinstance<py::float_>(src)) {
            value = pyslam::Value(py::cast<double>(src));
            return true;
        }

        if (py::isinstance<py::str>(src)) {
            value = pyslam::Value(py::cast<std::string>(src));
            return true;
        }

        if (py::isinstance<py::list>(src) || py::isinstance<py::tuple>(src)) {
            pyslam::List out;
            py::sequence seq = py::reinterpret_borrow<py::sequence>(src);
            out.reserve(py::len(seq));
            for (py::handle item : seq) {
                type_caster<pyslam::Value> sub;
                if (!sub.load(item, true))
                    return false;
                out.emplace_back(std::move(sub.value));
            }
            value = pyslam::Value(std::move(out));
            return true;
        }

        if (py::isinstance<py::dict>(src)) {
            pyslam::Dict out;
            py::dict d = py::reinterpret_borrow<py::dict>(src);
            for (auto kv : d) {
                // Require string-like keys; change to accept more if desired
                std::string key = py::cast<std::string>(kv.first);
                type_caster<pyslam::Value> sub;
                if (!sub.load(kv.second, true))
                    return false;
                out.emplace(std::move(key), std::move(sub.value));
            }
            value = pyslam::Value(std::move(out));
            return true;
        }

        return false; // unsupported Python type
    }

    // C++ -> Python
    static handle cast(const pyslam::Value &src, return_value_policy, handle) {
        const auto &var = src.data;

        if (std::holds_alternative<std::monostate>(var))
            return py::none().inc_ref();

        if (auto p = std::get_if<bool>(&var))
            return py::bool_(*p).release();

        if (auto p = std::get_if<int>(&var))
            return py::int_(*p).release();

        if (auto p = std::get_if<int64_t>(&var))
            return py::int_(*p).release();

        if (auto p = std::get_if<double>(&var))
            return py::float_(*p).release();

        if (auto p = std::get_if<std::string>(&var))
            return py::str(*p).release();

        if (auto p = std::get_if<std::vector<double>>(&var)) {
            py::list out;
            for (const auto &e : *p)
                out.append(py::cast(e));
            return out.release();
        }

        // Use the Value class's get_if method for recursive types
        if (auto p = src.get_if<pyslam::List>()) {
            py::list out;
            for (const auto &e : *p)
                out.append(py::cast(e));
            return out.release();
        }

        if (auto p = src.get_if<pyslam::Dict>()) {
            py::dict out;
            for (const auto &[k, v] : *p)
                out[py::str(k)] = py::cast(v);
            return out.release();
        }

        return py::none().inc_ref();
    }
};

} // namespace pybind11::detail