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
            // Will throw if out of int64 range; to adjust if we need big ints in the future
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
                if (!sub.load(item, true)) {
                    // If conversion fails, try string representation
                    try {
                        std::string val_str = py::str(item);
                        out.emplace_back(pyslam::Value(val_str));
                    } catch (const py::error_already_set &) {
                        // Skip problematic items
                        continue;
                    }
                } else {
                    out.emplace_back(std::move(sub.value));
                }
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
                if (!sub.load(kv.second, true)) {
                    // If recursive conversion fails, try to convert to string representation
                    // This handles complex nested structures that can't be directly converted
                    try {
                        std::string val_str = py::str(kv.second);
                        out.emplace(std::move(key), pyslam::Value(val_str));
                    } catch (const py::error_already_set &) {
                        // If even string conversion fails, skip this key-value pair
                        continue;
                    }
                } else {
                    out.emplace(std::move(key), std::move(sub.value));
                }
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