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

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyslam {

static py::object json_to_py(const nlohmann::json &j);

static py::dict json_to_dict(const nlohmann::json &j) {
    if (!j.is_object())
        throw std::runtime_error("Expected a JSON object to convert to dict");
    py::dict d;
    for (const auto &[k, v] : j.items()) {
        d[py::str(k)] = json_to_py(v);
    }
    return d;
}

static py::object json_to_py(const nlohmann::json &j) {
    using nlohmann::json;
    switch (j.type()) {
    case json::value_t::null:
        return py::none();
    case json::value_t::boolean:
        return py::bool_(j.get<bool>());
    case json::value_t::number_integer:
        return py::int_(j.get<long long>());
    case json::value_t::number_unsigned:
        return py::int_(j.get<unsigned long long>());
    case json::value_t::number_float:
        return py::float_(j.get<double>());
    case json::value_t::string:
        return py::str(j.get<std::string>());
    case json::value_t::array: {
        py::list lst;
        for (const auto &e : j)
            lst.append(json_to_py(e));
        return lst;
    }
    case json::value_t::object:
        return json_to_dict(j);
    default:
        // binary, discarded, etc. — to handle if we use them in the future
        throw std::runtime_error("Unsupported JSON value type");
    }
}

} // namespace pyslam

/* Example usage in your bindings:
nlohmann::json make_json(); // your function producing JSON

PYBIND11_MODULE(example, m) {
    m.def("get_config_dict", []() -> py::dict {
        nlohmann::json j = make_json();
        return json_to_dict(j); // guarantees a Python dict
    });
}
*/

namespace pybind11 {
namespace detail {

template <> struct type_caster<nlohmann::json> {
  public:
    PYBIND11_TYPE_CASTER(nlohmann::json, _("json"));

#if 0
    // Python -> C++ (optional;  if we also want to accept Python objects)
    bool load(handle src, bool) {
        try {
            // TODO: If needed, implement a py_to_json recursively too
            return false; // not implemented yet
        } catch (...) {
            return false;
        }
    }
#endif

    // C++ -> Python (automatic)
    static handle cast(const nlohmann::json &src, return_value_policy, handle) {
        return pyslam::json_to_py(src).release();
    }
};

} // namespace detail
} // namespace pybind11