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

#include <any>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

namespace pyslam {

// Forward declarations
struct Value;

using ValueVariant = std::variant<std::monostate,                                 // null / unset
                                  bool, int, int64_t, float, double, std::string, // scalars
                                  std::vector<bool>, std::vector<int>, std::vector<int64_t>,
                                  std::vector<double>, // vectors
                                  std::any             // For recursive types (List and Dict)
                                  >;

constexpr size_t VariantSize = std::variant_size_v<ValueVariant> - 1;

// Define type aliases
using List = std::vector<Value>;
using Dict = std::unordered_map<std::string, Value>;

// The Value class is a variant that can store any type of data.
struct Value {
    ValueVariant data;

    // Convenience constructors
    Value() : data(std::monostate{}) {}
    Value(bool v) : data(v) {}
    Value(int v) : data(v) {} // Use int directly now
    Value(int64_t v) : data(v) {}
    Value(float v) : data(v) {}
    Value(double v) : data(v) {}
    Value(const char *s) : data(std::string(s)) {}
    Value(std::string s) : data(std::move(s)) {}
    Value(std::vector<bool> v) : data(std::move(v)) {}
    Value(std::vector<int> v) : data(std::move(v)) {}
    Value(std::vector<int64_t> v) : data(std::move(v)) {}
    Value(std::vector<double> v) : data(std::move(v)) {}
    Value(List v) : data(std::any(std::move(v))) {}
    Value(Dict v) : data(std::any(std::move(v))) {}

    // Constructor for nlohmann::json to resolve ambiguity
    Value(const nlohmann::json &json_val) {
        if (json_val.is_null()) {
            data = std::monostate{};
        } else if (json_val.is_boolean()) {
            data = json_val.get<bool>();
        } else if (json_val.is_number_integer()) {
            data = json_val.get<int64_t>();
        } else if (json_val.is_number_float()) {
            data = json_val.get<double>();
        } else if (json_val.is_string()) {
            data = json_val.get<std::string>();
        } else if (json_val.is_array() && !json_val.empty() && json_val.front().is_boolean()) {
            data = json_val.get<std::vector<bool>>();
        } else if (json_val.is_array() && !json_val.empty() &&
                   json_val.front().is_number_integer()) {
            data = json_val.get<std::vector<int>>();
        } else if (json_val.is_array() && !json_val.empty() && json_val.front().is_number_float()) {
            data = json_val.get<std::vector<double>>();
        } else if (json_val.is_array()) {
            List list;
            for (const auto &item : json_val) {
                list.emplace_back(Value(item));
            }
            data = std::any(std::move(list));
        } else if (json_val.is_object()) {
            Dict dict;
            for (auto it = json_val.begin(); it != json_val.end(); ++it) {
                dict[it.key()] = Value(it.value());
            }
            data = std::any(std::move(dict));
        } else {
            data = std::monostate{}; // fallback for unknown types
        }
    }

    Value(const Value &other) : data(other.data) {}
    Value &operator=(const Value &other) {
        data = other.data;
        return *this;
    }
    Value(Value &&other) noexcept : data(std::move(other.data)) {}
    Value &operator=(Value &&other) noexcept {
        data = std::move(other.data);
        return *this;
    }

    // Type checks
    template <typename T> bool is() const {
        if constexpr (std::is_same_v<T, List>) {
            return data.index() == VariantSize &&
                   std::any_cast<List>(&std::get<std::any>(data)) != nullptr;
        } else if constexpr (std::is_same_v<T, Dict>) {
            return data.index() == VariantSize &&
                   std::any_cast<Dict>(&std::get<std::any>(data)) != nullptr;
        } else {
            return std::holds_alternative<T>(data);
        }
    }

    // Safe access; returns nullptr if wrong type
    template <typename T> const T *get_if() const {
        if constexpr (std::is_same_v<T, List>) {
            if (data.index() == VariantSize) {
                return std::any_cast<List>(&std::get<std::any>(data));
            }
            return nullptr;
        } else if constexpr (std::is_same_v<T, Dict>) {
            if (data.index() == VariantSize) {
                return std::any_cast<Dict>(&std::get<std::any>(data));
            }
            return nullptr;
        } else {
            return std::get_if<T>(&data);
        }
    }

    template <typename T> T *get_if() {
        if constexpr (std::is_same_v<T, List>) {
            if (data.index() == VariantSize) {
                return std::any_cast<List>(&std::get<std::any>(data));
            }
            return nullptr;
        } else if constexpr (std::is_same_v<T, Dict>) {
            if (data.index() == VariantSize) {
                return std::any_cast<Dict>(&std::get<std::any>(data));
            }
            return nullptr;
        } else {
            return std::get_if<T>(&data);
        }
    }

    // Throws std::bad_variant_access if wrong type
    template <typename T> const T &get() const {
        if constexpr (std::is_same_v<T, List>) {
            return std::any_cast<const List &>(std::get<std::any>(data));
        } else if constexpr (std::is_same_v<T, Dict>) {
            return std::any_cast<const Dict &>(std::get<std::any>(data));
        } else {
            return std::get<T>(data);
        }
    }

    template <typename T> T &get() {
        if constexpr (std::is_same_v<T, List>) {
            return std::any_cast<List &>(std::get<std::any>(data));
        } else if constexpr (std::is_same_v<T, Dict>) {
            return std::any_cast<Dict &>(std::get<std::any>(data));
        } else {
            return std::get<T>(data);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Value &v);

    std::string to_string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }
};

// Helper function for recursive printing of the Value class
inline void print_value_recursive(std::ostream &os, const Value &val, int indent) {
    auto pad = [&](int n) {
        for (int i = 0; i < n; i++)
            os << ' ';
    };

    std::visit(
        [&](auto const &val_data) {
            using T = std::decay_t<decltype(val_data)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                os << "null";
            } else if constexpr (std::is_same_v<T, bool>) {
                os << (val_data ? "true" : "false");
            } else if constexpr (std::is_same_v<T, int>) {
                os << val_data;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                os << val_data;
            } else if constexpr (std::is_same_v<T, float>) {
                os << val_data;
            } else if constexpr (std::is_same_v<T, double>) {
                os << val_data;
            } else if constexpr (std::is_same_v<T, std::string>) {
                os << '"' << val_data << '"';
            } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
                os << "[";
                for (size_t i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, std::vector<int>>) {
                os << "[";
                for (size_t i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                os << "[";
                for (size_t i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                os << "[";
                for (size_t i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, std::any>) {
                // Handle recursive types
                try {
                    if (auto list_ptr = std::any_cast<List>(&val_data)) {
                        os << "[\n";
                        for (size_t i = 0; i < list_ptr->size(); ++i) {
                            pad(indent + 2);
                            print_value_recursive(os, (*list_ptr)[i], indent + 2);
                            os << (i + 1 < list_ptr->size() ? ",\n" : "\n");
                        }
                        pad(indent);
                        os << "]";
                    } else if (auto dict_ptr = std::any_cast<Dict>(&val_data)) {
                        os << "{\n";
                        size_t i = 0;
                        for (auto const &[k, subv] : *dict_ptr) {
                            pad(indent + 2);
                            os << '"' << k << "\": ";
                            print_value_recursive(os, subv, indent + 2);
                            os << (++i < dict_ptr->size() ? ",\n" : "\n");
                        }
                        pad(indent);
                        os << "}";
                    }
                } catch (const std::bad_any_cast &) {
                    os << "unknown";
                }
            }
        },
        val.data);
}

inline std::ostream &operator<<(std::ostream &os, const Value &v) {
    print_value_recursive(os, v, 0);
    return os;
}

// The ConfigDict class is a dictionary that can be used to store configuration settings.
// It is able to convert between different types and handle nested dictionaries.
class ConfigDict : public Dict {
  public:
    using Dict::Dict; // inherit constructors

    using iterator = Dict::iterator;
    using const_iterator = Dict::const_iterator;

    // Copy/move ops (rule of zero-ish; but show explicitly for clarity)
    ConfigDict() = default;
    ConfigDict(const Dict &d) : Dict(d) {}
    ConfigDict(const ConfigDict &d) : Dict(d) {}
    ConfigDict(ConfigDict &&d) noexcept : Dict(std::move(d)) {}
    ConfigDict &operator=(const ConfigDict &d) {
        Dict::operator=(d);
        return *this;
    }
    ConfigDict &operator=(ConfigDict &&d) noexcept {
        Dict::operator=(std::move(d));
        return *this;
    }

    template <typename T> T get(const std::string &key, const T &default_value) const {
        const auto it = this->find(key);
        if (it == this->end()) {
            return default_value;
        }

        const Value &v = it->second;

        // Handle numeric conversions gracefully between int/int64_t/float/double
        if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t>) {
            if (auto p = v.get_if<int>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<int64_t>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<double>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<float>())
                return static_cast<T>(*p);
        } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
            if (auto p = v.get_if<double>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<float>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<int64_t>())
                return static_cast<T>(*p);
            if (auto p = v.get_if<int>())
                return static_cast<T>(*p);
        } else if constexpr (std::is_same_v<T, ConfigDict>) {
            // Special case for ConfigDict - convert from Dict if needed
            if (auto p = v.get_if<Dict>()) {
                return ConfigDict(*p);
            }
        } else {
            // Exact type match (includes std::string, std::vector<double>, etc.)
            if (auto p = v.get_if<T>()) {
                return *p;
            }
        }

        return default_value;
    }

    bool has(const std::string &key) const { return this->find(key) != this->end(); }

    bool is_empty() const { return this->empty(); }

    bool is_null() const { return this->empty(); }

    static ConfigDict from_json(const nlohmann::json &json_data) {
        ConfigDict config;
        for (auto it = json_data.begin(); it != json_data.end(); ++it) {
            config[it.key()] = Value(it.value());
        }
        return config;
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "ConfigDict: {" << std::endl;
        for (auto it = this->begin(); it != this->end(); ++it) {
            ss << "  " << it->first << ": " << it->second.to_string() << std::endl;
        }
        ss << "}" << std::endl;
        return ss.str();
    }
};

using FrameDataDict = Dict;

} // namespace pyslam

#if 0
// Example usage
int main() {
    using namespace pyslam;

    Dict d;
    d["name"] = "Luigi";
    d["age"] = 37;
    d["height_m"] = 1.78;
    d["is_engineer"] = true;
    d["skills"] = List{ "SLAM", "C++", "GTSAM" };
    d["address"] = Dict{
        {"city", "Rome"},
        {"zip",  "00100"}
    };

    // Access safely
    if (auto age = d["age"].get_if<int64_t>()) {
        std::cout << "Age + 1: " << (*age + 1) << "\n";
    }

    // Mutate nested
    d["skills"].get<List>().push_back("ROS2");

    // Print all
    print_value(Value{d});
    std::cout << "\n";
}
#endif
