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

#include "eigen_aliases.h"

#include <any>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include <DBoW2/DBoW2.h>
#include <DBoW3/DBoW3.h>
#include <opencv2/opencv.hpp>

namespace pyslam {

// TODO: This is a work in progress in the case USE_PYTHON is not defined

// Forward declare self-type to enable recursive containers via std::any.
class GlobalDescriptor;

// Recursive container aliases stored inside std::any
using GlobalList = std::vector<GlobalDescriptor>;
using GlobalDict = std::unordered_map<std::string, GlobalDescriptor>;

using GlobalDescriptorDataVariant =
    std::variant<std::monostate, // null / unset
                 std::vector<bool>, std::vector<int>, std::vector<int64_t>, std::vector<float>,
                 std::vector<double>, pyslam::VecNf, pyslam::VecNd, // vectors
                 DBoW3::BowVector, DBoW2::BowVector, cv::Mat, std::any>;

constexpr size_t GlobalDescriptorDataVariantAnyIndex =
    std::variant_size_v<GlobalDescriptorDataVariant> - 1;

class GlobalDescriptor {

  private:
    GlobalDescriptorDataVariant data;

    // helper for printing/serialization
    static void print_recursive(std::ostream &os, const GlobalDescriptor &gd, int indent);

    static void pad(std::ostream &os, int n) {
        for (int i = 0; i < n; ++i)
            os << ' ';
    }

  public:
    // Constructors
    GlobalDescriptor() : data(std::monostate{}) {}
    GlobalDescriptor(const GlobalDescriptorDataVariant &data_) : data(data_) {}
    GlobalDescriptor(GlobalDescriptorDataVariant &&data_) : data(std::move(data_)) {}

    // Convenience constructors for each supported type
    GlobalDescriptor(const std::vector<bool> &v) : data(v) {}
    GlobalDescriptor(std::vector<bool> &&v) : data(std::move(v)) {}

    GlobalDescriptor(const std::vector<int> &v) : data(v) {}
    GlobalDescriptor(std::vector<int> &&v) : data(std::move(v)) {}

    GlobalDescriptor(const std::vector<int64_t> &v) : data(v) {}
    GlobalDescriptor(std::vector<int64_t> &&v) : data(std::move(v)) {}

    GlobalDescriptor(const std::vector<float> &v) : data(v) {}
    GlobalDescriptor(std::vector<float> &&v) : data(std::move(v)) {}

    GlobalDescriptor(const std::vector<double> &v) : data(v) {}
    GlobalDescriptor(std::vector<double> &&v) : data(std::move(v)) {}

    GlobalDescriptor(const pyslam::VecNf &v) : data(v) {}
    GlobalDescriptor(pyslam::VecNf &&v) : data(std::move(v)) {}

    GlobalDescriptor(const pyslam::VecNd &v) : data(v) {}
    GlobalDescriptor(pyslam::VecNd &&v) : data(std::move(v)) {}

    GlobalDescriptor(const DBoW3::BowVector &v) : data(v) {}
    GlobalDescriptor(DBoW3::BowVector &&v) : data(std::move(v)) {}

    GlobalDescriptor(const DBoW2::BowVector &v) : data(v) {}
    GlobalDescriptor(DBoW2::BowVector &&v) : data(std::move(v)) {}

    GlobalDescriptor(const cv::Mat &m) : data(m) {}
    GlobalDescriptor(cv::Mat &&m) : data(std::move(m)) {}

    // Recursive containers via std::any
    GlobalDescriptor(const GlobalList &l) : data(std::any(l)) {}
    GlobalDescriptor(GlobalList &&l) : data(std::any(std::move(l))) {}

    GlobalDescriptor(const GlobalDict &d) : data(std::any(d)) {}
    GlobalDescriptor(GlobalDict &&d) : data(std::any(std::move(d))) {}

    // Rule of 5
    GlobalDescriptor(const GlobalDescriptor &other) : data(other.data) {}
    GlobalDescriptor(GlobalDescriptor &&other) noexcept : data(std::move(other.data)) {}
    GlobalDescriptor &operator=(const GlobalDescriptor &other) {
        data = other.data;
        return *this;
    }
    GlobalDescriptor &operator=(GlobalDescriptor &&other) noexcept {
        data = std::move(other.data);
        return *this;
    }
    ~GlobalDescriptor() = default;

    // Type checks
    template <typename T> bool is() const {
        if constexpr (std::is_same_v<T, GlobalList>) {
            return data.index() == GlobalVariantAnyIndex &&
                   std::any_cast<GlobalList>(&std::get<std::any>(data)) != nullptr;
        } else if constexpr (std::is_same_v<T, GlobalDict>) {
            return data.index() == GlobalVariantAnyIndex &&
                   std::any_cast<GlobalDict>(&std::get<std::any>(data)) != nullptr;
        } else {
            return std::holds_alternative<T>(data);
        }
    }

    // Safe access; returns nullptr if wrong type
    template <typename T> const T *get_if() const {
        if constexpr (std::is_same_v<T, GlobalList>) {
            if (data.index() == GlobalVariantAnyIndex) {
                return std::any_cast<GlobalList>(&std::get<std::any>(data));
            }
            return nullptr;
        } else if constexpr (std::is_same_v<T, GlobalDict>) {
            if (data.index() == GlobalVariantAnyIndex) {
                return std::any_cast<GlobalDict>(&std::get<std::any>(data));
            }
            return nullptr;
        } else {
            return std::get_if<T>(&data);
        }
    }

    template <typename T> T *get_if() {
        if constexpr (std::is_same_v<T, GlobalList>) {
            if (data.index() == GlobalVariantAnyIndex) {
                return std::any_cast<GlobalList>(&std::get<std::any>(data));
            }
            return nullptr;
        } else if constexpr (std::is_same_v<T, GlobalDict>) {
            if (data.index() == GlobalVariantAnyIndex) {
                return std::any_cast<GlobalDict>(&std::get<std::any>(data));
            }
            return nullptr;
        } else {
            return std::get_if<T>(&data);
        }
    }

    // Throws std::bad_variant_access if wrong type
    template <typename T> const T &get() const {
        if constexpr (std::is_same_v<T, GlobalList>) {
            return std::any_cast<const GlobalList &>(std::get<std::any>(data));
        } else if constexpr (std::is_same_v<T, GlobalDict>) {
            return std::any_cast<const GlobalDict &>(std::get<std::any>(data));
        } else {
            return std::get<T>(data);
        }
    }

    template <typename T> T &get() {
        if constexpr (std::is_same_v<T, GlobalList>) {
            return std::any_cast<GlobalList &>(std::get<std::any>(data));
        } else if constexpr (std::is_same_v<T, GlobalDict>) {
            return std::any_cast<GlobalDict &>(std::get<std::any>(data));
        } else {
            return std::get<T>(data);
        }
    }

    // Misc helpers
    bool is_null() const { return std::holds_alternative<std::monostate>(data); }

    // Access to underlying variant (const and non-const)
    const GlobalDescriptorDataVariant &variant() const { return data; }
    GlobalDescriptorDataVariant &variant() { return data; }

    // String conversion / printing
    std::string to_string() const {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const GlobalDescriptor &gd) {
        GlobalDescriptor::print_recursive(os, gd, 0);
        return os;
    }
};

// ---- implementation details ----
inline void GlobalDescriptor::print_recursive(std::ostream &os, const GlobalDescriptor &gd,
                                              int indent) {
    std::visit(
        [&](auto const &val_data) {
            using T = std::decay_t<decltype(val_data)>;

            if constexpr (std::is_same_v<T, std::monostate>) {
                os << "null";
            } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
                os << "[";
                for (size_t i = 0; i < val_data.size(); ++i) {
                    os << (val_data[i] ? "true" : "false");
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
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
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
            } else if constexpr (std::is_same_v<T, pyslam::VecNf>) {
                os << "VecNf[";
                for (int i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, pyslam::VecNd>) {
                os << "VecNd[";
                for (int i = 0; i < val_data.size(); ++i) {
                    os << val_data[i];
                    if (i + 1 < val_data.size())
                        os << ", ";
                }
                os << "]";
            } else if constexpr (std::is_same_v<T, DBoW3::BowVector>) {
                os << "DBoW3::BowVector{";
                size_t i = 0;
                for (auto const &kv : val_data) {
                    os << kv.first << ":" << kv.second;
                    if (++i < val_data.size())
                        os << ", ";
                }
                os << "}";
            } else if constexpr (std::is_same_v<T, DBoW2::BowVector>) {
                os << "DBoW2::BowVector{";
                size_t i = 0;
                for (auto const &kv : val_data) {
                    os << kv.first << ":" << kv.second;
                    if (++i < val_data.size())
                        os << ", ";
                }
                os << "}";
            } else if constexpr (std::is_same_v<T, cv::Mat>) {
                os << "cv::Mat(" << val_data.rows << "x" << val_data.cols
                   << ", type=" << val_data.type() << ")";
            } else if constexpr (std::is_same_v<T, std::any>) {
                // recursive containers
                try {
                    if (auto list_ptr = std::any_cast<GlobalList>(&val_data)) {
                        os << "[\n";
                        for (size_t i = 0; i < list_ptr->size(); ++i) {
                            pad(os, indent + 2);
                            print_recursive(os, (*list_ptr)[i], indent + 2);
                            os << (i + 1 < list_ptr->size() ? ",\n" : "\n");
                        }
                        pad(os, indent);
                        os << "]";
                    } else if (auto dict_ptr = std::any_cast<GlobalDict>(&val_data)) {
                        os << "{\n";
                        size_t i = 0;
                        for (auto const &kv : *dict_ptr) {
                            pad(os, indent + 2);
                            os << '"' << kv.first << "\": ";
                            print_recursive(os, kv.second, indent + 2);
                            os << (++i < dict_ptr->size() ? ",\n" : "\n");
                        }
                        pad(os, indent);
                        os << "}";
                    } else {
                        os << "unknown";
                    }
                } catch (const std::bad_any_cast &) {
                    os << "unknown";
                }
            }
        },
        gd.data);
}

/*

Usage example:

pyslam::GlobalDescriptor gd_vec(pyslam::VecNf::Ones(128));
pyslam::GlobalDescriptor gd_dbow3(DBoW3::BowVector{});
pyslam::GlobalList lst{ gd_vec, gd_dbow3 };
pyslam::GlobalDescriptor gd_list(lst);

std::cout << gd_list << std::endl;

*/

} // namespace pyslam