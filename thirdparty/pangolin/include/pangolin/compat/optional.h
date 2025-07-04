#pragma once

#include <pangolin/platform.h>

// Use either C++17 optional, or the standalone backwards compatible version
#if (__cplusplus >= 201703L)
#   include <optional>
#else
#   include <experimental/optional.hpp>
#endif

namespace pangolin {
#if (__cplusplus >= 201703L)
template <typename T>
using optional = std::optional<T>;
#else
template <typename T>
using optional = std::experimental::optional<T>;
#endif
}
