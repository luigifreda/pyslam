#pragma once

#include <pangolin/platform.h>

// Use either C++17 variant, or the standalone backwards compatible version
// of M. Park.
#if (__cplusplus >= 201703L)
#   include <variant>
#else
#   include <mpark/variant.hpp>
#endif

namespace pangolin {
#if (__cplusplus >= 201703L)
using std::variant;
using std::get;
using std::get_if;
using std::visit;
#else
using mpark::variant;
using mpark::get;
using mpark::get_if;
using mpark::visit;
#endif
}
