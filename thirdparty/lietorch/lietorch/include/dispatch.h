#ifndef DISPATCH_H
#define DISPATCH_H

#include <torch/extension.h>

#include "so3.h"
#include "rxso3.h"
#include "se3.h"
#include "sim3.h"


#define PRIVATE_CASE_TYPE(group_index, enum_type, type, ...)    \
  case enum_type: {                                             \
    using scalar_t = type;                                      \
    switch (group_index) {                                      \
      case 1: {                                                 \
        using group_t = SO3<type>;                              \
        return __VA_ARGS__();                                   \
      }                                                         \
      case 2: {                                                 \
        using group_t = RxSO3<type>;                            \
        return __VA_ARGS__();                                   \
      }                                                         \
      case 3: {                                                 \
        using group_t = SE3<type>;                              \
        return __VA_ARGS__();                                   \
      }                                                         \
      case 4: {                                                 \
        using group_t = Sim3<type>;                             \
        return __VA_ARGS__();                                   \
      }                                                         \
    }                                                           \
  }                                                             \

#define DISPATCH_GROUP_AND_FLOATING_TYPES(GROUP_INDEX, TYPE, NAME, ...)              \
  [&] {                                                                              \
    const auto& the_type = TYPE;                                                     \
    /* don't use TYPE again in case it is an expensive or side-effect op */          \
    at::ScalarType _st = ::detail::scalar_type(the_type);                            \
    switch (_st) {                                                                   \
      PRIVATE_CASE_TYPE(GROUP_INDEX, at::ScalarType::Double, double, __VA_ARGS__)    \
      PRIVATE_CASE_TYPE(GROUP_INDEX, at::ScalarType::Float, float, __VA_ARGS__)      \
      default: break;                                                                \
    }                                                                                \
  }()

#endif

