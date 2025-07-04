#pragma once

#include <pangolin/platform.h>

#ifdef HAVE_EIGEN
#  include <Eigen/Core>
#endif

namespace pangolin
{

// Scalar / Vector agnostic static_cast-like thing
//
// e.g. Promote float to double:
//   ComponentCast<double,float>::cast(0.14f);
//
// e.g. Promote Eigen::Vector2f to Eigen::Vector2d:
//   ComponentCast<Eigen::Vector2d,Eigen::Vector2f>::cast(Eigen::Vector2f(0.1,0.2);

template <typename To, typename From>
struct ComponentCast
{
    PANGO_HOST_DEVICE
    static To cast(const From& val)
    {
        return static_cast<To>(val);
    }
};

#ifdef HAVE_EIGEN
template <typename To, typename FromDerived>
struct ComponentCast<To, Eigen::MatrixBase<FromDerived> >
{
    PANGO_HOST_DEVICE
    static To cast(const Eigen::MatrixBase<FromDerived>& val)
    {
        return val.template cast<typename To::Scalar>();
    }
};
#endif

}
