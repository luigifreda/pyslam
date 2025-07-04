#pragma once

#include <tuple>

namespace pangolin {

template<typename TPred, typename T>
bool all_of(TPred pred, const T& i)
{
    return pred(i);
}

template<typename TPred, typename T, typename... Targs>
bool all_of(const TPred& pred, const T& i, const Targs& ... ins)
{
    return pred(i) && all_of(pred, ins...);
}

template<typename TPred, typename T>
bool any_of(TPred pred, const T& i)
{
    return pred(i);
}

template<typename TPred, typename T, typename... Targs>
bool any_of(const TPred& pred, const T& i, Targs& ... ins)
{
    return pred(i) || any_of(pred, ins...);
}

template<typename TContainer, typename... Targs>
bool all_found(const TContainer& c, const Targs& ... its)
{
    using T1 = typename std::tuple_element<0, std::tuple<Targs...>>::type;
    return all_of([&c](const T1& it){return it != c.end();}, its...);
}

template<typename T, typename... Targs>
bool all_equal(const T& v1, const Targs& ... its)
{
    return all_of([v1](const T& o){return v1 == o;}, its...);
}

}
