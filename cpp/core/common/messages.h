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

#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace IoColor {
inline constexpr std::string_view reset = "\033[0m";
inline constexpr std::string_view red = "\033[31m";
inline constexpr std::string_view green = "\033[32m";
inline constexpr std::string_view yellow = "\033[33m";
inline constexpr std::string_view blue = "\033[34m";
} // namespace IoColor

// ---------- Log level control ----------
#ifndef LOG_LEVEL
// 0=OFF, 1=ERROR, 2=WARN, 3=INFO
#define LOG_LEVEL 3
#endif

namespace detail {
inline std::ostream &log_stream() { return std::clog; } // buffered
inline std::ostream &err_stream() { return std::cerr; } // for fatals

inline void log_impl(std::ostream &os, std::string_view color, std::string_view prefix,
                     std::string_view msg) {
#if 0
    static std::mutex m; // to enable if we need thread-safety
    std::lock_guard<std::mutex> lock(m);
#endif
    os << color << prefix << msg << IoColor::reset << '\n';
}
} // namespace detail

// ---------- Macros (safe do/while(0) form) ----------
#define MSG_LOG_RAW(message, prefix, color, stream)                                                \
    do {                                                                                           \
        ::detail::log_impl(stream, color, prefix, std::string_view(message));                      \
    } while (0)

#if LOG_LEVEL >= 3
#define MSG_INFO(message) MSG_LOG_RAW(message, "INFO: ", IoColor::blue, ::detail::log_stream())
#define MSG_INFO_STREAM(args)                                                                      \
    do {                                                                                           \
        std::ostringstream ss;                                                                     \
        ss << args;                                                                                \
        MSG_INFO(ss.str());                                                                        \
    } while (0)
#else
#define MSG_INFO(message)                                                                          \
    do {                                                                                           \
    } while (0)
#define MSG_INFO_STREAM(args)                                                                      \
    do {                                                                                           \
    } while (0)
#endif

#if LOG_LEVEL >= 2
#define MSG_WARN(message) MSG_LOG_RAW(message, "WARNING: ", IoColor::yellow, ::detail::log_stream())
#define MSG_WARN_STREAM(args)                                                                      \
    do {                                                                                           \
        std::ostringstream ss;                                                                     \
        ss << args;                                                                                \
        MSG_WARN(ss.str());                                                                        \
    } while (0)
#else
#define MSG_WARN(message)                                                                          \
    do {                                                                                           \
    } while (0)
#define MSG_WARN_STREAM(args)                                                                      \
    do {                                                                                           \
    } while (0)
#endif

#if LOG_LEVEL >= 2
#define MSG_RED_WARN(message)                                                                      \
    MSG_LOG_RAW(message, "WARNING: ", IoColor::red, ::detail::log_stream())
#define MSG_RED_WARN_STREAM(args)                                                                  \
    do {                                                                                           \
        std::ostringstream ss;                                                                     \
        ss << args;                                                                                \
        MSG_RED_WARN(ss.str());                                                                    \
    } while (0)
#else
#define MSG_RED_WARN(message)                                                                      \
    do {                                                                                           \
    } while (0)
#define MSG_RED_WARN_STREAM(args)                                                                  \
    do {                                                                                           \
    } while (0)
#endif

#if LOG_LEVEL >= 1
#define MSG_ERROR(message)                                                                         \
    do {                                                                                           \
        MSG_LOG_RAW(message, "ERROR: ", IoColor::red, ::detail::err_stream());                     \
        std::abort();                                                                              \
    } while (0)
#define MSG_ERROR_STREAM(args)                                                                     \
    do {                                                                                           \
        std::ostringstream ss;                                                                     \
        ss << args;                                                                                \
        MSG_ERROR(ss.str());                                                                       \
    } while (0)
#else
#define MSG_ERROR(message)                                                                         \
    do {                                                                                           \
    } while (0)
#define MSG_ERROR_STREAM(args)                                                                     \
    do {                                                                                           \
    } while (0)
#endif

// ---------- Assertions ----------
#ifndef NDEBUG
#define MSG_ASSERT(condition, message)                                                             \
    do {                                                                                           \
        if (__builtin_expect(!(condition), 0)) { /* [[unlikely]] alt */                            \
            ::detail::log_impl(::detail::err_stream(), IoColor::red, "ASSERT: ", #condition);      \
            ::detail::log_impl(::detail::err_stream(), IoColor::red,                               \
                               "WHERE: ", __FILE__ ":" + std::to_string(__LINE__));                \
            ::detail::log_impl(::detail::err_stream(), IoColor::red, "FUNC:  ", __func__);         \
            ::detail::log_impl(::detail::err_stream(), IoColor::red,                               \
                               "MSG:   ", std::string_view(message));                              \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)
#else
#define MSG_ASSERT(condition, message)                                                             \
    do {                                                                                           \
        (void)sizeof(condition);                                                                   \
    } while (0)
#endif

#define MSG_FORCED_ASSERT(condition, message)                                                      \
    do {                                                                                           \
        if (__builtin_expect(!(condition), 0)) { /* [[unlikely]] alt */                            \
            ::detail::log_impl(::detail::err_stream(), IoColor::red, "ASSERT: ", #condition);      \
            ::detail::log_impl(::detail::err_stream(), IoColor::red,                               \
                               "WHERE: ", __FILE__ ":" + std::to_string(__LINE__));                \
            ::detail::log_impl(::detail::err_stream(), IoColor::red, "FUNC:  ", __func__);         \
            ::detail::log_impl(::detail::err_stream(), IoColor::red,                               \
                               "MSG:   ", std::string_view(message));                              \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)
