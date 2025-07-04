/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2014 Hauke Strasdat, Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <sstream>

namespace pangolin {
namespace details {

// Following: http://stackoverflow.com/a/22759544
template <typename T>
class IsStreamable {
private:
    template <typename TT>
    static auto test(int) -> decltype( (std::declval<std::stringstream&>() << std::declval<TT>(), std::true_type()) );

    template <typename>
    static auto test(...) -> std::false_type;

public:
    static const bool value = decltype(test<T>(0))::value;
};

inline void FormatStream(std::stringstream& stream, const char* text)
{
    stream << text;
}

// Following: http://en.cppreference.com/w/cpp/language/parameter_pack
template <typename T, typename... Args>
void FormatStream(std::stringstream& stream, const char* text, T arg, Args... args)
{
    static_assert(IsStreamable<T>::value,
                  "One of the args has not an ostream overload!");
    for (; *text != '\0'; ++text) {
        if (*text == '%') {
            stream << arg;
            FormatStream(stream, text + 1, args...);
            return;
        }
        stream << *text;
    }
    stream << "\nFormat-Warning: There are " << sizeof...(Args) + 1
           << " args unused.";
}

}  // namespace details

template <typename... Args>
std::string FormatString(const char* text, Args... args)
{
    std::stringstream stream;
    details::FormatStream(stream, text, args...);
    return stream.str();
}

inline std::string FormatString()
{
    return std::string();
}

}
