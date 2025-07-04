/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2014 Steven Lovegrove
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

#include <ostream>
#include <functional>

namespace pangolin
{

// Find the open brace preceeded by '$'
inline const char* FirstOpenBrace(const char* str, char token = '$', char open = '{')
{
    bool symbol = false;

    for(; *str != '\0'; ++str ) {
        if( *str == token) {
            symbol = true;
        }else{
            if( symbol ) {
                if( *str == open ) {
                    return str;
                } else {
                    symbol = false;
                }
            }
        }
    }
    return 0;
}

// Find the first matching end brace. str includes open brace
inline const char* MatchingEndBrace(const char* str, char open = '{', char close = '}')
{
    int b = 0;
    for(; *str != '\0'; ++str ) {
        if( *str == open ) {
            ++b;
        }else if( *str == close ) {
            --b;
            if( b == 0 ) {
                return str;
            }
        }
    }
    return 0;
}

inline std::string Transform(const std::string& val, std::function<std::string(const std::string&)> dictionary, char token = '$', char open = '{', char close = '}')
{
    std::string expanded = val;

    while(true)
    {
        const char* brace = FirstOpenBrace(expanded.c_str(), token, open);
        if(brace)
        {
            const char* endbrace = MatchingEndBrace(brace);
            if( endbrace )
            {
                std::ostringstream oss;
                oss << std::string(expanded.c_str(), brace-1);

                const std::string inexpand = Transform( std::string(brace+1,endbrace), dictionary, token, open, close );
                oss << dictionary(inexpand);
                oss << std::string(endbrace+1, expanded.c_str() + expanded.length() );
                expanded = oss.str();
                continue;
            }
        }
        break;
    }

    return expanded;
}

}
