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

#ifndef PANGOLIN_PARSE_H
#define PANGOLIN_PARSE_H

#include <string>
#include <cctype>

namespace pangolin
{

const unsigned int parse_max_token_size = 1024;

inline void ConsumeWhitespace(std::istream& is)
{
    while(is.good() && std::isspace(is.peek()) ) {
        is.get();
    }
}

inline bool ConsumeToNewline(std::istream& is)
{
    while(is.good()) {
        if(is.get() == '\n') {
            return true;
        }
    }
    return false;
}

template<size_t buffer_size> inline
size_t ReadToken(std::istream& is, char buffer[buffer_size])
{
    size_t r = 0;
    while(is.good() && r < buffer_size-1) {
        int c = is.peek();
        if( std::isgraph(c) ) {
            buffer[r++] = (char)is.get();
        }else{
            break;
        }
    }
    buffer[r] = '\0';
    return r;
}

inline std::string ReadToken(std::istream &is)
{
    char str_token[parse_max_token_size];
    ReadToken<parse_max_token_size>(is, str_token);
    return std::string(str_token);
}

template<size_t buffer_size> inline
size_t ConsumeWhitespaceReadToken(std::istream& is, char buffer[buffer_size])
{
    ConsumeWhitespace(is);
    return ReadToken<buffer_size>(is, buffer);
}

inline int ParseToken(const char* token, const char* token_list[], size_t token_list_size)
{
    for(size_t i=0; i < token_list_size; ++i) {
        if( strcmp(token, token_list[i]) == 0 ) {
            return i;
        }
    }
    return -1;
}

#define PANGOLIN_DEFINE_PARSE_TOKEN(x) \
    inline x ParseToken##x(const char* token) { \
        return (x)ParseToken(token, x##String, x##Size); \
    } \
    inline x ParseToken##x(std::istream& is) { \
        char str_token[parse_max_token_size]; \
        ReadToken<parse_max_token_size>(is, str_token); \
        return ParseToken##x( str_token ); \
    }

}

#endif // PANGOLIN_PARSE_H
