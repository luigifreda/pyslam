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

#include <pangolin/gl/glplatform.h>

#ifdef HAVE_EIGEN
#  include <Eigen/Core>
#endif

namespace pangolin
{

template<typename T>
struct GlFormatTraits;
//{
//    static const GLint glinternalformat = 0;
//    static const GLenum glformat = 0;
//    static const GLenum gltype = 0;
//    static const T glmin = 0;
//    static const T glmax = 0;
//};

template<>
struct GlFormatTraits<unsigned char>
{
    static const GLint glinternalformat = GL_LUMINANCE8;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_UNSIGNED_BYTE;
    static const size_t components = 1;
};

template<>
struct GlFormatTraits<unsigned short>
{
    static const GLint glinternalformat = GL_LUMINANCE16;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_UNSIGNED_SHORT;
    static const size_t components = 1;
};

template<>
struct GlFormatTraits<unsigned int>
{
    static const GLint glinternalformat = GL_LUMINANCE32I_EXT;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_UNSIGNED_INT;
    static const size_t components = 1;
};

template<>
struct GlFormatTraits<int>
{
    static const GLint glinternalformat = GL_LUMINANCE32I_EXT;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_INT;
    static const size_t components = 1;
};

template<>
struct GlFormatTraits<float>
{
    static const GLint glinternalformat = GL_LUMINANCE32F_ARB;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_FLOAT;
    static const size_t components = 1;
};

template<>
struct GlFormatTraits<double>
{
    static const GLint glinternalformat = GL_LUMINANCE32F_ARB;
    static const GLenum glformat = GL_LUMINANCE;
    static const GLenum gltype = GL_DOUBLE;
    static const size_t components = 1;
};



#ifdef HAVE_EIGEN

//////////////////////////////////////////////////////////////////

template <>
struct GlFormatTraits<Eigen::Vector2i>
{
    static const GLint glinternalformat = GL_RG32I;
    static const GLenum glformat = GL_RG;
    static const GLenum gltype = GL_INT;
    static const size_t components = 2;
};

template <>
struct GlFormatTraits<Eigen::Vector2f>
{
    static const GLint glinternalformat = GL_RG32F;
    static const GLenum glformat = GL_RG;
    static const GLenum gltype = GL_FLOAT;
    static const size_t components = 2;
};

template <>
struct GlFormatTraits<Eigen::Vector2d>
{
    static const GLint glinternalformat = GL_RG32F;
    static const GLenum glformat = GL_RG;
    static const GLenum gltype = GL_DOUBLE;
    static const size_t components = 2;
};

//////////////////////////////////////////////////////////////////

template <>
struct GlFormatTraits<Eigen::Matrix<unsigned char,3,1>>
{
    static const GLint glinternalformat = GL_RGB8;
    static const GLenum glformat = GL_RGB;
    static const GLenum gltype = GL_UNSIGNED_BYTE;
    static const size_t components = 3;
};

template <>
struct GlFormatTraits<Eigen::Matrix<unsigned short,3,1>>
{
    static const GLint glinternalformat = GL_RGBA16;
    static const GLenum glformat = GL_RGB;
    static const GLenum gltype = GL_UNSIGNED_SHORT;
    static const size_t components = 3;
};

template <>
struct GlFormatTraits<Eigen::Vector3f>
{
    static const GLint glinternalformat = GL_RGB32F;
    static const GLenum glformat = GL_RGB;
    static const GLenum gltype = GL_FLOAT;
    static const size_t components = 3;
};

template <>
struct GlFormatTraits<Eigen::Vector3d>
{
    static const GLint glinternalformat = GL_RGB32F;
    static const GLenum glformat = GL_RGB;
    static const GLenum gltype = GL_DOUBLE;
    static const size_t components = 3;
};

//////////////////////////////////////////////////////////////////

template <>
struct GlFormatTraits<Eigen::Matrix<unsigned char,4,1>>
{
    static const GLint glinternalformat = GL_RGBA8;
    static const GLenum glformat = GL_RGBA;
    static const GLenum gltype = GL_UNSIGNED_BYTE;
    static const size_t components = 4;
};

template <>
struct GlFormatTraits<Eigen::Matrix<unsigned short,4,1>>
{
    static const GLint glinternalformat = GL_RGBA16;
    static const GLenum glformat = GL_RGBA;
    static const GLenum gltype = GL_UNSIGNED_SHORT;
    static const size_t components = 4;
};

template <>
struct GlFormatTraits<Eigen::Vector4f>
{
    static const GLint glinternalformat = GL_RGBA32F;
    static const GLenum glformat = GL_RGBA;
    static const GLenum gltype = GL_FLOAT;
    static const size_t components = 4;
};

template <>
struct GlFormatTraits<Eigen::Vector4d>
{
    static const GLint glinternalformat = GL_RGBA32F;
    static const GLenum glformat = GL_RGBA;
    static const GLenum gltype = GL_DOUBLE;
    static const size_t components = 4;
};

#endif // HAVE_EIGEN

}
