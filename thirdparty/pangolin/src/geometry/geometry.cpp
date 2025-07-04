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

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/geometry_ply.h>
#include <pangolin/geometry/geometry_obj.h>

#include <pangolin/utils/file_extension.h>
#include <pangolin/utils/file_utils.h>

// TODO: Should really get rid of this from API
#include <pangolin/gl/gl.h>

namespace pangolin {

// TODO: Replace this with proper factory registry
pangolin::Geometry LoadGeometry(const std::string& filename)
{
    const std::string expanded_filename = PathExpand(filename);
    const ImageFileType ft = FileType(expanded_filename);
    if(ft == ImageFileTypePly) {
        return LoadGeometryPly(expanded_filename);
    }else if(ft == ImageFileTypeObj) {
        return LoadGeometryObj(expanded_filename);
    }else{
        throw std::runtime_error("Unsupported geometry file type.");
    }
}

pangolin::Geometry::Element::Attribute MakeAttribute(GLenum datatype, size_t num_items, size_t count_per_item, void* ptr, size_t pitch_bytes)
{
    switch(datatype) {
    case GL_FLOAT:
        return Image<float>   (    (float*)ptr, count_per_item, num_items, pitch_bytes);
    case GL_INT:
    case GL_UNSIGNED_INT:
        return Image<uint32_t>( (uint32_t*)ptr, count_per_item, num_items, pitch_bytes);
    case GL_SHORT:
    case GL_UNSIGNED_SHORT:
        return Image<uint16_t>( (uint16_t*)ptr, count_per_item, num_items, pitch_bytes);
    case GL_BYTE:
    case GL_UNSIGNED_BYTE:
        return Image<uint8_t> (  (uint8_t*)ptr, count_per_item, num_items, pitch_bytes);
    default:
        throw std::runtime_error("Unsupported type");
    };
}

}
