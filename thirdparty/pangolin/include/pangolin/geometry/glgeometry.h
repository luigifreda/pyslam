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

#include <pangolin/geometry/geometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/glsl.h>

namespace pangolin {

struct GlGeometry
{
    GlGeometry() = default;
    GlGeometry(GlGeometry&&) = default;
    GlGeometry& operator=(GlGeometry&&) = default;

    struct Element : public GlBufferData
    {
        Element() = default;
        Element(Element&&) = default;
        Element& operator=(Element&&) = default;

        Element(GlBufferType buffer_type, size_t size_bytes, GLenum gluse, uint8_t* data)
            : GlBufferData(buffer_type, size_bytes, gluse, data)
        {}

        inline bool HasAttribute(const std::string& name) const {
            return attributes.find(name) != attributes.end();
        }

        struct Attribute {
            // Stuff needed by glVertexAttribPointer
            GLenum gltype;
            size_t count_per_element;
            size_t num_elements;
            size_t offset;
            size_t stride_bytes;
        };
        std::map<std::string, Attribute> attributes;
    };

    inline bool HasAttribute(const std::string& name) const
    {
        for(const auto& b : buffers) if(b.second.HasAttribute(name)) return true;
        return false;
    }

    // Store vertices and attributes
    std::map<std::string, Element> buffers;
    // Stores index buffers for each sub-object
    std::multimap<std::string, Element> objects;
    // Stores pixmaps
    std::map<std::string, GlTexture> textures;
};

GlGeometry::Element ToGlGeometry(const Geometry::Element& el, GlBufferType buffertype);

GlGeometry ToGlGeometry(const Geometry& geom);

void GlDraw(GlSlProgram& prog, const GlGeometry& geom, const GlTexture *matcap);

}
