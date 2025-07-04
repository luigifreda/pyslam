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

#include <pangolin/geometry/glgeometry.h>

#include <pangolin/gl/glformattraits.h>

namespace pangolin {

GlGeometry::Element ToGlGeometryElement(const Geometry::Element& el, GlBufferType buffertype)
{
    GlGeometry::Element glel(buffertype, el.SizeBytes(), GL_STATIC_DRAW, el.ptr );
    for(const auto& attrib_variant : el.attributes) {
        visit([&](auto&& attrib){
            using T = std::decay_t<decltype(attrib)>;
            auto& glattrib = glel.attributes[attrib_variant.first];
            glattrib.gltype = GlFormatTraits<typename T::PixelType>::gltype;
            glattrib.count_per_element = attrib.w;
            glattrib.num_elements = attrib.h;
            glattrib.offset = (uint8_t*)attrib.ptr - el.ptr;
            glattrib.stride_bytes = attrib.pitch;
        }, attrib_variant.second);
    }
    return glel;
}

GlGeometry ToGlGeometry(const Geometry& geom)
{
    GlGeometry gl;
    for(const auto& b : geom.buffers)
        gl.buffers[b.first] = ToGlGeometryElement(b.second, GlArrayBuffer);

    for(const auto& b : geom.objects)
        gl.objects.emplace(b.first, ToGlGeometryElement(b.second, GlElementArrayBuffer));

    for(const auto& tex : geom.textures) {
        auto& gltex = gl.textures[tex.first];
        gltex.Load(tex.second);
    }
    return gl;
}

void BindGlElement(GlSlProgram& prog, const GlGeometry::Element& el)
{
    el.Bind();
    for(auto& a : el.attributes) {
        const GLint attrib_handle = prog.GetAttributeHandle(a.first);
        const GlGeometry::Element::Attribute& attr = a.second;
        if(attrib_handle >= 0) {
            glEnableVertexAttribArray(attrib_handle);
            glVertexAttribPointer(
                attrib_handle, attr.count_per_element, attr.gltype, GL_TRUE,
                attr.stride_bytes,
                (uint8_t*)0 + attr.offset
            );
        }
    }
}

void UnbindGlElements(GlSlProgram& prog, const GlGeometry::Element& el)
{
    for(auto& a : el.attributes) {
        const GLint attrib_handle = prog.GetAttributeHandle(a.first);
        if(attrib_handle >= 0) {
            glDisableVertexAttribArray(attrib_handle);
        }
    }
    el.Unbind();
}

void GlDraw(GlSlProgram& prog, const GlGeometry& geom, const GlTexture* matcap)
{
    // Bind textures
    int num_tex_bound = 0;
    for(auto& tex : geom.textures) {
        glActiveTexture(GL_TEXTURE0 + num_tex_bound);
        tex.second.Bind();
        prog.SetUniform(tex.first, (int)num_tex_bound);
        ++num_tex_bound;
    }

    if(matcap) {
        glActiveTexture(GL_TEXTURE0 + num_tex_bound);
        matcap->Bind();
        prog.SetUniform("matcap", (int)num_tex_bound);
        ++num_tex_bound;
    }

    // Bind all attribute buffers
    for(auto& buffer : geom.buffers) {
        BindGlElement(prog, buffer.second);
    }

    // Draw all geometry
    for(auto& buffer : geom.objects) {
        auto it_indices = buffer.second.attributes.find("vertex_indices");
        if(it_indices != buffer.second.attributes.end()) {
            buffer.second.Bind();
            auto& attrib = it_indices->second;
            glDrawElements(
               GL_TRIANGLES, attrib.count_per_element * attrib.num_elements,
               attrib.gltype, (uint8_t*)0 + attrib.offset
            );
            buffer.second.Unbind();
        }
    }

    // Unbind attribute buffers
    for(auto& buffer : geom.buffers) {
        UnbindGlElements(prog, buffer.second);
    }

    // Unbind textures
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
}

}
