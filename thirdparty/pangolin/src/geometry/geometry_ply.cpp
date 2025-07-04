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

#include <pangolin/geometry/geometry_ply.h>

#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/variadic_all.h>
#include <pangolin/utils/parse.h>
#include <pangolin/utils/type_convert.h>
#include <pangolin/utils/simple_math.h>

// TODO: Should really remove need for GL here.
#include <pangolin/gl/gl.h>

namespace pangolin {

#define FORMAT_STRING_LIST(x) #x,

const char* PlyHeaderString[] = {
   PLY_HEADER_LIST(FORMAT_STRING_LIST)
};

const char* PlyFormatString[] = {
   PLY_FORMAT_LIST(FORMAT_STRING_LIST)
};

const char* PlyTypeString[] = {
   PLY_TYPE_LIST(FORMAT_STRING_LIST)
};

#undef FORMAT_STRING_LIST

PLY_GROUP_LIST(PANGOLIN_DEFINE_PARSE_TOKEN)

void ParsePlyHeader(PlyHeaderDetails& ply, std::istream& is)
{
    // 'Active' element for property definitions.
    int current_element = -1;

    // Check header is correct
    PlyHeader token = ParseTokenPlyHeader(is);
    if( token != PlyHeader_ply) {
        throw std::runtime_error("Bad PLY header magic.");
    }
    ConsumeToNewline(is);

    while(is.good() && token != PlyHeader_end_header) {
        token = ParseTokenPlyHeader(is);
        switch (token) {
        case PlyHeader_format:
            // parse PLY format and version
            ConsumeWhitespace(is);
            ply.format = ParseTokenPlyFormat(is);
            ConsumeWhitespace(is);
            ply.version = ReadToken(is);
            break;
        case PlyHeader_element:
        {
            current_element = ply.elements.size();
            PlyElementDetails el;
            el.stride_bytes = 0;
            ConsumeWhitespace(is);
            el.name = ReadToken(is);
            ConsumeWhitespace(is);
            el.num_items = FromString<int>(ReadToken(is));
            ply.elements.push_back(el);
            break;
        }
        case PlyHeader_property:
            if(current_element >= 0) {
                PlyElementDetails& el = ply.elements[current_element];
                PlyPropertyDetails prop;
                ConsumeWhitespace(is);
                const PlyType t = ParseTokenPlyType(is);
                if( t == PlyType_list) {
                    ConsumeWhitespace(is);
                    const PlyType idtype = ParseTokenPlyType(is);
                    ConsumeWhitespace(is);
                    const PlyType itemtype = ParseTokenPlyType(is);
                    prop.list_index_type = PlyTypeGl[idtype];
                    prop.type = PlyTypeGl[itemtype];
                    prop.offset_bytes = el.stride_bytes;
                    prop.num_items = -1;
                    el.stride_bytes = -1;
                }else{
                    prop.type = PlyTypeGl[t];
                    prop.list_index_type = GL_NONE;
                    prop.offset_bytes = el.stride_bytes;
                    prop.num_items = 1;
                    const size_t size_bytes = GlDataTypeBytes(prop.type);
                    if( el.stride_bytes >= 0) {
                        el.stride_bytes += size_bytes;
                    }
                }
                ConsumeWhitespace(is);
                prop.name = ReadToken(is);
                el.properties.push_back(prop);
            }else{
                pango_print_warn("PLY Parser: property declaration before element. Ignoring line.");
            }
            break;
        case PlyHeader_comment:
        case PlyHeader_end_header:
            break;
        default:
            pango_print_warn("PLY Parser: Unknown token - ignoring line.");
        }
        ConsumeToNewline(is);
    }
}

void ParsePlyAscii(pangolin::Geometry& /*geom*/, const PlyHeaderDetails& /*ply*/, std::istream& /*is*/)
{
    throw std::runtime_error("Not implemented.");
}

void AddVertexNormals(pangolin::Geometry& geom)
{
    auto it_geom = geom.buffers.find("geometry");
    auto it_face = geom.objects.find("default");

    if(it_geom != geom.buffers.end() && it_face != geom.objects.end())
    {
        const auto it_vbo = it_geom->second.attributes.find("vertex");
        const auto it_ibo = it_face->second.attributes.find("vertex_indices");

        if(it_vbo != it_geom->second.attributes.end() && it_ibo != it_face->second.attributes.end()) {
            const auto& ibo = get<Image<uint32_t>>(it_ibo->second);
            const auto& vbo = get<Image<float>>(it_vbo->second);

            // Assume we have triangles.
            PANGO_ASSERT(ibo.w == 3 && vbo.w == 3);

            ManagedImage<float> vert_normals(3, vbo.h);
            ManagedImage<size_t> vert_face_count(1, vbo.h);
            vert_normals.Fill(0.0f);
            vert_face_count.Fill(0);

            float ab[3];
            float ac[3];
            float fn[3];

            for(size_t i=0; i < ibo.h; ++i) {
                uint32_t i0 = ibo(0,i);
                uint32_t i1 = ibo(1,i);
                uint32_t i2 = ibo(2,i);
                MatSub<3,1>(ab, vbo.RowPtr(i1), vbo.RowPtr(i0));
                MatSub<3,1>(ac, vbo.RowPtr(i2), vbo.RowPtr(i0));
                VecCross3(fn, ab, ac);
                Normalise<3>(fn);
                for(size_t v=0; v < 3; ++v) {
                    MatAdd<3,1>(vert_normals.RowPtr(ibo(v,i)), vert_normals.RowPtr(ibo(v,i)), fn);
                    ++vert_face_count(0,ibo(v,i));
                }
            }

            for(size_t v=0; v < vert_normals.h; ++v) {
                // Compute average
                MatMul<3,1>(vert_normals.RowPtr(v), 1.0f / vert_face_count(0,v));
            }

            auto& el = geom.buffers["normal"];
            (ManagedImage<float>&)el = std::move(vert_normals);
            auto& attr_norm = el.attributes["normal"];
            attr_norm = MakeAttribute(GL_FLOAT, el.h, 3, el.ptr, el.pitch);
        }
    }

}

void StandardizeXyzToVertex(pangolin::Geometry& geom)
{
    auto it_verts = geom.buffers.find("geometry");

    if(it_verts != geom.buffers.end()) {
        auto& verts = it_verts->second;
        auto it_x = verts.attributes.find("x");
        auto it_y = verts.attributes.find("y");
        auto it_z = verts.attributes.find("z");
        if(all_found(verts.attributes, it_x, it_y, it_z)) {
            if(verts.attributes.find("vertex") == verts.attributes.end()) {
                auto& vertex = verts.attributes["vertex"];
                auto& imx = get<Image<float>>(it_x->second);
                auto& imy = get<Image<float>>(it_y->second);
                auto& imz = get<Image<float>>(it_z->second);

                if(imx.ptr + 1 == imy.ptr && imy.ptr + 1 == imz.ptr) {
                    vertex = MakeAttribute(GL_FLOAT, verts.h, 3, imx.ptr, imx.pitch);
                }else{
                    throw std::runtime_error("Ooops");
                }
            }
            verts.attributes.erase(it_x);
            verts.attributes.erase(it_y);
            verts.attributes.erase(it_z);
        }
    }
}

void StandardizeRgbToColor(pangolin::Geometry& geom)
{
    auto it_verts = geom.buffers.find("geometry");

    if(it_verts != geom.buffers.end()) {
        auto& verts = it_verts->second;
        auto it_r = verts.attributes.find("r");
        auto it_g = verts.attributes.find("g");
        auto it_b = verts.attributes.find("b");
        auto it_a = verts.attributes.find("a");

        if(!all_found(verts.attributes, it_r, it_b, it_g)) {
            it_r = verts.attributes.find("red");
            it_g = verts.attributes.find("green");
            it_b = verts.attributes.find("blue");
            it_a = verts.attributes.find("alpha");
        }

        if(all_found(verts.attributes, it_r, it_g, it_b)) {
            const bool have_alpha = it_a != verts.attributes.end();

            if(verts.attributes.find("color") == verts.attributes.end()) {
                Geometry::Element::Attribute& red = it_r->second;
                Geometry::Element::Attribute& color = verts.attributes["color"];

                // TODO: Check that these really are contiguous in memory...
                if(auto attrib = get_if<Image<float>>(&red)) {
                    color = Image<float>(attrib->ptr, have_alpha ? 4 : 3, verts.h, verts.pitch);
                }else if(auto attrib = get_if<Image<uint8_t>>(&red)) {
                    color = Image<uint8_t>(attrib->ptr, have_alpha ? 4 : 3, verts.h, verts.pitch);
                }else if(auto attrib = get_if<Image<uint16_t>>(&red)) {
                    color = Image<uint16_t>(attrib->ptr, have_alpha ? 4 : 3, verts.h, verts.pitch);
                }else if(auto attrib = get_if<Image<uint32_t>>(&red)) {
                    color = Image<uint32_t>(attrib->ptr, have_alpha ? 4 : 3, verts.h, verts.pitch);
                }
            }
            verts.attributes.erase(it_r);
            verts.attributes.erase(it_g);
            verts.attributes.erase(it_b);
            if(have_alpha) verts.attributes.erase(it_a);
        }
    }
}

void StandardizeMultiTextureFaceToXyzuv(pangolin::Geometry& geom)
{
    const auto it_multi_texture_face = geom.buffers.find("multi_texture_face");
    const auto it_multi_texture_vertex = geom.buffers.find("multi_texture_vertex");
    const auto it_geom = geom.buffers.find("geometry");
    const auto it_face = geom.objects.find("default");

    if(it_geom != geom.buffers.end() && it_face != geom.objects.end())
    {
        const auto it_vbo = it_geom->second.attributes.find("vertex");
        const auto it_ibo = it_face->second.attributes.find("vertex_indices");

        if(all_found(geom.buffers, it_multi_texture_face, it_multi_texture_vertex) &&
                it_vbo != it_geom->second.attributes.end() &&
                it_ibo != it_face->second.attributes.end()
        ) {
            const auto it_uv_ibo = it_multi_texture_face->second.attributes.find("texture_vertex_indices");
            const auto it_tx = it_multi_texture_face->second.attributes.find("tx");
            const auto it_tn = it_multi_texture_face->second.attributes.find("tn");

            const auto it_u = it_multi_texture_vertex->second.attributes.find("u");
            const auto it_v = it_multi_texture_vertex->second.attributes.find("v");

            if(all_found(it_multi_texture_vertex->second.attributes, it_u, it_v) &&
                    it_uv_ibo != it_multi_texture_face->second.attributes.end()
            ) {
                // We're going to create a new vertex buffer to hold uv's too
                auto& orig_ibo = get<Image<uint32_t>>(it_ibo->second);
                const auto& orig_xyz = get<Image<float>>(it_vbo->second);
                const auto& uv_ibo = get<Image<uint32_t>>(it_uv_ibo->second);
                const auto& u = get<Image<float>>(it_u->second);
                const auto& v = get<Image<float>>(it_v->second);
                const auto& tx = get<Image<uint8_t>>(it_tx->second);
                const auto& tn = get<Image<uint32_t>>(it_tn->second);

                PANGO_ASSERT(u.h == v.h);
                PANGO_ASSERT(orig_ibo.w == 3 && uv_ibo.w == 3);

                pangolin::Geometry::Element new_xyzuv(5*sizeof(float), u.h);
                Image<float> new_xyz = new_xyzuv.UnsafeReinterpret<float>().SubImage(0,0,3,new_xyzuv.h);
                Image<float> new_uv = new_xyzuv.UnsafeReinterpret<float>().SubImage(3,0,2,new_xyzuv.h);
                new_xyzuv.attributes["vertex"] = new_xyz;
                new_xyzuv.attributes["uv"] = new_uv;

                for(size_t face=0; face < orig_ibo.h; ++face) {
                    uint32_t vtn = tn(0,face);
                    uint8_t vtx = tx(0,face);
                    PANGO_ASSERT(vtx==0, "Haven't implemented multi-texture yet.");

                    for(size_t vert=0; vert < 3; ++vert)
                    {
                        uint32_t& orig_xyz_index = orig_ibo(vert,vtn);
                        const uint32_t uv_index = uv_ibo(vert,face);
                        PANGO_ASSERT(uv_index < new_xyzuv.h && orig_xyz_index < orig_xyz.h);

                        for(int el=0; el < 3; ++el) {
                            new_xyz(el,uv_index) = orig_xyz(el,orig_xyz_index);
                        }
                        new_uv(0,uv_index) = u(0,uv_index);
                        new_uv(1,uv_index) = v(0,uv_index);
                        orig_xyz_index = uv_index;
                    }
                }

                geom.buffers["geometry"] = std::move(new_xyzuv);
                geom.buffers.erase(it_multi_texture_face);
                geom.buffers.erase(it_multi_texture_vertex);
            }

        }
    }
}

void Standardize(pangolin::Geometry& geom)
{
    StandardizeXyzToVertex(geom);
    StandardizeRgbToColor(geom);
    StandardizeMultiTextureFaceToXyzuv(geom);
    AddVertexNormals(geom);
}

inline int ReadGlIntType(GLenum type, std::istream& is)
{
    // TODO: This seems really dodgey...
    // int may not be big enough and if the datatype is smaller will it be padded?
    int v = 0;
    is.read( (char*)&v, GlDataTypeBytes(type));
    return v;
}

inline void ReadInto(std::vector<unsigned char>& vec, std::istream& is, size_t num_bytes)
{
    const size_t current_size = vec.size();
    vec.resize(current_size + num_bytes);
    is.read((char*)vec.data() + current_size, num_bytes);
}

void ParsePlyLE(pangolin::Geometry& geom, PlyHeaderDetails& ply, std::istream& is)
{
    std::vector<uint8_t> buffer;

    for(auto& el : ply.elements) {
        pangolin::Geometry::Element geom_el;

        if(el.stride_bytes > 0) {
            // This will usually be the case for vertex buffers with a known number of attributes
            PANGO_ASSERT(el.num_items > 0);
            geom_el.Reinitialise(el.stride_bytes, el.num_items);
            is.read((char*)geom_el.ptr, geom_el.Area());
        }else {
            // This will generally be the case for face data (containing a list of vertex indices)

            // Reserve enough space for a list of quads
            buffer.clear();
            buffer.reserve(el.num_items * el.properties.size() * 4);

            for(int i=0; i< el.num_items; ++i) {
                size_t offset_bytes = 0;
                for(auto& prop : el.properties) {
                    if(prop.isList()) {
                        const int list_items = ReadGlIntType(prop.list_index_type, is);
                        if(prop.num_items == -1) {
                            prop.num_items = list_items;
                            prop.offset_bytes = offset_bytes;
                        }else{
                            PANGO_ASSERT(prop.num_items == list_items);
                        }
                    }
                    const size_t num_bytes = prop.num_items * GlDataTypeBytes(prop.type);
                    ReadInto(buffer, is, num_bytes);
                    offset_bytes += num_bytes;
                }
            }

            // Update element stride now we know
            el.stride_bytes = 0;
            for(auto& prop : el.properties) {
                el.stride_bytes += prop.num_items * GlDataTypeBytes(prop.type);
            }

            geom_el.Reinitialise(el.stride_bytes, el.num_items);
            PANGO_ASSERT(geom_el.SizeBytes() == buffer.size());
            std::memcpy(geom_el.ptr, buffer.data(), buffer.size());
        }

        for(auto& prop : el.properties) {
            geom_el.attributes[prop.name] = MakeAttribute(
                prop.type, el.num_items, prop.num_items, geom_el.ptr + prop.offset_bytes, geom_el.pitch
            );
        }
        if(el.name == "vertex") {
            geom.buffers["geometry"] = std::move(geom_el);
        }else if(el.name == "face") {
            geom.objects.emplace("default", std::move(geom_el));
        }else{
            geom.buffers[el.name] = std::move(geom_el);
        }
    }

    Standardize(geom);
}

void ParsePlyBE(pangolin::Geometry& /*geom*/, const PlyHeaderDetails& /*ply*/, std::istream& /*is*/)
{
    throw std::runtime_error("Not implemented.");
}

void AttachAssociatedTexturesPly(pangolin::Geometry& geom, const std::string& filename)
{
    // For PLY, texture names are generally implicit
    auto dot = filename.find_last_of('.');
    if(dot != filename.npos) {
        const std::string base = filename.substr(0, dot);
        for(int i=0; i < 10; ++i) {
            const std::string glob = FormatString("%_%.*", base, i);
            std::vector<std::string> file_vec;
            if(FilesMatchingWildcard(glob, file_vec)) {
                for(const auto& file : file_vec) {
                    try {
                        geom.textures[FormatString("texture_%",i)] = LoadImage(file);
                        break;
                    }catch(std::runtime_error&)
                    {
                    }
                }
            }
        }
    }
}

pangolin::Geometry LoadGeometryPly(const std::string& filename)
{
    std::ifstream bFile( filename.c_str(), std::ios::in | std::ios::binary );
    if( !bFile.is_open() ) throw std::runtime_error("Unable to open PLY file: " + filename);

    PlyHeaderDetails ply;
    ParsePlyHeader(ply, bFile);

    // Initialise geom object
    pangolin::Geometry geom;

    // Fill in geometry from file.
    if(ply.format == PlyFormat_ascii) {
        ParsePlyAscii(geom, ply, bFile);
    }else if(ply.format == PlyFormat_binary_little_endian) {
        ParsePlyLE(geom, ply, bFile);
    }else if(ply.format == PlyFormat_binary_big_endian) {
        ParsePlyBE(geom, ply, bFile);
    }

    AttachAssociatedTexturesPly(geom, filename);

    return geom;
}

}
