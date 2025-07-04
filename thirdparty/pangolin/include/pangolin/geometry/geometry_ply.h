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

#include <pangolin/platform.h>
#include <pangolin/compat/variant.h>
#include <pangolin/geometry/geometry.h>

#include <fstream>
#include <vector>
#include <algorithm>

namespace pangolin
{

#define PLY_GROUP_LIST(m)  m(PlyHeader) m(PlyFormat) m(PlyType)
#define PLY_HEADER_LIST(m) m(ply) m(format) m(comment) m(property) m(element) m(end_header)
#define PLY_FORMAT_LIST(m) m(ascii) m(binary_big_endian) m(binary_little_endian)
#define PLY_TYPE_LIST(m)   m(char) m(int8) m(uchar) m(uint8) m(short) m(int16) m(ushort) m(uint16) m(int) m(int32) m(uint) m(uint32) m(float) m(float32) m(double) m(float64) m(list)

// Define Enums / strings
enum PlyHeader {
#define FORMAT_ENUM(x) PlyHeader_##x,
    PLY_HEADER_LIST(FORMAT_ENUM)
    PlyHeaderSize
#undef FORMAT_ENUM
};

enum PlyFormat {
#define FORMAT_ENUM(x) PlyFormat_##x,
    PLY_FORMAT_LIST(FORMAT_ENUM)
    PlyFormatSize
#undef FORMAT_ENUM
};

enum PlyType {
#define FORMAT_ENUM(x) PlyType_##x,
    PLY_TYPE_LIST(FORMAT_ENUM)
    PlyTypeSize
#undef FORMAT_ENUM
};
const size_t PlyTypeGl[] = {
//  char, int8 -> GL_BYTE
    0x1400, 0x1400,
//  uchar, uint8 -> GL_UNSIGNED_BYTE
    0x1401, 0x1401,
//  short, int16 -> GL_SHORT
    0x1402, 0x1402,
//  ushort, uint16 -> GL_UNSIGNED_SHORT
    0x1403, 0x1403,
//  int, int32 -> GL_INT
    0x1404, 0x1404,
//  uint, uint32 -> GL_UNSIGNED_INT
    0x1405, 0x1405,
//  float, float32 -> GL_FLOAT
    0x1406, 0x1406,
//  double, float64 -> GL_DOUBLE
    0x140A, 0x140A,
//  list -> GL_NONE
    0
};

#undef FORMAT_ENUM

struct PlyPropertyDetails
{
    std::string name;

    // Type of property (GLenum)
    size_t type;

    // Type of list index if a list, or 0 otherwise. (GLenum)
    size_t list_index_type;

    // Offset from element start
    size_t offset_bytes;

    // Number of items in the list. 1 if not a list. -1 if unknown.
    int num_items;

    bool isList() const {
        return list_index_type > 0;
    }
};

struct PlyElementDetails
{
    std::string name;
    int num_items;
    int stride_bytes;
    std::vector<PlyPropertyDetails> properties;

    inline std::vector<PlyPropertyDetails>::iterator FindProperty(const std::string& name)
    {
        return std::find_if(properties.begin(), properties.end(),
            [&name](const PlyPropertyDetails& p){ return p.name == name;}
        );
    }
};

struct PlyHeaderDetails
{
    PlyFormat format;
    std::string version;
    std::vector<PlyElementDetails> elements;

    inline std::vector<PlyElementDetails>::iterator FindElement(const std::string& name)
    {
        return std::find_if(elements.begin(), elements.end(),
            [&name](const PlyElementDetails& el){ return el.name == name;}
        );
    }
};

void ParsePlyHeader(PlyHeaderDetails& ply, std::istream& is);

struct PlyBuffer
{
    size_t index_size_bytes;
    size_t element_item_size_bytes;
    std::vector<unsigned char> data;
};

void ParsePlyAscii(pangolin::Geometry& /*geom*/, const PlyHeaderDetails& /*ply*/, std::istream& /*is*/);

// Convert Seperate "x","y","z" attributes into a single "vertex" attribute
void StandardizeXyzToVertex(pangolin::Geometry& geom);

// The Artec scanner saves with these attributes, for example
void StandardizeMultiTextureFaceToXyzuv(pangolin::Geometry& geom);

void Standardize(pangolin::Geometry& geom);

void ParsePlyLE(pangolin::Geometry& geom, PlyHeaderDetails& ply, std::istream& is);

void ParsePlyBE(pangolin::Geometry& /*geom*/, const PlyHeaderDetails& /*ply*/, std::istream& /*is*/);

void AttachAssociatedTexturesPly(pangolin::Geometry& geom, const std::string& filename);

pangolin::Geometry LoadGeometryPly(const std::string& filename);

}
