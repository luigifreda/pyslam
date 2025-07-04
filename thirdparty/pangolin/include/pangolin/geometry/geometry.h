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

#ifndef PANGOLIN_GEOMETRY_H
#define PANGOLIN_GEOMETRY_H

#include <map>
#include <unordered_map>
#include <vector>
#include <pangolin/image/typed_image.h>
#include <pangolin/compat/variant.h>

#ifdef HAVE_EIGEN
#include <Eigen/Geometry>
#endif

namespace pangolin
{

struct Geometry
{
    struct Element : public ManagedImage<uint8_t> {
        Element() = default;
        Element(Element&&) = default;
        Element& operator=(Element&&) = default;

        Element(size_t stride_bytes, size_t num_elements)
            : ManagedImage<uint8_t>(stride_bytes, num_elements)
        {}

        using Attribute = variant<Image<float>,Image<uint32_t>,Image<uint16_t>,Image<uint8_t>>;
        // "vertex", "rgb", "normal", "uv", "tris", "quads", ...
        std::map<std::string, Attribute> attributes;
    };

    // Store vertices and attributes
    std::map<std::string, Element> buffers;
    // Stores index buffers for each sub-object
    std::multimap<std::string, Element> objects;
    // Stores pixmaps
    std::map<std::string, TypedImage> textures;
};

pangolin::Geometry::Element::Attribute MakeAttribute(uint32_t gldatatype, size_t num_items, size_t count_per_item, void* ptr, size_t pitch_bytes);

pangolin::Geometry LoadGeometry(const std::string& filename);

#ifdef HAVE_EIGEN
inline Eigen::AlignedBox3f GetAxisAlignedBox(const Geometry& geom)
{
    Eigen::AlignedBox3f box;
    box.setEmpty();

    for(const auto& b : geom.buffers) {
        const auto& it_vert = b.second.attributes.find("vertex");
        if(it_vert != b.second.attributes.end()) {
            const Image<float>& vs = get<Image<float>>(it_vert->second);
            for(size_t i=0; i < vs.h; ++i) {
                const Eigen::Map<const Eigen::Vector3f> v(vs.RowPtr(i));
                box.extend(v);
            }
        }
    }

    return box;
}
#endif

}

#endif // PANGOLIN_GEOMETRY_H
