/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
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

#include <memory>
#include <map>
#include <random>

#include <pangolin/display/opengl_render_state.h>
#include <pangolin/scene/interactive.h>

namespace pangolin {

class Renderable
{
public:
    using guid_t = GLuint;

    static guid_t UniqueGuid()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return (guid_t)gen();
    }

    Renderable(const std::weak_ptr<Renderable>& parent = std::weak_ptr<Renderable>())
        : guid(UniqueGuid()), parent(parent), T_pc(IdentityMatrix()), should_show(true)
    {
    }

    virtual ~Renderable()
    {
    }

    // Default implementation simply renders children.
    virtual void Render(const RenderParams& params = RenderParams()) {
        RenderChildren(params);
    }

    void RenderChildren(const RenderParams& params)
    {
        for(auto& p : children) {
            Renderable& r = *p.second;
            if(r.should_show) {
                glPushMatrix();
                r.T_pc.Multiply();
                r.Render(params);
                if(r.manipulator) {
                    r.manipulator->Render(params);
                }
                glPopMatrix();
            }
        }
    }

    std::shared_ptr<Renderable> FindChild(guid_t guid)
    {
        auto o = children.find(guid);
        if(o != children.end()) {
            return o->second;
        }

        for(auto& kv : children ) {
            std::shared_ptr<Renderable> c = kv.second->FindChild(guid);
            if(c) return c;
        }

        return std::shared_ptr<Renderable>();
    }

    Renderable& Add(const std::shared_ptr<Renderable>& child)
    {
        if(child) {
            children[child->guid] = child;
        };
        return *this;
    }

    // Renderable properties
    const guid_t guid;
    std::weak_ptr<Renderable> parent;
    pangolin::OpenGlMatrix T_pc;
    std::shared_ptr<Renderable> child;
    bool should_show;

    // Children
    std::map<guid_t, std::shared_ptr<Renderable>> children;

    // Manipulator (handler, thing)
    std::shared_ptr<Manipulator> manipulator;
};

}
