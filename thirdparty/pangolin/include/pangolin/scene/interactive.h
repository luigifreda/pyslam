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

#include <pangolin/gl/glplatform.h>
#include <pangolin/display/opengl_render_state.h>

namespace pangolin {


struct Interactive
{
    static __thread GLuint current_id;

    virtual ~Interactive() {}

    virtual bool Mouse(
        int button,
        const GLprecision win[3], const GLprecision obj[3], const GLprecision normal[3],
        bool pressed, int button_state, int pickId
    )  = 0;

    virtual bool MouseMotion(
        const GLprecision win[3], const GLprecision obj[3], const GLprecision normal[3],
        int button_state, int pickId
    ) = 0;
};

struct RenderParams
{
    RenderParams()
      : render_mode(GL_RENDER)
    {
    }

    GLint render_mode;
};

struct Manipulator : public Interactive
{
    virtual void Render(const RenderParams& params) = 0;
};

}
