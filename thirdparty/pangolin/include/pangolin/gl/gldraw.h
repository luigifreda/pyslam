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

#include <pangolin/gl/glinclude.h>
#include <pangolin/gl/glformattraits.h>
#include <pangolin/display/opengl_render_state.h>

#include <vector>
#include <math.h>

#if defined(HAVE_EIGEN) && !defined(__CUDACC__) //prevent including Eigen in cuda files
#define USE_EIGEN
#endif

#ifdef USE_EIGEN
#include <Eigen/Core>
#include <Eigen/src/Geometry/AlignedBox.h>
#endif // USE_EIGEN

namespace pangolin
{

// h [0,360)
// s [0,1]
// v [0,1]
inline void glColorHSV( GLfloat hue, GLfloat s=1.0f, GLfloat v=1.0f )
{
    const GLfloat h = hue / 60.0f;
    const int i = (int)floor(h);
    const GLfloat f = (i%2 == 0) ? 1-(h-i) : h-i;
    const GLfloat m = v * (1-s);
    const GLfloat n = v * (1-s*f);
    switch(i)
    {
    case 0: glColor4f(v,n,m,1); break;
    case 1: glColor4f(n,v,m,1); break;
    case 2: glColor4f(m,v,n,1); break;
    case 3: glColor4f(m,n,v,1); break;
    case 4: glColor4f(n,m,v,1); break;
    case 5: glColor4f(v,m,n,1); break;
    default:
        break;
    }
}

inline void glColorBin( int bin, int max_bins, GLfloat sat=1.0f, GLfloat val=1.0f )
{
    if( bin >= 0 ) {
        const GLfloat hue = (GLfloat)(bin%max_bins) * 360.0f / (GLfloat)max_bins;
        glColorHSV(hue,sat,val);
    }else{
        glColor4f(1,1,1,1);
    }
}

template<typename T>
inline void glDrawVertices(
    size_t num_vertices, const T* const vertex_ptr, GLenum mode,
    size_t elements_per_vertex = GlFormatTraits<T>::components,
    size_t vertex_stride_bytes = 0 )
{
    if(num_vertices > 0)
    {
        PANGO_ENSURE(vertex_ptr != nullptr);
        PANGO_ENSURE(mode != GL_LINES || num_vertices % 2 == 0, "number of vertices (%) must be even in GL_LINES mode", num_vertices );

        glVertexPointer(elements_per_vertex, GlFormatTraits<T>::gltype, vertex_stride_bytes, vertex_ptr);
        glEnableClientState(GL_VERTEX_ARRAY);
        glDrawArrays(mode, 0, num_vertices);
        glDisableClientState(GL_VERTEX_ARRAY);
    }
}

template<typename TV, typename TC>
inline void glDrawColoredVertices(
    size_t num_vertices, const TV* const vertex_ptr, const TC* const color_ptr, GLenum mode,
    size_t elements_per_vertex = GlFormatTraits<TV>::components,
    size_t elements_per_color = GlFormatTraits<TC>::components,
    size_t vertex_stride_bytes = 0,
    size_t color_stride_bytes = 0
) {
    if(color_ptr) {
        glColorPointer(elements_per_color, GlFormatTraits<TC>::gltype, color_stride_bytes, color_ptr);
        glEnableClientState(GL_COLOR_ARRAY);
        glDrawVertices<TV>(num_vertices, vertex_ptr, mode, elements_per_vertex, vertex_stride_bytes);
        glDisableClientState(GL_COLOR_ARRAY);
    }else{
        glDrawVertices<TV>(num_vertices, vertex_ptr, mode, elements_per_vertex, vertex_stride_bytes);
    }
}

inline void glDrawLine( GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2 )
{
    const GLfloat verts[] = { x1,y1,  x2,y2 };
    glDrawVertices<float>(2, verts, GL_LINES, 2);
}

inline void glDrawLine( GLfloat x1, GLfloat y1, GLfloat z1, GLfloat x2, GLfloat y2, GLfloat z2)
{
    const GLfloat verts[] = { x1,y1,z1,  x2,y2,z2 };
    glDrawVertices<float>(2, verts, GL_LINES, 3);
}

inline void glDrawCross( GLfloat x, GLfloat y, GLfloat rad )
{
    const GLfloat verts[] = { x-rad,y, x+rad, y, x,y-rad, x, y+rad};
    glDrawVertices<float>(4, verts, GL_LINES, 2);
}

inline void glDrawCross( GLfloat x, GLfloat y, GLfloat z, GLfloat rad )
{
    const GLfloat verts[] = { x-rad,y,z, x+rad,y,z, x,y-rad,z, x,y+rad,z, x,y,z-rad, x,y,z+rad };
    glDrawVertices<float>(6, verts, GL_LINES, 3);
}

inline void glDrawAxis(float s)
{
    const GLfloat cols[]  = { 1,0,0, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1 };
    const GLfloat verts[] = { 0,0,0, s,0,0, 0,0,0, 0,s,0, 0,0,0, 0,0,s };
    glDrawColoredVertices<float,float>(6, verts, cols, GL_LINES, 3, 3);
}

inline void glDrawRect( GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2, GLenum mode = GL_TRIANGLE_FAN )
{
    const GLfloat verts[] = { x1,y1,  x2,y1,  x2,y2,  x1,y2 };
    glDrawVertices<float>(4, verts, mode, 2);
}

inline void glDrawRectPerimeter( GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2 )
{
    glDrawRect(x1,y1, x2,y2, GL_LINE_LOOP);
}

inline void glDrawCirclePerimeter( float x, float y, float rad )
{
    const int N = 50;
    GLfloat verts[N*2];

    const float TAU_DIV_N = 2*(float)M_PI/N;
    for(int i = 0; i < N*2; i+=2) {
        verts[i] =   x + rad * cos(i*TAU_DIV_N);
        verts[i+1] = y + rad * sin(i*TAU_DIV_N);
    }

    glDrawVertices<float>(N, verts, GL_LINES, 2);
}

inline void glDrawCircle( GLfloat x, GLfloat y, GLfloat rad )
{    
    const int N = 50;
    GLfloat verts[N*2];
    
    // Draw vertices anticlockwise for front face
    const float TAU_DIV_N = 2*(float)M_PI/N;
    for(int i = 0; i < N*2; i+=2) {
        verts[i] =   x + rad * cos(-i*TAU_DIV_N);
        verts[i+1] = y + rad * sin(-i*TAU_DIV_N);
    }
    
    // Render filled shape and outline (to make it look smooth)
    glVertexPointer(2, GL_FLOAT, 0, verts);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_TRIANGLE_FAN, 0, N);
    glDrawArrays(GL_LINE_STRIP, 0, N);
    glDisableClientState(GL_VERTEX_ARRAY);
}

inline void glDrawColouredCube(GLfloat axis_min=-0.5f, GLfloat axis_max = +0.5f)
{
    const GLfloat l = axis_min;
    const GLfloat h = axis_max;
    
    const GLfloat verts[] = {
        l,l,h,  h,l,h,  l,h,h,  h,h,h,  // FRONT
        l,l,l,  l,h,l,  h,l,l,  h,h,l,  // BACK
        l,l,h,  l,h,h,  l,l,l,  l,h,l,  // LEFT
        h,l,l,  h,h,l,  h,l,h,  h,h,h,  // RIGHT
        l,h,h,  h,h,h,  l,h,l,  h,h,l,  // TOP
        l,l,h,  l,l,l,  h,l,h,  h,l,l   // BOTTOM
    };
    
    glVertexPointer(3, GL_FLOAT, 0, verts);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);
    
    glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 8, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 12, 4);
    
    glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
    glDrawArrays(GL_TRIANGLE_STRIP, 16, 4);
    glDrawArrays(GL_TRIANGLE_STRIP, 20, 4);
    
    glDisableClientState(GL_VERTEX_ARRAY);
}

inline void glDraw_x0(GLfloat scale, int grid)
{
    const GLfloat maxord = grid*scale;
    for (int i = -grid; i <= grid; ++i) {
        glDrawLine(0.0, i*scale, -maxord, 0.0, i*scale, +maxord);
        glDrawLine(0.0, -maxord, i*scale, 0.0, +maxord, i*scale);
    }
}

inline void glDraw_y0(GLfloat scale, int grid)
{
    const GLfloat maxord = grid*scale;
    for (int i = -grid; i <= grid; ++i) {
        glDrawLine(i*scale, 0.0, -maxord, i*scale, 0.0, +maxord);
        glDrawLine(-maxord, 0.0, i*scale, +maxord, 0.0, i*scale);
    }
}

inline void glDraw_z0(GLfloat scale, int grid)
{
    const GLfloat maxord = grid*scale;
    for(int i=-grid; i<=grid; ++i ) {
        glDrawLine(i*scale,-maxord,   i*scale,+maxord);
        glDrawLine(-maxord, i*scale,  +maxord, i*scale);
    }
}

inline void glDrawFrustum( GLfloat u0, GLfloat v0, GLfloat fu, GLfloat fv, int w, int h, GLfloat scale )
{
    const GLfloat xl = scale * u0;
    const GLfloat xh = scale * (w*fu + u0);
    const GLfloat yl = scale * v0;
    const GLfloat yh = scale * (h*fv + v0);

    const GLfloat verts[] = {
        xl,yl,scale,  xh,yl,scale,
        xh,yh,scale,  xl,yh,scale,
        xl,yl,scale,  0,0,0,
        xh,yl,scale,  0,0,0,
        xl,yh,scale,  0,0,0,
        xh,yh,scale
    };

    glDrawVertices(11, verts, GL_LINE_STRIP, 3);
}

inline void glDrawTexture(GLenum target, GLuint texid)
{
    glBindTexture(target, texid);
    glEnable(target);
    
    const GLfloat sq_vert[] = { -1,-1,  1,-1,  1, 1,  -1, 1 };
    glVertexPointer(2, GL_FLOAT, 0, sq_vert);
    glEnableClientState(GL_VERTEX_ARRAY);   

    const GLfloat sq_tex[]  = { 0,0,  1,0,  1,1,  0,1  };
    glTexCoordPointer(2, GL_FLOAT, 0, sq_tex);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
         
    glColor4f(1,1,1,1);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glDisable(target);
}

inline void glDrawTextureFlipY(GLenum target, GLuint texid)
{
    glBindTexture(target, texid);
    glEnable(target);
    
    const GLfloat sq_vert[] = { -1,-1,  1,-1,  1, 1,  -1, 1 };
    glVertexPointer(2, GL_FLOAT, 0, sq_vert);
    glEnableClientState(GL_VERTEX_ARRAY);   

    const GLfloat sq_tex[]  = { 0,1,  1,1,  1,0,  0,0  };
    glTexCoordPointer(2, GL_FLOAT, 0, sq_tex);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
         
    glColor4f(1,1,1,1);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);

    glDisable(target);
}


#ifdef USE_EIGEN

#ifndef HAVE_GLES
inline void glVertex( const Eigen::Vector3d& p )
{
    glVertex3dv(p.data());
}
#endif

inline void glDrawLine( const Eigen::Vector2d& p1, const Eigen::Vector2d& p2 )
{
    glDrawLine((GLfloat)p1(0), (GLfloat)p1(1), (GLfloat)p2(0), (GLfloat)p2(1));
}

// Draws a vector of 2d or 3d vertices using provided ``mode``.
//
// Preconditions:
//  - ``mode`` must be GL_POINTS, GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP, etc
//  - If ``mode == GL_LINES``, then ``vertices.size()`` must be a multiple of 2.
//
template<typename P, int N, class Allocator>
void glDrawVertices(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& vertices, GLenum mode)
{
    glDrawVertices(vertices.size(), vertices.data(), mode);
}

// Draws a vector of 2d or 3d points.
//
template<typename P, int N, class Allocator>
void glDrawPoints(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& vertices)
{
    glDrawVertices(vertices, GL_POINTS);
}

// Draws a vector of 2d or 3d lines.
//
//  Precondition: ``vertices.size()`` must be a multiple of 2.
//
template<typename P, int N, class Allocator>
void glDrawLines(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& vertices)
{
    glDrawVertices(vertices, GL_LINES);
}

// Draws a 2d or 3d line strip.
//
template<typename P, int N, class Allocator>
void glDrawLineStrip(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& vertices)
{
    glDrawVertices(vertices, GL_LINE_STRIP);
}

// Draws a 2d or 3d line loop.
//
template<typename P, int N, class Allocator>
void glDrawLineLoop(const std::vector<Eigen::Matrix<P, N, 1>, Allocator>& vertices)
{
    glDrawVertices(vertices, GL_LINE_LOOP);
}

inline void glDrawCross( const Eigen::Vector2d& p, double r = 5.0 )
{
    glDrawCross((GLfloat)p(0), (GLfloat)p(1), (GLfloat)r);
}

inline void glDrawCross( const Eigen::Vector3d& p, double r = 5.0 )
{
    glDrawCross((GLfloat)p(0), (GLfloat)p(1), (GLfloat)p(2), (GLfloat)r);
}

inline void glDrawCircle( const Eigen::Vector2d& p, double radius = 5.0 )
{
    glDrawCircle((GLfloat)p(0), (GLfloat)p(1), (GLfloat)radius);
}

inline void glDrawCirclePerimeter( const Eigen::Vector2d& p, double radius = 5.0 )
{
    glDrawCirclePerimeter((GLfloat)p(0), (GLfloat)p(1), (GLfloat)radius);
}

inline void glSetFrameOfReference( const Eigen::Matrix4f& T_wf )
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf( T_wf.data() );
}

inline void glSetFrameOfReference( const Eigen::Matrix4d& T_wf )
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
#ifndef HAVE_GLES
    glMultMatrixd( T_wf.data() );
#else
    const Eigen::Matrix4f fT_wf = T_wf.cast<GLfloat>();
    glMultMatrixf( fT_wf.data() );
#endif
}

inline void glSetFrameOfReference( const pangolin::OpenGlMatrix& T_wf )
{
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixd( T_wf.m );
}

inline void glUnsetFrameOfReference()
{
    glPopMatrix();
}

template<typename T, typename S>
inline void glDrawAxis( const T& T_wf, S scale )
{
    glSetFrameOfReference(T_wf);
    glDrawAxis(scale);
    glUnsetFrameOfReference();
}

template<typename T>
inline void glDrawFrustum( const Eigen::Matrix<T,3,3>& Kinv, int w, int h, GLfloat scale )
{
    glDrawFrustum((GLfloat)Kinv(0,2), (GLfloat)Kinv(1,2), (GLfloat)Kinv(0,0), (GLfloat)Kinv(1,1), w, h, scale);
}

template<typename T>
inline void glDrawFrustum( const Eigen::Matrix<T,3,3>& Kinv, int w, int h, const Eigen::Matrix<T,4,4>& T_wf, T scale )
{
    glSetFrameOfReference(T_wf);
    glDrawFrustum(Kinv,w,h,scale);
    glUnsetFrameOfReference();
}

template<typename T>
inline void glDrawAlignedBox( const Eigen::AlignedBox<T,2>& box, GLenum mode = GL_TRIANGLE_FAN )
{
    const Eigen::Matrix<float,2,1> l = box.min().template cast<float>();
    const Eigen::Matrix<float,2,1> h = box.max().template cast<float>();

    GLfloat verts[] = {
        l[0], l[1],
        h[0], l[1],
        h[0], h[1],
        l[0], h[1]
    };

    glDrawVertices(4, verts, mode, 2);
}

template<typename T>
inline void glDrawAlignedBoxPerimeter( const Eigen::AlignedBox<T,2>& box)
{
    glDrawAlignedBox<T>(box, GL_LINE_LOOP);
}

template<typename T>
inline void glDrawAlignedBox( const Eigen::AlignedBox<T,3>& box)
{
    const Eigen::Matrix<float,3,1> l = box.min().template cast<float>();
    const Eigen::Matrix<float,3,1> h = box.max().template cast<float>();

    GLfloat verts[] = {
        l[0], l[1], l[2],
        l[0], l[1], h[2],
        h[0], l[1], h[2],
        h[0], l[1], l[2],
        l[0], l[1], l[2],
        l[0], h[1], l[2],
        h[0], h[1], l[2],
        h[0], l[1], l[2],
        h[0], h[1], l[2],
        h[0], h[1], h[2],
        l[0], h[1], h[2],
        l[0], h[1], l[2],
        l[0], h[1], h[2],
        l[0], l[1], h[2],
        h[0], l[1], h[2],
        h[0], h[1], h[2]
    };

    glDrawVertices(16, verts, GL_LINE_STRIP, 3);
}

#endif // USE_EIGEN

#ifndef HAVE_GLES
inline void glPixelTransferScale( float r, float g, float b )
{
    glPixelTransferf(GL_RED_SCALE,r);
    glPixelTransferf(GL_GREEN_SCALE,g);
    glPixelTransferf(GL_BLUE_SCALE,b);
}

inline void glPixelTransferScale( float scale )
{
    glPixelTransferScale(scale,scale,scale);
}
#endif

void glRecordGraphic(float x, float y, float radius);

}
