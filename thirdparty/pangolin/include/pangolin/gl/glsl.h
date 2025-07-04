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

#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <map>
#include <cctype>

#include <pangolin/gl/glplatform.h>
#include <pangolin/gl/colour.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/display/opengl_render_state.h>

#ifdef HAVE_GLES
    #define GLhandleARB GLuint
#endif

#if defined(HAVE_EIGEN) && !defined(__CUDACC__) //prevent including Eigen in cuda files
#define USE_EIGEN
#endif

#ifdef USE_EIGEN
#include <Eigen/Core>
#endif // USE_EIGEN

namespace pangolin
{

////////////////////////////////////////////////
// Standard attribute locations
////////////////////////////////////////////////

const GLuint DEFAULT_LOCATION_POSITION = 0;
const GLuint DEFAULT_LOCATION_COLOUR   = 1;
const GLuint DEFAULT_LOCATION_NORMAL   = 2;
const GLuint DEFAULT_LOCATION_TEXCOORD = 3;

const char DEFAULT_NAME_POSITION[] = "a_position";
const char DEFAULT_NAME_COLOUR[]   = "a_color";
const char DEFAULT_NAME_NORMAL[]   = "a_normal";
const char DEFAULT_NAME_TEXCOORD[] = "a_texcoord";

////////////////////////////////////////////////
// Interface
////////////////////////////////////////////////

enum GlSlShaderType
{
    GlSlAnnotatedShader = 0,
    GlSlFragmentShader = GL_FRAGMENT_SHADER,
    GlSlVertexShader = GL_VERTEX_SHADER,
    GlSlGeometryShader = 0x8DD9 /*GL_GEOMETRY_SHADER*/,
    GlSlComputeShader = 0x91B9 /*GL_COMPUTE_SHADER*/
};

class GlSlProgram
{
public:
    GlSlProgram();

    //! Move Constructor
    GlSlProgram(GlSlProgram&& tex);

    ~GlSlProgram();
    
    bool AddShader(
        GlSlShaderType shader_type,
        const std::string& source_code,
        const std::map<std::string,std::string>& program_defines = std::map<std::string,std::string>(),
        const std::vector<std::string>& search_path = std::vector<std::string>()
    );

    bool AddShaderFromFile(
        GlSlShaderType shader_type,
        const std::string& filename,
        const std::map<std::string,std::string>& program_defines = std::map<std::string,std::string>(),
        const std::vector<std::string>& search_path = std::vector<std::string>()
    );

    bool Link();

    // Remove all shaders from this program, and reload from files.
    bool ReloadShaderFiles();
    
    GLint GetAttributeHandle(const std::string& name);
    GLint GetUniformHandle(const std::string& name);

    // Before setting uniforms, be sure to Bind() the GlSl program first.
    void SetUniform(const std::string& name, int x);
    void SetUniform(const std::string& name, int x1, int x2);
    void SetUniform(const std::string& name, int x1, int x2, int x3);
    void SetUniform(const std::string& name, int x1, int x2, int x3, int x4);

    void SetUniform(const std::string& name, float f);
    void SetUniform(const std::string& name, float f1, float f2);
    void SetUniform(const std::string& name, float f1, float f2, float f3);
    void SetUniform(const std::string& name, float f1, float f2, float f3, float f4);

    void SetUniform(const std::string& name, Colour c);

    void SetUniform(const std::string& name, const OpenGlMatrix& m);

#ifdef HAVE_EIGEN
    void SetUniform(const std::string& name, const Eigen::Matrix3f& m);
    void SetUniform(const std::string& name, const Eigen::Matrix4f& m);
    void SetUniform(const std::string& name, const Eigen::Matrix3d& m);
    void SetUniform(const std::string& name, const Eigen::Matrix4d& m);
#endif

#if GL_VERSION_4_3
    GLint GetProgramResourceIndex(const std::string& name);
    void SetShaderStorageBlock(const std::string& name, const int& bindingIndex);
#endif

    void Bind();
    void SaveBind();
    void Unbind();


    void BindPangolinDefaultAttribLocationsAndLink();

    // Unlink all shaders from program
    void ClearShaders();

    GLint ProgramId() const {
        return prog;
    }

    bool Valid() const {
        return ProgramId() != 0;
    }

protected:
    struct ShaderFileOrCode
    {
        GlSlShaderType shader_type;
        std::string filename;
        std::string code;
        std::map<std::string,std::string> program_defines;
        std::vector<std::string> search_path;
    };


    // Convenience method to load shader description
    bool AddShaderFile(const ShaderFileOrCode &shader_file);

    std::string ParseIncludeFilename(
        const std::string& location
    );

    std::string SearchIncludePath(
        const std::string& filename,
        const std::vector<std::string>& search_path,
        const std::string& current_path
    );

    bool AddPreprocessedShader(
        GlSlShaderType shader_type,
        const std::string& source_code,
        const std::string& name_for_errors
    );

    void PreprocessGLSL(
        std::istream& input,
        std::ostream& output,
        const std::map<std::string,std::string>& program_defines,
        const std::vector<std::string>& search_path,
        const std::string& current_path
    );

    // Split 'code' into several code blocks per shader type
    // shader blocks in 'code' must be annotated with:
    // @start vertex, @start fragment, @start geometry or @start compute
    static std::map<GlSlShaderType,std::string>
    SplitAnnotatedShaders(const std::string& code);

    bool linked;
    std::vector<GLhandleARB> shaders;
    GLenum prog;
    GLint prev_prog;
    std::vector<ShaderFileOrCode> shader_files;
};

class GlSlUtilities
{
public:
    inline static GlSlProgram& OffsetAndScale(float offset, float scale) {
        GlSlProgram& prog = Instance().prog_offsetscale;
        prog.Bind();
        prog.SetUniform("offset", offset);
        prog.SetUniform("scale",  scale);
        return prog;
    }

    inline static GlSlProgram& Scale(float scale, float bias = 0.0f) {
        GlSlProgram& prog = Instance().prog_scale;
        prog.Bind();
        prog.SetUniform("scale", scale);
        prog.SetUniform("bias",  bias);
        return prog;
    }
    
    inline static void UseNone()
    {
        glUseProgram(0);
    }
    
protected:
    static GlSlUtilities& Instance() {
        // TODO: BUG: The GlSLUtilities instance needs to be tied
        // to the OpenGL context, not the thread, or globally.
#ifndef PANGO_NO_THREADLOCAL
        thread_local
#else
        static
#endif
        GlSlUtilities instance;
        return instance;
    }
    
    // protected constructor
    GlSlUtilities() {
        const char* source_scale =
                "uniform float scale;"
                "uniform float bias;"
                "uniform sampler2D tex;"
                "void main() {"
                "  vec2 uv = gl_TexCoord[0].st;"
                "  if(0.0 <= uv.x && uv.x <= 1.0 && 0.0 <= uv.y && uv.y <= 1.0) {"
                "    gl_FragColor = texture2D(tex,uv);"
                "    gl_FragColor.xyz *= scale;"
                "    gl_FragColor.xyz += vec3(bias,bias,bias);"
                "  }else{"
                "    float v = 0.1;"
                "    gl_FragColor.xyz = vec3(v,v,v);"
                "  }"
                "}";
        prog_scale.AddShader(GlSlFragmentShader, source_scale);
        prog_scale.Link();

        const char* source_offsetscale =
                "uniform float offset;"
                "uniform float scale;"
                "uniform sampler2D tex;"
                "void main() {"
                "  vec2 uv = gl_TexCoord[0].st;"
                "  if(0.0 <= uv.x && uv.x <= 1.0 && 0.0 <= uv.y && uv.y <= 1.0) {"
                "    gl_FragColor = texture2D(tex,gl_TexCoord[0].st);"
                "    gl_FragColor.xyz += vec3(offset,offset,offset);"
                "    gl_FragColor.xyz *= scale;"
                "  }else{"
                "    float v = 0.1;"
                "    gl_FragColor.xyz = vec3(v,v,v);"
                "  }"
                "}";
        prog_offsetscale.AddShader(GlSlFragmentShader, source_offsetscale);
        prog_offsetscale.Link();
    }
    
    GlSlProgram prog_scale;
    GlSlProgram prog_offsetscale;
};


////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////

inline bool IsLinkSuccessPrintLog(GLhandleARB prog)
{
    GLint status;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if(status != GL_TRUE) {
        pango_print_error("GLSL Program link failed: ");
        const int PROGRAM_LOG_MAX_LEN = 10240;
        char infolog[PROGRAM_LOG_MAX_LEN];
        GLsizei len;
        glGetProgramInfoLog(prog, PROGRAM_LOG_MAX_LEN, &len, infolog);
        if(len) {
            pango_print_error("%s\n",infolog);
        }else{
            pango_print_error("No details provided.\n");
        }
        return false;
    }
    return true;
}

inline bool IsCompileSuccessPrintLog(GLhandleARB shader, const std::string& name_for_errors)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if(status != GL_TRUE) {
        pango_print_error("GLSL Shader compilation failed: ");
        const int SHADER_LOG_MAX_LEN = 10240;
        char infolog[SHADER_LOG_MAX_LEN];
        GLsizei len;
        glGetShaderInfoLog(shader, SHADER_LOG_MAX_LEN, &len, infolog);
        if(len) {
            pango_print_error("%s:\n%s\n",name_for_errors.c_str(), infolog);
        }else{
            pango_print_error("%s:\nNo details provided.\n",name_for_errors.c_str());
        }
        return false;
    }
    return true;
}

inline GlSlProgram::GlSlProgram()
    : linked(false), prog(0), prev_prog(0)
{
}

//! Move Constructor
inline GlSlProgram::GlSlProgram(GlSlProgram&& o)
    : linked(o.linked), shaders(o.shaders), prog(o.prog), prev_prog(o.prev_prog)
{
    o.prog = 0;
}

inline GlSlProgram::~GlSlProgram()
{
    if(prog) {
        ClearShaders();
        glDeleteProgram(prog);
    }
}

inline void PrintSourceCode(const std::string& src)
{
    std::stringstream ss(src);
    std::string line;

    for(int linenum=1; std::getline(ss,line,'\n'); ++linenum) {
        std::cout << linenum << ":\t" << line << std::endl;
    }
}

inline bool GlSlProgram::AddPreprocessedShader(
    GlSlShaderType shader_type,
    const std::string& source_code,
    const std::string& name_for_errors
) {
    if(!prog) {
        prog = glCreateProgram();
    }

//    PrintSourceCode(source_code);

    GLhandleARB shader = glCreateShader(shader_type);
    const char* source = source_code.c_str();
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    bool success = IsCompileSuccessPrintLog(shader, name_for_errors);
    if(success) {
        glAttachShader(prog, shader);
        shaders.push_back(shader);
        linked = false;
    }
    return success;
}

inline std::string GlSlProgram::ParseIncludeFilename(const std::string& location)
{
    size_t start = location.find_first_of("\"<");
    if(start != std::string::npos) {
        size_t end = location.find_first_of("\">", start+1);
        if(end != std::string::npos) {
            return location.substr(start+1, end - start - 1);
        }
    }
    throw std::runtime_error("GLSL Parser: Unable to parse include location " + location );
}

inline std::string GlSlProgram::SearchIncludePath(
    const std::string& filename,
    const std::vector<std::string>& search_path,
    const std::string& current_path
) {
    if(FileExists(current_path + "/" + filename)) {
        return current_path + "/" + filename;
    }else{
        for(size_t i=0; i < search_path.size(); ++i) {
            const std::string hypoth = search_path[i] + "/" + filename;
            if( FileExists(hypoth) ) {
                return hypoth;
            }
        }
    }
    return "";
}

inline void GlSlProgram::PreprocessGLSL(
        std::istream& input, std::ostream& output,
        const std::map<std::string,std::string>& program_defines,
        const std::vector<std::string> &search_path,
        const std::string &current_path
) {
    const size_t MAXLINESIZE = 10240;
    char line[MAXLINESIZE] = "";

    while(!input.eof()) {
        // Take like from source
        input.getline(line,MAXLINESIZE);

        // Transform
        if( !strncmp(line, "#include", 8 ) ) {
            // C++ / G3D style include directive
            const std::string import_file = ParseIncludeFilename(line+8);
            const std::string resolved_file = SearchIncludePath(import_file, search_path, current_path);

            std::ifstream ifs(resolved_file.c_str());
            if(ifs.good()) {
                const std::string file_path = pangolin::PathParent(resolved_file);
                PreprocessGLSL(ifs, output, program_defines, search_path, file_path);
            }else{
                throw std::runtime_error("GLSL Parser: Unable to open " + import_file );
            }
        }else if( !strncmp(line, "#expect", 7) ) {
            // G3D style 'expect' directive, annotating expected preprocessor
            // definition with document string

            // Consume whitespace before token
            size_t token_start = 7;
            while( std::isspace(line[token_start]) ) ++token_start;

            // Iterate over contigous charecters until \0 or whitespace
            size_t token_end = token_start;
            while( line[token_end] && !std::isspace(line[token_end]) ) ++token_end;

            std::string token(line+token_start, line+token_end);
            std::map<std::string,std::string>::const_iterator it = program_defines.find(token);
            if( it == program_defines.end() ) {
                pango_print_warn("Expected define missing (defaulting to 0): '%s'\n%s\n", token.c_str(), line + token_end );
                output << "#define " << token << " 0" << std::endl;
            }else{
                output << "#define " << token << " " << it->second << std::endl;
            }
        }else{
            // Output directly
            output << line << std::endl;
        }
    }
}

inline void GlSlProgram::ClearShaders()
{
    // Remove and delete each shader
    for(size_t i=0; i<shaders.size(); ++i ) {
        glDetachShader(prog, shaders[i]);
        glDeleteShader(shaders[i]);
    }
    shaders.clear();
}

inline bool GlSlProgram::AddShaderFile(const ShaderFileOrCode& shader_file)
{
    std::stringstream buffer;

    if(shader_file.code.empty()) {
        std::ifstream ifs(shader_file.filename.c_str());
        if(ifs.is_open()) {
            PreprocessGLSL(ifs, buffer, shader_file.program_defines, shader_file.search_path, ".");
        }else{
            throw std::runtime_error(FormatString("Unable to open shader file '%'", shader_file.filename));
        }
    }else{
        std::istringstream iss(shader_file.code);
        PreprocessGLSL(iss, buffer, shader_file.program_defines, shader_file.search_path, ".");
    }

    const std::string code = buffer.str();
    const std::string input_name = !shader_file.filename.empty() ? shader_file.filename : "<string>";

    if(shader_file.shader_type == GlSlAnnotatedShader) {
        const std::map<GlSlShaderType,std::string> split_progs = SplitAnnotatedShaders(code);
        for(const auto& type_code : split_progs) {
            if(!AddPreprocessedShader(type_code.first, type_code.second, input_name )) {
                return false;
            }
        }
        return true;
    }else{
        return AddPreprocessedShader(shader_file.shader_type, code, input_name);
    }
}

inline bool GlSlProgram::AddShaderFromFile(
    GlSlShaderType shader_type,
    const std::string& filename,
    const std::map<std::string,std::string>& program_defines,
    const std::vector<std::string>& search_path
) {
    ShaderFileOrCode shader_file = {
        shader_type,
        pangolin::PathExpand(filename),
        std::string(),
        program_defines,
        search_path
    };
    shader_files.push_back(shader_file);
    return AddShaderFile(shader_file);
}

inline bool GlSlProgram::AddShader(
    GlSlShaderType shader_type,
    const std::string& source_code,
    const std::map<std::string,std::string>& program_defines,
    const std::vector<std::string>& search_path
) {
    ShaderFileOrCode shader_file = {
        shader_type,
        std::string(),
        source_code,
        program_defines,
        search_path
    };
    shader_files.push_back(shader_file);
    return AddShaderFile(shader_file);
}

inline bool GlSlProgram::ReloadShaderFiles()
{
    ClearShaders();

    for(const auto& sf : shader_files) {
        if(!AddShaderFile(sf)) {
            return false;
        }
    }

    Link();
    return true;
}

inline std::map<GlSlShaderType,std::string>
GlSlProgram::SplitAnnotatedShaders(const std::string& code)
{
    std::map<GlSlShaderType,std::string> ret;

    std::stringstream input(code);
    std::stringstream output;

    const size_t MAXLINESIZE = 10240;
    char line[MAXLINESIZE];

    GlSlShaderType current_type = GlSlAnnotatedShader;
    auto finish_block = [&](GlSlShaderType type){
        if(current_type != GlSlAnnotatedShader) {
            ret[current_type] = output.str();
        }
        output.str(std::string());
        current_type = type;
    };

    while(!input.eof()) {
        // Take like from source
        input.getline(line,MAXLINESIZE);

        // Transform
        if( !strncmp(line, "@start", 6 ) ) {
            const std::string str_shader_type = pangolin::Trim(std::string(line).substr(6));
            if(str_shader_type == "vertex") {
                finish_block(GlSlVertexShader);
            }else if(str_shader_type == "fragment") {
                finish_block(GlSlFragmentShader);
            }else if(str_shader_type == "geometry") {
                finish_block(GlSlGeometryShader);
            }else if(str_shader_type == "compute") {
                finish_block(GlSlComputeShader);
            }
        }else{
            output << line << std::endl;
        }
    }

    finish_block(GlSlAnnotatedShader);

    return ret;
}

inline bool GlSlProgram::Link()
{
    glLinkProgram(prog);
    return IsLinkSuccessPrintLog(prog);
}

inline void GlSlProgram::Bind()
{
    prev_prog = 0;
    glUseProgram(prog);
}

inline void GlSlProgram::SaveBind()
{
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    glUseProgram(prog);
}

inline void GlSlProgram::Unbind()
{
    glUseProgram(prev_prog);
}

inline GLint GlSlProgram::GetAttributeHandle(const std::string& name)
{
    return glGetAttribLocation(prog, name.c_str());
}

inline GLint GlSlProgram::GetUniformHandle(const std::string& name)
{
    return glGetUniformLocation(prog, name.c_str());
}

inline void GlSlProgram::SetUniform(const std::string& name, int x)
{
    glUniform1i( GetUniformHandle(name), x);
}

inline void GlSlProgram::SetUniform(const std::string& name, int x1, int x2)
{
    glUniform2i( GetUniformHandle(name), x1, x2);
}

inline void GlSlProgram::SetUniform(const std::string& name, int x1, int x2, int x3)
{
    glUniform3i( GetUniformHandle(name), x1, x2, x3);
}

inline void GlSlProgram::SetUniform(const std::string& name, int x1, int x2, int x3, int x4)
{
    glUniform4i( GetUniformHandle(name), x1, x2, x3, x4);
}

inline void GlSlProgram::SetUniform(const std::string& name, float f)
{
    glUniform1f( GetUniformHandle(name), f);
}

inline void GlSlProgram::SetUniform(const std::string& name, float f1, float f2)
{
    glUniform2f( GetUniformHandle(name), f1,f2);
}

inline void GlSlProgram::SetUniform(const std::string& name, float f1, float f2, float f3)
{
    glUniform3f( GetUniformHandle(name), f1,f2,f3);
}

inline void GlSlProgram::SetUniform(const std::string& name, float f1, float f2, float f3, float f4)
{
    glUniform4f( GetUniformHandle(name), f1,f2,f3,f4);
}

inline void GlSlProgram::SetUniform(const std::string& name, Colour c)
{
    glUniform4f( GetUniformHandle(name), c.r, c.g, c.b, c.a);
}

inline void GlSlProgram::SetUniform(const std::string& name, const OpenGlMatrix& mat)
{
    // glUniformMatrix4dv seems to be crashing...
    float m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = (float)mat.m[i];
    }
    glUniformMatrix4fv( GetUniformHandle(name), 1, GL_FALSE, m);
}

#ifdef HAVE_EIGEN
inline void GlSlProgram::SetUniform(const std::string& name, const Eigen::Matrix3f& m)
{
    glUniformMatrix3fv( GetUniformHandle(name), 1, GL_FALSE, m.data());
}
inline void GlSlProgram::SetUniform(const std::string& name, const Eigen::Matrix4f& m)
{
    glUniformMatrix4fv( GetUniformHandle(name), 1, GL_FALSE, m.data());
}
inline void GlSlProgram::SetUniform(const std::string& name, const Eigen::Matrix3d& m)
{
    glUniformMatrix3dv( GetUniformHandle(name), 1, GL_FALSE, m.data());
}
inline void GlSlProgram::SetUniform(const std::string& name, const Eigen::Matrix4d& m)
{
    glUniformMatrix4dv( GetUniformHandle(name), 1, GL_FALSE, m.data());
}
#endif

inline void GlSlProgram::BindPangolinDefaultAttribLocationsAndLink()
{
    glBindAttribLocation(prog, DEFAULT_LOCATION_POSITION, DEFAULT_NAME_POSITION);
    glBindAttribLocation(prog, DEFAULT_LOCATION_COLOUR,   DEFAULT_NAME_COLOUR);
    glBindAttribLocation(prog, DEFAULT_LOCATION_NORMAL,   DEFAULT_NAME_NORMAL);
    glBindAttribLocation(prog, DEFAULT_LOCATION_TEXCOORD, DEFAULT_NAME_TEXCOORD);
    Link();
}

#if GL_VERSION_4_3
inline GLint GlSlProgram::GetProgramResourceIndex(const std::string& name)
{
    return glGetProgramResourceIndex(prog, GL_SHADER_STORAGE_BLOCK, name.c_str());
}

inline void GlSlProgram::SetShaderStorageBlock(const std::string& name, const int& bindingIndex)
{
    glShaderStorageBlockBinding(prog, GetProgramResourceIndex(name), bindingIndex);
}
#endif

}
