#pragma once

namespace pangolin {

const std::string default_model_shader = R"Shader(
/////////////////////////////////////////
@start vertex
#version 120

#expect SHOW_COLOR
#expect SHOW_NORMAL
#expect SHOW_TEXTURE
#expect SHOW_MATCAP
#expect SHOW_UV

    uniform mat4 T_cam_norm;
    uniform mat4 KT_cw;
    attribute vec3 vertex;

#if SHOW_COLOR
    attribute vec4 color;
    varying vec4 vColor;
    void main() {
        vColor = color;
#elif SHOW_NORMAL
    attribute vec3 normal;
    varying vec3 vNormal;
    void main() {
        vNormal = mat3(T_cam_norm) * normal;
#elif SHOW_TEXTURE
    attribute vec2 uv;
    varying vec2 vUV;
    void main() {
        vUV = uv;
#elif SHOW_MATCAP
    attribute vec3 normal;
    varying vec3 vNormalCam;
    void main() {
        vNormalCam = mat3(T_cam_norm) * normal;
#elif SHOW_UV
    attribute vec2 uv;
    varying vec2 vUV;
    void main() {
        vUV = uv;
#else
    varying vec3 vP;
    void main() {
        vP = vertex;
#endif
        gl_Position = KT_cw * vec4(vertex, 1.0);
    }

/////////////////////////////////////////
@start fragment
#version 120
#expect SHOW_COLOR
#expect SHOW_NORMAL
#expect SHOW_TEXTURE
#expect SHOW_MATCAP
#expect SHOW_UV

#if SHOW_COLOR
    varying vec4 vColor;
#elif SHOW_NORMAL
    varying vec3 vNormal;
#elif SHOW_TEXTURE
    varying vec2 vUV;
    uniform sampler2D texture_0;
#elif SHOW_MATCAP
    varying vec3 vNormalCam;
    uniform sampler2D matcap;
#elif SHOW_UV
    varying vec2 vUV;
#else
    varying vec3 vP;
#endif

void main() {
#if SHOW_COLOR
    gl_FragColor = vColor;
#elif SHOW_NORMAL
    gl_FragColor = vec4((vNormal + vec3(1.0,1.0,1.0)) / 2.0, 1.0);
#elif SHOW_TEXTURE
    gl_FragColor = texture2D(texture_0, vUV);
#elif SHOW_MATCAP
    vec2 uv = 0.5 * vNormalCam.xy + vec2(0.5, 0.5);
    gl_FragColor = texture2D(matcap, uv);
#elif SHOW_UV
    gl_FragColor = vec4(vUV,1.0-vUV.x,1.0);
#else
    gl_FragColor = vec4(vP / 100.0,1.0);
#endif
}
)Shader";

const std::string equi_env_shader = R"Shader(
/////////////////////////////////////////
@start vertex
#version 120
attribute vec2 vertex;
attribute vec2 xy;
varying vec2 vXY;

void main() {
    vXY = xy;
    gl_Position = vec4(vertex,0.0,1.0);
}

@start fragment
#version 120
#define M_PI 3.1415926538
uniform sampler2D texture_0;
uniform mat3 R_env_camKinv;
varying vec2 vXY;

vec2 RayToEquirect(vec3 ray)
{
    float n = 1.0;
    float m = 1.0;
    float lamda = acos(ray.y/sqrt(1.0-ray.z*ray.z));
    if(ray.x < 0) lamda = -lamda;
    float phi = asin(ray.z);
    float u = n*lamda/(2.0*M_PI)+n/2.0;
    float v = m/2.0 + m*phi/M_PI;
    return vec2(u,v);
}

void main() {
    vec3 ray_env = normalize(R_env_camKinv * vec3(vXY, 1.0));
    gl_FragColor = texture2D(texture_0, RayToEquirect(ray_env));
}
)Shader";

}
