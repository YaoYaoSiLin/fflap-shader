#version 130

#define Enabled_TAA

in vec4 at_tangent;
in vec4 mc_Entity;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;

#ifndef GSH
    #define vtexcoord texcoord
    #define vlmcoord lmcoord

    #define vnormal normal
    #define vtangent tangent
    #define vbinormal binormal

    #define vcolor color
#endif

out vec2 vlmcoord;
out vec2 vtexcoord;

out vec3 vnormal;
out vec3 vbinormal;
out vec3 vtangent;

out vec4 vcolor;

uniform vec2 jitter;

void main() {
    vtexcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vlmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    vcolor = gl_Color;

    vnormal  = normalize(gl_NormalMatrix * gl_Normal);
    vtangent = normalize(gl_NormalMatrix * at_tangent.xyz);
    vbinormal = cross(vtangent, vnormal);

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

    #ifdef Enabled_TAA
    gl_Position.xy += jitter * 2.0 * gl_Position.w;
    #endif
}