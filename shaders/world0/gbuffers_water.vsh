#version 130

#define GSH

#ifndef GSH
    #define vertex_material_id material_id

    #define vertex_texcoord texcoord
    #define vertex_lmcoord lmcoord

    #define vertex_normal normal
    #define vertex_tangent tangent
    #define vertex_binormal binormal

    #define vertex_color color
#endif

in vec4 at_tangent;
in vec3 at_midBlock;
in vec3 mc_Entity;
in vec2 mc_midTexCoord;

uniform mat4 gbufferModelViewInverse;

out vec2 mid_coord;

out float vertex_material_id;

out vec2 vertex_texcoord;
out vec2 vertex_lmcoord;

out vec3 vertex_normal;
out vec3 vertex_tangent;
out vec3 vertex_binormal;

out vec3 vertexWorldPosition;

out vec4 vertex_color;

#define Enabled_TAA

#include "/libs/common.inc"
#include "/libs/materials/material_data.glsl"

void main() {
    mid_coord = mc_midTexCoord;

    vertex_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    vertex_lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    vertex_material_id = 0.0;

    vertex_color = gl_Color;

    vertex_normal = normalize(gl_NormalMatrix * gl_Normal);
    vertex_tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
    vertex_binormal = cross(vertex_tangent, vertex_normal);

    gl_Position = gl_Vertex;

    if(mc_Entity.x == 8) {
        vertex_material_id = 8.0;

        vertex_color.rgb = CalculateWaterColor(vertex_color).rgb;
        vertex_color.a = 0.1;
    } else 
    if(mc_Entity.x == 79) {
        vertex_material_id = 79.0;
    } else 
    if(mc_Entity.x == 90) {
        vertex_material_id = 90.0;
    }else 
    if(mc_Entity.x == 165) {
        vertex_material_id = 165.0;
    }else 
    if(mc_Entity.x == 20.0) {
        vertex_material_id = 20.0;
    }else 
    if(mc_Entity.x == 95.0) {
        vertex_material_id = 95.0;
    }else
    if(mc_Entity.x == 102) {
        vertex_material_id = 102.0;
    }else 
    if(mc_Entity.x == 160) {
        vertex_material_id = 160.0;
    }

    gl_Position = gl_ModelViewMatrix * gl_Position;

    vertexWorldPosition = mat3(gbufferModelViewInverse) * gl_Position.xyz;

    gl_Position = gl_ProjectionMatrix * gl_Position;

    #ifdef Enabled_TAA
    gl_Position.xy += jitter * 2.0 * gl_Position.w;
    #endif
}