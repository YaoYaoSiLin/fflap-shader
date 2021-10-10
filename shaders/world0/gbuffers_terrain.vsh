#version 130

in vec4 at_tangent;
in vec3 mc_Entity;
in vec2 mc_midTexCoord;

#define GSH

#ifndef GSH
    #define v_viewVector viewVector

    #define v_material_id material_id

    #define v_texcoord texcoord
    #define v_lmcoord lmcoord

    #define v_normal normal
    #define v_tangent tangent
    #define v_binormal binormal

    #define v_color color
#endif

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform vec2 jitter;

out vec2 mid_coord;

out vec3 v_viewVector;

out vec3 worldPosition;
out vec3 worldNormal;

out float v_material_id;

out vec2 v_texcoord;
out vec2 v_lmcoord;

out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_binormal;

out vec4 v_color;

#define Enabled_TAA

void main() {
    v_material_id = 0.0;

    mid_coord = mc_midTexCoord;

    v_texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
    v_lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

    v_color = gl_Color;

    v_normal = normalize(mat3(gl_NormalMatrix) * gl_Normal);
    v_tangent = normalize(mat3(gl_NormalMatrix) * at_tangent.xyz);
    v_binormal = cross(v_tangent, v_normal);

    bool herbaceous_plants = mc_Entity.x == 31.0;
    bool double_plant_upper = mc_Entity.x == 175.0;
    bool double_plant_lower = mc_Entity.x == 176.0;
    bool double_plant = double_plant_upper || double_plant_lower;

    bool leveas = mc_Entity.x == 18.0 || (mc_Entity.x > 1799.0 && mc_Entity.x < 1806.0);

    bool vine = mc_Entity.x == 106.0;

    bool lily_pad = mc_Entity.x == 111.0;

    bool stem_plants = mc_Entity.x == 59.0;

    gl_Position = gl_ModelViewMatrix * gl_Vertex;

    if(herbaceous_plants || double_plant) {
        v_material_id = 31.0;
    }

    if(stem_plants) {
        v_material_id = 31.0;
    }

    if(leveas) {
        v_material_id = 18.0;
    }

    if(vine) {
        v_material_id = 18.0;
    }

    if(lily_pad) {
        v_material_id = 18.0;
    }

    if(mc_Entity.x == 96.0) {
        v_material_id = 96.0;
    }

    if(mc_Entity.x == 83.0) {
        v_material_id = 31.0;
    }

    v_viewVector = gl_Position.xyz;

    worldPosition = mat3(gbufferModelViewInverse) * v_viewVector;
    worldNormal = mat3(gbufferModelViewInverse) * v_normal;

    gl_Position = gl_ProjectionMatrix * gl_Position;

    #ifdef Enabled_TAA
        gl_Position.xy += jitter * gl_Position.w * 2.0;
    #endif
}