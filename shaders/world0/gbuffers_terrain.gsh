#version 330 compatibility

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform ivec2 atlasSize;

uniform vec3 cameraPosition;

uniform vec3 lightVectorWorld;

in vec2[3] mid_coord;

in float[3] v_material_id;

in vec2[3] v_texcoord;
in vec2[3] v_lmcoord;

in vec3[3] v_normal;
in vec3[3] v_tangent;
in vec3[3] v_binormal;

in vec3[3] v_viewVector;

in vec3[3] worldPosition;
in vec3[3] worldNormal;

in vec4[3] v_color;

out float material_id;

out float solid;

out vec2 tileCoord0;
out vec2 tileCoord1;
out vec2 tileCoord2;
out vec2 tileCenter;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 tangent;
out vec3 binormal;

out vec3 lightVector;
out vec3 viewVector;

out vec4 color;

#define Enabled_Door_Parallax_Fix

float sdBox( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

void main() {
    vec3 vertexCenterPosition = (worldPosition[0].xyz + worldPosition[1].xyz + worldPosition[2].xyz) / 3.0;
    vec3 vertexCenterWorldPosition = vertexCenterPosition + cameraPosition;
    vec3 blockCenter = floor(vertexCenterWorldPosition - worldNormal[0] * 0.1) + 0.5;
    vec3 h = abs(vertexCenterWorldPosition - blockCenter);

    solid = 1.0;

    if(max(h.x, max(h.y, h.z)) < 0.5 - 0.025) {
        solid = 0.5;
    }

    tileCoord0 = v_texcoord[0] * vec2(atlasSize);
    tileCoord1 = v_texcoord[1] * vec2(atlasSize);
    tileCoord2 = v_texcoord[2] * vec2(atlasSize);
    tileCenter = mid_coord[0] * vec2(atlasSize);

    vec2 p = tileCenter - (tileCoord0 + tileCoord1 + tileCoord2) / 3.0;

    vec3 n = (worldNormal[0] + worldNormal[1] + worldNormal[2]) / 3.0;

    float e = dot(vec3(1.0, 0.0, 0.0), n);
    float t = dot(vec3(0.0, 1.0, 0.0), n);
    float s = dot(vec3(0.0, 0.0, 1.0), n);

    vec3 r0 = vec3(1.0, 0.0, 0.0);
    vec3 r1 = vec3(0.0, 1.0, 0.0);
    vec3 r2 = vec3(0.0, 0.0, 1.0);

    #ifdef Enabled_Door_Parallax_Fix
    if(v_material_id[0] == 96.0) {
        if((p.x > 0.0 && p.y > 0.0) || (p.x < 0.0 && p.y < 0.0)) {
            r1 *= (1.0 - step(abs(t), 0.5)) * 2.0 - 1.0;
            r2 *= step(t, 0.5) * 2.0 - 1.0;
        } else {
            r0 *= -((1.0 - step(t, 0.5)) * 2.0 - 1.0);
        }
    }
    #endif

    vec3 rotate = vec3(r0.x, r1.y, r2.z);
    lightVector = mat3(gbufferModelView) * (lightVectorWorld * rotate);

    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;

        material_id = v_material_id[i];

        texcoord    = v_texcoord[i];
        lmcoord     = v_lmcoord[i];

        normal      = v_normal[i];
        binormal    = v_binormal[i];
        tangent     = v_tangent[i];

        color       = v_color[i];

        //viewVector = v_viewVector[i];
        viewVector = mat3(gbufferModelView) * (rotate * worldPosition[i]);

        EmitVertex();
    } EndPrimitive();    
}