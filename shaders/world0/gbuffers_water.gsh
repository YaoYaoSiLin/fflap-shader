#version 330 compatibility

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec2[3] mid_coord;

in float[3] vertex_material_id;

in vec2[3] vertex_lmcoord;
in vec2[3] vertex_texcoord;

in vec3[3] vertex_normal;
in vec3[3] vertex_tangent;
in vec3[3] vertex_binormal;

in vec3[3] vertexWorldPosition;

in vec4[3] vertex_color;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

uniform ivec2 atlasSize;

uniform vec3 cameraPosition;

out float material_id;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 tangent;
out vec3 binormal;

out vec4 color;

float sdBox( vec3 p, vec3 b ) {
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

void main() {
    vec3 vertex0Normal = mat3(gbufferModelViewInverse) * vertex_normal[0];
    vec3 vertexCenterPosition = (vertexWorldPosition[0].xyz + vertexWorldPosition[1].xyz + vertexWorldPosition[2].xyz) / 3.0;
    vec3 vertexCenterWorldPosition = vertexCenterPosition + cameraPosition;
    vec3 blockCenter = floor(vertexCenterWorldPosition - vertex0Normal * 0.1) + 0.5;

    if(sdBox(vertexCenterWorldPosition - blockCenter, vec3(0.0)) < 0.5 && vertex_material_id[0] > 164.5 && vertex_material_id[0] < 165.5) {
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        EndPrimitive();
    }

    vec2 tileSize = abs(mid_coord[0] * vec2(atlasSize) - min(vertex_texcoord[0], min(vertex_texcoord[1], vertex_texcoord[2])) * vec2(atlasSize)) * 2.0;

    if(round(max(tileSize.x, tileSize.y)) < 16.0) {/*
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        EndPrimitive();*/    
    }

    for(int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;

        material_id = vertex_material_id[i];

        texcoord    = vertex_texcoord[i];
        lmcoord     = vertex_lmcoord[i];

        normal      = vertex_normal[i];
        binormal    = vertex_binormal[i];
        tangent     = vertex_tangent[i];

        color       = vertex_color[i];

        EmitVertex();
    } EndPrimitive();
}