#version 330 compatibility

#pragma optimize(on)

#define SHADOW_MAP_BIAS 0.9
#define SHADOW_MAP_BIAS_Mul 0.95

layout (triangles) in;
layout (triangle_strip, max_vertices = 6) out;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;

uniform vec4 gbufferProjection0;
uniform vec4 gbufferProjection1;
uniform vec4 gbufferProjection2;
uniform vec4 gbufferProjection3;

uniform vec2 pixel;
uniform vec2 resolution;

uniform float far;
uniform float near;
uniform float aspectRatio;

in float[3] vertex_ID;
in float[3] vertexBlockID;

in vec2[3] vertex_texcoord;
in vec2[3] vertex_lmcoord;
in vec3[3] vertex_normal;
in vec4[3] vertex_color;

in vec4[3] worldPosition;

out float isSphereMap;
out float sphereMapDepth;
out vec4 sphereViewPosition;

out float blockID;

out vec3 vworldPosition;
out float vLength;

out vec2 texcoord;
out vec2 lmcoord;
out vec3 world_normal;
out vec4 color;

vec3 nvec3(in vec4 x){
  return x.xyz / x.w;
}

vec3 wP2sP(in vec4 wP){
	vec4 coord = shadowProjection * shadowModelView * wP;
       coord /= coord.w;
       coord.xy /= mix(1.0, length(coord.xy), SHADOW_MAP_BIAS) / SHADOW_MAP_BIAS_Mul;
       coord = coord * 0.5 + 0.5;
       coord.xy = min(coord.xy, vec2(1.0)) * vec2(1.0, 0.5);

	return coord.xyz;
}

void SphereMapping(inout vec4 position){
    #if MC_VERSION >= 11605
    position.xy /= aspectRatio;
    position.xy *= resolution.y / 1080.0;
    #endif

    float Z = position.z;
    float L = length(position.xyz);
    
    position.xyz = (position.xyz) / L;
    position.xy /= 1.0 + (position.z);
    position.z = L / far * 2.0 - 1.0;

    position.w = 1.0;
}

void main() {
    vworldPosition = vec3(1000.0);

    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;

        gl_Position = shadowProjection * gl_Position;

        float distortion = length(gl_Position.xy);
        gl_Position.xy /= mix(1.0, distortion, SHADOW_MAP_BIAS) / 0.95;

        gl_Position.xy = gl_Position.xy * 0.5 + 0.5;
        gl_Position.xy *= vec2(1.0, 0.5);
        //gl_Position.xy = clamp(gl_Position.xy, vec2(0.0), vec2(1.0, 0.5));
        gl_Position.xy = gl_Position.xy * 2.0 - 1.0;

        blockID = vertexBlockID[i];
        texcoord = vertex_texcoord[i];
        lmcoord = vertex_lmcoord[i];
        world_normal = vertex_normal[i];
        color = vertex_color[i];

        isSphereMap = 0.0;

        EmitVertex();
    } EndPrimitive();

    #if MC_VERSION >= 11605
    mat4 gbufferProjection_ = mat4(gbufferProjection0, -gbufferProjection1, gbufferProjection2, gbufferProjection3);
    #else
    mat4 gbufferProjection_ = gbufferProjection;
    #endif

    float angle = 0.25 * 2.0 * 3.14159265;
    mat2 facingUp = mat2(cos(-angle), sin(-angle), -sin(-angle), cos(-angle));
    mat2 facingDown = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

    vec3 triangleCenter = (worldPosition[0].xyz + worldPosition[1].xyz + worldPosition[2].xyz) / 3.0;

    bool isPlayer = vertex_ID[0] > 0.5 && length(triangleCenter.xz) < 0.7 && length(triangleCenter.y + 1.62 * 0.5) < 1.7;

    if(isPlayer) {
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        gl_Position = vec4(0.0);
        EmitVertex();
        EndPrimitive();
    } else if(min(worldPosition[0].y, min(worldPosition[1].y, worldPosition[2].y)) < 0.0) {
        for (int i = 0; i < 3; i++) {
            vec4 position = worldPosition[i];
            vworldPosition = position.xyz;

            sphereViewPosition.xyz = wP2sP(position);

            position.yz = facingDown * position.yz;
            position = gbufferProjection_ * position;

            SphereMapping(position);

            position.xy = position.xy * 0.5 + 0.5;

            position.xy *= vec2(0.25, 0.25);
            position.xy += vec2(0.125, 0.625);
            position.xy = clamp(position.xy, vec2(0.0, 0.5), vec2(0.5, 1.0));
            position.xy = position.xy * 2.0 - 1.0;

            gl_Position = position;

            blockID = vertexBlockID[i];
            texcoord = vertex_texcoord[i];
            lmcoord = vertex_lmcoord[i];
            world_normal = vertex_normal[i];
            color = vertex_color[i];

            isSphereMap = 1.0;

            EmitVertex();
        } EndPrimitive();
    } else {
        for (int i = 0; i < 3; i++) {
            vec4 position = worldPosition[i];
            vworldPosition = position.xyz;

            sphereViewPosition.xyz = wP2sP(position);

            position.yz = facingUp * position.yz;
            position = gbufferProjection_ * position;

            SphereMapping(position);

            position.xy = position.xy * 0.5 + 0.5;
            position.xy *= vec2(0.25, 0.25);
            position.xy += vec2(0.625, 0.625);
            
            position.xy = clamp(position.xy, vec2(0.5, 0.5), vec2(1.0, 1.0));
            position.xy = position.xy * 2.0 - 1.0;

            gl_Position = position;

            blockID = vertexBlockID[i];
            texcoord = vertex_texcoord[i];
            lmcoord = vertex_lmcoord[i];
            world_normal = vertex_normal[i];
            color = vertex_color[i];

            isSphereMap = 1.0;

            EmitVertex();
        } EndPrimitive();
    }
}