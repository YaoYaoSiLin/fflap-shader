#version 450 core
#pragma optimize(on)

#define SHADOW_MAP_BIAS 0.9

layout (triangles) in;
layout (triangle_strip, max_vertices = 9) out;

uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowModelView;
uniform mat4 shadowProjection;

uniform vec3 cameraPosition;
uniform vec3 shadowLightPosition;

uniform float far;
uniform float near;

in float[] vertex_ID;
in float[] vertexBlockID;

in vec2[] vertex_texcoord;
in vec2[] vertex_lmcoord;
in vec3[] vertex_normal;
in vec4[] vertex_color;

in vec4[] worldPosition;

out float isSphereMap;
out float sphereMapDepth;
out vec4 sphereViewPosition;

out float blockID;

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
       coord.xy /= mix(1.0, length(coord.xy), SHADOW_MAP_BIAS) / 0.95;
       coord = coord * 0.5 + 0.5;
       coord.xy *= 0.8;
       coord.xy = min(coord.xy, 0.8);

	return coord.xyz;
}

void SphereMapping(inout vec4 position){
  float Z = position.z;
  float L = length(position.xyz);

  position.xyz = normalize(position.xyz);
  position.xy /= position.z + 1.0;
  position.z = L * sign(Z) / far * 2.0 - 1.0;
  position.w = 1.0;
}

void main(){
  float angle = 0.25 * 2.0 * 3.14159265;
  mat2 facingUp = mat2(cos(-angle), sin(-angle), -sin(-angle), cos(-angle));
  mat2 facingDown = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

  sphereMapDepth = 0.0;

  float diff = far - near;

  for (int i = 0; i < 3; i++) {
    vec4 position = worldPosition[i];
    float worldLength = length(position.xyz);

    if(length(position.xz) < 0.5 && length(position.y + 1.62 * 0.5) < 1.62 && bool(step(0.5, vertex_ID[i]))) break;
    if(worldLength < near) break;

    //position = gbufferModelView * position;

    sphereViewPosition.xyz = wP2sP(position);

    position.yz = facingUp * position.yz;
    position = gbufferProjection * position;
    SphereMapping(position);

    position.xy = position.xy * 0.5 + 0.5;
    position.xy *= 0.2; position.x += 0.8; position.y += 0.05;
    position.xy = position.xy * 2.0 - 1.0;

    gl_Position = position;

    texcoord = vertex_texcoord[i];
    lmcoord = vertex_lmcoord[i];
    world_normal = vertex_normal[i];
    color = vertex_color[i];

    isSphereMap = 1.0;

    EmitVertex();
  } EndPrimitive();

  angle = 0.5 * 2.0 * 3.14159265;
  mat2 rotate2 = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));

  for (int i = 0; i < 3; i++) {
    vec4 position = worldPosition[i];
    float worldLength = length(position.xyz);

    if(length(position.xz) < 0.5 && length(position.y + 1.62 * 0.5) < 1.62 && bool(step(0.5, vertex_ID[i]))) break;
    if(worldLength < near) break;

    //position = gbufferModelView * position;

    sphereViewPosition.xyz = wP2sP(position);

    position.yz = facingDown * position.yz;
    position = gbufferProjection * position;
    SphereMapping(position);

    position.xy = position.xy * 0.5 + 0.5;
    position.xy *= 0.2; position.x += 0.8; position.y += 0.3;
    position.xy = position.xy * 2.0 - 1.0;

    gl_Position = position;

    texcoord = vertex_texcoord[i];
    lmcoord = vertex_lmcoord[i];
    world_normal = vertex_normal[i];
    color = vertex_color[i];

    isSphereMap = 1.0;

    EmitVertex();
  } EndPrimitive();

  //Shadow Map
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;

    gl_Position = shadowProjection * gl_Position;

    float distortion = length(gl_Position.xy);
    gl_Position.xy /= mix(1.0, distortion, SHADOW_MAP_BIAS) / 0.95;

    //gl_Position.z = distortion * sign(gl_Position.z) / renderDistance * 2.0 - 1.0;

    gl_Position.xy = gl_Position.xy * 0.5 + 0.5;
    gl_Position.xy *= 0.8;
    gl_Position.xy = gl_Position.xy * 2.0 - 1.0;

    blockID = vertexBlockID[i];
    texcoord = vertex_texcoord[i];
    lmcoord = vertex_lmcoord[i];
    world_normal = vertex_normal[i];
    color = vertex_color[i];

    isSphereMap = 0.0;

    EmitVertex();
  } EndPrimitive();
}
