#version 120

attribute vec4 mc_Entity;
attribute vec4 at_tangent;

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;

varying float isWater;
varying float isGlass;
varying float isIce;

varying vec2 uv;
varying vec2 lmcoord;

varying vec3 vP;

varying vec3 normal;
varying vec3 tangent;
varying vec3 binormal;

varying vec4 color;

void main() {
  isWater = 0.0;
  if(mc_Entity.x == 8 || mc_Entity.x == 9) isWater = 1.0;

  isGlass = 0.0;
  if(mc_Entity.x == 95 || mc_Entity.x == 160) isGlass = 1.0;

  isIce = 0;
  if(mc_Entity.x == 79) isIce = 1;

  uv = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  color = gl_Color;

  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

  vec4 position = gbufferModelViewInverse * gl_ModelViewMatrix * gl_Vertex;
  gl_Position = gl_ProjectionMatrix * gbufferModelView * position;

  normal  = normalize(gl_NormalMatrix * gl_Normal);
  tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
  binormal = cross(tangent, normal);
}
