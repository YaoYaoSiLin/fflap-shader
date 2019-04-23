#version 120

attribute vec4 at_tangent;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;
varying vec3 tangent;
varying vec3 binormal;
varying vec3 vP;

varying vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

  color = gl_Color;

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif

  normal  = normalize(gl_NormalMatrix * gl_Normal);
  tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
  binormal = cross(tangent, normal);
}
