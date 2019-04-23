#version 120

#define Enabled_TAA

uniform mat4 gbufferProjection;

varying vec3 vP;

varying vec2 texcoord;
varying vec2 lmcoord;

varying vec3 normal;

varying vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  color = gl_Color;

  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif

  normal  = normalize(gl_NormalMatrix * gl_Normal);
}
