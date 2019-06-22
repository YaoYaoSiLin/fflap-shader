#version 120

#define Enabled_TAA

varying vec2 texcoord;

varying vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;

  color = gl_Color;

  gl_Position = ftransform();

  #ifdef Enabled_TAA
  //gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif
}
