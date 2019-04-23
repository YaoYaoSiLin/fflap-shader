#version 120

#define Enabled_TAA

varying vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  color = gl_Color;

  gl_Position = ftransform();

  #ifdef Enabled_TAA
  //gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif
}
