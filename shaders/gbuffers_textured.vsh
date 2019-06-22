#version 130

#define Enabled_TAA

out float id;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 vP;

out vec4 color;

#define Taa_Support 1

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

void main() {
  id = 0.0;

  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  normal = normalize(gl_NormalMatrix * gl_Normal);
  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;

  color = gl_Color;

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif
}
