#version 130

#define Enabled_TAA

attribute vec4 at_tangent;
attribute vec4 mc_Entity;

uniform mat4 gbufferModelViewInverse;

uniform vec3 sunPosition;
uniform vec3 upPosition;

out float id;

out vec2 texcoord;
out vec2 lmcoord;

out vec3 normal;
out vec3 binormal;
out vec3 tangent;
out vec3 vP;

out float fading;
out vec3 sunLightingColorRaw;
out vec3 skyLightingColorRaw;

out vec4 biomesColor;

#define Taa_Support 1

#include "libs/common.inc"

#include "libs/jittering.glsl"
#include "libs/taa.glsl"

#include "libs/atmospheric.glsl"

float CalculateSunLightFading(in vec3 wP, in vec3 sP){
  float h = playerEyeLevel + defaultHightLevel;
  return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}

void main() {
  id = 0.0;

  biomesColor = gl_Color;
  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  if(mc_Entity.x == 8.0) {
    id = 8.0;
    //biomesColor.rgb = mix(biomesColor.rgb, vec3())
    //biomesColor.rgb = vec3(38.0, 140.0, 0.0) / 255.0 * 0.08;
    //biomesColor.rgb = vec3(0.682, 0.415, 0.0) * 0.5;
  }

  if(mc_Entity.x == 20 || mc_Entity.x == 95) id = 20.0;
  if(mc_Entity.x == 79) id = 79.0;
  if(mc_Entity.x == 90) id = 90.0;
  if(mc_Entity.x == 106 || mc_Entity.x == 160) id = 106.0;

  normal  = normalize(gl_NormalMatrix * gl_Normal);
  tangent = normalize(gl_NormalMatrix * at_tangent.xyz);
  binormal = cross(tangent, normal);
  vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
  vec3 wP = mat3(gbufferModelViewInverse) * vP;

  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  fading = CalculateSunLightFading(normalize(wP.xyz), sP);

  sunLightingColorRaw = (CalculateSky(normalize(sunPosition), sP, 0.0, 0.375));
  skyLightingColorRaw = (CalculateSky(normalize(upPosition), sP, 0.0, 0.5));

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif
}
