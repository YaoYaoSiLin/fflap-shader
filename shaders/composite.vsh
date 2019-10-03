#version 130

uniform mat4 gbufferModelViewInverse;

uniform vec3 sunPosition;
uniform vec3 upPosition;

out vec2 texcoord;

out float fading;
out vec3 sunLightingColorRaw;
out vec3 skyLightingColorRaw;

#include "libs/common.inc"
#include "libs/atmospheric.glsl"

float CalculateSunLightFading(in vec3 wP, in vec3 sP){
  float h = playerEyeLevel + defaultHightLevel;
  return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}

void main() {
  vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

  vec3 vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
  vec3 wP = mat3(gbufferModelViewInverse) * vP;

  fading = CalculateSunLightFading(normalize(wP.xyz), sP);
  sunLightingColorRaw = CalculateSky(normalize(sunPosition), sP, 63.0, 0.7);
  skyLightingColorRaw = CalculateSky(normalize(upPosition), sP, 63.0, 1.0);

  texcoord = gl_MultiTexCoord0.st;

  gl_Position = ftransform();
}
