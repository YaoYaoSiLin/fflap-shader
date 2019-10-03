#version 130

uniform mat4 gbufferModelViewInverse;

uniform vec3 sunPosition;
uniform vec3 upPosition;

uniform vec3 vanillaWaterColor;

out vec2 texcoord;

out float fading;

out vec3 sunLightingColorRaw;
out vec3 skyLightingColorRaw;

out vec4 waterColor;

#include "libs/common.inc"
#include "libs/atmospheric.glsl"
#include "libs/water.glsl"

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

  waterColor = vec4(vanillaWaterColor, 0.05);

  #if Water_Color_Test > disable
    #if Water_Color_Test == normal_biomes
      waterColor.rgb = vec3(0.247 , 0.4627, 0.8941);
    #elif Water_Color_Test == swamp
      waterColor.rgb = vec3(0.3803, 0.4823, 0.3921);
    #elif Water_Color_Test == frozen_ocean_and_river
      waterColor.rgb = vec3(0.2235, 0.2196, 0.7882);
    #elif Water_Color_Test == warm_ocean
      waterColor.rgb = vec3(0.2627, 0.8352, 0.9333);
    #elif Water_Color_Test == lukewarm_ocean
      waterColor.rgb = vec3(0.2705, 0.6784, 0.949 );
    #elif Water_Color_Test == cold_ocean
      waterColor.rgb = vec3(0.2392, 0.3411, 0.8392);
    #endif
  #endif

  #if MC_VERSION < 11202 || !defined(MC_VERSION)
  waterColor.rgb = vec3(0.247 , 0.4627, 0.8941);
  #endif

  waterColor = CalculateWaterColor(waterColor);
  //waterColor.rgb = rgb2L(waterColor.rgb / overRange);

  texcoord = gl_MultiTexCoord0.st;

  gl_Position = ftransform();
}
