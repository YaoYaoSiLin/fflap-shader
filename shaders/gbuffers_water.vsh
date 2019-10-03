#version 130

#define Enabled_TAA

attribute vec4 at_tangent;
attribute vec3 mc_Entity;

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

#define disable -255
#define swamp 6
#define frozen_ocean_and_river 10
#define warm_ocean 44
#define lukewarm_ocean 45
#define cold_ocean 46

#if !defined(default)
#define default 0
#endif

#define Water_Color_Test disable //[disable default swamp frozen_ocean_and_river warm_ocean lukewarm_ocean cold_ocean]

float CalculateSunLightFading(in vec3 wP, in vec3 sP){
  float h = playerEyeLevel + defaultHightLevel;
  return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}

void main() {
  id = 0.0;

  texcoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
  lmcoord  = (gl_TextureMatrix[1] * gl_MultiTexCoord1).xy;

  biomesColor = gl_Color;

  if(mc_Entity.x == 8.0) {
    id = 8.0;

    #if MC_VERSION <= 11300 || !defined(MC_VERSION)
    if(biomesColor.r + biomesColor.g + biomesColor.b > 2.999) biomesColor.rgb = vec3(0.247 , 0.4627, 0.8941);
    #endif

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

  sunLightingColorRaw = (CalculateSky(normalize(sunPosition), sP, 0.0, 0.7));
  skyLightingColorRaw = (CalculateSky(normalize(upPosition), sP, 0.0, 1.0));

  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;
  #ifdef Enabled_TAA
  gl_Position.xy += haltonSequence_2n3[int(mod(frameCounter, 16))] * gl_Position.w * pixel;
  #endif


  #if Water_Color_Test > disable
    if(id == 8.0){
    #if Water_Color_Test == default
      biomesColor.rgb = vec3(0.247 , 0.4627, 0.8941);
    #elif Water_Color_Test == swamp
      biomesColor.rgb = vec3(0.3803, 0.4823, 0.3921);
    #elif Water_Color_Test == frozen_ocean_and_river
      biomesColor.rgb = vec3(0.2235, 0.2196, 0.7882);
    #elif Water_Color_Test == warm_ocean
      biomesColor.rgb = vec3(0.2627, 0.8352, 0.9333);
    #elif Water_Color_Test == lukewarm_ocean
      biomesColor.rgb = vec3(0.2705, 0.6784, 0.949 );
    #elif Water_Color_Test == cold_ocean
      biomesColor.rgb = vec3(0.2392, 0.3411, 0.8392);
    #endif
    }
  #endif
}
