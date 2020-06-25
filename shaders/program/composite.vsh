out vec2 texcoord;

#if defined(Calc_Natural_Light_Color) || defined(Gen_Water_Color) || defined(Gen_TnR)
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferModelView;

uniform float rainStrength;

uniform vec3 sunPosition;
uniform vec3 upPosition;

#include "../libs/common.inc"
#endif

#if defined(Calc_Natural_Light_Color)
out float fading;

uniform int worldTime;

out vec3 sunLightingColorRaw;
out vec3 skyLightingColorRaw;
out vec3 sunLightingColor;
out vec3 skyLightingColor;

#include "../libs/atmospheric.glsl"

float CalculateSunLightFading(in vec3 wP, in vec3 sP){

  return clamp01(sP.y - 0.0);

  //float h = playerEyeLevel + defaultHightLevel;
  //return clamp01(dot(sP * defaultHightLevel, vec3(0.0, h, 0.0)) / (h * defaultHightLevel) * 10.0);
}
#endif

#if defined(Gen_TnR) || defined(Gen_Water_Color)
#include "../libs/biomes.glsl"
#endif

#ifdef Gen_TnR
out float temperature;
out float rainfall;
out float dayTemp;
out float nightTemp;

out vec3 biomeDustColor;

#ifdef Calc_Sky_With_Fog
out float skyDustFactor;
#endif
#endif

#ifdef Gen_Water_Color
out vec4 eyesWaterColor;
#endif

void main(){
  gl_Position = ftransform();
  texcoord = gl_MultiTexCoord0.st;

  #if defined(Calc_Natural_Light_Color) || defined(Gen_Water_Color) || defined(Gen_TnR)
    vec3 sP = mat3(gbufferModelViewInverse) * normalize(sunPosition);

    bool night = sP.y < -0.1;

    vec3 lightPosition = sunPosition;
    if(night) lightPosition = -lightPosition;
    vec3 nlightPosition = normalize(lightPosition);
    vec3 worldLightPosition = mat3(gbufferModelViewInverse) * nlightPosition;
    //vec3 worldSunPosition = mat3(gbufferModelViewInverse) * normalize(sunPosition);

    vec3 vP = (gl_ModelViewMatrix * gl_Vertex).xyz;
    vec3 wP = mat3(gbufferModelViewInverse) * vP;
  #endif

  #ifdef Gen_TnR
    biomeDustColor = vec3(0.0);
    temperature = 0.0;
    rainfall = 0.0;
    CalculateBiomeData(temperature, rainfall, biomeDustColor);

    float t = temperature;
    float irainfall = 1.0 - rainfall;

    dayTemp = clamp01(sP.y);
    dayTemp += clamp01((sP.y - sP.x * 0.9)) * dayTemp;
    dayTemp *= 1.0 - rainStrength * 0.7;

    #if defined(Calc_Sky_With_Fog) || defined(Calc_Lighting_Color_With_Fog)
      float tempAvg = 0.7;
            tempAvg = tempAvg + dayTemp * tempAvg * tempAvg;

      float dustFactorTemp = t + dayTemp * t * t;
            dustFactorTemp = (dustFactorTemp - 1.5 - tempAvg) / (tempAvg + 1.5);
            dustFactorTemp *= 1.0 + 15.0 * rainStrength;

      float dustFactor = 128.0 * dustFactorTemp * 0.005 * irainfall;
            dustFactor = 1.0 - clamp01(exp(-(dustFactor)));
            dustFactor *= min(1.0, dustFactor) * irainfall;

      skyDustFactor = dustFactor;
    #endif
  #endif

  #ifdef Gen_Water_Color
    eyesWaterColor = vec4(vanillaWaterColor, 0.05);
    eyesWaterColor = CalculateWaterColor(eyesWaterColor);
    //eyesWaterColor.rgb = pow(eyesWaterColor.rgb, vec3(2.2));

    #ifdef Calc_Sky_With_Fog
      //eyesWaterColor.rgb = mix(eyesWaterColor.rgb, biomeDustColor, skyDustFactor);
    #endif
  #endif

  #ifdef Calc_Natural_Light_Color
    vec3 nupPosition = normalize(upPosition);

    vec3 worldSkyPosition = mat3(gbufferModelViewInverse) * normalize(vP.xyz);
    fading = clamp01((worldLightPosition.y - 0.1) * 6.6667);
    sunLightingColorRaw = CalculateSky(nlightPosition, sP, 63.0, 1.0);
    skyLightingColorRaw = CalculateSky(nupPosition, sP, 63.0, 1.0);

    if(night){
      //sunLightingColorRaw *= 2.0;
      skyLightingColorRaw *= 4.0;
    }

    sunLightingColor = sunLightingColorRaw;
    skyLightingColor = skyLightingColorRaw;

    #ifdef Calc_Lighting_Color_With_Fog
    /*
      sunLightingColor = pow(biomeDustColor, vec3(2.2)) * sunLightingColor;
      sunLightingColor *= 1.0 - dustFactor;
      sunLightingColor = mix(sunLightingColorRaw, sunLightingColor * 0.07, dustFactor);
      skyLightingColor = mix(skyLightingColor, sunLightingColor * 0.5, dustFactor);
      */
      //sunLightingColor = sunLightingColorRaw * rgb2L(biomeDustColor);
      //sunLightingColor = mix(sunLightingColorRaw, sunLightingColor * 0.037, dustFactor);
      //skyLightingColor = mix(skyLightingColorRaw, sunLightingColor * 0.26, dustFactor);
      //skyLightingColor = rgb2L(biomeDustColor) * (skyLightingColorRaw + sunLightingColorRaw) * 0.5;
      //skyLightingColor *= min(1.0, (1.0 - dustFactor) * 1.0);
    #endif
  #endif
}
